#!/usr/bin/env python3
"""
Trace Generator and Visualizer

Generates a trace file with tenant and interval pairs and visualizes the trace.
Each line contains: tenant_id interval

The trace is first filtered by the specified time window (--start-time and --duration) to extract
a time window from the original trace. Tenants are then selected from the filtered trace according
to the specified selection method. The duration parameter is used both for filtering and as the
makespan (total time span from first to last request across all tenants). The output file contains
intervals between consecutive requests, which follow the distribution pattern of the filtered trace
(scaled proportionally to fit within the specified duration).

Usage:
    # Generate and visualize trace
    python trace_generator.py --num-tenants 4 --request-scale 2.0 --start-time 1000 --duration 600
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print(
        "Warning: matplotlib is not installed. Install it with: pip install matplotlib numpy"
    )


def load_trace(trace_file: str) -> List[Dict[str, Any]]:
    """Load trace data from JSON file."""
    print(f"Loading trace from {trace_file}...")
    with open(trace_file, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} requests from trace")
    return data


def print_timestamp_range(trace_data: List[Dict[str, Any]]):
    """Print the range of timestamps in the trace data."""
    if not trace_data:
        print("No trace data available")
        return

    timestamps = [
        request.get("timestamp", 0) for request in trace_data if "timestamp" in request
    ]

    if not timestamps:
        print("Warning: No timestamps found in trace data")
        return

    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    total_duration = max_timestamp - min_timestamp

    print(f"\nTimestamp range in trace:")
    print(f"  Minimum timestamp: {min_timestamp:.2f}")
    print(f"  Maximum timestamp: {max_timestamp:.2f}")
    print(
        f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
    )
    print(f"  Valid --start-time range: [{min_timestamp:.2f}, {max_timestamp:.2f}]")


def filter_trace_by_time_window(
    trace_data: List[Dict[str, Any]], start_time: float, duration: float
) -> List[Dict[str, Any]]:
    """
    Filter trace data to include only requests within the specified time window.

    Args:
        trace_data: List of request dictionaries
        start_time: Starting timestamp (inclusive)
        duration: Duration of the time window in seconds

    Returns:
        Filtered list of requests within [start_time, start_time + duration]
    """
    end_time = start_time + duration
    filtered_data = []

    for request in trace_data:
        timestamp = request.get("timestamp", 0)
        if start_time <= timestamp <= end_time:
            filtered_data.append(request)

    print(
        f"Filtered trace: {len(filtered_data)} requests in time window "
        f"[{start_time:.2f}, {end_time:.2f}] (duration: {duration:.2f}s)"
    )

    return filtered_data


def create_tenant_id(request: Dict[str, Any]) -> str:
    """Create a unique tenant ID from request data."""
    scene_id = request.get("scene_id", "")
    base_model_id = request.get("base_model_id", "")
    controlnets = (
        sorted(request.get("controlnets", [])) if request.get("controlnets") else []
    )
    loras = sorted(request.get("loras", [])) if request.get("loras") else []

    return f"{scene_id}|{base_model_id}|{','.join(controlnets)}|{','.join(loras)}"


def analyze_tenants(trace_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze tenants and return request counts for each tenant."""
    tenant_counts = defaultdict(int)

    for request in trace_data:
        tenant_id = create_tenant_id(request)
        tenant_counts[tenant_id] += 1

    return dict(tenant_counts)


def select_tenants_top_n(
    tenant_counts: Dict[str, int], num_tenants: int
) -> List[Tuple[str, int]]:
    """Select top N tenants by request count."""
    sorted_tenants = sorted(tenant_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_tenants[:num_tenants]


def select_tenants_similar_frequency(
    tenant_counts: Dict[str, int], num_tenants: int
) -> List[Tuple[str, int]]:
    """Select tenants with similar request frequencies."""
    if len(tenant_counts) < num_tenants:
        # If not enough tenants, return all sorted by count
        return sorted(tenant_counts.items(), key=lambda x: x[1], reverse=True)

    # Calculate statistics
    counts = list(tenant_counts.values())
    mean_count = statistics.mean(counts)
    median_count = statistics.median(counts)

    # Find tenants with counts close to median/mean
    sorted_tenants = sorted(
        tenant_counts.items(), key=lambda x: abs(x[1] - median_count)
    )

    # Select tenants with similar frequencies
    selected = []
    seen_counts = set()

    for tenant_id, count in sorted_tenants:
        if len(selected) >= num_tenants:
            break
        # Try to avoid selecting tenants with exactly the same count
        if count not in seen_counts or len(selected) < num_tenants // 2:
            selected.append((tenant_id, count))
            seen_counts.add(count)

    # If we don't have enough, fill with closest to median
    if len(selected) < num_tenants:
        remaining = [t for t in sorted_tenants if t not in selected]
        selected.extend(remaining[: num_tenants - len(selected)])

    return selected[:num_tenants]


def select_tenants_random(
    tenant_counts: Dict[str, int], num_tenants: int
) -> List[Tuple[str, int]]:
    """Randomly select tenants."""
    all_tenants = list(tenant_counts.items())
    if len(all_tenants) <= num_tenants:
        return all_tenants
    return random.sample(all_tenants, num_tenants)


def select_tenants_balanced(
    tenant_counts: Dict[str, int], num_tenants: int
) -> List[Tuple[str, int]]:
    """Select tenants to balance the load (mix of high, medium, and low frequency)."""
    if len(tenant_counts) < num_tenants:
        return sorted(tenant_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_tenants = sorted(tenant_counts.items(), key=lambda x: x[1], reverse=True)
    total_tenants = len(sorted_tenants)

    # Divide into high, medium, low frequency groups
    high_end = total_tenants // 3
    medium_end = 2 * total_tenants // 3

    high_freq = sorted_tenants[:high_end]
    medium_freq = sorted_tenants[high_end:medium_end]
    low_freq = sorted_tenants[medium_end:]

    # Select balanced mix
    selected = []
    per_group = num_tenants // 3
    remainder = num_tenants % 3

    # Select from each group
    selected.extend(high_freq[: per_group + (1 if remainder > 0 else 0)])
    selected.extend(medium_freq[: per_group + (1 if remainder > 1 else 0)])
    selected.extend(low_freq[:per_group])

    # Fill remaining slots
    all_available = high_freq + medium_freq + low_freq
    for tenant in all_available:
        if len(selected) >= num_tenants:
            break
        if tenant not in selected:
            selected.append(tenant)

    return selected[:num_tenants]


def select_tenants(
    tenant_counts: Dict[str, int], num_tenants: int, method: str = "top_n"
) -> List[Tuple[str, int]]:
    """
    Select tenants based on the specified method.

    Args:
        tenant_counts: Dictionary mapping tenant_id to request count
        num_tenants: Number of tenants to select
        method: Selection method ('top_n', 'similar_frequency', 'random', 'balanced')

    Returns:
        List of (tenant_id, count) tuples
    """
    if method == "top_n":
        return select_tenants_top_n(tenant_counts, num_tenants)
    elif method == "similar_frequency":
        return select_tenants_similar_frequency(tenant_counts, num_tenants)
    elif method == "random":
        return select_tenants_random(tenant_counts, num_tenants)
    elif method == "balanced":
        return select_tenants_balanced(tenant_counts, num_tenants)
    else:
        raise ValueError(
            f"Unknown selection method: {method}. "
            f"Choose from: 'top_n', 'similar_frequency', 'random', 'balanced'"
        )


def generate_trace(
    num_tenants: int,
    request_scale: float,
    start_time: float,
    duration: float,
    trace_file: str = "diffusion_model_request_trace.json",
    selection_method: str = "top_n",
    seed: int = None,
) -> List[tuple]:
    """
    Generate a trace with tenant and interval pairs, preserving original interleaved order.

    The trace is first filtered by the specified time window [start_time, start_time + duration],
    then tenants are selected from the filtered trace according to the selection method.
    After selecting tenants, the number of requests is scaled using a unified approach:
    - First duplicate requests to ceiling(request_scale) times
    - Then sample down to the target number if needed (using seed for reproducibility)
    - Intervals are duplicated accordingly and then adjusted proportionally
    The intervals are then scaled proportionally so that all requests fit within the specified
    duration (global makespan) while preserving the relative distribution.

    Args:
        num_tenants: Number of tenants to generate
        request_scale: Scale factor for number of requests (e.g., 2.0 doubles requests, 0.5 halves requests, 1.5 scales to 1.5x)
        start_time: Starting timestamp for filtering the trace
        duration: Duration of time window in seconds (used for both filtering and as makespan)
        trace_file: Path to the original trace file
        selection_method: Method for selecting tenants ('top_n', 'similar_frequency', 'random', 'balanced')
        seed: Random seed for reproducible results (used for random selection method and sampling)

    Returns:
        List of (tenant_label, interval) tuples where interval is between consecutive requests
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    # Load trace
    trace_data = load_trace(trace_file)

    # Print timestamp range for reference
    print_timestamp_range(trace_data)

    # Filter trace by time window
    trace_data = filter_trace_by_time_window(trace_data, start_time, duration)
    if len(trace_data) == 0:
        raise ValueError(
            f"No requests found in time window [{start_time}, {start_time + duration}]"
        )

    # Analyze tenants from filtered trace
    tenant_counts = analyze_tenants(trace_data)

    num_active_tenants = len(tenant_counts)
    print(
        f"\nActive tenants in time window [{start_time:.2f}, {start_time + duration:.2f}]: {num_active_tenants}"
    )

    # Select tenants based on the specified method
    selected_tenants = select_tenants(tenant_counts, num_tenants, selection_method)

    if len(selected_tenants) < num_tenants:
        print(
            f"Warning: Only {len(selected_tenants)} tenants available, requested {num_tenants}"
        )

    print(f"\nSelection method: {selection_method}")

    # Create mapping from tenant_id to label (A, B, C, D...)
    tenant_id_to_label = {}
    selected_tenant_ids = set()
    print(f"\nSelected {len(selected_tenants)} tenants:")
    for i, (tenant_id, count) in enumerate(selected_tenants):
        tenant_label = chr(65 + i)  # A=65, B=66, C=67, D=68, ...
        tenant_id_to_label[tenant_id] = tenant_label
        selected_tenant_ids.add(tenant_id)
        print(f"  {tenant_label}: {tenant_id} ({count} requests)")

    # Go through original trace in order and collect requests from selected tenants
    # Also collect timestamps to extract interval patterns
    original_requests = []
    original_request_data = []  # Store (tenant_id, timestamp) pairs

    for request in trace_data:
        tenant_id = create_tenant_id(request)
        if tenant_id in selected_tenant_ids:
            timestamp = request.get("timestamp", 0)
            original_requests.append(tenant_id)
            original_request_data.append((tenant_id, timestamp))

    print(
        f"\nFound {len(original_requests)} requests from selected tenants in original trace"
    )

    # Calculate global intervals between consecutive requests (any tenant) based on timestamp
    # Sort requests by timestamp to get global order
    sorted_request_data = sorted(
        original_request_data, key=lambda x: x[1]
    )  # Sort by timestamp

    # Calculate intervals between consecutive requests in global timestamp order
    global_original_intervals = []
    if len(sorted_request_data) > 1:
        for i in range(len(sorted_request_data) - 1):
            interval = (
                sorted_request_data[i + 1][1] - sorted_request_data[i][1]
            )  # timestamp difference
            global_original_intervals.append(interval)
    else:
        global_original_intervals = []

    # Scale the requests: tenants are already selected, now scale the number of requests
    # Unified approach: duplicate to ceiling(request_scale), then sample down to target
    num_original_requests = len(sorted_request_data)
    num_target_requests = int(num_original_requests * request_scale)

    # Ensure at least 1 request if we have any
    if num_original_requests > 0:
        num_target_requests = max(1, num_target_requests)

    scaled_request_data = []
    scaled_duplicated_positions = (
        []
    )  # Track positions in duplicated sequence for interval calculation

    if num_original_requests > 0:
        # Step 1: Duplicate to ceiling(request_scale) times
        ceiling_scale = math.ceil(request_scale)
        duplicated_request_data = []

        for rep in range(ceiling_scale):
            duplicated_request_data.extend(sorted_request_data)

        # Step 2: Sample down to target number if needed
        num_duplicated = len(duplicated_request_data)
        if num_target_requests < num_duplicated:
            # Sample indices to keep, preserving relative order
            indices_to_keep = sorted(
                random.sample(range(num_duplicated), num_target_requests)
            )
            scaled_request_data = [duplicated_request_data[i] for i in indices_to_keep]
            scaled_duplicated_positions = indices_to_keep
        else:
            scaled_request_data = duplicated_request_data
            scaled_duplicated_positions = list(range(num_duplicated))
    else:
        scaled_request_data = sorted_request_data
        scaled_duplicated_positions = []

    print(
        f"Scaled from {num_original_requests} to {len(scaled_request_data)} requests (scale: {request_scale}x)"
    )

    # Calculate intervals based on unified scaling approach
    # First duplicate intervals, then adjust based on actual final count
    if len(scaled_request_data) > 1 and len(global_original_intervals) > 0:
        # Step 1: Duplicate intervals to match ceiling(request_scale) duplication
        ceiling_scale = math.ceil(request_scale)
        duplicated_intervals = []

        for _ in range(ceiling_scale):
            duplicated_intervals.extend(global_original_intervals)

        # Step 2: Calculate intervals between consecutive sampled requests
        # The scaled_duplicated_positions refer to positions in the duplicated sequence
        if len(scaled_duplicated_positions) > 1:
            # Calculate intervals between consecutive sampled requests
            # Note: intervals[i] is the interval between request i and i+1
            # So to get interval between request start_pos and end_pos, we sum intervals[start_pos:end_pos]
            global_intervals_unscaled = []
            for i in range(len(scaled_duplicated_positions) - 1):
                start_pos = scaled_duplicated_positions[i]
                end_pos = scaled_duplicated_positions[i + 1]
                # Sum all intervals between start_pos and end_pos in duplicated sequence
                if start_pos < len(duplicated_intervals):
                    if end_pos <= len(duplicated_intervals):
                        interval_sum = sum(duplicated_intervals[start_pos:end_pos])
                    else:
                        # Handle edge case where end_pos exceeds duplicated intervals
                        interval_sum = sum(duplicated_intervals[start_pos:])
                else:
                    interval_sum = 0.0
                global_intervals_unscaled.append(interval_sum)
        else:
            global_intervals_unscaled = []

        # Step 3: Adjust intervals proportionally to match request_scale
        # The total time should remain roughly the same, but adjusted for the actual scale
        if len(global_intervals_unscaled) > 0:
            interval_scale_factor = 1.0 / request_scale
            global_intervals_unscaled = [
                interval * interval_scale_factor
                for interval in global_intervals_unscaled
            ]
    else:
        global_intervals_unscaled = []

    # Extract tenant IDs from scaled request data
    scaled_requests = [tenant_id for tenant_id, _ in scaled_request_data]

    # Calculate the original total time span from the original trace
    # This is the time from first request to last request across all selected tenants
    if original_request_data:
        original_first_timestamp = min(
            timestamp for _, timestamp in original_request_data
        )
        original_last_timestamp = max(
            timestamp for _, timestamp in original_request_data
        )
        original_total_span = original_last_timestamp - original_first_timestamp
    else:
        original_total_span = 1.0  # Fallback

    # Calculate what the total time span would be with unscaled intervals
    # The makespan is the sum of all global intervals (single global timeline)
    total_unscaled_time = (
        sum(global_intervals_unscaled) if global_intervals_unscaled else 0.0
    )

    # If total_unscaled_time is 0, use a default
    if total_unscaled_time == 0.0:
        total_unscaled_time = 1.0

    # Calculate global scale factor to fit within duration
    global_scale_factor = (
        duration / total_unscaled_time if total_unscaled_time > 0 else 1.0
    )

    # Apply global scaling to all intervals
    global_intervals = []
    for interval in global_intervals_unscaled:
        global_intervals.append(interval * global_scale_factor)

    # Verify the total time span fits within duration
    # Calculate the actual total time after scaling
    actual_total_time = sum(global_intervals) if global_intervals else 0.0

    # If actual_total_time exceeds duration, scale down proportionally
    if actual_total_time > duration + 1e-6:
        additional_scale = duration / actual_total_time
        global_intervals = [
            interval * additional_scale for interval in global_intervals
        ]
        actual_total_time = duration

    print(f"Global makespan: {duration}s, actual total time: {actual_total_time:.6f}s")

    # Generate trace entries in the scaled order, with intervals
    # Intervals are based on global order (consecutive requests of any tenant)
    # Each request's interval is the time gap to the next request
    trace_entries = []

    for i, tenant_id in enumerate(scaled_requests):
        tenant_label = tenant_id_to_label[tenant_id]
        # Interval is the time gap to the next request
        # Last request has interval 0.0 (no next request)
        if i < len(global_intervals):
            interval = global_intervals[i]
        else:
            interval = 0.0  # Last request has no next request
        trace_entries.append((tenant_label, interval))

    print(f"\nGenerated {len(trace_entries)} trace entries")
    print(f"  Request scale: {request_scale}x")
    print(f"  Global makespan: {duration}s ({duration/60:.1f} minutes)")

    return trace_entries


def write_trace_file(trace_entries: List[tuple], output_file: str):
    """Write trace entries to output file, one per line."""
    with open(output_file, "w") as f:
        for tenant, interval in trace_entries:
            f.write(f"{tenant} {interval}\n")
    print(f"\nTrace written to {output_file}")


def load_trace_file(trace_file: str) -> List[Tuple[str, float]]:
    """
    Load trace data from file.

    Format: Each line contains "tenant_label interval"
    Example:
        A 10.5
        B 2.3
        A 5.1

    Returns:
        List of (tenant_label, interval) tuples
    """
    trace_entries = []

    try:
        with open(trace_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()
                if len(parts) != 2:
                    print(f"Warning: Skipping invalid line {line_num}: {line}")
                    continue

                tenant_label = parts[0]
                try:
                    interval = float(parts[1])
                    trace_entries.append((tenant_label, interval))
                except ValueError:
                    print(
                        f"Warning: Skipping line {line_num} with invalid interval: {line}"
                    )
                    continue

        print(f"Loaded {len(trace_entries)} trace entries from {trace_file}")
        return trace_entries

    except FileNotFoundError:
        print(f"Error: File {trace_file} not found")
        sys.exit(1)


def reconstruct_timeline(
    trace_entries: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """
    Reconstruct timeline from intervals.

    Args:
        trace_entries: List of (tenant_label, interval) tuples

    Returns:
        List of (tenant_label, timestamp) tuples where timestamp is cumulative time
    """
    timeline = []
    current_time = 0.0

    for tenant_label, interval in trace_entries:
        timeline.append((tenant_label, current_time))
        current_time += interval

    return timeline


def analyze_trace(trace_entries: List[Tuple[str, float]]) -> Dict:
    """
    Analyze trace and compute statistics.

    Returns:
        Dictionary containing various statistics
    """
    if not trace_entries:
        return {}

    # Reconstruct timeline
    timeline = reconstruct_timeline(trace_entries)

    # Extract data
    tenants = [t for t, _ in trace_entries]
    intervals = [interval for _, interval in trace_entries]

    # Tenant statistics
    tenant_counts = Counter(tenants)
    tenant_intervals = defaultdict(list)
    for tenant, interval in trace_entries:
        tenant_intervals[tenant].append(interval)

    # Timeline statistics
    timestamps = [ts for _, ts in timeline]
    total_duration = max(timestamps) if timestamps else 0.0

    # Calculate per-tenant statistics
    tenant_stats = {}
    for tenant in tenant_counts.keys():
        tenant_ints = tenant_intervals[tenant]
        tenant_stats[tenant] = {
            "count": tenant_counts[tenant],
            "total_intervals": len(tenant_ints),
            "mean_interval": statistics.mean(tenant_ints) if tenant_ints else 0.0,
            "median_interval": statistics.median(tenant_ints) if tenant_ints else 0.0,
            "min_interval": min(tenant_ints) if tenant_ints else 0.0,
            "max_interval": max(tenant_ints) if tenant_ints else 0.0,
            "std_interval": (
                statistics.stdev(tenant_ints) if len(tenant_ints) > 1 else 0.0
            ),
        }

    # Overall statistics
    overall_stats = {
        "total_requests": len(trace_entries),
        "total_duration": total_duration,
        "num_tenants": len(tenant_counts),
        "mean_interval": statistics.mean(intervals) if intervals else 0.0,
        "median_interval": statistics.median(intervals) if intervals else 0.0,
        "min_interval": min(intervals) if intervals else 0.0,
        "max_interval": max(intervals) if intervals else 0.0,
        "std_interval": statistics.stdev(intervals) if len(intervals) > 1 else 0.0,
        "requests_per_second": (
            len(trace_entries) / total_duration if total_duration > 0 else 0.0
        ),
    }

    return {
        "trace_entries": trace_entries,
        "timeline": timeline,
        "tenant_counts": dict(tenant_counts),
        "tenant_intervals": dict(tenant_intervals),
        "tenant_stats": tenant_stats,
        "overall_stats": overall_stats,
    }


def print_statistics(analysis: Dict):
    """Print analysis statistics to console."""
    if not analysis:
        print("No data to analyze")
        return

    stats = analysis["overall_stats"]
    tenant_stats = analysis["tenant_stats"]

    print("\n" + "=" * 80)
    print("Trace Analysis Summary")
    print("=" * 80)
    print(f"Total Requests: {stats['total_requests']}")
    print(
        f"Total Duration: {stats['total_duration']:.2f} seconds ({stats['total_duration']/60:.2f} minutes)"
    )
    print(f"Number of Tenants: {stats['num_tenants']}")
    print(f"Overall Requests Per Second: {stats['requests_per_second']:.4f}")

    print(f"\nOverall Interval Statistics:")
    print(f"  Mean: {stats['mean_interval']:.4f} seconds")
    print(f"  Median: {stats['median_interval']:.4f} seconds")
    print(f"  Min: {stats['min_interval']:.4f} seconds")
    print(f"  Max: {stats['max_interval']:.4f} seconds")
    print(f"  Std Dev: {stats['std_interval']:.4f} seconds")

    print(f"\nPer-Tenant Statistics:")
    print(
        f"{'Tenant':<10} {'Requests':<12} {'Mean Int':<12} {'Median Int':<12} {'Min Int':<12} {'Max Int':<12}"
    )
    print("-" * 80)

    for tenant in sorted(tenant_stats.keys()):
        ts = tenant_stats[tenant]
        print(
            f"{tenant:<10} {ts['count']:<12} {ts['mean_interval']:<12.4f} {ts['median_interval']:<12.4f} "
            f"{ts['min_interval']:<12.4f} {ts['max_interval']:<12.4f}"
        )


def plot_timeline(analysis: Dict, output_file: str = None, show_plot: bool = True):
    """
    Plot timeline showing when each tenant makes requests.

    Creates a scatter plot with different colors for each tenant.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib is not available. Cannot generate plot.")
        return

    timeline = analysis["timeline"]
    if not timeline:
        print("No timeline data to plot")
        return

    # Separate by tenant
    tenant_timelines = defaultdict(list)
    for tenant, timestamp in timeline:
        tenant_timelines[tenant].append(timestamp)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Assign colors to tenants
    tenants = sorted(tenant_timelines.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(tenants)))

    for i, tenant in enumerate(tenants):
        timestamps = tenant_timelines[tenant]
        y_pos = [i] * len(timestamps)  # Position on y-axis
        ax.scatter(
            timestamps,
            y_pos,
            label=f"Tenant {tenant} ({len(timestamps)} requests)",
            color=colors[i],
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Tenant", fontsize=12)
    ax.set_title("Request Timeline by Tenant", fontsize=14, fontweight="bold")
    ax.set_yticks(range(len(tenants)))
    ax.set_yticklabels(tenants)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nTimeline plot saved to {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_interval_distribution(
    analysis: Dict, output_file: str = None, show_plot: bool = True
):
    """
    Plot interval distribution per tenant.

    Creates histograms showing the distribution of intervals for each tenant.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib is not available. Cannot generate plot.")
        return

    tenant_intervals = analysis["tenant_intervals"]
    if not tenant_intervals:
        print("No interval data to plot")
        return

    tenants = sorted(tenant_intervals.keys())
    num_tenants = len(tenants)

    # Create subplots
    fig, axes = plt.subplots(num_tenants, 1, figsize=(12, 3 * num_tenants))
    if num_tenants == 1:
        axes = [axes]

    for i, tenant in enumerate(tenants):
        intervals = tenant_intervals[tenant]
        ax = axes[i]

        ax.hist(
            intervals, bins=min(30, len(set(intervals))), alpha=0.7, edgecolor="black"
        )
        ax.set_xlabel("Interval (seconds)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(
            f"Tenant {tenant} - Interval Distribution (n={len(intervals)})",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats = analysis["tenant_stats"][tenant]
        stats_text = (
            f"Mean: {stats['mean_interval']:.2f}s | "
            f"Median: {stats['median_interval']:.2f}s | "
            f"Min: {stats['min_interval']:.2f}s | "
            f"Max: {stats['max_interval']:.2f}s"
        )
        ax.text(
            0.5,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nInterval distribution plot saved to {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_request_frequency(
    analysis: Dict, output_file: str = None, show_plot: bool = True
):
    """
    Plot request frequency per tenant (bar chart).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib is not available. Cannot generate plot.")
        return

    tenant_counts = analysis["tenant_counts"]
    if not tenant_counts:
        print("No tenant data to plot")
        return

    tenants = sorted(tenant_counts.keys())
    counts = [tenant_counts[t] for t in tenants]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(tenants, counts, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Tenant", fontsize=12)
    ax.set_ylabel("Number of Requests", fontsize=12)
    ax.set_title("Request Frequency by Tenant", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nRequest frequency plot saved to {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_cumulative_requests(
    analysis: Dict, output_file: str = None, show_plot: bool = True
):
    """
    Plot cumulative number of requests over time, separated by tenant.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib is not available. Cannot generate plot.")
        return

    timeline = analysis["timeline"]
    if not timeline:
        print("No timeline data to plot")
        return

    # Build cumulative counts per tenant
    tenant_cumulative = defaultdict(list)
    tenant_timestamps = defaultdict(list)

    cumulative_counts = defaultdict(int)
    for tenant, timestamp in timeline:
        cumulative_counts[tenant] += 1
        tenant_cumulative[tenant].append(cumulative_counts[tenant])
        tenant_timestamps[tenant].append(timestamp)

    fig, ax = plt.subplots(figsize=(12, 6))

    tenants = sorted(tenant_cumulative.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(tenants)))

    for i, tenant in enumerate(tenants):
        timestamps = tenant_timestamps[tenant]
        cumulative = tenant_cumulative[tenant]
        ax.plot(
            timestamps,
            cumulative,
            label=f"Tenant {tenant}",
            color=colors[i],
            linewidth=2,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Cumulative Requests", fontsize=12)
    ax.set_title(
        "Cumulative Requests Over Time by Tenant", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nCumulative requests plot saved to {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all(analysis: Dict, output_prefix: str = None, show_plot: bool = True):
    """
    Generate all plots.
    """
    if output_prefix:
        timeline_file = f"{output_prefix}_timeline.png"
        interval_file = f"{output_prefix}_intervals.png"
        frequency_file = f"{output_prefix}_frequency.png"
        cumulative_file = f"{output_prefix}_cumulative.png"
    else:
        timeline_file = interval_file = frequency_file = cumulative_file = None

    plot_timeline(analysis, timeline_file, show_plot)
    plot_interval_distribution(analysis, interval_file, show_plot)
    plot_request_frequency(analysis, frequency_file, show_plot)
    plot_cumulative_requests(analysis, cumulative_file, show_plot)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate trace files with tenant and interval pairs and visualize them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate trace and visualize with all plots
  python trace_generator.py --num-tenants 4 --request-scale 2.0 --start-time 1000 --duration 600

Selection Methods (for generation):
  - top_n: Select top N tenants by request count (default)
  - similar_frequency: Select tenants with similar request frequencies
  - random: Randomly select tenants
  - balanced: Select a balanced mix of high, medium, and low frequency tenants

Note: The trace is first filtered to include only requests within the time window [start_time, start_time + duration],
      then tenants are selected from the filtered trace. After tenant selection, requests are scaled using a unified approach:
      - First duplicate requests to ceiling(request_scale) times
      - Then sample down to the target number if needed (using seed for reproducibility)
      - Intervals are duplicated accordingly and then adjusted proportionally
      The duration parameter is used both for filtering and as the makespan (total time span from first to last request
      across all tenants). The output file contains intervals between consecutive requests, which are scaled
      proportionally to fit within the specified duration.
        """,
    )

    # Generation arguments
    parser.add_argument(
        "--num-tenants", type=int, help="Number of tenants to generate (e.g., 4, 8)"
    )
    parser.add_argument(
        "--request-scale",
        type=float,
        help="Scale factor for number of requests (e.g., 2.0 to double requests)",
    )
    parser.add_argument(
        "--selection-method",
        default="top_n",
        choices=["top_n", "similar_frequency", "random", "balanced"],
        help="Method for selecting tenants: top_n (top N by count), "
        "similar_frequency (similar request frequencies), "
        "random (random selection), balanced (mix of high/medium/low frequency)",
    )
    parser.add_argument(
        "--start-time", type=float, help="Starting timestamp for filtering the trace"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration of time window in seconds (used for both filtering and as makespan)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (used with random selection method and sampling)",
    )
    parser.add_argument(
        "--trace-file",
        default="diffusion_model_request_trace.json",
        help="Path to input trace file (default: diffusion_model_request_trace.json)",
    )
    parser.add_argument(
        "--output-file",
        default="generated_trace.txt",
        help="Path to output trace file (default: generated_trace.txt)",
    )

    args = parser.parse_args()

    # Check required arguments
    if not all([args.num_tenants, args.request_scale, args.start_time, args.duration]):
        parser.error(
            "Generation requires --num-tenants, --request-scale, --start-time, and --duration"
        )

    # Generate trace
    trace_entries = generate_trace(
        num_tenants=args.num_tenants,
        request_scale=args.request_scale,
        start_time=args.start_time,
        duration=args.duration,
        trace_file=args.trace_file,
        selection_method=args.selection_method,
        seed=args.seed,
    )

    # Write to file
    write_trace_file(trace_entries, args.output_file)

    # Print summary
    tenant_dist = Counter(tenant for tenant, _ in trace_entries)
    print(f"\n=== SUMMARY ===")
    print(f"Total entries: {len(trace_entries)}")
    print(f"Tenant distribution:")
    for tenant in sorted(tenant_dist.keys()):
        print(f"  {tenant}: {tenant_dist[tenant]} requests")

    # Always analyze and visualize trace
    print("\nAnalyzing trace...")
    analysis = analyze_trace(trace_entries)

    # Print statistics
    print_statistics(analysis)

    # Generate all plots
    # Use output file name (without extension) as plot prefix
    plot_prefix = os.path.splitext(args.output_file)[0] if args.output_file else None
    plot_all(analysis, plot_prefix, show_plot=True)


if __name__ == "__main__":
    main()