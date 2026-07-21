import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

BASELINES = ("basic", "clockwork", "shepherd", "diflow")
SETTING = "flux.1-schnell"
LOG_ROOT = Path(__file__).resolve().parents[1] / "ae-results" / "client_logs"
LOG_TIMESTAMP_RE = re.compile(r"benchmark_client_(\d{8}_\d{6})\.log$")
RS_RE = re.compile(r"generated_trace_[^\s/]*_rs([0-9]+(?:\.[0-9]+)?)_")
SLO_RE = re.compile(r"SLO Attainment:\s*([0-9]+(?:\.[0-9]+)?)%")


def _log_sort_key(log_path):
    match = LOG_TIMESTAMP_RE.search(log_path.name)
    if match:
        return (match.group(1), log_path.stat().st_mtime)
    return ("", log_path.stat().st_mtime)


def _parse_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rs_match = RS_RE.search(text)
    slo_matches = SLO_RE.findall(text)
    if not rs_match or not slo_matches:
        return None
    return float(rs_match.group(1)), float(slo_matches[-1])


def parse_results(log_root=LOG_ROOT):
    parsed_results = {baseline: {SETTING: {}} for baseline in BASELINES}

    for baseline in BASELINES:
        baseline_dir = log_root / baseline
        latest_by_rs = {}
        for log_path in baseline_dir.glob("*.log"):
            parsed_log = _parse_log(log_path)
            if parsed_log is None:
                continue
            rs, slo_attainment = parsed_log
            log_sort_key = _log_sort_key(log_path)
            if rs not in latest_by_rs or log_sort_key > latest_by_rs[rs][0]:
                latest_by_rs[rs] = (log_sort_key, slo_attainment)

        parsed_results[baseline][SETTING] = {
            rs: latest_by_rs[rs][1] for rs in sorted(latest_by_rs)
        }

    return parsed_results


results = parse_results()

def print_improvement(results_dict):
    display_names = {
        "diflow": "DiFlow\t",
        "basic": "Diffusers",
        "clockwork": "Diffusers-C",
        "shepherd": "Diffusers-S",
    }
    baseline_order = ["basic", "clockwork", "shepherd", "diflow"]
    rs_values = sorted({
        rs
        for baseline in baseline_order
        for rs in results_dict.get(baseline, {}).get(SETTING, {})
    })

    print("\t\t" + "\t".join(f"RS={rs:g}" for rs in rs_values))
    for baseline in baseline_order:
        data = results_dict.get(baseline, {}).get(SETTING, {})
        values = [
            f"{_aggregate_value(data[rs]):.2f}" if rs in data else ""
            for rs in rs_values
        ]
        print(f"{display_names[baseline]}\t" + "\t".join(values))


def _aggregate_value(v):
    if isinstance(v, (list, tuple)):
        return sum(v) / len(v) if v else float("nan")
    return v


def plot_results(results_dict):
    baseline_order = ["basic", "clockwork", "shepherd", "diflow"]
    baselines = [b for b in baseline_order if b in results_dict]

    setting = SETTING
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharey=True)
    fontsize = 28

    colors = {
        "basic": "tab:blue",
        "clockwork": "tab:orange",
        "shepherd": "tab:red",
        "diflow": "tab:green",
    }

    linestyles = {
        "diflow": "-",
        "basic": "--",
        "clockwork": "-.",
        "shepherd": ":",
    }

    display_names = {
        "diflow": "DiFlow",
        "basic": "Diffusers",
        "clockwork": "Diffusers-C",
        "shepherd": "Diffusers-S",
    }

    all_xs = sorted({
        rs
        for baseline in baselines
        for rs in results_dict[baseline].get(setting, {})
    })
    x_positions = {rs: idx for idx, rs in enumerate(all_xs)}

    for baseline in baselines:
        data = results_dict[baseline].get(setting, {})
        if not data:
            continue
        xs = sorted(data.keys())
        ys = [_aggregate_value(data[x]) for x in xs]
        ax.plot(
            [x_positions[x] for x in xs],
            ys,
            marker="o",
            label=display_names.get(baseline, baseline),
            color=colors.get(baseline),
            linestyle=linestyles.get(baseline, "-"),
            linewidth=3,
            markersize=10,
        )

    ax.set_title("Flux.1 Schnell", fontsize=fontsize)
    ax.set_xlabel("Rate Scale", fontsize=fontsize-4)
    ax.set_ylabel("SLO Attainment (%)", fontsize=fontsize-4)
    ax.set_xticks(range(len(all_xs)))
    ax.set_xticklabels([str(x) for x in all_xs], fontsize=fontsize-4)
    ax.set_yticks([0, 50, 90, 100])
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=fontsize-4)
    ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        fontsize=fontsize-2,
        handlelength=3.0
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.98], w_pad=0.03, h_pad=0.03)
    return fig, ax


if __name__ == "__main__":
    print_improvement(results)
    fig, axes = plot_results(results)
    fig.savefig("./ae-results/e2e_results.pdf", bbox_inches="tight", pad_inches=0.03)
