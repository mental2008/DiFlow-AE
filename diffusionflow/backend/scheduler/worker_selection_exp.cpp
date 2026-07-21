#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using TensorSource = std::tuple<int, int, long long>;  // worker rank, host id, bytes

struct Candidate {
  int worker_rank = -1;
  std::size_t worker_index = 0;
  double execution_latency = std::numeric_limits<double>::infinity();
  double overall_latency = std::numeric_limits<double>::infinity();
};

double lookup_fetch_latency_seconds(
    long long tensor_size_bytes,
    const std::vector<long long>& block_sizes,
    const std::vector<double>& fetch_overheads_us) {
  if (block_sizes.empty() || fetch_overheads_us.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const auto block_iter =
      std::lower_bound(block_sizes.begin(), block_sizes.end(), tensor_size_bytes);
  std::size_t index = 0;
  if (block_iter == block_sizes.end()) {
    index = block_sizes.size() - 1;
  } else {
    index = static_cast<std::size_t>(block_iter - block_sizes.begin());
  }
  if (index >= fetch_overheads_us.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return fetch_overheads_us[index] / 1e6;
}

std::pair<int, double> select_worker(
    const std::vector<int>& worker_ranks,
    const std::vector<int>& worker_host_ids,
    const std::vector<double>& queue_latencies,
    const std::vector<bool>& model_loaded,
    double worker_latency_threshold,
    double loading_latency,
    const std::vector<std::vector<TensorSource>>& tensor_sources,
    const std::vector<long long>& intra_block_sizes,
    const std::vector<double>& intra_fetch_overheads_us,
    const std::vector<long long>& inter_block_sizes,
    const std::vector<double>& inter_fetch_overheads_us) {
  if (worker_ranks.size() != worker_host_ids.size() ||
      worker_ranks.size() != queue_latencies.size() ||
      worker_ranks.size() != model_loaded.size()) {
    throw std::invalid_argument(
        "worker_ranks, worker_host_ids, queue_latencies, and model_loaded must have the same length");
  }

  int selected_worker_rank = -1;
  double selected_execution_latency = std::numeric_limits<double>::infinity();

  py::gil_scoped_release release;

  auto score_worker_range = [&](std::size_t begin, std::size_t end) {
    Candidate best;
    for (std::size_t dst_index = begin; dst_index < end; ++dst_index) {
      const double queue_latency = queue_latencies[dst_index];
      if (queue_latency > worker_latency_threshold) {
        continue;
      }

      const int dst_worker_rank = worker_ranks[dst_index];
      const int dst_host_id = worker_host_ids[dst_index];
      double tensor_transfer_latency = 0.0;

      for (const auto& sources_for_tensor : tensor_sources) {
        if (sources_for_tensor.empty()) {
          continue;
        }

        // Match Python behavior: prefer a source worker on the destination host,
        // otherwise use the first source listed for that tensor.
        const TensorSource* selected_source = &sources_for_tensor.front();
        for (const auto& source : sources_for_tensor) {
          const int src_host_id = std::get<1>(source);
          if (src_host_id == dst_host_id) {
            selected_source = &source;
            break;
          }
        }

        const int src_worker_rank = std::get<0>(*selected_source);
        if (src_worker_rank == dst_worker_rank) {
          continue;
        }

        const long long tensor_size_bytes = std::get<2>(*selected_source);
        const bool intra_node = std::get<1>(*selected_source) == dst_host_id;
        if (intra_node) {
          tensor_transfer_latency += lookup_fetch_latency_seconds(
              tensor_size_bytes, intra_block_sizes, intra_fetch_overheads_us);
        } else {
          tensor_transfer_latency += lookup_fetch_latency_seconds(
              tensor_size_bytes, inter_block_sizes, inter_fetch_overheads_us);
        }
      }

      const double model_loading_latency =
          model_loaded[dst_index] ? 0.0 : loading_latency;
      const double execution_latency = tensor_transfer_latency + model_loading_latency;
      const double estimated_overall_latency = queue_latency + execution_latency;

      if (estimated_overall_latency < best.overall_latency) {
        best.overall_latency = estimated_overall_latency;
        best.execution_latency = execution_latency;
        best.worker_rank = dst_worker_rank;
        best.worker_index = dst_index;
      }
    }
    return best;
  };

  Candidate best_candidate;
  const std::size_t worker_count = worker_ranks.size();
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  const std::size_t thread_count =
      worker_count >= 64 && hardware_threads > 1
          ? std::min<std::size_t>(worker_count, hardware_threads)
          : 1;

  std::cerr << "[worker_selection] using " << thread_count
      << " threads for " << worker_count << " workers" << std::endl;

  if (thread_count == 1) {
    best_candidate = score_worker_range(0, worker_count);
  } else {
    std::vector<Candidate> thread_candidates(thread_count);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
      const std::size_t begin = worker_count * thread_index / thread_count;
      const std::size_t end = worker_count * (thread_index + 1) / thread_count;
      threads.emplace_back([&, thread_index, begin, end]() {
        thread_candidates[thread_index] = score_worker_range(begin, end);
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }

    // Reduce in worker order so ties behave like the serial Python scan.
    for (const auto& candidate : thread_candidates) {
      if (candidate.worker_rank != -1 &&
          (candidate.overall_latency < best_candidate.overall_latency ||
           (candidate.overall_latency == best_candidate.overall_latency &&
            candidate.worker_index < best_candidate.worker_index))) {
        best_candidate = candidate;
      }
    }
  }

  selected_worker_rank = best_candidate.worker_rank;
  selected_execution_latency = best_candidate.execution_latency;

  return {selected_worker_rank, selected_execution_latency};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "C++ worker-selection helpers for DiffusionFlow dynamic scheduling";
  m.def(
      "select_worker",
      &select_worker,
      "Select the minimum-latency worker for a task group");
}
