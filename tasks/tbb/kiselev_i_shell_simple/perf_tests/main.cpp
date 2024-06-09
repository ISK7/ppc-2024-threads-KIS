// Copyright 2024 Kiselev Igor
#include <gtest/gtest.h>
#include <oneapi/tbb.h>
#include <tbb/tbb.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/kiselev_i_shell_simple/include/shell_simple.hpp"

using namespace Kiselev_tbb;

TEST(kiselev_i_shell_simple_tbb, test_pipeline_run) {
  const int count = 10000;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);
  std::vector<int> res(count, 0);
  for (int i = 0; i < count; i++) {
    in[i] = count - 1 - i;
    res[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTBB->inputs_count.emplace_back(in.size());
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  // Create Task
  auto kiselevTaskTBB = std::make_shared<KiselevTaskTBB>(taskDataTBB);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(kiselevTaskTBB);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, res);
}

// task_run

TEST(kiselev_i_shell_simple_tbb, test_task_run) {
  const int count = 10000000;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);
  std::vector<int> res(count, 0);
  for (int i = 0; i < count; i++) {
    in[i] = count - 1 - i;
    res[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTBB->inputs_count.emplace_back(in.size());
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  // Create Task
  auto kiselevTaskTBB = std::make_shared<KiselevTaskTBB>(taskDataTBB);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(kiselevTaskTBB);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(out, res);
}
