// Copyright 2024 Kiselev Igor
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "tbb/kiselev_i_shell_simple/include/shell_simple.hpp"

using namespace Kiselev_tbb;

TEST(kiselev_i_shell_simple_tbb, check_8_size) {
  std::vector<int> arr = {10, 5, 3, 8, 12, 6, 9, 2, 7, 1, 4, 11};
  std::vector<int> res(arr.size(), 0);
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  taskDataTBB->inputs_count.emplace_back(arr.size());
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataTBB->outputs_count.emplace_back(arr.size());

  KiselevTaskTBB testTaskSequential(taskDataTBB);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  ASSERT_EQ(expected, res);
}

TEST(kiselev_i_shell_simple_tbb, check_100_size_reverse) {
  const int count = 100;

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
  KiselevTaskTBB kiselevTaskTBB(taskDataTBB);
  ASSERT_EQ(kiselevTaskTBB.validation(), true);
  ASSERT_EQ(kiselevTaskTBB.pre_processing(), true);
  ASSERT_EQ(kiselevTaskTBB.run(), true);
  ASSERT_EQ(kiselevTaskTBB.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_tbb, check_100_size_sorted) {
  const int count = 100;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);
  std::vector<int> res(count, 0);
  for (int i = 0; i < count; i++) {
    in[i] = i;
    res[i] = i;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTBB->inputs_count.emplace_back(in.size());
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskTBB kiselevTaskTBB(taskDataTBB);
  ASSERT_EQ(kiselevTaskTBB.validation(), true);
  ASSERT_EQ(kiselevTaskTBB.pre_processing(), true);
  ASSERT_EQ(kiselevTaskTBB.run(), true);
  ASSERT_EQ(kiselevTaskTBB.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_tbb, check_100_size_bias) {
  const int count = 100;
  const int bias = 50;

  // Create data
  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);
  std::vector<int> res(count, 0);
  for (int i = 0; i < count; i++) {
    in[i] = count - 1 - i + bias;
    res[i] = i + bias;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTBB->inputs_count.emplace_back(in.size());
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskTBB kiselevTaskTBB(taskDataTBB);
  ASSERT_EQ(kiselevTaskTBB.validation(), true);
  ASSERT_EQ(kiselevTaskTBB.pre_processing(), true);
  ASSERT_EQ(kiselevTaskTBB.run(), true);
  ASSERT_EQ(kiselevTaskTBB.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_tbb, check_incorrect_input) {
  const int count = 4;

  std::vector<std::vector<int>> in = {{2, 1}, {4, 3}};
  std::vector<int> out(count, 0);
  std::vector<int> res = {1, 2, 3, 4};

  std::shared_ptr<ppc::core::TaskData> taskDataTBB = std::make_shared<ppc::core::TaskData>();
  taskDataTBB->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataTBB->inputs_count.emplace_back(in.size());
  taskDataTBB->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataTBB->outputs_count.emplace_back(out.size());

  KiselevTaskTBB kiselevTaskTBB(taskDataTBB);
  ASSERT_EQ(kiselevTaskTBB.validation(), false);
}
