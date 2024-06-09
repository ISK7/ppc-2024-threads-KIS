// Copyright 2024 Kiselev Igor
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "stl/kiselev_i_shell_simple/include/shell_simple.hpp"

using namespace Kiselev_stl;

TEST(kiselev_i_shell_simple_stl, check_8_size) {
  std::vector<int> arr = {10, 5, 3, 8, 12, 6, 9, 2, 7, 1, 4, 11};
  std::vector<int> res(arr.size(), 0);
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  std::shared_ptr<ppc::core::TaskData> taskDataSTL = std::make_shared<ppc::core::TaskData>();
  taskDataSTL->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
  taskDataSTL->inputs_count.emplace_back(arr.size());
  taskDataSTL->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSTL->outputs_count.emplace_back(arr.size());

  KiselevTaskSTL testTaskSequential(taskDataSTL);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_EQ(testTaskSequential.run(), true);
  testTaskSequential.post_processing();
  ASSERT_EQ(expected, res);
}

TEST(kiselev_i_shell_simple_stl, check_100_size_reverse) {
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
  std::shared_ptr<ppc::core::TaskData> taskDataSTL = std::make_shared<ppc::core::TaskData>();
  taskDataSTL->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSTL->inputs_count.emplace_back(in.size());
  taskDataSTL->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSTL->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskSTL kiselevTaskSTL(taskDataSTL);
  ASSERT_EQ(kiselevTaskSTL.validation(), true);
  ASSERT_EQ(kiselevTaskSTL.pre_processing(), true);
  ASSERT_EQ(kiselevTaskSTL.run(), true);
  ASSERT_EQ(kiselevTaskSTL.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_stl, check_100_size_sorted) {
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
  std::shared_ptr<ppc::core::TaskData> taskDataSTL = std::make_shared<ppc::core::TaskData>();
  taskDataSTL->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSTL->inputs_count.emplace_back(in.size());
  taskDataSTL->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSTL->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskSTL kiselevTaskSTL(taskDataSTL);
  ASSERT_EQ(kiselevTaskSTL.validation(), true);
  ASSERT_EQ(kiselevTaskSTL.pre_processing(), true);
  ASSERT_EQ(kiselevTaskSTL.run(), true);
  ASSERT_EQ(kiselevTaskSTL.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_stl, check_100_size_bias) {
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
  std::shared_ptr<ppc::core::TaskData> taskDataSTL = std::make_shared<ppc::core::TaskData>();
  taskDataSTL->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSTL->inputs_count.emplace_back(in.size());
  taskDataSTL->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSTL->outputs_count.emplace_back(out.size());

  // Create Task
  KiselevTaskSTL kiselevTaskSTL(taskDataSTL);
  ASSERT_EQ(kiselevTaskSTL.validation(), true);
  ASSERT_EQ(kiselevTaskSTL.pre_processing(), true);
  ASSERT_EQ(kiselevTaskSTL.run(), true);
  ASSERT_EQ(kiselevTaskSTL.post_processing(), true);
  ASSERT_EQ(out, res);
}

TEST(kiselev_i_shell_simple_stl, check_incorrect_input) {
  const int count = 4;

  std::vector<std::vector<int>> in = {{2, 1}, {4, 3}};
  std::vector<int> out(count, 0);
  std::vector<int> res = {1, 2, 3, 4};

  std::shared_ptr<ppc::core::TaskData> taskDataSTL = std::make_shared<ppc::core::TaskData>();
  taskDataSTL->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSTL->inputs_count.emplace_back(in.size());
  taskDataSTL->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSTL->outputs_count.emplace_back(out.size());

  KiselevTaskSTL kiselevTaskSTL(taskDataSTL);
  ASSERT_EQ(kiselevTaskSTL.validation(), false);
}
