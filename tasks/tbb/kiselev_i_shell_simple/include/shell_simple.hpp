// Copyright 2024 Kiselev Igor
#ifndef SHELL_SIMPLE_HPP_INCLUDED
#define SHELL_SIMPLE_HPP_INCLUDED
#pragma once
#include <tbb/tbb.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace Kiselev_tbb {

class KiselevTaskTBB : public ppc::core::Task {
 public:
  explicit KiselevTaskTBB(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> arr;
  std::vector<int> input_;
  int ThreadNum, DimSize;
  bool IsSorted();
  static int exp(int arg, int exp);
  void FindThreadVariables();
  void MergeBlocks(int Index1, int BlockSize1, int Index2, int BlockSize2, std::vector<int> &array);
  void SeqSorter(int start, int end, std::vector<int> &array);
};
}  // namespace Kiselev_tbb
#endif  // SHELL_SIMPLE_HPP_INCLUDED
