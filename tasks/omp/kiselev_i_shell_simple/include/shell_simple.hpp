// Copyright 2024 Kiselev Igor
#ifndef SHELL_SIMPLE_HPP_INCLUDED
#define SHELL_SIMPLE_HPP_INCLUDED
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace Kiselev_omp {

class KiselevTaskOMP : public ppc::core::Task {
 public:
  explicit KiselevTaskOMP(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> arr;
  std::vector<int> input_;
  int ThreadNum, ThreadID, DimSize;
  bool IsSorted();
  int exp(int arg, int exp);
  void FindThreadVariables();
  int GrayCode(int RingID, int DimSize);
  int ReverseGrayCode(int CubeID, int DimSize);
  void SetBlockPairs(int* BlockPairs, int Iter);
  int FindPair(int* BlockPairs, int ThreadID, int Iter);
  void MergeBlocks(std::vector<int> pData, int Index1, int BlockSize1, int Index2, int BlockSize2);
  void SeqSorter(int start, int end);
  void CompareSplitBlocks(int start1, int size1, int start2, int size2);
};
}  // namespace Kiselev_omp
#endif  // SHELL_SIMPLE_HPP_INCLUDED
