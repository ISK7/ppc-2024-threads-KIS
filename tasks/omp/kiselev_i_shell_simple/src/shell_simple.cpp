// Copyright 2024 Kiselev Igor
#include "omp/kiselev_i_shell_simple/include/shell_simple.hpp"

#include <thread>
#include <omp.h>

using namespace std::chrono_literals;

bool Kiselev_omp::KiselevTaskOMP::pre_processing() {
  try {
    internal_order_test();
    size_t n = taskData->inputs_count[0];
    arr = std::vector<int>(n, 0);
    for (size_t i = 0; i < n; ++i) {
      int *elem = reinterpret_cast<int *>(taskData->inputs[0] + i * sizeof(int));
      arr[i] = *elem;
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool Kiselev_omp::KiselevTaskOMP::validation() {
  try {
    internal_order_test();
    return taskData->inputs_count[0] != 0 && taskData->inputs_count[0] == taskData->outputs_count[0];
  } catch (...) {
    return false;
  }
}

bool Kiselev_omp::KiselevTaskOMP::run() {
  try {
    internal_order_test();
    int n = arr.size();
    FindThreadVariables();
    int *Index = new int[2 * ThreadNum];
    int *BlockSize = new int[2 * ThreadNum];
    int *BlockPairs = new int[2 * ThreadNum];
    for (int i = 0; i < 2 * ThreadNum; i++) {
      Index[i] = int((i * n) / double(2 * ThreadNum));
      if (i < 2 * ThreadNum - 1)
        BlockSize[i] = int(((i + 1) * n) / double(2 * ThreadNum)) - Index[i];
      else
        BlockSize[i] = n - Index[i];
    }
#pragma omp parallel
    {
      int BlockID = ReverseGrayCode(ThreadNum + ThreadID, DimSize);
      SeqSorter(arr, Index[BlockID], Index[BlockID] + BlockSize[BlockID] - 1);
      BlockID = ReverseGrayCode(ThreadID, DimSize);
      SeqSorter(arr, Index[BlockID], Index[BlockID] + BlockSize[BlockID] - 1);
    }
    for (int Iter = 0; (Iter < DimSize) && (!IsSorted(arr)); Iter++) {
      SetBlockPairs(BlockPairs, Iter);
#pragma omp parallel
      {
        int MyPairNum = FindPair(BlockPairs, ThreadID, Iter);
        int FirstBlock = ReverseGrayCode(BlockPairs[2 * MyPairNum], DimSize);
        int SecondBlock = ReverseGrayCode(BlockPairs[2 * MyPairNum + 1], DimSize);
        CompareSplitBlocks(arr, Index[FirstBlock], BlockSize[FirstBlock], Index[SecondBlock], BlockSize[SecondBlock]);
      }
    }
    int Iter = 1;
    while (!IsSorted(arr)) {
#pragma omp parallel
      {
        if (Iter % 2 == 0)
          MergeBlocks(arr, Index[2 * ThreadID], BlockSize[2 * ThreadID], Index[2 * ThreadID + 1],
                      BlockSize[2 * ThreadID + 1]);
        else if (ThreadID < ThreadNum - 1)
          MergeBlocks(arr, Index[2 * ThreadID + 1], BlockSize[2 * ThreadID + 1], Index[2 * ThreadID + 2],
                      BlockSize[2 * ThreadID + 2]);
        Iter++;
      }
      delete[] Index;
      delete[] BlockSize;
      delete[] BlockPairs;
      return true;
    }
  } catch (...) {
    return false;
  }
}

bool Kiselev_omp::KiselevTaskOMP::post_processing() {
  try {
    internal_order_test();
    size_t n = arr.size();
    for (size_t i = 0; i < n; i++) {
      int *res = reinterpret_cast<int *>(taskData->outputs[0] + i * sizeof(int));
      *res = arr[i];
      (void)res;
    }
    return true;
  } catch (...) {
    return false;
  }
}
// Can do better
void Kiselev_omp::KiselevTaskOMP::MergeBlocks(std::vector<int> pData, int Index1, int BlockSize1, int Index2,
                                              int BlockSize2) {
  int *pTempArray = new int[BlockSize1 + BlockSize2];
  int i1 = Index1, i2 = Index2, curr = 0;
  while ((i1 < Index1 + BlockSize1) && (i2 < Index2 + BlockSize2)) {
    if (pData[i1] < pData[i2])
      pTempArray[curr++] = pData[i1++];
    else {
      pTempArray[curr++] = pData[i2++];
    }
    while (i1 < Index1 + BlockSize1) pTempArray[curr++] = pData[i1++];
    while (i2 < Index2 + BlockSize2) pTempArray[curr++] = pData[i2++];
    for (int i = 0; i < BlockSize1 + BlockSize2; i++) pData[Index1 + i] = pTempArray[i];
    delete[] pTempArray;
  }
}

bool Kiselev_omp::KiselevTaskOMP::IsSorted(std::vector<int> arr) {
  int n = arr.size();
  for (int i = 1; i < n; i++) {
    if (arr[i - 1] > arr[i]) return false;
  }
  return true;
}

void Kiselev_omp::KiselevTaskOMP::FindThreadVariables() {
#pragma omp parallel
  {
    ThreadID = omp_get_thread_num();
#pragma omp single
    ThreadNum = omp_get_num_threads();
  }
  DimSize = int(log10(double(ThreadNum)) / log10(2.0)) + 1;
}

int Kiselev_omp::KiselevTaskOMP::GrayCode(int RingID, int DimSize) {
  if ((RingID == 0) && (DimSize == 1)) return 0;
  if ((RingID == 1) && (DimSize == 1)) return 1;
  int res;
  if (RingID < (1 << (DimSize - 1)))
    res = GrayCode(RingID, DimSize - 1);
  else
    res = (1 << (DimSize - 1)) + GrayCode((1 << DimSize) - 1 - RingID, DimSize - 1);
  return res;
}

int Kiselev_omp::KiselevTaskOMP::ReverseGrayCode(int CubeID, int DimSize) {
  for (int i = 0; i < (1 << DimSize); i++) {
    if (CubeID == GrayCode(i, DimSize)) return i;
  }
}

void Kiselev_omp::KiselevTaskOMP::SetBlockPairs(int *BlockPairs, int Iter) {
  int PairNum = 0, FirstValue, SecondValue;
  bool Exist;
  for (int i = 0; i < 2 * ThreadNum; i++) {
    FirstValue = GrayCode(i, DimSize);
    Exist = false;
    for (int j = 0; (j < PairNum) && (!Exist); j++)
      if (BlockPairs[2 * j + 1] == FirstValue) Exist = true;
    if (!Exist) {
      SecondValue = FirstValue ^ (1 << (DimSize - Iter - 1));
      BlockPairs[2 * PairNum] = FirstValue;
      BlockPairs[2 * PairNum + 1] = SecondValue;
      PairNum++;
    }
  }
}

int Kiselev_omp::KiselevTaskOMP::FindPair(int *BlockPairs, int ThreadID, int Iter) {
  int BlockID = 0, index, result;
  for (int i = 0; i < ThreadNum; i++) {
    BlockID = BlockPairs[2 * i];
    if (Iter == 0) index = BlockID % (1 << DimSize - Iter - 1);
    if ((Iter > 0) && (Iter < DimSize - 1))
      index = ((BlockID >> (DimSize - Iter)) << (DimSize - Iter - 1)) | (BlockID % (1 << (DimSize - Iter - 1)));
    if (Iter == DimSize - 1) index = BlockID >> 1;
    if (index == ThreadID) {
      result = i;
      break;
    }
  }
  return result;
}

void Kiselev_omp::KiselevTaskOMP::SeqSorter(std::vector<int>, int start, int end) {
  int n = end - start;
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i += 1) {
      int temp = arr[i + start];
      int j;
      for (j = i; j >= gap && arr[start + j - gap] > temp; j -= gap) arr[start + j] = arr[start + j - gap];
      arr[start + j] = temp;
    }
  }
}

void Kiselev_omp::KiselevTaskOMP::CompareSplitBlocks(std::vector<int> &arr, int start1, int size1, int start2,
                                                     int size2) {
  std::vector<int> temp(size1 + size2);
  int i = start1, j = start2, k = 0;
  while (i < start1 + size1 && j < start2 + size2) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }
  while (i < start1 + size1) {
    temp[k++] = arr[i++];
  }
  while (j < start2 + size2) {
    temp[k++] = arr[j++];
  }
  for (int m = 0; m < size1 + size2; m++) {
    arr[start1 + m] = temp[m];
  }
}
