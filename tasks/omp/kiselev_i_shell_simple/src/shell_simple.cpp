// Copyright 2024 Kiselev Igor
#include "omp/kiselev_i_shell_simple/include/shell_simple.hpp"

#include <omp.h>

#include <memory>

using namespace std::chrono_literals;

using namespace std;
using namespace Kiselev_omp;

bool KiselevTaskOMP::pre_processing() {
  try {
    internal_order_test();
    size_t n = taskData->inputs_count[0];
    arr = ::std::vector<int>(n, 0);
    for (size_t i = 0; i < n; ++i) {
      int *elem = reinterpret_cast<int *>(taskData->inputs[0] + i * sizeof(int));
      arr[i] = *elem;
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool KiselevTaskOMP::validation() {
  try {
    internal_order_test();
    return taskData->inputs_count[0] != 0 && taskData->inputs_count[0] == taskData->outputs_count[0];
  } catch (...) {
    return false;
  }
}

bool KiselevTaskOMP::run() {
  try {
    internal_order_test();
    int n = (int)arr.size();
    FindThreadVariables();
    if (ThreadNum == 0) return false;
    int *Index = new int[(unsigned long)2 * ThreadNum];
    int *BlockSize = new int[(unsigned long)2 * ThreadNum];
    int *BlockPairs = new int[(unsigned long)4 * ThreadNum + 1];
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
      SeqSorter(Index[BlockID], Index[BlockID] + BlockSize[BlockID] - 1);
      BlockID = ReverseGrayCode(ThreadID, DimSize);
      SeqSorter(Index[BlockID], Index[BlockID] + BlockSize[BlockID] - 1);
    }
    for (int Iter = 0; (Iter < DimSize) && (!IsSorted()); Iter++) {
      SetBlockPairs(BlockPairs, Iter);
#pragma omp parallel
      {
        int MyPairNum = FindPair(BlockPairs, ThreadID, Iter);
        int FirstBlock = ReverseGrayCode(BlockPairs[2 * MyPairNum], DimSize);
        int SecondBlock = ReverseGrayCode(BlockPairs[2 * MyPairNum + 1], DimSize);
        MergeBlocks(arr, Index[FirstBlock], BlockSize[FirstBlock], Index[SecondBlock], BlockSize[SecondBlock]);
      }
    }
    int Iter = 1;
    while (!IsSorted() && Iter < 2 * DimSize) {
#pragma omp parallel
      {
        if (Iter % 2 == 0)
          MergeBlocks(arr, Index[2 * ThreadID], BlockSize[2 * ThreadID], Index[2 * ThreadID + 1],
                      BlockSize[2 * ThreadID + 1]);
        else if (ThreadID < ThreadNum - 1)
          MergeBlocks(arr, Index[2 * ThreadID + 1], BlockSize[2 * ThreadID + 1], Index[2 * ThreadID + 2],
                      BlockSize[2 * ThreadID + 2]);
      }
      Iter++;
    }
    delete[] Index;
    delete[] BlockSize;
    delete[] BlockPairs;
    return true;
  } catch (...) {
    return false;
  }
  return true;
}

bool KiselevTaskOMP::post_processing() {
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
void KiselevTaskOMP::MergeBlocks(::std::vector<int> pData, int Index1, int BlockSize1, int Index2, int BlockSize2) {
  int *pTempArray = new int[BlockSize1 + BlockSize2];
  int i1 = Index1, i2 = Index2, curr = 0;
  while ((i1 < Index1 + BlockSize1) || (i2 < Index2 + BlockSize2)) {
    if (((i1 < Index1 + BlockSize1) && (pData[i1] < pData[i2])) || (i2 >= Index2 + BlockSize2))
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

bool KiselevTaskOMP::IsSorted() {
  int n = arr.size();
  for (int i = 1; i < n; i++) {
    if (arr[i - 1] > arr[i]) return false;
  }
  return true;
}

int KiselevTaskOMP::exp(int arg, int exp) {
  int res = arg;
  for (int i = 1; i < exp; i++) {
    res *= arg;
  }
  return res;
}

void KiselevTaskOMP::FindThreadVariables() {
#pragma omp parallel
  {
    ThreadID = omp_get_thread_num();
#pragma omp single
    ThreadNum = omp_get_num_threads();
  }
  int helper = 1;
  DimSize = 1;
  while (ThreadNum >= helper) {
    DimSize++;
    helper = helper << 1;
  }
}

int KiselevTaskOMP::GrayCode(int RingID, int _DimSize) {
  if ((RingID == 0) && (_DimSize == 1)) return 0;
  if ((RingID == 1) && (_DimSize == 1)) return 1;
  int res;
  if (RingID < (1 << (_DimSize - 1)))
    res = GrayCode(RingID, _DimSize - 1);
  else
    res = (1 << (_DimSize - 1)) + GrayCode((1 << _DimSize) - 1 - RingID, _DimSize - 1);
  return res;
}

int KiselevTaskOMP::ReverseGrayCode(int CubeID, int _DimSize) {
  for (int i = 0; i < (1 << _DimSize); i++) {
    if (CubeID == GrayCode(i, _DimSize)) return i;
  }
  return 0;
}

void KiselevTaskOMP::SetBlockPairs(int *BlockPairs, int Iter) {
  int PairNum = 0, FirstValue, SecondValue;
  bool Exist;
  for (int i = 0; i < 2 * ThreadNum; i++) {
    FirstValue = GrayCode(i, DimSize);
    Exist = false;
    for (int j = 0; (j < PairNum) && (!Exist); j++)
      if (BlockPairs[2 * j + 1] == FirstValue) Exist = true;
    if (!Exist) {
      SecondValue = exp(FirstValue, (1 << (DimSize - Iter - 1)));
      BlockPairs[2 * PairNum] = FirstValue;
      BlockPairs[2 * PairNum + 1] = SecondValue;
      PairNum++;
    }
  }
}

int KiselevTaskOMP::FindPair(int *BlockPairs, int _ThreadID, int Iter) {
  int BlockID = 0, index = 0, result = 0;
  for (int i = 0; i < ThreadNum; i++) {
    BlockID = BlockPairs[2 * i];
    if (Iter == 0) index = BlockID % (1 << (DimSize - Iter - 1));
    if ((Iter > 0) && (Iter < DimSize - 1))
      index = ((BlockID >> (DimSize - Iter)) << (DimSize - Iter - 1)) | (BlockID % (1 << (DimSize - Iter - 1)));
    if (Iter == DimSize - 1) index = BlockID >> 1;
    if (index == _ThreadID) {
      result = i;
      break;
    }
  }
  return result;
}

void KiselevTaskOMP::SeqSorter(int start, int end) {
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
