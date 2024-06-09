// Copyright 2024 Kiselev Igor
#include "stl/kiselev_i_shell_simple/include/shell_simple.hpp"

#include <execution>
#include <memory>

using namespace std::chrono_literals;

using namespace std;
using namespace Kiselev_stl;

bool KiselevTaskSTL::pre_processing() {
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

bool KiselevTaskSTL::validation() {
  try {
    internal_order_test();
    return taskData->inputs_count[0] != 0 && taskData->inputs_count[0] == taskData->outputs_count[0];
  } catch (...) {
    return false;
  }
}

bool KiselevTaskSTL::run() {
  try {
    internal_order_test();
    int n = (int)arr.size();
    FindThreadVariables();
    if (ThreadNum == 0) return false;
    int *Index = new int[ThreadNum + 1];
    int *BlockSize = new int[ThreadNum];
    int *BlockIndices = new int[ThreadNum];
    for (int i = 0; i <= ThreadNum; i++) {
      Index[i] = (i * n) / ThreadNum;
    }
    for (int i = 0; i < ThreadNum; i++) {
      BlockSize[i] = Index[i + 1] - Index[i];
      BlockIndices[i] = i;
    }
    std::vector<int> helper = vector<int>(ThreadNum);
    for (int i = 0; i < ThreadNum; i++) {
      helper[i] = i;
    }
    std::for_each_n(helper.begin(), ThreadNum, [&](int i) { SeqSorter(Index[i], Index[i] + BlockSize[i], arr); });
    for (int i = 1; i < ThreadNum; i *= 2) {
      std::for_each_n(helper.begin(), ThreadNum / (2 * i), [&](int j) {
        int left = BlockIndices[j * 2 * i];
        int right = (j * 2 * i + i < ThreadNum) ? BlockIndices[j * 2 * i + i] : -1;
        if (right != -1) {
          MergeBlocks(Index[left], BlockSize[left], Index[right], BlockSize[right], arr);
          BlockIndices[j * 2 * i] = left;
          BlockSize[j * 2 * i] = BlockSize[left] + BlockSize[right];
        } else {
          BlockIndices[j * 2 * i] = left;
          BlockSize[j * 2 * i] = BlockSize[left];
        }
      });
    }
    delete[] Index;
    delete[] BlockSize;
    delete[] BlockIndices;
    return true;
  } catch (...) {
    return false;
  }
  return true;
}

bool KiselevTaskSTL::post_processing() {
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
void KiselevTaskSTL::MergeBlocks(int Index1, int BlockSize1, int Index2, int BlockSize2, std::vector<int> &array) {
  int *pTempArray = new int[BlockSize1 + BlockSize2];
  for (int i = BlockSize1 + BlockSize2 - 1; i >= 0; i--) {
    pTempArray[i] = 0;
  }
  int i1 = Index1;
  int i2 = Index2;
  int curr = 0;
  if (BlockSize1 == 0) {
    for (int i = 0; i < BlockSize2; i++) {
      pTempArray[curr++] = array[i2++];
    }
  } else if (BlockSize2 == 0) {
    for (int i = 0; i < BlockSize1; i++) {
      pTempArray[curr++] = array[i1++];
    }
  } else {
    while (i1 < Index1 + BlockSize1 && i2 < Index2 + BlockSize2) {
      if (array[i1] < array[i2])
        pTempArray[curr++] = array[i1++];
      else
        pTempArray[curr++] = array[i2++];
    }
    while (i1 < Index1 + BlockSize1) pTempArray[curr++] = array[i1++];
    while (i2 < Index2 + BlockSize2) pTempArray[curr++] = array[i2++];
  }
  for (int i = 0; i < BlockSize1 + BlockSize2; i++) array[Index1 + i] = pTempArray[i];
  delete[] pTempArray;
}

bool KiselevTaskSTL::IsSorted() {
  int n = arr.size();
  for (int i = 1; i < n; i++) {
    if (arr[i - 1] > arr[i]) return false;
  }
  return true;
}

int KiselevTaskSTL::exp(int arg, int exp) {
  int res = arg;
  for (int i = 1; i < exp; i++) {
    res *= arg;
  }
  return res;
}

void KiselevTaskSTL::FindThreadVariables() {
  ThreadNum = 8;
  int helper = 1;
  DimSize = 1;
  while (ThreadNum >= helper) {
    DimSize++;
    helper = helper << 1;
  }
}

void KiselevTaskSTL::SeqSorter(int start, int end, std::vector<int> &array) {
  int n = end - start;
  for (int i = 1; i < n; i++) {
    int j = i - 1;
    while (j >= 0 && array[j + start] > array[j + 1 + start]) {
      array[j + start] += array[j + 1 + start];
      array[j + 1 + start] = array[j + start] - array[j + 1 + start];
      array[j + start] = array[j + start] - array[j + 1 + start];
      j--;
    }
  }
}
