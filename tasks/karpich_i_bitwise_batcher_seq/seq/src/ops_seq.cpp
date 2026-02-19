#include "karpich_i_bitwise_batcher_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace karpich_i_bitwise_batcher_seq {

namespace {

void RadixSortPositive(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n <= 1) {
    return;
  }

  int max_val = *std::max_element(arr.begin(), arr.end());
  if (max_val == 0) {
    return;
  }

  std::vector<int> buffer(n);

  for (int shift = 0; shift < 32 && (max_val >> shift) > 0; shift += 8) {
    std::vector<int> count(256, 0);
    for (int i = 0; i < n; i++) {
      count[(arr[i] >> shift) & 0xFF]++;
    }
    for (int i = 1; i < 256; i++) {
      count[i] += count[i - 1];
    }
    for (int i = n - 1; i >= 0; i--) {
      buffer[--count[(arr[i] >> shift) & 0xFF]] = arr[i];
    }
    arr = buffer;
  }
}

void RadixSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n <= 1) {
    return;
  }

  std::vector<int> negative;
  std::vector<int> positive;
  for (int i = 0; i < n; i++) {
    if (arr[i] < 0) {
      negative.push_back(-arr[i]);
    } else {
      positive.push_back(arr[i]);
    }
  }

  RadixSortPositive(positive);
  RadixSortPositive(negative);

  int idx = 0;
  for (int i = static_cast<int>(negative.size()) - 1; i >= 0; i--) {
    arr[idx++] = -negative[i];
  }
  for (int i = 0; i < static_cast<int>(positive.size()); i++) {
    arr[idx++] = positive[i];
  }
}

void BatcherMerge(std::vector<int> &arr, int lo, int hi, int r) {
  int step = r * 2;
  if (step < hi - lo) {
    BatcherMerge(arr, lo, hi, step);
    BatcherMerge(arr, lo + r, hi, step);
    for (int i = lo + r; i + r <= hi; i += step) {
      if (arr[i] > arr[i + r]) {
        std::swap(arr[i], arr[i + r]);
      }
    }
  } else if (lo + r <= hi) {
    if (arr[lo] > arr[lo + r]) {
      std::swap(arr[lo], arr[lo + r]);
    }
  }
}

}  // namespace

KarpichIBitwiseBatcherSEQ::KarpichIBitwiseBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KarpichIBitwiseBatcherSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool KarpichIBitwiseBatcherSEQ::PreProcessingImpl() {
  int n = GetInput();
  data_.resize(n);
  std::mt19937 gen(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (int i = 0; i < n; i++) {
    data_[i] = dist(gen);
  }
  return true;
}

bool KarpichIBitwiseBatcherSEQ::RunImpl() {
  int n = static_cast<int>(data_.size());
  if (n <= 1) {
    return true;
  }

  int padded = 1;
  while (padded < n) {
    padded *= 2;
  }

  int max_elem = *std::max_element(data_.begin(), data_.end());
  data_.resize(padded, max_elem);

  int half = padded / 2;
  std::vector<int> left(data_.begin(), data_.begin() + half);
  std::vector<int> right(data_.begin() + half, data_.end());

  RadixSort(left);
  RadixSort(right);

  std::copy(left.begin(), left.end(), data_.begin());
  std::copy(right.begin(), right.end(), data_.begin() + half);

  BatcherMerge(data_, 0, padded - 1, 1);

  data_.resize(n);
  return true;
}

bool KarpichIBitwiseBatcherSEQ::PostProcessingImpl() {
  for (int i = 1; i < static_cast<int>(data_.size()); i++) {
    if (data_[i] < data_[i - 1]) {
      return false;
    }
  }
  GetOutput() = GetInput();
  return true;
}

}  // namespace karpich_i_bitwise_batcher_seq
