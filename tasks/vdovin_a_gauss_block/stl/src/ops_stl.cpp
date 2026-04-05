#include "vdovin_a_gauss_block/stl/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <thread>
#include <vector>

#include "util/include/util.hpp"
#include "vdovin_a_gauss_block/common/include/common.hpp"

namespace vdovin_a_gauss_block {

namespace {
constexpr int kChannels = 3;
constexpr int kKernelSize = 3;
constexpr int kKernelSum = 16;
constexpr std::array<std::array<int, kKernelSize>, kKernelSize> kKernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
}  // namespace

VdovinAGaussBlockSTL::VdovinAGaussBlockSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VdovinAGaussBlockSTL::ValidationImpl() {
  return GetInput() >= 3;
}

bool VdovinAGaussBlockSTL::PreProcessingImpl() {
  width_ = GetInput();
  height_ = GetInput();
  if (width_ < 3 || height_ < 3) {
    input_image_.clear();
    output_image_.clear();
    return false;
  }
  int total = width_ * height_ * kChannels;
  input_image_.assign(total, 100);
  output_image_.assign(total, 0);
  return true;
}

void VdovinAGaussBlockSTL::ApplyGaussianToPixel(int py, int px) {
  for (int ch = 0; ch < kChannels; ch++) {
    int sum = 0;
    for (int ky = -1; ky <= 1; ky++) {
      for (int kx = -1; kx <= 1; kx++) {
        int ny = std::clamp(py + ky, 0, height_ - 1);
        int nx = std::clamp(px + kx, 0, width_ - 1);
        sum += input_image_[(((ny * width_) + nx) * kChannels) + ch] * kKernel.at(ky + 1).at(kx + 1);
      }
    }
    output_image_[(((py * width_) + px) * kChannels) + ch] = static_cast<uint8_t>(std::clamp(sum / kKernelSum, 0, 255));
  }
}

bool VdovinAGaussBlockSTL::RunImpl() {
  if (input_image_.empty() || output_image_.empty()) {
    return false;
  }

  int num_threads = ppc::util::GetNumThreads();
  int block_height = std::max(1, height_ / 4);
  int block_width = std::max(1, width_ / 4);

  std::vector<std::pair<int, int>> blocks;
  for (int by = 0; by < height_; by += block_height) {
    for (int bx = 0; bx < width_; bx += block_width) {
      blocks.emplace_back(by, bx);
    }
  }

  int total_blocks = static_cast<int>(blocks.size());
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto worker = [&](int start, int end) {
    for (int b = start; b < end; b++) {
      int by = blocks[b].first;
      int bx = blocks[b].second;
      int y_end = std::min(by + block_height, height_);
      int x_end = std::min(bx + block_width, width_);
      for (int py = by; py < y_end; py++) {
        for (int px = bx; px < x_end; px++) {
          ApplyGaussianToPixel(py, px);
        }
      }
    }
  };

  int blocks_per_thread = total_blocks / num_threads;
  int remainder = total_blocks % num_threads;
  int offset = 0;

  for (int t = 0; t < num_threads; t++) {
    int count = blocks_per_thread + (t < remainder ? 1 : 0);
    threads.emplace_back(worker, offset, offset + count);
    offset += count;
  }

  for (auto &th : threads) {
    th.join();
  }

  return true;
}

bool VdovinAGaussBlockSTL::PostProcessingImpl() {
  if (output_image_.empty()) {
    return false;
  }
  auto total = static_cast<int64_t>(output_image_.size());
  if (total == 0) {
    return false;
  }
  int64_t sum = 0;
  for (int64_t idx = 0; idx < total; idx++) {
    sum += output_image_[idx];
  }
  GetOutput() = static_cast<int>(sum / total);
  return true;
}

}  // namespace vdovin_a_gauss_block
