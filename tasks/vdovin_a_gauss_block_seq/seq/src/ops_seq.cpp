#include "vdovin_a_gauss_block_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vdovin_a_gauss_block_seq {

VdovinAGaussBlockSEQ::VdovinAGaussBlockSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VdovinAGaussBlockSEQ::ValidationImpl() {
  return GetInput() >= 3;
}

bool VdovinAGaussBlockSEQ::PreProcessingImpl() {
  width_ = GetInput();
  height_ = GetInput();
  if (width_ < 3 || height_ < 3) {
    input_image_.clear();
    output_image_.clear();
    return false;
  }
  int total = width_ * height_ * 3;
  input_image_.assign(total, 100);
  output_image_.assign(total, 0);
  return true;
}

bool VdovinAGaussBlockSEQ::RunImpl() {
  if (input_image_.empty() || output_image_.empty()) {
    return false;
  }

  const int kChannels = 3;
  const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  const int kKernelSum = 16;

  int block_height = std::max(1, height_ / 4);
  int block_width = std::max(1, width_ / 4);

  for (int block_y = 0; block_y < height_; block_y += block_height) {
    for (int block_x = 0; block_x < width_; block_x += block_width) {
      int y_end = std::min(block_y + block_height, height_);
      int x_end = std::min(block_x + block_width, width_);

      for (int y = block_y; y < y_end; y++) {
        for (int x = block_x; x < x_end; x++) {
          for (int c = 0; c < kChannels; c++) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
              for (int kx = -1; kx <= 1; kx++) {
                int ny = std::clamp(y + ky, 0, height_ - 1);
                int nx = std::clamp(x + kx, 0, width_ - 1);
                sum += input_image_[(ny * width_ + nx) * kChannels + c] * kernel[ky + 1][kx + 1];
              }
            }
            output_image_[(y * width_ + x) * kChannels + c] =
                static_cast<uint8_t>(std::clamp(sum / kKernelSum, 0, 255));
          }
        }
      }
    }
  }

  return true;
}

bool VdovinAGaussBlockSEQ::PostProcessingImpl() {
  if (output_image_.empty()) {
    return false;
  }
  int total = static_cast<int>(output_image_.size());
  long long sum = 0;
  for (int i = 0; i < total; i++) {
    sum += output_image_[i];
  }
  GetOutput() = static_cast<int>(sum / total);
  return true;
}

}  // namespace vdovin_a_gauss_block_seq
