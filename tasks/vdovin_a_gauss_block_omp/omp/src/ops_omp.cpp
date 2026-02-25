#include "vdovin_a_gauss_block_omp/omp/include/ops_omp.hpp"

#include <algorithm>
#include <cstdint>

namespace vdovin_a_gauss_block_omp {

VdovinAGaussBlockOMP::VdovinAGaussBlockOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VdovinAGaussBlockOMP::ValidationImpl() {
  return GetInput() >= 3 && GetOutput() == 0;
}

bool VdovinAGaussBlockOMP::PreProcessingImpl() {
  int side = GetInput();
  width_ = side;
  height_ = side;
  auto size = static_cast<std::size_t>(side) * side * 3;
  input_image_.assign(size, 100);
  output_image_.resize(size);
  return true;
}

void VdovinAGaussBlockOMP::ApplyGaussianToPixel(int py, int px) {
  const int kKernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  for (int c = 0; c < 3; c++) {
    int sum = 0;
    for (int ky = -1; ky <= 1; ky++) {
      for (int kx = -1; kx <= 1; kx++) {
        int ny = std::clamp(py + ky, 0, height_ - 1);
        int nx = std::clamp(px + kx, 0, width_ - 1);
        sum += kKernel[ky + 1][kx + 1] * input_image_[(ny * width_ + nx) * 3 + c];
      }
    }
    output_image_[(py * width_ + px) * 3 + c] = static_cast<uint8_t>(sum / 16);
  }
}

bool VdovinAGaussBlockOMP::RunImpl() {
  const int kBlockSize = 32;
  int num_blocks_y = (height_ + kBlockSize - 1) / kBlockSize;
  int num_blocks_x = (width_ + kBlockSize - 1) / kBlockSize;
  int total_blocks = num_blocks_y * num_blocks_x;

#pragma omp parallel for schedule(static)
  for (int b = 0; b < total_blocks; b++) {
    int by = (b / num_blocks_x) * kBlockSize;
    int bx = (b % num_blocks_x) * kBlockSize;
    int y_end = std::min(by + kBlockSize, height_);
    int x_end = std::min(bx + kBlockSize, width_);
    for (int py = by; py < y_end; py++) {
      for (int px = bx; px < x_end; px++) {
        ApplyGaussianToPixel(py, px);
      }
    }
  }
  return true;
}

bool VdovinAGaussBlockOMP::PostProcessingImpl() {
  auto size = static_cast<long long>(output_image_.size());
  if (size == 0) {
    GetOutput() = 0;
    return true;
  }
  long long sum = 0;
  for (long long i = 0; i < size; i++) {
    sum += output_image_[i];
  }
  GetOutput() = static_cast<int>(sum / size);
  return true;
}

}  // namespace vdovin_a_gauss_block_omp
