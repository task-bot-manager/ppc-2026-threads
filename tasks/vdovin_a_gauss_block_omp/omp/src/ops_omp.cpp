#include "vdovin_a_gauss_block_omp/omp/include/ops_omp.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "vdovin_a_gauss_block_omp/common/include/common.hpp"

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
  const std::array<std::array<int, 3>, 3> k_kernel = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
  for (int ch = 0; ch < 3; ch++) {
    int sum = 0;
    for (int ky = -1; ky <= 1; ky++) {
      for (int kx = -1; kx <= 1; kx++) {
        int ny = std::clamp(py + ky, 0, height_ - 1);
        int nx = std::clamp(px + kx, 0, width_ - 1);
        sum += k_kernel.at(ky + 1).at(kx + 1) * input_image_[((ny * width_) + nx) * 3 + ch];
      }
    }
    output_image_[((py * width_) + px) * 3 + ch] = static_cast<uint8_t>(sum / 16);
  }
}

bool VdovinAGaussBlockOMP::RunImpl() {
  const int k_block_size = 32;
  int num_blocks_y = (height_ + k_block_size - 1) / k_block_size;
  int num_blocks_x = (width_ + k_block_size - 1) / k_block_size;
  int total_blocks = num_blocks_y * num_blocks_x;

#pragma omp parallel for schedule(static) default(none) shared(total_blocks, num_blocks_x)
  for (int bi = 0; bi < total_blocks; bi++) {
    int by = (bi / num_blocks_x) * k_block_size;
    int bx = (bi % num_blocks_x) * k_block_size;
    int y_end = std::min(by + k_block_size, height_);
    int x_end = std::min(bx + k_block_size, width_);
    for (int py = by; py < y_end; py++) {
      for (int px = bx; px < x_end; px++) {
        ApplyGaussianToPixel(py, px);
      }
    }
  }
  return true;
}

bool VdovinAGaussBlockOMP::PostProcessingImpl() {
  auto size = static_cast<int64_t>(output_image_.size());
  if (size == 0) {
    GetOutput() = 0;
    return true;
  }
  int64_t sum = 0;
  for (int64_t idx = 0; idx < size; idx++) {
    sum += output_image_[idx];
  }
  GetOutput() = static_cast<int>(sum / size);
  return true;
}

}  // namespace vdovin_a_gauss_block_omp
