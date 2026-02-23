#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "vdovin_a_gauss_block_seq/common/include/common.hpp"

namespace vdovin_a_gauss_block_seq {

class VdovinAGaussBlockSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VdovinAGaussBlockSEQ(const InType &in);

  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> input_image_;
  std::vector<uint8_t> output_image_;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vdovin_a_gauss_block_seq
