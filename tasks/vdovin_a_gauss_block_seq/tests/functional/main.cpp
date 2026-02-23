#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vdovin_a_gauss_block_seq/common/include/common.hpp"
#include "vdovin_a_gauss_block_seq/seq/include/ops_seq.hpp"

namespace vdovin_a_gauss_block_seq {

class VdovinAGaussBlockFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int width = -1;
    int height = -1;
    int channels = -1;
    {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(std::string(PPC_ID_vdovin_a_gauss_block_seq), "pic.ppm");
      auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);
      if (data == nullptr) {
        throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
      }
      stbi_image_free(data);
    }
    ASSERT_GT(width, 0);
    ASSERT_GT(channels, 0);
    ASSERT_EQ(width, height);

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 100;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(VdovinAGaussBlockFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VdovinAGaussBlockSEQ, InType>(kTestParam, PPC_SETTINGS_vdovin_a_gauss_block_seq));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VdovinAGaussBlockFuncTests::PrintFuncTestName<VdovinAGaussBlockFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VdovinAGaussBlockFuncTests, kGtestValues, kPerfTestName);

TEST(VdovinAGaussBlockExtra, ConstantImageThree) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(3);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, ConstantImageTen) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(10);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, AllZerosImage) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(4);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  std::fill(task->input_image_.begin(), task->input_image_.end(), static_cast<uint8_t>(0));
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 0);
}

TEST(VdovinAGaussBlockExtra, AllMaxImage) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(4);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  std::fill(task->input_image_.begin(), task->input_image_.end(), static_cast<uint8_t>(255));
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 255);
}

TEST(VdovinAGaussBlockExtra, CenterBrightPixel) {
  int n = 3;
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(n);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  std::fill(task->input_image_.begin(), task->input_image_.end(), static_cast<uint8_t>(0));
  int center_idx = (1 * n + 1) * 3;
  task->input_image_[center_idx] = 240;
  task->input_image_[center_idx + 1] = 240;
  task->input_image_[center_idx + 2] = 240;
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 26);
}

TEST(VdovinAGaussBlockExtra, ValidationFailsForTwo) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(2);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(VdovinAGaussBlockExtra, ValidationFailsForZero) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(0);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(VdovinAGaussBlockExtra, ValidationFailsForNegative) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(-1);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(VdovinAGaussBlockExtra, EvenSizedImage) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(6);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, LargeImage) {
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(50);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, ProcessRealImage) {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::string abs_path = ppc::util::GetAbsoluteTaskPath(std::string(PPC_ID_vdovin_a_gauss_block_seq), "pic.ppm");
  auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);
  ASSERT_NE(data, nullptr);
  channels = STBI_rgb;
  ASSERT_GT(width, 0);
  ASSERT_GT(height, 0);

  int side = std::max({width, height, 3});
  std::vector<uint8_t> padded(side * side * channels, 0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        padded[(y * side + x) * channels + c] = data[(y * width + x) * channels + c];
      }
    }
  }
  stbi_image_free(data);

  auto task = std::make_shared<VdovinAGaussBlockSEQ>(side);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  task->input_image_ = padded;
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_GE(task->GetOutput(), 0);
  EXPECT_LE(task->GetOutput(), 255);
}

TEST(VdovinAGaussBlockExtra, OutputImageSize) {
  int n = 8;
  auto task = std::make_shared<VdovinAGaussBlockSEQ>(n);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  EXPECT_EQ(static_cast<int>(task->output_image_.size()), n * n * 3);
}

}  // namespace

}  // namespace vdovin_a_gauss_block_seq
