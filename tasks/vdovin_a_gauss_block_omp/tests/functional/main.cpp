#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vdovin_a_gauss_block_omp/common/include/common.hpp"
#include "vdovin_a_gauss_block_omp/omp/include/ops_omp.hpp"

namespace vdovin_a_gauss_block_omp {

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
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(std::string(PPC_ID_vdovin_a_gauss_block_omp), "pic.ppm");
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

bool RunFullPipeline(VdovinAGaussBlockOMP &task) {
  return task.Validation() && task.PreProcessing() && task.Run() && task.PostProcessing();
}

std::pair<std::vector<uint8_t>, int> LoadAndPadPicPpm() {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::string abs_path = ppc::util::GetAbsoluteTaskPath(std::string(PPC_ID_vdovin_a_gauss_block_omp), "pic.ppm");
  unsigned char *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);
  if (data == nullptr) {
    return {{}, 0};
  }
  channels = STBI_rgb;
  int side = std::max({width, height, 3});
  std::vector<uint8_t> padded(
      static_cast<std::size_t>(side) * static_cast<std::size_t>(side) * static_cast<std::size_t>(channels), 0);
  for (int py = 0; py < height; py++) {
    for (int px = 0; px < width; px++) {
      for (int ch = 0; ch < channels; ch++) {
        padded[(((py * side) + px) * channels) + ch] = data[(((py * width) + px) * channels) + ch];
      }
    }
  }
  stbi_image_free(data);
  return {padded, side};
}

bool ProcessRealImageSucceedsAndOutputInRange() {
  auto [padded, side] = LoadAndPadPicPpm();
  if (padded.empty() || side < 3) {
    return false;
  }
  auto task = std::make_shared<VdovinAGaussBlockOMP>(side);
  if (!task->Validation() || !task->PreProcessing()) {
    return false;
  }
  task->InputImage() = padded;
  if (!task->Run() || !task->PostProcessing()) {
    return false;
  }
  int out = task->GetOutput();
  return out >= 0 && out <= 255;
}

TEST_P(VdovinAGaussBlockFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VdovinAGaussBlockOMP, InType>(kTestParam, PPC_SETTINGS_vdovin_a_gauss_block_omp));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VdovinAGaussBlockFuncTests::PrintFuncTestName<VdovinAGaussBlockFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VdovinAGaussBlockFuncTests, kGtestValues, kPerfTestName);

TEST(VdovinAGaussBlockExtra, ConstantImageThree) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(3);
  ASSERT_TRUE(RunFullPipeline(*task));
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, ConstantImageTen) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(10);
  ASSERT_TRUE(RunFullPipeline(*task));
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, AllZerosImage) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(4);
  ASSERT_TRUE(task->Validation() && task->PreProcessing());
  std::fill(task->InputImage().begin(), task->InputImage().end(), static_cast<uint8_t>(0));
  ASSERT_TRUE(task->Run() && task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 0);
}

TEST(VdovinAGaussBlockExtra, AllMaxImage) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(4);
  ASSERT_TRUE(task->Validation() && task->PreProcessing());
  std::fill(task->InputImage().begin(), task->InputImage().end(), static_cast<uint8_t>(255));
  ASSERT_TRUE(task->Run() && task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 255);
}

TEST(VdovinAGaussBlockExtra, CenterBrightPixel) {
  int side = 3;
  auto task = std::make_shared<VdovinAGaussBlockOMP>(side);
  ASSERT_TRUE(task->Validation() && task->PreProcessing());
  std::fill(task->InputImage().begin(), task->InputImage().end(), static_cast<uint8_t>(0));
  int center_idx = (1 * side + 1) * 3;
  task->InputImage()[center_idx] = 240;
  task->InputImage()[center_idx + 1] = 240;
  task->InputImage()[center_idx + 2] = 240;
  ASSERT_TRUE(task->Run() && task->PostProcessing());
  EXPECT_EQ(task->GetOutput(), 26);
}

TEST(VdovinAGaussBlockExtra, ValidationFailsForTwo) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(2);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(VdovinAGaussBlockExtra, ValidationFailsForZero) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(0);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(VdovinAGaussBlockExtra, ValidationFailsForNegative) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(-1);
  EXPECT_FALSE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();
}

TEST(VdovinAGaussBlockExtra, EvenSizedImage) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(6);
  ASSERT_TRUE(RunFullPipeline(*task));
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, LargeImage) {
  auto task = std::make_shared<VdovinAGaussBlockOMP>(50);
  ASSERT_TRUE(RunFullPipeline(*task));
  EXPECT_EQ(task->GetOutput(), 100);
}

TEST(VdovinAGaussBlockExtra, ProcessRealImage) {
  ASSERT_TRUE(ProcessRealImageSucceedsAndOutputInRange());
}

TEST(VdovinAGaussBlockExtra, OutputImageSize) {
  int side = 8;
  auto task = std::make_shared<VdovinAGaussBlockOMP>(side);
  ASSERT_TRUE(RunFullPipeline(*task));
  EXPECT_EQ(static_cast<int>(task->OutputImage().size()), side * side * 3);
}

}  // namespace

}  // namespace vdovin_a_gauss_block_omp
