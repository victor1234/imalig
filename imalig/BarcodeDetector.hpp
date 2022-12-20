#pragma once

#include <opencv2/core/core.hpp>
#include <tuple>
#include <vector>

namespace imalig {
class BarcodeDetector {
  public:
	BarcodeDetector() = default;

	std::tuple<std::vector<int>, std::vector<std::vector<cv::Point2f>>> detect(const cv::Mat image);
};
} // namespace imalig
