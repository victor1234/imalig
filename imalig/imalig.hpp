#pragma once

#include <opencv2/core.hpp>

namespace imalig {
  std::vector<cv::Point2f> imalig(const cv::Mat barcode, const cv::Mat image, const int markerId, const std::vector<cv::Point2f> markerCorners);
}
