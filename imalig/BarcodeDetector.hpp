#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/aruco.hpp>
#include <tuple>
#include <vector>

namespace imalig {
class BarcodeDetector {
  public:
    BarcodeDetector();
    cv::Mat drawMarker(int id, int size);
    std::tuple<std::vector<int>, std::vector<std::vector<cv::Point2f>>> detect(const cv::Mat image);

  private:
    cv::Ptr<cv::aruco::Dictionary> dictionary;
};
} // namespace imalig
