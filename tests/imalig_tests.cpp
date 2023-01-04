#include "imalig/BarcodeDetector.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>
#include <opencv2/imgproc.hpp>

TEST_CASE("Main")
{
	cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);

	imalig::BarcodeDetector barcodeDetector;
	auto [markersId, markersCorners] = barcodeDetector.detect(image);

	REQUIRE_FALSE(markersId.empty());

  float size = std::max(cv::norm(markersCorners[0][0] - markersCorners[0][2]),
                        cv::norm(markersCorners[0][1] - markersCorners[0][3]));
  std::cout << "size: " << size << std::endl;
	cv::Mat barcode = barcodeDetector.drawMarker(markersId[0], size);

  std::vector<cv::Mat> cornersList;
  constexpr float m = 20;
	for (size_t i = 0; i < 5; ++i) {
		auto markerCorners = markersCorners[0];
		/* Add random noise */
		for (auto &c : markerCorners) {
			float dx = GENERATE(take(1, random(-m, m)));
			float dy = GENERATE(take(1, random(-m, m)));
			c.x += dx;
			c.y += dy;
		}

		auto corners = imalig::Imalig().process(barcode, image, markersId[0], markerCorners);
		REQUIRE_FALSE(corners.empty());

		cornersList.push_back(cv::Mat(corners).clone());
	}

  /* Calculate mean */
  cv::Mat mean = cv::Mat::zeros(cornersList[0].size(), cornersList[0].type());
  for (const auto m: cornersList) {
    mean += m;
  }
  mean /= cornersList.size();

  /* Calculate std */
  cv::Mat std(cornersList[0].size(), cornersList[0].type(), cv::Scalar(0));
  for (const auto m: cornersList) {
    std += (m - mean).mul(m - mean);
  }

  CAPTURE(mean);
  CAPTURE(std);

  double maxValue;
  cv::minMaxLoc(std, nullptr, &maxValue);
  REQUIRE(false);
  REQUIRE(maxValue < 1e-5);
}
