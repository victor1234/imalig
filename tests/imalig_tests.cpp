#include "imalig/BarcodeDetector.hpp"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>

TEST_CASE("Main")
{
	cv::Mat barcode = cv::imread("fixtures/barcode.png");
	cv::Mat image = cv::imread("fixtures/image.jpg");

  imalig::BarcodeDetector barcodeDetector;
  auto [markersId, markersCorners] = barcodeDetector.detect(image);

	auto corners = imalig::Imalig().process(barcode, image, markersId[0], markersCorners[0]);

	REQUIRE_FALSE(corners.empty());
}
