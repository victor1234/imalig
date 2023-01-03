#include "imalig/BarcodeDetector.hpp"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>

TEST_CASE("Main")
{
	cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);

	imalig::BarcodeDetector barcodeDetector;
	auto [markersId, markersCorners] = barcodeDetector.detect(image);

  REQUIRE_FALSE(markersId.empty());

	cv::Mat barcode = barcodeDetector.drawMarker(markersId[0], 200);
	auto corners = imalig::Imalig().process(barcode, image, markersId[0], markersCorners[0]);

	REQUIRE_FALSE(corners.empty());
}
