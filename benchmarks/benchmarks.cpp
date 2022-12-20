#include "imalig/BarcodeDetector.hpp"
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>

TEST_CASE("BarcodeDetector")
{
	cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);

	BENCHMARK("BarcodeDetector::detect")
	{
		imalig::BarcodeDetector barcodeDetector;
		auto [markersId, markersCorners] = barcodeDetector.detect(image);
		return markersCorners;
	};
}

TEST_CASE("Imalig")
{
	cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);

	imalig::BarcodeDetector barcodeDetector;
	auto [markersId, markersCorners] = barcodeDetector.detect(image);

	cv::Mat barcode = barcodeDetector.drawMarker(markersId[0], 200);

	imalig::Imalig imalig;
	BENCHMARK("Imalig::process()")
	{
		auto corners = imalig.process(barcode, image, markersId[0], markersCorners[0]);
		return corners;
	};
}
