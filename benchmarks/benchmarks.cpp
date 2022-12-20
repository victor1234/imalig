#include "imalig/BarcodeDetector.hpp"
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>

TEST_CASE("Main")
{
	cv::Mat barcode = cv::imread("fixtures/barcode.png", cv::IMREAD_GRAYSCALE);
	cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);

	imalig::BarcodeDetector barcodeDetector;
	auto [markersId, markersCorners] = barcodeDetector.detect(image);

  imalig::Imalig imalig;
	BENCHMARK("Imalig::process()")
	{
		auto corners = imalig.process(barcode, image, markersId[0], markersCorners[0]);
	};
}
