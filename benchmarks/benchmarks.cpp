#include "imalig/BarcodeDetector.hpp"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>

TEST_CASE("BarcodeDetector")
{
	const cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);

	BENCHMARK_ADVANCED("BarcodeDetector::detect")(Catch::Benchmark::Chronometer meter)
	{
		imalig::BarcodeDetector barcodeDetector;
		meter.measure([&]() {
			auto [markersId, markersCorners] = barcodeDetector.detect(image);
			return markersCorners;
		});
	};
}

TEST_CASE("Imalig")
{
	const cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);

	imalig::BarcodeDetector barcodeDetector;
	auto [markersId, markersCorners] = barcodeDetector.detect(image);

	const cv::Mat barcode = barcodeDetector.drawMarker(markersId[0], markersCorners[0]);

	// imalig::Imalig imalig;
	BENCHMARK("Imalig::process()")
	{
		auto corners = imalig::Imalig().process(barcode, image, markersId[0], markersCorners[0]);
		return corners;
	};
}
