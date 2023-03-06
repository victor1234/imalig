#include "imalig/BarcodeDetector.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>
#include <opencv2/imgproc.hpp>

TEST_CASE("Main")
{
	/* Load image */
	cv::Mat image = cv::imread("fixtures/image.jpg", cv::IMREAD_GRAYSCALE);
	constexpr float k = 0.25;
	cv::resize(image, image, {}, k, k);

	/* Detect marker */
	imalig::BarcodeDetector barcodeDetector;
	auto [markersId, markersCorners] = barcodeDetector.detect(image);

	REQUIRE_FALSE(markersId.empty());
	REQUIRE(markersId.size() == 1);

	/* Create synth marker image */
	cv::Mat barcode = barcodeDetector.drawMarker(markersId[0], markersCorners[0]);

	std::vector<cv::Mat> cornersList;
	constexpr float d = 5;
	for (size_t i = 0; i < 5; ++i) {
		auto &markerCorners = markersCorners[0];
		/* Add random noise */
		for (auto &c : markerCorners) {
			float dx = GENERATE(take(1, random(-d, d)));
			float dy = GENERATE(take(1, random(-d, d)));
			c.x += dx;
			c.y += dy;
		}

		const auto corners = imalig::Imalig().process(barcode, image, markersId[0], markerCorners);
		REQUIRE_FALSE(corners.empty());

		cornersList.push_back(cv::Mat(corners).clone());
	}

	/* Calculate mean */
	cv::Mat mean = cv::Mat::zeros(cornersList[0].size(), cornersList[0].type());
	for (const auto m : cornersList) {
		mean += m;
	}
	mean /= cornersList.size();

	/* Calculate std */
	cv::Mat std(cornersList[0].size(), cornersList[0].type(), cv::Scalar(0));
	for (const auto m : cornersList) {
		std += (m - mean).mul(m - mean);
	}

	CAPTURE(mean);
	CAPTURE(std);

	double maxValue;
	cv::minMaxLoc(std, nullptr, &maxValue);

	REQUIRE(maxValue < 1e-4);
	REQUIRE(false);
}
