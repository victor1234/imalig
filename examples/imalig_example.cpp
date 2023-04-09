#include "imalig/BarcodeDetector.hpp"
#include "imalig/imalig.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <argumentum/argparse.h>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

using namespace argumentum;

int main(int argc, char *argv[])
{
	int marker_id;
	std::string image_filename;

	/* Configure argumentum */
	auto parser = argument_parser{};
	auto params = parser.params();
	parser.config().program(argv[0]).description("imalig example");
	params.add_parameter(marker_id, "id").help("Marker ID");
	params.add_parameter(image_filename, "image").help("Image filepath");

	if (!parser.parse_args(argc, argv, 1))
		return 1;

	/* Load image */
	const cv::Mat image = cv::imread(image_filename, cv::IMREAD_COLOR);
	if (image.empty()) {
		spdlog::error("Could not read the image: {}", image_filename);
		return 1;
	} else {
		spdlog::info("Image resolution: {}x{}", image.cols, image.rows);
	}

	/* Create barcode detector */
	imalig::BarcodeDetector barcodeDetector;

	/* Detect markers in grayscale */
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
	auto [markersId, markersCorners] = barcodeDetector.detect(imageGray);

	spdlog::info("Detected markers ids: [{}]", markersId[0]);

	/* Show result */
	// cv::imshow("barcode", barcode);
	// cv::imshow("image", image);
	cv::Mat outImage = image.clone();
	cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId);
	imshow("result", outImage);

	/* Run imalig */
	cv::Mat barcode = barcodeDetector.drawMarker(markersId[0], markersCorners[0]);
	auto corners = imalig::Imalig().process(barcode, imageGray, markersId[0], markersCorners[0]);
	std::cout << "corners = " << corners << std::endl;

	/* Draw aligned marker */
	markersCorners[0] = corners;
	cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId, {0, 0, 255});

	/* Pose estimation */
	cv::Matx33d cameraMatrix{image.cols * 0.8, 0, image.cols / 2., 0, image.cols * 0.8, image.rows / 2., 0, 0, 1};
	cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
	std::vector<cv::Vec3d> rvec, tvec;
	cv::aruco::estimatePoseSingleMarkers(markersCorners, 0.1, cameraMatrix, distCoeffs, rvec, tvec);
	cv::aruco::drawAxis(outImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);

	imshow("result", outImage);
	cv::waitKey(0);

	return 0;
}
