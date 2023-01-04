#include "imalig/BarcodeDetector.hpp"
#include "imalig/imalig.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <marker_id> <image>" << std::endl;
		return 1;
	}

	const int markerId = std::stoi(argv[1]);
	const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);

	/* Create barcode detector */
	imalig::BarcodeDetector barcodeDetector;

	/* Detect markers in grayscale */
	cv::Mat imageGray;
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
	auto [markersId, markersCorners] = barcodeDetector.detect(imageGray);

	/* Show result */
	// cv::imshow("barcode", barcode);
	// cv::imshow("image", image);
	cv::Mat outImage = image.clone();
	cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId);
	imshow("result", outImage);

	/* Run imalig */
	cv::Mat barcode = barcodeDetector.drawMarker(markerId, markersCorners[0]);
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
