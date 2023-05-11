#include "imalig/BarcodeDetector.hpp"
#include "imalig/imalig.hpp"

#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <argumentum/argparse.h>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

using namespace argumentum;

cv::Vec3d getCameraPosion(cv::Vec3d rvec, cv::Vec3d tvec)
{
	cv::Matx33d R;
	cv::Rodrigues(rvec, R);
	// cv::Matx31d T(tvec[0], tvec[1], tvec[2]);
	cv::Vec3d C = -R.t() * tvec;
	return C;
}
void saveResult(const std::string &originalFilepath, const std::string &directory, const cv::Mat &outImage)
{
	std::filesystem::path p(originalFilepath);
	std::filesystem::path filename = p.filename();

	p.remove_filename();
	p.append("result");

	std::filesystem::create_directories(p);

	p /= filename;

	spdlog::info("Save result in {}", p.string());

	cv::imwrite(p.string(), outImage);
}

int main(int argc, char *argv[])
{
	int marker_id;
	std::string image_filename;
	float resizeRatio;
	std::string outputDirectory;

	/* Configure argumentum */
	auto parser = argument_parser{};
	auto params = parser.params();
	parser.config().program(argv[0]).description("imalig example");
	params.add_parameter(marker_id, "id").help("Marker ID");
	params.add_parameter(image_filename, "image").help("Image filepath");
	params.add_parameter(resizeRatio, "--resize", "-r").nargs(1).help("Resize ratio").default_value(0.5);
	params.add_parameter(outputDirectory, "--output-subdirectory", "-o")
		.nargs(1)
		.help("Output subdirectory")
		.default_value("result");

	if (!parser.parse_args(argc, argv, 1)) {
		return 1;
	}

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

	/* Check if markers is detected */
	if (markersId.empty()) {
		spdlog::error("No markers detected");
		saveResult(image_filename, outputDirectory, image);
		return 1;
	}

	spdlog::info("Detected markers ids: [{}]", markersId[0]);

	/* Show result */
	// cv::imshow("barcode", barcode);
	// cv::imshow("image", image);
	cv::Mat outImage = image.clone();
	cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId);

	/* Run imalig */
	cv::Mat barcode = barcodeDetector.drawMarker(markersId[0], markersCorners[0]);
	auto corners = imalig::Imalig().process(barcode, imageGray, markersId[0], markersCorners[0]);
	std::cout << "corners = " << corners << std::endl;

	/* Draw aligned marker */
	markersCorners[0] = corners;
	cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId, {0, 0, 255});

	/* Pose estimation */
	double fx = 2573.10;
	double fy = fx;
	double cx = image.cols / 2.;
	double cy = image.rows / 2.;

	cv::Matx33d cameraMatrix{fx, 0, cx, 0, fy, cy, 0, 0, 1};
	cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
	std::vector<cv::Vec3d> rvec, tvec;
	cv::aruco::estimatePoseSingleMarkers(markersCorners, 0.25956, cameraMatrix, distCoeffs, rvec, tvec);
	cv::aruco::drawAxis(outImage, cameraMatrix, distCoeffs, rvec, tvec, 0.5);

	/* Print rvec and tvec */
	// spdlog::info("rvec = {}", rvec[0]);
	// spdlog::info("tvec = {}", tvec[0]);
	std::cout << "rvec = " << rvec[0] << std::endl;
	std::cout << "rvec = " << tvec[0] << std::endl;

	/* Print camera position */
	cv::Vec3d cameraPosition = getCameraPosion(rvec[0], tvec[0]);
	std::cout << "cameraPosition = " << cameraPosition << std::endl;

	/* Print camera position on image */
	double x, y, z;
	x = cameraPosition[0];
	y = cameraPosition[1];
	z = cameraPosition[2];
	cv::putText(outImage, fmt::format("cameraPosition = x:{:.2} y:{:.2} z:{:.2}", x, y, z),
				cv::Point(10, outImage.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 200, 0), 3);

	/* Save result in diffirent directory */
	saveResult(image_filename, outputDirectory, outImage);

	/* Show result */
	if (resizeRatio > 0) {
		cv::resize(outImage, outImage, cv::Size(), resizeRatio, resizeRatio);
		imshow("result", outImage);
		cv::waitKey(0);
	}

	return 0;
}
