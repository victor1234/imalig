#include "imalig/imalig.hpp"
#include <iostream>

#include <opencv2/core/types.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace imalig {

std::vector<cv::Point2f> imalig(const cv::Mat barcode, cv::Mat image, const int markerId,
								const std::vector<cv::Point2f> markerCorners)
{
	std::vector<cv::Point2i> markerCorners2i = {
		{0, 0}, {barcode.rows, 0}, {barcode.rows, barcode.cols}, {0, barcode.cols}};

	cv::Mat markerCorners0;
	cv::Mat(markerCorners2i).convertTo(markerCorners0, CV_32F);

	cv::Mat H = cv::getPerspectiveTransform(markerCorners0, markerCorners);
	std::cout << "H = " << H << std::endl;

	return {};
}

} // namespace imalig
