#include "imalig/imalig.hpp"
#include <iostream>

#include <opencv2/core/types.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <Eigen/Dense>


namespace imalig {

struct CostFunctor {
	CostFunctor(const cv::Mat barcode, const cv::Mat image)
		: barcode(barcode), barcodePoints(3, barcode.rows * barcode.cols), image(image)
	{
		barcode.convertTo(barcode, CV_64FC1);
		image.convertTo(image, CV_64FC1);

		/* Create barcode pixel coordinate matrix */
		int j = 0;
		for (int y = 0; y < barcode.rows; y++) {
			for (int x = 0; x < barcode.cols; x++) {
				barcodePoints(0, j) = x;
				barcodePoints(1, j) = y;
				barcodePoints(2, j) = 1;
				j++;
			}
		}

		/* Create image interpolator */
		ceres::Grid2D<double, 1> imageGrid(reinterpret_cast<double*>(barcode.data), 0, barcode.rows, 0, barcode.cols);
		ceres::BiCubicInterpolator interpolator(imageGrid);
	}

	template <typename T>
	bool operator()(const T* const x, T* residual) const {
		residual[0] = T(10.0) - x[0];
		return true;
	}

private:
	cv::Mat barcode;
	cv::Mat image;
	Eigen::MatrixXd barcodePoints;
};

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
