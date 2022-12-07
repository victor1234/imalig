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
	using Grid2D = ceres::Grid2D<double, 1>;
	using Interpolator = ceres::BiCubicInterpolator<Grid2D>;

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
		interpolator = std::move(ceres::BiCubicInterpolator<Grid2D>{imageGrid});
	}

	template <typename T>
	bool operator()(const T* const h, T* residual) const {
		Eigen::Matrix<T, 3, 3> H{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};

		Eigen::MatrixX<T> imagePoints = H * barcodePoints;
		
		for (int j = 0; j < barcodePoints.cols(); j++) {
			T x = imagePoints(0, j) / imagePoints(2, j);
			T y = imagePoints(1, j) / imagePoints(2, j);

			T value;
			interpolator.Evaluate(y, x, &value);

			residual[j] = value - barcode.data[sizeof(double) * j];
		}

		return true;
	}

private:
	cv::Mat barcode;
	cv::Mat image;
	Eigen::MatrixXd barcodePoints;
	Interpolator interpolator;
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
