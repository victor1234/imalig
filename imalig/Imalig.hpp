#pragma once

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <opencv2/core.hpp>
#include <vector>

#include <spdlog/spdlog.h>

namespace imalig {

class Imalig {
  public:
	Imalig() = default;

	std::vector<cv::Point2f> process(const cv::Mat &barcode, const cv::Mat &image, const int markerId,
									 const std::vector<cv::Point2f> markerCorners);
	cv::Mat getMask(const cv::Mat &barcode, const int radious);
};

struct CostFunctor {
	using Grid2D = ceres::Grid2D<double, 1>;
	using Interpolator = ceres::BiCubicInterpolator<Grid2D>;

	CostFunctor(const cv::Mat barcode, const cv::Mat mask, const cv::Mat image)
		: barcodePoints(3, cv::countNonZero(mask))
	{
		barcode.convertTo(barcode64, CV_64F); //, 1. / 255.);
		image.convertTo(image64, CV_64F);	  //, 1./ 255.);

		// log barcode mean and std
		//  cv::Scalar mean, stddev;
		//  cv::meanStdDev(barcode64, mean, stddev);
		//  spdlog::info("barcode mean: {:.2f}, std: {:.2f}", mean[0], stddev[0]);
		//
		//  //log image mean and std
		//  cv::meanStdDev(image64, mean, stddev);
		//  spdlog::info("image mean: {:.2f}, std: {:.2f}", mean[0], stddev[0]);

		/* Create barcode pixel coordinate matrix */
		int j = 0;
		for (int y = 0; y < barcode.rows; y++) {
			for (int x = 0; x < barcode.cols; x++) {
				if (mask.at<uchar>(y, x) > 0) {
					barcodePoints(0, j) = x;
					barcodePoints(1, j) = y;
					barcodePoints(2, j) = 1;
					j++;
				}
			}
		}

		/* Create image interpolator */
		ceres::Grid2D<double, 1> imageGrid(reinterpret_cast<double *>(image64.data), 0, image64.rows, 0, image64.cols);
		interpolator = std::make_unique<Interpolator>(imageGrid);
	}

	template <typename T> bool operator()(const T *const h, T *residual) const
	{
		Eigen::Matrix<T, 3, 3> H{{h[0], h[1], h[2]}, {h[3], h[4], h[5]}, {h[6], h[7], T(1)}};

		Eigen::MatrixX<T> imagePoints = H * barcodePoints;

		for (int j = 0; j < barcodePoints.cols(); j++) {
			T x = imagePoints(0, j) / imagePoints(2, j);
			T y = imagePoints(1, j) / imagePoints(2, j);

			T value;
			interpolator->Evaluate(y, x, &value);

			residual[j] = value - T(barcode64.data[sizeof(double) * j]);
		}

		return true;
	}

  private:
	cv::Mat barcode64;
	cv::Mat image64;
	Eigen::MatrixXd barcodePoints;
	std::unique_ptr<Interpolator> interpolator;
};

} // namespace imalig
