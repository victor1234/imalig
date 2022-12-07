#include "imalig/imalig.hpp"
#include <iostream>

#include <memory>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace imalig {

struct CostFunctor {
	using Grid2D = ceres::Grid2D<double, 1>;
	using Interpolator = ceres::BiCubicInterpolator<Grid2D>;

	CostFunctor(const cv::Mat barcode, const cv::Mat image)
		: barcodePoints(3, barcode.rows * barcode.cols)
	{
		barcode.convertTo(barcode64, CV_64F);
		image.convertTo(image64, CV_64F);

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

		std::cout << "=====" << std::endl;
		/* Create image interpolator */
		ceres::Grid2D<double, 1> imageGrid(reinterpret_cast<double *>(image64.data), 0, image64.rows, 0, image64.cols);
		interpolator = std::make_unique<Interpolator>(imageGrid);
	}

	template <typename T> bool operator()(const T *const h, T *residual) const
	{
		Eigen::Matrix<T, 3, 3> H;
		H << h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8];

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

std::vector<cv::Point2f> imalig(const cv::Mat barcode, cv::Mat image, const int markerId,
								const std::vector<cv::Point2f> markerCorners)
{
	std::vector<cv::Point2i> markerCorners2i = {
		{0, 0}, {barcode.rows, 0}, {barcode.rows, barcode.cols}, {0, barcode.cols}};

	cv::Mat markerCorners0;
	cv::Mat(markerCorners2i).convertTo(markerCorners0, CV_32F);

	cv::Mat H = cv::getPerspectiveTransform(markerCorners0, 10 + cv::Mat(markerCorners));
	std::cout << "H = " << H << std::endl;

	/* Create ceres problem */
	ceres::Problem problem;

	ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<CostFunctor, ceres::DYNAMIC, 9>(
		new CostFunctor(barcode, image), barcode.rows * barcode.cols);

	problem.AddResidualBlock(costFunction, nullptr, H.ptr<double>());

	/* Run solver */
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 1000;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;

	return {};
}

} // namespace imalig
