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

std::vector<cv::Point2f> Imalig::process(const cv::Mat barcode, cv::Mat image, const int markerId,
										 const std::vector<cv::Point2f> markerCorners)
{
	std::vector<cv::Point2i> markerCorners2i = {
		{0, 0}, {barcode.rows, 0}, {barcode.rows, barcode.cols}, {0, barcode.cols}};

	cv::Mat markerCorners0;
	cv::Mat(markerCorners2i).convertTo(markerCorners0, CV_32F);

	cv::Mat H = cv::getPerspectiveTransform(markerCorners0, 0 + cv::Mat(markerCorners));
	// std::cout << "H = " << H << std::endl;

	/* Create ceres problem */
	ceres::Problem problem;

	ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<CostFunctor, ceres::DYNAMIC, 9>(
		new CostFunctor(barcode, image), barcode.rows * barcode.cols);

	problem.AddResidualBlock(costFunction, nullptr, H.ptr<double>());

	/* Run solver */
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	options.max_num_iterations = 1000;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	// std::cout << summary.FullReport() << std::endl;

	/* Compute new marker corners */
	std::vector<cv::Point2f> preciseMarkerCorners;
	for (const auto &cp : markerCorners2i) {
		// std::cout << cp << std::endl;
		cv::Vec3d p(cp.x, cp.y, 1);
		cv::Vec3d p2 = cv::Matx33d(H) * p;
		// std::cout << p2 << std::endl;
		preciseMarkerCorners.emplace_back(cv::Point2d(p2[0] / p2[2], p2[1] / p2[2]));
	}

	return preciseMarkerCorners;
}

} // namespace imalig
