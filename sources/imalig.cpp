#include "imalig/imalig.hpp"

#include <iostream>

#include <memory>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <types.h>

namespace imalig {

std::vector<cv::Point2f> cornersFromH(const std::vector<cv::Point2f> corners0, const cv::Mat H)
{
	std::vector<cv::Point2f> corners;
	for (const auto &cp : corners0) {
		// std::cout << cp << std::endl;
		cv::Vec3d p(cp.x, cp.y, 1);
		cv::Vec3d p2 = cv::Matx33d(H) * p;
		// std::cout << p2 << std::endl;
		corners.emplace_back(cv::Point2d(p2[0] / p2[2], p2[1] / p2[2]));
	}
	return corners;
}

class DrawCornersCallback : public ceres::IterationCallback {
  public:
	DrawCornersCallback(const cv::Mat &image, const std::vector<cv::Point2f> &corners0, const cv::Mat &H)
		: image_(image), corners0_(corners0), H_(H)
	{
	}

	ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) override
	{
		cv::Mat image = image_.clone();
		cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

		const auto corners_ = cornersFromH(corners0_, H_);

		cv::line(image, corners_[0], corners_[1], cv::Scalar(0, 0, 255), 1);
		cv::line(image, corners_[1], corners_[2], cv::Scalar(0, 0, 255), 1);
		cv::line(image, corners_[2], corners_[3], cv::Scalar(0, 0, 255), 1);
		cv::line(image, corners_[3], corners_[0], cv::Scalar(0, 0, 255), 1);

		// draw iteration
		cv::putText(image, std::to_string(summary.iteration), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
					cv::Scalar(0, 0, 255), 1);
		// draw cost
		cv::putText(image, std::to_string(summary.cost), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
					cv::Scalar(0, 0, 255), 1);

		// print H
		// std::cout << "H: " << summary.user_state[0] << std::endl;

		cv::imshow("corners", image);
		cv::waitKey(10);
		return ceres::SOLVER_CONTINUE;
	}

  private:
	cv::Mat image_;
	std::vector<cv::Point2f> corners0_;
	cv::Mat H_;
};

std::vector<cv::Point2f> Imalig::process(const cv::Mat &barcode, const cv::Mat &image, const int markerId,
										 const std::vector<cv::Point2f> markerCorners)
{
	/* Set synthetic marker corners */
	const std::vector<cv::Point2i> markerCorners2i = {
		{0, 0}, {barcode.rows, 0}, {barcode.rows, barcode.cols}, {0, barcode.cols}};

	/* Convert to cv::Mat32 */
	cv::Mat markerCorners0;
	cv::Mat(markerCorners2i).convertTo(markerCorners0, CV_32F);

	/* Get initial H */
	cv::Mat H = cv::getPerspectiveTransform(markerCorners0, cv::Mat(markerCorners));
	// std::cout << "H: " << H << std::endl;

	/* Get barcode mask */
	cv::Mat mask = getMask(barcode, 20);
	spdlog::info("Masking ratio: {}", cv::countNonZero(mask) / (double)mask.total());

	/* Create ceres problem */
	ceres::Problem problem;

	ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<CostFunctor, ceres::DYNAMIC, 8>(
		new CostFunctor(barcode, mask, image), cv::countNonZero(mask));

	problem.AddResidualBlock(costFunction, nullptr, H.ptr<double>());

	/* Run solver */
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	// options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;
	options.max_num_iterations = 1000;
	options.logging_type = ceres::SILENT;
	options.update_state_every_iteration = true;
	// options.callbacks = {new DrawCornersCallback(image, markerCorners0, H)};
	options.use_nonmonotonic_steps = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	spdlog::info(summary.FullReport());

	/* Compute new marker corners */
	auto preciseMarkerCorners = cornersFromH(markerCorners0, H);

	// spdlog::info("d: {:.2f} {:.2F}", d[0], d[1]);

	return preciseMarkerCorners;
}

cv::Mat Imalig::getMask(const cv::Mat &barcode, const int radious)
{
	cv::Mat mask;
	if (radious == -1) {
		mask = cv::Mat(barcode.rows, barcode.cols, CV_8UC1, cv::Scalar(255));
	} else {
		cv::blur(barcode, mask, cv::Size(radious, radious));
		// set 255 to zero
		cv::threshold(mask, mask, 254, 0, cv::THRESH_TOZERO_INV);
	}

	return mask;
}

} // namespace imalig
