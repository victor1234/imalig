#include "imalig/BarcodeDetector.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace imalig {

std::tuple<std::vector<int>, std::vector<std::vector<cv::Point2f>>> BarcodeDetector::detect(const cv::Mat image)
{
	/* Create aruco dictionary */
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

	auto parameters = cv::aruco::DetectorParameters::create();

	std::vector<int> markersId;
	std::vector<std::vector<cv::Point2f>> markersCorners, rejectedCandidates;
	cv::aruco::detectMarkers(image, dictionary, markersCorners, markersId, parameters, rejectedCandidates);

	cv::Mat outImage = image.clone();
	cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId);
	imshow("result", outImage);
	cv::waitKey(0);
	return {markersId, markersCorners};
}
} // namespace imalig
