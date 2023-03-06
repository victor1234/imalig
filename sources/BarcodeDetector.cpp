#include "imalig/BarcodeDetector.hpp"
#include <opencv2/highgui/highgui.hpp>

namespace imalig {

BarcodeDetector::BarcodeDetector() { dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250); }

cv::Mat BarcodeDetector::drawMarker(int id, int size)
{
	cv::Mat markerImage;
	cv::aruco::drawMarker(dictionary, id, size, markerImage, 1);

	return markerImage;
}

cv::Mat BarcodeDetector::drawMarker(int id, std::vector<cv::Point2f> corners)
{
	cv::Mat markerImage;
	int size = std::max(cv::norm(corners[0] - corners[2]), cv::norm(corners[1] - corners[3]));
	return drawMarker(id, size);
}

std::tuple<std::vector<int>, std::vector<std::vector<cv::Point2f>>> BarcodeDetector::detect(const cv::Mat image)
{
	const auto parameters = cv::aruco::DetectorParameters::create();

	std::vector<int> markersId;
	std::vector<std::vector<cv::Point2f>> markersCorners, rejectedCandidates;
	cv::aruco::detectMarkers(image, dictionary, markersCorners, markersId, parameters, rejectedCandidates);

	return {markersId, markersCorners};
}
} // namespace imalig
