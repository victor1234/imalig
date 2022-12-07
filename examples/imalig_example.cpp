#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>

int main(int argc, char *argv[])
{
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <barcode> <image>" << std::endl;
		return 1;
	}

	/*Load images in grayscale */
	cv::Mat barcode = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	cv::Mat image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

	/* Create aruco dictionary */
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

	/* Detect aruco markers */
	auto parameters = cv::aruco::DetectorParameters::create();
	std::vector<int> markersId;
	std::vector<std::vector<cv::Point2f>> markersCorners, rejectedCandidates;
	cv::aruco::detectMarkers(image, dictionary, markersCorners, markersId, parameters, rejectedCandidates);

	/* Show result */
	//cv::imshow("barcode", barcode);
	//cv::imshow("image", image);
	//cv::Mat outImage = image.clone();
	//cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId);
	//imshow("result", outImage);
	//cv::waitKey(0);

	/* Run imalig */
	auto corners = imalig::imalig(barcode, image, markersId[0], markersCorners[0]);
	std::cout << "corners = " << corners << std::endl;

	//cv::Mat outImage2 = image.clone();
	//cv::aruco::drawDetectedMarkers(outImage2, corners, markersId);

	//imshow("result", outImage2);
	//cv::waitKey(0);

	return 0;
}
