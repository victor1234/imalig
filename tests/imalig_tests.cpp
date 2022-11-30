#include  <catch2/catch_test_macros.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>

#include <imalig/imalig.hpp>


TEST_CASE("Main")
{
  cv::Mat barcode = cv::imread("fixtures/barcode.png");
  cv::Mat image = cv::imread("fixtures/image.jpg");

  /* Create aruco dictionary */
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  
  auto parameters = cv::aruco::DetectorParameters::create();
  std::vector<int> markersId;
  std::vector<std::vector<cv::Point2f>> markersCorners, rejectedCandidates;
  cv::aruco::detectMarkers(image, dictionary, markersCorners, markersId, parameters, rejectedCandidates);

  cv::imshow("barcode", barcode);
  cv::imshow("image", image);
  cv::Mat outImage = image.clone();
  cv::aruco::drawDetectedMarkers(outImage, markersCorners, markersId);
  imshow("result", outImage);
  cv::waitKey(0);


	bool success = imalig::imalig(barcode, image);
	REQUIRE(success);
}
