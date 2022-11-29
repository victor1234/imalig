#include  <catch2/catch_test_macros.hpp>
#include <opencv2/highgui.hpp>

#include <imalig/imalig.hpp>


TEST_CASE("Main")
{
  cv::Mat barcode = cv::imread("fixtures/barcode.png");
  cv::Mat image = cv::imread("fixtures/image.jpg");
  
  cv::imshow("barcode", barcode);
  cv::imshow("image", image);
  cv::waitKey(0);


	bool success = imalig::imalig(barcode, image);
	REQUIRE(success);
}
