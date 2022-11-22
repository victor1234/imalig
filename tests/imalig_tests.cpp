#include  <catch2/catch_test_macros.hpp>

#include <imalig/imalig.hpp>


TEST_CASE("Main")
{
	bool success = imalig::imalig();
	REQUIRE(success);
}
