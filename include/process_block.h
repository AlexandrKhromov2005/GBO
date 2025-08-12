#pragma once
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <vector>

const int jpeg_zigzag[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

arma::vec matToZigzag(const cv::Mat& block);
cv::Mat zigzagToMat(const arma::vec& zz);
cv::Mat applyVectorToBlock(const arma::vec& vec, const cv::Mat& block, int scheme = 0);
unsigned char getBitFromBlock(const cv::Mat& block, int scheme = 0);
double calcFitnessValue(const cv::Mat& block, const arma::vec& vec, unsigned char bit, int scheme = 0);
double compute_psnr(const cv::Mat& orig, const cv::Mat& test);
double getRegionSum(const cv::Mat& dctBlock, std::vector<int> region);

// Region definition updated from user-provided 8×8 masks (1 = s1, 2 = s0)
// Scheme 0 mask:
// 00000012
// 00000212
// 00001210
// 00021200
// 00121000
// 02120000
// 12100000
// 12000000
//
// Scheme 1 mask:
// 00000012
// 00010212
// 00201210
// 01021200
// 00121000
// 02120000
// 12100000
// 12000000
// Values are zig-zag indices corresponding to the linear positions above
const std::vector<std::vector<int>> embeding_region = {
    {3, 4, 5, 6, 7, 10, 11, 14, 15, 22, 23, 40, 41, 48, 49, 52, 53, 56, 57, 58, 59, 60},
    {3, 4, 5, 6, 7, 10, 11, 14, 15, 20, 22, 23, 25, 26, 40, 41, 48, 49, 52, 53, 56, 57, 58, 59, 60}
};
const std::vector<std::vector<int>> s1_region = {
    /*{3, 4, 7, 15, 40, 41, 49, 52, 53, 57, 58}*/ //base case
    {3, 4, 5, 6, 7, 10, 11, 15, 22, 40, 41},
    {3, 4, 7, 15, 20, 25, 40, 41, 49, 52, 53, 57, 58}
};
const std::vector<std::vector<int>> s0_region = {
    /*{5, 6, 10, 11, 14, 22, 23, 48, 56, 59, 60}*/ //base case
    {14, 23, 48, 49, 52, 53, 56, 57, 58, 59, 60},
    {5, 6, 10, 11, 14, 22, 23, 26, 48, 56, 59, 60}
};