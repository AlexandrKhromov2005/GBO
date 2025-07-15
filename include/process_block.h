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

// Updated embedding region for scheme 0: mask
// 00000001
// 00000011
// 00000110
// 00001100
// 00011000
// 00110000
// 01100000
// 11000000
// Values are zig-zag indices corresponding to the linear positions above
const std::vector<std::vector<int>> embeding_region  = {{4, 5, 7, 10, 14, 15, 23, 41, 48, 52, 53, 56, 57, 59, 60},
                                                        {11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42}};
const std::vector<std::vector<int>> s1_region        = {{5, 7, 15, 23, 41, 53, 57, 59},
                                                        {11, 13, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41}};
const std::vector<std::vector<int>> s0_region        = {{4, 10, 14, 48, 52, 56, 60},
                                                        {12, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42}};