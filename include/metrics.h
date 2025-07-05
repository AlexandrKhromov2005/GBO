#pragma once

#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>

double computeBER(const std::vector<unsigned char>& wm1, const std::vector<unsigned char>& wm2);
double computePSNR(const cv::Mat& img1, const cv::Mat& img2);
double computeSSIM(const cv::Mat& img1, const cv::Mat& img2);
double computeNCC(const cv::Mat& img1, const cv::Mat& img2);
double computeMSE(const cv::Mat& img1, const cv::Mat& img2);

