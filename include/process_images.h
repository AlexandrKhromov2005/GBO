#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <armadillo>

std::vector<cv::Mat> splitImageInto8x8Blocks(const cv::Mat& image);

cv::Mat assembleImageFrom8x8Blocks(const std::vector<cv::Mat>& blocks);

std::vector<unsigned char> extract_watermark_bits(const cv::Mat& image);

cv::Mat reconstruct_watermark_image(const std::vector<unsigned char>& bits);