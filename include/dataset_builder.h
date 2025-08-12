#pragma once
#include <opencv2/opencv.hpp>
#include "gbo.h"
#include <vector>
#include <algorithm>

const int amount_of_schemes = 2;

const std::vector<std::string> images = {
    "images/airplane.png",
    "images/baboon.png",
    "images/boat.png",
    "images/bridge.png",
    "images/earth_from_space.png",
    "images/lake.png",
    "images/lenna.png",
    "images/pepper.png"
};
    
/**
 * @brief Embed the same bit (0 or 1) into every 8x8 block of the provided image
 *        using the specified embedding scheme.
 *
 * The function keeps the original image unchanged and returns a new image
 * with the embedded data.
 *
 * @param src      Grayscale source image (type CV_8UC1, size multiple of 8).
 * @param bit      Bit value to embed (must be 0 or 1).
 * @param scheme   Embedding scheme index (0 or 1) determining coefficient regions.
 * @return cv::Mat Image of the same size/type as src with the embedded bit.
 *
 * @throws std::invalid_argument if input image is empty, not CV_8UC1, or size not divisible by 8.
 */
cv::Mat embedUniformBits(const cv::Mat& src, unsigned char bit, int scheme = 0);

// Types of attacks available in attacks.h
enum class AttackType {
    BrightnessIncrease,
    BrightnessDecrease,
    ContrastIncrease,
    ContrastDecrease,
    SaltPepperNoise,
    SpeckleNoise,
    HistogramEqualization,
    Sharpening,
    JPEGCompression,
    GaussianFiltering,
    MedianFiltering,
    AverageFiltering
};

/**
 * @brief Simulate an attack on the given image.
 *
 * @param src      Input image (CV_8UC1).
 * @param type     Attack type (see AttackType).
 * @param param1   First parameter (meaning depends on attack).
 * @param param2   Second parameter (meaning depends on attack).
 * @return cv::Mat Attacked image.
 */
cv::Mat simulateAttack(const cv::Mat& src, AttackType type, double param1 = 10.0, int param2 = 3);

// Build dataset using embedding and attacks
auto buildDataset(int tau_max = 3) -> void;

