#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <armadillo>
#include "../include/process_block.h"

// Helper to build a sequential 8x8 double matrix
static cv::Mat buildSequentialMat() {
    cv::Mat mat(8, 8, CV_64FC1);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            mat.at<double>(r, c) = r * 8 + c; // 0..63
        }
    }
    return mat;
}

TEST(ZigzagConversion, MatToZigzagOrder) {
    cv::Mat mat = buildSequentialMat();
    arma::vec zz = matToZigzag(mat);
    ASSERT_EQ(zz.n_elem, 64u);

    // Verify each element matches expected order defined by jpeg_zigzag
    for (size_t k = 0; k < 64; ++k) {
        int linear = jpeg_zigzag[k];
        EXPECT_DOUBLE_EQ(zz(k), static_cast<double>(linear)) << "Mismatch at index " << k;
    }
}

TEST(ZigzagConversion, RoundTripIntegrity) {
    cv::Mat original = buildSequentialMat();
    arma::vec zz = matToZigzag(original);
    cv::Mat reconstructed = zigzagToMat(zz);

    // Compare matrix elements
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            double orig_val = original.at<double>(r, c);
            double recon_val = reconstructed.at<double>(r, c);
            EXPECT_DOUBLE_EQ(orig_val, recon_val) << "Mismatch at (" << r << "," << c << ")";
        }
    }
}
