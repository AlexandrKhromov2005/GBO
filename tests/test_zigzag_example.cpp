#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <armadillo>
#include <iostream>
#include "../include/process_block.h"

static cv::Mat buildSequentialMat() {
    cv::Mat mat(8, 8, CV_64FC1);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            mat.at<double>(r, c) = r * 8 + c;
    return mat;
}

// Тест-демонстрация: печатает пример работы функций
TEST(ZigzagConversion, ExamplePrint) {
    cv::Mat original = buildSequentialMat();
    arma::vec zz = matToZigzag(original);
    cv::Mat reconstructed = zigzagToMat(zz);

    // Выводим первую строку исходной матрицы
    std::cout << "\n[Example] Original first row: ";
    for (int c = 0; c < 8; ++c) std::cout << original.at<double>(0, c) << " ";
    std::cout << std::endl;

    // Выводим первые 16 элементов zigzag-вектора
    std::cout << "[Example] Zigzag first 16: ";
    for (int i = 0; i < 16; ++i) std::cout << zz(i) << " ";
    std::cout << std::endl;

    // Считаем максимальную абсолютную разницу
    cv::Mat diff; cv::absdiff(original, reconstructed, diff);
    double maxDiff; cv::minMaxLoc(diff, nullptr, &maxDiff);
    std::cout << "[Example] Max abs diff after round-trip: " << maxDiff << (maxDiff < 1e-9 ? " (OK)" : " (FAIL)") << std::endl;

    EXPECT_LT(maxDiff, 1e-9);
}
