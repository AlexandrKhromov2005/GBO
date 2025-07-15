#include <gtest/gtest.h>
#include "process_block.h"
#include "random_utils.h"
#include <opencv2/opencv.hpp>
#include <armadillo>

// Тест для проверки функций зигзаг-преобразования
TEST(ZigzagConversion, Reversibility) {
    // Создаем тестовую матрицу 8x8
    cv::Mat original_mat(8, 8, CV_64FC1);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            original_mat.at<double>(i, j) = i * 8 + j;
        }
    }

    // Преобразуем матрицу в зигзаг-вектор
    arma::vec zigzag_vec = matToZigzag(original_mat);

    // Проверяем, что размер вектора равен 64
    ASSERT_EQ(zigzag_vec.n_elem, 64);

    // Преобразуем вектор обратно в матрицу
    cv::Mat result_mat = zigzagToMat(zigzag_vec);

    // Сравниваем исходную и результирующую матрицы
    cv::Mat diff;
    cv::absdiff(original_mat, result_mat, diff);
    double total_diff = cv::sum(diff)[0];

    // Различия должны быть пренебрежимо малы
    EXPECT_NEAR(total_diff, 0.0, 1e-9);
}

// Тест, демонстрирующий правильный порядок формирования зигзаг-вектора
TEST(ZigzagConversion, MappingOrder) {
    // Матрица со значениями 0..63 по строкам
    cv::Mat mat(8, 8, CV_64FC1);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            mat.at<double>(r, c) = r * 8 + c;
        }
    }

    arma::vec zz = matToZigzag(mat);

    // Ожидаемый вектор: элемент k содержит linear index jpeg_zigzag[k]
    arma::vec expected(64);
    for (int k = 0; k < 64; ++k) {
        expected(k) = static_cast<double>(jpeg_zigzag[k]);
    }

    for (size_t k = 0; k < 64; ++k) {
        EXPECT_DOUBLE_EQ(zz(k), expected(k)) << "Mismatch at position " << k;
    }
}
