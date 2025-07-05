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
    cv::Mat mat(8, 8, CV_64FC1);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            mat.at<double>(i, j) = i * 8 + j; // уникальные значения
        }
    }

    arma::vec zz = matToZigzag(mat);

    // Формируем эталонный вектор с использованием jpeg_zigzag
    arma::vec expected = arma::zeros<arma::vec>(64);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int idx = jpeg_zigzag[i * 8 + j];
            expected(idx) = mat.at<double>(i, j);
        }
    }

    for (size_t k = 0; k < 64; ++k) {
        ASSERT_EQ(zz(k), expected(k));
    }
}



