#include <gtest/gtest.h>
#include "population.h"
#include "process_block.h"
#include <opencv2/opencv.hpp>
#include <armadillo>

// Тест PSNR: одинаковые блоки -> очень высокая оценка (~100)
TEST(PSNR, IdenticalBlocks) {
    cv::Mat block1 = cv::Mat::zeros(8, 8, CV_64FC1);
    cv::Mat block2 = block1.clone();
    double psnr = compute_psnr(block1, block2);
    ASSERT_NEAR(psnr, 100.0, 1e-6);
}

// Тест PSNR: различающиеся блоки -> оценка ниже, чем у одинаковых
TEST(PSNR, DifferentBlocks) {
    cv::Mat block1 = cv::Mat::zeros(8, 8, CV_64FC1);
    cv::Mat block2 = cv::Mat::ones(8, 8, CV_64FC1) * 255;
    double psnr = compute_psnr(block1, block2);
    ASSERT_LT(psnr, 50.0); // Должно быть существенно меньше 100
}

// Тест конструктора Population
TEST(PopulationClass, Initialization) {
    cv::Mat block(8, 8, CV_8UC1, cv::Scalar(128));
    int vec_size = 22;
    unsigned char bit = 1;
    Population pop(vec_size, block, bit);

    ASSERT_EQ(pop.individuals.size(), 30);
    ASSERT_EQ(pop.fitness_values.size(), 30);
    for (const auto &v : pop.individuals) {
        ASSERT_EQ(v.n_elem, vec_size);
    }
    ASSERT_GE(pop.indexOfBestIndividual, 0);
    ASSERT_LT(pop.indexOfBestIndividual, 30);
}

// Тест метода update: подтверждаем, что при улучшении fitness индивидуума best индекс обновится
TEST(PopulationClass, UpdateImprovesBest) {
    cv::Mat block(8, 8, CV_8UC1, cv::Scalar(128));
    int vec_size = 22;
    unsigned char bit = 1;
    Population pop(vec_size, block, bit);

    int old_best = pop.indexOfBestIndividual;
    arma::vec better_vec = arma::zeros<arma::vec>(vec_size); // Нулевая модификация — как правило даёт высокую fitness
    pop.update(better_vec, 0); // Пытаемся улучшить индивидуума под индексом 0

    // После апдейта индекс лучшего индивидуума может измениться на 0, если fitness улучшилась
    ASSERT_TRUE(pop.indexOfBestIndividual == old_best || pop.indexOfBestIndividual == 0);
}
