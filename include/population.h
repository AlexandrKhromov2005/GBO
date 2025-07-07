#pragma once
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <vector>
#include "process_block.h"


class Population {
private:
    const int gbo_iterations    = 40;
    const int population_size   = 30;
    const double th             = 10.0;
public:
    cv::Mat block; 
    int vector_size;
    int indexOfBestIndividual;
    arma::vec worstIndividual;
    double worstFitnessValue;
    unsigned char bit;
    int scheme;
    std::vector<arma::vec> individuals;
    std::vector<double> fitness_values;

    Population() = default;
    Population(int vector_size, const cv::Mat& block, unsigned char bit, int scheme = 0);
    void update(arma::vec& vec, int index);
    double get_th() const { return th; }
};