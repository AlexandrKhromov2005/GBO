#pragma once
#include "population.h"
#include "random_utils.h"
#include <cmath>

class GBO {
private:
    static constexpr double betta_min   = 0.2;
    static constexpr double betta_max   = 1.2;
    static constexpr double PI          = 3.14159265358979323846;
    static constexpr double angle       = 1.5 * PI;
    static constexpr double PR          = 0.5; // Probability of LEO
    const int iterations                = 40;

public:
    GBO() = default;
    Population population;
    double th = population.get_th();
    cv::Mat main_loop(cv::Mat& block, int vector_size, unsigned char bit);
};