#include "../include/random_utils.h"

// Generates a random double in the range [0, 1]
double uniform_random_0_1() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dist(std::nextafter(0.0, 1.0), 1.0);

    return dist(gen);
}

// Generates a random integer in the range [0, N-1]
int random_index(int N) {
    if (N <= 0)
        throw std::invalid_argument("N must be positive");

    static std::random_device rd;      
    static std::mt19937 gen(rd());     

    std::uniform_int_distribution<> dist(0, N - 1);
    return dist(gen);
}

// Generates 4 unique random indices in the range [0, N-1] excluding best_index and current_index
std::vector<int> generate_random_indices(int N, int best_index, int current_index) {
    if (N < 6)
        throw std::invalid_argument("N must be at least 6 to generate 4 unique indices excluding best_index and current_index");

    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(0, N - 1);
    std::set<int> excluded = {best_index, current_index};
    std::set<int> selected;

    while (selected.size() < 4) {
        int idx = dist(gen);
        if (excluded.count(idx) == 0 && selected.count(idx) == 0) {
            selected.insert(idx);
        }
    }

    return std::vector<int>(selected.begin(), selected.end());
}

// Generates a Gaussian random number in the range [0, 1] with specified mean and standard deviation
double gaussian_random_0_1() {
    double mean = 0.5;
    double stddev = 0.15;
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::normal_distribution<> dist(mean, stddev);

    double x;
    do {
        x = dist(gen);
    } while (x < 0.0 || x > 1.0); // жёсткое обрезание: только [0, 1]

    return x;
}
