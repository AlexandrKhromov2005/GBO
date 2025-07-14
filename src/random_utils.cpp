// random_utils.cpp
#include "../include/random_utils.h"

// Global RNG initialized once at program start
static std::mt19937 generator([]{
    std::random_device rd;
    return rd();
}());

// Generates a random double in [0, 1]
double uniform_random_0_1() {
    static std::uniform_real_distribution<double> dist(
        std::nextafter(0.0, 1.0), 1.0);
    return dist(generator);
}

// Generates a random integer in [0, N-1]
int random_index(int N) {
    if (N <= 0) {
        throw std::invalid_argument("N must be positive");
    }
    std::uniform_int_distribution<int> dist(0, N - 1);
    return dist(generator);
}

// Generates 4 unique random indices excluding specified indices
std::vector<int> generate_random_indices(int N, int best_index, int current_index) {
    if (N < 6) {
        throw std::invalid_argument(
            "N must be at least 6 to generate 4 unique indices excluding best_index and current_index"
        );
    }

    std::set<int> excluded = {best_index, current_index};
    std::vector<int> indices;
    indices.reserve(4);

    while (indices.size() < 4) {
        int candidate = random_index(N);
        if (excluded.count(candidate) == 0 &&
            std::find(indices.begin(), indices.end(), candidate) == indices.end()) {
            indices.push_back(candidate);
        }
    }

    return indices;
}

// Generates Gaussian random number in [0, 1] with clamping
double gaussian_random_0_1() {
    static std::normal_distribution<> dist(0.5, 0.15);
    
    // Fast path - 99.7% values will be within 3 sigma (0.05-0.95)
    double value = dist(generator);
    if (value >= 0.0 && value <= 1.0) {
        return value;
    }

    // Slow path for out-of-range values (should occur ~0.3% of time)
    return std::clamp(value, 0.0, 1.0);
}