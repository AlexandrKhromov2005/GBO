// random_utils.h
#pragma once
#include <random>
#include <stdexcept>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>

// Generates a random double in [0, 1] using uniform distribution
double uniform_random_0_1();

// Generates a random integer in [0, N-1]
int random_index(int N);

// Generates 4 unique random indices in [0, N-1] excluding best_index and current_index
std::vector<int> generate_random_indices(int N, int best_index, int current_index);

// Generates a Gaussian random number in [0, 1] (mean=0.5, stddev=0.15)
double gaussian_random_0_1();