#pragma once
#include <random>
#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm> 

double uniform_random_0_1();
int random_index(int N);
std::vector<int> generate_random_indices(int N, int best_index, int current_index);
double gaussian_random_0_1();