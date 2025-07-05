#include "../include/population.h"

/**
 * 
    * @brief Calculates the fitness value for a given vector and block.
    * @param block Input OpenCV block of size 8x8, type CV_64FC1 (64-bit floating point, single channel).
    * @param vec Input vector of size 22, containing the values to be applied to the block.
    * @param bit The bit to be used in the fitness calculation (0 or
 */
Population::Population(int vector_size, const cv::Mat& block, unsigned char bit) : vector_size(vector_size), block(block), bit(bit) {
    if (block.empty()) {
        throw std::invalid_argument("Population: empty block");
    }
    if (block.rows != 8 || block.cols != 8) {
        throw std::invalid_argument("Population: block must be 8x8");
    }
    if (block.type() != CV_64FC1) {
        throw std::invalid_argument("Population: block must be CV_64FC1 (double)");
    }

    individuals.resize(population_size, arma::vec(vector_size));
    fitness_values.resize(population_size, 0.0);

    // Initialize the first individual
    individuals[0].randu(vector_size);
    individuals[0] = 2.0 * th * individuals[0] - th;
    fitness_values[0] = calcFitnessValue(block, individuals[0], bit);

    // Set initial best and worst
    indexOfBestIndividual = 0;
    worstIndividual = individuals[0];
    worstFitnessValue = fitness_values[0];

    // Iterate through the rest of the population
    for (int i = 1; i < population_size; ++i) {
        individuals[i].randu(vector_size); 
        individuals[i] = 2.0 * th * individuals[i] - th;
        fitness_values[i] = calcFitnessValue(block, individuals[i], bit);
        if (fitness_values[i] < fitness_values[indexOfBestIndividual]) {
            indexOfBestIndividual = i;
        }
        if (fitness_values[i] > worstFitnessValue) {
            worstIndividual = individuals[i];
            worstFitnessValue = fitness_values[i];
        }
    }
}

/**
 * @brief Updates the population with a new vector and its fitness value.
 * If the new fitness value is better than the current one, it replaces the individual at the given index.
 * If the new fitness value is worse than the worst fitness value, it updates the worst individual.
 * @param vec The new vector to be added to the population.
 * @param index The index of the individual to be updated.
 */
void Population::update(arma::vec& vec, int index) {
    double fitness_value = calcFitnessValue(block, vec, bit);
    if (fitness_value < fitness_values[index]) {
        individuals[index] = vec;
        fitness_values[index] = fitness_value;
        if (fitness_value < fitness_values[indexOfBestIndividual]) {
            indexOfBestIndividual = index;
        }
        
    } else if (fitness_value > worstFitnessValue) {
        worstIndividual = vec;
        worstFitnessValue = fitness_value;
    }
}