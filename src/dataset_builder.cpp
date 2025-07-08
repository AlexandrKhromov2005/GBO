#include "../include/dataset_builder.h"
#include "../include/process_block.h"
#include <stdexcept>
#include "../include/attacks.h"
#include "../include/process_images.h"

cv::Mat embedUniformBits(const cv::Mat& src, unsigned char bit, int scheme) {
    if (src.empty()) {
        throw std::invalid_argument("embedUniformBits: empty input image");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("embedUniformBits: input image must be CV_8UC1 (grayscale 8-bit)");
    }
    if (src.rows % 8 != 0 || src.cols % 8 != 0) {
        throw std::invalid_argument("embedUniformBits: image size must be divisible by 8 in both dimensions");
    }
    if (bit != 0 && bit != 1) {
        throw std::invalid_argument("embedUniformBits: bit must be 0 or 1");
    }
    if (scheme < 0 || scheme >= static_cast<int>(embeding_region.size())) {
        throw std::invalid_argument("embedUniformBits: invalid scheme index");
    }

    const int vector_size = static_cast<int>(embeding_region[scheme].size());
    cv::Mat dst = src.clone();
    GBO optimizer;

    for (int y = 0; y < src.rows; y += 8) {
        for (int x = 0; x < src.cols; x += 8) {
            cv::Rect roi(x, y, 8, 8);
            cv::Mat block = src(roi);
            cv::Mat embedded_block = optimizer.main_loop(block, vector_size, bit, scheme);
            embedded_block.copyTo(dst(roi));
        }
    }
    return dst;
}


cv::Mat simulateAttack(const cv::Mat& src, AttackType type, double param1, int param2) {
    if (src.empty()) {
        throw std::invalid_argument("simulateAttack: empty input image");
    }
    if (src.type() != CV_8UC1) {
        throw std::invalid_argument("simulateAttack: input image must be CV_8UC1");
    }

    switch (type) {
        case AttackType::BrightnessIncrease:
            return brightnessIncrease(src, static_cast<int>(param1));
        case AttackType::BrightnessDecrease:
            return brightnessDecrease(src, static_cast<int>(param1));
        case AttackType::ContrastIncrease:
            return contrastIncrease(src, param1);
        case AttackType::ContrastDecrease:
            return contrastDecrease(src, param1);
        case AttackType::SaltPepperNoise:
            return saltPepperNoise(src, param1 / 100.0); // param1 as percentage
        case AttackType::SpeckleNoise:
            return speckleNoise(src, param1);
        case AttackType::HistogramEqualization:
            return histogramEqualization(src);
        case AttackType::Sharpening:
            return sharpening(src);
        case AttackType::JPEGCompression:
            return jpegCompression(src, static_cast<int>(param1));
        case AttackType::GaussianFiltering:
            return gaussianFiltering(src, param2);
        case AttackType::MedianFiltering:
            return medianFiltering(src, param2);
        case AttackType::AverageFiltering:
            return averageFiltering(src, param2);
        default:
            throw std::invalid_argument("simulateAttack: unsupported attack type");
    }
}

const std::vector<AttackType> attacks = {AttackType::ContrastIncrease, AttackType::JPEGCompression};

struct ISB { // Image, Scheme and Bit
    cv::Mat image;
    int scheme;
    unsigned char bit;

    ISB(cv::Mat img, int s, unsigned char b ) : image(img), scheme(s), bit(b) {}
};

struct BSB { // Block, Scheme and Bit
    cv::Mat block;
    int scheme;
    unsigned char bit;

    BSB(cv::Mat blk, int s, unsigned char b) : block(blk), scheme(s), bit(b) {}
};

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <random>

void buildDataset() {
size_t total_blocks_estimate = 0;
for (const auto& img_path : images) {
    cv::Mat tmp = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (tmp.empty()) continue;
    size_t blocks_in_img = (tmp.rows / 8) * (tmp.cols / 8);
    size_t variants = amount_of_schemes * 2 * (1 + attacks.size());
    total_blocks_estimate += blocks_in_img * variants;
}
size_t processed_blocks = 0;
std::vector<size_t> scheme_counts(amount_of_schemes, 0);

int image_index = 0;
for (auto& image : images) {
        std::cout << "\n\n[" << ++image_index << "/" << images.size() << "] Processing image: " << image << std::endl;
        std::vector<int> nums(amount_of_schemes, 0);
        std::vector<ISB> images_copies;
        cv::Mat img = cv::imread(image, cv::IMREAD_GRAYSCALE);
        for (int i = 0; i < amount_of_schemes; ++i) {
            cv::Mat black_img = embedUniformBits(img, 0, i);
            cv::Mat white_img = embedUniformBits(img, 1, i);
            images_copies.emplace_back(black_img, i,  0);
            images_copies.emplace_back(white_img, i,  1);
        }
        std::cout << "Geberated clear copies with 0 and 1 bits for all schemes" << std::endl;

        int n = images_copies.size();
        for (int i = 0; i < n; ++i) {
            for (auto& attack : attacks) {
                cv::Mat attacked_image = simulateAttack(images_copies[i].image, attack);
                images_copies.emplace_back(attacked_image, images_copies[i].scheme, images_copies[i].bit);
            }
        }
        std::cout << "Geberated attacked copies with 0 and 1 bits for all schemes" << std::endl;

        std::vector<std::vector<BSB>> sets_of_blocks;
        for (const auto& item : images_copies) {
            std::vector<BSB> blocks;
            std::vector<cv::Mat> image_blocks = splitImageInto8x8Blocks(item.image);
            for (const auto& block : image_blocks) {
                blocks.emplace_back(block, item.scheme, item.bit);
            }
            sets_of_blocks.push_back(blocks);
            std::cout << "    Generated " << blocks.size() << " blocks for scheme " << item.scheme 
                      << " bit " << static_cast<int>(item.bit) << std::endl;
        }

        for (int i = 0; i < sets_of_blocks[0].size(); ++i) {
            std::vector<int> errors(amount_of_schemes, 0);
            for (auto& s : sets_of_blocks) {
                unsigned char bit = getBitFromBlock(s[i].block, s[i].scheme);
                if (bit != s[i].bit) errors[s[i].scheme]++; 
            }
            int min_error = *std::min_element(errors.begin(), errors.end());
            std::vector<int> min_indices;
            for (int idx = 0; idx < errors.size(); ++idx) {
                if (errors[idx] == min_error) {
                    min_indices.push_back(idx);
                }
            }
            // Randomly select one of the schemes with the minimal number of errors
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, static_cast<int>(min_indices.size()) - 1);
            int min_index = min_indices[dis(gen)];

            std::string name_of_dir = "dataset/scheme_" + std::to_string(min_index) + "/block_" + std::to_string(nums[min_index]) + ".png";
            nums[min_index]++;
            scheme_counts[min_index]++;
            std::filesystem::create_directories("dataset/scheme_" + std::to_string(min_index));
            cv::imwrite(name_of_dir, sets_of_blocks[0][i].block);

            processed_blocks++;
            double perc = 100.0 * static_cast<double>(processed_blocks) / static_cast<double>(total_blocks_estimate);
            std::cout << "\rBuilding dataset: " << std::fixed << std::setprecision(1) << perc << "%" << std::flush;

        }

    std::cout << "\n\nDataset building complete. Blocks per scheme:" << std::endl;
    for (int i = 0; i < amount_of_schemes; ++i) {
        std::cout << "  Scheme " << i << ": " << scheme_counts[i] << " blocks" << std::endl;
    }

    }
}