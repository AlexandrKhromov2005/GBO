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

const std::vector<AttackType> attacks = {AttackType::JPEGCompression, AttackType::ContrastIncrease};

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
#include <unordered_map>

void buildDataset(int tau_max) {
    std::filesystem::create_directories("dataset/Dir1");
    std::filesystem::create_directories("dataset/Dir2");
    std::filesystem::create_directories("dataset/Dirrand");
    
    int dir1_count = 0, dir2_count = 0, dirrand_count = 0;
    size_t total_blocks_estimate = 0;
    
    for (const auto& img_path : images) {
        cv::Mat tmp = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (tmp.empty()) continue;
        total_blocks_estimate += (tmp.rows / 8) * (tmp.cols / 8);
    }
    
    size_t processed_blocks = 0;
    int image_index = 0;
    
    for (const auto& image_path : images) {
        std::cout << "\n\n[" << ++image_index << "/" << images.size() << "] Processing image: " << image_path << std::endl;
        
        cv::Mat original_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (original_img.empty()) {
            std::cout << "Error: Could not load image " << image_path << std::endl;
            continue;
        }
        
        // Шаг 1: Создаем 4 копии изображения для каждого класса (I0^1, I1^1, I0^2, I1^2)
        std::vector<ISB> image_copies;
        for (int scheme = 0; scheme < 2; ++scheme) {
            cv::Mat img_bit0 = embedUniformBits(original_img, 0, scheme);
            cv::Mat img_bit1 = embedUniformBits(original_img, 1, scheme);
            image_copies.emplace_back(img_bit0, scheme, 0);
            image_copies.emplace_back(img_bit1, scheme, 1);
        }
        std::cout << "Generated embedded copies for both schemes" << std::endl;
        
        // Шаг 3: Атакуем каждую копию отдельно (JPEG70 или Увеличение контрастности)
        int original_copies_count = image_copies.size();
        for (int i = 0; i < original_copies_count; ++i) {
            // JPEG70 атака
            cv::Mat jpeg_attacked = simulateAttack(image_copies[i].image, AttackType::JPEGCompression, 70);
            image_copies.emplace_back(jpeg_attacked, image_copies[i].scheme, image_copies[i].bit);
            
            // Увеличение контрастности
            cv::Mat contrast_attacked = simulateAttack(image_copies[i].image, AttackType::ContrastIncrease, 1.2);
            image_copies.emplace_back(contrast_attacked, image_copies[i].scheme, image_copies[i].bit);
        }
        std::cout << "Generated attacked copies: JPEG70 and Contrast increase (separately)" << std::endl;
        
        // Шаг 4: Делим все копии на блоки 8x8
        std::vector<std::vector<BSB>> block_groups;
        for (const auto& img_copy : image_copies) {
            std::vector<cv::Mat> blocks_8x8 = splitImageInto8x8Blocks(img_copy.image);
            std::vector<BSB> blocks_with_info;
            for (const auto& block : blocks_8x8) {
                blocks_with_info.emplace_back(block, img_copy.scheme, img_copy.bit);
            }
            block_groups.push_back(blocks_with_info);
        }
        
        // Шаг 5-7: Классификация блоков по новому алгоритму
        // Теперь у нас 12 групп блоков: 4 схемы×биты (оригинал) + 4 JPEG + 4 контраст
        // Для каждой схемы: 6 вариантов (без атаки 0, без атаки 1, JPEG 0, JPEG 1, контраст 0, контраст 1)
        int total_blocks_in_image = block_groups[0].size();
        for (int block_idx = 0; block_idx < total_blocks_in_image; ++block_idx) {
            // Подсчитываем ошибки для каждой схемы (tau может быть от 0 до 6)
            std::vector<int> scheme_errors(2, 0);
            
            for (const auto& group : block_groups) {
                unsigned char extracted_bit = getBitFromBlock(group[block_idx].block, group[block_idx].scheme);
                if (extracted_bit != group[block_idx].bit) {
                    scheme_errors[group[block_idx].scheme]++;
                }
            }
            
            int tau1 = scheme_errors[0];  // ошибки для схемы 0 (от 0 до 6)
            int tau2 = scheme_errors[1];  // ошибки для схемы 1 (от 0 до 6)
            
            // Классификация согласно алгоритму из PDF
            cv::Mat original_block = splitImageInto8x8Blocks(original_img)[block_idx];
            std::string output_path;
            
            if (tau1 < tau_max) {
                // Класс 1
                output_path = "dataset/Dir1/block_" + std::to_string(dir1_count++) + ".png";
            } else if (tau2 <= tau_max && tau_max <= tau1) {
                // Класс 2  
                output_path = "dataset/Dir2/block_" + std::to_string(dir2_count++) + ".png";
            } else {
                // Неопределенные
                output_path = "dataset/Dirrand/block_" + std::to_string(dirrand_count++) + ".png";
            }
            
            cv::imwrite(output_path, original_block);
            
            processed_blocks++;
            double perc = 100.0 * static_cast<double>(processed_blocks) / static_cast<double>(total_blocks_estimate);
            std::cout << "\rBuilding dataset: " << std::fixed << std::setprecision(1) << perc << "%" << std::flush;
        }
    }
    
    std::cout << "\n\nDataset building complete (tau_max = " << tau_max << "):" << std::endl;
    std::cout << "  Dir1 (Scheme 1): " << dir1_count << " blocks" << std::endl;
    std::cout << "  Dir2 (Scheme 2): " << dir2_count << " blocks" << std::endl;
    std::cout << "  Dirrand (Undefined): " << dirrand_count << " blocks" << std::endl;
    
    double total = dir1_count + dir2_count + dirrand_count;
    if (total > 0) {
        std::cout << "\nDistribution:" << std::endl;
        std::cout << "  Dir1: " << std::fixed << std::setprecision(1) << (dir1_count/total*100) << "%" << std::endl;
        std::cout << "  Dir2: " << std::fixed << std::setprecision(1) << (dir2_count/total*100) << "%" << std::endl;
        std::cout << "  Dirrand: " << std::fixed << std::setprecision(1) << (dirrand_count/total*100) << "%" << std::endl;
        
        if (dirrand_count/total > 0.3) {
            std::cout << "\nWarning: High percentage of undefined blocks (>30%). Consider adjusting tau_max." << std::endl;
        }
    }
}