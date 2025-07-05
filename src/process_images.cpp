#include "../include/process_images.h"

/**
 * @brief Splits a 512x512 image into 8x8 blocks.
 * 
 * @param image Input image of size 512x512 pixels, type CV_64FC1 (64-bit floating point, single channel).
 * @return std::vector<cv::Mat> A vector containing the 8x8 blocks of the input image.
 * @throws std::runtime_error If the input image is empty, not of size 512x512, or not of type CV_64FC1.
 */
std::vector<cv::Mat> splitImageInto8x8Blocks(const cv::Mat& image) {
    const int BLOCK_SIZE = 8;
    const int IMG_SIZE = 512;
    const int NUM_BLOCKS = IMG_SIZE / BLOCK_SIZE;

    if(image.empty()) {
        throw std::runtime_error("Empty image provided");
    }
    if(image.rows != IMG_SIZE || image.cols != IMG_SIZE) {
        throw std::runtime_error("Image must be 512x512 pixels");
    }
    if(image.type() != CV_64FC1) {
        throw std::runtime_error("Image must be of type CV_64FC1 (64-bit floating point, single channel)");
    }

    std::vector<cv::Mat> blocks;
    blocks.reserve(NUM_BLOCKS * NUM_BLOCKS);

    for(int y = 0; y < IMG_SIZE; y += BLOCK_SIZE) {
        for(int x = 0; x < IMG_SIZE; x += BLOCK_SIZE) {
           cv::Rect roi(x, y, BLOCK_SIZE, BLOCK_SIZE);
            blocks.push_back(image(roi).clone()); 
        }
    }
    
    return blocks;
}

/**
 * @brief Assembles a 512x512 image from 8x8 blocks.
 * 
 * @param blocks A vector containing the 8x8 blocks of the image.
 * @return cv::Mat The reconstructed image of size 512x512 pixels, type CV_64FC1.
 * @throws std::runtime_error If the number of blocks is incorrect or if any block has an invalid size or type.
 */
cv::Mat assembleImageFrom8x8Blocks(const std::vector<cv::Mat>& blocks) {
    const int BLOCK_SIZE = 8;
    const int IMG_SIZE = 512;
    const int NUM_BLOCKS = IMG_SIZE / BLOCK_SIZE;
    const int TOTAL_BLOCKS = NUM_BLOCKS * NUM_BLOCKS;

    if(blocks.size() != TOTAL_BLOCKS) {
        throw std::runtime_error("Incorrect amount of blocks. Need " + 
                           std::to_string(TOTAL_BLOCKS) + ", get " + 
                           std::to_string(blocks.size()));
    }

    cv::Mat image(IMG_SIZE, IMG_SIZE, CV_64FC1, cv::Scalar(0));
    
    int block_index = 0;
    for(int y = 0; y < IMG_SIZE; y += BLOCK_SIZE) {
        for(int x = 0; x < IMG_SIZE; x += BLOCK_SIZE) {
            const cv::Mat& block = blocks[block_index];
            if(block.rows != BLOCK_SIZE || block.cols != BLOCK_SIZE) {
                throw std::runtime_error("Block " + std::to_string(block_index) + 
                                   " has wrong size: " + 
                                   std::to_string(block.rows) + "x" + 
                                   std::to_string(block.cols));
            }
            if(block.type() != CV_64FC1) {
                throw std::runtime_error("Block " + std::to_string(block_index) + 
                                   " has wrong type");
            }
            
            cv::Rect roi(x, y, BLOCK_SIZE, BLOCK_SIZE);
            block.copyTo(image(roi));
            
            block_index++;
        }
    }
    
    return image;
}

/**
 * @brief Extracts watermark bits from a 32x32 grayscale image.
 * 
 * @param image Input image of size 32x32 pixels, type CV_8UC1 (8-bit single channel).
 * @return std::vector<unsigned char> A vector containing the extracted watermark bits.
 * @throws std::runtime_error If the input image is not of size 32x32 or not of type CV_8UC1.
 */
std::vector<unsigned char> extract_watermark_bits(const cv::Mat& image) {
    const int wm_size = 32;
    if (image.rows != wm_size || image.cols != wm_size || image.type() != CV_8UC1) {
        throw std::runtime_error("Input must be 32x32 grayscale image");
    }

    std::vector<unsigned char> bits;
    bits.reserve(wm_size * wm_size);

    for (int i = 0; i < wm_size; ++i) {
        for (int j = 0; j < wm_size; ++j) {
            unsigned char pixel = image.at<uchar>(i, j);
            bits.push_back(pixel > 127 ? 1 : 0); // бинаризация
        }
    }

    return bits;
}

/**
 * @brief Reconstructs a 32x32 grayscale image from a vector of watermark bits.
 * 
 * @param bits A vector containing the watermark bits (0 or 1).
 * @return cv::Mat The reconstructed image of size 32x32 pixels, type CV_8UC1.
 * @throws std::runtime_error If the size of the bit vector is not 1024 (32x32).
 */
cv::Mat reconstruct_watermark_image(const std::vector<unsigned char>& bits) {
    const int wm_size = 32;
    if (bits.size() != wm_size * wm_size) {
        throw std::runtime_error("Bit vector size must be 1024");
    }

    cv::Mat image(wm_size, wm_size, CV_8UC1);

    for (int i = 0; i < wm_size; ++i) {
        for (int j = 0; j < wm_size; ++j) {
            int idx = i * wm_size + j;
            image.at<uchar>(i, j) = bits[idx] ? 255 : 0; // 1 → белый, 0 → чёрный
        }
    }

    return image;
}