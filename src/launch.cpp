#include "../include/launch.h"

void embedWatermark(std::string image_path, std::string watermark_path, std::string output_path, int scheme) {
    cv::Mat image = cv::imread(image_path, CV_8UC1);
    if (image.empty()) {
        throw std::runtime_error("Could not open or find the image: " + image_path);
    }

    cv::Mat watermark = cv::imread(watermark_path, CV_8UC1);
    if (watermark.empty()) {
        throw std::runtime_error("Could not open or find the watermark: " + watermark_path);
    }

    std::vector<cv::Mat> blocks = splitImageInto8x8Blocks(image);
    std::vector<cv::Mat> new_blocks;

    std::vector<unsigned char> watermark_bits = extract_watermark_bits(watermark);

    for (size_t i = 0; i < blocks.size(); ++i) {
        GBO gbo;
        cv::Mat new_block = gbo.main_loop(blocks[i], 22, watermark_bits[i % watermark_bits.size()], scheme);
        new_blocks.push_back(new_block);
    }

    cv::Mat result_image = assembleImageFrom8x8Blocks(new_blocks);
    result_image.convertTo(result_image, CV_8UC1);
    cv::imwrite(output_path, result_image);
}

void extractWatermark(std::string watermarked_image_path, std::string extracted_watermark_path, int scheme) {
    cv::Mat watermarked_image = cv::imread(watermarked_image_path, CV_8UC1);
    if (watermarked_image.empty()) {
        throw std::runtime_error("Could not open or find the watermarked image: " + watermarked_image_path);
    }
    watermarked_image.convertTo(watermarked_image, CV_8UC1);

    std::vector<cv::Mat> blocks = splitImageInto8x8Blocks(watermarked_image);
    std::vector<unsigned char> extracted_bits(1024, 0);
    for (size_t i = 0; i < blocks.size(); ++i) {
        unsigned char bit = getBitFromBlock(blocks[i], scheme);
        extracted_bits[i % 1024] += bit;
    }

    for (auto& bit : extracted_bits) {
        if (bit >= 3) { 
            bit = 1;
        } else if (bit <= 1) {
            bit = 0;
        } else if (bit == 2) {
            bit = uniform_random_0_1() < 0.5 ? 0 : 1; 
        }
    }

    cv::Mat extracted_watermark = reconstruct_watermark_image(extracted_bits);
    cv::imwrite(extracted_watermark_path, extracted_watermark);
}


void launchGBO(const std::string& image_path,
               const std::string& watermark_path,
               const std::string& watermarked_output_path,
               const std::string& extracted_output_path,
               int scheme){

    try {
        embedWatermark(image_path, watermark_path, watermarked_output_path, scheme);
        std::cout << "Watermark embedded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error embedding watermark: " << e.what() << std::endl;
    }

    try {
        extractWatermark(watermarked_output_path, extracted_output_path, scheme);
        std::cout << "Watermark extracted successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error extracting watermark: " << e.what() << std::endl;
    }

    //Calculation of metrics
    cv::Mat original_image = cv::imread(image_path, CV_8UC1);
    cv::Mat watermark_image = cv::imread(watermark_path, CV_8UC1);
    cv::Mat watermarked_image = cv::imread(watermarked_output_path, CV_8UC1);
    cv::Mat extracted_watermark = cv::imread(extracted_output_path, CV_8UC1);

    if (watermark_image.empty() || extracted_watermark.empty() || original_image.empty() || watermarked_image.empty()) {
        std::cerr << "Error: One or more images could not be read for metrics calculation." << std::endl;
        return;
    }

    double ber = computeBER(extract_watermark_bits(watermark_image), extract_watermark_bits(extracted_watermark));
    double psnr = computePSNR(original_image, watermarked_image);
    double ssim = computeSSIM(original_image, watermarked_image);
    double ncc = computeNCC(original_image, watermarked_image);
    double mse = computeMSE(original_image, watermarked_image);
    std::cout << "Metrics:" << std::endl;
    std::cout << "BER: " << ber << std::endl;
    std::cout << "PSNR: " << psnr << std::endl;
    std::cout << "SSIM: " << ssim << std::endl;
    std::cout << "NCC: " << ncc << std::endl;
    std::cout << "MSE: " << mse << std::endl;

}