#include "../include/launch.h"
#include "../include/attacks.h"
#include <filesystem>

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

// New: run GBO for a single 8x8 block and print fitness evolution
void launchSingleBlockGBO(const std::string& image_path,
                          const std::string& watermark_path,
                          int scheme){
    try {
        cv::Mat image = cv::imread(image_path, CV_8UC1);
        if (image.empty()) {
            throw std::runtime_error("Could not open or find the image: " + image_path);
        }
        cv::Mat watermark = cv::imread(watermark_path, CV_8UC1);
        if (watermark.empty()) {
            throw std::runtime_error("Could not open or find the watermark: " + watermark_path);
        }
        // Use first block (top-left 8x8) as example
        cv::Rect roi(0,0,8,8);
        cv::Mat block = image(roi).clone();

        // Extract first watermark bit as target bit
        std::vector<unsigned char> wm_bits = extract_watermark_bits(watermark);
        unsigned char bit = wm_bits.empty() ? 0 : wm_bits[0];

        std::cout << "Running GBO for single 8x8 block (scheme=" << scheme << ")" << std::endl;
        GBO gbo;
        // Verbose flag prints fitness after every iteration
        cv::Mat new_block = gbo.main_loop(block, 22, bit, scheme, true);
        (void)new_block; // suppress unused warning

    } catch(const std::exception& e){
        std::cerr << "launchSingleBlockGBO error: " << e.what() << std::endl;
    }
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

    // ------------------ BASELINE (без атаки) ------------------
    std::string extracted_noattack_path = "images/extracted_no_attack.png";
    try {
        extractWatermark(watermarked_output_path, extracted_noattack_path, scheme);
    } catch (const std::exception &e) {
        std::cerr << "Error extracting baseline watermark: " << e.what() << std::endl;
    }

    cv::Mat original_image = cv::imread(image_path, CV_8UC1);
    cv::Mat watermark_image = cv::imread(watermark_path, CV_8UC1);
    cv::Mat watermarked_image = cv::imread(watermarked_output_path, CV_8UC1);
    cv::Mat extracted_noattack = cv::imread(extracted_noattack_path, CV_8UC1);

    if (watermark_image.empty() || extracted_noattack.empty() || original_image.empty() || watermarked_image.empty()) {
        std::cerr << "Error: images could not be read for baseline metric calculation." << std::endl;
        return;
    }

    std::cout << "\n========== METRICS ==========" << std::endl;
    auto printMetrics = [](const std::string &title, double ber, double psnr, double ssim, double ncc, double mse) {
        std::cout << "\n" << title << std::endl;
        std::cout << "BER : " << ber << std::endl;
        std::cout << "PSNR: " << psnr << std::endl;
        std::cout << "SSIM: " << ssim << std::endl;
        std::cout << "NCC : " << ncc << std::endl;
        std::cout << "MSE : " << mse << std::endl;
    };

    double ber_base  = computeBER(extract_watermark_bits(watermark_image), extract_watermark_bits(extracted_noattack));
    double psnr_base = computePSNR(original_image, watermarked_image);
    double ssim_base = computeSSIM(original_image, watermarked_image);
    double ncc_base  = computeNCC(original_image, watermarked_image);
    double mse_base  = computeMSE(original_image, watermarked_image);
    printMetrics("NO ATTACK", ber_base, psnr_base, ssim_base, ncc_base, mse_base);

    // ------------------ СПИСОК АТАК ------------------
    cv::Mat wm_img_gray = watermarked_image; // already CV_8UC1

    struct AttackInfo {
        std::string name;
        cv::Mat (*func)(const cv::Mat &, double);
        double param;
    };

    std::vector<AttackInfo> attacks = {
        {"Brightness +30", [](const cv::Mat &img, double v){return brightnessIncrease(img, static_cast<int>(v));}, 30},
        {"Brightness -30", [](const cv::Mat &img, double v){return brightnessDecrease(img, static_cast<int>(v));}, 30},
        {"Contrast *1.2",   [](const cv::Mat &img, double v){return contrastIncrease(img, v);}, 1.2},
        {"Contrast *0.8",   [](const cv::Mat &img, double v){return contrastDecrease(img, v);}, 0.8},
        {"Salt&Pepper 5%", [](const cv::Mat &img, double v){return saltPepperNoise(img, v);}, 0.05},
        {"Speckle 20",      [](const cv::Mat &img, double v){return speckleNoise(img, v);}, 20},
        {"Histogram Eq",    [](const cv::Mat &img, double){return histogramEqualization(img);}, 0},
        {"Sharpen",         [](const cv::Mat &img, double){return sharpening(img);}, 0},
        {"JPEG q=70",       [](const cv::Mat &img, double v){return jpegCompression(img, static_cast<int>(v));}, 70},
        {"JPEG q=80",       [](const cv::Mat &img, double v){return jpegCompression(img, static_cast<int>(v));}, 80},
        {"JPEG q=90",       [](const cv::Mat &img, double v){return jpegCompression(img, static_cast<int>(v));}, 90},
        {"Gaussian k=3",    [](const cv::Mat &img, double v){return gaussianFiltering(img, static_cast<int>(v));}, 3},
        {"Median k=3",      [](const cv::Mat &img, double v){return medianFiltering(img, static_cast<int>(v));}, 3},
        {"Average k=3",     [](const cv::Mat &img, double v){return averageFiltering(img, static_cast<int>(v));}, 3}
    };

    auto sanitize = [](std::string s){
        for (char &c : s) {
            if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
        }
        return s;
    };

    namespace fs = std::filesystem;
    fs::path img_path(image_path);
    std::string stem = img_path.stem().string();

    for (const auto &atk : attacks) {
        cv::Mat attacked = atk.func(wm_img_gray, atk.param);
        std::string attack_name_sanitized = sanitize(atk.name);
        std::string tmp_attack_path = (img_path.parent_path() / (stem + "_" + attack_name_sanitized + ".png")).string();
        std::string tmp_extr_path   = (img_path.parent_path() / ("extracted_watermark_" + stem + "_" + attack_name_sanitized + ".png")).string();
        cv::imwrite(tmp_attack_path, attacked);

        try {
            extractWatermark(tmp_attack_path, tmp_extr_path, scheme);
        } catch (const std::exception &e) {
            std::cerr << "Error extracting watermark after " << atk.name << ": " << e.what() << std::endl;
            continue;
        }

        cv::Mat extracted_wm = cv::imread(tmp_extr_path, CV_8UC1);
        if (extracted_wm.empty()) {
            std::cerr << "Could not read extracted watermark for " << atk.name << std::endl;
            continue;
        }

        double ber  = computeBER(extract_watermark_bits(watermark_image), extract_watermark_bits(extracted_wm));
        double psnr = computePSNR(original_image, attacked);
        double ssim = computeSSIM(original_image, attacked);
        double ncc  = computeNCC(original_image, attacked);
        double mse  = computeMSE(original_image, attacked);
        printMetrics(atk.name, ber, psnr, ssim, ncc, mse);

    }

}