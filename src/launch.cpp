#include "../include/launch.h"
#include "../include/attacks.h"
#include <filesystem>
#include "../include/process_block.h"
#include <iomanip>
#include <algorithm>

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

    // Debug/tracing modes have been removed for production build.
    const bool debug = false;
    const bool traceUnchanged = false;
    const bool traceFailed = false;

    for (size_t i = 0; i < blocks.size(); ++i) {
        GBO gbo;
        unsigned char target_bit = watermark_bits[i % watermark_bits.size()];
        cv::Mat new_block = gbo.main_loop(blocks[i], embeding_region[scheme].size(), target_bit, scheme);
        // Debug: verify embedding success
        if (debug || traceUnchanged || traceFailed) {
            unsigned char extracted_bit = getBitFromBlock(new_block, scheme);
            bool identical = cv::countNonZero(blocks[i] != new_block) == 0;
            if (identical && traceUnchanged) {
                // Block was not modified at all
                cv::Mat floatOld;
                blocks[i].convertTo(floatOld, CV_64FC1);
                cv::Mat dctOld;
                cv::dct(floatOld, dctOld);
                double s1_old = getRegionSum(dctOld, s1_region[scheme]);
                double s0_old = getRegionSum(dctOld, s0_region[scheme]);

                // For consistency compute PSNR (will be high if identical)
                double psnr_val = compute_psnr(blocks[i], new_block);

                std::cout << "[DEBUG] Unchanged block " << i
                          << " bit=" << static_cast<int>(target_bit)
                          << " s1=" << s1_old << " s0=" << s0_old
                          << " psnr=" << psnr_val << std::endl;

                // Additionally run GBO with verbose to trace fitness evolution
                std::cout << "[DEBUG] Fitness evolution for block " << i << std::endl;
                GBO gbo_trace;
                cv::Mat traced_block = gbo_trace.main_loop(blocks[i], embeding_region[scheme].size(), target_bit, scheme, true);

                // Compute region sums after GBO attempt
                cv::Mat tracedFloat;
                traced_block.convertTo(tracedFloat, CV_64FC1);
                cv::Mat dctTraced;
                cv::dct(tracedFloat, dctTraced);
                double s1_traced = getRegionSum(dctTraced, s1_region[scheme]);
                double s0_traced = getRegionSum(dctTraced, s0_region[scheme]);

                double psnr_after = compute_psnr(blocks[i], traced_block);
                std::cout << "[DEBUG] After GBO: new_s1=" << s1_traced << " new_s0=" << s0_traced
                          << " psnr_after=" << psnr_after << std::endl;

                std::cout << "[DEBUG] Trace complete. Exiting." << std::endl;
                std::exit(0);
            }
            if (traceFailed && extracted_bit != target_bit) {
                std::cout << "[TRACE_FAILED] Block " << i << " target=" << static_cast<int>(target_bit)
                          << " extracted=" << static_cast<int>(extracted_bit) << std::endl;
                std::cout << "[TRACE_FAILED] Per-iteration metrics:" << std::endl;
                GBO gbo_trace;
                cv::Mat traced_block = gbo_trace.main_loop(blocks[i], embeding_region[scheme].size(), target_bit, scheme, true);

                // Print DCT coefficients before and after
                auto printDCT = [&](const cv::Mat &mat, const std::string &label){
                    std::cout << label << std::endl;
                    for (int r = 0; r < 8; ++r) {
                        for (int c = 0; c < 8; ++c) {
                            // Determine zigzag index for (r,c)
                            int linear = r * 8 + c;
                            int zz_idx = 0;
                            for (; zz_idx < 64; ++zz_idx) {
                                if (jpeg_zigzag[zz_idx] == linear) break;
                            }
                            bool isEmbed = std::find(embeding_region[scheme].begin(), embeding_region[scheme].end(), zz_idx) != embeding_region[scheme].end();
                            if (isEmbed) std::cout << "\033[32m"; // green
                            std::cout << std::setw(9) << std::fixed << std::setprecision(2) << mat.at<double>(r, c);
                            if (isEmbed) std::cout << "\033[0m";
                        }
                        std::cout << std::endl;
                    }
                };

                cv::Mat origF, dctOrig;
                blocks[i].convertTo(origF, CV_64FC1);
                cv::dct(origF, dctOrig);

                cv::Mat tracedF, dctTraced;
                traced_block.convertTo(tracedF, CV_64FC1);
                cv::dct(tracedF, dctTraced);

                printDCT(dctOrig,  "[TRACE_FAILED] DCT ORIGINAL:");
                printDCT(dctTraced,"[TRACE_FAILED] DCT MODIFIED:");

                std::cout << "[TRACE_FAILED] Trace complete. Exiting." << std::endl;
                std::exit(0);
            }
            if (!identical && debug && extracted_bit != target_bit) {
                // Compute old region sums
                cv::Mat oldFloat;
                blocks[i].convertTo(oldFloat, CV_64FC1);
                cv::Mat dctOld;
                cv::dct(oldFloat, dctOld);
                double s1_old = getRegionSum(dctOld, s1_region[scheme]);
                double s0_old = getRegionSum(dctOld, s0_region[scheme]);

                // Compute new region sums
                cv::Mat newFloat;
                new_block.convertTo(newFloat, CV_64FC1);
                cv::Mat dctNew;
                cv::dct(newFloat, dctNew);
                double s1_new = getRegionSum(dctNew, s1_region[scheme]);
                double s0_new = getRegionSum(dctNew, s0_region[scheme]);

                // Compute PSNR between original and modified block
                double psnr_val = compute_psnr(blocks[i], new_block);

                std::cout << "[DEBUG] Mismatch at block " << i
                          << " target=" << static_cast<int>(target_bit)
                          << " extracted=" << static_cast<int>(extracted_bit)
                          << " old_s1=" << s1_old << " old_s0=" << s0_old
                          << " new_s1=" << s1_new << " new_s0=" << s0_new
                          << " psnr=" << psnr_val << std::endl;
            }
        }
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
        cv::Mat new_block = gbo.main_loop(block, embeding_region[scheme].size(), bit, scheme, true);
        // Print DCT coefficients before and after with embedding region highlighted
        auto printDCT = [&](const cv::Mat &mat, const std::string &label){
            std::cout << label << std::endl;
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    int linear = r * 8 + c;
                    int zz_idx = 0;
                    for (; zz_idx < 64; ++zz_idx) {
                        if (jpeg_zigzag[zz_idx] == linear) break;
                    }
                    bool isEmbed = std::find(embeding_region[scheme].begin(), embeding_region[scheme].end(), zz_idx) != embeding_region[scheme].end();
                    if (isEmbed) std::cout << "\033[32m"; // green
                    std::cout << std::setw(9) << std::fixed << std::setprecision(2) << mat.at<double>(r, c);
                    if (isEmbed) std::cout << "\033[0m";
                }
                std::cout << std::endl;
            }
        };

        cv::Mat origF, dctOrig;
        block.convertTo(origF, CV_64FC1);
        cv::dct(origF, dctOrig);

        cv::Mat newF, dctNew;
        new_block.convertTo(newF, CV_64FC1);
        cv::dct(newF, dctNew);

        printDCT(dctOrig,  "[SINGLE_BLOCK] DCT ORIGINAL:");
        printDCT(dctNew,   "[SINGLE_BLOCK] DCT MODIFIED:");

        double psnr_val = compute_psnr(block, new_block);
        std::cout << "[SINGLE_BLOCK] PSNR=" << psnr_val << std::endl;

    } catch(const std::exception& e){
        std::cerr << "launchSingleBlockGBO error: " << e.what() << std::endl;
    }
}

void launchGBO(const std::string& image_path,
               const std::string& watermark_path,
               const std::string& watermarked_output_path,
               const std::string& extracted_output_path,
               int scheme) {

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

// Simplified overload: auto-generate temp paths and invoke main variant
void launchGBO(const std::string& image_path,
               const std::string& watermark_path,
               int scheme) {
    std::string tmp_wm = "tmp_wm_single.png";
    std::string tmp_extract = "tmp_extract_single.png";
    launchGBO(image_path, watermark_path, tmp_wm, tmp_extract, scheme);
    std::filesystem::remove(tmp_wm);
    std::filesystem::remove(tmp_extract);
}