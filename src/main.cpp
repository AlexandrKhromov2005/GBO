#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include "../include/launch.h"
#include "../include/dataset_builder.h"
#include "../include/attacks.h"
#include "../include/metrics.h"
#include "../include/process_images.h"
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

struct MetricResult {
    double ber;
    double psnr;
    double ssim;
    double ncc;
    double mse;
};

struct MetricAgg {
    double minBer, maxBer, sumBer;
    double minPSNR, maxPSNR, sumPSNR;
    double minSSIM, maxSSIM, sumSSIM;
    double minNCC,  maxNCC,  sumNCC;
    double minMSE,  maxMSE,  sumMSE;
    MetricAgg() { reset(); }
    void reset() {
        minBer = minPSNR = minSSIM = minNCC = minMSE = std::numeric_limits<double>::infinity();
        maxBer = maxPSNR = maxSSIM = maxNCC = maxMSE = -std::numeric_limits<double>::infinity();
        sumBer = sumPSNR = sumSSIM = sumNCC = sumMSE = 0.0;
    }
    void update(const MetricResult &m) {
        minBer  = std::min(minBer,  m.ber);   maxBer  = std::max(maxBer,  m.ber);   sumBer  += m.ber;
        minPSNR = std::min(minPSNR, m.psnr);  maxPSNR = std::max(maxPSNR, m.psnr); sumPSNR += m.psnr;
        minSSIM = std::min(minSSIM, m.ssim);  maxSSIM = std::max(maxSSIM, m.ssim); sumSSIM += m.ssim;
        minNCC  = std::min(minNCC,  m.ncc);   maxNCC  = std::max(maxNCC,  m.ncc);  sumNCC  += m.ncc;
        minMSE  = std::min(minMSE,  m.mse);   maxMSE  = std::max(maxMSE,  m.mse);  sumMSE  += m.mse;
    }
    void print(const std::string &title, int runs) const {
        auto avg = [runs](double s){ return s / runs; };
        std::cout << "\n" << title << std::endl;
        std::cout << "BER  min:"  << minBer  << "  avg:" << avg(sumBer)  << "  max:" << maxBer  << std::endl;
        std::cout << "PSNR min:" << minPSNR << "  avg:" << avg(sumPSNR) << "  max:" << maxPSNR << std::endl;
        std::cout << "SSIM min:" << minSSIM << "  avg:" << avg(sumSSIM) << "  max:" << maxSSIM << std::endl;
        std::cout << "NCC  min:" << minNCC  << "  avg:" << avg(sumNCC)  << "  max:" << maxNCC  << std::endl;
        std::cout << "MSE  min:" << minMSE  << "  avg:" << avg(sumMSE)  << "  max:" << maxMSE  << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--build-dataset") {
        buildDataset();
        return 0;
    }

    bool single_block      = false;
    bool debug_embed       = false;
    bool trace_unchanged   = false;
    int trials             = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--single-block")          single_block    = true;
        else if (arg == "--debug-embed")      debug_embed     = true;
        else if (arg == "--trace-unchanged")  trace_unchanged = true;
        else if (arg == "--trials" && i + 1 < argc) {
            trials = std::max(1, std::atoi(argv[++i]));
        }
    }

    int scheme = 0;
    std::string image_path = "images/pepper.png"; 
    std::string watermark_path = "images/watermark.png";

    try {
        if (single_block) {
            launchSingleBlockGBO(image_path, watermark_path, scheme);
        } else if (trials > 1) {
            std::unordered_map<std::string, MetricAgg> agg;
            for (int t = 0; t < trials; ++t) {
                std::string wm_out    = "tmp_watermarked_" + std::to_string(t) + ".png";
                std::string ext_base  = "tmp_extracted_base_" + std::to_string(t) + ".png";

                // 1) Embed watermark
                embedWatermark(image_path, watermark_path, wm_out, scheme, debug_embed, trace_unchanged);

                // 2) Baseline (no attack)
                extractWatermark(wm_out, ext_base, scheme);

                cv::Mat original_image    = cv::imread(image_path, CV_8UC1);
                cv::Mat watermark_image   = cv::imread(watermark_path, CV_8UC1);
                cv::Mat watermarked_image = cv::imread(wm_out, CV_8UC1);
                cv::Mat extracted_base    = cv::imread(ext_base, CV_8UC1);

                if (original_image.empty() || watermark_image.empty() || watermarked_image.empty() || extracted_base.empty()) {
                    throw std::runtime_error("Could not read images for metric calculation in trial " + std::to_string(t));
                }

                MetricResult base;
                base.ber  = computeBER(extract_watermark_bits(watermark_image), extract_watermark_bits(extracted_base));
                base.psnr = computePSNR(original_image, watermarked_image);
                base.ssim = computeSSIM(original_image, watermarked_image);
                base.ncc  = computeNCC(original_image, watermarked_image);
                base.mse  = computeMSE(original_image, watermarked_image);

                agg.try_emplace("NO ATTACK", MetricAgg());
                agg["NO ATTACK"].update(base);

                // ------------ Attacks -------------
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

                for (size_t idx = 0; idx < attacks.size(); ++idx) {
                    const auto &atk = attacks[idx];
                    cv::Mat attacked = atk.func(watermarked_image, atk.param);

                    std::string attack_img_path = "tmp_attacked_" + std::to_string(t) + "_" + std::to_string(idx) + ".png";
                    std::string extracted_wm_path = "tmp_extracted_" + std::to_string(t) + "_" + std::to_string(idx) + ".png";
                    cv::imwrite(attack_img_path, attacked);

                    extractWatermark(attack_img_path, extracted_wm_path, scheme);
                    cv::Mat extracted_wm = cv::imread(extracted_wm_path, CV_8UC1);

                    MetricResult m;
                    m.ber  = computeBER(extract_watermark_bits(watermark_image), extract_watermark_bits(extracted_wm));
                    m.psnr = computePSNR(original_image, attacked);
                    m.ssim = computeSSIM(original_image, attacked);
                    m.ncc  = computeNCC(original_image, attacked);
                    m.mse  = computeMSE(original_image, attacked);

                    agg.try_emplace(atk.name, MetricAgg());
                    agg[atk.name].update(m);

                    // remove temp attack files
                    std::filesystem::remove(attack_img_path);
                    std::filesystem::remove(extracted_wm_path);
                }
            // remove temp baseline files
                std::filesystem::remove(wm_out);
                std::filesystem::remove(ext_base);
            }

            std::cout << "\n====== AGGREGATED RESULTS OVER " << trials << " RUNS ======" << std::endl;
            for (const auto &kv : agg) {
                kv.second.print(kv.first, trials);
            }
        } else {
            launchGBO(image_path, watermark_path, scheme, debug_embed, trace_unchanged);
        }
        std::cout << "GBO process finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;   
    }

    return 0;
}
