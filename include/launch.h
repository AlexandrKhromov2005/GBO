#pragma once
#include "gbo.h"
#include "process_images.h"
#include <string>
#include <opencv2/opencv.hpp>
#include "metrics.h"

void embedWatermark(std::string image_path, std::string watermark_path, std::string output_path, int scheme);
void extractWatermark(std::string watermarked_image_path, std::string extracted_watermark_path, int scheme);
// Original variant with explicit paths
void launchGBO(const std::string& image_path,
              const std::string& watermark_path,
              const std::string& watermarked_output_path,
              const std::string& extracted_output_path,
              int scheme = 0);

// Simplified variant: paths will be auto-generated inside the function
void launchGBO(const std::string& image_path,
              const std::string& watermark_path,
              int scheme = 0);