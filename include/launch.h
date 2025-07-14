#pragma once
#include "gbo.h"
#include "process_images.h"
#include <string>
#include <opencv2/opencv.hpp>
#include "metrics.h"

void embedWatermark(std::string image_path, std::string watermark_path, std::string output_path, int scheme, bool debug = false, bool traceUnchanged = false);
void extractWatermark(std::string watermarked_image_path, std::string extracted_watermark_path, int scheme);
// Run GBO for a single 8x8 block and print fitness value changes
void launchSingleBlockGBO(const std::string& image_path,
                          const std::string& watermark_path,
                          int scheme = 0);
// Original variant with explicit paths
void launchGBO(const std::string& image_path,
              const std::string& watermark_path,
              const std::string& watermarked_output_path,
              const std::string& extracted_output_path,
              int scheme = 0,
              bool debug = false,
              bool traceUnchanged = false);

// Simplified variant: paths will be auto-generated inside the function
void launchGBO(const std::string& image_path,
              const std::string& watermark_path,
              int scheme = 0,
              bool debug = false,
              bool traceUnchanged = false);