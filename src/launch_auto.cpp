#include "../include/launch.h"
#include <filesystem>

// Auto-generated path variant of launchGBO
void launchGBO(const std::string& image_path,
               const std::string& watermark_path,
               int scheme) {
    namespace fs = std::filesystem;
    fs::path img_path(image_path);
    std::string stem = img_path.stem().string();

    // Output paths reside in same directory as source image
    std::string watermarked_output_path = (img_path.parent_path() / ("watermarked_" + stem + img_path.extension().string())).string();
    std::string extracted_output_path  = (img_path.parent_path() / ("extracted_watermark_" + stem + "_no_attack.png")).string();

    launchGBO(image_path, watermark_path, watermarked_output_path, extracted_output_path, scheme);
}
