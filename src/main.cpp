#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include "../include/launch.h"

// Проверка Armadillo и OpenCV
int main() {
    std::string image_path = "images/pepper.png"; 
    std::string watermark_path = "images/watermark.png";
    std::string watermarked_output_path = "images/watermarked_pepper.png";
    std::string extracted_output_path = "images/extracted_watermark.png";
    try {
        launchGBO(image_path, watermark_path, watermarked_output_path, extracted_output_path);
        std::cout << "GBO launched successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;   
    }

    return 0;
}
