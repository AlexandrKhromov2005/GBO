#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include "../include/launch.h"

int main() {
    int scheme = 0;
    std::string image_path = "images/airplane.png"; 
    std::string watermark_path = "images/watermark.png";
    std::string watermarked_output_path = "images/watermarked_airplane.png";
    std::string extracted_output_path = "images/extracted_watermark.png";
    try {
        launchGBO(image_path, watermark_path, watermarked_output_path, extracted_output_path, scheme);
        std::cout << "GBO launched successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;   
    }

    return 0;
}
