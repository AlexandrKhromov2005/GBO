#include <iostream>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include "../include/launch.h"
#include "../include/dataset_builder.h"

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--build-dataset") {
        buildDataset();
        return 0;
    }

    int scheme = 0;
    std::string image_path = "images/pepper.png"; 
    std::string watermark_path = "images/watermark.png";
    try {
        launchGBO(image_path, watermark_path, scheme);
        std::cout << "GBO launched successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;   
    }

    return 0;
}
