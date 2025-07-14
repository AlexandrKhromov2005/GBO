#include "../include/process_block.h"


arma::vec matToZigzag(const cv::Mat& block) {
    CV_Assert(!block.empty() && block.rows == 8 && block.cols == 8 && block.type() == CV_64FC1);

    arma::vec zz = arma::zeros<arma::vec>(64);
    for (int k = 0; k < 64; ++k) {
        int i = jpeg_zigzag[k] / 8;
        int j = jpeg_zigzag[k] % 8;
        zz(k) = block.at<double>(i, j);
    }
    return zz;
}

cv::Mat zigzagToMat(const arma::vec& zz) {
    CV_Assert(zz.n_elem == 64);

    cv::Mat block(8, 8, CV_64FC1, cv::Scalar(0));
    for (int k = 0; k < 64; ++k) {
        int i = jpeg_zigzag[k] / 8;
        int j = jpeg_zigzag[k] % 8;
        block.at<double>(i, j) = zz(k);
    }
    return block;
}

double getRegionSum(const cv::Mat& block, std::vector<int> region) {
    arma::vec zzBlock = matToZigzag(block);
    double sum = 0.0;
    for (int i : region) {
        sum += std::fabs(zzBlock(i));
    }
    return sum > 0.001 ? sum : 0.001; // Avoid division by zero
}

double compute_psnr(const cv::Mat& orig, const cv::Mat& test) {
    cv::Mat diff;
    cv::absdiff(orig, test, diff);
    diff = diff.mul(diff); 

    double mse = cv::sum(diff)[0] / 64.0;
    if (mse == 0.0) {
        return 100.0; 
    }

    const double MAX_I = 255.0;
    return 10.0 * std::log10((MAX_I * MAX_I) / mse);
}

/**
 * @brief Applies a vector to an 8x8 block using DCT and zigzag transformation.
 * @param vec Input vector of size 22, containing the values to be applied to the block.
 * @param block Input OpenCV block of size 8x8, type CV_8UC1.
 * @return cv::Mat The modified block after applying the vector, of size 8x8, type CV_8UC1.
 */
cv::Mat applyVectorToBlock(const arma::vec& vec, const cv::Mat& block, int scheme) {
    if (block.empty()) {
        throw std::invalid_argument("applyVectorToBlock: empty block");
    }

    if (block.rows != 8 || block.cols != 8) {
        throw std::invalid_argument("applyVectorToBlock: block must be 8x8");
    }
    if (block.type() != CV_8UC1) {
        throw std::invalid_argument("applyVectorToBlock: block must be CV_8UC1");
    }
    cv::Mat floatBlock;
    block.convertTo(floatBlock, CV_64FC1); 
    cv::Mat dctBlock;
    cv::dct(floatBlock, dctBlock);
    arma::vec zzBlock = matToZigzag(dctBlock);

    for (int idx = 0; idx < embeding_region[scheme].size(); ++idx) {
        int coeff = embeding_region[scheme][idx];
        double sign = (zzBlock(coeff) >= 0.0) ? 1.0 : -1.0;
        zzBlock(coeff) = sign * std::fabs(std::fabs(zzBlock(coeff)) + vec(idx));
    }

    cv::Mat modifiedBlock = zigzagToMat(zzBlock);
    cv::dct(modifiedBlock, modifiedBlock, cv::DCT_INVERSE);
    cv::Mat modifiedBlock8U;
    modifiedBlock.convertTo(modifiedBlock8U, CV_8UC1);
    return modifiedBlock8U;
}

/**
 * @brief Extracts a bit from an 8x8 block based on the sum of specific regions in the zigzag transformed block.
 * @param block Input OpenCV block of size 8x8, type CV_8UC1.
 * @return unsigned char The extracted bit, either 0 or 1, based on the comparison of sums from two regions.
 */
unsigned char getBitFromBlock(const cv::Mat& block, int scheme){
    cv::Mat dctBlock, floatBlock;
    block.convertTo(floatBlock, CV_64FC1);
    cv::dct(floatBlock, dctBlock);
    double s1 = getRegionSum(dctBlock, s1_region[scheme]);
    double s0 = getRegionSum(dctBlock, s0_region[scheme]);
    return (s1 >= s0) ? 1 : 0;
}

/**
 * @brief Calculates the fitness value for a given block and vector, based on PSNR and region sums.
 * @param block Input OpenCV block of size 8x8, type CV_8UC1.
 * @param vec Input vector of size 22, containing the values to be applied to the block.
 * @param bit The bit to be used in the fitness calculation (0 or 1).
 * @return double The calculated fitness value.
 */
double calcFitnessValue(const cv::Mat& block, const arma::vec& vec, unsigned char bit, int scheme) {
    if (block.empty()) {
        throw std::invalid_argument("calcFitnessValue: empty block");
    }

    if (block.rows != 8 || block.cols != 8) {
        throw std::invalid_argument("calcFitnessValue: block must be 8x8");
    }
    if (block.type() != CV_8UC1) {
        throw std::invalid_argument("calcFitnessValue: block must be CV_8UC1");
    }

    cv::Mat modifiedBlock = applyVectorToBlock(vec, block, scheme);
    cv::Mat modifiedFloatBlock;
    modifiedBlock.convertTo(modifiedFloatBlock, CV_64FC1);
    double psnr = compute_psnr(block, modifiedBlock);
    cv::Mat modifiedBlockDCT;
    cv::dct(modifiedFloatBlock, modifiedBlockDCT);
    double s1 = getRegionSum(modifiedBlockDCT, s1_region[scheme]);
    double s0 = getRegionSum(modifiedBlockDCT, s0_region[scheme]);
    return (bit == 0 ? s1 / s0 : s0 / s1) - 0.01 * psnr;
}