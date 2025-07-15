#include "../include/metrics.h"

double computeBER(const std::vector<unsigned char>& wm1, const std::vector<unsigned char>& wm2) {
    if (wm1.size() != wm2.size())
        throw std::invalid_argument("Watermark vectors must be the same size");

    int errors = 0;
    for (size_t i = 0; i < wm1.size(); ++i) {
        if (wm1[i] != wm2[i])
            ++errors;
    }

    return static_cast<double>(errors) / wm1.size();
}

double computePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    double mse = computeMSE(img1, img2);
    if (mse <= 0) {
        return 0;
    }

    double max_pixel_value = 255.0;
    double psnr = 10.0 * log10((max_pixel_value * max_pixel_value) / mse);
    return psnr;
}

double computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    CV_Assert(img1.size() == img2.size() && img1.type() == CV_8UC1 && img2.type() == CV_8UC1);

    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat img1f, img2f;
    img1.convertTo(img1f, CV_32F);
    img2.convertTo(img2f, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1f, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2f, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(img1f.mul(img1f), sigma1_sq, cv::Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;

    cv::GaussianBlur(img2f.mul(img2f), sigma2_sq, cv::Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;

    cv::GaussianBlur(img1f.mul(img2f), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat t3 = t1.mul(t2);

    t1 = mu1_sq + mu2_sq + C1;
    t2 = sigma1_sq + sigma2_sq + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    return cv::mean(ssim_map)[0];
}

double computeNCC(const cv::Mat& img1, const cv::Mat& img2) {
    CV_Assert(img1.size() == img2.size() && img1.type() == CV_8UC1 && img2.type() == CV_8UC1);

    cv::Mat img1f, img2f;
    img1.convertTo(img1f, CV_32F);
    img2.convertTo(img2f, CV_32F);

    cv::Mat img1_zero_mean = img1f - cv::mean(img1f)[0];
    cv::Mat img2_zero_mean = img2f - cv::mean(img2f)[0];

    double numerator = cv::sum(img1_zero_mean.mul(img2_zero_mean))[0];
    double denominator = std::sqrt(cv::sum(img1_zero_mean.mul(img1_zero_mean))[0] *
                                   cv::sum(img2_zero_mean.mul(img2_zero_mean))[0]);

    if (denominator == 0) return 0.0;
    return numerator / denominator;
}

double computeMSE(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cerr << "Error: Images must have the same size and type." << std::endl;
        return -1;
    }

    cv::Mat diff;
    cv::Mat img1f, img2f;
    img1.convertTo(img1f, CV_32F);
    img2.convertTo(img2f, CV_32F);
    cv::absdiff(img1f, img2f, diff);
    diff = diff.mul(diff);

    cv::Scalar mse_scalar = cv::mean(diff);
    int channels = img1.channels();
    double mse;
    if (channels == 1) {
        mse = mse_scalar[0];
    } else {
        // Assume 3-channel RGB
        mse = (mse_scalar[0] + mse_scalar[1] + mse_scalar[2]) / 3.0;
    }

    return mse;
}