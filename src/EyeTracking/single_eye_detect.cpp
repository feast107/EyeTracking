#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


// 添加 GradientIntersect 类
class GradientIntersect {
private:
    // 创建网格向量场
    cv::Mat createGrid(int rows, int cols) {
        cv::Mat grid(2 * rows - 1, 2 * cols - 1, CV_32FC2);

        for (int y = 1 - rows; y < rows; y++) {
            for (int x = 1 - cols; x < cols; x++) {
                float norm = std::sqrt(x * x + y * y);
                if (norm == 0) norm = 1;
                int grid_y = y + rows - 1;
                int grid_x = x + cols - 1;
                grid.at<cv::Vec2f>(grid_y, grid_x) = cv::Vec2f(y / norm, x / norm);
            }
        }
        return grid;
    }

    // 计算归一化梯度
    cv::Mat createGradient(const cv::Mat& image) {
        cv::Mat grad_x, grad_y;
        cv::Sobel(image, grad_x, CV_32F, 1, 0);
        cv::Sobel(image, grad_y, CV_32F, 0, 1);

        cv::Mat gradient(image.size(), CV_32FC2);
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                float gx = grad_x.at<float>(y, x);
                float gy = grad_y.at<float>(y, x);
                float norm = std::sqrt(gx * gx + gy * gy);
                if (norm == 0) norm = 1;
                gradient.at<cv::Vec2f>(y, x) = cv::Vec2f(gy / norm, gx / norm);
            }
        }
        return gradient;
    }

public:
    cv::Point locate(const cv::Mat& image, double sigma = 2, int accuracy = 1) {
        cv::Mat float_image;
        image.convertTo(float_image, CV_32F);
        cv::normalize(float_image, float_image, 0, 1, cv::NORM_MINMAX);

        cv::Mat blurred;
        cv::GaussianBlur(float_image, blurred, cv::Size(0, 0), sigma);

        // 限制搜索范围到中心区域
        int border = 5;  // 边界像素
        int start_y = border;
        int end_y = image.rows - border;
        int start_x = border;
        int end_x = image.cols - border;

        cv::Mat grid = createGrid(image.rows, image.cols);
        cv::Mat gradient = createGradient(float_image);

        cv::Mat scores = cv::Mat::zeros(image.size(), CV_32F);

        // 使用OpenMP并行计算
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int cy = start_y; cy < end_y; cy += accuracy) {
            for (int cx = start_x; cx < end_x; cx += accuracy) {
                float score = 0;
                float blur_val = blurred.at<float>(cy, cx);

                int window_size = 10;
                int start_wy = std::max(0, cy - window_size);
                int end_wy = std::min(image.rows, cy + window_size);
                int start_wx = std::max(0, cx - window_size);
                int end_wx = std::min(image.cols, cx + window_size);

                for (int y = start_wy; y < end_wy; y++) {
                    for (int x = start_wx; x < end_wx; x++) {
                        cv::Vec2f disp = grid.at<cv::Vec2f>(
                            image.rows - cy - 1 + y,
                            image.cols - cx - 1 + x
                        );
                        cv::Vec2f grad = gradient.at<cv::Vec2f>(y, x);
                        float dot = disp[0] * grad[0] + disp[1] * grad[1];
                        score += dot * dot;
                    }
                }
                scores.at<float>(cy, cx) = score * (1 - blur_val);
            }
        }

        cv::Point maxLoc;
        cv::minMaxLoc(scores, nullptr, nullptr, nullptr, &maxLoc);
        return maxLoc;
    }

    cv::Mat getGradientImage(const cv::Mat& image) {
        cv::Mat float_image;
        image.convertTo(float_image, CV_32F);
        cv::normalize(float_image, float_image, 0, 1, cv::NORM_MINMAX);

        cv::Mat gradient = createGradient(float_image);

        cv::Mat gradient_vis(image.size(), CV_8UC1);
        for (int y = 0; y < image.rows; y += 2) {
            for (int x = 0; x < image.cols; x += 2) {
                cv::Vec2f grad = gradient.at<cv::Vec2f>(y, x);
                float magnitude = std::sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
                gradient_vis.at<uchar>(y, x) = cv::saturate_cast<uchar>(magnitude * 255);
            }
        }
        return gradient_vis;
    }
};

int main() {
    // 读取单个眼睛区域图像
    cv::Mat eye_image = cv::imread("21.png", cv::IMREAD_GRAYSCALE);
    if (eye_image.empty()) {
        std::cerr << "Error: Unable to load eye image!" << std::endl;
        return -1;
    }
    std::cout << "Eye image loaded successfully. Size: " << eye_image.size() << std::endl;

    try {
        // 创建瞳孔检测器
        GradientIntersect detector;

        // 定位瞳孔中心
        cv::Point pupil = detector.locate(eye_image);
        cv::Mat gradient = detector.getGradientImage(eye_image);

        // 在图像上标记结果
        cv::Mat result = eye_image.clone();
        cv::circle(result, pupil, 3, cv::Scalar(255), -1);
        cv::circle(result, pupil, 10, cv::Scalar(255), 1);

        // 显示结果
        cv::namedWindow("Original Eye", cv::WINDOW_NORMAL);
        cv::namedWindow("Result", cv::WINDOW_NORMAL);
        cv::namedWindow("Gradient", cv::WINDOW_NORMAL);

        cv::resizeWindow("Original Eye", 400, 300);
        cv::resizeWindow("Result", 400, 300);
        cv::resizeWindow("Gradient", 400, 300);

        cv::imshow("Original Eye", eye_image);
        cv::imshow("Result", result);
        cv::imshow("Gradient", gradient);

        // 输出瞳孔坐标
        std::cout << "Pupil position: (" << pupil.x << ", " << pupil.y << ")" << std::endl;
        std::cout << "Press any key to exit..." << std::endl;

        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}