#ifndef __EDGES_SUBPIX_H__
#define __EDGES_SUBPIX_H__
#include <opencv2/opencv.hpp>

struct EdgePoints {
    std::vector<cv::Point2f> points;
    std::vector<float> direction;
    std::vector<float> response;
};

// only 8-bit images are supported
CV_EXPORTS void EdgesSubPix(cv::Mat& gray, double alpha, int low, int high,
    int blocksize, EdgePoints& edge_points);

// only 8-bit images are supported
CV_EXPORTS void ContoursSubPix(cv::Mat& gray, cv::Mat& binary, double alpha,
    int low, int high, int blocksize, std::vector<EdgePoints>& contours);

CV_EXPORTS void DrawEdges(cv::Mat& rgb, cv::Mat& gray,
    const EdgePoints& edge_points, const cv::Scalar& color,
    const int scaleFactor = 2);

CV_EXPORTS void DrawContours(cv::Mat& rgb, cv::Mat& gray,
    const std::vector<EdgePoints>& contours, const cv::Scalar& color,
    const int scaleFactor);

#endif // __EDGES_SUBPIX_H__
