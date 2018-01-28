#ifndef __EDGES_SUBPIX_H__
#define __EDGES_SUBPIX_H__
#include <opencv2/opencv.hpp>
#include <vector>

struct Contour
{
    std::vector<cv::Point2f> points;
    std::vector<float> direction;  
    std::vector<float> response;
};
// only 8-bit
CV_EXPORTS void EdgesSubPix(cv::Mat &gray, cv::Mat &binary, double alpha, int low, int high,
                            std::vector<Contour> &contours, cv::OutputArray hierarchy,
                            int mode);

CV_EXPORTS void EdgesSubPix(cv::Mat &gray, cv::Mat &binary, double alpha, int low, int high,
                           std::vector<Contour> &contours);

CV_EXPORTS void DrawContours(cv::Mat &rgb, cv::Mat &gray,
                             const std::vector<Contour> &contours, const cv::Scalar &color,
                             const int scaleFactor = 2);

#endif // __EDGES_SUBPIX_H__
