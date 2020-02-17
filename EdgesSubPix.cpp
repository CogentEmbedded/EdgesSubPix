#include "EdgesSubPix.h"

using namespace cv;
using namespace std;

const double scale = 128.0; // sum of half Canny filter is 128

static void getCannyKernel(OutputArray _d, double alpha)
{
    int r = cvRound(alpha * 3);
    int ksize = 2 * r + 1;

    _d.create(ksize, 1, CV_16S, -1, true);

    Mat k = _d.getMat();

    vector<float> kerF(ksize, 0.0f);
    kerF[r] = 0.0f;
    double a2 = alpha * alpha;
    float sum = 0.0f;
    for (int x = 1; x <= r; ++x) {
        float v = (float)(x * std::exp(-x * x / (2 * a2)));
        sum += v;
        kerF[r + x] = v;
        kerF[r - x] = -v;
    }
    float scale = 128 / sum;
    for (int i = 0; i < ksize; ++i) {
        kerF[i] *= scale;
    }
    Mat temp(ksize, 1, CV_32F, &kerF[0]);
    temp.convertTo(k, CV_16S);
}

static inline double getAmplitude(Mat& dx, Mat& dy, int i, int j)
{
    Point2d mag(dx.at<short>(i, j), dy.at<short>(i, j));
    return norm(mag);
}

static inline void getMagNeighbourhood(
    Mat& dx, Mat& dy, Point& p, int w, int h, vector<double>& mag)
{
    int top = p.y - 1 >= 0 ? p.y - 1 : p.y;
    int down = p.y + 1 < h ? p.y + 1 : p.y;
    int left = p.x - 1 >= 0 ? p.x - 1 : p.x;
    int right = p.x + 1 < w ? p.x + 1 : p.x;

    mag[0] = getAmplitude(dx, dy, top, left);
    mag[1] = getAmplitude(dx, dy, top, p.x);
    mag[2] = getAmplitude(dx, dy, top, right);
    mag[3] = getAmplitude(dx, dy, p.y, left);
    mag[4] = getAmplitude(dx, dy, p.y, p.x);
    mag[5] = getAmplitude(dx, dy, p.y, right);
    mag[6] = getAmplitude(dx, dy, down, left);
    mag[7] = getAmplitude(dx, dy, down, p.x);
    mag[8] = getAmplitude(dx, dy, down, right);
}

static inline void get2ndFacetModelIn3x3(
    vector<double>& mag, vector<double>& a)
{
    a[0] = (-mag[0] + 2.0 * mag[1] - mag[2] + 2.0 * mag[3] + 5.0 * mag[4]
               + 2.0 * mag[5] - mag[6] + 2.0 * mag[7] - mag[8])
        / 9.0;
    a[1] = (-mag[0] + mag[2] - mag[3] + mag[5] - mag[6] + mag[8]) / 6.0;
    a[2] = (mag[6] + mag[7] + mag[8] - mag[0] - mag[1] - mag[2]) / 6.0;
    a[3] = (mag[0] - 2.0 * mag[1] + mag[2] + mag[3] - 2.0 * mag[4] + mag[5]
               + mag[6] - 2.0 * mag[7] + mag[8])
        / 6.0;
    a[4] = (-mag[0] + mag[2] + mag[6] - mag[8]) / 4.0;
    a[5] = (mag[0] + mag[1] + mag[2] - 2.0 * (mag[3] + mag[4] + mag[5])
               + mag[6] + mag[7] + mag[8])
        / 6.0;
}
/*
   Compute the eigenvalues and eigenvectors of the Hessian matrix given by
   dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
   their absolute values.
*/
static inline void eigenvals(
    vector<double>& a, double eigval[2], double eigvec[2][2])
{
    // derivatives
    // fx = a[1], fy = a[2]
    // fxy = a[4]
    // fxx = 2 * a[3]
    // fyy = 2 * a[5]
    double dfdrc = a[4];
    double dfdcc = a[3] * 2.0;
    double dfdrr = a[5] * 2.0;
    double theta, t, c, s, e1, e2, n1, n2; /* , phi; */

    /* Compute the eigenvalues and eigenvectors of the Hessian matrix. */
    if (dfdrc != 0.0) {
        theta = 0.5 * (dfdcc - dfdrr) / dfdrc;
        t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
        if (theta < 0.0)
            t = -t;
        c = 1.0 / sqrt(t * t + 1.0);
        s = t * c;
        e1 = dfdrr - t * dfdrc;
        e2 = dfdcc + t * dfdrc;
    } else {
        c = 1.0;
        s = 0.0;
        e1 = dfdrr;
        e2 = dfdcc;
    }
    n1 = c;
    n2 = -s;

    /* If the absolute value of an eigenvalue is larger than the other, put
    that eigenvalue into first position.  If both are of equal absolute value,
    put the negative one first. */
    if (fabs(e1) > fabs(e2)) {
        eigval[0] = e1;
        eigval[1] = e2;
        eigvec[0][0] = n1;
        eigvec[0][1] = n2;
        eigvec[1][0] = -n2;
        eigvec[1][1] = n1;
    } else if (fabs(e1) < fabs(e2)) {
        eigval[0] = e2;
        eigval[1] = e1;
        eigvec[0][0] = -n2;
        eigvec[0][1] = n1;
        eigvec[1][0] = n1;
        eigvec[1][1] = n2;
    } else {
        if (e1 < e2) {
            eigval[0] = e1;
            eigval[1] = e2;
            eigvec[0][0] = n1;
            eigvec[0][1] = n2;
            eigvec[1][0] = -n2;
            eigvec[1][1] = n1;
        } else {
            eigval[0] = e2;
            eigval[1] = e1;
            eigvec[0][0] = -n2;
            eigvec[0][1] = n1;
            eigvec[1][0] = n1;
            eigvec[1][1] = n2;
        }
    }
}

std::tuple<cv::Point2f, float, float> improveSubPixPoint(
    Mat& dx, Mat& dy, cv::Point pt)
{
    int w = dx.cols;
    int h = dx.rows;

    vector<double> magNeighbour(9);
    getMagNeighbourhood(dx, dy, pt, w, h, magNeighbour);
    vector<double> a(9);
    get2ndFacetModelIn3x3(magNeighbour, a);

    // Hessian eigen vector
    double eigvec[2][2], eigval[2];
    eigenvals(a, eigval, eigvec);
    double t = 0.0;
    double ny = eigvec[0][0];
    double nx = eigvec[0][1];
    if (eigval[0] < 0.0) {
        double rx = a[1], ry = a[2], rxy = a[4], rxx = a[3] * 2.0,
               ryy = a[5] * 2.0;
        t = -(rx * nx + ry * ny)
            / (rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny);
    }
    double px = nx * t;
    double py = ny * t;
    float x = (float)pt.x;
    float y = (float)pt.y;
    if (fabs(px) <= 0.5 + 10 * FLT_EPSILON
        && fabs(py) <= 0.5 + 10 * FLT_EPSILON) {
        x += (float)px;
        y += (float)py;
    }

    return std::make_tuple(
        Point2f(x, y), (float)(a[0] / scale), (float)std::atan2(ny, nx));
}

void extractSubPixPoints(Mat& dx, Mat& dy, Mat& edge, EdgePoints& edge_points)
{
    edge_points.points.clear();
    edge_points.direction.clear();
    edge_points.response.clear();
#if defined(_OPENMP) && defined(NDEBUG)
#pragma omp parallel for
#endif
    for (int i = 0; i < edge.rows; i++) {
        for (int j = 0; j < edge.cols; j++) {
            if (edge.at<uchar>(i, j) > 0) {
                Point pt(j, i);

                cv::Point2f improved;
                float response;
                float direction;

                std::tie(improved, response, direction)
                    = improveSubPixPoint(dx, dy, pt);

                edge_points.points.push_back(improved);
                edge_points.response.push_back(response);
                edge_points.direction.push_back(direction);
            }
        }
    }
}

//---------------------------------------------------------------------
//          INTERFACE FUNCTIONS
//---------------------------------------------------------------------
void EdgesSubPix(cv::Mat& gray, double alpha, int low, int high,
    int blocksize, EdgePoints& edge_points)
{
    Mat blur;
    GaussianBlur(gray, blur, Size(0, 0), alpha, alpha);

    Mat d;
    getCannyKernel(d, alpha);
    Mat one = Mat::ones(Size(1, 1), CV_16S);
    Mat dx, dy;
    sepFilter2D(blur, dx, CV_16S, d, one);
    sepFilter2D(blur, dy, CV_16S, one, d);

    Mat edge;
    Canny(gray, edge, low, high, blocksize);

    // subpixel position extraction with steger's method and facet model 2nd
    // polynominal in 3x3 neighbourhood
    extractSubPixPoints(dx, dy, edge, edge_points);
}

void ContoursSubPix(Mat& gray, Mat& binary, double alpha, int low, int high,
    int blocksize, std::vector<EdgePoints>& contours)
{
    Mat blur;
    GaussianBlur(gray, blur, Size(0, 0), alpha, alpha);

    Mat d;
    getCannyKernel(d, alpha);
    Mat one = Mat::ones(Size(1, 1), CV_16S);
    Mat dx, dy;
    sepFilter2D(blur, dx, CV_16S, d, one);
    sepFilter2D(blur, dy, CV_16S, one, d);

    Mat edge;
    Canny(gray, edge, low, high, blocksize);

    // contours in pixel precision
    vector<vector<Point>> contoursInPixel;
    findContours(binary, contoursInPixel, cv::noArray(), cv::RETR_LIST,
        CHAIN_APPROX_NONE);

    // subpixel position extraction with steger's method and facet model 2nd
    // polynominal in 3x3 neighbourhood
    for (const auto& pixelContour : contoursInPixel) {
        contours.emplace_back();
        EdgePoints& edge_points = contours.back();

        for (const auto& pt : pixelContour) {
            cv::Point2f improved;
            float response;
            float direction;

            std::tie(improved, response, direction)
                = improveSubPixPoint(dx, dy, pt);

            edge_points.points.push_back(improved);
            edge_points.response.push_back(response);
            edge_points.direction.push_back(direction);
        }
    }
}

void DrawEdges(cv::Mat& rgb, cv::Mat& gray, const EdgePoints& edge_points,
    const cv::Scalar& color, const int scaleFactor)
{
    cv::Mat gray2;

    cv::resize(gray, gray2, gray.size() * scaleFactor, 0, 0, INTER_LINEAR);
    cv::cvtColor(gray2, rgb, cv::COLOR_GRAY2BGR);

    cv::Point2f offset(scaleFactor / 2. - 0.5, scaleFactor / 2. - 0.5);
    for (size_t i = 0; i < edge_points.points.size(); i++) {
        cv::Point2f b = scaleFactor * edge_points.points[i] + offset;
        cv::line(rgb, b, b, color);
    }
}

void DrawContours(cv::Mat& rgb, cv::Mat& gray,
    const std::vector<EdgePoints>& contours, const cv::Scalar& color,
    const int scaleFactor)
{
    cv::Mat gray2;

    cv::resize(gray, gray2, gray.size() * scaleFactor, 0, 0, INTER_LINEAR);
    cv::cvtColor(gray2, rgb, cv::COLOR_GRAY2BGR);

    cv::Point2f offset(scaleFactor / 2. - 0.5, scaleFactor / 2. - 0.5);
    for (const auto& edge_points : contours)
        for (size_t i = 0; i < edge_points.points.size(); i++) {
            cv::Point2f b = scaleFactor * edge_points.points[i] + offset;
            cv::line(rgb, b, b, color);
        }
}
