#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include "EdgesSubPix.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    ocl::setUseOpenCL(false);
    const String keys
        = "{help h usage ? |          | print this message            }"
          "{@image         |          | image for edge detection      }"
          "{@output        |edge.tiff | image for draw edges          }"
          "{data           |          | edges data in txt format      }"
          "{low            |40        | low threshold                 }"
          "{high           |100       | high threshold                }"
          "{alpha          |1.0       | gaussian alpha                }";
    CommandLineParser parser(argc, argv, keys);
    parser.about("subpixel edge detection");

    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.has("@image")) {
        parser.printMessage();
        return 0;
    }

    String imageFile = parser.get<String>(0);
    String outputFile = parser.get<String>("@output");
    int low = parser.get<int>("low");
    int high = parser.get<int>("high");
    double alpha = parser.get<double>("alpha");

    Mat image = imread(imageFile, IMREAD_GRAYSCALE);
    EdgePoints edge_points;
    int64 t0 = getCPUTickCount();
    EdgesSubPix(image, alpha, low, high, edge_points);
    int64 t1 = getCPUTickCount();
    cout << "execution time is " << (t1 - t0) / (double)getTickFrequency()
         << " seconds" << endl;

    if (parser.has("data")) {
        FileStorage fs(parser.get<String>("data"),
            FileStorage::WRITE | FileStorage::FORMAT_YAML);
        fs << "edges"
           << "{";
        fs << "points" << edge_points.points;
        fs << "response" << edge_points.response;
        fs << "direction" << edge_points.direction;
        fs << "}";
        fs.release();
    }

    cv::Mat rgb;
    DrawEdges(rgb, image, edge_points, cv::Scalar(0, 255, 0), 10);

    cv::imwrite(outputFile, rgb);

    return 0;
}
