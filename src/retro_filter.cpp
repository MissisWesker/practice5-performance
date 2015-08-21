#include "retro_filter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <time.h>

using namespace std;
using namespace cv;

inline void alphaBlend(const Mat& src, Mat& dst, const Mat& alpha)
{
    Mat w, d, s, dw, sw;
    alpha.convertTo(w, CV_32S);
    src.convertTo(s, CV_32S);
    dst.convertTo(d, CV_32S);

    multiply(s, w, sw);
    multiply(d, -w, dw);
    d = (d*255 + sw + dw)/255.0;
    d.convertTo(dst, CV_8U);
}

RetroFilter::RetroFilter(const Parameters& params) : rng_(time(0))
{
    params_ = params;

    resize(params_.fuzzyBorder, params_.fuzzyBorder, params_.frameSize);

    if (params_.scratches.rows < params_.frameSize.height ||
        params_.scratches.cols < params_.frameSize.width)
    {
        resize(params_.scratches, params_.scratches, params_.frameSize);
    }

    hsvScale_ = 1;
    hsvOffset_ = 20;
}

void RetroFilter::applyToVideo(const Mat& frame, Mat& retroFrame)
{
    Mat luminance;
    cvtColor(frame, luminance, CV_BGR2GRAY);

    // Add scratches
    Scalar meanColor = mean(luminance.row(luminance.rows / 2)) * 2.0;
	int x = rng_.uniform(0, params_.scratches.cols - luminance.cols);
    int y = rng_.uniform(0, params_.scratches.rows - luminance.rows);
	Mat scratches_mask = params_.scratches(Rect(Point(x, y), luminance.size()));
	luminance.setTo(meanColor, scratches_mask);

    // Add fuzzy border
	Mat borderColor(params_.frameSize, CV_8UC1, meanColor[0] * 1.5);
    alphaBlend(borderColor, luminance, params_.fuzzyBorder);


    // Apply sepia-effect
	vector<Mat> hsv(3);
	hsv[0] = Mat(luminance.size(), CV_8UC1, 19);
	hsv[1] = Mat(luminance.size(), CV_8UC1, 78);
	hsv[2] = luminance * hsvScale_ + hsvOffset_;
	merge(hsv, retroFrame);
	cvtColor(retroFrame, retroFrame, COLOR_HSV2BGR);
}
