// Preprocess.h

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(3, 3);
const int ADAPTIVE_THRESH_BLOCK_SIZE = 21;
const int ADAPTIVE_THRESH_WEIGHT = 3;

// function prototypes ////////////////////////////////////////////////////////////////////////////

void preprocess(cv::Mat &matTestingNumbers, cv::Mat &matGrayscale, cv::Mat &matThresh);

cv::Mat extractValue(cv::Mat &matTestingNumbers);

cv::Mat maximizeContrast(cv::Mat &matGrayscale);


#endif	// PREPROCESS_H

