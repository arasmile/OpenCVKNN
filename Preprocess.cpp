// Preprocess.cpp

#include "Preprocess.h"

void preprocess(cv::Mat &matTestingNumbers, cv::Mat &matGrayscale, cv::Mat &matThresh) {
    matGrayscale = extractValue(matTestingNumbers);                           

    cv::Mat imgMaxContrastGrayscale = maximizeContrast(matGrayscale);       

    cv::Mat matBlurred;

    cv::GaussianBlur(imgMaxContrastGrayscale, matBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);          
																									
	cv::threshold(matBlurred, matThresh, 0.0, 255.0, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
}


cv::Mat extractValue(cv::Mat &matTestingNumbers) {
    cv::Mat imgHSV;
    std::vector<cv::Mat> vectorOfHSVImages;
    cv::Mat imgValue;

    cv::cvtColor(matTestingNumbers, imgHSV, CV_BGR2HSV);

    cv::split(imgHSV, vectorOfHSVImages);

    imgValue = vectorOfHSVImages[2];

    return(imgValue);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat maximizeContrast(cv::Mat &matGrayscale) {
    cv::Mat imgTopHat;
    cv::Mat imgBlackHat;
    cv::Mat imgGrayscalePlusTopHat;
    cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

    cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));

    cv::morphologyEx(matGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);
    cv::morphologyEx(matGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

    imgGrayscalePlusTopHat = matGrayscale + imgTopHat;
    imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

    return(imgGrayscalePlusTopHatMinusBlackHat);
}


