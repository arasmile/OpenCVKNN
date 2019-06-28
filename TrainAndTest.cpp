// TrainAndTest.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include"Preprocess.h"
#include<iostream>
#include<sstream>
#include<fstream>

const int MIN_CONTOUR_AREA = 80;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;


class ContourWithData {
public:
    std::vector<cv::Point> ptContour;           // contour
    cv::Rect boundingRect;                      // bounding rect for contour
    float fltArea;                              // area of contour

    bool checkIfContourIsValid() {                              
        if (fltArea < MIN_CONTOUR_AREA) return false;           
        return true;                                            
    }

    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   
    }

};


int main() {
    std::vector<ContourWithData> allContoursWithData;           // empty vectors
    std::vector<ContourWithData> validContoursWithData;         

	
    cv::Mat matClassificationInts;      
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        
    if (fsClassifications.isOpened() == false) {                                                    
        std::cout << "error, unable to open training classifications file, exiting program\n\n";    
        return(0);                                                                                  
    }
    fsClassifications["classifications"] >> matClassificationInts;      
    fsClassifications.release();                                        

    cv::Mat matTrainingImagesAsFlattenedFloats;        
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);        
    if (fsTrainingImages.isOpened() == false) {                                   
        std::cout << "error, unable to open training images file, exiting program\n\n";         
        return(0);                                                                              
    }
    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           
    fsTrainingImages.release();                                                

																				// KNN training
    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // KNN object

                                                                               
																			    // train KNN
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

 
																	// testing
    cv::Mat matTestingNumbers = cv::imread("image23.jpg");          // read test image

    if (matTestingNumbers.empty()) {                                // check file open, show error and exit
        std::cout << "error: image not read from file\n\n";         
        return(0);                                                  
    }

    cv::Mat matGrayscale;           
    cv::Mat matBlurred;             // 
    cv::Mat matThresh;              //
    cv::Mat matThreshCopy;          //
	
	preprocess(matTestingNumbers, matGrayscale, matThresh);
	
	cv::imshow("threshold", matThresh);

    matThreshCopy = matThresh.clone();              // thresh image copy -- findContours will change image

    std::vector<std::vector<cv::Point> > ptContours;        
    std::vector<cv::Vec4i> v4iHierarchy;                    

    cv::findContours(matThreshCopy,             // input image must be thresh copy
        ptContours,                             
        v4iHierarchy,                          
        cv::RETR_EXTERNAL,                     
        cv::CHAIN_APPROX_SIMPLE);               

    for (int i = 0; i < ptContours.size(); i++) {               
        ContourWithData contourWithData;                                                    
        contourWithData.ptContour = ptContours[i];                                          
        contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // bounding rect
        contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate area
        allContoursWithData.push_back(contourWithData);                                     
    }

    for (int i = 0; i < allContoursWithData.size(); i++) {                      // CHECK valid contour
        if (allContoursWithData[i].checkIfContourIsValid()) {                   
            validContoursWithData.push_back(allContoursWithData[i]);            
        }
    }
	
    // contour found sorting
    std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

    std::string strFinalString;         // final string declaration

    for (int i = 0; i < validContoursWithData.size(); i++) {            // for each valid contour

                                                                        
        cv::rectangle(matTestingNumbers,                            
            validContoursWithData[i].boundingRect,        
            cv::Scalar(0, 255, 0),                        // green
            2);                                           // thick

        cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          

        cv::Mat matROIResized;
        cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     

        cv::Mat matROIFloat;
        matROIResized.convertTo(matROIFloat, CV_32FC1);             

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

        cv::Mat matCurrentChar(0, 0, CV_32F);

        kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     
		
        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

        strFinalString = strFinalString + char(int(fltCurrentChar));        
    }

    std::cout << "\n\n" << "characters read = " << strFinalString << "\n\n";       

    cv::imshow("matTestingNumbers", matTestingNumbers);     // show input image with green boxes drawn around found digits

	// save characters read to file ///////////////////////////////////////////////////////
	std::ofstream fsText("text.txt");           // open the classifications file

	if (fsText.is_open() == false) {                                                        // if the file was not opened successfully
		std::cout << "error, unable to open training classifications file, exiting program\n\n";        // show error message
		return(0);                                                                                      // and exit program
	}

	fsText << "characters read: " << strFinalString;        // write classifications into classifications section of classifications file
	fsText.close();                                            // close the classifications file

																			// save training images to file ///////////////////////////////////////////////////////
	///////////////
    cv::waitKey(0);                                         // wait for user key press

    return(0);
}


