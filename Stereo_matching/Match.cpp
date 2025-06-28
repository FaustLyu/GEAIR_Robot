#define _CRT_SECURE_NO_WARNINGS


#include <iostream>
#include <json/json.h>
#include <string>
#include <fstream>
#include <time.h>
#include <ctime>
#include <io.h>
#include <conio.h>
#include <chrono>
#include <sstream>
#include <direct.h>

#include <opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>

#include <librealsense2/rs.hpp>

#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>

#include <ur_rtde/robotiq_gripper.h>

#include <Windows.h>

#define pi 3.1415926
#define width 1280 
#define height 720 

using namespace std;
using namespace cv;
using namespace ur_rtde;
using namespace rs2;

vector <vector<double>> vBoxes;
vector <vector<vector<double>>> vMasks;
vector <int> vClasses;
vector <double> vScores;

time_t TimeStamp;
string ImgPath;
string ImgBakPath;
string ImgName;
string ImgDir = "../../data/220714FlowerProj/image/";
string JsonDir = "../../data/220714FlowerProj/json/";
string DataDir = "../../data/220714FlowerProj/time/";
string VideoDir = "../../data/220714FlowerProj/video/";
string PathDir = "../../data/220714FlowerProj/path/";


char strVideoDir[1024] = "../../data/220714FlowerProj/video/";
char strVideoDirCpy[1024] = "../../data/220714FlowerProj/video/";

Mat mCurImg = Mat::zeros(1280, 720, CV_8UC3);
unsigned char *ucImgBuf = new unsigned char[width * height * 3];




int b = 10;
int focal_length = 910;

double brushlen = 120.0;

int nRow = 720;
int nCol = 1280;

bool Test = false;
bool Display = true;
bool Flag = true;


bool IsSafeRange(vector<double> p)
{
    if (p[0] > -0.5 && p[0] < 0.5 &&
        p[1] < -0.1 &&
        p[2] > 0.1)
    {
        return true;
    }
    else
    {
        return false;
    }
}   

void PrintMat(Mat matrix)
{
    int n = matrix.rows;
    int m = matrix.cols;
    std::cout << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            std::cout << double(matrix.ptr<double>(i)[j]) << "\t";
        }
        std::cout << endl;
    }
    std::cout << endl;
}

void PrintVct(vector<double> vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << "\t";
    }
    std::cout << endl;
}   

//read detection result from json
void ReadDetection(string json,
	vector <vector<double>> &boxes,
	vector <vector<vector<double>>> &masks,
	vector <int> &classes,
	vector <double> &scores)
{
	int nBegin = cv::getTickCount();
	Json::Reader reader;
	Json::Value root;
	ifstream srcFile(json, ios::binary);/*定义一个ifstream流对象，与文件demo.json进行关联*/


	if (!srcFile.is_open())
	{
        std::cout << "Fail to open src.json" << endl;
		return;
	}

	if (reader.parse(srcFile, root))
	{
		int count = root["box"].size();
        //std::cout << count << endl;
		boxes.resize(count);
		classes.resize(count);
		scores.resize(count);
		masks.resize(count);

		for (int i = 0; i < count; i++)
		{
			Mat mask(138, 138, CV_8UC1, Scalar(0));

			int nBoxSize = root["box"][i].size();
			boxes[i].resize(nBoxSize);
			for (int j = 0; j < nBoxSize; j++)
			{

				boxes[i][j] = root["box"][i][j].asFloat();
			}

			classes[i] = root["class"][i].asInt();
			scores[i] = root["score"][i].asFloat();

			//int nMaskSize = root["mask"][i].size();
			//masks[i].resize(nMaskSize);

			//for (int k = 0; k < nMaskSize; k++)
			//{
			//	int nMaskRow = root["mask"][i][k].size();
			//	masks[i][k].resize(nMaskRow);
			//	uchar* cMaskPtr = mask.ptr(k);
			//	for (int m = 0; m < nMaskRow; m++)
			//	{
			//		masks[i][k][m] = root["mask"][i][k][m].asFloat();
			//		cMaskPtr[m] = floor(255 * root["mask"][i][k][m].asFloat());
			//	}
			//}
			//resize(mask, mask, Size(1080, 1080), 0, 0, INTER_LINEAR);
			//inRange(mask, 128, 256, mask);
			////imshow("mask", mask);
			////waitKey(0.5);
	

		}
		int nEnd = cv::getTickCount();
		double dTdetec = 1000.0 * (nEnd - nBegin) / cv::getTickFrequency();
		//std::cout << dTdetec << endl;
	}
}

//check whether a file exits
bool IsFileExist(const string& FilePath)
{
	return _access_s(FilePath.c_str(), 0) == 0;
}

//func for GetDepth
bool comp(const DMatch& a, const DMatch& b)
{
	return a.distance < b.distance;
}

struct CameraPoint
{
	int x;
	int y;
	double depth;
};

//match and get depth
double GetDepth(Mat img1, Mat img2, int bias)
{
    //begin time
    clock_t t1, t2;
    t1 = clock();

    //imshow("1", img1);
    //imshow("2", img2);
    //waitKey(0);
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	Mat gray1, gray2;
	cv::cvtColor(img1, gray1, CV_BGR2GRAY);
	cv::cvtColor(img2, gray2, CV_BGR2GRAY);

	int minHessian = 5000;
	Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(minHessian);
	vector<KeyPoint> keypoints1, keypoints2;
	//int t1 = getTickCount();

	detector->detect(gray1, keypoints1, Mat());
	detector->detect(gray2, keypoints2, Mat());


	Mat desp1, desp2;
	detector->compute(gray1, keypoints1, desp1);
	detector->compute(gray2, keypoints2, desp2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	vector<vector<DMatch>> knn_matches;
	double ratio_threshold = 0.9;
	vector<DMatch> good_matches;
	matcher->knnMatch(desp1, desp2, knn_matches, 2);
	vector<CameraPoint> points;

	 
	for (auto& knn_matche : knn_matches)
	{
        int leftX = knn_matche[0].queryIdx;
        int rightX = knn_matche[0].trainIdx;
        if (knn_matche[0].distance < ratio_threshold * knn_matche[1].distance &&
            abs(keypoints1[leftX].pt.y - keypoints2[rightX].pt.y) < 30)
		{
			good_matches.push_back(knn_matche[0]);
			CameraPoint point;
			int disparity = abs(keypoints1[knn_matche[0].queryIdx].pt.x + bias - keypoints2[knn_matche[0].trainIdx].pt.x);
			point.x = round(keypoints1[knn_matche[0].queryIdx].pt.x);
			point.y = round(keypoints1[knn_matche[0].queryIdx].pt.y);
			if (disparity == 0)
			{
				point.depth = 0.0;
			}
			else
			{
				point.depth = b * focal_length / disparity;
			}
			points.push_back(point);
		}
	}
    Mat img_matches_knn;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches_knn, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// get depth 
	// a better method needed
	double sum = 0.0;
	for (int i = 0; i < points.size(); i++)
	{
		sum += points[i].depth;
	}

    double depth = sum / points.size();

    vector<CameraPoint> goodpoints;
    for (int i = 0; i < points.size(); i++)
    {
        if (abs(points[i].depth - depth) < 0.5*depth)
        {
            goodpoints.push_back(points[i]);
        }
    }

    sum = 0.0;
    for (int i = 0; i < goodpoints.size(); i++)
    {
        sum += goodpoints[i].depth;
    }
    // end time
    t2 = clock();
    double timecost = (double)(t2 - t1) / CLOCKS_PER_SEC;

    // write time to file
    if (sum != 0.0)
    {
        ofstream fp;
        fp.open(DataDir + "location_time.txt", ofstream::app);
        fp << (double)timecost << endl;
        fp.close();
        //
    }
    


    if (Display)
    {
        imshow("match", img_matches_knn);
        waitKey(0);
    }
	return sum / goodpoints.size(); //unit: mm

}

//convert rx, ry, ry from robot to rotation matrix
Mat Rxyz2Mat(vector<double> Rxyz)
{
    double dThetaRad = sqrt(pow(Rxyz[0], 2) + pow(Rxyz[1], 2) + pow(Rxyz[2], 2));
    double scale = abs(dThetaRad);
    double fx = Rxyz[0] / scale;
    double fy = Rxyz[1] / scale;
    double fz = Rxyz[2] / scale;
    Mat T(Size(3, 3), CV_64FC1, Scalar(0));
    //todo
    T.ptr<double>(0)[0] = fx * fx * (1 - cos(dThetaRad)) + cos(dThetaRad);
    T.ptr<double>(0)[1] = fy * fx * (1 - cos(dThetaRad)) - fz * sin(dThetaRad);
    T.ptr<double>(0)[2] = fz * fx * (1 - cos(dThetaRad)) + fy * sin(dThetaRad);
    T.ptr<double>(1)[0] = fx * fy * (1 - cos(dThetaRad)) + fz * sin(dThetaRad);
    T.ptr<double>(1)[1] = fy * fy * (1 - cos(dThetaRad)) + cos(dThetaRad);
    T.ptr<double>(1)[2] = fz * fy * (1 - cos(dThetaRad)) - fx * sin(dThetaRad);
    T.ptr<double>(2)[0] = fx * fz * (1 - cos(dThetaRad)) - fy * sin(dThetaRad);
    T.ptr<double>(2)[1] = fy * fz * (1 - cos(dThetaRad)) + fx * sin(dThetaRad);
    T.ptr<double>(2)[2] = fz * fz * (1 - cos(dThetaRad)) + cos(dThetaRad);

    return T;
}

//convert rotation matrix to rx, ry, rz for robot
vector<double> Mat2Rxyz(Mat T)
{
    vector<double> Rxyz;

    double nx = T.ptr<double>(0)[0];
    double ny = T.ptr<double>(1)[0];
    double nz = T.ptr<double>(2)[0];

    double ox = T.ptr<double>(0)[1];
    double oy = T.ptr<double>(1)[1];
    double oz = T.ptr<double>(2)[1];

    double ax = T.ptr<double>(0)[2];
    double ay = T.ptr<double>(1)[2];
    double az = T.ptr<double>(2)[2];

    double dTheta = atan2(sqrt(pow(oz - ay, 2) + pow(ax - nz, 2) + pow(ny - ox, 2)), nx + oy + az - 1);
    double fx = (oz - ay) / (2 * sin(dTheta));
    double fy = (ax - nz) / (2 * sin(dTheta));
    double fz = (ny - ox) / (2 * sin(dTheta));

    //归一化todo
    double scale = sqrt(pow(dTheta, 2) / (pow(fx, 2) + pow(fy, 2) + pow(fz, 2)));
    double RX = scale * fx;
    double RY = scale * fy;
    double RZ = scale * fz;
    Rxyz.push_back(RX);
    Rxyz.push_back(RY);
    Rxyz.push_back(RZ);
    return Rxyz;
}

//move camera to the positon of tcp
vector<double> Robot2Camera(vector<double> vRobotTcp)
{
    Mat mRobotPose(Size(4, 4), CV_64FC1, Scalar(0));
    mRobotPose.ptr<double>(0)[3] = vRobotTcp[0];
    mRobotPose.ptr<double>(1)[3] = vRobotTcp[1];
    mRobotPose.ptr<double>(2)[3] = vRobotTcp[2];
    mRobotPose.ptr<double>(3)[3] = 1.0;
    vector<double> vRobotRxyz{ vRobotTcp[3],  vRobotTcp[4], vRobotTcp[5] };

    //std::cout << "vRobotRxyz" << endl;
    //for (int i = 0; i < 3; i++)
    //{
    //    std::cout << vRobotRxyz[i] << endl;
    //}
    Mat mRobotRotate = Rxyz2Mat(vRobotRxyz);
    //std::cout << "mRobotRotate" << endl;
    //for (int i = 0; i < 3; i++)
    //{
    //    for (int j = 0; j < 3; j++)
    //    {
    //        std::cout << double(mRobotRotate.ptr<double>(i)[j]) << "\t";
    //    }
    //    std::cout << endl;
    //}
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            mRobotPose.ptr<double>(i)[j] = double(mRobotRotate.ptr<double>(i)[j]);
        }
    }
    //std::cout << "mRobotPose" << endl;

    //for (int i = 0; i < 4; i++)
    //{
    //    for (int j = 0; j < 4; j++)
    //    {
    //        std::cout << double(mRobotPose.ptr<double>(i)[j]) << "\t";
    //    }
    //    std::cout << endl;
    //}
    Mat Trc = (Mat_<double>(4, 4) << 1, 0, 0, 0.0329,
        0, 0.73, -0.67, 0.0543,
        0, 0.67, 0.73, 0.13975,
        0, 0, 0, 1);

    Mat mCameraPose = mRobotPose * Trc;
    //std::cout << "mCameraPose" << endl;

    //for (int i = 0; i < 4; i++)
    //{
    //    for (int j = 0; j < 4; j++)
    //    {
    //        std::cout << double(mCameraPose.ptr<double>(i)[j]) << "\t";
    //    }
    //    std::cout << endl;
    //}
    Mat mCameraRotate(mCameraPose, Range(0, 3), Range(0, 3));
    vector<double> vCameraRotate = Mat2Rxyz(mCameraRotate);

    vector<double> vCameraTcp{ double(mCameraPose.ptr<double>(0)[3]),
                                double(mCameraPose.ptr<double>(1)[3]),
                                double(mCameraPose.ptr<double>(2)[3]),
                                vCameraRotate[0],
                                vCameraRotate[1],
                                vCameraRotate[2],
    };
    //std::cout << "camera tcp:" << endl;
    //for (int i = 0; i < 6; i++)
    //{
    //    std::cout << vCameraTcp[i] << endl;
    //}
    return vCameraTcp;
}

//move tcp to the position of camera
vector<double> Camera2Robot(vector<double> vCameraTcp)
{
    Mat mCameraPose(Size(4, 4), CV_64FC1, Scalar(0));
    mCameraPose.ptr<double>(0)[3] = vCameraTcp[0];
    mCameraPose.ptr<double>(1)[3] = vCameraTcp[1];
    mCameraPose.ptr<double>(2)[3] = vCameraTcp[2];
    mCameraPose.ptr<double>(3)[3] = 1.0;
    vector<double> vCameraRxyz{ vCameraTcp[3],  vCameraTcp[4], vCameraTcp[5] };

    //std::cout << "vRobotRxyz" << endl;
    //for (int i = 0; i < 3; i++)
    //{
    //    std::cout << vCameraRxyz[i] << endl;
    //}
    Mat mCameraRotate = Rxyz2Mat(vCameraRxyz);
    //std::cout << "mCameraRotate" << endl;
    //for (int i = 0; i < 3; i++)
    //{
    //    for (int j = 0; j < 3; j++)
    //    {
    //        std::cout << double(mCameraRotate.ptr<double>(i)[j]) << "\t";
    //    }
    //    std::cout << endl;
    //}
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            mCameraPose.ptr<double>(i)[j] = double(mCameraRotate.ptr<double>(i)[j]);
        }
    }
    //std::cout << "mCameraPose" << endl;

    //for (int i = 0; i < 4; i++)
    //{
    //    for (int j = 0; j < 4; j++)
    //    {
    //        std::cout << double(mCameraPose.ptr<double>(i)[j]) << "\t";
    //    }
    //    std::cout << endl;
    //}

    Mat Trc = (Mat_<double>(4, 4) << 1, 0, 0, 0.0329,
        0, 0.73, -0.67, 0.0543,
        0, 0.67, 0.73, 0.13975,
        0, 0, 0, 1);
    Mat Tcr;
    invert(Trc, Tcr);
    
    //1       0             0           32.9
    //0       0.743532      0.68242     - 135.742
    //0       - 0.68242     0.743532    - 66.8532
    //0       0             0     1

    PrintMat(Tcr);
    Mat mRobotPose = mCameraPose * Tcr;
    PrintMat(mCameraPose);
    PrintMat(mRobotPose);
    //std::cout << "mRobotPose" << endl;

    //for (int i = 0; i < 4; i++)
    //{
    //    for (int j = 0; j < 4; j++)
    //    {
    //        std::cout << double(mRobotPose.ptr<double>(i)[j]) << "\t";
    //    }
    //    std::cout << endl;
    //}
    Mat mRobotRotate(mRobotPose, Range(0, 3), Range(0, 3));
    vector<double> vRobotRotate = Mat2Rxyz(mRobotRotate);

    vector<double> vRobotTcp{ double(mRobotPose.ptr<double>(0)[3]),
                                double(mRobotPose.ptr<double>(1)[3]),
                                double(mRobotPose.ptr<double>(2)[3]),
                                vRobotRotate[0],
                                vRobotRotate[1],
                                vRobotRotate[2],
    };
    //std::cout << "Robot tcp:" << endl;
    //for (int i = 0; i < 6; i++)
    //{
    //    std::cout << vRobotTcp[i] << endl;
    //}
    return vRobotTcp;
}

//get camera coordiantes from image coordiantes
vector<double> Image2Camera(double fU, double fV, double fZ)
{
    vector<double> vCameraCoordinate;
    vCameraCoordinate.push_back((fU - 645.67) * fZ / 909.53 / 1000); // 1000 : mm to m
    vCameraCoordinate.push_back((fV - 355.99) * fZ / 913.03 / 1000);
    vCameraCoordinate.push_back(fZ / 1000);
    return vCameraCoordinate;
}

//get world coordiantes from camera coordiantes
//TODO
vector<double> Camera2World(vector<double> vCameraP, vector<double> vRobotTcp)
{
    Mat mRobotPose(Size(4, 4), CV_64FC1, Scalar(0));
    mRobotPose.ptr<double>(0)[3] = vRobotTcp[0];
    mRobotPose.ptr<double>(1)[3] = vRobotTcp[1];
    mRobotPose.ptr<double>(2)[3] = vRobotTcp[2];
    mRobotPose.ptr<double>(3)[3] = 1.0;
    vector<double> vRobotRxyz{ vRobotTcp[3],  vRobotTcp[4], vRobotTcp[5] };

    
    Mat mRobotRotate = Rxyz2Mat(vRobotRxyz);
    //PrintMat(mRobotRotate);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            mRobotPose.ptr<double>(i)[j] = double(mRobotRotate.ptr<double>(i)[j]);
        }
    }
    //PrintMat(mRobotPose);
    //Tcw = Trw * Tcr;
    Mat Trc = (Mat_<double>(4, 4) << 1, 0, 0, 0.0329,
        0, 0.73, -0.67, 0.0543,
        0, 0.67, 0.73, 0.13975,
        0, 0, 0, 1);
    Mat Tcr;
    invert(Trc, Tcr);
    //PrintMat(Tcr);
    Mat Tcw = mRobotPose * Tcr;
    //PrintMat(Tcw);
    Mat Pc = (Mat_<double>(4, 1) << vCameraP[0], vCameraP[1], vCameraP[2], 1);
    //PrintMat(Pc);
    Mat Pw = Tcw * Pc;
    //PrintMat(Pw);
    vector<double> vWorldP;
    for (int i = 0; i<3;i++)
    {
        vWorldP.push_back(double(Pw.ptr<double>(i)[0]));
    }
    return vWorldP;
}

//rotate src ? degree along ? axis to dst
void Rotate(Mat src, Mat dst, int axis, double degree)
{
    double rad = pi * degree / 180;
    Mat T(3, 3, CV_64FC1, 0);

    switch (axis)
    {
    case 0:
    {
        T.ptr<double>(1)[1] = cos(rad);
        T.ptr<double>(1)[2] = -sin(rad);
        T.ptr<double>(2)[1] = sin(rad);
        T.ptr<double>(2)[2] = cos(rad);
    }
    case 1:
    {
        T.ptr<double>(0)[0] = cos(rad);
        T.ptr<double>(0)[2] = sin(rad);
        T.ptr<double>(2)[0] = -sin(rad);
        T.ptr<double>(2)[2] = cos(rad);
    }
    case 2:
    {
        T.ptr<double>(0)[0] = cos(rad);
        T.ptr<double>(0)[1] = -sin(rad);
        T.ptr<double>(1)[0] = sin(rad);
        T.ptr<double>(1)[1] = cos(rad);
    }
    }
    dst = src * T;
}

double GetDistance(vector<double> vPoint1, vector<double> vPoint2)
{
    double sum = 0;
    for (int i = 0; i < 3; i++)
    {
        sum += pow((vPoint1[i] - vPoint2[i]), 2);
    }
    return sqrt(sum);
}

bool IsReach(vector<double> vPoint1, vector<double> vPoint2, double threshold = 0.01)
{
    double sum = 0;
    for (int i = 0; i < 3; i++)
    {
        sum += abs(vPoint1[i] - vPoint2[i]);
    }
    for (int j = 3; j < 6; j++)
    {
        sum += abs(vPoint1[j] - vPoint1[j]);
    }
    if (sum < threshold)
    {
        return true;
    }
    else
    {
        return false;
    }

}

string GetTimeStampMs(int time_stamp_type = 0)
{

}

bool ConvertMatToUChar_3Bit(
    unsigned char* imgDstBuf3Bit,
    Mat aMatSrc,

    int nWidth,
    int nHeight
)
{
    memset(&(imgDstBuf3Bit[0]), 0x00, nWidth * nHeight * 3);

    for (int i = 0; i < nWidth * nHeight * 3; i++)
    {
        imgDstBuf3Bit[i] = aMatSrc.at<Vec3b>(i / (nWidth * 3), (i % (nWidth * 3)) / 3)[i % 3];
    }

    return 0;
}

// 把 unsigned char * 转为  Mat
Mat ConvertUCharToMat_3Bit(
    unsigned char* imgSrcBuf3Bit,

    int nWidth,
    int nHeight
)
{
    Mat aMat(Size(nWidth, nHeight), CV_8UC3);

    for (int i = 0; i < nWidth * nHeight * 3; i++)
    {
        aMat.at<Vec3b>(i / (nWidth * 3), (i % (nWidth * 3)) / 3)[i % 3] = imgSrcBuf3Bit[i];
    }

    return aMat;
}

void WriteTrackPoint(vector<double> TrackPoint, string FileName)
{
    FILE* fp;
    const char* p = FileName.data();
    fp = fopen(p, "a+");
    for (int i = 0; i < TrackPoint.size(); i++)
    {
        fprintf(fp, "%0.5f\t", TrackPoint[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    return;
}

DWORD WINAPI ThreadProc_RS2ImgProc(LPVOID lppara)
{
    colorizer gColorMap;
    pipeline gPipe;

    gPipe.start();

    int nCount = 0;
    char strTmp[8] = "";

    //get time stamp
    TimeStamp = time(NULL);
    string SubVideoDir = to_string(time(&TimeStamp));


    if (access((VideoDir + SubVideoDir).c_str(), 0) != 0)
    {
        mkdir((VideoDir + SubVideoDir).c_str());
    }


    //updata frame
    for (int j = 0; j < 30; j++)
    {
        frameset data = gPipe.wait_for_frames(); // Wait for next set of frames from the camera
        frame color = data.get_color_frame().apply_filter(gColorMap);
    }


    //collect image
    while (1)
    {

        ////get time stamp
        //TimeStamp = time(NULL);
        //ImgName = to_string(time(&TimeStamp));
        //ImgPath = VideoDir + ImgName + ".png";

        //strcpy(strVideoDir, strVideoDirCpy);
        ////
        string FrameName;
        if ((nCount >= 0) && (nCount < 10))
        {
            FrameName = "0000" + to_string(nCount);
        }
        if ((nCount >= 10) && (nCount < 100))
        {
            FrameName = "000" + to_string(nCount);
        }
        if ((nCount >= 100) && (nCount < 1000))
        {
            FrameName = "00" + to_string(nCount);
        }
        if ((nCount >= 1000) && (nCount < 10000))
        {
            FrameName = "0" + to_string(nCount);
        }
        nCount += 1;
        string FramePath = VideoDir + SubVideoDir + "/" + FrameName + ".jpg";

        //_itoa(nCount, strTmp, 10);
        //nCount += 1;
        //strcat(strVideoDir, strTmp);+
        //strcat(strVideoDir, ".jpg");

        //

        //
        Sleep(100);
        frameset data = gPipe.wait_for_frames(); // Wait for next set of frames from the camera
        frame color = data.get_color_frame().apply_filter(gColorMap);
        const int w = color.as<video_frame>().get_width();
        const int h = color.as<video_frame>().get_height();

        mCurImg = Mat(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
        //memset(&(ucImgBuf[0]), 0x00, width * height * 3);

        //ConvertMatToUChar_3Bit(&(ucImgBuf[0]), mCurImg, width, height);

        cv::cvtColor(mCurImg, mCurImg, COLOR_RGB2BGR);
        cv::imwrite(FramePath, mCurImg);
        //std::cout << strVideoDir << endl;

    }
    

}

void main()
{
    DWORD dwThread_RS2Img;
    ::CreateThread(NULL, 0, ThreadProc_RS2ImgProc, (LPVOID)0, 1024, &dwThread_RS2Img);

    std::cout << "Camera Started" << endl;
    FILE* fp;
    fp = fopen((PathDir + "path.txt").data(), "w");
    fclose(fp);

    //connect robot
    RTDEControlInterface rtde_control("192.168.1.77");
    RTDEReceiveInterface rtde_receive("192.168.1.77");
    RobotiqGripper gripper("192.168.1.77", 63352, true);
    std::cout << "Robot Connected" << endl;

    //rtde_control.stopScript();
    //vector<double> p = rtde_receive.getActualTCPPose();
    //PrintVct(p);

    //Move to home point
    vector<double> vHomePoint(6);
    vHomePoint[0] = 0.0;
    vHomePoint[1] = -0.3;
    vHomePoint[2] = 0.6;
    vHomePoint[3] = 0.0;
    vHomePoint[4] = -2.228;
    vHomePoint[5] = 2.222;
    if (IsSafeRange(vHomePoint))
    {
        vHomePoint = Robot2Camera(vHomePoint);
        rtde_control.moveL(vHomePoint, 0.05, 0.25, true);
        Sleep(5000);
        WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");

        std::cout << "Moving to Home Point" << endl;
    }
    else
    {
        std::cout << "Position:";
        PrintVct(vHomePoint);
        std::cout << "Out of Range!" << endl;
        exit(0);                
    }


    //gripper initialize
    gripper.connect();
/*    gripper.activate();
    gripper.setUnit(RobotiqGripper::POSITION, RobotiqGripper::UNIT_MM);
    gripper.setForce(0.0);
    gripper.setSpeed(0.2);
    gripper.move(0);
    gripper.waitForMotionComplete();
    std::cout << "Gripper Initialized" << endl;   */ 

    //read image from camera
    //connect realsense
    //colorizer color_map;
    //pipeline pipe;
    ////
    //pipe.start();

    //while (1)
    //{
    //    Sleep(30);
    //}

    clock_t tPBegin, tPEnd;

    //search track
    double dVelocity = 0.01;
    double dAcce = 0.2;
    double dBlend = 0.0;
    vector <double> vDTrackP1 = { -0.2, -0.4, 0.88, 0.0, -2.228, 2.222 };
    vector <double> vDTrackP2 = { 0.2, -0.4, 0.88, 0.0, -2.228, 2.222 };
    vector <double> vDTrackP3 = { 0.2, -0.4, 0.65, 0.0, -2.228, 2.222 };
    vector <double> vDTrackP4 = { -0.2, -0.4, 0.65, 0.0, -2.228, 2.222 };
    vector <double> vDTrackP5 = { -0.2, -0.4, 0.4, 0.0, -2.228, 2.222 };
    vector <double> vDTrackP6 = { 0.2, -0.4, 0.4, 0.0, -2.228, 2.222 };
    vector <vector<double>> vVDTrack;
    vVDTrack.push_back(vDTrackP1);
    vVDTrack.push_back(vDTrackP2);
    vVDTrack.push_back(vDTrackP3);
    vVDTrack.push_back(vDTrackP4);
    vVDTrack.push_back(vDTrackP5);
    vVDTrack.push_back(vDTrackP6);
    vector <double> vDCameraTrack;
    vector <vector<double>> vVDValidTrack;

    for (int i = 0; i < vVDTrack.size(); i++)
    {
        if (IsSafeRange(vVDTrack[i]))
        {
            vDCameraTrack = Robot2Camera(vVDTrack[i]);
            rtde_control.moveL(vDCameraTrack, 0.05, 0.05, true);
           
            std::cout << "Moving to Track Point " << i+1 << endl;
            while (!IsReach(rtde_receive.getActualTCPPose(), vDCameraTrack))
            {
                Sleep(100);
            }
            WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");
            Sleep(200);

        }
        else
        {
            std::cout << "Position:";
            PrintVct(vVDTrack[i]);
            std::cout << "Out of Range!" << endl;
            exit(0);
        }

        //get time stamp
        TimeStamp = time(NULL);
        ImgName = to_string(time(&TimeStamp));
        ImgPath = ImgDir + ImgName + ".png";
        ImgBakPath = ImgDir + ImgName + "_bak.png";

        ////updata frame
        //for (int j = 0; j < 30; j++)
        //{
        //    frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        //    frame color = data.get_color_frame().apply_filter(color_map);
        //}

        ////collect image
        //Sleep(30);
        //frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        //frame color = data.get_color_frame().apply_filter(color_map);
        //const int w = color.as<video_frame>().get_width();
        //const int h = color.as<video_frame>().get_height();

        //Mat image(mCurImg);
        Mat image = ConvertUCharToMat_3Bit(&(ucImgBuf[0]), width, height);
        cv::cvtColor(image, image, COLOR_RGB2BGR);
        imwrite(ImgPath, image);
        imwrite(ImgBakPath, image);
        imwrite(ImgDir + "left.jpg", image);
        std::cout << "Image Saved, Waiting for Detection ..." << endl;

        //read detection result
        while (!IsFileExist(JsonDir + ImgName + "_bak.json"))
        {
            Sleep(100);
        }
        std::cout << "Dectecion Received" << endl;

        vBoxes.clear();
        vMasks.clear();
        vClasses.clear();
        vScores.clear();

        //the detections have been sorted by score
        ReadDetection(JsonDir + ImgName + "_det.json", vBoxes, vMasks, vClasses, vScores);

        if (vScores.size() == 0)
        {
            std::cout << "No Flower Detected" << endl;

            //move robot
            //
        }
        else
        {
            vVDValidTrack.push_back(vVDTrack[i]);
        }



    }
      
    for (int m = 0; m < vVDValidTrack.size(); m++)
    {
        if (IsSafeRange(vVDValidTrack[m]))
        {
            vDCameraTrack = Robot2Camera(vVDValidTrack[m]);
            rtde_control.moveL(vDCameraTrack, 0.05, 0.05, true);

            std::cout << "Moving to Track Point " << m + 1 << endl;
            while (!IsReach(rtde_receive.getActualTCPPose(), vDCameraTrack))
            {
                Sleep(100);
            }
            Sleep(200);
            WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");


        }
        else
        {
            std::cout << "Position:";
            PrintVct(vVDValidTrack[m]);
            std::cout << "Out of Range!" << endl;
            exit(0);
        }

        Flag = true;
        tPBegin = clock();
        while (Flag)
        {
            
            //get time stamp
            TimeStamp = time(NULL);
            ImgName = to_string(time(&TimeStamp));
            ImgPath = ImgDir + ImgName + ".png";
            ImgBakPath = ImgDir + ImgName + "_bak.png";

            ////updata frame
            //for (int i = 0; i < 30; i++)
            //{
            //    frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
            //    frame color = data.get_color_frame().apply_filter(color_map);
            //}

            ////collect image
            //Sleep(30);
            //frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
            //frame color = data.get_color_frame().apply_filter(color_map);
            //const int w = color.as<video_frame>().get_width();
            //const int h = color.as<video_frame>().get_height();

            //Mat image(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
            //Mat image(mCurImg);
            Mat image = ConvertUCharToMat_3Bit(&(ucImgBuf[0]), width, height);
            cv::cvtColor(image, image, COLOR_RGB2BGR);
            imwrite(ImgPath, image);
            imwrite(ImgBakPath, image);
            imwrite(ImgDir + "left.jpg", image);
            std::cout << "Left Image Saved, Waiting for Detection ..." << endl;



            //read detection result
            while (!IsFileExist(JsonDir + ImgName + "_bak.json"))
            {
                Sleep(100);
            }
            std::cout << "Dectecion Received" << endl;


            vBoxes.clear();
            vMasks.clear();
            vClasses.clear();
            vScores.clear();

            //the detections have been sorted by score
            ReadDetection(JsonDir + ImgName + "_det.json", vBoxes, vMasks, vClasses, vScores);


            if (vScores.size() == 0)
            {
                std::cout << "No Flower Detected" << endl;

                //move robot
                //
            }
            else
            {
                std::cout << "Flower Detected" << endl;
                //read the first box, keep the data in box and remove the remaining
                //TODO: mask sure the box not on the bounding of the whole image left, so that the box must be in image right;
                vector<double> Box = vBoxes[0];
                double fU = (Box[0] + (Box[2] - Box[0]) / 2) * nCol;
                double fV = (Box[1] + (Box[3] - Box[1]) / 2) * nRow;
                //Mat ImgLeft(Size(nCol, nRow), CV_8UC3, Scalar(0, 0, 0));
                Rect Roi(round(Box[0] * nCol),
                    round(Box[1] * nRow),
                    round((Box[2] - Box[0]) * nCol),
                    round((Box[3] - Box[1]) * nRow));
                std::cout << "Rect of Left:" << round(Box[0] * nCol) << "\t" <<
                    round(Box[1] * nRow) << "\t" <<
                    round((Box[2] - Box[0]) * nCol) << "\t" <<
                    round((Box[3] - Box[1]) * nRow) << "\t" << endl;
                Mat ImgLeft = image(Roi);
                //cvtColor(ImgLeft, ImgLeft, COLOR_RGB2BGR);
                if (Display)
                {
                    imshow("imgleft", ImgLeft);
                    waitKey(0);
                }
                imwrite(ImgDir + "left_roi.jpg", ImgLeft);
                //imshow("1", Region);
                //waitKey(0);
                //Region.copyTo(ImgLeft(Roi));

                ////just for test
                //ImgLeft = Region;


                //move robot 10mm along x-aixs
                vector<double> vCurPoint(6);
                vCurPoint = rtde_receive.getActualTCPPose();
                vCurPoint[0] -= 0.01; //unit: m
                if (IsSafeRange(vCurPoint))
                {
                    rtde_control.moveL(vCurPoint, 0.05, 0.25, true);
                    Sleep(2000);
                    WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");

                    std::cout << "Moving to the Right Position" << endl;

                }
                else
                {
                    std::cout << "Position:";
                    PrintVct(vCurPoint);
                    std::cout << "Out of Range!" << endl;
                    exit(0);
                }


                //get time stamp
                TimeStamp = time(NULL);
                ImgName = to_string(time(&TimeStamp));
                ImgPath = ImgDir + ImgName + ".png";
                ImgBakPath = ImgDir + ImgName + "_bak.png";


                //collect the right image

                //for (int i = 0; i < 30; i++)
                //{
                //    frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
                //    frame color = data.get_color_frame().apply_filter(color_map);
                //}
                //Sleep(30);
                //frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
                //frame color = data.get_color_frame().apply_filter(color_map);
                //const int w = color.as<video_frame>().get_width();
                //const int h = color.as<video_frame>().get_height();
                //Mat ImgRight(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
                
                Mat ImgRight = ConvertUCharToMat_3Bit(&(ucImgBuf[0]), width, height);

                cv::cvtColor(ImgRight, ImgRight, COLOR_RGB2BGR);
                std::cout << "Right Image Collected" << endl;
                imwrite(ImgDir + "right.jpg", ImgRight);

                Rect Cut(0,
                    round(Box[1] * ImgRight.rows),
                    ImgRight.cols,
                    round((Box[3] - Box[1]) * ImgRight.rows));
                ImgRight = ImgRight(Cut);
                imwrite(ImgDir + "right_roi.jpg", ImgRight);
                if (Display)
                {
                    imshow("imgright", ImgRight);
                    waitKey(0);
                }

                //match and get depth
                double fZ = GetDepth(ImgLeft, ImgRight, round(Box[0] * nCol));
                std::cout << "Flower Depth: " << fZ << "mm" << endl;

                vector<double> vCPoint = Image2Camera(fU, fV, fZ);


                //get robot position
                //todo: move to the left position before capture 
                //todo: change camera to robot
                vCurPoint = rtde_receive.getActualTCPPose();
                vCurPoint[0] += 0.01;
                rtde_control.moveL(vCurPoint, 0.05, 0.25, true);
                Sleep(2000);
                WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");

                std::cout << "Moving Back to Left" << endl;


                vector<double> vCamPoint = Camera2Robot(vCurPoint);

                //calculate the world coordinate of flower
                vector<double> vWPoint = Camera2World(vCPoint, vCurPoint);

                //decide whether move or operate
                double fDistance = GetDistance(vWPoint, vCamPoint);
                std::cout << "Flower Distance: " << fDistance << endl;
                if (fDistance < 0.3)
                {

                    //make sure the image is in the center of camera
                    // 
                    //get time stamp
                    TimeStamp = time(NULL);
                    ImgName = to_string(time(&TimeStamp));
                    ImgPath = ImgDir + ImgName + ".png";
                    ImgBakPath = ImgDir + ImgName + "_bak.png";

                    //for (int i = 0; i < 30; i++)
                    //{
                    //    frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
                    //    frame color = data.get_color_frame().apply_filter(color_map);
                    //}
                    ////collect the image
                    //Sleep(30);
                    //frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
                    //frame color = data.get_color_frame().apply_filter(color_map);
                    //const int w = color.as<video_frame>().get_width();
                    //const int h = color.as<video_frame>().get_height();
                    //Mat ImgCur(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

                    Mat ImgCur = ConvertUCharToMat_3Bit(&(ucImgBuf[0]), width, height);


                    cv::cvtColor(ImgCur, ImgCur, COLOR_RGB2BGR);
                    std::cout << "Collecting Image" << endl;
                    imwrite(ImgPath, ImgCur);
                    imwrite(ImgBakPath, ImgCur);
                    imwrite(ImgDir + "right.jpg", ImgCur);
                    //if (Display)
                    //{
                    //    imshow("imgcur", ImgCur);
                    //    waitKey(0);
                    //}

                    //read detection result
                    while (!IsFileExist(JsonDir + ImgName + "_bak.json"))
                    {
                        Sleep(100);
                    }
                    std::cout << "Dectecion Received" << endl;

                    //the detections have been sorted by score
                    vBoxes.clear();
                    vMasks.clear();
                    vClasses.clear();
                    vScores.clear();

                    ReadDetection(JsonDir + ImgName + "_det.json", vBoxes, vMasks, vClasses, vScores);

                    if (vScores.size() == 0)
                    {
                        std::cout << "No Flower Detected" << endl;

                        //move robot
                        //
                    }
                    else
                    {
                        std::cout << "Flower Detected" << endl;
                        //read the first box, keep the data in box and remove the remaining
                        //TODO: mask sure the box not on the bounding of the whole image left, so that the box must be in image right;
                        vector<double> Box = vBoxes[0];
                        double fU = (Box[0] + (Box[2] - Box[0]) / 2) * nCol;
                        double fV = (Box[1] + (Box[3] - Box[1]) / 2) * nRow;
                        std::cout << "Flower Postion in Image:" << fU << "\t" << fV << endl;
                        if (abs(fU - nCol / 2) > 30)
                        {
                            std::cout << "AdjustING FU" << endl;
                            vCurPoint = rtde_receive.getActualTCPPose();
                            vCurPoint[0] -= (fU - 645.67) * fZ / 909.53 / 1000; //unit: m
                            if (IsSafeRange(vCurPoint))
                            {
                                rtde_control.moveL(vCurPoint, 0.05, 0.25, true);
                                Sleep(2000);
                                WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");

                                std::cout << "Collecting the Right Image" << endl;

                            }
                            else
                            {
                                std::cout << "Position:";
                                PrintVct(vCurPoint);
                                std::cout << "Out of Range!" << endl;
                                exit(0);
                            }
                        }

                        if (abs(fV - nRow / 2) > 30)
                        {
                            std::cout << "Adjusting fV" << endl;
                            vCurPoint = rtde_receive.getActualTCPPose();
                            vCurPoint[2] -= (fV - 355.99) * fZ / 913.03 / 1000; //unit: m
                            if (IsSafeRange(vCurPoint))
                            {
                                rtde_control.moveL(vCurPoint, 0.05, 0.25, true);

                                Sleep(2000);
                                WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");


                            }
                            else
                            {
                                std::cout << "Position:";
                                PrintVct(vCurPoint);
                                std::cout << "Out of Range!" << endl;
                                exit(0);
                            }
                        }

                    }


                    // ready to operate

                    vCurPoint = rtde_receive.getActualTCPPose();
                    vCamPoint = Camera2Robot(vCurPoint);
                    vCamPoint[1] -= (fZ - brushlen) / 1000;
                    vCamPoint[2] -= 0.01;


                    if (IsSafeRange(vCamPoint))
                    {
                        rtde_control.moveL(vCamPoint, 0.05, 0.25, true);
                        Sleep(5000);
                        WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");

                        std::cout << "Ready to Operate" << endl;
                    }
                    else
                    {
                        std::cout << "Position:";
                        PrintVct(vCamPoint);
                        std::cout << "Out of Range!" << endl;
                        exit(0);
                    }
                    imshow("imgcur", ImgCur);
                    waitKey(0);

                    //a servoing
                    vCurPoint = rtde_receive.getActualTCPPose();

                    double velocity = 0.01;
                    double acceleration = 0.2;
                    double blend_1 = 0.00;
                    double blend_2 = 0.003;
                    double blend_3 = 0.00;
                    double r = 0.005;
                    double step_f = 0.5;
                    std::vector<std::vector<double>> path;

                    for (int i = 0; i < 10; i++)
                    {
                        std::vector<double> path_pose1 = { vCurPoint[0], vCurPoint[1] - i * r * step_f, vCurPoint[2] + r, vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        std::vector<double> path_pose2 = { vCurPoint[0] + r, vCurPoint[1] - i * r * step_f, vCurPoint[2], vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        std::vector<double> path_pose3 = { vCurPoint[0], vCurPoint[1] - i * r * step_f, vCurPoint[2] - r, vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        std::vector<double> path_pose4 = { vCurPoint[0] - r, vCurPoint[1] - i * r * step_f, vCurPoint[2], vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        std::vector<double> path_pose5 = { vCurPoint[0], vCurPoint[1] - i * r * step_f, vCurPoint[2] + 2 * r, vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        std::vector<double> path_pose6 = { vCurPoint[0] + 2 * r, vCurPoint[1] - i * r * step_f, vCurPoint[2], vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        std::vector<double> path_pose7 = { vCurPoint[0], vCurPoint[1] - i * r * step_f, vCurPoint[2] - 2 * r, vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        std::vector<double> path_pose8 = { vCurPoint[0] - 2 * r, vCurPoint[1] - i * r * step_f, vCurPoint[2], vCurPoint[3], vCurPoint[4], vCurPoint[5], velocity, acceleration, r * 0.6 };
                        path.push_back(path_pose1);
                        path.push_back(path_pose2);
                        path.push_back(path_pose3);
                        path.push_back(path_pose4);
                        path.push_back(path_pose5);
                        path.push_back(path_pose6);
                        path.push_back(path_pose7);
                        path.push_back(path_pose8);

                    }

                    tPEnd = clock();
                    double dPTime = (double)(tPEnd - tPBegin) / CLOCKS_PER_SEC;

                    // write time to file

                    ofstream fp;
                    fp.open(DataDir + "pollination_time.txt", ofstream::app);
                    fp << (double)dPTime << endl;
                    fp.close();
                    //


                    rtde_control.moveL(path, true);



                    //vector<double> vOPoint(6);
                    //for (int i = 0; i < 6; i++)
                    //{
                    //    vOPoint[i] = vCurPoint[i];
                    //}
                    //vCurPoint[0] -= 0.02;
                    //vCurPoint[2] -= 0.02;

                    //for (int i = 0; i < 3; i++)
                    //{   
                    //    vCurPoint[1] -= 0.01;
                    //    for (int j = 0; j < 3; j++)
                    //    {
                    //        vCurPoint[0] += 0.01;
                    //        for (int k = 0; k < 3; k++)
                    //        {
                    //            vCurPoint[2] += 0.01;
                    //            rtde_control.moveL(vCurPoint, 0.02, 0.25, true);
                    //            Sleep(2000);

                    //            vector<double> vCurJoint = rtde_receive.getActualQ();
                    //            vCurJoint[5] += 1.0;
                    //            rtde_control.moveJ(vCurJoint);
                    //            Sleep(1000);
                    //            vCurJoint[5] -= 1.0;
                    //            rtde_control.moveJ(vCurJoint);
                    //            Sleep(1000);

                    //            rtde_control.moveL(vOPoint, 0.05, 0.25, true);
                    //            Sleep(2000);

                    //            

                    //        }
                    //        vCurPoint[2] -= 0.03;
                    //    }
                    //    vCurPoint[0] -= 0.03;
                    //}

                    //start vision servoing
                    //while (1)
                    //{
                    //    //get time stamp
                    //    TimeStamp = time(NULL);
                    //    ImgName = to_string(time(&TimeStamp));
                    //    ImgPath = ImgDir + ImgName + ".png";
                    //    ImgBakPath = ImgDir + ImgName + "_bak.png";

                    //    for (int i = 0; i < 30; i++)
                    //    {
                    //        frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
                    //        frame color = data.get_color_frame().apply_filter(color_map);
                    //    }
                    //    //collect the right image
                    //    Sleep(30);
                    //    data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
                    //    color = data.get_color_frame().apply_filter(color_map);
                    //    //const int w = color.as<video_frame>().get_width();
                    //    //const int h = color.as<video_frame>().get_height();
                    //    ImgCur = Mat(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);
                    //    cv::cvtColor(ImgCur, ImgCur, COLOR_RGB2BGR);
                    //    imwrite(ImgPath, ImgCur);
                    //    imwrite(ImgBakPath, ImgCur);
                    //    std::cout << "Collecting Image" << endl;
                    //    if (Display)
                    //    {
                    //        imshow("curimg", ImgCur);
                    //        waitKey(0);
                    //    }
                    //    //imwrite(ImgDir + "right.jpg", ImgCur);

                    //    //read detection result
                    //    while (!IsFileExist(JsonDir + ImgName + "_bak.json"))
                    //    {
                    //        Sleep(100);
                    //    }
                    //    std::cout << "Dectecion Received" << endl;

                    //    //the detections have been sorted by score
                    //    vBoxes.clear();
                    //    vMasks.clear();
                    //    vClasses.clear();
                    //    vScores.clear();

                    //    ReadDetection(JsonDir + ImgName + "_det.json", vBoxes, vMasks, vClasses, vScores);

                    //    if (vScores.size() == 0)
                    //    {
                    //        std::cout << "No Flower Detected" << endl;

                    //        //move robot
                    //        //
                    //    }
                    //    else
                    //    {
                    //        std::cout << "Flower Detected" << endl;
                    //        //read the first box, keep the data in box and remove the remaining
                    //        //TODO: mask sure the box not on the bounding of the whole image left, so that the box must be in image right;
                    //        vector<double> Box = vBoxes[0];
                    //        double fU = (Box[0] + (Box[2] - Box[0]) / 2) * nCol;
                    //        double fV = (Box[1] + (Box[3] - Box[1]) / 2) * nRow;
                    //        std::cout << "Servoing" << endl;
                    //        if (818 - fU > 30 || 409 - fV > 30)
                    //        {
                    //            vCurPoint = rtde_receive.getActualTCPPose();
                    //            vCurPoint[1] -= 0.005;
                    //            if (IsSafeRange(vCurPoint))
                    //            {
                    //                rtde_control.moveL(vCurPoint, 0.05, 0.25, true);
                    //                Sleep(2000);
                    //                std::cout << "Approaching" << endl;
                    //            }
                    //            else
                    //            {
                    //                std::cout << "Position:";
                    //                PrintVct(vCamPoint);
                    //                std::cout << "Out of Range!" << endl;
                    //                exit(0);
                    //            }
                    //        }
                    //        else
                    //        {
                    //            std::cout << "I DID IT" << endl;
                    //            exit(0);
                    //        }
                    //        
                    //    }
                    //}
                    //Sleep(1000000);
                    while (!IsReach(rtde_receive.getActualTCPPose(), path.back()) && Flag)
                    {
                        if (_kbhit())
                        {
                            int ch = _getch();
                            std::cout << ch << std::endl;;
                            if (ch == 115) //press "s" to interupt
                            {
                                rtde_control.stopL();
                                Flag = false;
                            }
                        }
                        else
                        {
                            Sleep(200);
                        }
                    }
                    Flag = false;


                }
                else
                {
                    //move
                    vector<double> vDstPoint(6);
                    vDstPoint[0] = vWPoint[0];
                    vDstPoint[1] = vCamPoint[1] + 0.2 * (vWPoint[1] - vCamPoint[1]);
                    vDstPoint[2] = vWPoint[2];

                    //for (int i = 0; i < 3; i++)
                    //{
                    //    vDstPoint[i] = 0.5 * (vWPoint[i] + vCamPoint[i]);
                    //}
                    for (int i = 3; i < 6; i++)
                    {
                        vDstPoint[i] = vCamPoint[i];
                    }
                    PrintVct(vDstPoint);

                    vDstPoint = Robot2Camera(vDstPoint);
                    //PrintVct(vWPoint);
                    //PrintVct(vCamPoint);
                    //PrintVct(vDstPoint);

                    if (IsSafeRange(vDstPoint))
                    {
                        rtde_control.moveL(vDstPoint, 0.05, 0.25, true);
                        std::cout << "Moving to Mid" << endl;
                        Sleep(8000);
                        WriteTrackPoint(rtde_receive.getActualTCPPose(), PathDir + "path.txt");

                    }
                    else
                    {
                        std::cout << "Position:";
                        PrintVct(vDstPoint);
                        std::cout << "Out of Range!" << endl;
                        exit(0);
                    }
                }
            }

        }
    }

    return;
}
