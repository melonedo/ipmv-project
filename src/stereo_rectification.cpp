#include <math.h>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"

using namespace Eigen;
using namespace cv;
using namespace std;



// //void stereo_rectification(const cv::Mat& img_L,const cv::Mat& img_R,
  //                        const cv::Mat& KL,const cv::Mat& KR,
  //                        const cv::Mat& DL, const cv::Mat& DR,
  //                        const cv::Mat& R,const cv::Mat& T,
  //                        cv::Mat& image_l_rected,cv::Mat& image_r_rected) {////两张原图，内参矩阵K(双目标定获得)，畸变系数向量D，R、T，最后两个是输出
//！！！如果你想把参数（内参矩阵、畸变系数向量、R，T）直接存在这个程序里，不用输入的话，用下面这段声明！！！
void stereo_rectification(const cv::Mat& img_L, const cv::Mat& img_R, Mat& image_l_rected, cv::Mat& image_r_rected) {                       
  

    //�������
 double d_left[1][5] = {-0.2476308740918039, 0.1428984605799336, -0.007308380442553203, 0.01834444017828064, -0.1575561255791122};
   Mat D1 = cv::Mat(1, 5, cv::DataType<double>::type, d_left);

   double d_right[1][5] = {-0.3123616831963305, 0.7492919242979774, -0.005600943091007264, 0.01601327051434557, -2.391415265128237};
     
   Mat D2 = cv::Mat(1, 5, cv::DataType<double>::type, d_right);
    
  double K_left[3][3] = {995.5872941102477, 0, 507.6322439538547,
 0, 993.1761254348543, 391.3618985410175,
 0, 0, 1
};
  Mat K_L = cv::Mat(3, 3, cv::DataType<double>::type, K_left); 
  double K_right[3][3] = {1059.943028057779, 0, 557.0605532130172,
 0, 1058.429107553662, 406.6782374209959,
 0, 0, 1
};
  Mat K_R = cv::Mat(3, 3, cv::DataType<double>::type, K_right);
  double R_stereo[3][3] = {
      0.9999592144503071, 0.001226048321462626, -0.0089479741527445,
 -0.001140372433315718, 0.9999535206231106, 0.009573721541786621,
 0.00895929610170791, -0.009563127049233127, 0.9999141351208124
};
  Mat  R= cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);
  Vec3d T = {-65.0649, -0.139223, 27.6227};

    //matlab����
  /*double d_left[1][5] = {-0.485164739871447,0.552024900666525,0,0,-0.336674278642270};                         
  Mat D1 = cv::Mat(1, 5, cv::DataType<double>::type, d_left);
  double d_right[1][5] = {-0.424674246902221, 0.00151675029453053, 0 ,0, 0.639005725201348};
  Mat D2 = cv::Mat(1, 5, cv::DataType<double>::type, d_right);
  double K_left[3][3] = {
      1388.19773824506, 0, 0, 0, 1389.01737025501, 0, 590.918909086584,
      360.567076302685, 1};
  Mat K_L = cv::Mat(3, 3, cv::DataType<double>::type, K_left);
  double K_right[3][3] = {
      1386.15194480925, 0, 0, 0, 1390.06465431073, 0, 631.010280569678,
      392.663204621441, 1};
  Mat K_R = cv::Mat(3, 3, cv::DataType<double>::type, K_right);
  double R_stereo[3][3] = {
      0.999971235342412,-0.00173830599783834,0.00738287071727767,0.00167481635173865,0.999961641851747,0.00859708179243690,-0.00739753188287186,- 0.00858446954772283,0.999935789640828};   
  Mat R = cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);
  Vec3d T = {-61.7483197328276,3.73154432852811,1.76558076147146};*/
  
  //用cv自带函数进行校正，对照组
  
   
   
  const size_t Row = img_L.size[0];//1080
  const size_t Col = img_L.size[1];//1920
    
  //Mat R, E, F;
  //vector<Mat> tvecsMat; /* 每幅图像的旋转向�?*/
  //vector<Mat> rvecsMat;      
  //Mat K_L ;
  //Mat K_R ;
  //Mat D1, D2;
  //Vec3d T;
  //Mat gray_L, gray_R;

  //vector<cv::Point3f> objectpoint;
  //vector<vector<cv::Point3f>> objpoint;
  //// image_points1、imagePoints2中每个元素都是一个小vector，每个小vector存储的每个元素都是opencv的cv::Point2f数据结构
  //vector<vector<Point2f> > imagePoints1, imagePoints2;
  //vector<Point2f> corner_L, corner_R;

  //int board_Row = 7;
  //int board_Col = 11;
  //int squaresize = 30;
  //Size boardsize = Size(board_Col, board_Row);
  //cvtColor(img_L, gray_L, CV_BGR2GRAY);
  //cvtColor(img_R, gray_R, CV_BGR2GRAY);
  //bool foundL=0, foundR=0;
  //foundL = findChessboardCorners(img_L, boardsize, corner_L);
  //foundR = findChessboardCorners(img_R, boardsize, corner_R);
  //cornerSubPix(gray_L, corner_L, cv::Size(5, 5), cv::Size(-1, -1),TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
  //cornerSubPix(gray_R, corner_R, cv::Size(5, 5), cv::Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
  ///*drawChessboardCorners(img_L, boardsize, corner_L, foundL);
  //cv::imshow("L", img_L);
  //cv::waitKey(1);
  //drawChessboardCorners(img_R, boardsize, corner_R, foundR);
  //cv::imshow("R", img_R);
  //cv::waitKey(10);*/
  //for (int i = 0; i < board_Row; i++) {
  //  for (int j = 0; j < board_Col; j++) {
  //    objectpoint.push_back(cv::Point3f(i * squaresize, j * squaresize, 0.0f));  
  //  }
  //}
  //objpoint.push_back(objectpoint);
  //imagePoints1.push_back(corner_L);
  //imagePoints2.push_back(corner_R);  
  //calibrateCamera(objpoint, imagePoints1, img_L.size(), K_L, D1, rvecsMat,tvecsMat, 0);       
  //calibrateCamera(objpoint, imagePoints2, img_R.size(), K_R, D2, rvecsMat,tvecsMat, 0);
  //stereoCalibrate(objpoint, imagePoints1, imagePoints2, K_L, D1, K_R, D2, img_L.size(), R,T, E, F,CALIB_USE_INTRINSIC_GUESS,cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,1e-6));
  //    




  /*cv::Mat R_l, R_r, P1, P2, Q;
  stereoRectify(K_L, D1, K_R, D2, img_L.size(), R, T, R_l, R_r, P1, P2, Q);*///������
           

  cout << "KL=" << K_L << endl;
  cout << "KR=" << K_R << endl;
  cout << "D1=" <<D1 << endl;
  cout << "D2=" << D2 << endl;
  cout << "R=" << R << endl;
  cout << "T=" << T << endl;
  
                 
               
  Matrix3d r; 
  Vector3d T1;
  Matrix3d KL;//
  Matrix3d KR;//
  cv2eigen(R,r);
  cv2eigen(T, T1);
  cv2eigen(K_L, KL);
  cv2eigen(K_R, KR);
   
  Vector3d I(0, 0, 1);
  double TT = T1.norm();
  Vector3d E1 = T1 / TT;
  Vector3d E2 = I.cross(T1) / (I.cross(T1)).norm();
  Vector3d E3 = E1.cross(E2);
  Matrix<double, 3, 3> temp;
  Matrix<double, 3, 3> Rrect;
  temp << E1, E2, E3;
  Rrect = temp.transpose();
  Matrix<double, 3, 3> R1 = Rrect;
  Matrix<double, 3, 3> R2 = r*Rrect;
  
  Matrix<double, 3, 4> PL;
  Matrix<double, 3, 4> PR;
  Vector3d ZERO(0, 0, 0);
  Matrix<double, 3, 4> M1_L;
  Matrix<double, 3, 4> M1_R;
  Matrix<double, 4, 4> M2;
  Matrix<double, 3, 4> top;
  Matrix<double, 1, 4> buttom(0,0,0,1);
  
  M1_L << KL, ZERO;
  M1_R << KR,ZERO;
  top << r, T1;
  M2 << top,
        buttom;
  PL = M1_L * M2;
  PR = M1_R * M2;
  cout << "PR=" << PR << endl;     
  cv::Mat r1, r2, p1, p2;
  eigen2cv(R1, r1);
  eigen2cv(R2, r2);
  eigen2cv(PL, p1);
  eigen2cv(PR, p2);
  cv::Mat lmapx, lmapy, rmapx, rmapy;
  
  cv::initUndistortRectifyMap(K_L, D1, r1, p1, img_L.size(), CV_32F, lmapx,
                              lmapy);
  cv::initUndistortRectifyMap(K_R, D2, r2, p2, img_R.size(), CV_32F, rmapx,
                              rmapy);
  //������
  /*cv::initUndistortRectifyMap(K_L, D1, R_l, P1, img_L.size(), CV_32F, lmapx,
                              lmapy);
  cv::initUndistortRectifyMap(K_R, D2, R_r, P2, img_R.size(), CV_32F, rmapx,
                              rmapy);*/
  cv::remap(img_L, image_l_rected, lmapx, lmapy, cv::INTER_LINEAR);
  cv::remap(img_R, image_r_rected, rmapx, rmapy, cv::INTER_LINEAR);
  cv::imshow("left.jpg", image_l_rected);
  cv::imshow("right.jpg", image_r_rected);
  cv::waitKey(0);    
    
 

  //    //测试用代�?
  //    //------------------------------------------------------------
  

   //#pragma omp parallel for
  // for (int x = 1 ; x < Row ; x++) {
  //  for (int y = 1 ; y < Col ; y++) {
  //    locat = {double(x), double(y), 1};
  //    locat_l = KL * R1  * KL.inverse()* locat;
  //    locat_r = KR * R2  * KR.inverse()* locat;
  //    /*locat_l = R1 * locat;
  //    locat_r = R2 * locat;*/
  //    /*double kl = f / locat_l[2];
  //    double kr = f / locat_r[2];*/
  //    //或�?
  //    /*Vector3d HL = KL * R1 * locat_l;
  //    Vector3d HR = KR * R2 * locat_r;*/

  //    /*cout << " K_L * R1 " << K_L * R1 * locat_l<< endl;
  //    cout << " K_R * R2 " << K_R * R2 * locat_r << endl;*/
  //    /*Vector3d templ = K_L * R1 * locat_l ;
  //    Vector3d tempr = K_R * R2 * locat_r;
  //    
   

  //    if (templ[2] > max) {
  //      max = templ[2];
  //    }
  //    if (templ[2] < min) {
  //      min = templ[2];
  //    }*/
  //    /*n++;*/
  //    /*std::cout << "n=" << n << std::endl;*/
  //    /*if (locat_l[1] > max) {
  //      max = locat_l[1];
  //    }
  //    if  (locat_l[1] < min) {
  //      min = locat_l[1];
  //    }*/
  //    /*
  //    x(1300 839018)
  //    y(-168455 1238593)
  //    z(-231 0)
  //    */
  //    /*std::cout << "N0." << n << ":" << locat_l[2] << std::endl;
  //    std::cout << "N0." << n << ":" <<locat_l[2] << std::endl;*/
  //    //-----------------------------------------------------------------------
  //    for (int v = 0; v < 3; v++) {
  //      image_l_rected.at<cv::Vec3b>(locat_l[0]/2, locat_l[1]/2)[v] = img_L.at<cv::Vec3b>(x, y)[v];//这里的abs（x,y/10000）只是用来暂时防止报�?
  //      image_r_rected.at<cv::Vec3b>(locat_r[0]/2, locat_r[1]/2)[v] = img_R.at<cv::Vec3b>(x, y)[v];//这里的赋值语句会导致溢出,原因已经查明，因为不是像素坐标，是毫米坐标的原因       
  //        //image_l_rected.at<cv::Vec3b>(x, y)[v] = img_L.at<cv::Vec3b>(x, y)[v];//测试代码
  //        //image_r_rected.at<cv::Vec3b>(x, y)[v] = img_R.at<cv::Vec3b>(x, y)[v];
  //                                   
  //          
  //    }
  //  }
  //  
  //}
  
 
} 

  