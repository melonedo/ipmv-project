#include <math.h>

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"

using namespace Eigen;
using namespace cv;
using namespace std;

void preset_steroparams(cv::Mat& R, cv::Mat& T, cv::Mat& K_L, cv::Mat& K_R,
                        cv::Mat& D1, cv::Mat& D2) {
  //代码计算结果
  /*double d_left[1][5] = {-0.2476308740918039, 0.1428984605799336,
 -0.007308380442553203, 0.01834444017828064, -0.1575561255791122}; Mat D1 =
 cv::Mat(1, 5, cv::DataType<double>::type, d_left);

    double d_right[1][5] = {-0.3123616831963305, 0.7492919242979774,
 -0.005600943091007264, 0.01601327051434557, -2.391415265128237};

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
   Vec3d T = {-65.0649, -0.139223, 27.6227};*/

  // matlab计算
  double d_left[1][5] = {-0.485164739871447, 0.552024900666525, 0, 0,
                         -0.336674278642270};
  D1 = Mat(1, 5, DataType<double>::type, d_left);
  double d_right[1][5] = {-0.424674246902221, 0.00151675029453053, 0, 0,
                          0.639005725201348};
  D2 = Mat(1, 5, DataType<double>::type, d_right);
  double K_left[3][3] = {
      1388.19773824506, 0, 0, 0, 1389.01737025501, 0, 590.918909086584,
      360.567076302685, 1};
  K_L = Mat(3, 3, DataType<double>::type, K_left);
  double K_right[3][3] = {
      1386.15194480925, 0, 0, 0, 1390.06465431073, 0, 631.010280569678,
      392.663204621441, 1};
  K_R = Mat(3, 3, DataType<double>::type, K_right);
  double R_stereo[3][3] = {
      0.999971235342412,    -0.00173830599783834, 0.00738287071727767,
      0.00167481635173865,  0.999961641851747,    0.00859708179243690,
      -0.00739753188287186, -0.00858446954772283, 0.999935789640828};
  R = Mat(3, 3, DataType<double>::type, R_stereo);
  T = Mat{-61.7483197328276, 3.73154432852811, 1.76558076147146};
}

void stereo_rectification(const Mat& img_L, const Mat& img_R, const Mat& R,
                          const Mat& T, const Mat& K_L, const Mat& K_R,
                          const Mat& D1, const Mat& D2, Mat& image_l_rected,
                          Mat& image_r_rected) {
  //对照组代码-----------------------------------------------------------------------------
  /*cv::Mat R_l, R_r, P1, P2, Q;
  stereoRectify(K_L, D1, K_R, D2, img_L.size(), R, T, R_l, R_r, P1, P2, Q);*/
  //-----------------------------------------------------------------------------------------

  //将cv传递过来的矩阵转为eigen能用的---------
  Matrix3d r;
  Vector3d T1;
  Matrix3d KL;
  Matrix3d KR;
  cv2eigen(R, r);
  cv2eigen(T, T1);
  cv2eigen(K_L, KL);
  cv2eigen(K_R, KR);
  //------------------------------------------
  //
  // Rrect计算--------------------
  double TT;
  Matrix<double, 3, 3> temp;
  Matrix<double, 3, 3> Rrect;  //旋转矫正
  Matrix<double, 3, 3> R1;     //左旋转矫正
  Matrix<double, 3, 3> R2;     //右旋转矫正
  Vector3d I;                  // 0,0,1
  Vector3d E1;                 // Rrect的三个向量
  Vector3d E2;                 // Rrect的三个向量
  Vector3d E3;                 // Rrect的三个向量

  I = {0, 0, 1};
  TT = T1.norm();  //范数计算
  E1 = T1 / TT;
  E2 = I.cross(T1) / (I.cross(T1)).norm();
  E3 = E1.cross(E2);
  temp << E1, E2, E3;
  Rrect = temp.transpose();
  R1 = Rrect;
  R2 = r * Rrect;
  //-----------------------------

  //投影矩阵P计算-------------------------------
  Matrix<double, 3, 4> PL;
  Matrix<double, 3, 4> PR;
  Vector3d ZERO;
  Matrix<double, 3, 4> M1_L;
  Matrix<double, 3, 4> M1_R;
  Matrix<double, 4, 4> M2;
  Matrix<double, 3, 4> top;
  Matrix<double, 1, 4> buttom;

  ZERO = {0, 0, 0};
  buttom = {0, 0, 0, 1};
  M1_L << KL, ZERO;  //扩充内参矩阵KL，变为M1_L
  M1_R << KR, ZERO;  //扩充内参矩阵KR,，变为M1_R
  top << r, T1;
  M2 << top,
      buttom;      //制作外参矩阵M2
  PL = M1_L * M2;  //投影矩阵计算，M1*M2，内参乘以外参
  PR = M1_R * M2;
  //----------------------------------------------
  //
  //将eigen的计算结果转为cv能用的--------------------------------------
  cv::Mat lmapx, lmapy, rmapx, rmapy;
  cv::Mat r1, r2, p1, p2;
  eigen2cv(R1, r1);
  eigen2cv(R2, r2);
  eigen2cv(PL, p1);
  eigen2cv(PR, p2);
  //-------------------------------------------------------------------

  //最后的投影-----------------------

  cv::initUndistortRectifyMap(K_L, D1, r1, p1, img_L.size(), CV_32F, lmapx,
                              lmapy);
  cv::initUndistortRectifyMap(K_R, D2, r2, p2, img_R.size(), CV_32F, rmapx,
                              rmapy);
  //对照组
  /*cv::initUndistortRectifyMap(K_L, D1, R_l, P1, img_L.size(), CV_32F, lmapx,
                              lmapy);
  cv::initUndistortRectifyMap(K_R, D2, R_r, P2, img_R.size(), CV_32F, rmapx,
                              rmapy);*/
  cv::remap(img_L, image_l_rected, lmapx, lmapy, cv::INTER_LINEAR);
  cv::remap(img_R, image_r_rected, rmapx, rmapy, cv::INTER_LINEAR);
  cv::imshow("left.jpg", image_l_rected);
  cv::imshow("right.jpg", image_r_rected);
  cv::waitKey(0);
}
