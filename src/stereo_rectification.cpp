#include <math.h>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"

using namespace cv;
using namespace std;



// 
//�������������Ѳ������ڲξ��󡢻���ϵ��������R��T��ֱ�Ӵ�������������������Ļ����������������������
//void stereo_rectification(const cv::Mat& img_L, const cv::Mat& img_R,
//                          Mat& image_l_rected,Mat& image_r_rected) {
//                          
  void stereo_rectification(const cv::Mat& img_L,const cv::Mat& img_R,
                          const cv::Mat& KL,const cv::Mat& KR,
                          const cv::Mat& DL, const cv::Mat& DR,
                          const cv::Mat& R,const cv::Mat& T,
                          cv::Mat& image_l_rected,cv::Mat& image_r_rected) {////����ԭͼ���ڲξ���K(˫Ŀ�궨���)������ϵ������D��R��T��������������
                          
                         

  /*double K_left[3][3] = {662.3562273563088,
                         0,
                         312.6263091035918,
                         0,
                         662.9296902690498,
                         258.9996285827844,
                         0,
                         0,
                         1};
  Mat KL = cv::Mat(3, 3, cv::DataType<double>::type, K_left);

  double d_left[1][5] = {0.06966962838870275, 0.02054263655123773,
                         0.001252584212211826, 0.002089077085379777,
                         -0.4096320330693385};
  Mat DL = cv::Mat(1, 5, cv::DataType<double>::type, d_left);

  double K_right[3][3] = {647.3402626821477,
                          0,
                          298.8766921846282,
                          0,
                          647.739941990085,
                          259.9519313557778,
                          0,
                          0,
                          1};
  Mat KR = cv::Mat(3, 3, cv::DataType<double>::type, K_right);

  double d_right[1][5] = {0.02272045036297292, -0.5565235313790773,
                          0.007380600944678045, -0.007607934580409265,
                          1.30443069011335};
  Mat DR = cv::Mat(1, 5, cv::DataType<double>::type, d_right);

  double R_stereo[3][3] = {
      0.9997826605620699,    0.004051019540949904,  0.02045044938645102,
      -0.004074743115292153, 0.999991072656389,     0.001118515117933302,
      -0.02044573569166274,  -0.001201602348328341, 0.9997902420227071};
  Mat R = cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);

  Vec3d T = {-0.0676242, -0.0119106, -0.0116169};*/

 /* cout << "˫Ŀ���������" << endl;
  cout << KL << endl;
  cout << KR << endl;
  cout << DL << endl;
  cout << DR << endl;
  cout << R << endl;
  cout << T << endl;*/

  // R1 �C Output 3x3 rectification transform (rotation matrix) for the first camera.
  // R2 �C Output 3x3 rectification transform (rotation matrix) for the second camera. 
  // P1 �C Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera. 
  // P2 �C Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera. 
  // Q �C Output disparity-to-depth mapping matrix
  cv::Mat RL, RR, PL, PR, Q;
  stereoRectify(KL, DL, KR, DR, img_L.size(), R, T, RL, RR, PL, PR, Q);

 /* cout << "��һ��˫ĿУ�������" << endl << endl;
  cout << RL << endl;
  cout << PL << endl;
  cout << RR << endl;
  cout << PR << endl;*/

  // ����ͼ��У��
  cv::Mat lmapx, lmapy, rmapx, rmapy;

  //-----------------------------
  //cv::Mat imgU1, imgU2;//��Щ����Ӧ���ҷŵ������������ˣ�
  //---------------------------

  cv::initUndistortRectifyMap(KL, DL, RL, PL, img_L.size(), CV_32F, lmapx,
                              lmapy);
  cv::initUndistortRectifyMap(KR, DR, RR, PR, img_R.size(), CV_32F, rmapx,
                              rmapy);
  cv::remap(img_L, image_l_rected, lmapx, lmapy, cv::INTER_LINEAR);
  cv::remap(img_R, image_r_rected, rmapx, rmapy, cv::INTER_LINEAR);

  cv::imshow("left.jpg", image_l_rected);
  cv::imshow("right.jpg", image_r_rected);
  cv::waitKey(0);

}