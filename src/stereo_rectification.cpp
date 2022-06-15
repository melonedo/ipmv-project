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
  //                        cv::Mat& image_l_rected,cv::Mat& image_r_rected) {////����ԭͼ���ڲξ���K(˫Ŀ�궨���)������ϵ������D��R��T��������������
//�������������Ѳ������ڲξ��󡢻���ϵ��������R��T��ֱ�Ӵ�������������������Ļ����������������������
void stereo_rectification(const cv::Mat& img_L, const cv::Mat& img_R, Mat& image_l_rected, cv::Mat& image_r_rected) {                       
  
  double K_left[3][3] = {662.3562273563088,
                         0,
                         312.6263091035918,
                         0,
                         662.9296902690498,
                         258.9996285827844,
                         0,
                         0,
                         1};
  Mat K_L = cv::Mat(3, 3, cv::DataType<double>::type, K_left); 
  double K_right[3][3] = {647.3402626821477,
                          0,
                          298.8766921846282,
                          0,
                          647.739941990085,
                          259.9519313557778,
                          0,
                          0,
                          1};
  Mat K_R = cv::Mat(3, 3, cv::DataType<double>::type, K_right);
  double R_stereo[3][3] = {
      0.9997826605620699,    0.004051019540949904,  0.02045044938645102,
      -0.004074743115292153, 0.999991072656389,     0.001118515117933302,
      -0.02044573569166274,  -0.001201602348328341, 0.9997902420227071};
  Mat  R= cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);
  Vec3d T = {0.0676242, 0.0119106,0.0116169};
  

  const size_t Row = img_L.size[0];//1080
  const size_t Col = img_L.size[1];//1920
    
 
  Matrix3d r; 
  Vector3d T1;
  Matrix3d KL;
  Matrix3d KR;
  cv2eigen(R,r);
  cv2eigen(T, T1);
  //---------------------------------------------
  cv2eigen(K_R, KR);
  cv2eigen(K_L, KL);
  //----------------------------------
    
 
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
  Vector3d locat;
  Vector3d locat_l;
  Vector3d locat_r;
    
  //�����ô���
  //------------------------------------------------------------
  
 /* cout << "I.cross(T1)" << I.cross(T1) << endl;
  cout << "E1=" << E1 << endl;
  cout << "E2=" << E2 << endl;
  cout << "E3=" << E3 << endl;
  cout << "Rrect=" << Rrect << endl;
  cout << "R2=" << R2 << endl;
  cout << "Rrect * E1="<< Rrect * E1<<endl;*/
  
  int n = 0;
  int max=-200;
  int min = 1300;
  //------------------------------------------------------------
    
    
 #pragma omp parallel for
  for (int x = 1 ; x < Row ; x++) {
    for (int y = 1 ; y < Col ; y++) {
      locat = {double(x), double(y), 1};
      locat_l = KL * R1 * locat;
      locat_r = KR * R2 * locat;
      /*locat_l = R1 * locat;
      locat_r = R2 * locat;*/
      /*double kl = f / locat_l[2];
      double kr = f / locat_r[2];*/
      //����
      /*Vector3d HL = KL * R1 * locat_l;
      Vector3d HR = KR * R2 * locat_r;*/


      //�����ô���
      //------------------------------------------------------------
      /*cout << " K_L * R1 " << K_L * R1 * locat_l<< endl;
      cout << " K_R * R2 " << K_R * R2 * locat_r << endl;*/
      /*Vector3d templ = K_L * R1 * locat_l ;
      Vector3d tempr = K_R * R2 * locat_r;
      

      if (templ[2] > max) {
        max = templ[2];
      }
      if (templ[2] < min) {
        min = templ[2];
      }*/
      /*n++;*/
      /*std::cout << "n=" << n << std::endl;*/
      /*if (kl*locat_l[0] > max) {
        max = kl*locat_l[0];
      }
      if (kl * locat_l[0] < min) {
        min = kl * locat_l[0];
      }*/
      /*std::cout << "N0." << n << ":" << locat_l[1] << std::endl;*/
      /*std::cout << "N0." << n << ":" <<locat_l[0] << std::endl;*/
      //-----------------------------------------------------------------------
      /*image_l_rected.at<double>(int(locat_l[0]), int(locat_l[1])) = img_L.at<uint8_t>(x, y);
      image_r_rected.at<double>(int(locat_r[0]), int(locat_r[1])) = img_R.at<uint8_t>(x, y); */
           //����ĸ�ֵ���ᵼ�����? �������ߵ����Ե�ʱ�򶼲�����ġ�                
    }
  }
  /*std::cout << "max=" << max << std::endl;
  std::cout << "min=" << min << std::endl;*/
 //double K_left[3][3] = {662.3562273563088,
 //                       0,
 //                       312.6263091035918,
 //                       0,
 //                       662.9296902690498,
 //                       258.9996285827844,
 //                       0,
 //                       0,
 //                       1};
 //Mat K1 = cv::Mat(3, 3, cv::DataType<double>::type, K_left);

 //double d_left[1][5] = {0.06966962838870275, 0.02054263655123773,
 //                       0.001252584212211826, 0.002089077085379777,
 //                       -0.4096320330693385};
 //Mat D1 = cv::Mat(1, 5, cv::DataType<double>::type, d_left);

 //double K_right[3][3] = {647.3402626821477,
 //                        0,
 //                        298.8766921846282,
 //                        0,
 //                        647.739941990085,
 //                        259.9519313557778,
 //                        0,
 //                        0,
 //                        1};
 //Mat K2 = cv::Mat(3, 3, cv::DataType<double>::type, K_right);

 //double d_right[1][5] = {0.02272045036297292, -0.5565235313790773,
 //                        0.007380600944678045, -0.007607934580409265,
 //                        1.30443069011335};
 //Mat D2 = cv::Mat(1, 5, cv::DataType<double>::type, d_right);

 //double R_stereo[3][3] = {
 //    0.9997826605620699,    0.004051019540949904,  0.02045044938645102,
 //    -0.004074743115292153, 0.999991072656389,     0.001118515117933302,
 //    -0.02044573569166274,  -0.001201602348328341, 0.9997902420227071};
 //Mat R = cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);

 //Vec3d T = {-0.0676242, -0.0119106, -0.0116169};


 //// ��һ������˫ĿУ��������stereoRectify�����������double���͵�Mat������ᱨ��
 //// -205:Formats of input arguments do not match) All the matrices must have the
 //// same data type in function 'cvRodrigues2'
 //cv::Mat R1, R2, P1, P2, Q;
 //stereoRectify(K1, D1, K2, D2, img_L.size(), R, T, R1, R2, P1, P2, Q);

 //cout << "��һ��˫ĿУ�������" << endl << endl;
 //cout << R1 << endl;
 //cout << P1 << endl;
 //cout << R2 << endl;
 //cout << P2 << endl;
 //
} 

  

/*���ʣ�
    1.��������λ����double���͵Ļ����᲻�������������ǿת��int������
    2.֮������double�������������������Ϊ�ڳ���R����double�ͣ���ʱ��������Ͳ�ͬ�ᷢ������ֻ�ܰ�һ���������󻻳�double����
    3.Ҫ������ٸ�ת�����������꣨[x' y' z']������f/z'�����������pdf P46�����ҷ���z'��С��Ϊ�����������Ժ�������������
    4.������T�Ϳ��Եó�Rrect����û��û���õ���˵��E^T*e=0ʲô��
!!! 5.У����ʱ�ǲ�ɫͼƬ���ǻ�ɫͼƬ��Ϊ���ҵĸ�ֵ�������������ȷ������������дdouble�ˣ���֮Ҳ���û�ɫ�ˣ���
    6.���ܻ�����ϸ���ϵ��©�����ٲ��
*/

//Mat image_l = imread(testset + "/im0.png");
//Mat image_r = imread(testset + "/im1.png");