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
                     
  double R_stereo[3][3] = {
      0.9997826605620699,    0.004051019540949904,  0.02045044938645102,
      -0.004074743115292153, 0.999991072656389,     0.001118515117933302,
      -0.02044573569166274,  -0.001201602348328341, 0.9997902420227071};
  Mat  R= cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);
  Vec3d T = {-0.0676242, -0.0119106, -0.0116169};
  double f=1;//�������ͺ���ֵ��ʱֱ���ô��ݵġ�

  const size_t Row = img_L.size[0];
  const size_t Col = img_L.size[1];

  Matrix3d r;
  cv2eigen(R,r);

  Vector3d T1(1, 2, 3);
  Vector3d I(0, 0, 1);
  double TT = T1.norm();
  Vector3d E1 = T1 / TT;
  Vector3d E2 = I.cross(T1) / (I.cross(T1)).norm();
  Vector3d E3 = E1.cross(E3);
  Matrix<double, 3, 3> temp;
  Matrix<double, 3, 3> Rrect;
  temp << E1, E2, E3;
  Rrect = temp.transpose();
  
  

  Matrix<double, 3, 3> R1 = Rrect;
  Matrix<double, 3, 3> R2 = r*Rrect;
  Vector3d locat;
  Vector3d locat_l;
  Vector3d locat_r;

   /* int n = 0;*/
 #pragma omp parallel for
  for (int x = 1 ; x < Row ; x++) {
    for (int y = 1 ; y < Col ; y++) {
      locat = {double(x), double(y), 1};
      locat_l = R1 * locat;
      locat_r = R2 * locat;
      double kl = f / locat_l[2];
      double kr = f / locat_r[2];
      /*n++;
      std::cout << "N0." << n << ":" << locat_l[2] << std::endl;*/
      /*std::cout << "N0." << n << ":" <<locat_l[0] << std::endl;*/
      image_l_rected.at<uint8_t>(int(locat_l[0]), int(locat_l[1])) = img_L.at<uint8_t>(x, y);
      image_r_rected.at<uint8_t>(int(locat_r[0]), int(locat_r[1])) = img_R.at<uint8_t>(x, y); 
           //����ĸ�ֵ���ᵼ�����? �������ߵ����Ե�ʱ�򶼲�����ġ�                
    }
  }
} 
/*cout << "Rrect * E1="<< Rrect * E1<<endl;*/
  //cout << "I.cross(T1)" << I.cross(T1) << endl;
  /*cout << "E1=" << E1 << endl;
  cout << "E2=" << E2 << endl;
  cout << "E3=" << E3 << endl;
  cout << "Rrect=" << Rrect << endl;*/

/*���ʣ�
    1.��������λ����double���͵Ļ����᲻�������������ǿת��int������
    2.֮������double�������������������Ϊ�ڳ���R����double�ͣ���ʱ��������Ͳ�ͬ�ᷢ������ֻ�ܰ�һ���������󻻳�double����
    3.Ҫ������ٸ�ת�����������꣨[x' y' z']������f/z'�����������pdf P46�����ҷ���z'��С��Ϊ�����������Ժ�������������
    4.������T�Ϳ��Եó�Rrect����û��û���õ���˵��E^T*e=0ʲô��
!!! 5.У����ʱ�ǲ�ɫͼƬ���ǻ�ɫͼƬ��Ϊ���ҵĸ�ֵ�������������ȷ������������дdouble�ˣ���֮Ҳ���û�ɫ�ˣ���
    6.���ܻ�����ϸ���ϵ��©�����ٲ��
*/
