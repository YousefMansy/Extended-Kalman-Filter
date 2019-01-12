#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   // check the validity of the following inputs:
   //  * the estimation vector size should not be zero
   //  * the estimation vector size should equal ground truth vector size
   if (!estimations.size() || estimations.size() != ground_truth.size())
      cout << "Error\n";

   // accumulate squared residuals
   for (int i = 0; i < estimations.size(); ++i)
   {
      VectorXd diff = estimations[i] - ground_truth[i];
      rmse.array() += diff.array().pow(2);
   }

   // calculate the mean
   rmse = rmse.array() * 1 / estimations.size();

   // calculate the squared root
   rmse = rmse.array().sqrt();

   // return the result
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   MatrixXd Hj(3, 4);

   // recover state parameters
   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);

   // check division by zero

   if (fabs(px) < 0.000001 && fabs(py) < 0.000001)
      px = py = 0.000001;
   
   // compute the Jacobian matrix
   float a = px*px+py*py;
   float b = sqrt(a);
   float c = (a*b);

   if (fabs(a) < 0.000001)
      a = 0.000001;

   Hj << px / b, py / b, 0, 0,
        -py / a, px / a, 0, 0,
        py*(vx*py - vy*px)/c, px*(px*vy - py*vx)/c, px/b, py/b;

        
   // Hj << px / sqrt(px * px + py * py), py / sqrt(px * px + py * py), 0, 0,
   //     -py / (px * px + py * py), px / (px * px + py * py), 0, 0,
   //     (py * (vx * py - vy * px)) / pow((double)(px * px) + (double)(py * py), 1.5), px * (vy * px - vx * py) / pow(px * px + py * py, 3 / 2), px / sqrt(px * px + py * py), py / sqrt(px * px + py * py);
  

   return Hj;
}

VectorXd Tools::CartesianToPolar(const VectorXd &x_state)
{
   float px, py, vx, vy;
   px = x_state[0];
   py = x_state[1];
   vx = x_state[2];
   vy = x_state[3];

   float rho, phi, rho_dot;
   rho = sqrt(px * px + py * py);
   phi = atan2(py, px); // returns values between -pi and pi

   // if rho is very small, set it to 0.0001 to avoid division by 0 in computing rho_dot
   if (rho < 0.000001)
      rho = 0.000001;

   rho_dot = (px * vx + py * vy) / rho;

   VectorXd z_pred = VectorXd(3);
   z_pred << rho, phi, rho_dot;

   return z_pred;
}
