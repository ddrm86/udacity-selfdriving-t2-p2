#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  unsigned int n_est = estimations.size();
  if ((n_est == 0) || (n_est != ground_truth.size())) {
    cout << "Zero estimations or estimations and ground truth data of different sizes" << endl;
    return rmse;
  }
  
  for (int i=0; i<n_est; i++) {
    VectorXd difference = estimations[i] - ground_truth[i];
    VectorXd quad_dif = difference.array() * difference.array();
    rmse += quad_dif;
  }
  rmse = rmse / n_est;
  rmse = rmse.array().sqrt();
  
  return rmse;
}

