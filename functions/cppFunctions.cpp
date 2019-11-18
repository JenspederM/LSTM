#include <Rcpp.h>
using namespace Rcpp;
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export(name = `%op%`)]]
NumericMatrix outerProduct(NumericVector v1, NumericVector v2) {
  Eigen::Map<Eigen::VectorXd> x(as<Eigen::Map<Eigen::VectorXd> >(v1));
  Eigen::Map<Eigen::VectorXd> y(as<Eigen::Map<Eigen::VectorXd> >(v2));
  
  Eigen::MatrixXd op = x * y.transpose();
  return Rcpp::wrap(op);
}

// [[Rcpp::export(name = `%m%`)]]
NumericVector matrixMultiplication(NumericMatrix mat, NumericVector vec) {
  Eigen::Map<Eigen::MatrixXd> x(as<Eigen::Map<Eigen::MatrixXd> >(mat));
  Eigen::Map<Eigen::MatrixXd> y(as<Eigen::Map<Eigen::MatrixXd> >(vec));
  
  Eigen::VectorXd m = x*y;
  return Rcpp::wrap(m);
}