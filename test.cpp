#include <Rcpp.h>
using namespace Rcpp;
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
NumericMatrix outerProd(NumericVector v1, NumericVector v2) {
  NumericVector xx(v1);
  NumericVector yy(v2);
  
  const Eigen::Map<Eigen::VectorXd> x(as<Eigen::Map<Eigen::VectorXd> >(xx));
  const Eigen::Map<Eigen::VectorXd> y(as<Eigen::Map<Eigen::VectorXd> >(yy));
  
  Eigen::MatrixXd op = x * y.transpose();
  return Rcpp::wrap(op);
}
  
// [[Rcpp::export]]
NumericMatrix MatMult(NumericMatrix mat, NumericVector vec) {
  NumericMatrix xx(mat);
  NumericVector yy(vec);
  
  const Eigen::Map<Eigen::MatrixXd> x(as<Eigen::Map<Eigen::MatrixXd> >(xx));
  const Eigen::Map<Eigen::MatrixXd> y(as<Eigen::Map<Eigen::MatrixXd> >(yy));
  
  Eigen::MatrixXd prod = x*y;
  return Rcpp::wrap(prod);
}