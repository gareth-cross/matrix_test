// Simple matrix performance test - July 21 2017.
// Compile with:
// g++ -std=c++11 -O3 -I<Eigen Install Path> matrix_test.cc

#include <chrono>
#include <iostream>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "tic_toc.h"

template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

// Test fixture w/ some common operations.
template <typename Scalar, int Dim>
class MatrixTest {
 public:
  using Matrix = Eigen::Matrix<Scalar, Dim, Dim>;
  using Vector = Eigen::Matrix<Scalar, Dim, 1>;

  MatrixTest(const int dimension = Dim)
      : dimension_(dimension),
        name_(std::string(typeid(Scalar).name()) + ", " +
              std::to_string(dimension) + ", ") {}

  void Run(const int num_runs, const int num_iterations) {
    for (int i = 0; i < num_runs; ++i) {
      Reset(num_iterations);
      MatrixVectorMul();
      MatrixRankUpdate();
      VectorRankUpdate();
      CholeskySolve();
      QRSolve();
      EigenDecomposition();
    }
  }

  // Reset objects we use for computations to random values.
  // Not included in timing.
  void Reset(const int num_iterations) {
    matrices_ = AlignedVector<Matrix>(num_iterations);
    vectors_ = AlignedVector<Vector>(num_iterations);
    for (auto& A : matrices_) {
      // Fill w/ PD matrix so we can do factorizations.
      A.noalias() = RandomPDMatrix();
    }
    for (auto& x : vectors_) {
      x.setRandom(dimension_, 1);
    }
  }

  // Do the summation b = A_i * x_i + ...
  void MatrixVectorMul() {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
    b.setZero(dimension_);
    // test
    {
      const TicToc tic(name_ + __FUNCTION__, matrices_.size());
      for (int i = 0; i < matrices_.size(); ++i) {
        b.noalias() += matrices_[i] * vectors_[i];
      }
    }
  }

  // Do the summation B = A_i^T * A_i + ...
  void MatrixRankUpdate() {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B;
    B.setZero(dimension_, dimension_);
    // test
    {
      const TicToc tic(name_ + __FUNCTION__, matrices_.size());
      for (const auto& A : matrices_) {
        B.template selfadjointView<Eigen::Lower>().rankUpdate(A.transpose());
      }
    }
  }

  // Do the summation B = x_i * x_i^T + ...
  void VectorRankUpdate() {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B;
    B.setZero(dimension_, dimension_);
    // test
    {
      const TicToc tic(name_ + __FUNCTION__, matrices_.size());
      for (const auto& x : vectors_) {
        B.template selfadjointView<Eigen::Lower>().rankUpdate(x);
      }
    }
  }

  // Do the summation b = A_i^-1 * x_i + ..., using cholesky decomposition.
  void CholeskySolve() {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
    b.setZero(dimension_);
    // test
    {
      const TicToc tic(name_ + __FUNCTION__, matrices_.size());
      for (int i = 0; i < matrices_.size(); ++i) {
        const Eigen::LLT<Matrix, Eigen::Lower> cholesky(matrices_[i]);
        b.noalias() += cholesky.solve(vectors_[i]);
      }
    }
  }

  // Do the summation b = A_i^-1 * x_i + ..., using QR decomposition.
  void QRSolve() {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
    b.setZero(dimension_);
    // test
    {
      const TicToc tic(name_ + __FUNCTION__, matrices_.size());
      for (int i = 0; i < matrices_.size(); ++i) {
        const Eigen::FullPivHouseholderQR<Matrix> qr(matrices_[i]);
        b.noalias() += qr.solve(vectors_[i]);
      }
    }
  }

  // Do the summation b = eig(A_i) + ...
  void EigenDecomposition() {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b;
    b.setZero(dimension_);
    // test
    {
      const TicToc tic(name_ + __FUNCTION__, matrices_.size());
      for (int i = 0; i < matrices_.size(); ++i) {
        const Eigen::SelfAdjointEigenSolver<Matrix> eigen(matrices_[i]);
        b.noalias() += eigen.eigenvalues();
      }
    }
  }

 private:
  // Generate positive definite matrix.
  Matrix RandomPDMatrix() {
    Matrix result(dimension_, dimension_);
    result.setZero();
    for (int i = 0; i < dimension_; ++i) {
      const Vector x = Vector::Random(dimension_);
      result.noalias() += (x * x.transpose());
    }
    result.diagonal().array() += dimension_;
    return result;
  }

  const int dimension_;
  const std::string name_;

  AlignedVector<Eigen::Matrix<Scalar, Dim, Dim>> matrices_;
  AlignedVector<Eigen::Matrix<Scalar, Dim, 1>> vectors_;
};

int main() {
  constexpr int kNumRuns = 100;

  MatrixTest<float, 3>().Run(kNumRuns, 1000);
  MatrixTest<float, 4>().Run(kNumRuns, 1000);
  MatrixTest<float, 6>().Run(kNumRuns, 1000);
  MatrixTest<float, 8>().Run(kNumRuns, 1000);
  MatrixTest<float, 9>().Run(kNumRuns, 1000);
  MatrixTest<float, Eigen::Dynamic>(50).Run(kNumRuns, 10);
  MatrixTest<float, Eigen::Dynamic>(100).Run(kNumRuns, 10);
  MatrixTest<float, Eigen::Dynamic>(200).Run(kNumRuns, 10);
  MatrixTest<float, Eigen::Dynamic>(300).Run(kNumRuns, 10);

  MatrixTest<double, 3>().Run(kNumRuns, 1000);
  MatrixTest<double, 4>().Run(kNumRuns, 1000);
  MatrixTest<double, 6>().Run(kNumRuns, 1000);
  MatrixTest<double, 8>().Run(kNumRuns, 1000);
  MatrixTest<double, 9>().Run(kNumRuns, 1000);
  MatrixTest<double, Eigen::Dynamic>(50).Run(kNumRuns, 10);
  MatrixTest<double, Eigen::Dynamic>(100).Run(kNumRuns, 10);
  MatrixTest<double, Eigen::Dynamic>(200).Run(kNumRuns, 10);
  MatrixTest<double, Eigen::Dynamic>(300).Run(kNumRuns, 10);
  return 0;
}
