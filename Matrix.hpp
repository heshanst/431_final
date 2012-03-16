//This code is distributed under the BSD 
//license and it is a rewrite of code shared 
//in class CSC431 at DePaul University 
//by Massimo Di Pierro
//Rewritten by Haohui Huang & Binfang Qiu

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

class Matrix {
public:
  typedef std::vector<double> Row;
  
public:
  Matrix(unsigned int rows, unsigned int cols, double fill = 0.0) :
    data(rows, Row(cols, fill))
  {
  }
  
  Matrix(const Matrix& mat) : data(mat.data) {
  }

  Matrix(std::vector<Row> mat) : data(mat) {
  }
  
  ~Matrix() {
  }

  Row& operator[](unsigned int i) {
    return const_cast<Row&>(const_cast<const Matrix&>(*this)[i]);
  }

  const Row& operator[] (unsigned int i) const {
    return data[i];
  }
  
  Row row(int i) const {
    return data[i];
  }

  std::vector<double> col(int i) const {
    std::vector<double> column;
    for (unsigned int j = 0; j < data.size(); ++j) {
      column.push_back(data[j][i]);
    }
    return column;
  }

  unsigned int rows() const {
    return data.size();
  }

  unsigned int cols() const {
    if (data.size() > 0) {
      return data[0].size();
    } else {
      return 0;
    }
  }

  static Matrix identity(unsigned int rows = 1, 
                         double one = 1.0, 
                         double fill = 0.0) {
    Matrix m = Matrix(rows, rows, fill);
    for (unsigned int i = 0; i < rows; ++i) {
      m[i][i] = one;
    }
    return m;
  }

  static Matrix diagonal(const std::vector<double>& d) {
    Matrix m = Matrix(d.size(), d.size());
    for (unsigned int i = 0; i < d.size(); ++i) {
      m[i][i] = d[i];
    }
    return m;
  }

  friend Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs);
    result += rhs;
    return result;
  }
  
  Matrix& operator+= (const Matrix& rhs) {
    unsigned int n = rows();
    unsigned int m = cols();
    if (rhs.rows() != n || rhs.cols() != m) {
      throw std::invalid_argument("Incompatible dimensions");
    }
    
    for (unsigned int i = 0; i < n; ++i) {
      for (unsigned int j = 0; j < m; ++j) {
        data[i][j] += rhs[i][j];
      }
    }
    return *this;
  }

  friend Matrix operator- (const Matrix& A, const Matrix& B) {
    Matrix m(A);
    m -= B;
    return m;
  }
  
  Matrix& operator-= (const Matrix& rhs) {
    unsigned int n = rows();
    unsigned int m = cols();
    if (rhs.rows() != n || rhs.cols() != m) {
      throw std::invalid_argument("Incompatible dimensions");
    }
    
    for (unsigned int i = 0; i < n; ++i) {
      for (unsigned int j = 0; j < m; ++j) {
        data[i][j] -= rhs[i][j];
      }
    }
    return *this;
  }

  friend bool operator==(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
      return false;
    }
    
    for (unsigned int i = 0; i < lhs.rows(); ++i) {
      for (unsigned int j = 0; j < lhs.cols(); ++j) {
        if (lhs[i][j] != rhs[i][j]) {
          return false;
        }
      }
    }
    
    return true;
  }

  friend bool operator != (const Matrix& lhs, const Matrix& rhs) {
    return !(lhs == rhs);
  }

  friend Matrix operator* (double x, const Matrix& rhs) {
    Matrix m(rhs);
    for (unsigned int r = 0; r < m.rows(); ++r) {
      for (unsigned int c = 0; c < m.cols(); ++c) {
        m[r][c] *= x;
      }
    }
    return m;
  }

  friend Matrix operator* (const Matrix& lhs, double x) {
    return x * lhs;
  }
  
  friend Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols() != rhs.rows()) {
      throw std::invalid_argument("Incompatible dimensions");
    }

    Matrix m(lhs.rows(), rhs.cols());
    for (unsigned int r = 0; r < lhs.rows(); ++r) {
      for (unsigned int c = 0; c < rhs.cols(); ++c) {
        for (unsigned int k = 0; k < lhs.cols(); ++k) {
          m[r][c] = lhs[r][k] * rhs[k][c];
        }
      }
    }
    return m;
  }
  
  // Computes x/A using Gauss-Jordan elimination where x is a scalar
  friend Matrix operator / (double x, const Matrix& rhs) {
    if (rhs.rows() != rhs.cols()) {
      throw std::invalid_argument("matrix not squared");
    }
    
    Matrix A(rhs);
    Matrix B = identity(A.rows(), x);
    for (unsigned int c = 0; c < A.rows(); ++c) {
      for (unsigned int r = c + 1; r < A.rows(); ++r) {
        if (abs(A[r][c]) > abs(A[c][c])) {
          A.swap_rows(r,c);
          B.swap_rows(r,c);
        }
      }
      
      double p = A[c][c];
      for (unsigned int k = 0; k < A.rows(); ++k) {
        A[c][k] /= p;
        B[c][k] /= p;
      }

      for (unsigned int r = 0; r < A.rows(); ++r) {
        if (r != c) {
          p = A[r][c];
          for (unsigned int k = 0; k < A.rows(); ++k) {
            A[r][k] -= A[c][k]*p;
            B[r][k] -= B[c][k]*p;
          }
        }
      }
    }

    return B;
  }

  friend Matrix operator / (const Matrix& lhs, double x) {
    return (1.0 / x) * lhs;
  }
  
  friend Matrix operator / (const Matrix& lhs, const Matrix& rhs) {
    return lhs * (1.0 / rhs);
  }

  void swap_rows(unsigned int i, unsigned int j) {
    std::swap(data[i], data[j]);
  }

  // Transposed of A
  Matrix t() {
      Matrix tt(cols(), rows());
      for (unsigned int r = 0; r < cols(); ++r) {
          for (unsigned int c = 0; c < rows(); ++c) {
              tt[r][c] = data[c][r];
          }
      }
      return tt;
  }
  
private:
  std::vector<Row> data;
};
