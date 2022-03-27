//
//  utils.hpp
//  helper
//
//  Created by Paul Rötzer on 11.04.21.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>

#define ASSERT_NEVER_REACH assert(false)
#define FLOAT_EPSI 1e-7

namespace utils {

template <typename T>
void writeMatrixToFile(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix, const std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file(fileName + ".csv");
    if (file.is_open()) {
        file << matrix.format(CSVFormat);
        file.close();
    }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> readMatrixFromFile(const std::string fileName) {
    std::vector<T> matrixEntries;
    std::ifstream matrixDataFile(fileName);
    if (matrixDataFile.fail()){
        std::cout << "File: " << fileName << " not found" << std::endl;
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>();
    }
    std::string matrixRowString;
    std::string matrixEntry;
    int matrixRowNumber = 0;
     
    while (getline(matrixDataFile, matrixRowString)) {
        std::stringstream matrixRowStringStream(matrixRowString);
 
        while (std::getline(matrixRowStringStream, matrixEntry, ',')) {
            matrixEntries.push_back(stod(matrixEntry));
        }
        matrixRowNumber++; //update the column numbers
    }
     
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

void addElement2IntVector(Eigen::VectorXi &vec, int val);

float safeLog(const  float x);

Eigen::ArrayXf arraySafeLog(const  Eigen::ArrayXf X);

float squaredNorm(const Eigen::Vector3f vec);

/* function all
    checks if all elements in two integer matrices are the equal
 */
inline bool allEqual(const Eigen::MatrixXi inp1, const Eigen::MatrixXi inp2) {
    assert(inp1.rows() == inp2.rows());
    assert(inp1.cols() == inp2.cols());
    
    return (inp1 - inp2).norm() == 0;
}

template <typename T>
int getFirstNonZeroIndexOfCol(const Eigen::SparseMatrix<T, Eigen::ColMajor> &mat, int col) {
    for (typename Eigen::SparseMatrix<T, Eigen::ColMajor>::InnerIterator it(mat, col); it; ++it) {
        return it.index();
    }
    return -1;
}

template <typename T>
int getFirstNonZeroIndexOfRow(const Eigen::SparseMatrix<T, Eigen::RowMajor> &mat, int row) {
    for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(mat, row); it; ++it) {
        return it.index();
    }
    return -1;
}

template <typename T>
bool isIn(const T val, const T* array, const size_t length) {
    for (int i = 0; i < length; i++) {
        if (val == array[i]) {
            return true;
        }
    }
    return false;
}

void setLinspaced(Eigen::MatrixXi& mat, int start);
Eigen::MatrixXi linspaced(int start, int end);
Eigen::MatrixXi linspaced(int start, int end, int step);

// these functions have to to go to the header since otherwise the linking does not work
template<typename XprType, typename RowFactorType, typename ColFactorType>
auto repelem(const XprType &xpr, RowFactorType row_factor, ColFactorType col_factor) {
    using namespace Eigen;

    const int RowFactor = internal::get_fixed_value<RowFactorType>::value;
    const int ColFactor = internal::get_fixed_value<ColFactorType>::value;
    const int NRows = XprType::RowsAtCompileTime == Dynamic || RowFactor == Dynamic ? Dynamic : XprType::RowsAtCompileTime*RowFactor;
    const int NCols = XprType::ColsAtCompileTime == Dynamic || ColFactor == Dynamic ? Dynamic : XprType::ColsAtCompileTime*ColFactor;
    const int nrows = internal::get_runtime_value(row_factor) * xpr.rows();
    const int ncols = internal::get_runtime_value(col_factor) * xpr.cols();

    return xpr(
        Array<int,NRows,1>::LinSpaced(nrows,0,xpr.rows()-1),
        Array<int,NCols,1>::LinSpaced(ncols,0,xpr.cols()-1)
    );
}

template <typename T, int majorType>
Eigen::SparseMatrix<T, majorType> sprepelem(const Eigen::SparseMatrix<T, majorType> &A, const int r, const int c) {
    assert(r>0);
    assert(c>0);
    Eigen::SparseMatrix<T, majorType> B(r*A.rows(), c*A.cols());
    std::vector<Eigen::Triplet<T>> b;
    b.reserve(r*c*A.nonZeros());
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            // Loop outer level
            for (int k = 0; k < A.outerSize(); ++k) {
                // loop inner level
                for (typename Eigen::SparseMatrix<T, majorType>::InnerIterator it(A,k); it; ++it) {
                    Eigen::Triplet<T> triplet(i + it.row() * r, j + it.col() * c, it.value());
                    b.push_back(triplet);
                }
            }
        }
    }
    
    B.setFromTriplets(b.begin(), b.end());
    return B;
}

template <typename T, int majorType>
Eigen::SparseMatrix<T, majorType> repmat(const Eigen::SparseMatrix<T, majorType> &A, const int r, const int c) {
    assert(r>0);
    assert(c>0);
    Eigen::SparseMatrix<T, majorType> B(r*A.rows(), c*A.cols());
    std::vector<Eigen::Triplet<T>> b;
    b.reserve(r*c*A.nonZeros());
    
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            // Loop outer level
            for (int k = 0; k < A.outerSize(); ++k) {
                // loop inner level
                for (typename Eigen::SparseMatrix<T, majorType>::InnerIterator it(A,k); it; ++it) {
                    Eigen::Triplet<T> triplet(i * A.rows() + it.row(), j * A.cols() + it.col(), it.value());
                    b.push_back(triplet);
                }
            }
        }
    }
    
    B.setFromTriplets(b.begin(), b.end());
    return B;
}

template <typename T>
int numElemEqualTo(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix, const T value) {
    int numElemEqual = 0;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            if (matrix(i, j) == value) {
                numElemEqual++;
            }
        }
    }
    return numElemEqual;
}

} // namespace utils

#endif /* utils_hpp */
