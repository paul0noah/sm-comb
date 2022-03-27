//
//  utils.cpp
//  helper
//
//  Created by Paul Rötzer on 11.04.21.
//

#include <Eigen/Dense>
#include <math.h>
#include "utils.hpp"

namespace utils {

void addElement2IntVector(Eigen::VectorXi &vec, int val) {
    vec.conservativeResize(vec.rows() + 1, Eigen::NoChange);
    vec(vec.rows()-1) = val;
}

/* function safeLog
 log which is linearly extended below a threshold epsi
 */
float safeLog(const float x) {
    float l;
    if (x > FLOAT_EPSI) {
        l = std::log(x);
    }
    else {
        l = (x - FLOAT_EPSI)/FLOAT_EPSI + std::log(FLOAT_EPSI);
    }
    return l;
}

Eigen::ArrayXf arraySafeLog(const Eigen::ArrayXf X) {
    Eigen::ArrayXf L = X;
    for (int i = 0; i < X.rows(); i++) {
        L(i) = safeLog(X(i));
    }
    return L;
}

float squaredNorm(const Eigen::Vector3f vec) {
    return vec(0)*vec(0) + vec(1)*vec(1) + vec(2)*vec(2);
}


/* function setLinspaced
    creates a increasing vector of fixed step of one
    e.g.
    mat = Eigen::MatrixXi(1, 5);
    setLinspaced(mat, 2);
    creates
    mat = [2 3 4 5 6]
 */
void setLinspaced(Eigen::MatrixXi& mat, int start) {
    assert(mat.rows() == 1 || mat.cols() == 1);
    int length;
    if (mat.rows() == 1) {
        length = mat.cols();
    }
    else {
        length = mat.rows();
    }
    for (int i = 0; i < length; i++) {
        mat(i) = i + start;
    }
}

Eigen::MatrixXi linspaced(int start, int end) {
    return linspaced(start, end, 1);
}

Eigen::MatrixXi linspaced(int start, int end, int step) {
    assert(step > 0);
    assert(end > start);
    int length = (end - start)/step;
    Eigen::MatrixXi A(length, 1);
    for (int i = 0; i < length; i++) {
        A(i, 0) = start + i * step;
    }
    return A;
}

} // namespace utils
