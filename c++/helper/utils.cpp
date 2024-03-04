//
//  utils.cpp
//  helper
//
//  Created by Paul Rötzer on 11.04.21.
//

#include <Eigen/Dense>
#include <math.h>
#include "utils.hpp"
#include <igl/adjacency_list.h>

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


Eigen::MatrixX<bool> computeP2PMat(Shape& shapeX,
                                   Shape& shapeY,
                                   const Eigen::MatrixXi& coarsep2pmap,
                                   const Eigen::MatrixXi& IXf2c,
                                   const Eigen::MatrixXi& IYf2c,
                                   const int c2fNeighborhood) {

    Eigen::MatrixX<bool> p2pMatCoarse(std::max(coarsep2pmap.col(0).maxCoeff(), IXf2c.maxCoeff())+1,
                                      std::max(coarsep2pmap.col(1).maxCoeff(), IYf2c.maxCoeff())+1);
    p2pMatCoarse.setZero();
    // init p2p matrix
    for (int i = 0; i < coarsep2pmap.rows(); i++) {
        p2pMatCoarse(coarsep2pmap(i, 0), coarsep2pmap(i, 1)) = 1;
    }

    std::vector<std::vector<int>> adjX, adjY;
    igl::adjacency_list(shapeX.getF(), adjX);
    igl::adjacency_list(shapeY.getF(), adjY);
    Eigen::MatrixX<bool> p2pMatFine(shapeX.getNumVertices(), shapeY.getNumVertices());
    p2pMatFine.setZero();
    for (int vx = 0; vx < p2pMatFine.rows(); vx++) {
        for (int vy = 0; vy < p2pMatFine.cols(); vy++) {
            bool set2one = false;
            set2one = p2pMatCoarse(IXf2c(vx), IYf2c(vy));
            if (!set2one) {
                // if not already setting to one we check one ring neighborhood
                if (c2fNeighborhood > 0) {
                    for (auto vxx: adjX.at(vx)) {
                        set2one = set2one || p2pMatCoarse(IXf2c(vxx), IYf2c(vy));
                        if (set2one) break;
                        if (c2fNeighborhood > 1) {
                            for (auto vxxx: adjX.at(vxx)) {
                                set2one = set2one || p2pMatCoarse(IXf2c(vxxx), IYf2c(vy));
                                if (set2one) break;
                            }
                        }
                    }

                    for (auto vyy: adjY.at(vy)) {
                        set2one = set2one || p2pMatCoarse(IXf2c(vx), IYf2c(vyy));
                        if (set2one) break;
                        if (c2fNeighborhood > 1) {
                            for (auto vyyy: adjY.at(vyy)) {
                                set2one = set2one ||p2pMatCoarse(IXf2c(vx), IYf2c(vyyy));
                                if (set2one) break;
                            }
                        }
                    }
                }
            }
            p2pMatFine(vx, vy) = set2one;
        }
    }
    return p2pMatFine;
}

} // namespace utils
