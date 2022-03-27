//
//  wksEnergy.hpp
//  shape-matching-dd
//
//  Created by Paul Rötzer on 28.10.21.
//

#ifndef wksEnergy_hpp
#define wksEnergy_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include "helper/shape.hpp"

class WKSEnergy {
private:
    float getVoronoiArea(int i, Shape &shape, int neighboor, Eigen::MatrixXf &cotTriangleAngles);
    float getMixedArea(int i, Shape &shape, Eigen::VectorXi &oneRingNeighboorhood, Eigen::MatrixXf &triangleAngles, Eigen::MatrixXf &cotTriangleAngles);
    void getA(Shape &shape, Eigen::VectorXf &A);
    void getWKS(Shape &shape, Eigen::MatrixXf& WKS, const int wksSize, const int wksVariance, const int numEigenFunctions);

public:
    WKSEnergy();
    Eigen::MatrixXf get(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo);
};

#endif /* wksEnergy_hpp */
