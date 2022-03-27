//
//  bendingEnergy.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#ifndef bendingEnergy_hpp
#define bendingEnergy_hpp
#include <Eigen/Dense>
#include "helper/shape.hpp"

class BendingEnergy {
private:
    float getVoronoiArea(int i, Shape &shape, int neighboor, Eigen::MatrixXf &cotTriangleAngles);
    float getMixedArea(int i, Shape &shape, Eigen::VectorXi &oneRingNeighboorhood, Eigen::MatrixXf &triangleAngles, Eigen::MatrixXf &cotTriangleAngles);
    void getMeanCurvatures(Shape &shape, Eigen::VectorXf &curvatures, Eigen::VectorXf &A);
    Eigen::MatrixXf getSumOverOneRingNeighborhood(int i, Shape &shape, Eigen::VectorXi &oneRingNeighboorhood, Eigen::MatrixXf &cotTriangleAngles);
    
public:
    BendingEnergy();
    Eigen::MatrixXf get(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo);
};

#endif /* bendingEnergy_hpp */

