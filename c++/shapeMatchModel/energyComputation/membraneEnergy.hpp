//
//  membraneEnergy.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#ifndef membraneEnergy_hpp
#define membraneEnergy_hpp
#include <Eigen/Dense>
#include "helper/shape.hpp"

class MembraneEnergy {
private:
    Eigen::ArrayXXf getG(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo);
    Eigen::ArrayXf getW(Eigen::ArrayXXf &A, float mu, float lambda);
    Eigen::ArrayXXf getEdges12(Shape &shape, Eigen::MatrixXi &FCombo);
    Eigen::ArrayXXf getg(Eigen::ArrayXXf &edges12);
    
public:
    MembraneEnergy();
    Eigen::MatrixXf get(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi FaCombo, Eigen::MatrixXi FbCombo);
};

#endif /* membraneEnergy_hpp */
