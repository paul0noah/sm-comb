//
//  combinations.hpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#ifndef combinations_hpp
#define combinations_hpp

#include <Eigen/Dense>
#include "helper/shape.hpp"

class Combinations {
private:
    Shape& shapeA;
    Shape& shapeB;
    int numCombinations;
    bool combosComputed;
    
    Eigen::MatrixXi FaCombo;
    Eigen::MatrixXi FbCombo;
    
    Eigen::MatrixXi computeNonDegenerateCombinations(Shape &shapeX, Shape &shapeY);
    Eigen::MatrixXi computeDegenerateCombinations(Shape &shapeX, Shape &shapeY);
    Eigen::MatrixXi getTriangle2EdgeMatching(Shape &shapeX, int numFacesY);
    
    
public:
    void init();
    Combinations(Shape& sA, Shape& sB);
    
    void computeCombinations();
    void prune(const Eigen::VectorX<bool>& pruneVec);
    Eigen::MatrixXi& getFaCombo();
    Eigen::MatrixXi& getFbCombo();
};

#endif /* combinations_hpp */
