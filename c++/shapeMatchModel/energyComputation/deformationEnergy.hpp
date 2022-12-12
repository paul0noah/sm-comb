//
//  deformationEnergy.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#ifndef deformationEnergy_hpp
#define deformationEnergy_hpp
#include <Eigen/Dense>
#include "helper/shape.hpp"
#include "shapeMatchModel/combinations/combinations.hpp"
#include "shapeMatchModel/energyComputation/membraneEnergy.hpp"
#include "shapeMatchModel/energyComputation/bendingEnergy.hpp"

class DeformationEnergy {
private:
    Shape& shapeA;
    Shape& shapeB;
    
    MembraneEnergy membraneEnergy;
    BendingEnergy bendingEnergy;
    
    Combinations& combos;

    
    Eigen::MatrixXi computeNonDegenerateCombinations(Shape &shapeX, Shape &shapeY);
    Eigen::MatrixXi computeDegenerateCombinations(Shape &shapeX, Shape &shapeY);
    Eigen::MatrixXi getTriangle2EdgeMatching(Eigen::MatrixXi &Fx, int numFacesY);
    
    void computeCombinations();
    
    bool computed;
    Eigen::MatrixXf defEnergy;
    
public:
    DeformationEnergy(Shape& sA, Shape& sB, Combinations& c);
    Eigen::MatrixXf get();
    void modifyEnergyVal(const int index, float newVal);
    void useCustomDeformationEnergy(const Eigen::MatrixXf& Vx2VyCostMatrix, bool useAreaWeighting);
};

#endif /* deformationEnergy_hpp */

