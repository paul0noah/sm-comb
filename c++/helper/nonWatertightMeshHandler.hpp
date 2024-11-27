//
//  nonWatertightMeshHandler.hpp
//  shape-matching-dd
//
//  Created by Paul Rötzer on 07.10.21.
//

#ifndef nonWatertightMeshHandler_hpp
#define nonWatertightMeshHandler_hpp

#include <Eigen/Dense>
#include <vector>
#include "helper/shape.hpp"
#include "helper/utils.hpp"
#include "shapeMatchModel/combinations/combinations.hpp"
#include "shapeMatchModel/energyComputation/deformationEnergy.hpp"
#include "shapeMatchModel/constraintsComputation/constraints.hpp"

#define NEW_ENERGY_VALUE 2 * FLOAT_EPSI
class NonWatertightMeshHandler {
private:
    std::vector<int> indicesForModification;
    int oldNumFacesX;
    int oldNumVerticesX;
    int oldNumEdgesX;
    int oldNumFacesY;
    int oldNumVerticesY;
    int oldNumEdgesY;
    int newNumFacesX;
    int newNumVerticesX;
    int newNumEdgesX;
    int newNumFacesY;
    int newNumVerticesY;
    int newNumEdgesY;
    float newEnergyValue;
    Shape shapeXHoles;
    Shape shapeYHoles;
    bool didFillHolesOfShapeX;
    bool didFillHolesOfShapeY;

public:
    bool filledHolesOfShapeX() const;
    bool filledHolesOfShapeY() const;
    Shape getShapeXWithHoles() const;
    Shape getShapeYWithHoles() const;
    NonWatertightMeshHandler();
    NonWatertightMeshHandler(const std::string modelName);
    void writeToFile(const std::string modelName);
    void computeModifyIndices(Constraints &constr);
    bool fillHoles(Shape &shapeX, Shape &shapeY, bool useTriangleFan=false);
    void modifyEnergy(DeformationEnergy &defEnergy, Constraints &constr);
    void modifyEnergy(DeformationEnergy &defEnergy, Constraints &constr, float newEnergyVal);
    void modifyGamma(Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> &Gamma, Constraints &constr, int8_t newGammaVal);
};


#endif /* nonWatertightMeshHandler_hpp */
