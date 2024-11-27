//
//  nonWatertightMeshHandler.cpp
//  shape-matching-dd
//
//  Created by Paul Rötzer on 07.10.21.
//


#include "nonWatertightMeshHandler.hpp"

const std::string dataMatrixName = "_NWMHData";
const std::string shapeXOrigName = "_Xoriginal.ply";
const std::string shapeYOrigName = "_Yoriginal.ply";

NonWatertightMeshHandler::NonWatertightMeshHandler() {
    oldNumFacesX    = -1;
    oldNumVerticesX = -1;
    oldNumEdgesX    = -1;
    oldNumFacesY    = -1;
    oldNumVerticesY = -1;
    oldNumEdgesY    = -1;
    newNumFacesX    = -1;
    newNumVerticesX = -1;
    newNumEdgesX    = -1;
    newNumFacesY    = -1;
    newNumVerticesY = -1;
    newNumEdgesY    = -1;
    newEnergyValue  = NEW_ENERGY_VALUE;
    didFillHolesOfShapeX = false;
    didFillHolesOfShapeY = false;
    shapeXHoles = Shape();
    shapeYHoles = Shape();
}

NonWatertightMeshHandler::NonWatertightMeshHandler(const std::string modelName) {
    Eigen::MatrixXi Data = utils::readMatrixFromFile<int>(modelName + dataMatrixName + ".csv");
    oldNumFacesX  = Data(0, 0);
    oldNumVerticesX = Data(0, 1);
    oldNumEdgesX =  Data(0, 2);
    newNumFacesX = Data(0, 3);
    newNumVerticesX = Data(0, 4);
    newNumEdgesX = Data(0, 5);
    didFillHolesOfShapeX = Data(0, 6);
    oldNumFacesY = Data(1, 0);
    oldNumVerticesY = Data(1, 1);
    oldNumEdgesY = Data(1, 2);
    newNumFacesY = Data(1, 3);
    newNumVerticesY = Data(1, 4);
    newNumEdgesY = Data(1, 5);
    didFillHolesOfShapeY = Data(1, 6);
    if (didFillHolesOfShapeX) {
        shapeXHoles = Shape(modelName + shapeXOrigName);
    }
    if (didFillHolesOfShapeY) {
        shapeYHoles = Shape(modelName + shapeYOrigName);
    }
}


void NonWatertightMeshHandler::writeToFile(const std::string modelName) {
    Eigen::MatrixXi Data(2, 7);
    Data(0, 0) = oldNumFacesX;
    Data(0, 1) = oldNumVerticesX;
    Data(0, 2) = oldNumEdgesX;
    Data(0, 3) = newNumFacesX;
    Data(0, 4) = newNumVerticesX;
    Data(0, 5) = newNumEdgesX;
    Data(0, 6) = didFillHolesOfShapeX;
    Data(1, 0) = oldNumFacesY;
    Data(1, 1) = oldNumVerticesY;
    Data(1, 2) = oldNumEdgesY;
    Data(1, 3) = newNumFacesY;
    Data(1, 4) = newNumVerticesY;
    Data(1, 5) = newNumEdgesY;
    Data(1, 6) = didFillHolesOfShapeY;
    utils::writeMatrixToFile(Data, modelName + dataMatrixName);
    if (didFillHolesOfShapeX) {
        shapeXHoles.writeToFile(modelName + shapeXOrigName);
    }
    if (didFillHolesOfShapeY) {
        shapeYHoles.writeToFile(modelName + shapeYOrigName);
    }
}

bool NonWatertightMeshHandler::fillHoles(Shape &shapeX, Shape &shapeY, bool useTriangleFan) {
    oldNumFacesX      = shapeX.getNumFaces();
    oldNumVerticesX   = shapeX.getNumVertices();
    oldNumEdgesX      = shapeX.getNumEdges();
    oldNumFacesY      = shapeY.getNumFaces();
    oldNumVerticesY   = shapeY.getNumVertices();
    oldNumEdgesY      = shapeY.getNumEdges();

    if (!shapeX.isWatertight()) {
        didFillHolesOfShapeX = true;
        shapeXHoles = shapeX;
        if (useTriangleFan)
            shapeX.closeHolesWithTriFan();
        else
            shapeX.closeHoles();
    }
    if (!shapeY.isWatertight()) {
        didFillHolesOfShapeY = true;
        shapeYHoles = shapeY;
        if (useTriangleFan)
            shapeY.closeHolesWithTriFan();
        else
            shapeY.closeHoles();
    }

    if (!shapeX.isWatertight() && !shapeY.isWatertight()) {
        std::cout << "Could not close holes of at least one shape" << std::endl;
        return false;
    }
    
    newNumFacesX      = shapeX.getNumFaces();
    newNumVerticesX   = shapeX.getNumVertices();
    newNumEdgesX      = shapeX.getNumEdges();
    newNumFacesY      = shapeY.getNumFaces();
    newNumVerticesY   = shapeY.getNumVertices();
    newNumEdgesY      = shapeY.getNumEdges();
    return true;
}

void NonWatertightMeshHandler::computeModifyIndices(Constraints &constr) {
    const int approxNumNewProdTriangles = 25 * (newNumFacesX * newNumFacesY - oldNumFacesX * oldNumFacesY);
    indicesForModification.reserve(approxNumNewProdTriangles);

    // non-degenerate
    SparseMatInt8 constrLHS = constr.getConstraintMatrix();
    const int numRowsDel = constrLHS.rows() - newNumFacesX - newNumFacesY;
    const int startRowPiXNewFX = numRowsDel + oldNumFacesX;
    const int endRowPiXNewFX = numRowsDel + newNumFacesX;
    for (int row = startRowPiXNewFX; row < endRowPiXNewFX; row++) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS, row); it; ++it) {
            indicesForModification.push_back(it.index());
        }
    }
    const int startRowPiYNewFY = numRowsDel + newNumFacesX + oldNumFacesY;
    const int endRowPiYNewFY = numRowsDel + newNumFacesX + newNumFacesY;
    for (int row = startRowPiYNewFY; row < endRowPiYNewFY; row++) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS, row); it; ++it) {
            indicesForModification.push_back(it.index());
        }
    }

    // degenerate
    if (oldNumFacesX != newNumFacesX) {
        // tri to vertex
        const int startIdxTriYtoVx = 3 * newNumFacesX * newNumFacesY;
        const int idxFirstTriYtoNewVx = startIdxTriYtoVx + oldNumVerticesX * newNumFacesY;
        const int idxLastTriYtoNewVx = startIdxTriYtoVx + newNumVerticesX * newNumFacesY;
        for (int i = idxFirstTriYtoNewVx; i < idxLastTriYtoNewVx; i++) {
            indicesForModification.push_back(i);
        }
        // tri to edge
        for (int rot = 0; rot < 3; rot++) {
            const int rotOffset = idxLastTriYtoNewVx + 2 * rot * newNumEdgesX * newNumFacesY;
            for (int j = 0; j < 2; j++) {
                const int jOffset = j * newNumEdgesX * newNumFacesY;
                const int startIdxTriYtoEx = rotOffset + jOffset;
                const int idxFirstTriYtoNewEx = startIdxTriYtoEx + oldNumEdgesX * newNumFacesY;
                const int idxLastTriYtoNewEx  = startIdxTriYtoEx + newNumEdgesX * newNumFacesY;
                for (int i = idxFirstTriYtoNewEx; i < idxLastTriYtoNewEx; i++) {
                    indicesForModification.push_back(i);
                }
            }
        }
    }
    if (oldNumFacesY != newNumFacesY) {
        // tri to vertex
        const int startIdxTriXtoVy = 3 * newNumFacesX * newNumFacesY + newNumVerticesX * newNumFacesY + 3 * 2 *newNumEdgesX * newNumFacesY;
        const int idxFirstTriXtoNewVy = startIdxTriXtoVy + oldNumVerticesY * newNumFacesX;
        const int idxLastTriXtoNewVy = startIdxTriXtoVy + newNumVerticesY * newNumFacesX;
        for (int i = idxFirstTriXtoNewVy; i < idxLastTriXtoNewVy; i++) {
            indicesForModification.push_back(i);
        }
        // tri to edge
        for (int rot = 0; rot < 3; rot++) {
            const int rotOffset = idxLastTriXtoNewVy + 2 * rot * newNumEdgesY * newNumFacesX;
            for (int j = 0; j < 2; j++) {
                const int jOffset = j * newNumEdgesY * newNumFacesX;
                const int startIdxTriXtoEy = rotOffset + jOffset;
                const int idxFirstTriXtoNewEy = startIdxTriXtoEy + oldNumEdgesY * newNumFacesX;
                const int idxLastTriXtoNewEy = startIdxTriXtoEy + newNumEdgesY * newNumFacesX;
                for (int i = idxFirstTriXtoNewEy; i < idxLastTriXtoNewEy; i++) {
                    indicesForModification.push_back(i);
                }
            }
        }
    }
}

void NonWatertightMeshHandler::modifyEnergy(DeformationEnergy &defEnergy, Constraints &constr) {
    computeModifyIndices(constr);
    for(std::vector<int>::iterator it = std::begin(indicesForModification); it != std::end(indicesForModification); ++it) {
        const int index = *it;
        defEnergy.modifyEnergyVal(index, newEnergyValue);
    }

}

void NonWatertightMeshHandler::modifyEnergy(DeformationEnergy &defEnergy, Constraints &constr, float newEnergyVal) {
    newEnergyValue = newEnergyVal;
    modifyEnergy(defEnergy, constr);
}

void NonWatertightMeshHandler::modifyGamma(Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> &Gamma, Constraints &constr, int8_t newGammaVal) {
    computeModifyIndices(constr);
    for(std::vector<int>::iterator it = std::begin(indicesForModification); it != std::end(indicesForModification); ++it) {
        const int index = *it;
        Gamma(index, 0) = newGammaVal;
    }

}

bool NonWatertightMeshHandler::filledHolesOfShapeX() const {
    return didFillHolesOfShapeX;
}

bool NonWatertightMeshHandler::filledHolesOfShapeY() const {
    return didFillHolesOfShapeY;
}

Shape NonWatertightMeshHandler::getShapeXWithHoles() const {
    return shapeXHoles;
}

Shape NonWatertightMeshHandler::getShapeYWithHoles() const {
    return shapeYHoles;
}
