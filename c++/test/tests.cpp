//
//  testEnergies.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 15.04.21.
//
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "tests.hpp"
#include "helper/shape.hpp"
#include "helper/nonWatertightMeshHandler.hpp"
#include "shapeMatchModel/energyComputation/deformationEnergy.hpp"
#include "shapeMatchModel/constraintsComputation/constraints.hpp"
#include "shapeMatchModel/ShapeMatchModel.hpp"
#include "test/exampleshapes.hpp";



namespace test {

void generateExamples(const std::string path) {
    std::string example = "example";
    Shape shapeA = getDavid(60);
    shapeA.reOrderTriangulation();
    Shape shapeB= getDavid(50);
    shapeB.reOrderTriangulation();
    float squashFactor = shapeA.squashInUnitBox();
    shapeB.squash(squashFactor);
    shapeB.translate( (Eigen::Vector3f() << 1.2, 0, 0).finished() );
    shapeA.writeToFile(path + example + "_X.ply");
    shapeB.writeToFile(path + example + "_Y.ply");

    std::string partialExample = "partialExample";
    Shape shape2h = getHorse2holes();
    shape2h.writeToFile(path + partialExample + "_Xoriginal.ply");
    Shape shape2c = getHorse2Closed();
    shape2c.writeToFile(path + partialExample + "_X.ply");
    Shape shape1h = getHorse1holes();
    shape1h.writeToFile(path + partialExample + "_Yoriginal.ply");
    Shape shape1c = getHorse1Closed();
    shape1c.writeToFile(path + partialExample + "_Y.ply");
}

Eigen::MatrixXf getEnergyFromFile(std::string filename) {
    Eigen::MatrixXf energy(4812, 1);
    std::ifstream file(filename);
    if (!file) {
        std::cout << "File: " << filename << " not found" << std::endl;
    }
    std::string line;
    int i = 0;
    while ( std::getline(file, line) ) {
        energy(i) = std::stof(line);
        i++;
    }
    file.close();
    return energy;
}
SparseMatInt8 getDelFromFile(std::string filename, size_t delRows) {
    SparseMatInt8 del(delRows, 4812);
    std::vector<TripletInt8> delEntries;
    delEntries.reserve(4812 * 3 + 5460);
    
    std::ifstream file(filename);
    if (!file) {
        std::cout << "File: " << filename << " not found" << std::endl;
    }
    std::string line;
    int row, col, pos, i=0;
    int8_t val;
    int maxRow = 0, maxCol = 0;
    while ( std::getline(file, line) ) {
        pos = line.find(",");
        row = std::stoi(line.substr(0, pos)) - 1;
        line = line.substr(pos + 1);
        
        pos = line.find(",");
        col = std::stoi(line.substr(0, pos)) - 1;
        line = line.substr(pos + 1);
        
        val = std::stoi(line);
        
        delEntries.push_back(TripletInt8(row, col, val));
        
        i++;
    }
    file.close();
    //assert(i == (4812 * 3 + 5460));
    del.setFromTriplets(delEntries.begin(), delEntries.end());
    return del;
}

void deformationEnergy() {
    std::cout << "+++ Testing Deformation Energy +++" << std::endl;
    Shape shapeA = getTestShapeBase();
    Shape shapeB = getTestShapeCplxTriTop();
    Eigen::MatrixXf defEnergyMatlab(4812, 1);
    defEnergyMatlab = getEnergyFromFile("test/data/deformationEnergy");
    
    Combinations combos = Combinations(shapeA, shapeB);
    combos.computeCombinations();
    
    DeformationEnergy deformationEnergy = DeformationEnergy(shapeA, shapeB, combos);
    
    
    Constraints constr = Constraints(shapeA, shapeB, combos);
    constr.getConstraintMatrix();
    
    
    Eigen::MatrixXf defEnergy(4812, 1);
    defEnergy = deformationEnergy.get();
    
    bool correcteness = true;
    float errorTol = 1e-4;
    float maxErr = 0;
    float err = 0;
    
    for(int i = 0; i < defEnergy.rows(); i++) {
        err = std::abs(defEnergy(i) - defEnergyMatlab(i));
        if (err > maxErr) {
            maxErr = err;
        }
        if (err > errorTol) {
            correcteness = false;
        }
    }
    std::string output = correcteness ? "CORRECT" : "WRONG";
    std::cout << " > Deformation Energy is implemented " <<
        output << " with max error: " << maxErr << std::endl;
     
}

void constraints() {
    Shape shapeA = getTestShapeBase();
    Shape shapeB = getTestShapeCplxTriTop();
#ifdef LARGE_EDGE_PRODUCT_SPACE
    SparseMatInt8 constrMatlab(1416, 4812);
    constrMatlab = getDelFromFile("test/data/constrLarge", 1416);
#else
    SparseMatInt8 constrMatlab(930, 4812);
    constrMatlab = getDelFromFile("test/data/constrSmall", 930);
#endif
    
    Combinations combos = Combinations(shapeA, shapeB);
    combos.computeCombinations();
    
    Constraints constr = Constraints(shapeA, shapeB, combos);
    SparseMatInt8 constraints = constr.getConstraintMatrix();

#ifdef LARGE_EDGE_PRODUCT_SPACE
    SparseMatInt8 del = constraints.topRows(1386);
    SparseMatInt8 projection = constraints.bottomRows(30);
    SparseMatInt8 delMatlab = constrMatlab.topRows(1386);
    SparseMatInt8 projectionMatlab = constrMatlab.bottomRows(30);
#else
    SparseMatInt8 del = constraints.topRows(900);
    SparseMatInt8 projection = constraints.bottomRows(30);
    SparseMatInt8 delMatlab = constrMatlab.topRows(900);
    SparseMatInt8 projectionMatlab = constrMatlab.bottomRows(30);
#endif
    
    // del
    bool equal = true;
    for (int k = 0; k < del.outerSize(); ++k) {
        int i = 0;
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(del,k); it; ++it) {
            if(it.value() != delMatlab.coeff(k, it.col())) {
                equal &= false;
            }
            i++;
        }
    }
    if(equal){
        std::cout << "Del is implemented correctly" << std::endl;
    }
    else {
        del = (del.toDense().cwiseAbs() - delMatlab.toDense().cwiseAbs()).sparseView();
        int numberOfDifferentElements = del.nonZeros();
        std::cout << "Del is not implemented correctly (" << numberOfDifferentElements << " different elements)" << std::endl;
    }
    
    // projection
    equal = true;
    for (int k = 0; k < projection.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(projection,k); it; ++it) {
            if(it.value() != projectionMatlab.coeff(k, it.col())) {
                //std::cout << (int) it.value() << "  " << (int)delMatlab.coeff(k, i) << std::endl;
                equal &= false;
            }
        }
    }
    if(equal){
        std::cout << "Projection is implemented correctly" << std::endl;
    }
    else {
        projection = (projection.toDense().cwiseAbs() - projectionMatlab.toDense().cwiseAbs()).sparseView();
        int numberOfDifferentElements = projection.nonZeros();
        std::cout << "Projection is not implemented correctly (" << numberOfDifferentElements << " different elements)" << std::endl;
    }
    
}


void saveLp() {
    Shape shapeA = getTestShapeBase();
    Shape shapeB = getTestShapeCplxTriTop();
    
    ShapeMatchModel model = ShapeMatchModel(shapeA, shapeB);
    
    model.saveAsLp("TestShape_TestShapeCplxTriTop.lp");
}

void saveIlpAsLp() {
    Shape shapeA = getTestShapeBase();
    Shape shapeB = getTestShapeCplxTriTop();
    
    ShapeMatchModel model = ShapeMatchModel(shapeA, shapeB);
    
    model.saveIlpAsLp("TestShape_TestShapeCplxTriTop.lp");
}

void primalHeuristic() {
    Shape shapeA = getTestShapeBase();
    Shape shapeB = getTestShapeCplxTriTop();
    
    ShapeMatchModel model = ShapeMatchModel(shapeA, shapeB);
    
    
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> G = model.solve();
    
}

void plot() {
    Shape shapeA = getTestShapeBase();
    Shape shapeB = getTestShapeCplxTriTop();
    float squashFactor = shapeA.squashInUnitBox();
    shapeB.squash(squashFactor);
    shapeB.translate( (Eigen::Vector3f() << 1.2, 0, 0).finished() );
    ShapeMatchModel model = ShapeMatchModel(shapeA, shapeB);
    
    
    const size_t numRowsGamma = 4812;
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> Gamma(numRowsGamma, 1);
    Gamma.setZero();
    
    // Solution computed by Gurobi and Matlab
    const int gammaOnesIdx[] = {0, 19, 38, 57, 76, 95, 114, 134,
        155, 195, 214, 611, 715, 716, 1609, 1611, 1612, 2046};
    const size_t numOnes = 18;
    for (int i = 0; i < numOnes; i++) {
        Gamma(gammaOnesIdx[i], 0) = 1;
    }
    
    model.plotSolution(Gamma.sparseView());
}

void solveDualProblem() {
    Shape shapeA = getTestShapeBase();
    Shape shapeB = getTestShapeCplxTriTop();
    ShapeMatchModel model = ShapeMatchModel(shapeA, shapeB);

    
    std::cout << model.getMinMarginals() << std::endl;
}

} // namespace test
