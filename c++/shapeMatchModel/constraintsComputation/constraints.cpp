//
//  constraints.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 17.04.21.
//

#include "constraints.hpp"
#include "helper/utils.hpp"
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <igl/repmat.h>
#include <igl/cumsum.h>

// indexing helpers
const EDGE idxEdge0((EDGE() << 0, 1).finished());
const EDGE idxEdge1((EDGE() << 1, 2).finished());
const EDGE idxEdge2((EDGE() << 2, 0).finished());
const EDGE idxEX((EDGE() << 0, 1).finished());
const EDGE idxEY((EDGE() << 2, 3).finished());
const EDGE minusEdge((EDGE() << 1, 0).finished());

#ifdef LARGE_EDGE_PRODUCT_SPACE
bool Constraints::checkAndAddToDel(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, EDGE &eX, EDGE &eY, int e, int f, uint8_t &numAdded) {
    if ( utils::allEqual( E(e, idxEX), eX) ) {
        if (utils::allEqual( E(e, idxEY), eY) ) {
            delEntries.push_back(TripletInt8(e, f, 1));
            numAdded++;
            return true;
        }
    }
    if ( utils::allEqual( E(e, idxEX), eX(minusEdge) ) ) {
        if ( utils::allEqual( E(e, idxEY), eY(minusEdge) ) ) {
            delEntries.push_back(TripletInt8(e, f, -1));
            numAdded++;
            return true;
        }
    }
    return false;
}

#else
bool Constraints::checkAndAddToDel(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, EDGE &eX, EDGE &eY, int e, int f, uint8_t &numAdded) {
    
    if ( utils::allEqual( E(e, idxEX), eX) ) {
        if (utils::allEqual( E(e, idxEY), eY ) ) {
            delEntries.push_back(TripletInt8(e, f, 1));
            numAdded++;
            return true;
        }
        if ( utils::allEqual( E(e, idxEY), eY(minusEdge) ) ) {
            delEntries.push_back(TripletInt8(e, f, -1));
            numAdded++;
            return true;
        }
    }
    if ( utils::allEqual( E(e, idxEX), eX(minusEdge) ) ) {
        if ( utils::allEqual( E(e, idxEY), eY(minusEdge) ) ) {
            delEntries.push_back(TripletInt8(e, f, -1));
            numAdded++;
            return true;
        }
        if ( utils::allEqual( E(e, idxEY), eY ) ) {
            delEntries.push_back(TripletInt8(e, f, 1));
            numAdded++;
            return true;
        }
    }
    return false;
}
#endif


Eigen::MatrixXi Constraints::constructEdgeProductSpace(int &rowsE) {
    // build the product edge space E
    Eigen::MatrixXi EX = shapeX.getE();
    Eigen::MatrixXi EXdegenerate(numVerticesX, 1);
    utils::setLinspaced(EXdegenerate, 0);
    Eigen::MatrixXi EY = shapeY.getE();
    Eigen::MatrixXi EYdegenerate(numVerticesY, 1);
    utils::setLinspaced(EYdegenerate, 0);
    /*
     we have to build up all combinations of [EX; EXdegenerate] with [EY; EYdegenerate].
     Note: we will not combine EXdegenerate with EYdegenerate
     since in the product face space there are only either in X degenerate or in
     Y degenerate faces.
     => combining EXdegenerate with EYdegenerate would give us zero rows in the
     constraint matrix which the solver unnecessarily has to remove
     */
#ifdef LARGE_EDGE_PRODUCT_SPACE
    /* for the large edge product space we comnine EX with EY and addtionally
        EX(:, [2 1]) with EY
     */
    rowsE = 2 * numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
    Eigen::MatrixXi E(rowsE, 4);
    // non-degenerate
    const int numNonDegenerate = numEdgesX * numEdgesY;
    E.block(0, 0, numNonDegenerate, 2) = utils::repelem(EX, numEdgesY, 1);
    E.block(numNonDegenerate, 0, numNonDegenerate, 2) = utils::repelem(EX(Eigen::all, minusEdge), numEdgesY, 1);
    E.block(0, 2, 2 * numNonDegenerate, 2) = EY.replicate(2 * numEdgesX, 1);
    
    // degenerate X
    const int numDegenerateX = numEdgesY * numVerticesX;
    E.block(2 * numNonDegenerate, 0, numDegenerateX, 2) =
            utils::repelem(EXdegenerate, numEdgesY, 2);
    E.block(2 * numNonDegenerate, 2, numDegenerateX, 2) =
            EY.replicate(numVerticesX, 1);
    
    // degenerate Y
    const int numDegenerateY = numEdgesX * numVerticesY;
    E.block(2 * numNonDegenerate + numDegenerateX, 0, numDegenerateY, 2) = utils::repelem(EX, numVerticesY, 1);
    E.block(2 * numNonDegenerate + numDegenerateX, 2, numDegenerateY, 2) = EYdegenerate.replicate(numEdgesX, 2);
    
    
#else
    rowsE = numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
    Eigen::MatrixXi E(rowsE, 4);
    // non-degenerate
    const int numNonDegenerate = numEdgesX * numEdgesY;
    E.block(0, 0, numNonDegenerate, 2) = utils::repelem(EX, numEdgesY, 1);
    E.block(0, 2, numNonDegenerate, 2) = EY.replicate(numEdgesX, 1);
    // degenerate X
    const int numDegenerateX = numEdgesY * numVerticesX;
    E.block(numNonDegenerate, 0, numDegenerateX, 2) =
            utils::repelem(EXdegenerate, numEdgesY, 2);
    E.block(numNonDegenerate, 2, numDegenerateX, 2) =
            EY.replicate(numVerticesX, 1);
    // degenerate Y
    const int numDegenerateY = numEdgesX * numVerticesY;
    E.block(numNonDegenerate + numDegenerateX, 0, numDegenerateY, 2) = utils::repelem(EX, numVerticesY, 1);
    E.block(numNonDegenerate + numDegenerateX, 2, numDegenerateY, 2) = EYdegenerate.replicate(numEdgesX, 2);
    
#endif
    return E;
}


/* function getDel()
  We can construct the del part as follows:
  iterate through all triangles in the product space => column pos in del
    iterate through all edges in the in the edge product space
        check both orientations of the edge
        if oriented as in edge list => 1 in del matrix
        if not oriented as in edge list => -1 in del matrix
 */
SparseMatInt8 Constraints::getDel() {
    // allocate del
    const int numNonZerosDel = numProductFaces * 3;
    SparseMatInt8 del(numProductEdges, numProductFaces);
    
    std::vector<TripletInt8> delEntries;
    delEntries.reserve(numProductFaces * 3);
    
    int rowsE;
    Eigen::MatrixXi E;
    E = constructEdgeProductSpace(rowsE);

    // intermediate variables
    EDGE edgeX0, edgeX1, edgeX2, edgeY0, edgeY1, edgeY2;

    uint8_t numAdded;
    
    for (int f = 0; f < numProductFaces; f++) {
        // extract product edges from the product triangles
        edgeX0 = FXCombo(f, idxEdge0);
        edgeX1 = FXCombo(f, idxEdge1);
        edgeX2 = FXCombo(f, idxEdge2);
        edgeY0 = FYCombo(f, idxEdge0);
        edgeY1 = FYCombo(f, idxEdge1);
        edgeY2 = FYCombo(f, idxEdge2);
        
        numAdded = 0;
        for( int e = 0; e < numProductEdges; e++) {

            checkAndAddToDel(delEntries, E, edgeX0, edgeY0, e, f, numAdded);
            checkAndAddToDel(delEntries, E, edgeX1, edgeY1, e, f, numAdded);
            checkAndAddToDel(delEntries, E, edgeX2, edgeY2, e, f, numAdded);
             
            // there are only three entries per column
            if (numAdded >= 3) {
                break;
            }
        }
    }
    del.setFromTriplets(delEntries.begin(), delEntries.end());
    return del;
}

/*function getProjection
    computes the projection of the product space
    
    Whenever a face in the product space consists of either a face of shapeX or a face of shapY
    there is a 1 in the respective projection matrix
 */
SparseMatInt8 Constraints::getProjection() {
    const int numFacesXxNumFacesY = numFacesX * numFacesY;
    const int numNonDegenerate = 3 * numFacesXxNumFacesY;
    const int numDegenerateX = 9 * numFacesXxNumFacesY + numVerticesX * numFacesY;
    const int numDegenerateY = 9 * numFacesXxNumFacesY + numVerticesY * numFacesX;;
    const int numDegenerate = numDegenerateX + numDegenerateY;
    const int nnzProj = 2 * numNonDegenerate + numDegenerate;
    SparseMatInt8 proj(numFacesX + numFacesY, numProductFaces);
    proj.reserve(nnzProj);
    
    SparseMatInt8 eyeNumFacesX(numFacesX, numFacesX); eyeNumFacesX.setIdentity();
    SparseMatInt8 eyeNumFacesY(numFacesY, numFacesY); eyeNumFacesY.setIdentity();

    // Non degenerate
    SparseMatInt8 ProjXNDegenerate = utils::repmat(utils::sprepelem(eyeNumFacesX, 1, numFacesY), 1, 3);
    SparseMatInt8 ProjYNDegenerate = utils::repmat(eyeNumFacesY, 1, 3 * numFacesX);
    
    // Degenerate
    Eigen::SparseMatrix<int8_t, Eigen::ColMajor> ProjXDegenerate(numFacesX, numDegenerate);
    ProjXDegenerate.rightCols(numDegenerateY) = utils::repmat(eyeNumFacesX, 1, 9 * numFacesY + numVerticesY);
    Eigen::SparseMatrix<int8_t, Eigen::ColMajor> ProjYDegenerate(numFacesX, numDegenerate);
    ProjYDegenerate.leftCols(numDegenerateX)  = utils::repmat(eyeNumFacesY, 1, 9 * numFacesX + numVerticesX);
    
    Eigen::SparseMatrix<int8_t, Eigen::ColMajor> ProjX(numFacesX, numNonDegenerate + numDegenerate);
    ProjX.leftCols(numNonDegenerate) = ProjXNDegenerate;
    ProjX.rightCols(numDegenerate)   = ProjXDegenerate;
    Eigen::SparseMatrix<int8_t, Eigen::ColMajor> ProjY(numFacesY, numNonDegenerate + numDegenerate);
    ProjY.leftCols(numNonDegenerate) = ProjYNDegenerate;
    ProjY.rightCols(numDegenerate)   = ProjYDegenerate;
    
    proj.topRows(numFacesX) = ProjX;
    proj.bottomRows(numFacesY) = ProjY;

    return proj;
}

void Constraints::computeConstraintVector() {
    constraintVector.reserve(numProjections);
    
    for (int k = numProductEdges; k < numProductEdges + numProjections; k++) {
        constraintVector.insert(k) = 1;
    }
}

void Constraints::computeConstraints() {
    const int rowsDel = numProductEdges;
    const int rowsProjection = numFacesX + numFacesY;

    constraintMatrix.topRows(rowsDel) = getDelOptimized();
    
    constraintMatrix.bottomRows(rowsProjection) = getProjection();
    
    computeConstraintVector();
}

void Constraints::computePrunedConstraints(const Eigen::VectorX<bool>& pruneVec, const Eigen::MatrixXi& coarsep2pmap, const Eigen::MatrixXi& IXf2c, const Eigen::MatrixXi& IYf2c) {


    numProductFaces = FXCombo.rows();

    std::vector<TripletInt8> constrEntries;
    constrEntries.reserve(numProductFaces * 3 + numProductFaces * 2);


    // del
    getDelOptimizedPRUNED(constrEntries, pruneVec, coarsep2pmap, IXf2c, IYf2c);

    // projection
    SparseMatInt8 proj = getProjection();
    Eigen::VectorX<long> cumSumPruneVec;
    igl::cumsum(pruneVec.cast<long>(), 1, cumSumPruneVec);
    for (long e = 0; e < proj.outerSize(); ++e) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(proj, e); it; ++it) {
            const int f = it.index();
            const int8_t val = it.value();
            if (pruneVec(f)) {
                const int colidx = cumSumPruneVec(f) - 1;
                constrEntries.push_back(TripletInt8(e + numProductEdges, colidx, val));
            }
        }
    }


    // create the matrix
    constraintMatrix = SparseMatInt8(numProductEdges + numProjections, numProductFaces);
    constraintMatrix.setFromTriplets(constrEntries.begin(), constrEntries.end());


    // Constraints vector
    constraintVector = SparseVecInt8(numProductEdges + numProjections);
    constraintVector.reserve(numProjections);
    for (int k = numProductEdges; k < numProductEdges + numProjections; k++) {
        constraintVector.insert(k) = 1;
    }

    computed = true;
}

void Constraints::init() {
    numFacesX = shapeX.getNumFaces();
    numFacesY = shapeY.getNumFaces();
    numFacesXxNumFacesY = numFacesX * numFacesY;
    numEdgesX = shapeX.getNumEdges();
    numEdgesY = shapeY.getNumEdges();
    numVerticesX = shapeX.getNumVertices();
    numVerticesY = shapeY.getNumVertices();

    numProductFaces = 21 * numFacesX * numFacesY + numVerticesX * numFacesY + numVerticesY * numFacesX;
#ifdef LARGE_EDGE_PRODUCT_SPACE
    numProductEdges = 2 * numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
#else
    numProductEdges = numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
#endif
    numProjections = numFacesX + numFacesY;

    constraintMatrix = SparseMatInt8(numProductEdges + numProjections, numProductFaces);
    constraintVector = SparseVecInt8(numProductEdges + numProjections);
}

Constraints::Constraints(Shape &sX, Shape &sY, Combinations& c) :
    shapeX(sX),
    shapeY(sY),
    FXCombo(c.getFaCombo()),
    FYCombo(c.getFbCombo()),
    computed(false) {
        
    init();
}

SparseMatInt8 Constraints::getConstraintMatrix() {
    if(!computed) {
        computeConstraints();
        computed = true;
    }
    return constraintMatrix;
}

SparseVecInt8 Constraints::getConstraintVector() {
    if(!computed) {
        computeConstraints();
        computed = true;
    }
    return constraintVector;
}


void Constraints::prune(const Eigen::VectorX<bool>& pruneVec) {
    const long numElements = pruneVec.cast<long>().sum();
    Eigen::VectorX<long> cumSumPruneVec;
    igl::cumsum(pruneVec.cast<long>(), 1, cumSumPruneVec);
    std::vector<TripletInt8> constrEntries;
    constrEntries.reserve(constraintMatrix.nonZeros() * 0.25);


    unsigned long rowOffset = 0;
    unsigned long maxColIdx = 0;
    unsigned long minColIdx = 10000000;
    for (long e = 0; e < constraintMatrix.outerSize(); ++e) {
        unsigned int numNonZerosRow = 0;
        int8_t prevVal = 0;
        bool signflip = false;
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constraintMatrix, e); it; ++it) {
            const int f = it.index();
            const int8_t val = it.value();
            if (pruneVec(f)) {
                if (prevVal == 0) {
                    prevVal = val;
                }
                if (prevVal != val) {
                    signflip = true;
                }
            }
        }
        if (signflip || e >= numProductEdges) { // make sure trivial constraints are pruned away
            for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constraintMatrix, e); it; ++it) {
                const int f = it.index();
                const int8_t val = it.value();
                if (pruneVec(f)) {
                    numNonZerosRow++;
                    const int colidx = cumSumPruneVec(f) - 1;
                    constrEntries.push_back(TripletInt8(e - rowOffset, colidx, val));
                    if (colidx > maxColIdx) {
                        maxColIdx = colidx;
                    }
                    if (colidx < minColIdx) {
                        minColIdx = colidx;
                    }
                }
            }
        }
        if (numNonZerosRow == 0) {
            rowOffset++;
        }
    }

    constraintMatrix = SparseMatInt8(numProductEdges - rowOffset + numProjections, numElements);
    constraintMatrix.setFromTriplets(constrEntries.begin(), constrEntries.end());

    /*for (long e = 0; e < constraintMatrix.outerSize(); ++e) {
        unsigned int numNonZerosRow = 0;
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constraintMatrix, e); it; ++it) {
                numNonZerosRow++;
        }
        if (!numNonZerosRow) {
            std::cout << "Row " << e << " of " << constraintMatrix.outerSize() << std::endl;
        }
    }*/

    constraintVector = SparseVecInt8(numProductEdges - rowOffset + numProjections);
    constraintVector.reserve(numProjections);
    for (int k = numProductEdges - rowOffset; k < numProductEdges - rowOffset + numProjections; k++) {
        constraintVector.insert(k) = 1;
    }
}
