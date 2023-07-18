//
//  combinations.cpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#include "combinations.hpp"
#include "helper/utils.hpp"


/*function computeNonDegenerateCombinations(...)
 Consider the face matrix of shape X  and the one of shape Y
      Fx = [1 2 3;         Fy = [ 7  8  9;
            4 5 6];              10 11 12];
 Than the matchings are definied as:
       Matchings = [FxCombo FyCombo   ] =
                   [[1 2 3] [ 7  8  9]] // no rotation
                   [[1 2 3] [10 11 12]]
                   [[4 5 6] [ 7  8  9]]
                   [[4 5 6] [10 11 12]]
                   [[2 3 1] [ 7  8  9]] // first rotation
                   [[2 3 1] [10 11 12]]
                   [[5 6 4] [ 7  8  9]]
                   [[5 6 4] [10 11 12]]
                   [[3 1 2] [ 7  8  9]] // second rotation
                   [[3 1 2] [10 11 12]]
                   [[6 4 5] [ 7  8  9]]
                   [[6 4 5] [10 11 12]]
*/
Eigen::MatrixXi Combinations::computeNonDegenerateCombinations(Shape &shapeX, Shape &shapeY) {
    const int numFacesX = shapeX.getNumFaces();
    const int numFacesY = shapeY.getNumFaces();
    const int numFacesXY = numFacesX * numFacesY;
    const int numNonDegenerateCombos = 3 * numFacesXY;
    
    Eigen::MatrixXi MatchingsNonDegenerate(numNonDegenerateCombos, 6);

    Eigen::MatrixXi Fx(shapeX.getNumFaces(), 3);
    Fx = shapeX.getF();
    
    Eigen::ArrayXi rot1(3); rot1<<1, 2, 0;
    Eigen::ArrayXi rot2(3); rot2<<2, 0, 1;

    // combos for Fx
    MatchingsNonDegenerate.block(0, 0, numFacesXY, 3)              = utils::repelem(Fx, numFacesY, 1);
    MatchingsNonDegenerate.block(    numFacesXY, 0, numFacesXY, 3) = utils::repelem(Fx(Eigen::all, rot1), numFacesY, 1);
    MatchingsNonDegenerate.block(2 * numFacesXY, 0, numFacesXY, 3) = utils::repelem(Fx(Eigen::all, rot2), numFacesY, 1);
    
    //  combos for Fy
    MatchingsNonDegenerate.block(0, 3, numNonDegenerateCombos, 3) = shapeY.getF().replicate(3 * numFacesX, 1);
    return MatchingsNonDegenerate;
}


/* function  getDegenerateCombinations
% Consider the face matrix of shape A  and the one of shape B
%      Fa = [1 2 3;         Fb = [ 7  8  9;
%            4 5 6];              10 11 12];
% => EdgesA = [1 2;
%              2 3;
%              3 1;
%              4 5;
%              5 6;
%              6 4];
% => From one edge we can generate two different degenerate triangles:
% e.g edge = [1 2]
%     tri1 = [1 1 2]
%     tri2 = [1 2 2]
%
%       Matchings = [FxCombo FyCombo   ] =
% The degenerate matchings in A are definied as:
 % Triangle to vertex
 %                   [[1 1 1] [ 7  8  9]]
 %                   [[1 1 1] [10 11 12]]
 %                   [[2 2 2] [ 7  8  9]]
 %                   [[2 2 2] [10 11 12]]
 %                   [[3 3 3] [ 7  8  9]]
 %                   [[3 3 3] [10 11 12]]
 %                   [[4 4 4] [ 7  8  9]]
 %                   [[4 4 4] [10 11 12]]
 %                   [[5 5 5] [ 7  8  9]]
 %                   [[5 5 5] [10 11 12]]
 %                   [[6 6 6] [ 7  8  9]]
 %                   [[6 6 6] [10 11 12]]
 % Triangle to edge (X)
 % 1. Rotation
 %                   [[1 1 2] [ 7  8  9]]
 %                   [[1 1 2] [10 11 12]]
 %                   [[2 2 3] [ 7  8  9]]
 %                   [[2 2 3] [10 11 12]]
 %                   [[3 3 1] [ 7  8  9]]
 %                   [[3 3 1] [10 11 12]]
 %                   [[4 4 5] [ 7  8  9]]
 %                   [[4 4 5] [10 11 12]]
 %                   [[5 5 6] [ 7  8  9]]
 %                   [[5 5 6] [10 11 12]]
 %                   [[6 6 4] [ 7  8  9]]
 %                   [[6 6 4] [10 11 12]]
 %
 %                   [[1 2 2] [ 7  8  9]]
 %                   [[1 2 2] [10 11 12]]
 %                   [[2 3 3] [ 7  8  9]]
 %                   [[2 3 3] [10 11 12]]
 %                   [[3 1 1] [ 7  8  9]]
 %                   [[3 1 1] [10 11 12]]
 %                   [[4 5 5] [ 7  8  9]]
 %                   [[4 5 5] [10 11 12]]
 %                   [[5 6 6] [ 7  8  9]]
 %                   [[5 6 6] [10 11 12]]
 %                   [[6 4 4] [ 7  8  9]]
 %                   [[6 4 4] [10 11 12]]
 % 2. Rotation
 %                   [[1 1 2] [ 8  9  7]]
 %                   [[1 1 2] [11 12 10]]
 %                   [[2 2 3] [ 8  9  7]]
 % ...
 % 3. Rotation
 %                   [[1 1 2] [ 9  7  8]]
 %                   [[1 1 2] [12 10 11]]
 %                   [[2 2 3] [ 9  7  8]]
% Than the degenerate matchings in B are definied as:
%   ... analogously to A => just call the function twice ;)
%
% returns the matching matrix
 */
Eigen::MatrixXi Combinations::computeDegenerateCombinations(Shape &shapeX, Shape &shapeY) {
    
    const int numFacesX = shapeX.getNumFaces();
    const int numFacesY = shapeY.getNumFaces();
    const int numVerticesX = shapeX.getNumVertices();
    const int numEdgesX = shapeX.getNumEdges();
    const int numFacesXY = numFacesX * numFacesY;
    const int numTriangle2Vertex = numVerticesX * numFacesY;
    const int numTriangle2Edges = 3 * 2 * numEdgesX * numFacesY;
    const int numDegenerateCombos = numTriangle2Vertex + numTriangle2Edges;
    
    Eigen::MatrixXi MatchingsDegenerate(numDegenerateCombos, 6);
    
    Eigen::MatrixXi Fx = shapeX.getF();
    Eigen::MatrixXi Fy = shapeY.getF();
    
    // triangle to vertex
    // for shape X
    Eigen::MatrixXi Ftemp(numVerticesX, 1);
    utils::setLinspaced(Ftemp, 0);

    MatchingsDegenerate.block(0, 0, numTriangle2Vertex, 3) =
        utils::repelem(Ftemp.replicate(1, 3), numFacesY, 1);
    // for shape Y
    MatchingsDegenerate.block(0, 3, numTriangle2Vertex, 3) = Fy.replicate(numVerticesX, 1);
    // triangle to edge
    // for shape X
    MatchingsDegenerate.block(numTriangle2Vertex, 0, numTriangle2Edges, 3) = getTriangle2EdgeMatching(shapeX, numFacesY);
    // for shape Y
    Eigen::ArrayXi rot1(3); rot1 << 1, 2, 0;
    Eigen::ArrayXi rot2(3); rot2 << 2, 0, 1;
    // no rotation
    MatchingsDegenerate.block( numTriangle2Vertex, 3, 2 * numEdgesX * numFacesY, 3) = Fy.replicate(2 * numEdgesX, 1);
    // first rotation
    MatchingsDegenerate.block( numTriangle2Vertex + 2 * numEdgesX * numFacesY, 3, 2 * numEdgesX * numFacesY, 3) =
            Fy(Eigen::all, rot1).replicate(2 * numEdgesX, 1);
    // second rotation
    MatchingsDegenerate.block(numTriangle2Vertex + 4 * numEdgesX * numFacesY, 3, 2 * numEdgesX * numFacesY, 3) =
        Fy(Eigen::all, rot2).replicate(2 * numEdgesX, 1);

    return MatchingsDegenerate;
}


/* function getTriangle2EdgeMatching(Fa, nFb)
    computes the triangle to edge matching part of the matching matrix (only the left part for Fx, since the right part is simple repeption of the Fy matrix )
 */
Eigen::MatrixXi Combinations::getTriangle2EdgeMatching(Shape &shapeX, int numFacesY) {
    Eigen::ArrayXi edge2Triangle0(3); edge2Triangle0 << 0, 0, 1;
    Eigen::ArrayXi edge2Triangle1(3); edge2Triangle1 << 0, 1, 1;

    const int numFacesX = shapeX.getNumFaces();
    const int numEdgesX = shapeX.getNumEdges();
    const int numCombos = 3 * 2 * numEdgesX * numFacesY;
    
    Eigen::MatrixXi Ex = shapeX.getE();
    Eigen::MatrixXi FxCombo(numCombos, 3);
    FxCombo.block(0         , 0, numEdgesX, 3) = Ex(Eigen::all, edge2Triangle0);
    FxCombo.block(numEdgesX , 0, numEdgesX, 3) = Ex(Eigen::all, edge2Triangle1);

    Eigen::MatrixXi Ftemp(2 * numEdgesX, 3);
    Ftemp = FxCombo.block(0, 0, 2 * numEdgesX, 3);
    FxCombo.block(0, 0, 2 * numEdgesX * numFacesY, 3) = utils::repelem(Ftemp, numFacesY, 1);

    FxCombo.block(0, 0, numCombos, 3) = FxCombo.block(0, 0, 2 * numEdgesX * numFacesY, 3).replicate(3, 1);
    return FxCombo;
}

/* function computeCombinations()
 
 */
void Combinations::computeCombinations() {
    const int numFacesA = shapeA.getNumFaces();
    const int numFacesB = shapeB.getNumFaces();
    const int numVerticesA = shapeA.getNumVertices();
    const int numVerticesB = shapeB.getNumVertices();
    const int numEdgesA = shapeA.getNumEdges();
    const int numEdgesB = shapeB.getNumEdges();
    const int numFacesAB = numFacesA * numFacesB;
    const int numNonDegenerateCombos = 3 * numFacesAB;
    const int numDegenerateA = 3 * 2 * numEdgesA * numFacesB + numVerticesA * numFacesB;
    const int numDegenerateB = 3 * 2 * numEdgesB * numFacesA + numVerticesB * numFacesA;
    

    Eigen::MatrixXi nonDegenerateMatchings(numNonDegenerateCombos, 6);
    nonDegenerateMatchings = computeNonDegenerateCombinations(shapeA, shapeB);
    
    // we have to call this function twice since:
    // -> 1st call: degenerate cases in shapeA
    Eigen::MatrixXi degenerateMatchingsA(numDegenerateA, 6);
    degenerateMatchingsA = computeDegenerateCombinations(shapeA, shapeB);
    // -> 2nd call: degenerate caseses in shapeB
    Eigen::MatrixXi degenerateMatchingsB(numDegenerateB, 6);
    degenerateMatchingsB = computeDegenerateCombinations(shapeB, shapeA);
    
    
    // build the combo matrices
    // non degenerate cases
    FaCombo.block(0, 0, numNonDegenerateCombos, 3) = nonDegenerateMatchings.block(0, 0, numNonDegenerateCombos, 3);
    FbCombo.block(0, 0, numNonDegenerateCombos, 3) = nonDegenerateMatchings.block(0, 3, numNonDegenerateCombos, 3);
    // degenerate cases A
    FaCombo.block(numNonDegenerateCombos, 0, numDegenerateA, 3) = degenerateMatchingsA.block(0, 0, numDegenerateA, 3);
    FbCombo.block(numNonDegenerateCombos, 0, numDegenerateA, 3) = degenerateMatchingsA.block(0, 3, numDegenerateA, 3);
    // degenerate cases B (Attention: we have to be careful with the degenerateMatchingsB matrix since it is computed the other way around)
    FaCombo.block(numNonDegenerateCombos + numDegenerateA, 0, numDegenerateB, 3) = degenerateMatchingsB.block(0, 3, numDegenerateB, 3);
    FbCombo.block(numNonDegenerateCombos + numDegenerateA, 0, numDegenerateB, 3) = degenerateMatchingsB.block(0, 0, numDegenerateB, 3);
}

void Combinations::init() {
    const int numFacesA = shapeA.getNumFaces();
    const int numFacesB = shapeB.getNumFaces();
    const int numVerticesA = shapeA.getNumVertices();
    const int numVerticesB = shapeB.getNumVertices();
    const int numEdgesA = shapeA.getNumEdges();
    const int numEdgesB = shapeB.getNumEdges();
    const int numFacesAB = numFacesA * numFacesB;
    const int numNonDegenerateCombos = 3 * numFacesAB;
    const int numDegenerateA = 3 * 2 * numEdgesA * numFacesB + numVerticesA * numFacesB;
    const int numDegenerateB = 3 * 2 * numEdgesB * numFacesA + numVerticesB * numFacesA;

    combosComputed = false;
    numCombinations = numNonDegenerateCombos + numDegenerateA + numDegenerateB;
    FaCombo = Eigen::MatrixXi(numCombinations, 3);
    FbCombo = Eigen::MatrixXi(numCombinations, 3);
}

Combinations::Combinations(Shape& sA, Shape& sB) :
    shapeA(sA),
    shapeB(sB) {
    init();
}

Eigen::MatrixXi& Combinations::getFaCombo() {
    if(!combosComputed) {
        computeCombinations();
        combosComputed = true;
    }
    return FaCombo;
    
}

Eigen::MatrixXi& Combinations::getFbCombo() {
    if(!combosComputed) {
        computeCombinations();
        combosComputed = true;
    }
    return FbCombo;
}

void Combinations::prune(Eigen::VectorX<bool>& pruneVec) {
    const long numElements = pruneVec.nonZeros();
    Eigen::MatrixXi FaComboRed(numElements, 3);
    FaComboRed = -FaComboRed.setOnes();
    Eigen::MatrixXi FbComboRed(numElements, 3);
    FbComboRed = -FbComboRed.setOnes();

    long elementCounter = 0;
    for (int i = 0; i < FaCombo.rows(); i++) {
        if (pruneVec(i)) {
            FaComboRed(elementCounter, Eigen::all) = FaCombo(i, Eigen::all);
            FbComboRed(elementCounter, Eigen::all) = FbCombo(i, Eigen::all);
            elementCounter++;
            if (true) {
                if (elementCounter >= numElements) {
                    std::cout << "ERROR in Combinations::prune. Num Elements not correct" << std::endl;
                }
            }
        }
    }
    FaCombo = FaComboRed;
    FbCombo = FbComboRed;
}
