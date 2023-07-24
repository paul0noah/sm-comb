//
//  constraints.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 17.04.21.
//

#ifndef constraints_hpp
#define constraints_hpp

#include <Eigen/Sparse>
#include "helper/shape.hpp"
#include "shapeMatchModel/combinations/combinations.hpp"
#include "helper/utils.hpp"

/* Large edge product space:
   if we combine EX and EX(:, [2 1]) with EY
    => results in more, but less coupled constraints
 */
#define LARGE_EDGE_PRODUCT_SPACE

/*
 constraint matrix as well as the constraint vector only
 contain {-1, 0, 1} => int8 is more than sufficient
 
 since we later compute
 constrMatrix * Gamma = constrVector
 we have to store it in RowMajor format since we read the rows first
*/
typedef Eigen::SparseMatrix<int8_t, Eigen::RowMajor> SparseMatInt8;
typedef Eigen::SparseVector<int8_t, Eigen::RowMajor> SparseVecInt8;
// we need this to fill sparse matrices
typedef Eigen::Triplet<int8_t> TripletInt8;

// just for convinience ;)
typedef Eigen::Matrix<int, 1, 2> EDGE;



class Constraints {
private:
    int numFacesX;
    int numFacesY;
    int numFacesXxNumFacesY;
    int numEdgesX;
    int numEdgesY;
    int numVerticesX;
    int numVerticesY;
    int numProductFaces;
    int numProductEdges;
    int numProjections;
    
    Shape& shapeX;
    Shape& shapeY;
    
    Eigen::MatrixXi& FXCombo;
    Eigen::MatrixXi& FYCombo;
    
    bool computed;
    
    SparseMatInt8 constraintMatrix;
    SparseVecInt8 constraintVector;
    
    bool checkAndAddToDel(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, EDGE &eX, EDGE &eY, int e, int f, uint8_t &numAdded);
    Eigen::MatrixXi constructEdgeProductSpace(int &rowsE);
    SparseMatInt8 getDel();
    SparseMatInt8 getDelOptimized();
    void getDelOptimizedPRUNED(std::vector<TripletInt8>& delEntries, const Eigen::VectorX<bool>& pruneVec, const Eigen::MatrixXi& coarsep2pmap, const Eigen::MatrixXi& IXf2c, const Eigen::MatrixXi& IYf2c);
    
    // helper functions for getDelOptimized
    Eigen::MatrixXi constructEtoEdgesXTranslator();
    Eigen::MatrixXi constructEtoEdgesYTranslator();
    void findVerticesInEdgesMatrix(const int vertexIdx, Shape &shape, Eigen::MatrixXi &idxes, Eigen::MatrixXi &numIdxes);
    void findVerticesInEdgesMatrixPRUNED(const int vertexIdx, Shape &shape, Eigen::MatrixXi &idxes, Eigen::MatrixXi &numIdxes);
    void checkFace(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, int f);
    void checkFacePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long> cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, int f);
    void searchInNonDegenerateFaces(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEXinFX, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator);
    void searchInNonDegenerateFacesPRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long> cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEXinFX, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator);
    void searchInVertex2Triangle(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numF, int offset);
    void searchInVertex2TrianglePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long> cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numF, int offset);
    void searchInEdges2Triangle(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numE, int numF, int offset);
    void searchInEdges2TrianglePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long> cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numE, int numF, int offset);
    void searchDegEdgesInEdges2Triangle(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, Shape &shape, int numEX, int numFX, int numEY, int numFY, int offset);
    void searchDegEdgesInEdges2TrianglePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long> cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, Shape &shape, int numEX, int numFX, int numEY, int numFY, int offset);
    // end of helper functions
    
    SparseMatInt8 getProjection();
    void computeConstraintVector();
    void computeConstraints();
    
public:
    void init();
    Constraints(Shape &sX, Shape &sY, Combinations& c);
    SparseMatInt8 getConstraintMatrix();
    SparseVecInt8 getConstraintVector();
    void prune(const Eigen::VectorX<bool>& pruneVec);
    void computePrunedConstraints(const Eigen::VectorX<bool>& pruneVec, const Eigen::MatrixXi& coarsep2pmap, const Eigen::MatrixXi& IXf2c, const Eigen::MatrixXi& IYf2c);
    
};
#endif /* constraints_hpp */
