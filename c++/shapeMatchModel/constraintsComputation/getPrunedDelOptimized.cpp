//
//  getDelOptimized.cpp
//  constraintsComputation
//
//  Created by Paul Rötzer on 14.05.21.
//

#include <stdio.h>
#include "constraints.hpp"
#include "helper/utils.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <iostream>
#include <igl/cumsum.h>
#include <igl/boundary_loop.h>
#include <tsl/robin_set.h>
#include <unordered_set>
#if defined(_OPENMP)
    #include <omp.h>
#else
    #warning OPENMP not found on this machine. Consider installing it for improved performance
#endif


const EDGE idxEdge0((EDGE() << 0, 1).finished());
const EDGE idxEdge1((EDGE() << 1, 2).finished());
const EDGE idxEdge2((EDGE() << 2, 0).finished());
inline void Constraints::checkFacePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long>& cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, int f) {

    if (!pruneVec(f)) {
        return;
    }

    f = cumSumPruneVec(f) - 1;

    /*if (f > numProductFaces ) {
        std::cout << "f to large" << std::endl;
        return;
    }

    if (e > numProductEdges ) {
        std::cout << "e to large" << std::endl;
        return;
    }*/
    
    uint8_t numAdded = 0;
    
    EDGE eX = FXCombo(f, idxEdge0);
    EDGE eY = FYCombo(f, idxEdge0);
    if (checkAndAddToDel(delEntries, E, eX, eY, e, f, numAdded)) {
        return;
    }
    
    eX = FXCombo(f, idxEdge1);
    eY = FYCombo(f, idxEdge1);
    if (checkAndAddToDel(delEntries, E, eX, eY, e, f, numAdded)) {
        return;
    }
    
    eX = FXCombo(f, idxEdge2);
    eY = FYCombo(f, idxEdge2);
    checkAndAddToDel(delEntries, E, eX, eY, e, f, numAdded);
}


static int oldVertexIdx = -1;
static Eigen::MatrixXi IDX(20, 2);
static Eigen::MatrixXi numIDX(1, 2);
void Constraints::findVerticesInEdgesMatrixPRUNED(const int vertexIdx, Shape &shape, Eigen::MatrixXi &idxes, Eigen::MatrixXi &numIdxes) {
    assert(idxes.rows() == IDX.rows());
    assert(idxes.cols() == IDX.cols());
    assert(numIdxes.rows() == numIDX.rows());
    assert(numIdxes.cols() == numIDX.cols());

    // using Openmp we cannot gurantee access to IDX and numIDX in order so we search each time for vertex
    // not super efficient but thread safe
#if !defined(_OPENMP)
    // reset
    if (vertexIdx == -1) {
        oldVertexIdx = -1;
        numIDX << 0, 0;
        return;
    }

    // if there is no change return the old value
    if(vertexIdx == oldVertexIdx){
        idxes = IDX;
        numIdxes = numIDX;
        return;
    }
#endif
    Eigen::MatrixXi EDG = shape.getE();
    int nEDG = shape.getNumEdges();

    numIdxes.setZero();

    // iterate through the edges of shape
    for(int e = 0; e < nEDG; e++) {
        if(EDG(e, 0) == vertexIdx) {
            idxes(numIdxes(0, 0), 0) = e;
            numIdxes(0, 0) = numIdxes(0, 0) + 1;
        }
        if(EDG(e, 1) == vertexIdx) {
            idxes(numIdxes(0, 1), 1) = e;
            numIdxes(0, 1) = numIdxes(0, 1) + 1;
        }

    }
#if !defined(_OPENMP)
    IDX = idxes;
    numIDX = numIdxes;
    oldVertexIdx = vertexIdx;
#endif
}


/*
 searchDegEdgesXInEdgesX2TriangleY:
     offset = 3 * numFacesXxNumFacesY + numVerticesX * numFacesY;
     => searchDegEdgesInEdges2Triangle(delEntries, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, shapeX, numEdgesX, numFacesX, numEdgesY, numFaceY, offset)
 
 searchDegEdgesYInEdgesY2TriangleX:
    offset = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY + ...
             numVerticesY * numFacesX;
    => searchDegEdgesInEdges2Triangle(delEntries, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, shapeY, numEdgesY, numFacesY, numEdgesX, numFaceX, offset)
 */
void Constraints::searchDegEdgesInEdges2TrianglePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long>& cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, Shape &shape, int numEX, int numFX, int numEY, int numFY, int offset) {
    
    const int vertexIdx = eToEXTranslator(e);
    Eigen::MatrixXi idxes(IDX.rows(), 2);
    Eigen::MatrixXi numIdxes(1, 2);
    // we don't know how often a vertex appears in the corresponding edge
    // matrix of a shape => this functions searches for the vertices
    findVerticesInEdgesMatrixPRUNED(vertexIdx, shape, idxes, numIdxes);
    
    // the i loop accounts for the possible rotations
    for (int i = 0; i < 3; i++) {
        int rotPos = i * 2 * numEX * numFY;
        
        // the vertex index can either be in the first or the second element
        // of an edge => k == 0 first element; k == 1 second element
        for(int k = 0; k < 2; k++) {
            for (int idx = 0; idx < numIdxes(k); idx++) {
                int edge2TriPos = idxes(idx, k) * numFY + k * numEX * numFY;

                int f = offset + edge2TriPos + rotPos;
            
                // first orientation of the edge
                int f1 = f + LocEYinFY(eToEYTranslator(e), 0);
                checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f1);
                
                // first orientation of the edge
                int f2 = f + LocEYinFY(eToEYTranslator(e), 1);
                checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f2);
            }
        }
    }
}



/*
 searchInEdgesX2TriangleY
    offset = 3 * numFacesXxNumFacesY + numVerticesX * numFacesY;
    => searchInEdges2Triangle(delEntries, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, numEdgesX, numFacesY, offset)
 searchInEdgesY2TriangleX
    offset = offset = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY + numVerticesY * numFacesX;
    => searchInEdges2Triangle(delEntries, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, numEdgesY, numFacesX, offset)
 */
void Constraints::searchInEdges2TrianglePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long>& cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numE, int numF, int offset) {
    
    // the i loop accounts for the possible rotations
    for (int i = 0; i < 3; i++) {
        int rotPos = i * 2 * numE * numF;
        
        // the ii loop accounts for different possibilities the triangles
        // can be constructet from one edge
        for (int ii = 0; ii < 2; ii++) {
            int edge2TriPos = ii * numE * numF;
            
            int f = offset + edge2TriPos + rotPos + eToEXTranslator(e) * numF;
                
            // first orientation of the edge
            int f1 = f + LocEYinFY(eToEYTranslator(e), 0);
            checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f1);
             
            // second orientation of the edge
            int f2 = f + LocEYinFY(eToEYTranslator(e), 1);
            checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f2);
        }
    }
}


/*
 searchInVertexX2TriangleY:
 offset 3 * numFacesXxNumFacesY
   => searchInVertex2Triangle(delEntries, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, numFacesY, offset);
 searchInVertexY2TriangleX:
 offset = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY;
   => searchInVertex2Triangle(delEntries, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, numFacesX, offset);
 */
void Constraints::searchInVertex2TrianglePRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long>& cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numF, int offset) {
    
    int f = offset + eToEXTranslator(e) * numF;
    
    // first orientation of the edge
    int f1 = f + LocEYinFY(eToEYTranslator(e), 0);
    
    checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f1);
    
    // second orientation of the edge
    int f2 = f + LocEYinFY(eToEYTranslator(e), 1);
    checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f2);
    
}

void Constraints::searchInNonDegenerateFacesPRUNED(const Eigen::VectorX<bool>& pruneVec, const Eigen::VectorX<long>& cumSumPruneVec, std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEXinFX, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator) {
    // this i-loop accounts for the different rotations of the triangles
    for (int i = 0; i < 3; i++) {
        int f = i * numFacesXxNumFacesY;
        // each product edge may be in one of 4 possible triangles
        
        int f11 = f + LocEXinFX(eToEXTranslator(e), 0) * numFacesY + LocEYinFY(eToEYTranslator(e), 0);
        checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f11);
        
        int f12 = f + LocEXinFX(eToEXTranslator(e), 0) * numFacesY + LocEYinFY(eToEYTranslator(e), 1);
        checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f12);
        
        int f21 = f + LocEXinFX(eToEXTranslator(e), 1) * numFacesY + LocEYinFY(eToEYTranslator(e), 0);
        checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f21);
        
        int f22 = f + LocEXinFX(eToEXTranslator(e), 1) * numFacesY + LocEYinFY(eToEYTranslator(e), 1);
        checkFacePRUNED(pruneVec, cumSumPruneVec, delEntries, E, e, f22);
    }
}

void pruneEdgeProductSpace(Eigen::MatrixXi& E,
                           Eigen::MatrixXi& eToEXTranslator,
                           Eigen::MatrixXi& eToEYTranslator,
                           const Eigen::MatrixXi& coarsep2pmap,
                           const Eigen::MatrixXi& IXf2c,
                           const Eigen::MatrixXi& IYf2c,
                           Shape shapeX,
                           Shape shapeY) {

    const Eigen::MatrixX<bool> p2pmatrix = utils::computeP2PMat(shapeX, shapeY, coarsep2pmap, IXf2c, IYf2c, 2);

    Eigen::MatrixXi prunedE(E.rows(), 4);
    Eigen::MatrixXi prunedEToEXTranslator(E.rows(), 1);
    Eigen::MatrixXi prunedEToEYTranslator(E.rows(), 1);
    long numE = 0;
    for (long e = 0; e < E.rows(); e++) {
        const bool firstp2p = p2pmatrix( E(e, 0), E(e, 2) );
        const bool seconp2p = p2pmatrix( E(e, 1), E(e, 3) );
        const bool thirdp2p = p2pmatrix( E(e, 0), E(e, 3) );
        const bool fourthp2p = p2pmatrix( E(e, 1), E(e, 2) );
        if (firstp2p || seconp2p || thirdp2p || fourthp2p) {
            prunedE(numE, Eigen::all) = E(e, Eigen::all);
            prunedEToEXTranslator(numE, 0) = eToEXTranslator(e, 0);
            prunedEToEYTranslator(numE, 0) = eToEYTranslator(e, 0);
            numE++;
        }
    }

    prunedE.conservativeResize(numE, 4);
    E = prunedE;
    prunedEToEXTranslator.conservativeResize(numE, 1);
    eToEXTranslator = prunedEToEXTranslator;
    prunedEToEYTranslator.conservativeResize(numE, 1);
    eToEYTranslator = prunedEToEYTranslator;
}


/*
 % We can construct the del part as follows:
 % iterate through all triangles in the product space => column pos in del
 %   iterate through all edges in the in the edge product space
 %       if oriented as in edge list => 1 in del matrix
 %       if not oriented as in edge list => -1 in del matrix
 % Since this is very computational demanding ( O(numProductFaces^2) ), we
 % propose an approach that cleverly utilizes the structure of the product
 % space and the location of the edges of each shape within their face
 % matrices.
 %
 % The del matrix can be seen as follows:
 %
 %                                 numProductFaces
 %         ┌──────────────────────────────────────────────────────────────┐
 %         │                                                              │
 %         │                                                              │
 %  num    │                                                              │
 % Product │                                                              │
 %  Edges  │                                                              │
 %         │                                                              │
 %         │                                                              │
 %         └──────────────────────────────────────────────────────────────┘
 % We now can split this into different sections:
 %
 %     non degenerate faces
 %              | vertex X to triangle Y         vertex Y to triangleX
 %              |      |    edgesX to triangleY     |   edgesY to triangleX
 %              |      |             |              |           |
 %          ┌──────|──────|─────────────────────|──────|──────────────────┐
 % non deg  │      |      |                     |      |                  │
 %   Edges  │      |      |                     |      |                  │
 % ---------- ----------------------------------------------------------- -
 % deg Edges│      |      |                     |      |                  │
 %  of X    │      |      |                     |      |                  │
 % ---------- ----------------------------------------------------------- -
 % deg Edges│      |      |                     |      |                  │
 %  of      │      |      |                     |      |                  │
 %          └─────────────────────────────────────────────────────────────┘
 %
 % We know:
 % * non deg Edges can't appear in vertex to triangle product faces
 % * deg edges in Z can only appear in vertex Z to triangle and in edgesZ to
 %   triangle product faces
 % * the ordering of the faces in the product space is according to their
 %   order in the respective face matrices
 % * the ordering of the edges in the edges product space is dependent on
 %   the order of the faces (if no sorting is performed)
 %
 % => combining these facts and cleverly indexing only the relevant faces by
 % combing the information of the location of the edges we can reduce the
 % complexity of the getDel function to O(numProductFaces)
 */
void Constraints::getDelOptimizedwithPruneVec(std::vector<TripletInt8>& delEntries,
                                              const Eigen::VectorX<bool>& pruneVec,
                                              Eigen::MatrixXi& E,
                                              Eigen::MatrixXi& eToEXTranslator,
                                              Eigen::MatrixXi& eToEYTranslator) {
    
    int rowsE;
    Eigen::VectorX<long> cumSumPruneVec;
    igl::cumsum(pruneVec.cast<long>(), 1, cumSumPruneVec);
    Eigen::MatrixXi LocEXinFX = shapeX.getLocEinF();
    Eigen::MatrixXi LocEYinFY = shapeY.getLocEinF();
    numProductEdges = E.rows();

    // find num nondeg and num deg
    const int numEdges = E.rows();
    int numEdgesNonDegenerate = 2 * numEdgesX * numEdgesY;
    for (long i = 0; i < numEdges; i++) {
        if (E(i, 0) == E(i, 1)) {
            numEdgesNonDegenerate = i;
            break;
        }
    }
    int numEdgesYNonDegenerate = numEdgesNonDegenerate + numVerticesX * numEdgesY;
    for (long i = numEdgesNonDegenerate; i < numEdges; i++) {
        if (E(i, 2) == E(i, 3)) {
            numEdgesYNonDegenerate = i;
            break;
        }
    }
    
    #if defined(_OPENMP)
    #pragma omp parallel
    #endif
    {
        #if defined(_OPENMP)
            const int numThreads = omp_get_num_threads();
        #else
            const int numThreads = 1;
        #endif
        std::vector<TripletInt8> delEntriesPriv;
        /*
            assume that each core is approx same speed => each core fills numThreads-th of del entries
            (and a little bit extra to avoid reallocation)
         */
        delEntriesPriv.reserve(numProductFaces * 3 / numThreads + numProductFaces / (20 * numThreads));
        /*
         non-degenerate edges
         */
        const int offsetEX2TY =  3 * numFacesXxNumFacesY + numVerticesX * numFacesY;
        const int offsetEY2TX = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY + numVerticesY * numFacesX;
        
        #if defined(_OPENMP)
        #pragma omp for nowait
        #endif
        for (int e = 0; e < numEdgesNonDegenerate; e++) {
            searchInNonDegenerateFacesPRUNED(pruneVec, cumSumPruneVec, delEntriesPriv, E, e, LocEXinFX, LocEYinFY, eToEXTranslator, eToEYTranslator);
            //searchInEdgesX2TriangleY
            searchInEdges2TrianglePRUNED(pruneVec, cumSumPruneVec, delEntriesPriv, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, numEdgesX, numFacesY, offsetEX2TY);
            //searchInEdgesY2TriangleX
            searchInEdges2TrianglePRUNED(pruneVec, cumSumPruneVec, delEntriesPriv, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, numEdgesY, numFacesX, offsetEY2TX);
        }

        /*
         degenerate edges
         */
        // reset persistent variables in this function
        findVerticesInEdgesMatrixPRUNED(-1, shapeX, IDX, numIDX);
        
        // degenerate edges in X
        const int offsetVX2TY = 3 * numFacesXxNumFacesY;
        const int offsetDegEX = 3 * numFacesXxNumFacesY + numVerticesX * numFacesY;
        #if defined(_OPENMP)
        #pragma omp for nowait
        #endif
        for (int e  = numEdgesNonDegenerate; e < numEdgesYNonDegenerate; e++) {
            searchInVertex2TrianglePRUNED(pruneVec, cumSumPruneVec, delEntriesPriv, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, numFacesY, offsetVX2TY);
            // searchDegEdgesXInEdgesX2TriangleY
            searchDegEdgesInEdges2TrianglePRUNED(pruneVec, cumSumPruneVec, delEntriesPriv, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, shapeX, numEdgesX, numFacesX, numEdgesY, numFacesY, offsetDegEX);
        }

        // reset persistent variables in this function
        findVerticesInEdgesMatrixPRUNED(-1, shapeX, IDX, numIDX);
    
        const int offsetVY2TX = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY;
        const int offsetDegEY = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY + numVerticesY * numFacesX;
        // degenerate edges in Y
        #if defined(_OPENMP)
        #pragma omp for nowait
        #endif
        for (int e  = numEdgesYNonDegenerate; e < numEdges; e++) {
            //searchInVertexY2TriangleX
            searchInVertex2TrianglePRUNED(pruneVec, cumSumPruneVec, delEntriesPriv, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, numFacesX, offsetVY2TX);
            
            // searchDegEdgesYInEdgesY2TriangleX:
            searchDegEdgesInEdges2TrianglePRUNED(pruneVec, cumSumPruneVec, delEntriesPriv, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, shapeY, numEdgesY, numFacesY, numEdgesX, numFacesX, offsetDegEY);
        }
        // assemble del entires
        #if defined(_OPENMP)
        #pragma omp critical
        #endif
        delEntries.insert(delEntries.end(), delEntriesPriv.begin(), delEntriesPriv.end());
        
    }
}


void Constraints::getDelOptimizedPRUNED(std::vector<TripletInt8>& delEntries, const Eigen::VectorX<bool>& pruneVec, const Eigen::MatrixXi& coarsep2pmap, const Eigen::MatrixXi& IXf2c, const Eigen::MatrixXi& IYf2c) {

    int rowsE;
    Eigen::MatrixXi E = constructEdgeProductSpace(rowsE);
    Eigen::MatrixXi eToEXTranslator = constructEtoEdgesXTranslator();
    Eigen::MatrixXi eToEYTranslator = constructEtoEdgesYTranslator();
    pruneEdgeProductSpace(E, eToEXTranslator, eToEYTranslator, coarsep2pmap, IXf2c, IYf2c, shapeX, shapeY);

    getDelOptimizedwithPruneVec(delEntries, pruneVec, E, eToEXTranslator, eToEYTranslator);

}


/*



 BoundaryCode




 */

struct EDG {
    int idx0;
    int idx1;
    EDG () {}
    EDG (int iidx0, int iidx1) {
        idx0 = iidx0;
        idx1 = iidx1;
    }
    EDG operator-() const {
        EDG minusEDG;
        minusEDG.idx0 = idx1;
        minusEDG.idx1 = idx0;
        return minusEDG;
    }
    bool operator==(const EDG& edg) const {
        return (idx0 == edg.idx0) && (idx1 == edg.idx1);
    }
};
struct PEDG {
    int idx0;
    int idx1;
    int idy0;
    int idy1;
    PEDG () {}
    PEDG (int iidx0, int iidx1, int iidy0, int iidy1) {
        idx0 = iidx0;
        idx1 = iidx1;
        idy0 = iidy0;
        idy1 = iidy1;
    }
    bool operator==(const PEDG& edg) const {
        return (idx0 == edg.idx0) && (idx1 == edg.idx1) && (idy0 == edg.idy0) && (idy1 == edg.idy1);
    }
};
namespace std {
    template<> struct hash<EDG> {
        std::size_t operator()(EDG const& edg) const noexcept {
            int k1 = edg.idx0;
            int k2 = edg.idx1;
            return (k1 + k2 ) * (k1 + k2 + 1) / 2 + k2;
        }
    };
    template<> struct equal_to<EDG>{
        constexpr bool operator()(const EDG &lhs, const EDG &rhs) const {
            return  lhs == rhs;
        }
    };
    template<> struct hash<PEDG> {
        std::size_t operator()(PEDG const& edg) const noexcept {
            int k1 = edg.idx0;
            int k2 = edg.idx1;
            int k3 = edg.idy0;
            int k4 = edg.idy1;
            return (k1 + k2 ) * (k1 + k2 + 1) / 2 + k2 + (k3 + k4 ) * (k3 + k4 + 1) / 2 + k4;
        }
    };
    template<> struct equal_to<PEDG>{
        constexpr bool operator()(const PEDG &lhs, const PEDG &rhs) const {
            return lhs == rhs;
        }
    };
}

bool findEDG(const std::unordered_set<EDG> &ELookup, const EDG &edg) {
    const auto it = ELookup.find(edg);
    if (it != ELookup.end())
        return true;
    return false;
}
bool findPEdge(const std::unordered_set<PEDG> &ELookup, const PEDG &edg) {
    const auto it = ELookup.find(edg);
    if (it != ELookup.end())
        return true;
    return false;
}

std::tuple<std::unordered_set<EDG>, std::unordered_set<EDG>> findDummyAndBoundaryEdges(Shape& shapeX, const int nFXHoles) {
    std::unordered_set<EDG> dummyEdgesX;
    const Eigen::MatrixXi FX = shapeX.getF();
    const int numDummyFaces = shapeX.getNumFaces() - nFXHoles;

    std::vector<size_t> boundary; boundary.reserve(numDummyFaces + 3);
    igl::boundary_loop(FX.block(nFXHoles, 0, numDummyFaces, 3), boundary);
    boundary.push_back(boundary.front());
    std::unordered_set<EDG> boundaryEdgesX;
    for (int i = 0; i < boundary.size()-1; i++) {
        const EDG e0 = EDG(boundary.at(i), boundary.at(i+1));
        const EDG e1 = EDG(boundary.at(i+1), boundary.at(i));
        const EDG ed = EDG(boundary.at(i), boundary.at(i));
        boundaryEdgesX.insert(e0);
        boundaryEdgesX.insert(e1);
        boundaryEdgesX.insert(ed);
    }

    for (int i = nFXHoles; i < shapeX.getNumFaces(); i++) { // this loop follows convention how the product space is built up
        const EDG e0_0 = EDG(FX(i, 0), FX(i, 1));
        const EDG e0_1 = EDG(FX(i, 1), FX(i, 2));
        const EDG e0_2 = EDG(FX(i, 2), FX(i, 0));

        const EDG e1_0 = EDG(FX(i, 1), FX(i, 0));
        const EDG e1_1 = EDG(FX(i, 2), FX(i, 1));
        const EDG e1_2 = EDG(FX(i, 0), FX(i, 2));


        // if the edge is no boundary edge it has to be a dummy edge
        if (!findEDG(boundaryEdgesX, e0_0))
            dummyEdgesX.insert(e0_0);
        if (!findEDG(boundaryEdgesX, e1_0))
            dummyEdgesX.insert(e1_0);
        if (!findEDG(boundaryEdgesX, e0_1))
            dummyEdgesX.insert(e0_1);
        if (!findEDG(boundaryEdgesX, e1_1))
            dummyEdgesX.insert(e1_1);
        if (!findEDG(boundaryEdgesX, e0_2))
            dummyEdgesX.insert(e0_2);
        if (!findEDG(boundaryEdgesX, e1_2))
            dummyEdgesX.insert(e1_2);

    }
    return std::make_tuple(boundaryEdgesX, dummyEdgesX);
}

void pruneEdgeProductSpaceWithBoundary(Eigen::MatrixXi& E,
                                       const Eigen::MatrixXi& boundaryMatching,
                                       Eigen::MatrixXi& eToEXTranslator,
                                       Eigen::MatrixXi& eToEYTranslator,
                                       const int nFXHoles,
                                       const int nFYHoles,
                                       std::vector<std::tuple<int, int>>& boundaryConstraints,
                                       Shape shapeX,
                                       Shape shapeY) {
    // compute boundary product edges
    Eigen::MatrixXi boundaryProductEdges(2 * boundaryMatching.rows()-2, 4); boundaryProductEdges.setConstant(-1);
    std::unordered_set<PEDG> boundaryProductEdgesHashMap, invboundaryProductEdgesHashMap;
    int numAdded = 0;
    for (int i = 0; i < boundaryMatching.rows()-1; i++) {
        boundaryProductEdges.row(numAdded) << boundaryMatching(i, 0), boundaryMatching(i+1, 0), boundaryMatching(i, 1), boundaryMatching(i+1, 1);
        numAdded++;
        boundaryProductEdges.row(numAdded) << boundaryMatching(i, 0), boundaryMatching(i+1, 0), boundaryMatching(i+1, 1), boundaryMatching(i, 1);
        numAdded++;
        boundaryProductEdgesHashMap.insert(    PEDG(boundaryProductEdges(numAdded-2, 0), boundaryProductEdges(numAdded-2, 1), boundaryProductEdges(numAdded-2, 2), boundaryProductEdges(numAdded-2, 3)) );
        //boundaryProductEdgesHashMap.insert(    PEDG(boundaryProductEdges(numAdded-1, 1), boundaryProductEdges(numAdded-1, 0), boundaryProductEdges(numAdded-1, 2), boundaryProductEdges(numAdded-1, 3)) );
        invboundaryProductEdgesHashMap.insert( PEDG(boundaryProductEdges(numAdded-2, 1), boundaryProductEdges(numAdded-2, 0), boundaryProductEdges(numAdded-2, 3), boundaryProductEdges(numAdded-2, 2)) );
        //invboundaryProductEdgesHashMap.insert( PEDG(boundaryProductEdges(numAdded-1, 0), boundaryProductEdges(numAdded-1, 1), boundaryProductEdges(numAdded-1, 3), boundaryProductEdges(numAdded-1, 2)) );
    }
    if ((boundaryProductEdges.array() == -1).any()) {
        std::cout << "ERROR: did not add as many boundary matchings as expected" << std::endl;
    }

    // find dummy edges
    const auto dummyAndBoundaryEdgesX = findDummyAndBoundaryEdges(shapeX, nFXHoles);
    const std::unordered_set<EDG> boundaryEdgesX = std::get<0>(dummyAndBoundaryEdgesX);
    const std::unordered_set<EDG> dummyEdgesX = std::get<1>(dummyAndBoundaryEdgesX);
    const auto dummyAndBoundaryEdgesY = findDummyAndBoundaryEdges(shapeY, nFYHoles);
    const std::unordered_set<EDG> boundaryEdgesY = std::get<0>(dummyAndBoundaryEdgesY);
    const std::unordered_set<EDG> dummyEdgesY = std::get<1>(dummyAndBoundaryEdgesY);

    Eigen::MatrixXi prunedE(E.rows(), 4);
    Eigen::MatrixXi prunedEToEXTranslator(E.rows(), 1);
    Eigen::MatrixXi prunedEToEYTranslator(E.rows(), 1);
    //std::cout << boundaryMatching << std::endl;
    //std::cout << boundaryProductEdgesHashMap.size() << " asdfasdf " << invboundaryProductEdgesHashMap.size() << std::endl;
    //std::cout << "++++" << std::endl;
    long numE = 0;
    for (long e = 0; e < E.rows(); e++) {
        bool prune = false;

        const EDG ex(E(e, 0), E(e, 1));
        const EDG ey(E(e, 2), E(e, 3));
        const bool isDeg = (E(e, 0) == E(e, 1)) || (E(e, 2) == E(e, 3));

        const bool exIsBoundaryEdge = findEDG(boundaryEdgesX, ex);
        const bool eyIsBoundaryEdge = findEDG(boundaryEdgesY, ey);
        // never!!!! :D prune boundary product edges if not degenerate and onle one of both is boundary edge
        const bool pruneBoundaryEdges = (!exIsBoundaryEdge != !eyIsBoundaryEdge) && !isDeg;
        const bool exIsDummyEdge = findEDG(dummyEdgesX, ex);
        const bool eyIsDummyEdge = findEDG(dummyEdgesY, ey);
        if (exIsDummyEdge || eyIsDummyEdge) {
            prune = true;
        }

        // setboundaryConstraints rhs
        if (exIsBoundaryEdge && eyIsBoundaryEdge) { // both edges are boundary
            const PEDG pe(E(e, 0), E(e, 1), E(e, 2), E(e, 3));
            if (findPEdge(boundaryProductEdgesHashMap, pe)) {
                boundaryConstraints.push_back(std::make_tuple(numE, -1)); // first orientation of product edge will be used as +1 in del operator => rhs must be 1
            }
            else if (findPEdge(invboundaryProductEdgesHashMap, pe)) {
                boundaryConstraints.push_back(std::make_tuple(numE, 1)); // second orientation of product edge will be used as -1 in del operator => rhs must be 1
            }
            else {
                // we didnt find boudnary edges in boundary matching => prune it
                prune = true;
            }
        }


        if (!prune) {
            prunedE(numE, Eigen::all) = E(e, Eigen::all);
            prunedEToEXTranslator(numE, 0) = eToEXTranslator(e, 0);
            prunedEToEYTranslator(numE, 0) = eToEYTranslator(e, 0);
            numE++;
        }
    }


    // overwrite existing matrices
    prunedE.conservativeResize(numE, 4);
    E = prunedE;
    prunedEToEXTranslator.conservativeResize(numE, 1);
    eToEXTranslator = prunedEToEXTranslator;
    prunedEToEYTranslator.conservativeResize(numE, 1);
    eToEYTranslator = prunedEToEYTranslator;
}

long Constraints::getDelOptimizedBoundary(std::vector<TripletInt8>& delEntries,
                                          const Eigen::VectorX<bool>& pruneVec,
                                          const Eigen::MatrixXi& boundaryMatching,
                                          const int nFXHoles,
                                          const int nFYHoles,
                                          std::vector<std::tuple<int, int>>& boundaryConstraints) {
    int rowsE;
    Eigen::MatrixXi E;
    E = constructEdgeProductSpace(rowsE);
    Eigen::MatrixXi eToEXTranslator = constructEtoEdgesXTranslator();
    Eigen::MatrixXi eToEYTranslator = constructEtoEdgesYTranslator();

    pruneEdgeProductSpaceWithBoundary(E, boundaryMatching, eToEXTranslator, eToEYTranslator, nFXHoles, nFYHoles, boundaryConstraints, shapeX, shapeY);

    getDelOptimizedwithPruneVec(delEntries, pruneVec, E, eToEXTranslator, eToEYTranslator);
    return E.rows();
}
