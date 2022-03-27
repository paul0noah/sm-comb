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
#if defined(_OPENMP)
    #include <omp.h>
#else
    #warning OPENMP not found on this machine. Consider installing it for improved performance
#endif


const EDGE idxEdge0((EDGE() << 0, 1).finished());
const EDGE idxEdge1((EDGE() << 1, 2).finished());
const EDGE idxEdge2((EDGE() << 2, 0).finished());
void Constraints::checkFace(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, int f) {
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

/*
 %findVerticesInEdgesMatrix finds a vertex represented by its idx within the
 %  edge matrix of the specified shape
 %
 % This function stores its result and if vertexIdx changes, it updates its values.
 */
static int oldVertexIdx = -1;
static Eigen::MatrixXi IDX(20, 2);
static Eigen::MatrixXi numIDX(1, 2);
void Constraints::findVerticesInEdgesMatrix(const int vertexIdx, Shape &shape, Eigen::MatrixXi &idxes, Eigen::MatrixXi &numIdxes) {
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
void Constraints::searchDegEdgesInEdges2Triangle(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, Shape &shape, int numEX, int numFX, int numEY, int numFY, int offset) {
    
    const int vertexIdx = eToEXTranslator(e);
    Eigen::MatrixXi idxes(IDX.rows(), 2);
    Eigen::MatrixXi numIdxes(1, 2);
    // we don't know how often a vertex appears in the corresponding edge
    // matrix of a shape => this functions searches for the vertices
    findVerticesInEdgesMatrix(vertexIdx, shape, idxes, numIdxes);
    
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
                checkFace(delEntries, E, e, f1);
                
                // first orientation of the edge
                int f2 = f + LocEYinFY(eToEYTranslator(e), 1);
                checkFace(delEntries, E, e, f2);
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
void Constraints::searchInEdges2Triangle(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numE, int numF, int offset) {
    
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
            checkFace(delEntries, E, e, f1);
             
            // second orientation of the edge
            int f2 = f + LocEYinFY(eToEYTranslator(e), 1);
            checkFace(delEntries, E, e, f2);
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
void Constraints::searchInVertex2Triangle(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator, int numF, int offset) {
    
    int f = offset + eToEXTranslator(e) * numF;
    
    // first orientation of the edge
    int f1 = f + LocEYinFY(eToEYTranslator(e), 0);
    
    checkFace(delEntries, E, e, f1);
    
    // second orientation of the edge
    int f2 = f + LocEYinFY(eToEYTranslator(e), 1);
    checkFace(delEntries, E, e, f2);
    
}

void Constraints::searchInNonDegenerateFaces(std::vector<TripletInt8> &delEntries, Eigen::MatrixXi &E, int e, Eigen::MatrixXi &LocEXinFX, Eigen::MatrixXi &LocEYinFY, Eigen::MatrixXi &eToEXTranslator, Eigen::MatrixXi &eToEYTranslator) {
    // this i-loop accounts for the different rotations of the triangles
    for (int i = 0; i < 3; i++) {
        int f = i * numFacesXxNumFacesY;
        // each product edge may be in one of 4 possible triangles
        
        int f11 = f + LocEXinFX(eToEXTranslator(e), 0) * numFacesY + LocEYinFY(eToEYTranslator(e), 0);
        checkFace(delEntries, E, e, f11);
        
        int f12 = f + LocEXinFX(eToEXTranslator(e), 0) * numFacesY + LocEYinFY(eToEYTranslator(e), 1);
        checkFace(delEntries, E, e, f12);
        
        int f21 = f + LocEXinFX(eToEXTranslator(e), 1) * numFacesY + LocEYinFY(eToEYTranslator(e), 0);
            checkFace(delEntries, E, e, f21);
        
        int f22 = f + LocEXinFX(eToEXTranslator(e), 1) * numFacesY + LocEYinFY(eToEYTranslator(e), 1);
        checkFace(delEntries, E, e, f22);
    }
}

Eigen::MatrixXi Constraints::constructEtoEdgesXTranslator() {
#ifdef LARGE_EDGE_PRODUCT_SPACE
    const int numProductEdges = 2 * numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
    Eigen::MatrixXi EtoEdgesXTranslator(numProductEdges, 1);
    
    EtoEdgesXTranslator.block(0, 0, numEdgesX * numEdgesY, 1) =
        utils::repelem(utils::linspaced(0, numEdgesX), numEdgesY, 1);
    EtoEdgesXTranslator.block(numEdgesX * numEdgesY, 0, numEdgesX * numEdgesY, 1) =
        utils::repelem(utils::linspaced(0, numEdgesX), numEdgesY, 1);
    EtoEdgesXTranslator.block(2 * numEdgesX * numEdgesY, 0, numVerticesX * numEdgesY, 1) =
        utils::repelem(utils::linspaced(0, numVerticesX), numEdgesY, 1);
    EtoEdgesXTranslator.block(2 * numEdgesX * numEdgesY + numVerticesX * numEdgesY, 0, numVerticesY * numEdgesX, 1) =
        utils::repelem(utils::linspaced(0, numEdgesX), numVerticesY, 1);
#else
    const int numProductEdges = numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
    Eigen::MatrixXi EtoEdgesXTranslator(numProductEdges, 1);
    
    EtoEdgesXTranslator.block(0, 0, numEdgesX * numEdgesY, 1) =
        utils::repelem(utils::linspaced(0, numEdgesX), numEdgesY, 1);
                                                
    EtoEdgesXTranslator.block(numEdgesX * numEdgesY, 0, numVerticesX * numEdgesY, 1) =
        utils::repelem(utils::linspaced(0, numVerticesX), numEdgesY, 1);
    EtoEdgesXTranslator.block(numEdgesX * numEdgesY + numVerticesX * numEdgesY, 0, numVerticesY * numEdgesX, 1) =
        utils::repelem(utils::linspaced(0, numEdgesX), numVerticesY, 1);
#endif
    return EtoEdgesXTranslator;
}

Eigen::MatrixXi Constraints::constructEtoEdgesYTranslator() {
#ifdef LARGE_EDGE_PRODUCT_SPACE
    const int numProductEdges = 2 * numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
    Eigen::MatrixXi EtoEdgesYTranslator(numProductEdges, 1);

    EtoEdgesYTranslator.block(0, 0, 2 * numEdgesX * numEdgesY, 1) =
        utils::linspaced(0, numEdgesY).replicate(2 * numEdgesX, 1);
    EtoEdgesYTranslator.block(2 * numEdgesX * numEdgesY, 0, numVerticesX * numEdgesY, 1) =
        utils::linspaced(0, numEdgesY).replicate(numVerticesX, 1);
    EtoEdgesYTranslator.block(2 * numEdgesX * numEdgesY + numVerticesX * numEdgesY, 0, numVerticesY * numEdgesX, 1) =
        utils::linspaced(0, numVerticesY).replicate(numEdgesX, 1);
#else
    const int numProductEdges = numEdgesX * numEdgesY + numEdgesX * numVerticesY + numEdgesY * numVerticesX;
    Eigen::MatrixXi EtoEdgesYTranslator(numProductEdges, 1);

    EtoEdgesYTranslator.block(0, 0, numEdgesX * numEdgesY, 1) =
        utils::linspaced(0, numEdgesY).replicate(numEdgesX, 1);
    EtoEdgesYTranslator.block(numEdgesX * numEdgesY, 0, numVerticesX * numEdgesY, 1) =
        utils::linspaced(0, numEdgesY).replicate(numVerticesX, 1);
    EtoEdgesYTranslator.block(numEdgesX * numEdgesY + numVerticesX * numEdgesY, 0, numVerticesY * numEdgesX, 1) =
        utils::linspaced(0, numVerticesY).replicate(numEdgesX, 1);
#endif
    return EtoEdgesYTranslator;
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
SparseMatInt8 Constraints::getDelOptimized() {
    // allocate del
    const int numNonZerosDel = numProductFaces * 3;
    SparseMatInt8 del(numProductEdges, numProductFaces);
    
    std::vector<TripletInt8> delEntries;
    delEntries.reserve(numProductFaces * 3);
    
    int rowsE;
    Eigen::MatrixXi E;
    E = constructEdgeProductSpace(rowsE);
    Eigen::MatrixXi LocEXinFX = shapeX.getLocEinF();
    Eigen::MatrixXi LocEYinFY = shapeY.getLocEinF();
    Eigen::MatrixXi eToEXTranslator = constructEtoEdgesXTranslator();
    Eigen::MatrixXi eToEYTranslator = constructEtoEdgesYTranslator();
    
    
    const int numEdges = E.rows();
    const int numEdgesNonDegenerate = numEdgesX * numEdgesY;
    const int numEdgesYNonDegenerate = numEdgesNonDegenerate + numVerticesX * numEdgesY;
    
#ifdef LARGE_EDGE_PRODUCT_SPACE
    int eLarger = numEdgesNonDegenerate;
#else
    int eLarger = 0;
#endif
    
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
        for (int e = 0; e < numEdgesNonDegenerate + eLarger; e++) {
            searchInNonDegenerateFaces(delEntriesPriv, E, e, LocEXinFX, LocEYinFY, eToEXTranslator, eToEYTranslator);
            //searchInEdgesX2TriangleY
            searchInEdges2Triangle(delEntriesPriv, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, numEdgesX, numFacesY, offsetEX2TY);
            //searchInEdgesY2TriangleX
            searchInEdges2Triangle(delEntriesPriv, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, numEdgesY, numFacesX, offsetEY2TX);
        }

        /*
         degenerate edges
         */
        // reset persistent variables in this function
        findVerticesInEdgesMatrix(-1, shapeX, IDX, numIDX);
        
        // degenerate edges in X
        const int offsetVX2TY = 3 * numFacesXxNumFacesY;
        const int offsetDegEX = 3 * numFacesXxNumFacesY + numVerticesX * numFacesY;
        #if defined(_OPENMP)
        #pragma omp for nowait
        #endif
        for (int e  = eLarger + numEdgesNonDegenerate; e < eLarger + numEdgesYNonDegenerate; e++) {
            searchInVertex2Triangle(delEntriesPriv, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, numFacesY, offsetVX2TY);
            // searchDegEdgesXInEdgesX2TriangleY
            searchDegEdgesInEdges2Triangle(delEntriesPriv, E, e, LocEYinFY, eToEXTranslator, eToEYTranslator, shapeX, numEdgesX, numFacesX, numEdgesY, numFacesY, offsetDegEX);
        }

        // reset persistent variables in this function
        findVerticesInEdgesMatrix(-1, shapeX, IDX, numIDX);
    
        const int offsetVY2TX = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY;
        const int offsetDegEY = 12 * numFacesXxNumFacesY + numVerticesX * numFacesY + numVerticesY * numFacesX;
        // degenerate edges in Y
        #if defined(_OPENMP)
        #pragma omp for nowait
        #endif
        for (int e  = eLarger + numEdgesYNonDegenerate; e < numEdges; e++) {
            //searchInVertexY2TriangleX
            searchInVertex2Triangle(delEntriesPriv, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, numFacesX, offsetVY2TX);
            
            // searchDegEdgesYInEdgesY2TriangleX:
            searchDegEdgesInEdges2Triangle(delEntriesPriv, E, e, LocEXinFX, eToEYTranslator, eToEXTranslator, shapeY, numEdgesY, numFacesY, numEdgesX, numFacesX, offsetDegEY);
        }
        // assemble del entires
        #if defined(_OPENMP)
        #pragma omp critical
        #endif
        delEntries.insert(delEntries.end(), delEntriesPriv.begin(), delEntriesPriv.end());
        
    }
    
    del.setFromTriplets(delEntries.begin(), delEntries.end());
    return del;
}
