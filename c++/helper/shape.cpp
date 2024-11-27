//
//  shape.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <igl/readOFF.h>
#include <igl/decimate.h>
#include <igl/qslim.h>
#include <igl/boundary_loop.h>
#include <igl/bfs_orient.h>
#include "shape.hpp"
#include "helper/utils.hpp"
#include <iostream>
#include <tmhc/close_holes.h>
#include <tmhc/find_holes.h>
#include <tsl/robin_set.h>


Shape::Shape() {
    V = Eigen::MatrixXf(0, 0);
    F = Eigen::MatrixXi(0, 0);
}

Shape::Shape(std::string filename, int numFaces) :
    Shape(filename) {
    reduce(numFaces);
    initShape();
}

Shape::Shape(std::string filename) {
    Eigen::MatrixXd Vinp;
    std::string fileextension = filename.substr(filename.find_last_of(".") + 1);
    if (fileextension == "ply") {
        igl::readPLY(filename, Vinp, F);
    }
    else if (fileextension == "off") {
        igl::readOFF(filename, Vinp, F);
    }
    else {
        std::cout << "FILE EXTENSION NOT SUPPORTED" << std::endl;
    }
    V = Vinp.cast<float>();
    initShape();
}

Shape::Shape(Eigen::MatrixXf Vinp, Eigen::MatrixXi Finp) {
    V = Vinp;
    F = Finp;
    initShape();
}

Shape::Shape(Eigen::MatrixXf Vinp, Eigen::MatrixXi Finp, int numFaces) {
    V = Vinp;
    F = Finp;
    reduce(numFaces);
    initShape();
}

void Shape::initShape() {
    E = Eigen::MatrixXi();
    LocEinF = Eigen::MatrixXi();
    triangleNeighbours = Eigen::MatrixXi();
    edgesComputed = false;
    triangleNeighboursComputed = false;
    watertight = true; // will be overwritten
    computeEdges();
    getTriangleNeighbours();
}

const long Shape::getNumFaces() {
    return F.rows();
}

const long Shape::getNumEdges() {
    // assumes that the shape is closed
    if (!edgesComputed) {
        computeEdges();
    }
    return E.rows();
}

const long Shape::getNumVertices(){
    return V.rows();
}

struct EDG {
    int idx0;
    int idx1;
    int e; // index of edge
    EDG () {}
    EDG (int iidx0, int iidx1, int ie) {
        idx0 = iidx0;
        idx1 = iidx1;
        e = ie;
    }
    EDG (Eigen::MatrixXi edge, int ie) {
        idx0 = edge(0);
        idx1 = edge(1);
        e = ie;
    }
    EDG (Eigen::MatrixXi edge) {
        idx0 = edge(0);
        idx1 = edge(1);
        e = -1;
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
namespace std {
    template<> struct hash<EDG> {
        std::size_t operator()(EDG const& edg) const noexcept {
            //size_t idx0hash = std::hash<int>()(edg.idx0);
            //size_t idx1hash = std::hash<int>()(edg.idx1) << 1;
            //return idx0hash ^ idx1hash;
            int k1 = edg.idx0;
            int k2 = edg.idx1;
            return (k1 + k2 ) * (k1 + k2 + 1) / 2 + k2;
        }
    };
    template<> struct equal_to<EDG>{
        constexpr bool operator()(const EDG &lhs, const EDG &rhs) const {
            return (lhs.idx0 == rhs.idx0) && (lhs.idx1 == rhs.idx1);
        }
    };
}

int findEdge(const tsl::robin_set<EDG> &ELookup, const EDG &edg) {
    auto it = ELookup.find(edg);
    if(it != ELookup.end()) {
        EDG foundEdg = *it;
        return foundEdg.e;
    }
    return -1;
}

void Shape::computeEdges() {
    const int numFaces = getNumFaces();
    const int numEdgesClosedMesh = numFaces * 3 / 2;
    const int maxNumEdges = numFaces * 3;
    Eigen::MatrixXi Ework(maxNumEdges, 2);
    Ework = -Ework.setOnes();
    Eigen::MatrixXi LocEinFwork(maxNumEdges, 2);
    LocEinFwork.setOnes();
    LocEinFwork = - LocEinFwork;
    tsl::robin_set<EDG> ELookup; ELookup.reserve(maxNumEdges);

    Eigen::Vector2i idxEdge0; idxEdge0 << 0, 1;
    Eigen::Vector2i idxEdge1; idxEdge1 << 1, 2;
    Eigen::Vector2i idxEdge2; idxEdge2 << 2, 0;

    // we add the first edges before the loop => otherwise first iteration does not work
    Ework(0, Eigen::all) = F(0, idxEdge0);
    ELookup.insert(EDG(Ework.row(0), 0));
    LocEinFwork(0, 0) = 0;
    Ework(1, Eigen::all) = F(0, idxEdge1);
    ELookup.insert(EDG(Ework.row(1), 1));
    LocEinFwork(1, 0) = 0;
    Ework(2, Eigen::all) = F(0, idxEdge2);
    ELookup.insert(EDG(Ework.row(2), 2));
    LocEinFwork(2, 0) = 0;
    int numEdgesAdded = 3;

    Eigen::Vector2i edge0, edge1, edge2;
    bool found0, found1, found2;
    Eigen::Vector2i minusEdge; minusEdge << 1, 0;

    // will be overwritten later
    watertight = true;

    for (int f = 1; f < numFaces; f++) {
        edge0 = F(f, idxEdge0); found0 = false;
        edge1 = F(f, idxEdge1); found1 = false;
        edge2 = F(f, idxEdge2); found2 = false;

        if (findEdge(ELookup, EDG(edge0)) != -1) {
            found0 = true;
        }
        else {
            int e = findEdge(ELookup, -EDG(edge0));
            if (e != -1) {
                found0 = true;
                LocEinFwork(e, 1) = f;
            }
        }

        if (findEdge(ELookup, EDG(edge1)) != -1) {
            found1 = true;
        }
        else {
            int e = findEdge(ELookup, -EDG(edge1));
            if (e != -1) {
                found1 = true;
                LocEinFwork(e, 1) = f;
            }
        }

        if (findEdge(ELookup, EDG(edge2)) != -1) {
            found2 = true;
        }
        else {
            int e = findEdge(ELookup, -EDG(edge2));
            if (e != -1) {
                found2 = true;
                LocEinFwork(e, 1) = f;
            }
        }

        // Add edges if any new were found
        if (!found0) {
            Ework(numEdgesAdded, Eigen::all) = edge0;
            ELookup.insert(EDG(edge0, numEdgesAdded));
            LocEinFwork(numEdgesAdded, 0) = f;
            numEdgesAdded++;

        }
        if (!found1) {
            Ework(numEdgesAdded, Eigen::all) = edge1;
            ELookup.insert(EDG(edge1, numEdgesAdded));
            LocEinFwork(numEdgesAdded, 0) = f;
            numEdgesAdded++;

        }
        if (!found2) {
            Ework(numEdgesAdded, Eigen::all) = edge2;
            ELookup.insert(EDG(edge2, numEdgesAdded));
            LocEinFwork(numEdgesAdded, 0) = f;
            numEdgesAdded++;
        }
    }

    LocEinF = LocEinFwork.block(0, 0, numEdgesAdded, 2);
    E = Ework.block(0, 0, numEdgesAdded, 2);

    // Watertightness check
    if (numEdgesAdded != numEdgesClosedMesh) {
        watertight = false;
    }
    if ((LocEinF.array() == -1).any()) {
        watertight = false;
    }
    edgesComputed = true;
}

/* function getTrianglesAttachedToVertex
    returns a vector containing the index of the triangles which are attached to vertex vertexIdx
 */
Eigen::VectorXi Shape::getTrianglesAttachedToVertex(unsigned int vertexIdx) {
    // find triangles in which vertexIdx exists
    Eigen::VectorXi vertexAttachments;
    Eigen::MatrixXi currentTriangle(1, 3);
    for (int i = 0; i < getNumFaces(); i++) {
        currentTriangle = getFi(i);
        for (int j = 0; j < 3; j++) {
            if (currentTriangle(j) == vertexIdx) {
                utils::addElement2IntVector(vertexAttachments, i);
                j = 3;
            }
        }
    }
    return vertexAttachments;
}

const float Shape::getTriangleArea(Eigen::MatrixXi triangle) {
    return 0.5 * getTwiceTriangleArea(triangle);
}

const float Shape::getTriangleArea(int triangleIdx) {
    return getTriangleArea(F(triangleIdx, Eigen::all));
}

Eigen::MatrixXf Shape::getTriangleAreas(Eigen::MatrixXi Fcombo) {
    Eigen::MatrixXf areas(Fcombo.rows(), 1);
    for (int i = 0; i < Fcombo.rows(); i++) {
        areas(i) = getTriangleArea(Fcombo(i, Eigen::all));
    }
    return areas;
}

Eigen::MatrixXf Shape::getTriangleAreas() {
    return getTriangleAreas(getF());
}

const float Shape::getTwiceTriangleArea(Eigen::MatrixXi triangle) {
    // get vertices of the triangle
    Eigen::Vector3f v0 = V.row(triangle(0));
    Eigen::Vector3f v1 = V.row(triangle(1));
    Eigen::Vector3f v2 = V.row(triangle(2));

    // extract two edges
    Eigen::Vector3f e1 = v0 - v1;
    Eigen::Vector3f e2 = v0 - v2;

    // triangle area via cross product
    return e1.cross(e2).norm();
}

float Shape::getAngle(Eigen::Vector3f e0, Eigen::Vector3f e1) {
    // Computes the angle between two vectors e0 and e1
    const float nominator = e0.transpose() * e1;
    const float denominator = e0.norm() * e1.norm() ;
    return acos( nominator /  denominator);
}

Eigen::Vector3f Shape::getTriangleAngles(unsigned int triangleIdx) {
    /*      2
            -
           / \
      e1  /   \  e0
         /     \
        /       \
       /_________\
     0      e2     1
    */
    // get vertices of the triangle
    Eigen::Vector3i triangle =  F.row(triangleIdx);
    Eigen::Vector3f v0 = V.row(triangle(0));
    Eigen::Vector3f v1 = V.row(triangle(1));
    Eigen::Vector3f v2 = V.row(triangle(2));

    // extract two edges
    Eigen::Vector3f e0 = v2 - v1;
    Eigen::Vector3f e1 = v0 - v2;
    Eigen::Vector3f e2 = v1 - v0;
    
    Eigen::Vector3f angles;
    angles << getAngle(-e1, e2), getAngle(-e2, e0), getAngle(-e0, e1);
    return angles;
}


void Shape::modifyV(Eigen::MatrixXf newV) {
    edgesComputed = false;
    triangleNeighboursComputed = false;
    V = newV;
}

void Shape::modifyV(Eigen::MatrixXd newV) {
    edgesComputed = false;
    triangleNeighboursComputed = false;
    V = newV.cast<float>();
}

void Shape::modifyF(Eigen::MatrixXi newF) {
    edgesComputed = false;
    triangleNeighboursComputed = false;
    F = newF;
}


Eigen::MatrixXf Shape::getVi(int vertexIdx) {
    return V.row(vertexIdx);
}

Eigen::MatrixXi Shape::getFi(int triangleIdx) {
    return F.row(triangleIdx);
}

Eigen::MatrixXf Shape::getV() {
    return V;
}

Eigen::MatrixXi Shape::getF() {
    return F;
}

Eigen::MatrixXi Shape::getE() {
    if (!edgesComputed) {
        computeEdges();
        edgesComputed = true;
    }
    return E;
}

Eigen::MatrixXi Shape::getLocEinF() {
    if (!edgesComputed) {
        computeEdges();
        edgesComputed = true;
    }
    return LocEinF;
}

Eigen::MatrixXi Shape::getLocFinE() {
    Eigen::MatrixXi LocFinE(getNumFaces(), 3);
    Eigen::MatrixXi counter(getNumFaces(), 2); counter.setZero();
    LocFinE = -LocFinE.setOnes();
    
    if (!edgesComputed) {
        computeEdges();
        edgesComputed = true;
    }
    
    for (int e = 0; e < LocEinF.rows(); e++) {
        const int f0 = LocEinF(e, 0);
        LocFinE(f0, counter(f0)) = e;
        counter(f0, 0) = counter(f0) + 1;
        const int f1 = LocEinF(e, 1);
        LocFinE(f1, counter(f1)) = e;
        counter(f1, 0) = counter(f1) + 1;
    }
    if (LocEinF.array().cwiseEqual(-1).any()) {
        std::cout << "Error while extracting LocEinF correctly. Aborting" << std::endl;
    }
    return LocFinE;
}

bool Shape::isWatertight() {
    if (!edgesComputed) {
        computeEdges();
        edgesComputed = true;
    }
    return watertight;
}

bool Shape::reduce(size_t numFaces) {
    Eigen::MatrixXd Vdec = V.cast<double>();
    Eigen::MatrixXi Fdec;
    Eigen::VectorXi J;
    Eigen::VectorXi I;
    edgesComputed = false;
    triangleNeighboursComputed = false;
    if (!igl::qslim(Vdec, F, numFaces, Vdec, Fdec, J, I)) {
        std::cout << "Couldn't reduce to requested amount of faces. Reduced shape has "
            << Fdec.rows() << " number of faces" << std::endl;
        V = Vdec.cast<float>();
        F = Fdec;
        return false;
    }
    V = Vdec.cast<float>();
    F = Fdec;
    return true;
}


bool Shape::writeToFile(std::string filename) {
    std::string fileextension = filename.substr(filename.find_last_of(".") + 1);
    if (fileextension == "ply") {
        return igl::writePLY(filename, V, F, igl::FileEncoding::Ascii);
    }
    else {
        std::cout << "FILE EXTENSION NOT SUPPORTED" << std::endl;
        return false;
    }
}

void Shape::translate(const Eigen::Vector3f translationVector) {
    for (int i = 0; i < getNumVertices(); i++) {
        for (int j = 0; j < 3; j++) {
            V(i, j) += translationVector(j);
        }
    }
}

float Shape::squashInUnitBox() {
    
    float minX = V(Eigen::all, 0).minCoeff();
    float minY = V(Eigen::all, 1).minCoeff();
    float minZ = V(Eigen::all, 2).minCoeff();
    Eigen::Vector3f translate2positive;
    translate2positive << minX, minY, minZ;
    translate(translate2positive);
    
    Eigen::Vector3f maxXYZ;
    
    maxXYZ << V(Eigen::all, 0).maxCoeff(),
              V(Eigen::all, 1).maxCoeff(),
              V(Eigen::all, 2).maxCoeff();
    
    float max = maxXYZ.maxCoeff();
    float squashFactor = 1 / max;
    
    squash(squashFactor);
    
    return squashFactor;
}

void Shape::squash(const float squashFactor) {
    for (int i = 0; i < V.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            V(i, j) *= squashFactor;
        }
    }
}


Eigen::MatrixXi Shape::getTriangleNeighbours() {
    if (!isWatertight()) {
        std::cout << "Cannot compute triangle neighbours because shape is not watertight" << std::endl;
        return triangleNeighbours;
    }
    if (triangleNeighboursComputed) {
        return triangleNeighbours;
    }
    const int numFaces = getNumFaces();
    
    Eigen::MatrixXi locEinF = getLocEinF();
    
    // 4 columns because we have 3 neighbours and one count column
    Eigen::MatrixXi triNeighbours(numFaces, 4);
    triNeighbours.setOnes(); triNeighbours = -triNeighbours;

    
    for (int e = 0; e < getNumEdges(); e++) {
        const int tri1 = locEinF(e, 0);
        const int neighbourOfTri1 = locEinF(e, 1);
        triNeighbours(tri1, 3) += 1;
        const int col1 = triNeighbours(tri1, 3);
        triNeighbours(tri1, col1) = neighbourOfTri1;
        
        const int tri2 = locEinF(e, 1);
        const int neighbourOfTri2 = locEinF(e, 0);
        triNeighbours(tri2, 3) += 1;
        const int col2 = triNeighbours(tri2, 3);
        triNeighbours(tri2, col2) = neighbourOfTri2;
    }
    
    if (triNeighbours.array().cwiseEqual(-1).any()) {
        std::cout << "Error while extracting neighbours" << std::endl;
    }
    triangleNeighbours = triNeighbours.block(0, 0, numFaces, 3);
    triangleNeighboursComputed = true;
    return triangleNeighbours;
}


bool alreadyInNewF(Eigen::MatrixXi& idxOldTriangles, const int numFacesReordered, const int newIdx) {
    for (int i = 0; i < numFacesReordered; i++) {
        if (idxOldTriangles(i) == newIdx) {
            return true;
        }
    }
    return false;
}


Eigen::MatrixXi Shape::reOrderTriangulation() {
    return reOrderTriangulation(0);
}

Eigen::MatrixXi Shape::reOrderTriangulation(int seed) {
    if (!triangleNeighboursComputed) {
        getTriangleNeighbours();
    }
    if (!isWatertight()) {
        std::cout << "Cannot reOrderTriangulation because shape is not watertight" << std::endl;
        return Eigen::MatrixXi();
    }
    
    Eigen::MatrixXi newF(getNumFaces(), 3); newF = -newF.setOnes();
    Eigen::MatrixXi permutation(getNumFaces(), 1); permutation = -permutation.setOnes();
    
    int numFacesReorderd = 0;

    if (seed != 0) {
        if (seed > getNumFaces() - 1) {
            std::cout << "Seed to big. Using seed = 0" << std::endl;
        }
        else {
            Eigen::MatrixXi temp = triangleNeighbours.row(seed);
            triangleNeighbours(seed, Eigen::all) = triangleNeighbours.row(0);
            triangleNeighbours(0, Eigen::all) = temp;
            temp = F.row(seed);
            F(seed, Eigen::all) = F.row(0);
            F(0, Eigen::all) = temp;
        }
    }


    // we have to add the first outside the for-loop
    permutation(numFacesReorderd) = triangleNeighbours(0, 0);
    newF.block(0, 0, 1, 3) = F.row(permutation(0));
    numFacesReorderd++;
    permutation(numFacesReorderd) = triangleNeighbours(0, 1);
    numFacesReorderd++;
    permutation(numFacesReorderd) = triangleNeighbours(0, 2);
    numFacesReorderd++;
    
    for (int i = 1; i < getNumFaces(); i++) {
        if (i >= numFacesReorderd) {
            ASSERT_NEVER_REACH;
        }
        
        const int curIdx = permutation(i);
        Eigen::MatrixXi neighboursOfCurrentTriangle = triangleNeighbours.block(curIdx, 0, 1, 3);
        
        for (int j = 0; j < 3; j++) {
            const int newIdx = neighboursOfCurrentTriangle(j);
            if (!alreadyInNewF(permutation, numFacesReorderd, newIdx)) {
                permutation(numFacesReorderd) = newIdx;
                numFacesReorderd++;
            }
        }
        
        newF.block(i, 0, 1, 3) = F.row(permutation(i   ));
    }
    
    F = newF;
    // the translation matrices are not correct anymore and thus have to be recomputed
    // TODO: improve efficiency by using permutation matrix
    triangleNeighboursComputed = false;
    edgesComputed = false;
    return permutation;
}


void Shape::reorderVertices() {
    reOrderTriangulation();
    Eigen::MatrixXf Vnew = V;
    Eigen::MatrixXi Fnew = F;
    Eigen::MatrixXi permutation(getNumVertices(), 1); permutation = -permutation.setOnes();
    int numVertPermuted = 0;
    for (int f = 0; f < getNumFaces(); f++) {
        for (int i = 0; i < 3; i++) {
            const int currentVidx = F(f, i);
            if (permutation(currentVidx) == -1) {
                permutation(currentVidx) = numVertPermuted;
                Vnew(numVertPermuted, Eigen::all) = V.row(currentVidx);
                numVertPermuted++;
            }
            Fnew(f, i) = permutation(currentVidx);
        }
    }
    V = Vnew;
    F = Fnew;
}


void Shape::plot() {
    Eigen::MatrixXd color(3, 1);
    color(0) = 0.905;
    color(1) = 0.623;
    color(2) = 0.333;
    plot(color);
}

void Shape::plot(Eigen::MatrixXd clr) {
    igl::opengl::glfw::Viewer viewer;
    plot(clr, viewer, 0);
    
    // White Background
    viewer.core().background_color.setOnes();
    
    viewer.core().camera_zoom = 0.5;
    viewer.core().set_rotation_type(
        igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_NO_ROTATION);
    viewer.core().lighting_factor = 0.5;
    viewer.launch();
}


Eigen::MatrixXd Shape::getCartesianColorMap() {
    Eigen::MatrixXf color(getNumVertices(), 3);
    color = (V.rowwise() - V.colwise().minCoeff()).array().rowwise() / (V.colwise().maxCoeff() - V.colwise().minCoeff()).array();
    // clip between 0.05 and 0.8
    color = color.array() * 0.75 + 0.05;
    return color.cast<double>();
}

Eigen::MatrixXd Shape::getCartesianColorMapForFaces() {
    Eigen::MatrixXf colorVertices = getCartesianColorMap().cast<float>();
    Eigen::MatrixXf colorFaces(getNumFaces(), 3); colorFaces.setZero();

    for (int f = 0; f < getNumFaces(); f++) {
        for (int i = 0; i < 3; i++) {
            colorFaces(f, Eigen::all) += colorVertices.row(F(f, i));
        }
        colorFaces(f, Eigen::all) = colorFaces(f, Eigen::all) * 0.33333;
    }
    return colorFaces.cast<double>();
}

void Shape::plot(Eigen::MatrixXd clr, igl::opengl::glfw::Viewer &viewer, int viewerDataIdx) {
    // plot shape
    viewer.data(viewerDataIdx).set_mesh(V.cast<double>(), F);
    viewer.data(viewerDataIdx).set_face_based(true);
    viewer.data(viewerDataIdx).set_data(clr);

    // plot open edges
    if (!isWatertight()) {
        Eigen::RowVector3d holeCol(3); holeCol.setZero();
        holeCol(0) = 1;
        const size_t VERTEX_THICKNESS = 15;
        const size_t EDGE_THICKNESS = 7;
        const float POINT_DISTANCE = (V.maxCoeff() - V.minCoeff()) / getNumFaces();
        std::vector<std::vector<int>> holeEdgeIndices;
        const int numHoles = tmhc::find_holes(F, E, LocEinF, holeEdgeIndices);
        int numPlotted = 0;
        for (int hole = 0; hole < numHoles; hole++) {
            for (int i = 0; i < holeEdgeIndices[hole].size(); i++) {
                numPlotted++;
                std::vector<int> holeBoundary = holeEdgeIndices[hole];
                Eigen::MatrixXi boundaryEdge = getE().row((holeBoundary[i]));
                Eigen::MatrixXf P1(1, 3); P1 = getVi( boundaryEdge(0) );
                Eigen::MatrixXf P2(1, 3); P2 = getVi( boundaryEdge(1) );
            #ifdef __APPLE__
                // why so ever we cant control the linewidth on apple computer so we have to use
                // this scetchy workaround
                Eigen::MatrixXf connectionVec = P1 - P2;
                const int numEpoints = connectionVec.norm() / POINT_DISTANCE + 1;
                const float dist = 1 / (float) numEpoints;
                for (int i = 0; i < numEpoints; i++) {
                    viewer.data(viewerDataIdx).add_points((P2 + i * dist * connectionVec).cast<double>(), holeCol);
                    viewer.data(viewerDataIdx).point_size = EDGE_THICKNESS;
                }
            #else
                viewer.data(viewerDataIdx).add_edges(P1.cast<double>(), P2.cast<double>(), holeCol);
                viewer.data(viewerDataIdx).line_width = EDGE_THICKNESS;
            #endif
                //viewer.data(numPlotted).set_colors(holeCol);
            }
        }
    }
}


bool Shape::closeHoles(bool doSurfaceFairing) {
    if (isWatertight()) {
        std::cout << "Shape does not contain holes" << std::endl;
        return true;
    }
    Eigen::MatrixXi Fclosed;
    Eigen::MatrixXd Vclosed;
    tmhc::close_holes(V.cast<double>(), F, Vclosed, Fclosed, doSurfaceFairing);
    F = Fclosed;
    V = Vclosed.cast<float>();

    edgesComputed = false;
    triangleNeighboursComputed = false;
    if (isWatertight()) {
        std::cout << "Successfully closed holes" << std::endl;
        return true;
    }
    std::cout << "Could not close holes successfully" << std::endl;
    return false;
}

bool Shape::closeHoles() {
    return closeHoles(false);
}

bool Shape::closeHolesWithTriFan() {
    std::vector<size_t> boundary_loop;
    igl::boundary_loop(F, boundary_loop);
    assert(boundary_loop.size() > 3);
    const int numNewFaces = F.rows() + boundary_loop.size() - 2;
    Eigen::MatrixXi Fclosed(numNewFaces, 3); Fclosed.setConstant(-1);
    Fclosed.block(0, 0, F.rows(), 3) = F;

    if (boundary_loop.front() == boundary_loop.back()) {
        std::cout << "Did not expect boundary loop to have front == back. PLs adjust code accordingly" << std::endl;
    }
    boundary_loop.push_back(boundary_loop.front());

    int f = F.rows();
    for (int i = 1; i < boundary_loop.size()-2; i++) {
        Fclosed.row(f) << boundary_loop.at(0), boundary_loop.at(i), boundary_loop.at(i+1);
        f++;
    }

    if ((Fclosed.array() == -1).any()) {
        std::cout << "Did not use as many triangles to close holes as expected" << std::endl;
    }

    Eigen::MatrixXi componentIds(Fclosed.rows(), 1); componentIds.setZero();
    igl::bfs_orient(Fclosed, Fclosed, componentIds);

    F = Fclosed;
    edgesComputed = false;
    triangleNeighboursComputed = false;
    if (isWatertight()) {
        std::cout << "Successfully closed holes" << std::endl;
        return true;
    }
    std::cout << "Could not close holes successfully" << std::endl;
    return false;
}
