//
//  shape.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#ifndef shape_hpp
#define shape_hpp

#include <Eigen/Dense>
#include <igl/opengl/glfw/Viewer.h>

class Shape {
private:
    // Vertices
    Eigen::MatrixXf V;
    // Faces
    Eigen::MatrixXi F;
    // Edges
    Eigen::MatrixXi E;
    // Location of Edges in F
    Eigen::MatrixXi LocEinF;
    
    Eigen::MatrixXi triangleNeighbours;
    
    bool edgesComputed;
    
    bool triangleNeighboursComputed;
    
    void computeEdges();
    float getAngle(Eigen::Vector3f e0, Eigen::Vector3f e1);
    void add2VertexAttachments(Eigen::VectorXi &vertexAttachments, int vertex0, int vertex1);
    bool watertight;
    void initShape();
    
public:
    Shape();
    Shape(std::string filename);
    Shape(Eigen::MatrixXf Vinp, Eigen::MatrixXi Finp);
    const long getNumFaces();
    const long getNumEdges();
    const long getNumVertices();
    Eigen::VectorXi getTrianglesAttachedToVertex(unsigned int vertexIdx);
    const float getTriangleArea(Eigen::MatrixXi triangle);
    const float getTriangleArea(int triangleIdx);
    Eigen::MatrixXf getTriangleAreas(Eigen::MatrixXi Fcombo);
    Eigen::MatrixXf getTriangleAreas();
    const float getTwiceTriangleArea(Eigen::MatrixXi triangle);
    Eigen::Vector3f getTriangleAngles(unsigned int triangleIdx);
    void modifyV(Eigen::MatrixXf newV);
    void modifyV(Eigen::MatrixXd newV);
    void modifyF(Eigen::MatrixXi newF);
    Eigen::MatrixXf getVi(int vertexIdx);
    Eigen::MatrixXi getFi(int triangleIdx);
    Eigen::MatrixXf getV();
    Eigen::MatrixXi getF();
    Eigen::MatrixXi getE();
    Eigen::MatrixXi getLocEinF();
    Eigen::MatrixXi getLocFinE();
    bool isWatertight();
    bool reduce(size_t numFaces);
    bool writeToFile(std::string filename);
    void translate(const Eigen::Vector3f translationVector);
    float squashInUnitBox();
    void squash(const float squashFactor);
    Eigen::MatrixXi getTriangleNeighbours();
    Eigen::MatrixXi reOrderTriangulation(int seed);
    Eigen::MatrixXi reOrderTriangulation();
    void reorderVertices();
    void plot();
    void plot(Eigen::MatrixXd clr);
    void plot(Eigen::MatrixXd clr, igl::opengl::glfw::Viewer &viewer, int viewerDataIdx);
    bool closeHoles();
    bool closeHoles(bool doSurfaceFairing);
    Eigen::MatrixXd getCartesianColorMap();
    Eigen::MatrixXd getCartesianColorMapForFaces();
};

#endif /* shape_hpp */
