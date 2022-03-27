//
//  plotSolution.cpp
//
//  Created by Paul Rötzer on 29.07.21.
//

#include "shapeMatchModel/shapeMatchModel.hpp"
#include <igl/opengl/glfw/Viewer.h>
#include <igl/colormap.h>

struct Matching {
    unsigned int idxVX;
    unsigned int idxVY;
    Matching(unsigned int idxVX, unsigned int idxVY):
    idxVX(idxVX), idxVY(idxVY) {}
    bool operator==(const Matching& match) const {
        return idxVX == match.idxVX && idxVY == match.idxVY;
    }
    bool operator<(const Matching& match) const
    {
        return idxVX < match.idxVX || idxVY < match.idxVY;
    }
    friend std::ostream& operator<<(std::ostream& os, const Matching& match){
        os << "(" << match.idxVX << ", " << match.idxVY << ")";
        return os;
    }
};


const size_t VERTEX_THICKNESS = 15;
const size_t EDGE_THICKNESS = 7;
const float POINT_DISTANCE = 0.005;

void ShapeMatchModel::plotSolution(const SparseVecInt8 &G) {
    
    const Eigen::MatrixXi FXCombo = getCombinations().getFaCombo();
    const Eigen::MatrixXi FYCombo = getCombinations().getFbCombo();
    
    const size_t numVerticesX = shapeX.getNumVertices();
    const size_t numVerticesY = shapeY.getNumVertices();
    const size_t numEdgesX = shapeX.getNumEdges();
    const size_t numEdgesY = shapeY.getNumEdges();
    const size_t numFacesX = shapeX.getNumFaces();
    const size_t numFacesY = shapeY.getNumFaces();
    
    const size_t numNonDegenerate = numFacesX * numFacesY * 3;
    const size_t numTriangleYToVertexX = numFacesY * numVerticesX;
    const size_t numTriangleYToEdgeX = numFacesY * 3 * 2 * numEdgesX;
    const size_t numDegenerateX = numTriangleYToVertexX + numTriangleYToEdgeX;
    const size_t numTriangleXToVertexY = numFacesX * numVerticesY;
    const size_t numTriangleXToEdgeY = numFacesX * 3 * 2 * numEdgesY;
    const size_t numDegenerateY = numTriangleXToVertexY + numTriangleXToEdgeY;
    
    const size_t numNonZerosInG = G.nonZeros();
    
    igl::opengl::glfw::Viewer viewer;
    int numFacesPlotted = 0;
    for (typename Eigen::SparseVector<int8_t, Eigen::RowMajor>::InnerIterator it(G); it; ++it) {
        const int idx = it.index();
        
        const double colorFactor =  (double) numFacesPlotted / numNonZerosInG;
        double col[3];
        igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_JET, colorFactor, col);
        Eigen::RowVector3d color = Eigen::RowVector3d(col[0], col[1], col[2]);
        
        // non-degenerate faces
        if (idx < numNonDegenerate) {
            const Eigen::Vector3i fX = FXCombo(idx, Eigen::all);
            const Eigen::Vector3i fY = FYCombo(idx, Eigen::all);
            const Eigen::Matrix3f vX = (shapeX.getV()) (fX, Eigen::all);
            const Eigen::Matrix3f vY = (shapeY.getV()) (fY, Eigen::all);
            Eigen::MatrixXf Vtemp(6, 3);
            Vtemp.block(0, 0, 3, 3) = vX;
            Vtemp.block(3, 0, 3, 3) = vY;
            Eigen::MatrixXi Ftemp(2, 3);
            Ftemp << 0, 1, 2, 3, 4, 5;
            viewer.data(numFacesPlotted).set_mesh(Vtemp.cast<double>(), Ftemp);
            viewer.data(numFacesPlotted).set_face_based(true);
            
            viewer.data(numFacesPlotted).set_colors(color);
        }
        
        // degenerate in X
        else if (idx < numNonDegenerate + numDegenerateX) {
            const Eigen::Vector3i fX = FXCombo(idx, Eigen::all);
            const Eigen::Vector3i fY = FYCombo(idx, Eigen::all);
            const Eigen::Matrix3f vY = (shapeY.getV()) (fY, Eigen::all);
            
            Eigen::MatrixXi Ftemp(1, 3);
            Ftemp << 0, 1, 2;
            // plot the triangle of shape Y
            viewer.data(numFacesPlotted).set_mesh(vY.cast<double>(), Ftemp);
            viewer.data(numFacesPlotted).set_face_based(true);
            viewer.data(numFacesPlotted).set_colors(color);
            
            // triangle to vertex
            if (idx < numNonDegenerate + numTriangleYToVertexX) {
                // plot the point of shape X
                Eigen::MatrixXf P(1, 3); P = shapeX.getVi( fX(0) );
                viewer.data(numFacesPlotted).add_points(P.cast<double>(), color);
                viewer.data(numFacesPlotted).point_size = VERTEX_THICKNESS;
            }
            
            // triangle to edge
            else {
                // find the two distinct points in the degenerate triangle
                int edgeidxX1 = fX(0);
                int edgeidxX2 = edgeidxX1;
                for (int i = 1; i < 3; i++) {
                    if (fX(i) != edgeidxX1) {
                        edgeidxX2 = fX(i);
                        break;
                    }
                }
                assert(edgeidxX1 != edgeidxX2);
                
                // plot edge
                Eigen::MatrixXf P1(1, 3); P1 = shapeX.getVi( edgeidxX1 );
                Eigen::MatrixXf P2(1, 3); P2 = shapeX.getVi( edgeidxX2 );
            #ifdef __APPLE__
                // why so ever we cant control the linewidth on apple computer so we have to use
                // this scetchy workaround
                Eigen::MatrixXf connectionVec = P1 - P2;
                const int numEpoints = connectionVec.norm() / POINT_DISTANCE + 1;
                const float dist = 1 / (float) numEpoints;
                for (int i = 0; i < numEpoints; i++) {
                    viewer.data(numFacesPlotted).add_points((P2 + i * dist * connectionVec).cast<double>(), color);
                    viewer.data(numFacesPlotted).point_size = EDGE_THICKNESS;
                }
            #else
                viewer.data(numFacesPlotted).add_edges(P1.cast<double>(), P2.cast<double>(), color);
                viewer.data(numFacesPlotted).line_width = EDGE_THICKNESS;
            #endif
            }

        }
        // degenerate in Y
        else if (idx < numNonDegenerate + numDegenerateX + numDegenerateY) {
            const Eigen::Vector3i fX = FXCombo(idx, Eigen::all);
            const Eigen::Vector3i fY = FYCombo(idx, Eigen::all);
            const Eigen::Matrix3f vX = (shapeX.getV()) (fX, Eigen::all);
            
            Eigen::MatrixXi Ftemp(1, 3);
            Ftemp << 0, 1, 2;
            // plot the triangle of shape Y
            viewer.data(numFacesPlotted).set_mesh(vX.cast<double>(), Ftemp);
            viewer.data(numFacesPlotted).set_face_based(true);
            viewer.data(numFacesPlotted).set_colors(color);
            
            // triangle to vertex
            if (idx < numNonDegenerate + numDegenerateX + numTriangleXToVertexY) {
                // plot the point of shape Y
                Eigen::MatrixXf P(1, 3); P = shapeY.getVi( fY(0) );
                viewer.data(numFacesPlotted).add_points(P.cast<double>(), color);
                viewer.data(numFacesPlotted).point_size = VERTEX_THICKNESS;
            }
            
            // triangle to edge
            else {
                // find the two distinct points in the degenerate triangle
                int edgeidxY1 = fY(0);
                int edgeidxY2 = edgeidxY1;
                for (int i = 1; i < 3; i++) {
                    if (fY(i) != edgeidxY1) {
                        edgeidxY2 = fY(i);
                        break;
                    }
                }
                assert(edgeidxY1 != edgeidxY2);
                
                // plot edge
                Eigen::MatrixXf P1(1, 3); P1 = shapeY.getVi( edgeidxY1 );
                Eigen::MatrixXf P2(1, 3); P2 = shapeY.getVi( edgeidxY2 );
            #ifdef __APPLE__
                // why so ever we cant control the linewidth on apple computer so we have to use
                // this scetchy workaround
                Eigen::MatrixXf connectionVec = P1 - P2;
                const int numEpoints = connectionVec.norm() / POINT_DISTANCE + 1;
                const float dist = 1 / (float) numEpoints;
                for (int i = 0; i < numEpoints; i++) {
                    viewer.data(numFacesPlotted).add_points((P2 + i * dist * connectionVec).cast<double>(), color);
                    viewer.data(numFacesPlotted).point_size = EDGE_THICKNESS;
                }
            #else
                viewer.data(numFacesPlotted).add_edges(P1.cast<double>(), P2.cast<double>(), color);
                viewer.data(numFacesPlotted).line_width = EDGE_THICKNESS;
            #endif
            }
            
            
        }
        else {
            ASSERT_NEVER_REACH;
        }
        viewer.append_mesh();
        numFacesPlotted++;

    }
    
    
    // White Background
    viewer.core().background_color.setOnes();
    
    // center camera
    Eigen::MatrixXd VBig(numVerticesX + numVerticesY, 3);
    VBig.block(0, 0, numVerticesX, 3) = shapeX.getV().cast<double>();
    VBig.block(numVerticesX, 0, numVerticesY, 3) = shapeY.getV().cast<double>();
    viewer.core().align_camera_center(VBig);
    
    viewer.core().camera_zoom = 0.5;
    viewer.core().set_rotation_type(
        igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_NO_ROTATION);
    viewer.core().lighting_factor = 0.5;
    viewer.launch();
}

void ShapeMatchModel::plotInterpolatedSolution(const SparseVecInt8 &G) {
    const Eigen::MatrixXi FXCombo = getCombinations().getFaCombo();
    const Eigen::MatrixXi FYCombo = getCombinations().getFbCombo();
    std::set<Matching> matchings;
    
    // extract all matchings
    for (typename Eigen::SparseVector<int8_t, Eigen::RowMajor>::InnerIterator it(G); it; ++it) {
        const int idx = it.index();
        
        Eigen::MatrixXi faceX = FXCombo.row(idx);
        Eigen::MatrixXi faceY = FYCombo.row(idx);
        
        for (int i = 0; i < 3; i++) {
            // std::set keeps track if we already added a matching
            matchings.insert(Matching(faceX(i), faceY(i)));
        }
    }
    
    // compute colorlabel for each shape
    Eigen::MatrixXd colorX(shapeX.getNumVertices(), 3);
    colorX = shapeX.getCartesianColorMap();
    Eigen::MatrixXd colorY(shapeY.getNumVertices(), 3);
    colorY = -colorY.setOnes();
    
    const int numMatchings = matchings.size();
    int i = 0;
    double col[3];
    for (std::set<Matching>::iterator it = matchings.begin(); it != matchings.end(); ++it) {
        const Matching match = *it;
        /*const double colorFactor = (double) i / numMatchings;
        igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_JET, colorFactor, col);
        if (colorX(match.idxVX, 0) < 0) {
            colorX(match.idxVX, 0) = col[0];
            colorX(match.idxVX, 1) = col[1];
            colorX(match.idxVX, 2) = col[2];
        }*/
        if (colorY(match.idxVY, 0) < 0) {
            colorY(match.idxVY, Eigen::all) = colorX.row(match.idxVX);
        }
        i++;
    }
    bool useFaceBasedPlotting = true;
    if (useFaceBasedPlotting) {
        colorX.resize(shapeX.getNumFaces(), 3);
        colorX = shapeX.getCartesianColorMapForFaces();

        Eigen::MatrixXd colorYFaces(shapeY.getNumFaces(), 3); colorYFaces.setZero();
        Eigen::MatrixXi FY = shapeY.getF();
        for (int f = 0; f < shapeY.getNumFaces(); f++) {
            bool anyNoMatch = false;
            for (int i = 0; i < 3; i++) {
                colorYFaces(f, Eigen::all) += colorY.row(FY(f, i));
                if (colorY(FY(f, i), 0) == -1)
                    anyNoMatch = true;
            }
            colorYFaces(f, Eigen::all) = colorYFaces(f, Eigen::all) * 0.3333;
            if (anyNoMatch)
                colorYFaces(f, Eigen::all) << 0, 0, 0;
        }
        colorY.resize(shapeX.getNumFaces(), 3);
        colorY = colorYFaces;
    }


    igl::opengl::glfw::Viewer viewer;

    if (nonWatertightMeshHandler.filledHolesOfShapeX()) {
        Shape shapeXHoles = nonWatertightMeshHandler.getShapeXWithHoles();
        if (useFaceBasedPlotting)
            shapeXHoles.plot(colorX.block(0, 0, shapeXHoles.getNumFaces(), 3), viewer, 0);
        else
            shapeXHoles.plot(colorX.block(0, 0, shapeXHoles.getNumVertices(), 3), viewer, 0);
    }
    else {
        shapeX.plot(colorX, viewer, 0);
    }
    viewer.append_mesh();
    if (nonWatertightMeshHandler.filledHolesOfShapeY()) {
        Shape shapeYHoles = nonWatertightMeshHandler.getShapeYWithHoles();
        if (useFaceBasedPlotting)
            shapeYHoles.plot(colorY.block(0, 0, shapeYHoles.getNumFaces(), 3), viewer, 1);
        else
            shapeYHoles.plot(colorY.block(0, 0, shapeYHoles.getNumVertices(), 3), viewer, 1);
    }
    else {
        shapeY.plot(colorY, viewer, 1);
    }
    
    // White Background
    viewer.core().background_color.setOnes();

    // center camera
    const int numVerticesX = shapeX.getNumVertices();
    const int numVerticesY = shapeY.getNumVertices();
    Eigen::MatrixXd VBig(numVerticesX + numVerticesY, 3);
    VBig.block(0, 0, numVerticesX, 3) = shapeX.getV().cast<double>();
    VBig.block(numVerticesX, 0, numVerticesY, 3) = shapeY.getV().cast<double>();
    viewer.core().align_camera_center(VBig);

    viewer.core().camera_zoom = 0.5;
    viewer.core().set_rotation_type(
        igl::opengl::ViewerCore::RotationType::ROTATION_TYPE_NO_ROTATION);
    viewer.core().lighting_factor = 0.2;
    viewer.launch();
}

