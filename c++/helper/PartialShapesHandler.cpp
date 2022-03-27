//
//  PartialShapesHandler.cpp
//  ILP_input
//
//  Created by Paul Rötzer on 13.11.21.
//

#include "partialShapesHandler.hpp"
#include <igl/qslim.h>


PartialShapesHandler::PartialShapesHandler(Shape& sX, int _numFacesShapeXWithHoles, Shape& sY, int _numFacesShapeYWithHoles) :
shapeX(sX),
shapeY(sY),
numFacesShapeXWithHoles(_numFacesShapeXWithHoles),
numFacesShapeYWithHoles(_numFacesShapeYWithHoles) {

}

PartialShapesHandler::PartialShapesHandler(std::string shapenameX, std::string shapenameY) {
    shapeX = Shape(shapenameX+ ".ply");
    shapeY = Shape(shapenameY + ".ply");

    numFacesShapeXWithHoles = shapeX.getNumFaces();
    if (!shapeX.isWatertight()) {
        shapeX.writeToFile(shapenameX + "original.ply");
        shapeX.closeHoles();
    }

    numFacesShapeYWithHoles = shapeY.getNumFaces();
    if (!shapeY.isWatertight()) {
        shapeY.writeToFile(shapenameY + "original.ply");
        shapeY.closeHoles();

    }
}

PartialShapesHandler::PartialShapesHandler(std::string shapenameX, bool xhasVersionWithHoles, std::string shapenameY, bool yhasVersionWithHoles) {
    shapeX = Shape(shapenameX+ ".ply");
    shapeY = Shape(shapenameY + ".ply");

    if (xhasVersionWithHoles) {
        Shape shapeXholes = Shape(shapenameX + "original.ply");
        numFacesShapeXWithHoles = shapeXholes.getNumFaces();
    }
    else {
        numFacesShapeXWithHoles = shapeX.getNumFaces();
    }

    if (yhasVersionWithHoles) {
        Shape shapeYholes = Shape(shapenameY + "original.ply");
        numFacesShapeYWithHoles = shapeYholes.getNumFaces();
    }
    else {
        numFacesShapeYWithHoles = shapeY.getNumFaces();
    }
}


Shape reducePartialShape(Shape &shape, int targetNumberOfFaces, int numFacesOriginShapeWithHoles) {

    Eigen::MatrixXd Vdec = shape.getV().cast<double>();
    Eigen::MatrixXi F = shape.getF();
    Eigen::MatrixXi Fdec;
    Eigen::VectorXi IndexInBirthFaces; //   J  #G list of indices into F of birth face
    Eigen::VectorXi IndexInBirthVertices; //   I  #U list of indices into V of birth vertices

    if (!igl::qslim(Vdec, F, targetNumberOfFaces, Vdec, Fdec, IndexInBirthFaces, IndexInBirthVertices)) {
        std::cout << "Cannot reduce shape because potentially not two-manifold" << std::endl;
        exit(0);
    }

    Eigen::MatrixXi isBirthFaceAHole(Fdec.rows(), 1); isBirthFaceAHole = isBirthFaceAHole.setZero();
    Eigen::MatrixXi isVertexOnlyPartOfHole(Vdec.rows(), 1); isVertexOnlyPartOfHole.setOnes();
    for (int f = 0; f < Fdec.rows(); f++) {
        if (IndexInBirthFaces(f) >= numFacesOriginShapeWithHoles) {
            isBirthFaceAHole(f) = 1;
        }
        else {
            for (int i = 0; i < 3; i++) {
                int vertex = Fdec(f, i);
                isVertexOnlyPartOfHole(vertex) = 0;
            }
        }
    }

    // reorder triangles so that the hole closing triangles are in the end
    Eigen::MatrixXi Freorderd(Fdec.rows(), 3); Freorderd = -Freorderd.setOnes();
    const int numTrianglesNotHoles = Fdec.rows() - isBirthFaceAHole.sum();
    int numNotHolesAdded = 0, numHolesAdded = 0;
    for (int f = 0; f < Fdec.rows(); f++) {
        if (!isBirthFaceAHole(f)) {
            Freorderd(numNotHolesAdded, Eigen::all) = Fdec.row(f);
            numNotHolesAdded++;
        }
        else {
            Freorderd(numTrianglesNotHoles + numHolesAdded, Eigen::all) = Fdec.row(f);
            numHolesAdded++;
        }
    }


    // reorder vertices so that the hole-closing vertices are in the end
    Eigen::MatrixXd Vreorderd(Vdec.rows(), 3); Vreorderd.setZero();
    Eigen::MatrixXi translation(Vdec.rows(), 1); translation = - translation.setOnes();
    const int numVerticesNotInHoles = Vdec.rows() - isVertexOnlyPartOfHole.sum();
    numNotHolesAdded = 0;
    numHolesAdded = 0;
    for (int v = 0; v < Vdec.rows();  v++) {
        int idx;
        if (!isVertexOnlyPartOfHole(v)) {
            idx = numNotHolesAdded;
            numNotHolesAdded++;
        }
        else {
            idx = numVerticesNotInHoles + numHolesAdded;
            numHolesAdded++;
        }
        Vreorderd(idx, Eigen::all) = Vdec.row(v);
        translation(v) = idx;
    }

    // change indices of F according to reorder V
    for (int f = 0; f < Fdec.rows(); f++) {
        for (int i = 0; i < 3; i++) {
            Fdec(f, i) = translation(Fdec(f, i));
        }
    }

    shape.modifyV(Vreorderd);
    shape.modifyF(Fdec);

    Shape reducedShapeWithHoles(Vreorderd.block(0, 0, numVerticesNotInHoles, 3).cast<float>(), Fdec.block(0, 0, numTrianglesNotHoles, 3));
    return reducedShapeWithHoles;
}


ShapeMatchModel PartialShapesHandler::generateShapeMatchoModel(ShapeMatchModelOpts opts,  int numberOfTriangles) {
    std::string modelname = opts.modelName;
    std::filesystem::create_directories(modelname);
    float triangleNumberRatio = shapeX.getNumFaces() / (float) shapeY.getNumFaces();
    int targetNumFacesX, targetNumFacesY;
    if (triangleNumberRatio >= 1) {
        targetNumFacesX = numberOfTriangles;
        targetNumFacesY = numberOfTriangles / triangleNumberRatio;
    }
    else {
        targetNumFacesX = numberOfTriangles * triangleNumberRatio;
        targetNumFacesY = numberOfTriangles;
    }

    bool shapeXContainsHoles = shapeX.getNumFaces() > numFacesShapeXWithHoles;
    bool shapeYContainsHoles = shapeY.getNumFaces() > numFacesShapeYWithHoles;

    // reduce shapes
    int numFacesXHoles, numVerticesXHoles, numEdgesXHoles;
    int numFacesYHoles, numVerticesYHoles, numEdgesYHoles;
    if (shapeXContainsHoles) {
        Shape shapeXredHoles = reducePartialShape(shapeX, targetNumFacesX, numFacesShapeXWithHoles);
        numFacesXHoles      = shapeXredHoles.getNumFaces();
        numVerticesXHoles   = shapeXredHoles.getNumVertices();
        numEdgesXHoles      = shapeXredHoles.getNumEdges();
        shapeXredHoles.writeToFile(modelname + "_Xoriginal.ply");

        //shapeXredHoles.plot();
        //shapeX.plot();
    }
    else {
        shapeX.reduce(targetNumFacesX);
    }
    if (shapeYContainsHoles) {
        Shape shapeYredHoles = reducePartialShape(shapeY, targetNumFacesY, numFacesShapeYWithHoles);
        numFacesYHoles      = shapeYredHoles.getNumFaces();
        numVerticesYHoles   = shapeYredHoles.getNumVertices();
        numEdgesYHoles      = shapeYredHoles.getNumEdges();
        shapeYredHoles.writeToFile(modelname + "_Yoriginal.ply");
        
        //shapeYredHoles.plot();
        //shapeY.plot();
    }
    else {
        shapeY.reduce(targetNumFacesY);
    }
    if (!shapeX.isWatertight()) {
        std::cout << "Cannot create shape match model because provided shape X is not watertigth" << std::endl;
        exit(0);
    }
    if (!shapeY.isWatertight()) {
        std::cout << "Cannot create shape match model because provided shape Y is not watertigth" << std::endl;
        exit(0);
    }
    shapeX.writeToFile(modelname + "_X.ply");
    shapeY.writeToFile(modelname + "_Y.ply");


    // create NWMH csv file
    Eigen::MatrixXi Data(2, 7);
    Data(0, 0) = numFacesXHoles;
    Data(0, 1) = numVerticesXHoles;
    Data(0, 2) = numEdgesXHoles;
    Data(0, 3) = shapeX.getNumFaces();
    Data(0, 4) = shapeX.getNumVertices();
    Data(0, 5) = shapeX.getNumEdges();
    Data(0, 6) = shapeXContainsHoles;
    Data(1, 0) = numFacesYHoles;
    Data(1, 1) = numVerticesYHoles;
    Data(1, 2) = numEdgesYHoles;
    Data(1, 3) = shapeY.getNumFaces();
    Data(1, 4) = shapeY.getNumVertices();
    Data(1, 5) = shapeY.getNumEdges();
    Data(1, 6) = shapeYContainsHoles;
    utils::writeMatrixToFile(Data, modelname + "_NWMHData");


    ShapeMatchModel model(modelname, opts);
    return model;
}

