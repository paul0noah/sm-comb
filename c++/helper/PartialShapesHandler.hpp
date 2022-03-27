//
//  PartialShapesHandler.hpp
//  ILP_input
//
//  Created by Paul Rötzer on 13.11.21.
//

#ifndef PartialShapesHandler_hpp
#define PartialShapesHandler_hpp

#include <stdio.h>
#include "helper/shape.hpp"
#include "shapeMatchModel/shapeMatchModel.hpp"
#include <Eigen/Dense>

/*
 PartialShapesHandler can be used to generate a shapematchmodel for partial shapes
 - provide a closed version of a partial shape (use e.g. meshlab)
 - ensure that the closing triangles are added in such a way:
    Fclosed = [Fholes; FclosingTriangles]
 - provide the number of triangles of the original shape (namely length(Fholes))
 - provide the number of face to which each shape should be reduced

 PartialShapesHandler will create a shape match model which will match reduced versions of the shapes.
 Note: the number of triangles of the reduced shapes will be in the same ratio as the number of triangles of the original shapes
 Additionally, when plotted, the additional triangles for closing the holes so that the partial plot is very nice ;)

you can call it in two way:
 1)
    PartialShapesHandler(Shape& sX, int _numFacesShapeXWithHoles, Shape& sY, int _numFacesShapeYWithHoles);
    sX: closed version of shapeX
    _numFacesShapeXWithHoles: number of triangles of shapeX with holes
    sY: closed version of shapeY
    _numFacesShapeYWithHoles: number of triangles of shapeY with holes
 2)
    PartialShapesHandler(std::string shapenameX, bool xhasVersionWithHoles, std::string shapenameY, bool yhasVersionWithHoles);
     shapenameX: shape name without the file ending. asserts shape is stored as .ply
     xhasVersionWithHoles: if this is true, you must provide a shape with filename "<shapenameX>original.ply"
        and this file must contain version of shapeX with holes
    shapenameY: shape name without the file ending. asserts shape is stored as .ply
    yhasVersionWithHoles: if this is true, you must provide a shape with filename "<shapenameY>original.ply"
        and this file must contain version of shapeY with holes

=> for both ways: when created the object you can create an shape match model as follows
 ShapeMatchModelOpts opts;
 opts.modelname = "path/to/where/you/want/to/store/the/model";
 opts.writeModelToFileAfterSolving = true; // make sure results get saved
 ShapeMatchModel model = partialShapesHandler.generateShapeMatchoModel(opts, numberOfTriangles);
 */
class PartialShapesHandler {
private:
    Shape shapeX;
    int numFacesShapeXWithHoles;
    Shape shapeY;
    int numFacesShapeYWithHoles;


public:
    PartialShapesHandler(std::string shapenameX, std::string shapenameY);
    PartialShapesHandler(std::string shapenameX, bool xhasVersionWithHoles, std::string shapenameY, bool yhasVersionWithHoles);
    PartialShapesHandler(Shape& sX, int _numFacesShapeXWithHoles, Shape& sY, int _numFacesShapeYWithHoles);
    ShapeMatchModel generateShapeMatchoModel(ShapeMatchModelOpts opts,  int numberOfTriangles);

};


#endif /* PartialShapesHandler_hpp */
