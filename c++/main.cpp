//
//  main.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//
#include "test/tests.hpp"
#include "helper/partialShapesHandler.hpp"
#include <filesystem>

int main(int argc, char *argv[]) {
    /*
     WARNING: we recommend matching shapes only with less than 1000 triangles. Otherwise the program may crash due to high memory usage.
     If you have a capable machine, try using more triangles at you own risk. Note: ProblemSize ~ 20 * NumTrianglesOneShape * NumTrianglesOneShape

     Remember to compile your code with >>>>release<<<< option to obtain maximum performance.

     Parts of the functions run on multiple cores => we recommend compiling with OpenMP
     */

    // create a directory to store the shape matching files into
    std::string path = "shapeMatchingExampleFiles/";
    std::filesystem::create_directories(path);
    // put the example files into this directory
    test::generateExamples(path);

    /*
     Matching two closed, oriented, two-manifold shapes (whenver this is not the case, our code should complain)
     */
    {
        // load shapes (supported file extensions are .ply and .off)
        Shape shapeX(path + "example_X.ply");
        Shape shapeY(path + "example_Y.ply");
        // reduce shapes to reasonable amount of triangles
        shapeX.reduce(60);
        shapeX.reduce(50);

        // generate options struct and modify accordingly
        ShapeMatchModelOpts opts;
        opts.modelName = path + "closedShapes";
        // you can write the solution as well as all shapes to csv file. It also allows to recreate the shape matching
        // instances by simply providing the model name via ShapeMatchModel model(modelname);
        opts.writeModelToFileAfterSolving = false;

        // create the shape matching problem
        ShapeMatchModel model(shapeX, shapeY, opts);

        // solve the model
        MatrixInt8 Gamma = model.solve();

        // plot solution
        model.plotInterpolatedSolution(Gamma.sparseView());
    }

    /*
        Matching two partial shapes, assumes you provide for each shape the open and closed version with the following naming:
         - shapeX: closed shape:     modelname + "_X.ply"
                   shape with holes: modelname + "_Xoriginal.ply"
         - shapeY: closed shape:     modelname + "_Y.ply"
                   shape with holes: modelname + "_Yoriginal.ply"
        Ín the "path" folder, you can find the "partialExample_*" shapes for partial matchings
     */
    {

        // create a partial shapes handler which takes care of reducing the shapes as well as identifying hole-closing triangles on reduced shapes
        bool shapeXcontainsHoles = true;
        bool shapeYcontainsHoles = true;
        std::string filenameX = path + "partialExample_X"; // dont add fileextension here!
        std::string filenameY = path + "partialExample_Y"; // dont add fileextension here!
        PartialShapesHandler psh(filenameX, shapeXcontainsHoles, filenameY, shapeYcontainsHoles);

        // generate options struct and modify accordingly
        ShapeMatchModelOpts opts;
        opts.modelName = path + "partialExample";
        opts.writeModelToFileAfterSolving = true;

        // generate shape match model with partial shapes handler
        int maxNumFacesShapes = 150; // shapes will be reduced to <=150 triangles
        ShapeMatchModel model = psh.generateShapeMatchoModel(opts, maxNumFacesShapes);

        // solve the model
        MatrixInt8 Gamma = model.solve();

        // plot solution (will not plot hole-closing triangles)
        model.plotInterpolatedSolution(Gamma.sparseView());

    }

}
