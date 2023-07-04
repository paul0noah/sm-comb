//
//  deformationEnergy.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#include "deformationEnergy.hpp"
#include "wksEnergy.hpp"
#include "helper/utils.hpp"
#include <chrono>
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif


DeformationEnergy::DeformationEnergy(Shape& sA, Shape& sB, Combinations& c) :
    membraneEnergy(),
    bendingEnergy(),
    shapeA(sA),
    shapeB(sB),
    combos(c),
    computed(false) {

}


/* function DeformationEnergy::get
   according to (1):
          | memE(ShapeA, ShapeB) + memE(ShapeB, ShapeA); A, B non-degenerate
   memE = | 2 * memE(ShapeB, ShapeA); A degenerate
          | 2 * memE(ShapeA, ShapeB); B degenerate
   and deformationEnergy = memE + lambda * bendE + mu * wksE
 */
Eigen::MatrixXf DeformationEnergy::get() {
    if(computed) {
        return defEnergy;
    }
    const int numFacesA = shapeA.getNumFaces();
    const int numFacesB = shapeB.getNumFaces();
    const int numVerticesA = shapeA.getNumVertices();
    const int numVerticesB = shapeB.getNumVertices();
    const int numEdgesA = shapeA.getNumEdges();
    const int numEdgesB = shapeB.getNumEdges();
    const int numFacesAB = numFacesA * numFacesB;
    const int numNonDegenerate = 3 * numFacesAB;
    const int numDegenerateA = 3 * 2 * numEdgesA * numFacesB + numVerticesA * numFacesB;
    const int numDegenerateB = 3 * 2 * numEdgesB * numFacesA + numVerticesB * numFacesA;
    
    // compute the combinations which get stored in FaCombo and FbCombo respectively
    Eigen::MatrixXi& FaCombo = combos.getFaCombo();
    Eigen::MatrixXi& FbCombo = combos.getFbCombo();
    
    // lambda models material properties
    const float lambda = 5;
    const float mu = 1;
    Eigen::MatrixXf deformationEnergy(numNonDegenerate + numDegenerateA + numDegenerateB, 1);
    
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    #if defined(_OPENMP)
    #pragma omp parallel
    #pragma omp sections
    #endif
    {
    
        // non-degenerate part of membrane energy
        #if defined(_OPENMP)
        #pragma omp section
        #endif
        deformationEnergy.block(0, 0, numNonDegenerate, 1) =
            membraneEnergy.get(shapeA, shapeB,
                               FaCombo.block(0, 0, numNonDegenerate, 3),
                               FbCombo.block(0, 0, numNonDegenerate, 3)) +
            membraneEnergy.get(shapeB, shapeA,
                               FbCombo.block(0, 0, numNonDegenerate, 3),
                               FaCombo.block(0, 0, numNonDegenerate, 3));
         
        // degenerate cases in A
        #if defined(_OPENMP)
        #pragma omp section
        #endif
        deformationEnergy.block(numNonDegenerate, 0, numDegenerateA, 1) =
            2 * membraneEnergy.get(shapeB, shapeA,
                               FbCombo.block(numNonDegenerate, 0, numDegenerateA, 3),
                               FaCombo.block(numNonDegenerate, 0, numDegenerateA, 3));

        // degenerate cases in B
        #if defined(_OPENMP)
        #pragma omp section
        #endif
        deformationEnergy.block(numNonDegenerate + numDegenerateA, 0, numDegenerateB, 1) =
            2 * membraneEnergy.get(shapeA, shapeB,
                               FaCombo.block(numNonDegenerate + numDegenerateA, 0, numDegenerateB, 3),
                               FbCombo.block(numNonDegenerate + numDegenerateA, 0, numDegenerateB, 3));
    }

    deformationEnergy = (deformationEnergy.array() * (1.0f/ deformationEnergy.mean()) ).matrix();
    //std::cout << "Mean membrane energy " <<  deformationEnergy.mean() << std::endl;
    // bending energy is easier to handle ;)
    const Eigen::MatrixXf bendE = bendingEnergy.get(shapeA, shapeB, FaCombo, FbCombo);
    deformationEnergy += lambda * (bendE.array() * (1.0f / bendE.mean())).matrix();
    if (mu > 0) {
        WKSEnergy wksEnergy = WKSEnergy();
        const Eigen::MatrixXf wksE = wksEnergy.get(shapeA, shapeB, FaCombo, FbCombo);
        //std::cout << "Mean wks energy " << wksE.mean() << std::endl;
        //std::cout << "Mean bending energy " << bendingEnergy.get(shapeA, shapeB, FaCombo, FbCombo).mean() << std::endl;
        deformationEnergy += mu * (wksE.array() * (1.0f / wksE.mean()) ).matrix();
    }
    
    const float floatEpsilon = 1e-7;
    const float minCoeff = deformationEnergy.minCoeff();
    if (minCoeff < -FLOAT_EPSI) {
        ASSERT_NEVER_REACH;
    }
    if (minCoeff < FLOAT_EPSI ) {
        deformationEnergy = (deformationEnergy.array() + std::abs(deformationEnergy.minCoeff())).matrix();
    }
    
    defEnergy = deformationEnergy;
    computed = true;
    return deformationEnergy;
}

void DeformationEnergy::modifyEnergyVal(const int index, float newVal) {
    assert(index > 0 && index < defEnergy.rows());
    defEnergy(index) = newVal;
}


void DeformationEnergy::useCustomDeformationEnergy(const Eigen::MatrixXf& Vx2VyCostMatrix, bool useAreaWeighting, bool membraneReg, float lambda) {
    const bool useTranspose = Vx2VyCostMatrix.rows() != shapeA.getNumVertices();

    const Eigen::MatrixXi& FaCombo = combos.getFaCombo();
    const Eigen::MatrixXi& FbCombo = combos.getFbCombo();

    Eigen::ArrayXXf energy(FaCombo.rows(), 3);

    if (useAreaWeighting) {
        // Init curvature and area vectors
        Eigen::VectorXf Aa(shapeA.getNumVertices());
        Eigen::VectorXf Ab(shapeB.getNumVertices());
        WKSEnergy wksEnergy = WKSEnergy();
        wksEnergy.getA(shapeA, Aa);
        wksEnergy.getA(shapeB, Ab);

        energy(Eigen::all, 0) = Aa(FaCombo(Eigen::all, 0)) + Ab(FbCombo(Eigen::all, 0));
        energy(Eigen::all, 1) = Aa(FaCombo(Eigen::all, 1)) + Ab(FbCombo(Eigen::all, 1));
        energy(Eigen::all, 2) = Aa(FaCombo(Eigen::all, 2)) + Ab(FbCombo(Eigen::all, 2));
    }
    else {
        energy.setOnes();

    }

    Eigen::ArrayXXf temp(FaCombo.rows(), 3);
    if (useTranspose) {
        for (int j = 0; j < FaCombo.rows(); j++) {
            for (int i = 0; i < 3; i++) {
                temp(Eigen::all, i) = Vx2VyCostMatrix(FaCombo(j, i), FbCombo(j, i));
            }
        }
    }
    else {
        for (int j = 0; j < FaCombo.rows(); j++) {
            for (int i = 0; i < 3; i++) {
                temp(Eigen::all, i) = Vx2VyCostMatrix(FbCombo(j, i), FaCombo(j, i));
            }
        }
    }

    // update energy
    energy = energy.cwiseProduct(temp.square());
    defEnergy = energy.matrix().rowwise().sum();


    /*
     Membrane Energy
     */
    if (membraneReg) {
        const long numFacesA = shapeA.getNumFaces();
        const long numFacesB = shapeB.getNumFaces();
        const long numVerticesA = shapeA.getNumVertices();
        const long numVerticesB = shapeB.getNumVertices();
        const long numEdgesA = shapeA.getNumEdges();
        const long numEdgesB = shapeB.getNumEdges();
        const long numFacesAB = numFacesA * numFacesB;
        const long numNonDegenerate = 3 * numFacesAB;
        const long numDegenerateA = 3 * 2 * numEdgesA * numFacesB + numVerticesA * numFacesB;
        const long numDegenerateB = 3 * 2 * numEdgesB * numFacesA + numVerticesB * numFacesA;
        Eigen::MatrixXf memE(numNonDegenerate + numDegenerateA + numDegenerateB, 1);
        #if defined(_OPENMP)
        #pragma omp parallel
        #pragma omp sections
        #endif
        {

            // non-degenerate part of membrane energy
            #if defined(_OPENMP)
            #pragma omp section
            #endif
            memE.block(0, 0, numNonDegenerate, 1) =
            membraneEnergy.get(shapeA, shapeB,
                               FaCombo.block(0, 0, numNonDegenerate, 3),
                               FbCombo.block(0, 0, numNonDegenerate, 3)) +
            membraneEnergy.get(shapeB, shapeA,
                               FbCombo.block(0, 0, numNonDegenerate, 3),
                               FaCombo.block(0, 0, numNonDegenerate, 3));

            // degenerate cases in A
            #if defined(_OPENMP)
            #pragma omp section
            #endif
            memE.block(numNonDegenerate, 0, numDegenerateA, 1) =
            2 * membraneEnergy.get(shapeB, shapeA,
                                   FbCombo.block(numNonDegenerate, 0, numDegenerateA, 3),
                                   FaCombo.block(numNonDegenerate, 0, numDegenerateA, 3));

            // degenerate cases in B
            #if defined(_OPENMP)
            #pragma omp section
            #endif
            memE.block(numNonDegenerate + numDegenerateA, 0, numDegenerateB, 1) =
            2 * membraneEnergy.get(shapeA, shapeB,
                                   FaCombo.block(numNonDegenerate + numDegenerateA, 0, numDegenerateB, 3),
                                   FbCombo.block(numNonDegenerate + numDegenerateA, 0, numDegenerateB, 3));
        }

        defEnergy = lambda * defEnergy + memE;
    }
}


/* REFERENCES:
 (1) WINDHEUSER, Thomas, et al. Large‐scale integer linear programming for
     orientation preserving 3d shape matching. In: Computer Graphics Forum.
     Oxford, UK: Blackwell Publishing Ltd, 2011. S. 1471-1480.
 */
