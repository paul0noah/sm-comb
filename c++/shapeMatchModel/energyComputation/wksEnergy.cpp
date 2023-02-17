//
//  wksEnergy.cpp
//  shape-matching-dd
//
//  Created by Paul Rötzer on 28.10.21.
//

#include "wksEnergy.hpp"
#include <math.h>
#include <iostream>
#include "helper/utils.hpp"
#include <igl/principal_curvature.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/eigs.h>
#include <igl/LinSpaced.h>


/* function getVoronoiArea
         R
         -
        / \
       /   \
      /     \
     /       \
    /_________\
  P             Q
  P is the vertex i for which we want to compute the Voronoi area:
  According to the paper by Meyer:
    A_voronoi = 1/8 * ( norm(PQ)^2 * cot(angle(R)) +  norm(PR)^2 * cot(angle(Q)))
 */
float WKSEnergy::getVoronoiArea(int i, Shape &shape, int neighboor, Eigen::MatrixXf &cotTriangleAngles) {
    Eigen::Vector3i triangle = shape.getFi(neighboor).transpose();
    Eigen::Vector3f P, Q, R;
    float cotAngleR, cotAngleQ;

    if (i == triangle(0)) {
        P = shape.getVi(triangle(0)).transpose();
        Q = shape.getVi(triangle(1)).transpose();
        R = shape.getVi(triangle(2)).transpose();
        cotAngleQ = cotTriangleAngles(neighboor, 1);
        cotAngleR = cotTriangleAngles(neighboor, 2);
    }
    else if (i == triangle(1)) {
        P = shape.getVi(triangle(1)).transpose();
        Q = shape.getVi(triangle(2)).transpose();
        R = shape.getVi(triangle(0)).transpose();
        cotAngleQ = cotTriangleAngles(neighboor, 2);
        cotAngleR = cotTriangleAngles(neighboor, 0);
    }
    else if (i == triangle(2)) {
        P = shape.getVi(triangle(2)).transpose();
        Q = shape.getVi(triangle(0)).transpose();
        R = shape.getVi(triangle(1)).transpose();
        cotAngleQ = cotTriangleAngles(neighboor, 0);
        cotAngleR = cotTriangleAngles(neighboor, 1);
    }
    return 0.125 * (utils::squaredNorm(P - Q) * cotAngleR + utils::squaredNorm(P - R) * cotAngleQ);
}


/* function getMixedArea

    computes the mixed area as proposed in (2) Fig. 4 (figure shows
    a pseudo-code version of the code which is implemented here)
    => in case of obtuse triangles the Voroni area around vertex i is not
    within the one-ring-neighboorhood or truncated and thus not plausible
    for the curvature computation.
 */
float WKSEnergy::getMixedArea(int i, Shape &shape, Eigen::VectorXi &oneRingNeighboorhood, Eigen::MatrixXf &triangleAngles, Eigen::MatrixXf &cotTriangleAngles) {

    float Amixed = 0;
    const double piHalf = M_PI * 0.5;

    for (int k = 0; k < oneRingNeighboorhood.rows(); k++) {

        // check if any angle in triangle is obtuse
        // => one angle has to be greater than piHalf
        bool obtuse = false;
        for (int j = 0; j < 3; j++) {
            if (triangleAngles(oneRingNeighboorhood(k), j) > piHalf) {
                obtuse = true;
                break;
            }
        }
        if (obtuse) {
            int idxAngle;
            // find the vertex we are currently talking about
            Eigen::Vector3i currTriangle = shape.getFi(oneRingNeighboorhood(k)).transpose();
            for (int j = 0; j < 3; j++) {
                if (currTriangle(j) == i){
                    idxAngle = j;
                    break;
                }
            }
            float factor = triangleAngles(oneRingNeighboorhood(k), idxAngle) >= piHalf ? 0.5 : 0.25;

            Amixed = Amixed + factor * shape.getTriangleArea(oneRingNeighboorhood(k));
        }
        // non-obtuse triangle
        else {
            Amixed = Amixed + getVoronoiArea(i, shape, oneRingNeighboorhood(k), cotTriangleAngles);
        }
    }
    return Amixed;
}


/* function getMeanCurvatures
    computes the mean curvatures for each vertex as proposed in (2)
 */
void WKSEnergy::getA(Shape &shape, Eigen::VectorXf &A) {

    Eigen::MatrixXf triangleAngles(shape.getNumFaces(), 3);
    for (int i = 0; i < shape.getNumFaces(); i++) {
        triangleAngles.row(i) = shape.getTriangleAngles(i);
    }

    Eigen::MatrixXf cotTriangleAngles(shape.getNumFaces(), 3);
    // cotan(x) = 1 / tan(x)
    cotTriangleAngles = triangleAngles.array().tan().inverse().matrix();

    Eigen::VectorXi oneRingNeighboorhood;
    Eigen::MatrixXf sumOverOneRingNeighborhood;
    for (int i = 0; i < A.rows(); i++) {
        oneRingNeighboorhood = shape.getTrianglesAttachedToVertex(i);
        A(i) = getMixedArea(i, shape, oneRingNeighboorhood, triangleAngles, cotTriangleAngles);
    }
}

WKSEnergy::WKSEnergy(){

}

bool WKSEnergy::getWKS(Shape &shape, Eigen::MatrixXf& WKS, const int wksSize, const int wksVariance, int numEigenFunctions) {
    Eigen::SparseMatrix<double> L, M;
    igl::cotmatrix(shape.getV(), shape.getF(), L);
    L = (-L).eval();
    igl::massmatrix(shape.getV().cast<double>(), shape.getF(), igl::MASSMATRIX_TYPE_VORONOI, M);

    Eigen::MatrixXd Evecs;
    Eigen::VectorXd Evals;
    if (!igl::eigs(L, M, numEigenFunctions+1, igl::EIGS_TYPE_SM, Evecs, Evals)) {
        std::cout<< "Failed to compute wks energy with expected number of eigen functions." << std::endl;
        numEigenFunctions = 2;
        if (!igl::eigs(L, M, numEigenFunctions+1, igl::EIGS_TYPE_SM, Evecs, Evals)) {
            std::cout << "Exiting program because of WKS errors"<< std::endl;
            return false;
        }
    }

    // resort them, i know unefficient but i dont have time to make it nice
    Eigen::VectorXd EvalsTemp = Evals;
    Eigen::MatrixXd EvecsTemp = Evecs;
    for (int i = 0; i < Evals.size(); i++) {
        Evals(0) = EvalsTemp(Evals.size() - 1 - i);
        Evecs(Eigen::all, 0) = EvecsTemp(Eigen::all, Evals.size() - 1 - i);
    }


    Eigen::MatrixXd Dsafe = Evals;
    for (int i = Dsafe.rows()-1; i >= 0; i--) {
        Dsafe(i) = std::max(Dsafe(i), 1e-6);
        if (Dsafe(i) > 1e-6) {
            break;
        }
    }

    Eigen::MatrixXf logE = Dsafe.cast<float>().array().log().matrix();

    Eigen::MatrixXf e(wksSize, 1);
    e = igl::LinSpaced<Eigen::VectorXf>(wksSize, (float) logE(1), (float) logE.maxCoeff() / 1.02);
    const float sigma = (e(1) - e(0)) * wksVariance;
    const float twosigmasquared = 2 * sigma * sigma;

    Eigen::MatrixXf C(1, wksSize);
    for (int i = 0; i < wksSize; i++) {
        Eigen::ArrayXf temp = ( (-(e(i) - logE.array()).square()) / twosigmasquared ).exp();

        WKS(Eigen::all, i) = (Evecs.cast<float>().array().square() * temp.transpose().replicate(shape.getNumVertices(), 1)).rowwise().sum().matrix();
        C(i) = temp.sum();
    }

    for (int i = 0; i < shape.getNumVertices(); i++) {
        WKS(i, Eigen::all) = ( WKS(i, Eigen::all).array() / C.array()).matrix();
    }
    return true;
}


/* function get
    According to (1):
       E_{bend}(f) = sum_{i=1}^3 A_i^A * (H_A(i) - H_B(i))^2 +
                     sum_{i=1}^3 A_i^B * (H_B(i) - H_A(i))^2
    => E_{bend}(f) = sum_{i=1}^3 (A_i^A + A_i^B) * (H_A(i) - H_B(i))^2
*/
Eigen::MatrixXf WKSEnergy::get(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo) {

    const int wksSize = 2;
    const int wksVariance = 6;
    int numEigenFunctions = 2;

    // Init curvature and area vectors
    Eigen::MatrixXf WKSa(shapeA.getNumVertices(), wksSize);
    Eigen::MatrixXf WKSb(shapeB.getNumVertices(), wksSize);

    Eigen::VectorXf Aa(shapeA.getNumFaces());
    Eigen::VectorXf Ab(shapeB.getNumFaces());

    getA(shapeA, Aa);
    getA(shapeB, Ab);


    if (!getWKS(shapeA, WKSa, wksSize, wksVariance, numEigenFunctions))
        return Eigen::MatrixXf(FaCombo.rows(), 1).setZero();
    if (!getWKS(shapeB, WKSb, wksSize, wksVariance, numEigenFunctions))
        return Eigen::MatrixXf(FaCombo.rows(), 1).setZero();


    Eigen::ArrayXXf bendingEnergy(FaCombo.rows(), 3);
    bendingEnergy(Eigen::all, 0) = Aa(FaCombo(Eigen::all, 0)) + Ab(FbCombo(Eigen::all, 0));
    bendingEnergy(Eigen::all, 1) = Aa(FaCombo(Eigen::all, 1)) + Ab(FbCombo(Eigen::all, 1));
    bendingEnergy(Eigen::all, 2) = Aa(FaCombo(Eigen::all, 2)) + Ab(FbCombo(Eigen::all, 2));
    Eigen::ArrayXXf temp(FaCombo.rows(), 3);
    temp.setZero();
    for (int i = 0; i < wksSize; i++) {
        temp(Eigen::all, 0) = temp(Eigen::all, 0) + (WKSa(FaCombo(Eigen::all, 0), i) - WKSb(FbCombo(Eigen::all, 0), i)).array().square();
        temp(Eigen::all, 1) = temp(Eigen::all, 1) + (WKSa(FaCombo(Eigen::all, 1), i) - WKSb(FbCombo(Eigen::all, 1), i)).array().square();
        temp(Eigen::all, 2) = temp(Eigen::all, 2) + (WKSa(FaCombo(Eigen::all, 2), i) - WKSb(FbCombo(Eigen::all, 2), i)).array().square();
    }
    temp = temp * 1.0f/wksSize;
    bendingEnergy = bendingEnergy.cwiseProduct(temp.square());
    return bendingEnergy.matrix().rowwise().sum();
}
