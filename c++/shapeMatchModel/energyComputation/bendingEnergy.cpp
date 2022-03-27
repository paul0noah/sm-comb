//  bendingEnergy.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#include <math.h>
#include <iostream>
#include "bendingEnergy.hpp"
#include "helper/utils.hpp"


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
float BendingEnergy::getVoronoiArea(int i, Shape &shape, int neighboor, Eigen::MatrixXf &cotTriangleAngles) {
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
float BendingEnergy::getMixedArea(int i, Shape &shape, Eigen::VectorXi &oneRingNeighboorhood, Eigen::MatrixXf &triangleAngles, Eigen::MatrixXf &cotTriangleAngles) {

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

/* function getSumOverOneRingNeighborhood
         i__________beta_ij
         /\        /
        /  \  B   /
       / A  \    /
      /      \  /
     /________\/
     alpha_ij   j
 
    i is the vertex for which we computed the oneRingNeighboorhood
    
     According to (2)
     sumOverOneRingNeighborhood =
           sum_{j in Neighboordhood} (cot(alpha_ij) + cot(beta_ij)) * (x_i - x_j)
 */
Eigen::MatrixXf BendingEnergy::getSumOverOneRingNeighborhood(int i, Shape &shape, Eigen::VectorXi &oneRingNeighboorhood, Eigen::MatrixXf &cotTriangleAngles) {
    
    Eigen::MatrixXi F(shape.getNumFaces(), 3); F = shape.getF();
    Eigen::Array3f sum; sum.setZero();
    Eigen::Vector3i triangleA, triangleB;

    for (int k = 0; k < oneRingNeighboorhood.rows(); k++) {
        triangleA = shape.getFi(oneRingNeighboorhood(k)).transpose();
        // find i and j in triangle A
        int idxiInTriangleA;
        for (int ii = 0; ii < 3; ii++) {
            if (triangleA(ii) == i) {
                idxiInTriangleA = ii;
            }
        }
        
        int idxjInTriangleA = idxiInTriangleA - 1;
        idxjInTriangleA = idxjInTriangleA == -1 ? 2 : idxjInTriangleA;
        int j = triangleA(idxjInTriangleA);
        // find adjacent triangle B (must contain j)
        int idxjInTriangleB, idxiInTriangleB, idxTriangleB;
        for (int kk = 0; kk < oneRingNeighboorhood.rows(); kk++) {
            if (kk != k) {
                bool adjacent = false;
                triangleB = shape.getFi(oneRingNeighboorhood(kk)).transpose();
                
                for (int ii = 0; ii < 3; ii++) {
                    // we already know that each in oneRingNeighboordhood contains i
                    if (triangleB(ii) == j) {
                        idxjInTriangleB = ii;
                        idxiInTriangleB = idxjInTriangleB - 1;
                        idxiInTriangleB = idxiInTriangleB == -1 ? 2 : idxiInTriangleB;
                        idxTriangleB = oneRingNeighboorhood(kk);
                        adjacent = true;
                    }
                }
                if (adjacent) {
                    break;
                }
            }
        }

        // compute sum
        int idxCotAlpha_ij = idxiInTriangleA + 1;
        idxCotAlpha_ij = idxCotAlpha_ij == 3 ? 0 : idxCotAlpha_ij;
        float cotAlpha_ij = cotTriangleAngles(oneRingNeighboorhood(k), idxCotAlpha_ij);
        int idxCotBeta_ij = idxjInTriangleB + 1;
        idxCotBeta_ij = idxCotBeta_ij == 3 ? 0 : idxCotBeta_ij;
        float cotBeta_ij = cotTriangleAngles(idxTriangleB, idxCotBeta_ij);
        
        sum = sum + (( cotAlpha_ij + cotBeta_ij) * (shape.getVi(i) - shape.getVi(j)).array().transpose()) ;

    }
    return sum.matrix();
}

/* function getMeanCurvatures
    computes the mean curvatures for each vertex as proposed in (2)
 */
void BendingEnergy::getMeanCurvatures(Shape &shape, Eigen::VectorXf &curvatures, Eigen::VectorXf &A) {
    
    Eigen::MatrixXf triangleAngles(shape.getNumFaces(), 3);
    for (int i = 0; i < shape.getNumFaces(); i++) {
        triangleAngles.row(i) = shape.getTriangleAngles(i);
    }
    
    Eigen::MatrixXf cotTriangleAngles(shape.getNumFaces(), 3);
    // cotan(x) = 1 / tan(x)
    cotTriangleAngles = triangleAngles.array().tan().inverse().matrix();
    
    Eigen::VectorXi oneRingNeighboorhood;
    Eigen::MatrixXf sumOverOneRingNeighborhood;
    for (int i = 0; i < curvatures.rows(); i++) {
        oneRingNeighboorhood = shape.getTrianglesAttachedToVertex(i);
        A(i) = getMixedArea(i, shape, oneRingNeighboorhood, triangleAngles, cotTriangleAngles);
        sumOverOneRingNeighborhood = getSumOverOneRingNeighborhood(i, shape, oneRingNeighboorhood, cotTriangleAngles);

        curvatures(i) = 0.25 / A(i) * sumOverOneRingNeighborhood.norm();
    }
}

BendingEnergy::BendingEnergy(){
    
}

/* function get
    According to (1):
       E_{bend}(f) = sum_{i=1}^3 A_i^A * (H_A(i) - H_B(i))^2 +
                     sum_{i=1}^3 A_i^B * (H_B(i) - H_A(i))^2
    => E_{bend}(f) = sum_{i=1}^3 (A_i^A + A_i^B) * (H_A(i) - H_B(i))^2
*/
Eigen::MatrixXf BendingEnergy::get(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo) {
    
    // Init curvature and area vectors
    Eigen::VectorXf Ca(shapeA.getNumVertices());
    Eigen::VectorXf Cb(shapeB.getNumVertices());
    
    Eigen::VectorXf Aa(shapeA.getNumFaces());
    Eigen::VectorXf Ab(shapeB.getNumFaces());
    
    getMeanCurvatures(shapeA, Ca, Aa);
    getMeanCurvatures(shapeB, Cb, Ab);
    
    Eigen::ArrayXXf bendingEnergy(FaCombo.rows(), 3);
    bendingEnergy(Eigen::all, 0) = Aa(FaCombo(Eigen::all, 0)) + Ab(FbCombo(Eigen::all, 0));
    bendingEnergy(Eigen::all, 1) = Aa(FaCombo(Eigen::all, 1)) + Ab(FbCombo(Eigen::all, 1));
    bendingEnergy(Eigen::all, 2) = Aa(FaCombo(Eigen::all, 2)) + Ab(FbCombo(Eigen::all, 2));
    Eigen::ArrayXXf temp(FaCombo.rows(), 3);
    temp(Eigen::all, 0) = Ca(FaCombo(Eigen::all, 0)) - Cb(FbCombo(Eigen::all, 0));
    temp(Eigen::all, 1) = Ca(FaCombo(Eigen::all, 1)) - Cb(FbCombo(Eigen::all, 1));
    temp(Eigen::all, 2) = Ca(FaCombo(Eigen::all, 2)) - Cb(FbCombo(Eigen::all, 2));
    bendingEnergy = bendingEnergy.cwiseProduct(temp.square());
    return bendingEnergy.matrix().rowwise().sum();
}
 
/* SOURCES:
 (1) WINDHEUSER, Thomas, et al.
 Large‐scale integer linear programming for orientation preserving 3d
 shape matching.
 In: Computer Graphics Forum. Oxford, UK: Blackwell Publishing Ltd,
 2011. S. 1471-1480.
 (2) MEYER, Mark, et al.
 Discrete differential-geometry operators for triangulated 2-manifolds.
 In: Visualization and mathematics III. Springer, Berlin, Heidelberg, 2003. S. 35-57.
 */
