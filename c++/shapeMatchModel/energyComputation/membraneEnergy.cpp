//
//  membraneEnergy.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#include "membraneEnergy.hpp"
#include "helper/utils.hpp"

/* function getG
 
  G = inv(gA) * gB = [G1 G2; G3 G4]
  for vectorized computation we output
  G = [G1 G2 G3 G4]
  where each GX is a colum vector
 
  G = inv(gA) * gB
  since we don't want to invert we use the following
  A = [a b; c d]
  inv(A) = 1/(ad - bc) * [d -b; -c a]
  consequently
  gA = [a1 a2; a3 a4];
  gB = [b1 b2; b3 b4];
  => inv(gA) * gB = 1 / (a1*a4 - a2*a3) * [a4 -a2; -a3 a1] * [b1 b2; b3 b4]
                  = 1 / (a1*a4 - a2*a3) * [a4*b1 + -a2*b3, a4*b2 + -a2*b4]
                                          [-a3*b1 + a1*b3, -a3*b2 + a1*b4]
                  = [G1 G2; G3 G4]
 */
Eigen::ArrayXXf MembraneEnergy::getG(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo) {
    
    Eigen::ArrayXXf edgesA12(FaCombo.rows(), 6); edgesA12 = getEdges12(shapeA, FaCombo);
    Eigen::ArrayXXf edgesB12(FbCombo.rows(), 6); edgesB12= getEdges12(shapeB, FbCombo);
    
    Eigen::ArrayXXf gA(edgesA12.rows(), 4); gA = getg(edgesA12);
    Eigen::ArrayXXf gB(edgesB12.rows(), 4); gB = getg(edgesB12);
    
    Eigen::ArrayXf detga =  gA(Eigen::all, 0).cwiseProduct( gA(Eigen::all, 3) ) - gA(Eigen::all, 1).cwiseProduct( gA(Eigen::all, 2) );
    
    detga = detga.inverse();

    Eigen:: ArrayXXf G(FaCombo.rows(), 4);
    G(Eigen::all, 0) =  ( gA(Eigen::all, 3).cwiseProduct(gB(Eigen::all, 0))
                         - gA(Eigen::all, 1).cwiseProduct(gB(Eigen::all, 2))
                        ).cwiseProduct(detga);
    G(Eigen::all, 1) =  ( gA(Eigen::all, 3).cwiseProduct(gB(Eigen::all, 1))
                         - gA(Eigen::all, 1).cwiseProduct(gB(Eigen::all, 3))
                        ).cwiseProduct(detga);
    G(Eigen::all, 2) = ( -gA(Eigen::all, 2).cwiseProduct(gB(Eigen::all, 0))
                        + gA(Eigen::all, 0).cwiseProduct(gB(Eigen::all, 2))
                       ).cwiseProduct(detga);
    G(Eigen::all, 3) = ( -gA(Eigen::all, 2).cwiseProduct(gB(Eigen::all, 1))
                        + gA(Eigen::all, 0).cwiseProduct(gB(Eigen::all, 3))
                       ).cwiseProduct(detga);
    return G;
}

/* function w = getW(G, mu, lam)
 According to (1)
  W(A) = mu / 2 * tr(A)+ lambda / 4 * det(A) - (mu / 2 + lambda / 4) *
            * log(det(A)) - (mu + lambda / 4)
  A actually is
  A = [A0 A1; A2 A3]
  but due to vectorization we input
  A = [A0 A1 A2 A3]
 */
Eigen::ArrayXf MembraneEnergy::getW(Eigen::ArrayXXf &A, float mu, float lambda) {
    // tr(A) = A0 + A3
    Eigen::ArrayXf trA(A.rows(), 1);
    trA = A(Eigen::all, 0) + A(Eigen::all, 3);
    
    // det(A) = A0 * A3 - A2 * A1
    Eigen::ArrayXf detA = A(Eigen::all, 0).cwiseProduct(A(Eigen::all, 3)) - A(Eigen::all, 2).cwiseProduct(A(Eigen::all, 1));

    return mu/2 * trA + lambda/4 * detA - (mu/2 + lambda/4) * utils::arraySafeLog(detA) - (mu + lambda/4);
}


/* function getEdges12()
 returns 2 edges of each triangle in the following form:
 edges12 = [[edge1_0 edge2_0]
            [edge1_1 edge2_1]
            ...]
 */
Eigen::ArrayXXf MembraneEnergy::getEdges12(Shape &shape, Eigen::MatrixXi &FCombo) {
    Eigen::ArrayXXf edges12(FCombo.rows(), 6);
    for (int i = 0; i < FCombo.rows(); i++) {
        edges12.block(i, 0, 1, 3) = (shape.getVi(FCombo(i, 1)) - shape.getVi(FCombo(i, 0))).array();
        edges12.block(i, 3, 1, 3) = (shape.getVi(FCombo(i, 2)) - shape.getVi(FCombo(i, 1))).array();
    }
    return edges12;
}


/* function getg()
 According to (1):
 g is actually a matrix
 g = [g0 g1; g2 g3];
 but we make a vector out of it
 => g = [g0 g1 g2 g3];
 according to the paper
 g = [e1; -e2]' * [e1; -e2];
 => g0 = e1' * e1;
    g1 = e1' * e2
    g2 = g1
    g3 = e2' * e2
 
 Assumes edges12 in the following format:
 edges12 = [edge1_0, edge2_0]
           [edge1_1, edge2_1]
           ...
 where edge1_X and edge2_X in IR^{1x3}
 */
Eigen::ArrayXXf MembraneEnergy::getg(Eigen::ArrayXXf &edges12) {
    Eigen::ArrayXXf g(edges12.rows(), 4);
    g(Eigen::all, 0) = edges12.block(0, 0, edges12.rows(), 3).square().rowwise().sum();
    g(Eigen::all, 3) = edges12.block(0, 3, edges12.rows(), 3).square().rowwise().sum();
    g(Eigen::all, 1) = (edges12.block(0, 0, edges12.rows(), 3).cwiseProduct(- edges12.block(0, 3, edges12.rows(), 3))).rowwise().sum();
    g(Eigen::all, 2) = g(Eigen::all, 1);
    return g;
}


MembraneEnergy::MembraneEnergy(){
    
}


/* function get
    computes membrane energy according to (1):
        Membrane Energy between Triangle A and B
        W_mem(A, B) = A_a * W(G_t)
        where
        G_t = g_at^-1 * g_bt;
 
               -
              / \
         eX1 /   \ eX1
            /  X  \
           /       \
          /_________\
             eX0
        g_t = [eX1; -eX2]' * [eX1; -eX2];
        => no parallel edges in triangle A!!!! (no degenerate cases in A)
        (so g_at is invertible)
 
        W(A) = mu / 2 * tr(A)+ lambda / 4 * det(A) - (mu / 2 + lambda / 4) *
            * log(det(A)) - (mu + lambda / 4)
 
 */
Eigen::MatrixXf MembraneEnergy::get(Shape &shapeA, Shape &shapeB, Eigen::MatrixXi FaCombo, Eigen::MatrixXi FbCombo) {
    
    // these are some material properties, maybe something we could tune
    float mu = 1;
    float lambda = 1;

    Eigen::ArrayXf areasA(FaCombo.rows(), 1);
    areasA = shapeA.getTriangleAreas(FaCombo);
    
    Eigen::ArrayXXf G(FaCombo.rows(), 4);
    G = getG(shapeA, shapeB, FaCombo, FbCombo);
    
    Eigen::MatrixXf membraneEnergy(FaCombo.rows(), 3);
    membraneEnergy = areasA.cwiseProduct(getW(G, mu, lambda)).matrix();
    
    return membraneEnergy;
}

/* REFERENCES
  (1) EZUZ, Danielle, et al. Elastic correspondence between triangle meshes.
      In: Computer Graphics Forum. 2019. S. 121-134.
 */
