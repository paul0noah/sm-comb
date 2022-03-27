//
//  testEnergies.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 15.04.21.
//

#ifndef testEnergies_hpp
#define testEnergies_hpp

#include <Eigen/Dense>	
#include <string.h>

namespace test {
void generateExamples(const std::string path);

Eigen::MatrixXf getDeformationEnergyComputedWithMatlab();
void deformationEnergy();
void membraneEnergy();
void bendingEnergy();
void constraints();
void saveLp();
void saveIlpAsLp();
void primalHeuristic();
void plot();
void solveDualProblem();

} // namespace test

#endif /* testEnergies_hpp */
