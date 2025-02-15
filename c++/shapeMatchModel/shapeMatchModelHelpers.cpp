//
//  ShapeMatchModelHelpers.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 23.05.21.
//

#include "shapeMatchModel.hpp"
#include "primalHeuristic.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include <igl/find.h>
#include <igl/adjacency_list.h>
#include <min_marginal_utils.h>
#include "helper/utils.hpp"
#include <chrono>
#include <filesystem>
#include <algorithm>
#define DEBUG_PRUNING false


bool ShapeMatchModel::checkWatertightness() {
    bool sXwatertight = shapeX.isWatertight();
    bool sYwatertight = shapeY.isWatertight();

    if (opts.fillHolesOfShapes && (!sXwatertight || !sYwatertight)) {
        if (nonWatertightMeshHandler.fillHoles(shapeX, shapeY)) {
            // when we successfully closed the holes we have to reinitialize the members
            combos.init();
            constr.init();
            const int numProductFaces = combos.getFaCombo().rows();
            minMarginals = Eigen::MatrixXf(numProductFaces, 1);
        }
        sXwatertight = shapeX.isWatertight();
        sYwatertight = shapeY.isWatertight();
    }
    if (!sXwatertight) {
        std::cout << "[ShapeMM] Shape X is not watertight. Cannot create shapeMatchModel." << std::endl;
    }
    if (!sYwatertight) {
        std::cout << "[ShapeMM] Shape Y is not watertight. Cannot create shapeMatchModel." << std::endl;
    }
    assert(sXwatertight && sYwatertight);
    return sXwatertight && sYwatertight;
}

std::string ShapeMatchModel::getVariableName(int idx, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo) {
    // name the the optimization variables according to their matchings
    std::string name = " x_" +
        std::to_string(FaCombo(idx, 0)) + "_" + std::to_string(FaCombo(idx, 1)) + "_" + std::to_string(FaCombo(idx, 2)) + "__"
    + std::to_string(FbCombo(idx, 0)) + "_" + std::to_string(FbCombo(idx, 1)) + "_" + std::to_string(FbCombo(idx, 2));
    return name;
}

void ShapeMatchModel::saveAsLp(const std::string& filename) {
    Eigen::MatrixXf objective = getDeformationEnergy();
    SparseMatInt8 constrLHS = getConstraintsMatrix();
    SparseVecInt8 constrRHS = getConstraintsVector();
    Eigen::MatrixXi FaCombo = combos.getFaCombo();
    Eigen::MatrixXi FbCombo = combos.getFbCombo();
    
    std::ifstream temp;
    temp.open(filename.c_str());
    if (temp) {
        std::cout << "[ShapeMM] File already exists. Do you want to overwrite (y/n)?" << std::endl;
        char inp;
        std::cin >> inp;
        if (inp == 'y' || inp == 'Y') {
            std::cout << "[ShapeMM] Overwriting file" << std::endl;
            temp.close();
            std::remove(filename.c_str());
        } else if (inp == 'n' || inp == 'N') {
            std::cout << "[ShapeMM] Aborting." << std::endl;
            temp.close();
            return;
        }
        else {
            std::cout << "[ShapeMM] Invalid input. Aborting." << std::endl;
            temp.close();
            return;
        }
    }
    
    std::ofstream file;
    file.open(filename.c_str());
    
    
    // objective
    file << "Minimize\n";
    for (int i = 0; i < objective.rows()-1; i++) {
            file << std::to_string(objective(i)) << getVariableName(i, FaCombo, FbCombo) << " +\n";
    }
    file << std::to_string(objective(objective.rows()-1, 0)) << getVariableName(objective.rows()-1, FaCombo, FbCombo) << "\n";
    
    
    // constraint
    file << "Subject To\n";
    // Loop outer level => rows
    for (int k = 0; k < constrLHS.outerSize(); ++k) {
        file << "R" << std::to_string(k) << ": ";
        // loop inner level => cols
        bool first = true;
        int i = 0;
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
            if (it.value() > 0) {
                if (first) {
                    file << getVariableName(it.col(), FaCombo, FbCombo);
                } else {
                    file << " +" << getVariableName(it.col(), FaCombo, FbCombo);
                }
            } else {
                file << " -" << getVariableName(it.col(), FaCombo, FbCombo);
            }
            if (i % 3 == 0) {
                file << "\n";
            }
            i++;
            first = false;
        }
        file << " = " << std::to_string(constrRHS.coeff(k)) << "\n";
    }
    
    // optimization varible binary
    file << "Bounds\nBinaries\n";
    for (int i = 0; i < objective.rows(); i++) {
        file << getVariableName(i, FaCombo, FbCombo);
        if (i % 3 == 0) {
            file << "\n";
        }
    }
    file << "\nEnd\n";
    std::cout << "[ShapeMM] Successfully written ShapeMatchModel to LP file." << std::endl;
    file.close();
}


Combinations& ShapeMatchModel::getCombinations() {
    return combos;
}

Eigen::MatrixXf ShapeMatchModel::getDeformationEnergy() {
    return deformationEnergy.get();
}

SparseMatInt8 ShapeMatchModel::getConstraintsMatrix() {
    return constr.getConstraintMatrix();
}

SparseVecInt8 ShapeMatchModel::getConstraintsVector() {
    return constr.getConstraintVector();
}

void ShapeMatchModel::setMaxNumDualSolverCalls(const int numcalls) {
    opts.maxNumDualSolverCalls = numcalls;
}

void ShapeMatchModel::setMaxNumBacktracks(const int maxbacktracks) {
    opts.primalHeuristicOpts.autoSetMaxBacktracks = false;
    opts.primalHeuristicOpts.maxBacktracks = maxbacktracks;
}

void ShapeMatchModel::setMaxPrimalHeuristicIters(const int maxiters) {
    opts.primalHeuristicOpts.maxIter = maxiters;
}

/* function saveIlpAsLp
 
 */
void ShapeMatchModel::saveIlpAsLp(const std::string& filename) {
    std::ofstream file;
    file.open(filename.c_str());
    if (!ilpGenerated) {
        getIlpObj();
    }
    ilp.write_lp(file);
    std::cout << "[ShapeMM] Successfully written lp file" << std::endl;
}

/* function saveIlpAsLp

 */
void ShapeMatchModel::updateEnergy(const Eigen::MatrixXf& Vx2VyCost, bool weightWithAreas, bool useMemReg, float lambda) {
    deformationEnergy.useCustomDeformationEnergy(Vx2VyCost, weightWithAreas, useMemReg, lambda);
}

void ShapeMatchModel::updateEnergy(const Eigen::MatrixXf& Vx2VyCost, bool useElastic, float weightFeature) {
    if (useElastic) {
        deformationEnergy.addFeatureDifference(Vx2VyCost, weightFeature);
    }
    else {
        deformationEnergy.useCustomDeformationEnergy(Vx2VyCost, false, false, 1.0);
    }
}


void ShapeMatchModel::constantPenaliseDegenerate(float addval) {
    deformationEnergy.constantPenaliseDegenerate(addval);
}

LPMP::ILP_input ShapeMatchModel::getIlpObj() {
    if (ilpGenerated) {
        return ilp;
    }
    ilp = LPMP::ILP_input();
    Eigen::MatrixXf objective = getDeformationEnergy();
    Eigen::MatrixXi FaCombo = combos.getFaCombo();
    Eigen::MatrixXi FbCombo = combos.getFbCombo();
    SparseMatInt8 constrLHS = getConstraintsMatrix();
    SparseVecInt8 constrRHS = getConstraintsVector();
    
    // Add variables to ilp
    for (int i = 0; i < objective.rows(); i++) {
        std::string varName = getVariableName(i, FaCombo, FbCombo);
        ilp.add_new_variable(varName);
        ilp.add_to_objective((double) objective(i), i);
    }
    assert(ilp.nr_variables() == constrLHS.cols());
    
    const int numRowsDel = constrLHS.rows() - shapeX.getNumFaces() - shapeY.getNumFaces();

    int constraintGroup[] = {0, 0, 0};
    // Add constraints to ilp
    for (int k = 0; k < constrLHS.outerSize(); ++k) {
        if (k < numRowsDel) {
            unsigned int numNonZerosRow = 0;
            int8_t prevVal = 0;
            bool signflip = false;
            for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS, k); it; ++it) {
                const int f = it.index();
                const int8_t val = it.value();
                if (prevVal == 0) {
                    prevVal = val;
                }
                if (prevVal != val) {
                    signflip = true;
                }
            }
            if (!signflip)
                continue; // make sure we do not add trivial constraints
        }

        /*
            beginNewInequality   => creates new constraint
            inequalityIdentifier => name of the constraint e.g. R101
            inequalityType       => in our case always "="
        */
        
        constraintGroup[0] = ilp.begin_new_inequality();
        const std::string identifier = "R" + std::to_string(k) + " ";
        ilp.set_inequality_identifier(identifier);
        ilp.set_inequality_type(LPMP::ILP_input::inequality_type::equal);
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
                ilp.add_to_constraint(it.value(), it.index());
        }
        const int rhs = constrRHS.coeff(k);
        ilp.set_right_hand_side(rhs);
        
        /*
         each row of del can at most contain one positive and one negative value
         */
        if (k < numRowsDel && opts.useConstraintsGroups) {
            // del plus
            constraintGroup[1] = ilp.begin_new_inequality();
            const std::string identifierPlus = "R" + std::to_string(k) + "_plus" + " ";
            ilp.set_inequality_identifier(identifierPlus);
            ilp.set_inequality_type(LPMP::ILP_input::inequality_type::smaller_equal);
            for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
                if (it.value() == 1) {
                    ilp.add_to_constraint(it.value(), it.index());
                }
            }
            ilp.set_right_hand_side(1);
            
            // del minus
            constraintGroup[2] = ilp.begin_new_inequality();
            const std::string identifierMinus = "R" + std::to_string(k) + "_minus" + " ";
            ilp.set_inequality_identifier(identifierMinus);
            ilp.set_inequality_type(LPMP::ILP_input::inequality_type::smaller_equal);
            for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
                if (it.value() == -1) {
                    ilp.add_to_constraint(1, it.index());
                }
            }
            ilp.set_right_hand_side(1);
            
            ilp.add_constraint_group(std::begin(constraintGroup), std::end(constraintGroup));
        }
    }

    /*if (pruned) {
        for (int i = 0; i < objective.rows(); i++) {
            if (!pruneveci(i)){
                const std::string identifier = "Prune" + std::to_string(i) + " ";
                ilp.set_inequality_identifier(identifier);
                ilp.set_inequality_type(LPMP::ILP_input::inequality_type::equal);
                ilp.add_to_constraint(1, i);
                ilp.set_right_hand_side(0);
            }
        }
    }*/
    
    if (opts.useConstraintsGroups) {
        assert(ilp.nr_constraints() == (constrLHS.rows() + 2 * numRowsDel));
        assert(ilp.nr_constraint_groups() == numRowsDel);
    }
    else {
        assert(ilp.nr_constraints() == constrLHS.rows());
    }

    ilpGenerated = true;
    return ilp;
}

Eigen::VectorX<bool> ShapeMatchModel::getPruneVec(Eigen::MatrixXi& coarsep2pmap, Eigen::MatrixXi& IXf2c, Eigen::MatrixXi& IYf2c) {
    const Eigen::MatrixXi FXCombo = getCombinations().getFaCombo();
    const Eigen::MatrixXi FYCombo = getCombinations().getFbCombo();

    Eigen::MatrixX<bool> p2pMatFine = utils::computeP2PMat(shapeX, shapeY, coarsep2pmap, IXf2c, IYf2c, c2fNeighborhood);


    // actually creating the pruning vector
    Eigen::VectorX<bool> PruneVec(FXCombo.rows());
    for (long f = 0; f < FXCombo.rows(); f++) {
        const bool firstp2p = p2pMatFine( FXCombo(f, 0), FYCombo(f, 0) );
        const bool seconp2p = p2pMatFine( FXCombo(f, 1), FYCombo(f, 1) );
        const bool thirdp2p = p2pMatFine( FXCombo(f, 2), FYCombo(f, 2) );

        const bool p2p4 = p2pMatFine( FXCombo(f, 0), FYCombo(f, 1) );
        const bool p2p5 = p2pMatFine( FXCombo(f, 1), FYCombo(f, 2) );
        const bool p2p6 = p2pMatFine( FXCombo(f, 2), FYCombo(f, 0) );

        const bool p2p7 = p2pMatFine( FXCombo(f, 0), FYCombo(f, 2) );
        const bool p2p8 = p2pMatFine( FXCombo(f, 1), FYCombo(f, 0) );
        const bool p2p9 = p2pMatFine( FXCombo(f, 2), FYCombo(f, 1) );

        if (firstp2p || seconp2p || thirdp2p || p2p4 || p2p5 || p2p6 || p2p7 || p2p8 || p2p9) {
            PruneVec(f) = true;
        }
        else {
            PruneVec(f) = false;
        }
    }
    return PruneVec;
}

void ShapeMatchModel::pruneWithCoarserMatching(Eigen::MatrixXi& coarsep2pmap, Eigen::MatrixXi& IXf2c, Eigen::MatrixXi& IYf2c) {
    const Eigen::VectorX<bool> PruneVec = getPruneVec(coarsep2pmap, IXf2c, IYf2c);

    if (opts.verbose) std::cout << "[ShapeMM] Pruning Shape Match Model..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   > Product Space" << std::endl;
    combos.prune(PruneVec);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;
    if (opts.verbose) std::cout << "[ShapeMM]   > Constraints" << std::endl;
    constr.prune(PruneVec);

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "  [ms])" << std::endl;
    if (opts.verbose) std::cout << "[ShapeMM]   > Energies" << std::endl;
    deformationEnergy.prune(PruneVec);
    
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "  [ms])" << std::endl;
    if (opts.verbose) std::cout << "[ShapeMM] Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() << "  [ms])" << std::endl;

    opts.useConstraintsGroups = false;
    pruned = true;
}

Eigen::MatrixXf ShapeMatchModel::getMinMarginals() {
    Eigen::SparseMatrix<int8_t> Gamma(combos.getFaCombo().rows(), 1);
    return getMinMarginals(Gamma, true);
}

Eigen::MatrixXf ShapeMatchModel::getMinMarginals(Eigen::SparseMatrix<int8_t> &Gamma, bool gammaEmpty) {
    if (minMarginalsComputed && gammaEmpty) {
        return minMarginals;
    }
    // generate solver with options struct
    LPMP::bdd_solver_options localBddOpts = opts.bddSolverOpts;
    if (bddsolver  == NULL) {
        // use a pointer for bddsolver so we can access its state of the previous iteration
        localBddOpts.ilp = getIlpObj();
        bddsolver = new LPMP::bdd_solver(localBddOpts);
    }

    if (!gammaEmpty) {
        for (typename SparseMatInt8Cm::InnerIterator it(Gamma, 0); it; ++it) {
            if (it.value() == 1) {
                bddsolver->fix_variable(it.index(), 1);
            }
            if (it.value() == 0) {
                bddsolver->fix_variable(it.index(), 0);
            }
        }
    }

    // solve and retrieve dual costs
    bddsolver->solve();
    if (initialLowerBound == -1)
        initialLowerBound = bddsolver->lower_bound();
    std::vector<double> minMargVec;

    const double eps = 1e-7;
    minMargVec = LPMP::min_marginal_differences( bddsolver->min_marginals(), eps);
    assert(minMargVec.size() == minMarginals.rows());
    if (minMargVec.size() != minMarginals.rows()) {
        std::cout << "[ShapeMM] BDD Solver returned not enough min marginals. Aborting Programm" << std::endl;
        exit(-1);
    }

    minMarginals = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(minMargVec.data(), minMargVec.size()).cast<float>();
    minMarginalsComputed = true;
    return minMarginals;
}

void ShapeMatchModel::printSolutionInfo(SparseMatInt8 Gamma) {
    std::cout << "[ShapeMM] Solution contains >>> " << Gamma.nonZeros() << " <<< matchings" << std::endl;
    SparseMatInt8 constrLHS = getConstraintsMatrix();
    SparseVecInt8 constrRHS = getConstraintsVector();
    if ((constrLHS * Gamma - constrRHS.transpose()).cwiseAbs().sum() == 0) {
        std::cout << "[ShapeMM] Solution fullfills constraints" << std::endl;
    }
    else {
        std::cout << "[ShapeMM] Solution does NOT fullfill constraints" << std::endl;
    }
    std::cout << "[ShapeMM] Solution Objective: " << deformationEnergy.get().transpose() * Gamma.cast<float>() << std::endl;
    
}

void ShapeMatchModel::writeSolutionToFile(MatrixInt8 &Gamma) {
    std::filesystem::create_directories(opts.modelName);
    std::cout << "[ShapeMM] Writing Solution to file..." << std::endl;
    utils::writeMatrixToFile(Gamma, opts.modelName + "_Gamma");
    std::cout << "[ShapeMM] Done" << std::endl;
    writeModelToFile();
}

void ShapeMatchModel::writeModelForMatlab(std::string filename) {
    utils::writeMatrixToFile(getDeformationEnergy(), filename + "_Energy");

    const SparseMatInt8 constrLHS = getConstraintsMatrix();
    SparseVecInt8 constrRHS = getConstraintsVector();
    const int nnz = constrLHS.nonZeros();
    Eigen::MatrixXi I(nnz, 1);
    Eigen::MatrixXi J(nnz, 1);
    Eigen::MatrixX<int8_t> V(nnz, 1);

    long c = 0;
    for (int k = 0; k < constrLHS.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS, k); it; ++it) {
            I(c) = k;
            J(c) = it.index();
            V(c) = it.value();
            c++;
        }
    }

    utils::writeMatrixToFile(I, filename + "_I");
    utils::writeMatrixToFile(J, filename + "_J");
    utils::writeMatrixToFile(V, filename + "_V");

    Eigen::MatrixXi dimension(1, 1);
    dimension << constrRHS.nonZeros();
    utils::writeMatrixToFile(dimension, filename + "_b");
}

Eigen::MatrixXf ShapeMatchModel::getEforPython() {
    return getDeformationEnergy();
}

Eigen::MatrixX<int8_t> ShapeMatchModel::getRHSforPython() {
    return getConstraintsVector();
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixX<int8_t>> ShapeMatchModel::getAforPython() {
    const SparseMatInt8 constrLHS = getConstraintsMatrix();
    const int nnz = constrLHS.nonZeros();
    Eigen::MatrixXi I(nnz, 1);
    Eigen::MatrixXi J(nnz, 1);
    Eigen::MatrixX<int8_t> V(nnz, 1);

    long c = 0;
    for (int k = 0; k < constrLHS.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS, k); it; ++it) {
            I(c) = k;
            J(c) = it.index();
            V(c) = it.value();
            c++;
        }
    }
    return std::make_tuple(I, J, V);
}

void ShapeMatchModel::writeModelToFile() {
    std::filesystem::create_directories(opts.modelName);
    std::cout << "[ShapeMM] Writing Model to file..." << std::endl;
    utils::writeMatrixToFile(getDeformationEnergy(), opts.modelName + "_Energy");
    shapeX.writeToFile(opts.modelName + "_X.ply");
    shapeY.writeToFile(opts.modelName + "_Y.ply");
    nonWatertightMeshHandler.writeToFile(opts.modelName);
    std::cout << "[ShapeMM] Done" << std::endl;
}

MatrixInt8 ShapeMatchModel::readSolutionFromFile(std::string filename) {
    return utils::readMatrixFromFile<int8_t>(filename);
}

MatrixInt8 ShapeMatchModel::readSolutionFromFile() {
    return readSolutionFromFile(opts.modelName + "_Gamma");
}

Eigen::MatrixXi ShapeMatchModel::getPointMatchesFromSolution(const SparseVecInt8 &Gamma) {
    const Eigen::MatrixXi FXCombo = getCombinations().getFaCombo();
    const Eigen::MatrixXi FYCombo = getCombinations().getFbCombo();
    std::set<std::tuple<int, int>> matchings;

    // extract all matchings
    for (typename Eigen::SparseVector<int8_t, Eigen::RowMajor>::InnerIterator it(Gamma); it; ++it) {
        const int idx = it.index();

        Eigen::MatrixXi faceX = FXCombo.row(idx);
        Eigen::MatrixXi faceY = FYCombo.row(idx);

        for (int i = 0; i < 3; i++) {
            // std::set keeps track if we already added a matching
            matchings.insert(std::make_tuple(faceX(i), faceY(i)));
        }
    }

    Eigen::MatrixXi matchingMatrix(matchings.size(), 2);
    int idx = 0;
    for (auto matching : matchings) {
        matchingMatrix(idx, 0) = std::get<0>(matching);
        matchingMatrix(idx, 1) = std::get<1>(matching);
        idx++;
    }
    return matchingMatrix;
}

float ShapeMatchModel::getFinalEnergy(const SparseVecInt8 &Gamma) {
    const Eigen::VectorXf cost = deformationEnergy.get().col(0);
    const Eigen::VectorXf result(Gamma.cast<float>());
    return cost.transpose() * result;
}

Eigen::MatrixXi& ShapeMatchModel::getFaCombo() {
    return combos.getFaCombo();
}
Eigen::MatrixXi& ShapeMatchModel::getFbCombo() {
    return combos.getFbCombo();
}

Eigen::MatrixXi ShapeMatchModel::getEdgeTriangleAdjacencyX() {
    return shapeX.getLocEinF();
}

Eigen::MatrixXi ShapeMatchModel::getEdgeTriangleAdjacencyY() {
    return shapeY.getLocEinF();
}

Eigen::MatrixXi ShapeMatchModel::getEdgesX(){
    return shapeX.getE();
}

Eigen::MatrixXi ShapeMatchModel::getEdgesY(){
    return shapeY.getE();
}

float ShapeMatchModel::getLowerBound() {
    return initialLowerBound;
}

bool ShapeMatchModel::constraintsFullfilled() const {
    return constraintsFulfilled;
}
