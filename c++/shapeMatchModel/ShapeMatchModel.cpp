//
//  ShapeMatchModel.cpp
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
#include <min_marginal_utils.h>
#include "helper/utils.hpp"
#include <chrono>
#include <filesystem>


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


void ShapeMatchModel::generate() {
    if (!checkWatertightness()) return;
    if (opts.verbose) std::cout << "[ShapeMM] Generating Shape Match Model..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   > Product Space" << std::endl;
    combos.getFaCombo();
    
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;
    if (opts.verbose) std::cout << "[ShapeMM]   > Constraints" << std::endl;
    constr.getConstraintMatrix();
    
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "  [ms])" << std::endl;
    if (opts.verbose) std::cout << "[ShapeMM]   > Energies" << std::endl;
    deformationEnergy.get();
    if (opts.assignZeroEnergyToFilledHoles) {
        if (opts.verbose) std::cout << "[ShapeMM]     > Updating Energy Values of filled holes" << std::endl;
        nonWatertightMeshHandler.modifyEnergy(deformationEnergy, constr);
    }
    
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "  [ms])" << std::endl;
    if (opts.verbose) std::cout << "[ShapeMM] Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() << "  [ms])" << std::endl;
}


ShapeMatchModel::ShapeMatchModel(std::string modelname) :
    shapeX(modelname + "_X.ply"),
    shapeY(modelname + "_Y.ply"),
    combos(shapeX, shapeY),
    constr(shapeX, shapeY, combos),
    deformationEnergy(shapeX, shapeY, combos),
    ilpGenerated(false),
    minMarginals(combos.getFaCombo().rows(), 1),
    minMarginalsComputed(false),
    bddsolver(NULL),
    nonWatertightMeshHandler(modelname) {
    ilp = LPMP::ILP_input();
    generate();
    opts.modelName = modelname;
}

ShapeMatchModel::ShapeMatchModel(std::string modelname, ShapeMatchModelOpts optsIn) :
ShapeMatchModel(modelname) {
    opts = optsIn;
}


ShapeMatchModel::ShapeMatchModel(Shape &sX, Shape &sY) :
    shapeX(sX),
    shapeY(sY),
    combos(shapeX, shapeY),
    constr(shapeX, shapeY, combos),
    deformationEnergy(shapeX, shapeY, combos),
    ilpGenerated(false),
    minMarginals(combos.getFaCombo().rows(), 1),
    minMarginalsComputed(false),
    bddsolver(NULL) {
    ilp = LPMP::ILP_input();
    generate();
}

ShapeMatchModel::ShapeMatchModel(Shape &sX, Shape & sY, ShapeMatchModelOpts optsIn):
    ShapeMatchModel(sX, sY) {
        opts = optsIn;
}

ShapeMatchModel::ShapeMatchModel(std::string filenameShapeX, std::string filenameShapeY):
    shapeX(filenameShapeX),
    shapeY(filenameShapeY),
    combos(shapeX, shapeY),
    constr(shapeX, shapeY, combos),
    deformationEnergy(shapeX, shapeY, combos),
    ilpGenerated(false),
    minMarginals(combos.getFaCombo().rows(), 1),
    minMarginalsComputed(false) {
    ilp = LPMP::ILP_input();
    generate();
}

ShapeMatchModel::ShapeMatchModel(std::string filenameShapeX, std::string filenameShapeY, ShapeMatchModelOpts opts):
    ShapeMatchModel(filenameShapeX, filenameShapeY) {
        opts = opts;
}

ShapeMatchModel::~ShapeMatchModel() {
    // make sure we delete the object under the bddsolver pointer which we created with "new"
    if (bddsolver != NULL)
        delete bddsolver;
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
        /*
            beginNewInequality   => creates new constraint
            inequalityIdentifier => name of the constraint e.g. R101
            inequalityType       => in our case always "="
        */
        
        constraintGroup[0] = ilp.begin_new_inequality();
        const std::string identifier = "R" + std::to_string(k) + ": ";
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
            const std::string identifierPlus = "R" + std::to_string(k) + "_plus" + ": ";
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
            const std::string identifierMinus = "R" + std::to_string(k) + "_minus" + ": ";
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

/*
Eigen::MatrixXf ShapeMatchModel::getMinMarginalsWithNonEmptyGamma(Eigen::SparseMatrix<int8_t> &Gamma) {

    LPMP::ILP_input reducedIlp = LPMP::ILP_input();
    Eigen::MatrixXf objective = getDeformationEnergy();
    Eigen::MatrixXi FaCombo = combos.getFaCombo();
    Eigen::MatrixXi FbCombo = combos.getFbCombo();
    SparseMatInt8 constrLHS = getConstraintsMatrix();
    SparseVecInt8 constrRHS = getConstraintsVector();

    // extract undecided variables
    std::vector<int> idxVarUndecided;
    Eigen::MatrixXi varIdxInUndecidedMap(Gamma.rows(), 1);
    varIdxInUndecidedMap = - varIdxInUndecidedMap.setOnes();
    idxVarUndecided.reserve(Gamma.nonZeros());
    int mapIdx = 0;
    for (typename SparseMatInt8Cm::InnerIterator it(Gamma, 0); it; ++it) {
        if (it.value() != 1) {
            idxVarUndecided.push_back(it.index());
            varIdxInUndecidedMap(it.index()) = mapIdx;
            mapIdx++;
        }
    }

    // Add undecided variables to reducedIlp
    for (std::vector<int>::iterator it = idxVarUndecided.begin() ; it != idxVarUndecided.end(); ++it) {
        const int idx = *it;
        std::string varName = getVariableName(idx, FaCombo, FbCombo);
        reducedIlp.add_new_variable(varName);
        reducedIlp.add_to_objective((double) objective(idx), varIdxInUndecidedMap(idx));
    }

    // Add constraints to ilp
    for (int k = 0; k < constrLHS.outerSize(); k++) {
        bool reducedVariableFound = false;
        int numReducedVariablesFound = 0;
        int varIdx;
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
            const int currentVarIdx = it.index();
            if (Gamma.coeff(currentVarIdx, 0) != 1 && Gamma.coeff(currentVarIdx, 0) != 0) {
                numReducedVariablesFound++;
                varIdx = currentVarIdx;
            }
            if (numReducedVariablesFound >= 2) {
                reducedVariableFound = true;
                break;
            }
        }
        if (numReducedVariablesFound == 1) {
            if (opts.verbose) std::cout << "[ShapeMM] ERROR: Found value in reduced lp which needs to be set to one or zero: " << varIdx << std::endl;
        }
        if (reducedVariableFound) {
            reducedIlp.begin_new_inequality();
            const std::string identifier = "R" + std::to_string(k) + ": ";
            reducedIlp.set_inequality_identifier(identifier);
            reducedIlp.set_inequality_type(LPMP::ILP_input::inequality_type::equal);
            int rhs = constrRHS.coeff(k);
            for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
                const int currentVarIdx = it.index();
                if (Gamma.coeff(currentVarIdx, 0) == 1) {
                    rhs -= it.value();
                }
                else if (Gamma.coeff(currentVarIdx, 0) != 0) {
                    reducedIlp.add_to_constraint(it.value(), varIdxInUndecidedMap(currentVarIdx));
                    if (opts.debugOutputMinMarginalsNonEmptyGamma) std::cout << (int) it.value() << " * (" << currentVarIdx << ") ";
                }
            }
            if (rhs != -1 && rhs != 0 && rhs != 1) {
                if (opts.verbose) std::cout << "[ShapeMM] ERROR: RHS of reduced lp not as expected" << std::endl;
            }
            reducedIlp.set_right_hand_side(rhs);


            if (opts.debugOutputMinMarginalsNonEmptyGamma) std::cout << " = " << (int) rhs << std::endl;
        }

    }

    LPMP::bdd_solver_options localBddOpts = opts.bddSolverOpts;
    localBddOpts.ilp = reducedIlp;
    LPMP::bdd_solver solver(localBddOpts);

    // solve and retrieve dual costs
    solver.solve();
    std::vector<double> minMargVec;

    const double eps = 1e-7;
    minMargVec = LPMP::min_marginal_differences( solver.min_marginals(), eps);

    Eigen::MatrixXf minMarginals(Gamma.rows(), 1);
    minMarginals.setZero();
    int i = 0;
    for (std::vector<int>::iterator it = idxVarUndecided.begin() ; it != idxVarUndecided.end(); ++it) {
        const int idx = *it;
        minMarginals(idx) = (float) minMargVec[i];
        i++;
    }
    return minMarginals;
}
 */

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

int numVarCandidates(Eigen::SparseMatrix<int8_t> &Gamma) {
    int numCandidates = 0;
    for (typename SparseMatInt8Cm::InnerIterator it(Gamma, 0); it; ++it) {
        if (it.value() != 1) {
            numCandidates++;
        }
    }
    return numCandidates;
}

MatrixInt8 ShapeMatchModel::solve() {

    bool gammaEmpty = true;
    MatrixInt8 Gamma(combos.getFaCombo().rows(), 1);

    if (!checkWatertightness()) {
        std::cout << "[ShapeMM] Matching of non-watertight meshes is not possible with this approach" << std::endl;
        return MatrixInt8();
    }

    if(!opts.useMinMarginals) {
        opts.primalHeuristicOpts.useMinMarginals = false;
    }

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    
    Eigen::MatrixXf costs(Gamma.rows(), 1);
    costs = deformationEnergy.get();
    Eigen::MatrixXf minMarginals(Gamma.rows(), 1);
    minMarginals.setZero();
    
    int solIters = 0;
    
    PrimalHeuristicOpts heuristicOpts = opts.primalHeuristicOpts;
    // propagate options to primalHeuristic
    heuristicOpts.useMinMarginals = opts.useMinMarginals;
    heuristicOpts.verbose = opts.verbose;
    PrimalHeuristic heuristic(shapeX, shapeY, combos, costs, minMarginals, constr, heuristicOpts);

    if (opts.initGammaBeforeComputingMinMarginals && std::min(shapeX.getNumFaces(), shapeY.getNumFaces()) > 175) {
        gammaEmpty = heuristic.initializeGamma(Gamma, nonWatertightMeshHandler);
    }
    PRIMAL_HEURISTIC_RETURN_FLAG heuristicStatus;

    std::chrono::duration<float> timeBDD = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::seconds(0));
    std::chrono::duration<float> timeHeuristic = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::seconds(0));
    while (solIters < opts.maxNumDualSolverCalls) {

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        if (opts.useMinMarginals) {

            std::cout << "[ShapeMM] Solving Dual:" << std::endl;

            Eigen::SparseMatrix<int8_t> GammaSparse = Gamma.sparseView();

            if ( numVarCandidates(GammaSparse) > 0 || gammaEmpty || solIters == 0) {
                minMarginals = getMinMarginals(GammaSparse, gammaEmpty);
            }
            else {
                printSolutionInfo(Gamma.sparseView());
                return Gamma;
            }
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        timeBDD += std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);

        /*  this function does exactly nothing because we are working with a reference to minMarginals
            -> i'll leave it here nevertheless to improve understandability
            heuristic.updateMinMarginals(minMarginals);
         */
        heuristicStatus = heuristic.apply(Gamma, gammaEmpty);
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        timeHeuristic += std::chrono::duration_cast<std::chrono::duration<float>>(t3 - t2);
        if (heuristicStatus == SOL_COMPLETE || heuristicStatus == SOL_INFEASIBLE || !opts.useMinMarginals) {
            break;
        }
        gammaEmpty = false;
        solIters++;

        if (!opts.employDualSolverMultipleTimes) {
            std::cout << "[ShapeMM] Exiting solution loop because 'employDualSolverMultipleTimes' is set to false" << std::endl;
            break;
        }
    }
    
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    Eigen::MatrixXf duration(1, 3);
    duration(0, 0) = std::chrono::duration_cast<std::chrono::seconds>(t4 - t0).count();
    std::cout << "[ShapeMM] TOTAL ELAPSED TIME: " << std::chrono::duration_cast<std::chrono::seconds>(t4 - t0).count() << " [s]" << std::endl;
    duration(0, 1) = std::chrono::duration_cast<std::chrono::seconds>(timeHeuristic).count();
    std::cout << "[ShapeMM]   Time heuristic: " << std::chrono::duration_cast<std::chrono::seconds>(timeHeuristic).count() << " [s]" << std::endl;
    if (opts.useMinMarginals) {
        duration(0, 2) = std::chrono::duration_cast<std::chrono::seconds>(timeBDD).count();
        std::cout << "[ShapeMM]   Time bdd: " << std::chrono::duration_cast<std::chrono::seconds>(timeBDD).count() << " [s]" << " ( in " << solIters+1 << " calls )" << std::endl;
    }

    if (opts.writeModelToFileAfterSolving)
        utils::writeMatrixToFile(duration, opts.modelName + "_time");
    
    // extract a Gamma which does not contain flags if the solution is incomplete
    if (heuristicStatus == SOL_INCOMPLETE) {
        SparseMatInt8Cm GammSparse = Gamma.sparseView();
        for(typename SparseMatInt8Cm::InnerIterator it(GammSparse, 0); it; ++it) {
            if (it.value() != 1) {
                Gamma(it.index()) = 0;
            }
        }
    }
    
    if (opts.verbose) printSolutionInfo(Gamma.sparseView());
    if (opts.writeModelToFileAfterSolving) writeSolutionToFile(Gamma);
    
    return Gamma;
    
}
