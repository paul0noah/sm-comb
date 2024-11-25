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
#include <igl/adjacency_list.h>
#include <min_marginal_utils.h>
#include "helper/utils.hpp"
#include <chrono>
#include <filesystem>
#include <algorithm>
#define DEBUG_PRUNING false

void ShapeMatchModel::generate() {
    constraintsFulfilled = false;
    pruned = false;
    initialLowerBound = -1;
    generationSuccessfull = false;
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
    generationSuccessfull = true;
}

bool ShapeMatchModel::smmCreatedSuccessFully() {
    return generationSuccessfull;
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

ShapeMatchModel::ShapeMatchModel(Eigen::MatrixXi FX, Eigen::MatrixXf VX, Eigen::MatrixXi FY, Eigen::MatrixXf VY, Eigen::MatrixXi coarsep2pmap, Eigen::MatrixXi IXf2c, Eigen::MatrixXi IYf2c, int c2fneighborhood) :
        ShapeMatchModel(FX, VX, FY, VY, coarsep2pmap, IXf2c, IYf2c) {
    c2fNeighborhood = c2fneighborhood;
    if (c2fneighborhood > 2) {
        if (opts.verbose) std::cout << "[ShapeMM] Warning: c2fneighborhood > 2 not supported. falling back to 2." << std::endl;
    }
}

ShapeMatchModel::ShapeMatchModel(Eigen::MatrixXi FX, Eigen::MatrixXf VX, Eigen::MatrixXi FY, Eigen::MatrixXf VY, Eigen::MatrixXi coarsep2pmap, Eigen::MatrixXi IXf2c, Eigen::MatrixXi IYf2c) :
shapeX(Shape(VX, FX)),
shapeY(Shape(VY, FY)),
combos(shapeX, shapeY),
constr(shapeX, shapeY, combos),
deformationEnergy(shapeX, shapeY, combos),
ilpGenerated(false),
minMarginals(combos.getFaCombo().rows(), 1),
minMarginalsComputed(false),
bddsolver(NULL),
c2fNeighborhood(1) {

    ilp = LPMP::ILP_input();
    pruned = true;
    initialLowerBound = -1;
    generationSuccessfull = false;
    if (!checkWatertightness()) return;
    if (opts.verbose) std::cout << "[ShapeMM] Generating Shape Match Model Pruned..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   > Product Space" << std::endl;
    combos.getFaCombo();
    const Eigen::VectorX<bool> PruneVec = getPruneVec(coarsep2pmap, IXf2c, IYf2c);


    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;

    if (opts.verbose) std::cout << "[ShapeMM]   > Energies" << std::endl;
    //deformationEnergy.get();
    //deformationEnergy.prune(PruneVec);
    if (opts.verbose) std::cout << "[ShapeMM]     => skipping (contact project owner if you need this)" << std::endl;
    combos.prune(PruneVec);
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "  [ms])" << std::endl;

    if (opts.verbose) std::cout << "[ShapeMM]   > Constraints" << std::endl;
    constr.computePrunedConstraints(PruneVec, coarsep2pmap, IXf2c, IYf2c);
    if (DEBUG_PRUNING) {
        std::cout << "      Debugging Pruning..." << std::endl;
        Constraints tempConstr = Constraints(shapeX, shapeY, combos);
        tempConstr.getConstraintMatrix();
        tempConstr.prune(PruneVec);
        auto Atemp = tempConstr.getConstraintMatrix();
        auto A =    constr.getConstraintMatrix();
        std::cout << "A " << (Atemp.cast<int>() - A.cast<int>()).norm() << std::endl;

        auto RHStemp = tempConstr.getConstraintVector();
        auto RHS =    constr.getConstraintVector();
        std::cout << "RHS " << (RHS.cast<int>() - RHStemp.cast<int>()).norm() << std::endl;
    }


    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (opts.verbose) std::cout << "[ShapeMM]   Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "  [ms])" << std::endl;
    if (opts.verbose) std::cout << "[ShapeMM] Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() << "  [ms])" << std::endl;
    generationSuccessfull = true;

    opts.useConstraintsGroups = false;
}

ShapeMatchModel::ShapeMatchModel(Eigen::MatrixXi FX, Eigen::MatrixXf VX, Eigen::MatrixXi FY, Eigen::MatrixXf VY):
    shapeX(Shape(VX, FX)),
    shapeY(Shape(VY, FY)),
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

ShapeMatchModel::ShapeMatchModel(std::string filenameShapeX, int numFacesX, std::string filenameShapeY, int numFacesY):
    shapeX(filenameShapeX, numFacesX),
    shapeY(filenameShapeY, numFacesY),
    combos(shapeX, shapeY),
    constr(shapeX, shapeY, combos),
    deformationEnergy(shapeX, shapeY, combos),
    ilpGenerated(false),
    minMarginals(combos.getFaCombo().rows(), 1),
    minMarginalsComputed(false)
{
    ilp = LPMP::ILP_input();
    generate();
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

    if (pruned) {
        for (int i = 0; i < 10; i++) {
            std::cout << "[ShapeMM] Warning" << std::endl;
        }
        std::cout << "[ShapeMM] The model was pruned with a previous solution." << std::endl;
        std::cout << "[ShapeMM] The following code paths are not tested for pruned solutions => expect issues." << std::endl;
    }

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
    constraintsFulfilled = heuristicStatus == SOL_COMPLETE;
    
    if (opts.verbose) printSolutionInfo(Gamma.sparseView());
    if (opts.writeModelToFileAfterSolving) writeSolutionToFile(Gamma);
    
    return Gamma;
    
}
