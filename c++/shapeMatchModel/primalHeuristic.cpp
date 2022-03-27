//
//  primalHeuristic.cpp
//
//  Created by Paul Rötzer on 01.07.21.
//

#include <stdio.h>
#include <iostream>
#include <igl/sort.h>
#include "shapeMatchModel.hpp"
#include "helper/utils.hpp"
#include "shapeMatchModel/constraintsComputation/constraints.hpp"
#include "primalHeuristic.hpp"
#if defined(_OPENMP)
    #include <omp.h>
#endif

const char* changeReasons[] = {
    "UNKNOWN",
    "INIT",
    "SOL_CANDIDATE",
    "PROPAGATION",
    "BACKTRACK",
    "ROTATION",
    "ELIMINATION"
};

const int8_t VAR_UNDECIDED = -1;
const int8_t VAR_ZERO = 0;
const int8_t VAR_ONE = 1;
const int8_t VAR_SOL_CANDIDATE = 2;
const int8_t VAR_LIKELY_SOL_CANDIDATE = 3;
const int8_t VAR_MUST_BE_SOL = 4;

void PrimalHeuristic::updateMinMarginals(Eigen::MatrixXf &newMinMarginals){
    assert(minMarginals.rows() == newMinMarginals.rows());
    assert(minMarginals.cols() == newMinMarginals.cols());
    
    minMarginals = newMinMarginals;
}

/*
 For better peformance the sparse matrices have to be available row-major as well col-major.
 Additionally, the del-matrix needs to be split up into the 1s and -1s (to find adjacent product triangles).
 */
void PrimalHeuristic::extractMatrices() {
    if (opts.verbose) std::cout << "[PrimHeu] Initing Primal Heuristic..." << std::endl;

    if (opts.autoSetMaxIter) {
        opts.maxIter = 0.3 * std::min(numFacesX, numFacesY);
        if (opts.verbose) std::cout << "[PrimHeu]     Auto-setting max-iter to " << opts.maxIter << std::endl;
    }

    if (opts.autoSetMaxBacktracks) {
        opts.maxBacktracks = 0.05 * std::max(numFacesX, numFacesY);
        if (opts.verbose) std::cout << "[PrimHeu]     Auto-setting max-backtrack to " << opts.maxBacktracks << std::endl;
    }

    SparseMatInt8Rm del =
        constr.getConstraintMatrix().block(0, 0, numProductEdges, numProductFaces);
    SparseMatInt8Rm piX =
        constr.getConstraintMatrix().block(numProductEdges, 0, numFacesX, numProductFaces);
    SparseMatInt8Rm piY =
        constr.getConstraintMatrix().block(numProductEdges + numFacesX, 0, numFacesY, numProductFaces);
    
    
    
    
    std::vector<Eigen::Triplet<int8_t>> entriesDelPlus;
    entriesDelPlus.reserve(del.nonZeros() / 2);
    std::vector<Eigen::Triplet<int8_t>> entriesDelMinus;
    entriesDelMinus.reserve(del.nonZeros() / 2);

    // Loop outer level
    for (int k = 0; k < del.outerSize(); ++k) {
        // loop inner level
        for (typename SparseMatInt8Rm::InnerIterator it(del,k); it; ++it) {
            Eigen::Triplet<int8_t> triplet(it.row(), it.col(), it.value());
            if (it.value() > 0) {
                entriesDelPlus.push_back(triplet);
            }
            else {
                entriesDelMinus.push_back(triplet);
            }
        }
    }
    delPlusRm.resize(del.rows(), del.cols());
    delPlusRm.setFromTriplets(entriesDelPlus.begin(), entriesDelPlus.end());
    delPlusCm.resize(del.rows(), del.cols());
    delPlusCm.setFromTriplets(entriesDelPlus.begin(), entriesDelPlus.end());
    delMinusRm.resize(del.rows(), del.cols());
    delMinusRm.setFromTriplets(entriesDelMinus.begin(), entriesDelMinus.end());
    delMinusCm.resize(del.rows(), del.cols());
    delMinusCm.setFromTriplets(entriesDelMinus.begin(), entriesDelMinus.end());
    
    std::vector<Eigen::Triplet<int8_t>> entriesPiX;
    entriesPiX.reserve(piX.nonZeros());
    // Loop outer level
    for (int k = 0; k < piX.outerSize(); ++k) {
        // loop inner level
        for (typename SparseMatInt8Rm::InnerIterator it(piX,k); it; ++it) {
            Eigen::Triplet<int8_t> triplet(it.row(), it.col(), it.value());
            entriesPiX.push_back(triplet);
        }
    }
    piXRm.resize(piX.rows(), piX.cols());
    piXRm.setFromTriplets(entriesPiX.begin(), entriesPiX.end());
    piXCm.resize(piX.rows(), piX.cols());
    piXCm.setFromTriplets(entriesPiX.begin(), entriesPiX.end());
    
    std::vector<Eigen::Triplet<int8_t>> entriesPiY;
    entriesPiY.reserve(piY.nonZeros());
    // Loop outer level
    for (int k = 0; k < piY.outerSize(); ++k) {
        // loop inner level
        for (typename SparseMatInt8Rm::InnerIterator it(piY,k); it; ++it) {
            Eigen::Triplet<int8_t> triplet(it.row(), it.col(), it.value());
            entriesPiY.push_back(triplet);
        }
    }
    piYRm.resize(piY.rows(), piY.cols());
    piYRm.setFromTriplets(entriesPiY.begin(), entriesPiY.end());
    piYCm.resize(piY.rows(), piY.cols());
    piYCm.setFromTriplets(entriesPiY.begin(), entriesPiY.end());
    if (opts.verbose) std::cout << "[PrimHeu] Done" << std::endl;
}


PrimalHeuristic::PrimalHeuristic(Shape &sX,
                Shape &sY,
                Combinations& cbs,
                Eigen::MatrixXf& costs,
                Eigen::MatrixXf& minMarginals,
                Constraints& cstrs,
                PrimalHeuristicOpts opts) :
                shapeX(sX),
                shapeY(sY),
                combos(cbs),
                costs(costs),
                minMarginals(minMarginals),
                constr(cstrs),
                opts(opts) {
        
    numFacesX = shapeX.getNumFaces();
    numFacesY = shapeY.getNumFaces();
    numEdgesX = shapeX.getNumEdges();
    numEdgesY = shapeY.getNumEdges();
    numVerticesX = shapeX.getNumVertices();
    numVerticesY = shapeY.getNumVertices();
    numProductEdges = 2 * numEdgesX * numEdgesY +
                    numVerticesX * numEdgesY + numEdgesX * numVerticesY;
    numProductFaces = 21 * numFacesX * numFacesY +
                    numVerticesX * numFacesY + numVerticesY * numFacesX;
    
    // init delPlus delMinus etc.
    extractMatrices();

    initializationInProgress = false;
    triangleNeighboursX = shapeX.getTriangleNeighbours();
    triangleNeighboursY = shapeY.getTriangleNeighbours();
}

void PrimalHeuristic::updateOpts(PrimalHeuristicOpts newOpts) {
    opts = newOpts;
}

void PrimalHeuristic::addNewChangeToSetOfChanges(tsl::robin_set<Change> &changesAtCurrentDepth, Change newChange) const {
    const auto it = changesAtCurrentDepth.find(newChange);
    // we already changed this index at the current depth ?
    if (it != changesAtCurrentDepth.end()) {
        Change oldChangeOnSameIdx = *it;
        newChange.oldValue = oldChangeOnSameIdx.oldValue;
        changesAtCurrentDepth.erase(it);
    }
    changesAtCurrentDepth.insert(newChange);
}

bool PrimalHeuristic::preCheckSettingVar(CORE_STATUS &coreStatus, const int idx, const int8_t val) const {
    size_t &numVarSolCandidate = coreStatus.numVarSolCandidate = 0;
    size_t &numVarLikelySolCandidate = coreStatus.numVarLikelySolCandidate = 0;
    size_t &numVarMustBeSol = coreStatus.numVarMustBeSol = 0;
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    tsl::robin_set<unsigned int> &idxVarOnes = coreStatus.idxVarOnes;
    MatrixInt8 &Gamma = coreStatus.Gamma;
    assert(val == 1);
    if (Gamma(idx) == VAR_UNDECIDED) {
        return false;
    }
    // check delPlus
    for (typename SparseMatInt8Cm::InnerIterator it(delPlusCm, idx); it; ++it) {
        const int row = it.row();
        const int rhs = constr.getConstraintVector().coeff(row);
        assert(rhs == 0);
        int numNegCoeff = 0, numPosCoeff = 0;
        for (typename SparseMatInt8Rm::InnerIterator iit(delPlusRm, row); iit; ++iit) {
            if (Gamma(iit.col()) == 1 || (iit.index() == idx && val)  ) numPosCoeff++;
        }
        for (typename SparseMatInt8Rm::InnerIterator iit(delMinusRm, row); iit; ++iit) {
            if (Gamma(iit.col()) != 0) numNegCoeff++;
        }
        const bool constraintFullfillable = (numPosCoeff > 0 && numNegCoeff > 0);
        if (!constraintFullfillable) {
            //std::cout << "  Prechecking: del not fullfillable (+)" << std::endl;
            return false;
        }
    }
    // check delMinus
    for (typename SparseMatInt8Cm::InnerIterator it(delMinusCm, idx); it; ++it) {
        const int row = it.row();
        const int rhs = constr.getConstraintVector().coeff(row);
        assert(rhs == 0);
        int numNegCoeff = 0, numPosCoeff = 0;
        for (typename SparseMatInt8Rm::InnerIterator iit(delPlusRm, row); iit; ++iit) {
            if (Gamma(iit.col()) != 0) numNegCoeff++;
        }
        for (typename SparseMatInt8Rm::InnerIterator iit(delMinusRm, row); iit; ++iit) {
            if (Gamma(iit.col()) == 1 || (iit.index() == idx && val)  ) numPosCoeff++;
        }
        const bool constraintFullfillable = (numPosCoeff > 0 && numNegCoeff > 0);
        if (!constraintFullfillable) {
            //std::cout << "  Prechecking: del not fullfillable (-)" << std::endl;
            return false;
        }
    }
    const float piXOffset = delPlusCm.rows();
    // check piX
    for (typename SparseMatInt8Cm::InnerIterator it(piXCm, idx); it; ++it) {
        const int row = it.row();
        int rhs = constr.getConstraintVector().coeff(piXOffset + row);
        assert(rhs == 1);
        int numPosCoeffWhereGamma1 = 0, numPosCoeffWhereGammaUndecided = 0;
        for (typename SparseMatInt8Rm::InnerIterator iit(piXRm, row); iit; ++iit) {
            if ( Gamma(iit.col()) == 1 || (iit.index() == idx && val) ) numPosCoeffWhereGamma1++;
            if ( Gamma(iit.col()) != 1 && Gamma(iit.col()) != 0 && iit.index() != idx ) numPosCoeffWhereGammaUndecided++;
        }
        const bool constraintFullfillable = (numPosCoeffWhereGamma1 == 1 || numPosCoeffWhereGammaUndecided > 1);
        if (val == 1) assert(numPosCoeffWhereGamma1 == 1);
        if (val == 0) assert(numPosCoeffWhereGammaUndecided > 1);
        if (!constraintFullfillable) {
            //std::cout << "  Prechecking: piX not fullfillable" << std::endl;
            return false;
        }
    }
    const float piYOffset = delPlusCm.rows() + piXCm.rows();
    // check piY
    for (typename SparseMatInt8Cm::InnerIterator it(piYCm, idx); it; ++it) {
        const int row = it.row();
        int rhs = constr.getConstraintVector().coeff(piYOffset + row);
        assert(rhs == 1);
        int numPosCoeffWhereGamma1 = 0, numPosCoeffWhereGammaUndecided = 0;
        for (typename SparseMatInt8Rm::InnerIterator iit(piYRm, row); iit; ++iit) {
            if ( Gamma(iit.col()) == 1 || (iit.index() == idx && val) ) numPosCoeffWhereGamma1++;
            if ( Gamma(iit.col()) != 1 && Gamma(iit.col()) != 0 && iit.index() != idx ) numPosCoeffWhereGammaUndecided++;
        }
        const bool constraintFullfillable = (numPosCoeffWhereGamma1 == 1 || numPosCoeffWhereGammaUndecided > 1);
        if (val == 1) assert(numPosCoeffWhereGamma1 == 1);
        if (val == 0) assert(numPosCoeffWhereGammaUndecided > 1);
        if (!constraintFullfillable) {
            //std::cout << "  Prechecking: piY not fullfillable" << std::endl;
            return false;
        }
    }
    //std::cout << "  Prechecking: variable good" << std::endl;
    return true;
}

/* function setGammaValue
    this function is used to apply changes to the binary vector Gamma while making sure no conflicts appear
    returns false if a conflict occured
 */
inline bool PrimalHeuristic::setGammaValue(CORE_STATUS &coreStatus,
                                           const int idx,
                                           int8_t newVal,
                                           tsl::robin_set<Change> &changes,
                                           CHANGE_REASON reason) const {
    size_t &numVarSolCandidate = coreStatus.numVarSolCandidate = 0;
    size_t &numVarLikelySolCandidate = coreStatus.numVarLikelySolCandidate = 0;
    size_t &numVarMustBeSol = coreStatus.numVarMustBeSol = 0;
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    tsl::robin_set<unsigned int> &idxVarOnes = coreStatus.idxVarOnes;
    MatrixInt8 &Gamma = coreStatus.Gamma;
    const int gammaVal = Gamma(idx);
    const int newValInt = newVal;
    const std::string outputPrefix = coreStatus.outputPrefix;
    
    // Only tri to tri matchings (dont set values with idx greater than 3 * numFacesX * numFacesY)
    if (opts.allowOnlyNonDegenerateMatchings && idx >= 3 * numFacesX * numFacesY) {
        return true;
    }
    
    // make sure we do not set a higher value than VAR_MUST_BE_SOL
    if (newVal > VAR_MUST_BE_SOL) {
        newVal = VAR_MUST_BE_SOL;
    }

    if (Gamma(idx) == VAR_ONE) {
        if (newVal == VAR_ONE) {
            return true;
        }
        if (newVal == VAR_ZERO && reason != BACKTRACK) {
            if (opts.debugOutput) {
                std::cout << outputPrefix << "      Error setting Gamma value. Already set 1 can't be reset with 0" << std::endl;
                std::cout << outputPrefix << "      -> REASON:" << changeReasons[reason] << std::endl;
            }
            return false;
        }
        return true;
    }
    if (Gamma(idx) == VAR_ZERO) {
        if (newVal == VAR_ZERO) {
            return true;
        }
        if (newVal == VAR_ONE) {
            if (opts.debugOutput) {
                std::cout << outputPrefix << "      Error setting Gamma value. Already set 0 can't be reset with 1" << std::endl;
                std::cout << outputPrefix << "      -> REASON:" << changeReasons[reason] << std::endl;
            }
            return false;
        }
        return true;
    }
    /*if (Gamma(idx) == VAR_SOL_CANDIDATE && newVal == VAR_SOL_CANDIDATE) {
        
        //return true;
    }
    if (Gamma(idx) == VAR_LIKELY_SOL_CANDIDATE ){
        if (newVal == VAR_LIKELY_SOL_CANDIDATE) {
            //return true;
        }
        if (newVal  == VAR_SOL_CANDIDATE) {
            newVal = VAR_LIKELY_SOL_CANDIDATE;
            //return true;
        }
    }*/
    if (Gamma(idx) == VAR_MUST_BE_SOL) {
        if (newVal == VAR_MUST_BE_SOL) {
            return true;
        }
        /*if (newVal  == VAR_SOL_CANDIDATE || newVal  == VAR_LIKELY_SOL_CANDIDATE) {
            newVal = VAR_MUST_BE_SOL;
            //return true;
        }*/
    }
    

    Change change;
    change.idx = idx;
    change.oldValue = Gamma(idx);
    change.reason = reason;
    addNewChangeToSetOfChanges(changes, change);
    Gamma(idx) = newVal;
    
    // make sure we add the candidates to the set and the candidates are distinct
    tsl::robin_set<Candidate>::iterator it = candidates.find(Candidate(idx, newVal));
    if (it != candidates.end()) { // this is true if we have found idx in candidates
        candidates.erase(it);
    }
    if (newVal >= VAR_SOL_CANDIDATE) {
        candidates.insert(Candidate(idx, newVal));
    }
    if (newVal == VAR_ONE) {
        idxVarOnes.insert(idx);
    }

    return true;
    
}

/* function propagate
    uses the constraints to propagate consequences of changes in the binary vector Gamma
 
 */
bool PrimalHeuristic::propagate(int idx,
                                CORE_STATUS &coreStatus,
                                tsl::robin_set<Change> &changesAtCurrentDepth) const {
    size_t &numVarSolCandidate = coreStatus.numVarSolCandidate = 0;
    size_t &numVarLikelySolCandidate = coreStatus.numVarLikelySolCandidate = 0;
    size_t &numVarMustBeSol = coreStatus.numVarMustBeSol = 0;
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    tsl::robin_set<unsigned int> &idxVarOnes = coreStatus.idxVarOnes;
    MatrixInt8 &Gamma = coreStatus.Gamma;
    const std::string outputPrefix = coreStatus.outputPrefix;
    
    unsigned int numberOfChangesAtCurrentDepth = changesAtCurrentDepth.size();
    
    /*
     Propagation Rule 1
     does the new matching belong to a triangle in shape X?
     */
    int row = -1;
    for (typename SparseMatInt8Cm::InnerIterator it(piXCm, idx); it; ++it) {
        if (it.value() == 1) {
            row = it.row();
            break;
        }
    }
    // set all other faces to zero
    if (row != -1) {
        numberOfChangesAtCurrentDepth += piXRm.row(row).nonZeros() - 1;
        changesAtCurrentDepth.reserve(numberOfChangesAtCurrentDepth);

        for (typename SparseMatInt8Rm::InnerIterator it(piXRm, row); it; ++it) {
            if (it.col() != idx) {
                bool setSuccess = setGammaValue(coreStatus, it.col(), VAR_ZERO, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
                if (!setSuccess) {
                    if (opts.debugOutput)  std::cout << outputPrefix << "      Set Failure with Propagtion X" << std::endl;
                    // some conflict occured => propagation not successfull
                    return false;
                }
            }
        }
    }
    
    
    /*
     Propagation Rule 2
     does the new matching belong to a triangle in shape Y?
     */
    row = -1;
    for (typename SparseMatInt8Cm::InnerIterator it(piYCm, idx); it; ++it) {
        if (it.value() == 1) {
            row = it.row();
            break;
        }
    }
    // set all other faces to zero
    if (row != -1) {
        numberOfChangesAtCurrentDepth += piYRm.row(row).nonZeros() - 1;
        changesAtCurrentDepth.reserve(numberOfChangesAtCurrentDepth);

        for (typename SparseMatInt8Rm::InnerIterator it(piYRm, row); it; ++it) {
            if (it.col() != idx) {
                bool setSuccess = setGammaValue(coreStatus, it.col(), VAR_ZERO, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
                if (!setSuccess) {
                    if (opts.debugOutput) std::cout << outputPrefix << "      Set Failure with Propagtion Y" << std::endl;
                    // some conflict occured => propagation not successfull
                    return false;
                }
            }
        }
    }
    
    
    /*
     Propagation Rule 3
     Use positive orientation of the edge
     */
    // at max there are three rows which are nonzero in del
    // => each triangle has three side ;)
    int rows[] = {-1, -1, -1};
    int numRows = 0;
    for (typename SparseMatInt8Cm::InnerIterator it(delPlusCm, idx); it; ++it) {
        if (it.value() == 1) { // i guess we dont need this check ?
            rows[numRows] = it.row();
            numRows++;
        }
    }
    for (int i = 0; i < numRows; i++) {
        numberOfChangesAtCurrentDepth += delPlusRm.row(rows[i]).nonZeros() - 1;
        changesAtCurrentDepth.reserve(numberOfChangesAtCurrentDepth);
        // set all other product faces corresponding to this orientation of this edge to zero
        for (typename SparseMatInt8Rm::InnerIterator it(delPlusRm, rows[i]); it; ++it) {
            if (it.col() != idx) {
                bool setSuccess = setGammaValue(coreStatus, it.col(), VAR_ZERO, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
                if (!setSuccess) {
                    if (opts.debugOutput) std::cout << outputPrefix << "      Set Failure with Propagtion delPlus: setting other edges to zero" << std::endl;
                    // some conflict occured => propagation not successfull
                    return false;
                }
            }
        }
        
        // if all except one of the negative versions of this edge are zero set the corresponding product face to 1
        int sumNegEdge = 0;
        int idxNegEdgeNotZero = -1;
        for (typename SparseMatInt8Rm::InnerIterator it(delMinusRm, rows[i]); it; ++it) {
            if (Gamma(it.col()) != 0) {
                sumNegEdge += 1;
                idxNegEdgeNotZero = it.col();
            }
        }
        if (sumNegEdge == 1) {
            bool success = setGammaValue(coreStatus, idxNegEdgeNotZero, VAR_MUST_BE_SOL, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
            if (!success) {
                if (opts.debugOutput)
                    std::cout << outputPrefix << "      Set Failure with Propagtion delPlus: setting last opposite edge to one" << std::endl;
                return false;
            }
        }
        else {
            // set all negative orientations of the edge to solution candidates
            for (typename SparseMatInt8Rm::InnerIterator it(delMinusRm, rows[i]); it; ++it) {
                setGammaValue(coreStatus, it.col(), VAR_SOL_CANDIDATE, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
            }
        }
    }
    
    /*
     Propagation Rule 4
     Use negative orientation of the edge
     */
    // at max there are three rows which are nonzero in del
    // => each triangle has three side ;)
    rows[0] = -1; rows[1] = -1; rows[2] = -1;
    numRows = 0;
    for (typename SparseMatInt8Cm::InnerIterator it(delMinusCm, idx); it; ++it) {
        if (it.value() == -1) { // i guess we dont need this check ?
            rows[numRows] = it.row();
            numRows++;
        }
    }
    for (int i = 0; i < numRows; i++) {
        numberOfChangesAtCurrentDepth += delMinusRm.row(rows[i]).nonZeros() - 1;
        changesAtCurrentDepth.reserve(numberOfChangesAtCurrentDepth);
        // set all other product faces corresponding to this orientation of this edge to zero
        for (typename SparseMatInt8Rm::InnerIterator it(delMinusRm, rows[i]); it; ++it) {
            if (it.col() != idx) {
                bool success = setGammaValue(coreStatus, it.col(), VAR_ZERO, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
                if (!success) {
                    if (opts.debugOutput) std::cout << outputPrefix << "      Set Failure with Propagtion delMinus: setting other edges to zero" << std::endl;
                    // some conflict occured => propagation not successfull
                    return false;
                }
            }
        }
        
        // if all except one of the positive versions of this edge are zero set the corresponding product face to 1
        int sumPosEdge = 0;
        int idxPosEdgeNotZero = -1;
        for (typename SparseMatInt8Rm::InnerIterator it(delPlusRm, rows[i]); it; ++it) {
            if (Gamma(it.col()) != 0) {
                sumPosEdge += 1;
                idxPosEdgeNotZero = it.col();
            }
        }
        if (sumPosEdge == 1) {
            bool success = setGammaValue(coreStatus, idxPosEdgeNotZero, VAR_MUST_BE_SOL, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
            if (!success) {
                if (opts.debugOutput)
                    std::cout << outputPrefix << "      Set Failure with Propagtion delMinus: setting last opposite edge to one" << std::endl;
                return false;
            }
        }
        else {
            // set all negative orientations of the edge to solution candidates
            for (typename SparseMatInt8Rm::InnerIterator it(delPlusRm, rows[i]); it; ++it) {
                setGammaValue(coreStatus, it.col(), VAR_SOL_CANDIDATE, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
            }
        }
    }
    
    /*
     Propagation Rule 5
     Whenever all neighbours of a solution candidate are already a solution
     => this solution candidate has to be in the solution (it somehow fills a hole)
     */
    for (tsl::robin_set<Candidate>::iterator it = candidates.begin(); it != candidates.end(); ++it) {
        const Candidate cand = *it;
        const int f = cand.idx;
        if (Gamma(f) == VAR_SOL_CANDIDATE || Gamma(f) == VAR_LIKELY_SOL_CANDIDATE) {
            int numNeighboursInSolution = 0;
            
            // check if three neighbours are within the solution
            for (typename SparseMatInt8Cm::InnerIterator it(delPlusCm, f); it; ++it) {
                for (typename SparseMatInt8Rm::InnerIterator iit(delMinusRm, it.index()); iit; ++iit) {
                    if (Gamma(iit.index()) == VAR_ONE) {
                        numNeighboursInSolution++;
                        break;
                    }
                }
            }
            for (typename SparseMatInt8Cm::InnerIterator it(delMinusCm, f); it; ++it) {
                for (typename SparseMatInt8Rm::InnerIterator iit(delPlusRm, it.index()); iit; ++iit) {
                    if (Gamma(iit.index()) == VAR_ONE) {
                        numNeighboursInSolution++;
                        break;
                    }
                }
            }
            
            if (numNeighboursInSolution == 2 && Gamma(f) != VAR_LIKELY_SOL_CANDIDATE) {
                const bool success = setGammaValue(coreStatus, f, VAR_LIKELY_SOL_CANDIDATE, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
                if (!success) ASSERT_NEVER_REACH;
            }
            if (numNeighboursInSolution == 3) {
                const bool success = setGammaValue(coreStatus, f, VAR_MUST_BE_SOL, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
                if (!success) ASSERT_NEVER_REACH;
            }
            if (numNeighboursInSolution > 3) {
                ASSERT_NEVER_REACH;
            }
        }
    }
    
    /*
     Propagation Rule 6
     Use minMarginals to upvote solution candidates
     */
    for (tsl::robin_set<Candidate>::iterator it = candidates.begin(); it != candidates.end(); ++it) {
        const Candidate cand = *it;
        const int f = cand.idx;
        if (Gamma(f) == VAR_SOL_CANDIDATE) {
            if (minMarginals(f) < -FLOAT_EPSI) {
                setGammaValue(coreStatus, f, VAR_LIKELY_SOL_CANDIDATE, changesAtCurrentDepth, CHANGE_REASON::PROPAGATION);
            }
        }
        
    }
    
    return true;
}

/*
 sort the variables with equal flags (e.g. VAR_LIKELY_SOL_CANDIDATE) w.r.t to min marginals and costs
 */
void PrimalHeuristic::getCandidatesAscending(CORE_STATUS &coreStatus,
                                       Eigen::MatrixXi &variableCandidates,
                                       const int8_t flag,
                                       const int count,
                                       const int offset) const
{
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    MatrixInt8 &Gamma = coreStatus.Gamma;
    Eigen::MatrixXf unsortedCostOfCandidates(count, 1);
    Eigen::MatrixXi localVarCandidates(count, 1);

    
    // extract var candidates
    int idx = 0;
    int numZeroMinMargs = 0;
    for (tsl::robin_set<Candidate>::iterator it = candidates.begin(); it != candidates.end(); ++it) {
        const Candidate currentCandidate = (*it);
        const int i = currentCandidate.idx;
        if (currentCandidate.flag == flag) {
            localVarCandidates(idx) = i;
            if (opts.useMinMarginals) {
                unsortedCostOfCandidates(idx) = minMarginals(i);
                if (minMarginals(i) > -FLOAT_EPSI && minMarginals(i) < FLOAT_EPSI) {
                    numZeroMinMargs++;
                }
            }
            else {
                unsortedCostOfCandidates(idx) = costs(i);
            }
            idx++;
        }
    }
    Eigen::MatrixXf sortedCostsOfCandidates(count, 1);
    Eigen::MatrixXi idxInVariableCandidates(count, 1);
    igl::sort(unsortedCostOfCandidates,
              1,    // sort each column
              true, // -> ascending order
              sortedCostsOfCandidates,
              idxInVariableCandidates);
    
    Eigen::MatrixXi sortedVariableCandidates(count, 1);
    for (int i = 0; i < localVarCandidates.rows(); i++) {
        sortedVariableCandidates(i) = localVarCandidates(idxInVariableCandidates(i));
    }
    
    /*
     Whenever we have zero min marginals within the candidates we want to sort those zero min margs
     according to their energy
     */
    if (numZeroMinMargs > 1) {
        Eigen::MatrixXi zeroMinMargs(numZeroMinMargs, 1);
        Eigen::MatrixXf energyZeroMinMargs(numZeroMinMargs, 1);
        int idxZeroMM = 0;
        int firstZeroMM = -1;
        for (int i = 0; i < count; i++) {
            if (sortedCostsOfCandidates(i) > -FLOAT_EPSI && sortedCostsOfCandidates(i) < FLOAT_EPSI) {
                zeroMinMargs(idxZeroMM) = sortedVariableCandidates(i);
                energyZeroMinMargs(idxZeroMM) = costs(sortedVariableCandidates(i));
                idxZeroMM++;
                if (firstZeroMM == -1) {
                    firstZeroMM = i;
                }
            }
        }
        assert(firstZeroMM != -1);
        
        Eigen::MatrixXf sortedEnergyOfZeroMM(numZeroMinMargs, 1);
        Eigen::MatrixXi idxInZeroMinMargs(numZeroMinMargs, 1);
        igl::sort(energyZeroMinMargs,
                  1,    // sort each column
                  true, // -> ascending order
                  sortedEnergyOfZeroMM,
                  idxInZeroMinMargs);
        Eigen::MatrixXi sortedZeroMinMargs(numZeroMinMargs, 1);
        for (int i = 0; i < numZeroMinMargs; i++) {
            sortedZeroMinMargs(i) = zeroMinMargs(idxInZeroMinMargs(i));
        }
        sortedVariableCandidates.block(firstZeroMM, 0, numZeroMinMargs, 1) = sortedZeroMinMargs;
    }
    
    variableCandidates.block(offset, 0, count, 1) = sortedVariableCandidates;
}


void countCandidates(CORE_STATUS &coreStatus) {
    size_t &numVarSolCandidate = coreStatus.numVarSolCandidate = 0;
    size_t &numVarLikelySolCandidate = coreStatus.numVarLikelySolCandidate = 0;
    size_t &numVarMustBeSol = coreStatus.numVarMustBeSol = 0;
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    numVarSolCandidate = 0;
    numVarLikelySolCandidate = 0;
    numVarMustBeSol = 0;
    for (tsl::robin_set<Candidate>::iterator it = candidates.begin(); it != candidates.end(); ++it) {
        const Candidate currentCandidate = (*it);
        if (currentCandidate.flag == VAR_SOL_CANDIDATE) numVarSolCandidate++;
        if (currentCandidate.flag == VAR_LIKELY_SOL_CANDIDATE) numVarLikelySolCandidate++;
        if (currentCandidate.flag == VAR_MUST_BE_SOL) numVarMustBeSol++;
    }
}

/*
 from the propagation rules we obtain so-called variable candidates (variabels which are likely to be set to one)
 this function sorts the candidates w.r.t. to the min-margianls and costs
 */
void PrimalHeuristic::getVariabelCandidatesInAscendingOrder(CORE_STATUS &coreStatus,
                                      Eigen::MatrixXi &variableCandidates,
                                      const Eigen::MatrixXf cost) const {


    size_t &numVarSolCandidate = coreStatus.numVarSolCandidate = 0;
    size_t &numVarLikelySolCandidate = coreStatus.numVarLikelySolCandidate = 0;
    size_t &numVarMustBeSol = coreStatus.numVarMustBeSol = 0;

    countCandidates(coreStatus);
    const int totalCandidates = numVarSolCandidate + numVarLikelySolCandidate + numVarMustBeSol;
    variableCandidates.resize(totalCandidates, 1);
    variableCandidates = -variableCandidates.setOnes();
    
    if (numVarMustBeSol) {
        const int offset = 0;
        getCandidatesAscending(coreStatus, variableCandidates, VAR_MUST_BE_SOL, numVarMustBeSol, offset);
    }
    if (numVarLikelySolCandidate) {
        const int offset = numVarMustBeSol;
        getCandidatesAscending(coreStatus, variableCandidates, VAR_LIKELY_SOL_CANDIDATE, numVarLikelySolCandidate, offset);
    }
    if (numVarSolCandidate) {
        const int offset = numVarMustBeSol + numVarLikelySolCandidate;
        getCandidatesAscending(coreStatus, variableCandidates, VAR_SOL_CANDIDATE, numVarSolCandidate, offset);
    }
}

/*
 function to check if the current partial solution is still feasible
 */
bool PrimalHeuristic::isGammaFeasible(MatrixInt8 &Gamma) const {

    // one face of shape X can't receive a matching anymore
    for (int row = 0; row < piXRm.rows(); row++) {
        int gammaValSum = 0;
        for (typename SparseMatInt8Rm::InnerIterator it(piXRm, row); it; ++it) {
            gammaValSum += std::abs( Gamma( it.col() ) );
        }
        if (gammaValSum == 0) {
            return false;
        }
    }
    // one face of shape Y can't receive a matching anymore
    for (int row = 0; row < piYRm.rows(); row++) {
        int gammaValSum = 0;
        for (typename SparseMatInt8Rm::InnerIterator it(piYRm, row); it; ++it) {
            gammaValSum += std::abs( Gamma( it.col() ) );
        }
        if (gammaValSum == 0) {
            return false;
        }
    }
    
    // del
    for (int col = 0; col < Gamma.rows(); col++) {
        if (Gamma(col) == 1) {
            // delPlus
            for (typename SparseMatInt8Cm::InnerIterator it(delPlusCm, col); it; ++it) {
                int row = it.row();
                // check if there is more of this orientation set to one
                for (typename SparseMatInt8Rm::InnerIterator it(delPlusRm, row); it; ++it) {
                    if (Gamma(it.col()) == 1 && col != it.col()) {
                        // => we have another matching set to one which holds also the orientation fo this edge
                        return false;
                    }
                }
                
                // check if it is still possible to set one opposite orienation of this edge to one
                bool possibleOppFound = false;
                for (typename SparseMatInt8Rm::InnerIterator it(delMinusRm, row); it; ++it) {
                    if (Gamma(it.col()) != 0) {
                        possibleOppFound = true;
                        break;
                    }
                }
                if (!possibleOppFound) {
                    return false;
                }
            }
        
        
            // del Minus
            for (typename SparseMatInt8Cm::InnerIterator it(delMinusCm, col); it; ++it) {
                int row = it.row();
                // check if there is more of this orientation set to one
                for (typename SparseMatInt8Rm::InnerIterator it(delMinusRm, row); it; ++it) {
                    if (Gamma(it.col()) == 1 && col != it.col()) {
                        // => we have another matching set to one which holds also the orientation fo this edge
                        return false;
                    }
                }
                
                // check if it is still possible to set one opposite orienation of this edge to one
                bool possibleOppFound = false;
                for (typename SparseMatInt8Rm::InnerIterator it(delPlusRm, row); it; ++it) {
                    if (Gamma(it.col()) != 0) {
                        possibleOppFound = true;
                        break;
                    }
                }
                if (!possibleOppFound) {
                    return false;
                }
            }
        }
    }
    
    return true;
}


void PrimalHeuristic::undoChanges(CORE_STATUS &coreStatus, tsl::robin_set<Change> &changes) const {
    MatrixInt8 &Gamma = coreStatus.Gamma;
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    tsl::robin_set<unsigned int> &idxVarOnes = coreStatus.idxVarOnes;
    for (tsl::robin_set<Change>::iterator it = changes.begin(); it != changes.end(); ++it) {
        const Change currentChange = *it;
        const int idx = currentChange.idx;
        const int8_t oldVal = currentChange.oldValue;
        if (Gamma(idx) == 1 && oldVal != 1) {
            idxVarOnes.erase(idx);
        }
        Gamma(idx) = oldVal;
        tsl::robin_set<Candidate>::iterator candidateIt = candidates.find(Candidate(idx, -2));
        if (candidateIt != candidates.end()) { // this is true if we have found idx in candidates
            candidates.erase(candidateIt);
        }
        if (oldVal >= VAR_SOL_CANDIDATE) {
            candidates.insert(Candidate(idx, oldVal));
        }
    }
    changes.clear();
}

void PrimalHeuristic::undoChangesAtLastLevel(CORE_STATUS &coreStatus, std::vector<tsl::robin_set<Change>> &changesInGamma) const {
    if (changesInGamma.size() < 1) return;
    tsl::robin_set<Change> changesAtLastDepth = changesInGamma.back();
    changesInGamma.pop_back();
    
    undoChanges(coreStatus, changesAtLastDepth);
}

bool constraintsSatisfied(const SparseMatInt8   GammaBar,
                          const SparseMatInt8Rm &delPlus,
                          const SparseMatInt8   &delMinus,
                          const SparseMatInt8Rm &piX,
                          const SparseMatInt8Rm &piY) {
    
    // less multiplications to do, so this comes first ;)
    const int numFacesXMatched = (piX * GammaBar).cast<int>().sum();
    if ( numFacesXMatched != piX.rows()) {
        return false;
    }
    const int numFacesYMatched = (piY * GammaBar).cast<int>().sum();
    if ( numFacesYMatched != piY.rows()) {
        return false;
    }
    
    if ( (delPlus * GammaBar + delMinus * GammaBar).cwiseAbs().cast<int>().sum() != 0 ) {
        return false;
    }
    
    return true;
}

Change PrimalHeuristic::setNewSolVar(CORE_STATUS &coreStatus, int idx) const {
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    tsl::robin_set<unsigned int> &idxVarOnes = coreStatus.idxVarOnes;
    MatrixInt8 &Gamma = coreStatus.Gamma;
    Change nextOne;
    nextOne.idx = idx;
    nextOne.oldValue = Gamma(nextOne.idx);
    nextOne.reason = CHANGE_REASON::SOL_CANDIDATE;
    if (Gamma(idx) != VAR_UNDECIDED ) {
        candidates.erase(Candidate(idx, -2));
    }
    Gamma(nextOne.idx) = VAR_ONE;
    idxVarOnes.insert(idx);
    
    return nextOne;
}

/*
 this function return the indices of rows of the constraint matrices which are affected by a changes
 */
tsl::robin_set<int> PrimalHeuristic::getConstraintIdxAffectedByChanges(tsl::robin_set<Change> &changes, int idxFirstChange) const {
    assert(idxFirstChange >= 0);
    tsl::robin_set<int> idxAffectedConstraints; idxAffectedConstraints.reserve(changes.size() * 3);

    for (tsl::robin_set<Change>::iterator it = changes.begin(); it != changes.end(); ++it) {
        const Change change = *it;
        const int col = change.idx;

        // delPlus
        for (typename Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(delPlusCm, col); it; ++it) {
            idxAffectedConstraints.insert(it.index());
        }
        // delMinus
        for (typename Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(delMinusCm, col); it; ++it) {
            idxAffectedConstraints.insert(it.index());
        }
        // piX
        const int piXOffset = delPlusCm.rows();
        for (typename Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(piXCm, col); it; ++it) {
            idxAffectedConstraints.insert( it.index() + piXOffset );
        }
        // piY
        const int piYOffset = piXOffset + piXRm.rows();
        for (typename Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(piYCm, col); it; ++it) {
            idxAffectedConstraints.insert( it.index() + piYOffset );
        }
    }

    return idxAffectedConstraints;
}

/*
 clean up propagate remaining changes of each row of the matrices
 returns false if an unfeasible constraint is detected => gamma is not feasible anymore
 */
bool PrimalHeuristic::doCleanUp(int k, CORE_STATUS &coreStatus,
                                tsl::robin_set<Change> &changes,
                                int &numVarsEliminated,
                                int &numVarsSet,
                                const SparseMatInt8 &constrLHS,
                                const SparseVecInt8 &constrRHS) const {
    MatrixInt8 &Gamma = coreStatus.Gamma;
    int rhs = constrRHS.coeff(k);
    int numPosCoeff = 0;
    int numNegCoeff = 0;
    for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
        const int currentVarIdx = it.index();
        const int currentCoeff = it.value();
        if (Gamma(currentVarIdx) == 1) {
            rhs -= currentCoeff;
        }
        else if (Gamma(currentVarIdx) != 0) {
            if ( currentCoeff > 0 ) {
                numPosCoeff++;
            }
            else if ( currentCoeff < 0 ) {
                numNegCoeff++;
            }
        }
    }
    const bool allVarsOfConstraintNeedToBeZero = ( numPosCoeff == 0 && numNegCoeff > 1 && rhs == 0 ) ||
                                                 ( numNegCoeff == 0 && numPosCoeff > 1 && rhs == 0 );
    if (allVarsOfConstraintNeedToBeZero) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
            const int currentVarIdx = it.index();
            if (Gamma(currentVarIdx) != 1 && Gamma(currentVarIdx) != 0) {
                numVarsEliminated++;
                setGammaValue(coreStatus, currentVarIdx, VAR_ZERO, changes, ELIMINATION);
            }
        }
    }

    const bool constraintNotFullfillabel = ( numPosCoeff == 0 && rhs == 1 ) ||
                                           ( numNegCoeff == 0 && rhs == -1 ) ||
                                           ( numPosCoeff == 0 && numNegCoeff == 0 && rhs == 1 ) ||
                                           ( numPosCoeff == 0 && numNegCoeff == 0 && rhs == -1 ) || // <- should never happen
                                            rhs > 1 || rhs < -1;
    if (constraintNotFullfillabel) {
        return false;
    }

    const bool valueNeedsToBeZeroOrOne = (numPosCoeff == 0 && numNegCoeff == 1) ||
                                          (numPosCoeff == 1 && numNegCoeff == 0);
    if (valueNeedsToBeZeroOrOne) {
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator it(constrLHS,k); it; ++it) {
            const int currentVarIdx = it.index();
            if (Gamma(currentVarIdx) != 1 && Gamma(currentVarIdx) != 0) {
                if (rhs == 1 || rhs == -1) {
                    numVarsSet++;
                    setGammaValue(coreStatus, currentVarIdx, VAR_ONE, changes, ELIMINATION);
                    propagate(currentVarIdx, coreStatus, changes);
                }
                else if (rhs == 0) {
                    setGammaValue(coreStatus, currentVarIdx, VAR_ZERO, changes, ELIMINATION);
                    numVarsEliminated++;
                }
                else
                    ASSERT_NEVER_REACH;
            }
        }
    }
    return true;
}

/* function cleanUpVarsInGammaWithConstraints
   iterates through all constraints and checks if variables in gamma need to be set to zero or one
   => usually allows eleminating large amounts of constraints
   => part of the early error detection strategy
   returns true if Gamma is still feasible
 */
bool PrimalHeuristic::cleanUpVarsInGammaWithConstraints(CORE_STATUS &coreStatus, tsl::robin_set<Change> &changes, bool allConstraints, bool quiet) const {
    if(!opts.useCleanUp) {
        return true;
    }
    MatrixInt8 &Gamma = coreStatus.Gamma;
    int numVarsEliminated = 0;
    int numVarsSet = 0;
    int oldNumValsChanged = 0;
    const SparseMatInt8 constrLHS = constr.getConstraintMatrix();
    const SparseVecInt8 constrRHS = constr.getConstraintVector();

    int oldNumChangesInGamma = 1;
    int newNumChangesInGamma = changes.size();
    if (opts.verbose && !quiet) std::cout << "  Cleaning up Gamma..." << std::endl;
    bool gammaFeasible = true;
    do {
        // if allConstraints we iterate through all constraints, otherwise we only iterate through the constraints which were affected by the changes
        if (allConstraints) {
            for (int k = 0; k < constrLHS.outerSize(); k++) {
                gammaFeasible = doCleanUp(k, coreStatus, changes, numVarsEliminated, numVarsSet, constrLHS, constrRHS);
            }
        }
        else {
            for (const auto& k: getConstraintIdxAffectedByChanges(changes, oldNumChangesInGamma-1)) {
                gammaFeasible = doCleanUp(k, coreStatus, changes, numVarsEliminated, numVarsSet, constrLHS, constrRHS);
            }
        }
        oldNumChangesInGamma = newNumChangesInGamma;
        newNumChangesInGamma = changes.size();
        if (!isGammaFeasible(Gamma) || !gammaFeasible) {
            if (opts.verbose && !quiet) std::cout << "  Gamma is not feasible anymore. (Detected while cleaning up)" << std::endl;
            return false;
        }
    } while (oldNumChangesInGamma != newNumChangesInGamma);

    if (opts.verbose && !quiet) std::cout << "      -> " << numVarsEliminated << " Vars have been eliminated" << std::endl;
    if (opts.verbose && !quiet) std::cout << "      -> " << numVarsSet << " Vars have been set" << std::endl;
    return true;
}

/*
 apply the primal heuristic to binary vector Gamma which might not contain any flagged variables yet
 */
PRIMAL_HEURISTIC_RETURN_FLAG PrimalHeuristic::apply(MatrixInt8 &Gamma, bool gammaEmpty) {
    /* when calling this function we have 3 options
        1) Gamma is empty => we need to initialize it
        2) Gamma not empty but Initialization still in progress, but needed to update min marginals first
            => choose best init candidate
        3) Gamma not empty => apply primal heuristic
     */

    // Keep track of the changes in Gamma at each level
    std::vector<tsl::robin_set<Change>> changesInGamma; changesInGamma.reserve(1.2 * std::max(numFacesX, numFacesY));
    // Init coreStatus
    CORE_STATUS coreStatus(Gamma);
    coreStatus.candidates.reserve(0.01 * numProductFaces);
    coreStatus.idxVarOnes.reserve(1.4 * std::max(numFacesX, numFacesY));

    // lambda function we need for sorting later
    auto cprMinMargs = [this] (int i, int j) { return minMarginals(i) < minMarginals(j); };

    if (gammaEmpty || initializationInProgress) {
        if (opts.autoSetMaxIter) {
            opts.maxIter = 0.05 * std::min(numFacesX, numFacesY);
            if (opts.verbose) std::cout << "[PrimHeu]     Auto-setting max-iter to " << opts.maxIter << std::endl;
        }
        int initFlag = INIT_UNSURE;
        initializationInProgress = false;

        // make sure min marginals dont cause any problems
        if (!opts.useMinMarginals) {
            minMarginals.setZero();
        }

        if (gammaEmpty) {
            // set all to undecided
            Gamma = - Gamma.setOnes();
            initSet = initialization(coreStatus, changesInGamma);
            // exctract the init flag
            initFlag = initSet.back(); initSet.pop_back();
        }
        else {
            std::sort(initSet.begin(), initSet.end(), cprMinMargs);
            int numNegMinMargVarsInInitSet = 0; int idxNegMinMargVarInInitSet;
            int numZeroMinMargVarsInInitSet = 0;
            for (auto it = std::begin(initSet); it != std::end(initSet); ++it) {
                const int f = *it;
                if (minMarginals(f) < - FLOAT_EPSI){
                    numNegMinMargVarsInInitSet++;
                    idxNegMinMargVarInInitSet = f;
                }
                else if (minMarginals(f) < FLOAT_EPSI){
                    numZeroMinMargVarsInInitSet++;
                }
            }
            if (opts.verbose) std::cout << "[PrimHeu]  Num candidates with negative min marginals " <<  numNegMinMargVarsInInitSet << std::endl;
            if (opts.verbose) std::cout << "[PrimHeu]  Num candidates with zero min marginals " <<  numZeroMinMargVarsInInitSet << std::endl;
            if (numNegMinMargVarsInInitSet == 1) { // => otherwise init unsure
                initFlag = INIT_SURE;
                initSet.clear();
                initSet.push_back(idxNegMinMargVarInInitSet);
                Change chg = setNewSolVar(coreStatus, initSet[0]);
                tsl::robin_set<Change> changes; changes.insert(chg);
                propagate(chg.idx, coreStatus, changes);
                changesInGamma.push_back(changes);
            }
            else if (numNegMinMargVarsInInitSet > 1 || numZeroMinMargVarsInInitSet > 1) {
                for (auto it = initSet.begin(); it != initSet.end();) {
                    const int f = *it;
                    if (minMarginals(f) > FLOAT_EPSI)
                        initSet.erase(it);
                    else
                        ++it;
                }
            }
        }

        int numInitCandidates = initSet.size();

        // if we have an unsure initialization => try different initial values if there are any
        if (initFlag == INIT_UNSURE && numInitCandidates > 1) {
            std::cout << "[PrimHeu] Initialization unsure" << std::endl;
            std::cout << "[PrimHeu] Trying multiple init candidates and choosing best one from them" << std::endl;
            int oldMaxIter = opts.maxIter;
            opts.maxIter = 0.1 * std::min(numFacesX, numFacesY);
            int oldMaxVarOnes = opts.maxVarOnes;
            opts.maxVarOnes = 0.09 * std::min(numFacesX, numFacesY);

            int numThreads = opts.maxNumInitCandidates;
            if (numThreads > numInitCandidates)
                numThreads = numInitCandidates;
            int idxBestInit = 0;
            double meanEnergies[numThreads]; meanEnergies[idxBestInit] = INFINITY;
            int numOnes[numThreads]; numOnes[idxBestInit] = 0;
            int numNegMinMargs[numThreads]; numNegMinMargs[idxBestInit] = 0;
            float deltaFXvsFYmatched[numThreads]; deltaFXvsFYmatched[idxBestInit] = numFacesX + numFacesY;
            float meanNegMinMargSum[numThreads]; meanNegMinMargSum[idxBestInit] = 0;
            float meanPosMinMargSum[numThreads]; meanPosMinMargSum[idxBestInit] = INFINITY;
            bool oldVerboseState = opts.verbose;
            bool oldDebugOutput = opts.debugOutput;
            opts.verbose = false;
            opts.debugOutput = false;

            // Try multiple candidates and compare their mean energies after the first matchings
            #if defined(_OPENMP)
            #pragma omp parallel for
            #endif
            for (int thread = 0; thread < numThreads; thread++) {
                std::vector<tsl::robin_set<Change>> threadPrivateChanges;
                tsl::robin_set<Change> threadPrivateInitChanges;
                MatrixInt8 GammaThreadPrivate(numProductFaces, 1);
                GammaThreadPrivate = -GammaThreadPrivate.setOnes();
                CORE_STATUS coreStatusThreadPrivate(GammaThreadPrivate);
                coreStatusThreadPrivate.candidates.reserve(0.01 * numProductFaces);
                coreStatusThreadPrivate.idxVarOnes.reserve(0.2 * std::max(numFacesX, numFacesY));
                coreStatusThreadPrivate.id = thread;

                const int initIdx = initSet[thread];
                Change chg = setNewSolVar(coreStatusThreadPrivate, initIdx);
                addNewChangeToSetOfChanges(threadPrivateInitChanges, chg);
                threadPrivateChanges.push_back(threadPrivateInitChanges);
                propagate(initIdx, coreStatusThreadPrivate, threadPrivateChanges.back());
                core(coreStatusThreadPrivate, threadPrivateChanges);

                // extract energy and num negative min marginal variables
                double meanEnergy = 0; float negMMSum = 0, posMMSum = 0; int numFacesXMatched = 0, numFacesYMatched = 0;
                int localNumNegMinMargs = 0;
                for (tsl::robin_set<unsigned int>::iterator it = coreStatusThreadPrivate.idxVarOnes.begin(); it != coreStatusThreadPrivate.idxVarOnes.end(); ++it) {
                    meanEnergy += costs(*it);
                    if (minMarginals(*it) < -FLOAT_EPSI) {
                        localNumNegMinMargs++;
                        negMMSum += minMarginals(*it);
                    }
                    if (minMarginals(*it) > FLOAT_EPSI) {
                        posMMSum += minMarginals(*it);
                    }
                    const int fX =  utils::getFirstNonZeroIndexOfCol(piXCm, *it);
                    numFacesXMatched += fX == -1 ? 0 : 1;
                    const int fY =  utils::getFirstNonZeroIndexOfCol(piYCm, *it);
                    numFacesYMatched += fY == -1 ? 0 : 1;
                }
                const int numVarOnes = coreStatusThreadPrivate.idxVarOnes.size();
                meanEnergy /= numVarOnes;
                negMMSum /= numVarOnes;
                posMMSum /= numVarOnes;
                const float delta = std::abs( numFacesXMatched/ (float)numFacesX - numFacesYMatched / (float)numFacesY);

                #if defined(_OPENMP)
                #pragma omp critical
                #endif
                {
                    
                    if (localNumNegMinMargs >= numNegMinMargs[idxBestInit] &&
                        negMMSum            <= meanNegMinMargSum[idxBestInit] &&
                        posMMSum            <= meanPosMinMargSum[idxBestInit] &&
                        delta               <= deltaFXvsFYmatched[idxBestInit] &&
                        //numVarOnes          >= numOnes[idxBestInit] &&
                        //meanEnergy          <= meanEnergies[idxBestInit] &&
                        numVarOnes          >= opts.maxVarOnes
                        ) {
                        idxBestInit = thread;
                        Gamma = GammaThreadPrivate;
                        
                    }
                    meanNegMinMargSum[thread] = negMMSum;
                    meanPosMinMargSum[thread] = posMMSum;
                    numNegMinMargs[thread] = localNumNegMinMargs;
                    numOnes[thread] = numVarOnes;
                    deltaFXvsFYmatched[thread] = delta;
                    meanEnergies[thread] = meanEnergy;

                    if (oldVerboseState) std::cout << "[PrimHeu] Trying init candidate " << thread+1 << " of " << numThreads << std::endl;
                    if (oldDebugOutput) {
                        std::cout << "   Mean Energy = " << meanEnergy << std::endl;
                        std::cout << "   Num Vars with neg Min Marginals = " << localNumNegMinMargs << std::endl;
                        std::cout << "   Num Var Ones =" << coreStatusThreadPrivate.idxVarOnes.size() << std::endl;
                        std::cout << "   Mean negative MM sum = " << negMMSum << std::endl;
                        std::cout << "   Mean positive MM sum = " << posMMSum << std::endl;
                        std::cout << "   Delta % Matches X Y  = " << delta << std::endl;
                    }
                } // end #pragma omp critical
            } // end #pragma omp parallel for
            std::cout << "[PrimHeu] Best init Candidate is " << idxBestInit+1 << std::endl;
            opts.verbose = oldVerboseState;
            opts.debugOutput = oldDebugOutput;
            opts.maxIter = oldMaxIter;
            opts.maxVarOnes = oldMaxVarOnes;
            if (opts.useMinMarginals) {
                std::cout << "[PrimHeu] Updating min marginals for this initialization => solve dual problem" << std::endl;
                return SOL_INCOMPLETE;
            }
        }
        return apply(Gamma, false);
    }

    // this is the case when we have a non-empty gamma and no initialization in progress anymore
    else {
        if (opts.autoSetMaxIter) {
            opts.maxIter = 0.2 * std::min(numFacesX, numFacesY);
            if (opts.verbose) std::cout << "[PrimHeu]     Auto-setting max-iter to " << opts.maxIter << std::endl;
        }
        tsl::robin_set<Change> currentChanges;
        bool success = true;
        // restore coreStatus
        for (int f = 0; f < numProductFaces; f++) {
            if (Gamma(f) == VAR_ONE) {
                coreStatus.idxVarOnes.insert(f);
            }
            if (Gamma(f) > VAR_ONE) {
                coreStatus.candidates.insert(Candidate(f, Gamma(f)));
            }
        }
        countCandidates(coreStatus);
        // propagate already existing changes in gamma
        for (tsl::robin_set<unsigned int>::iterator it = coreStatus.idxVarOnes.begin(); it != coreStatus.idxVarOnes.end(); ++it) {
            success &= propagate(*it, coreStatus, currentChanges);
        }
        if (!success && !cleanUpVarsInGammaWithConstraints(coreStatus, currentChanges, false, !opts.debugOutput)) {
            std::cout << "[PrimHeu] Non-Empty Gamma vector contains conflicting values. Aborting" << std::endl;
            ASSERT_NEVER_REACH;
            exit(0);
        }
        changesInGamma.push_back(currentChanges);
        return core(coreStatus, changesInGamma);
    }
}


void removeFlagsFromGamma(MatrixInt8& GammaBar, tsl::robin_set<unsigned int> &idxVarOnes) {
    GammaBar.setZero();
    for (tsl::robin_set<unsigned int>::iterator it = idxVarOnes.begin(); it != idxVarOnes.end(); ++it) {
        GammaBar((*it)) = 1;
    }
}


/*  core of the primal heuristic
    sets/propagates/undos changes as long as no stopping criterion is fulfilled
 
 */
PRIMAL_HEURISTIC_RETURN_FLAG PrimalHeuristic::core(CORE_STATUS &coreStatus,
                                                   std::vector<tsl::robin_set<Change>> &changesInGamma) const {
    bool verbose = true;

    // references, so we dont need to change to much code
    size_t &numVarSolCandidate = coreStatus.numVarSolCandidate = 0;
    size_t &numVarLikelySolCandidate = coreStatus.numVarLikelySolCandidate = 0;
    size_t &numVarMustBeSol = coreStatus.numVarMustBeSol = 0;
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    tsl::robin_set<unsigned int> &idxVarOnes = coreStatus.idxVarOnes;
    MatrixInt8 &Gamma = coreStatus.Gamma;
    if (coreStatus.id != -1) {
        coreStatus.outputPrefix = "[PrimHeu " + std::to_string(coreStatus.id) + "] ";
    }
    const std::string outputPrefix = coreStatus.outputPrefix;
    if(opts.verbose) std::cout << outputPrefix << "Assembling Primal Solution" << std::endl;

    assert(Gamma.cols() == 1 && Gamma.rows() == numProductFaces);

    
    /*
     Iterate
     */
    int currentDepth = 0;
    MatrixInt8 GammaBar(Gamma.rows(), 1);
    GammaBar.setZero();
    Eigen::MatrixXi variableCandidates;
    bool newMatching;
    int iter = 0;
    int numBackTracks = 0;
    Change previousOne;
    std::vector<int> idxDecisions; idxDecisions.reserve(opts.maxIter);
    std::vector<std::vector<int>> idxBadDecisions; idxBadDecisions.reserve(opts.maxBacktracks);

    // vector to keep track of changes at current level
    tsl::robin_set<Change> changesAtCurrentDepth; changesAtCurrentDepth.reserve(0.3 * numProductFaces);

    std::vector<int> idxBadDecisionsCurrentDepth; idxBadDecisionsCurrentDepth.reserve(300);

    
    while (!constraintsSatisfied(GammaBar.sparseView(), delPlusRm, delMinusRm, piXRm, piYRm)) {

        if (changesInGamma.size() > opts.maxBacktracks + 1) {
            changesInGamma.erase(changesInGamma.begin());
        }
        if (idxBadDecisions.size() > opts.maxBacktracks + 1) {
            idxBadDecisions.erase(idxBadDecisions.begin());
        }
        
        newMatching = false;
        
        // find new variable candidates
        getVariabelCandidatesInAscendingOrder(coreStatus, variableCandidates, costs);
        if (opts.verbose) std::cout << outputPrefix << "Depth: " << currentDepth << " (ITER: "<< iter << ", #ONE: " << idxVarOnes.size() << ")" <<std::endl;
        if (opts.debugOutput) {
            const int numVarZero = numProductFaces - (Gamma.array() == 0).count();
            const int numVarOne = idxVarOnes.size();
            const int numVarUndecided = numProductFaces - numVarZero - numVarOne - numVarSolCandidate - numVarLikelySolCandidate - numVarMustBeSol;
            std::cout << "  # VAR_ONE                      : " << numVarOne <<std::endl;
            std::cout << "  # VAR_ZERO                     : " << numVarZero << std::endl;
            std::cout << "  # VAR_UNDECIDED                : " << numVarUndecided << std::endl;
            std::cout << "  # VAR_SOL_CANDIDATE            : " << numVarSolCandidate << std::endl;
            std::cout << "  # VAR_LIKELY_SOL_CANDIDATE     : " << numVarLikelySolCandidate << std::endl;
            std::cout << "  # VAR_MUST_BE_SOL              : " << numVarMustBeSol << std::endl;
            std::cout << "  Constraints satisfied          : " << constraintsSatisfied(GammaBar.sparseView(), delPlusRm, delMinusRm, piXRm, piYRm) << std::endl;
            std::cout << "  Gamma feasibel                 : " << isGammaFeasible(Gamma) << std::endl;
        }
        
        Change nextOne;

        // ascending order of variable candidates
        if (candidates.size() > 0) {
        for (int i = 0; i < variableCandidates.size(); i++) {

            if (opts.precheckVarCandidates) {
                bool needToBackTrack = false;
                while (!preCheckSettingVar(coreStatus, variableCandidates(i), VAR_ONE)) {
                    if (opts.debugOutput) std::cout << "  Prechecking detected unfeasible variable candidate => setting it to zero" << std::endl;
                    if (!setGammaValue(coreStatus, variableCandidates(i), VAR_ZERO, changesAtCurrentDepth, SOL_CANDIDATE)) {
                        if (opts.debugOutput) std::cout << "  Cannot set variable candidate to zero or to one => need to backtrack" << std::endl;
                        needToBackTrack = true;
                        break;
                    }
                    i++;
                    if ( i == variableCandidates.size() ) {
                        if (opts.verbose) std::cout << outputPrefix << "  Didnt find a feasible varibale candidate => need to backtrack" << std::endl;
                        needToBackTrack = true;
                        break;
                    }
                }
                if (needToBackTrack) break; // => exit for-loop
            }
            
            // set next element to one
            nextOne = setNewSolVar(coreStatus, variableCandidates(i));
            addNewChangeToSetOfChanges(changesAtCurrentDepth, nextOne);
            idxDecisions.push_back(variableCandidates(i));
            
            // propagate
            bool noErrorWhilePropagation =
            propagate(nextOne.idx,
                      coreStatus,
                      changesAtCurrentDepth);

            if (nextOne.oldValue == VAR_MUST_BE_SOL && !noErrorWhilePropagation) {
                if (opts.debugOutput) std::cout << outputPrefix << "Value that needs to be in solution yields conflicts -> Backtrack" << std::endl;
                break;
            }
            
            // check feasibility
            bool goLevelDeeper = false;
            if (noErrorWhilePropagation) {
                goLevelDeeper = isGammaFeasible(Gamma);
                // clean up gamma to eleminate as many variables as possible and to detect errors early
                if (goLevelDeeper && opts.cleanUpGammAfterEachIteration) {
                    if (!cleanUpVarsInGammaWithConstraints(coreStatus, changesAtCurrentDepth, false, !opts.debugOutput)) {
                        // if we end up here then Gamma was not feasible and we have to backtrack
                        goLevelDeeper = false;
                    }
                }
            }


            if (goLevelDeeper) {
                currentDepth = currentDepth + 1;
                if (opts.debugOutput) { std::cout << "     New VAR_ONE idx: " << nextOne.idx << " (" << i+1 << "-th candidate";
                    if (minMarginals(nextOne.idx) > -FLOAT_EPSI) {
                        std::cout << ", minMarginal= " << minMarginals(nextOne.idx) << ")" << std::endl;
                    }
                    else
                        std::cout << ")" << std::endl;
                }
                newMatching = true;
                changesInGamma.push_back(changesAtCurrentDepth);
                changesAtCurrentDepth.clear();
                previousOne = nextOne;
                const int oldSIze = idxBadDecisions.size();
                idxBadDecisions.push_back(idxBadDecisionsCurrentDepth);
                break; // -> exit for loop and go a level deeper
            }
            else {
                if (opts.debugOutput) std::cout << "     No success with " << i+1 << "-th candidate" << std::endl;
                // undo the changes of the current level and set the element to zero which led to a non-feasible solution
                undoChanges(coreStatus, changesAtCurrentDepth);
                if (i < variableCandidates.size() - 1) {
                    idxBadDecisionsCurrentDepth.push_back(idxDecisions.back());
                    idxDecisions.pop_back();
                    // add all previously problematic variables back to Gamma
                    for (auto idx : idxBadDecisionsCurrentDepth) {
                        setGammaValue(coreStatus, idx, VAR_ZERO, changesAtCurrentDepth, BACKTRACK);
                    }
                }

            }
        }
        } // candidates.size() > 0

        idxBadDecisionsCurrentDepth.clear(); idxBadDecisionsCurrentDepth.reserve(candidates.size());
        
        if (!newMatching) {
            if (opts.verbose) std::cout << outputPrefix << "Backtrack" << std::endl;
            numBackTracks++;
            // go one level up
            currentDepth = currentDepth - 1;
            if (currentDepth <= 0) {
                if (opts.verbose) std::cout << outputPrefix << "Tried all remaining candidates. Gamma infeasible. Aborting..." << std::endl;
                removeFlagsFromGamma(Gamma, idxVarOnes);
                return SOL_INFEASIBLE;
            }
            else if (changesInGamma.size() == 0) {
                if (opts.verbose) std::cout << outputPrefix << "No more changes to backtrack" << std::endl;
                removeFlagsFromGamma(Gamma, idxVarOnes);
                return SOL_INFEASIBLE;
            }
            else {
                if (changesAtCurrentDepth.size() > 0) {
                    undoChanges(coreStatus, changesAtCurrentDepth);
                }
                // undo changes of upper level
                changesAtCurrentDepth = changesInGamma.back();
                changesInGamma.pop_back();
                if (changesAtCurrentDepth.size() > 0) {
                    undoChanges(coreStatus, changesAtCurrentDepth);
                }

                // restore all problematic values which were set to zero of last level
                idxBadDecisionsCurrentDepth = idxBadDecisions.back();
                idxBadDecisions.pop_back();
                for (auto idx : idxBadDecisionsCurrentDepth) {
                    setGammaValue(coreStatus, idx, VAR_ZERO, changesAtCurrentDepth, BACKTRACK);
                }
                // set level up value which was set to one to zero since it caused infeasibility
                setGammaValue(coreStatus, idxDecisions.back(), VAR_ZERO, changesAtCurrentDepth, BACKTRACK);
                idxBadDecisionsCurrentDepth.push_back(idxDecisions.back());
                idxDecisions.pop_back();
            }
        }
        
        // extract Gamma Bar
        removeFlagsFromGamma(GammaBar, idxVarOnes);

        
        iter++;
        if (iter > opts.maxIter || numBackTracks >= opts.maxBacktracks || idxVarOnes.size() >= opts.maxVarOnes) {
            if (opts.verbose) {
                std::cout << outputPrefix << "Aborting solving, returning unfinished solution";
                if (iter > opts.maxIter)
                    std::cout << " (maxIter reached)" << std::endl;
                if (numBackTracks > opts.maxBacktracks)
                    std::cout << " (maxBacktracks reached)" << std::endl;
                if (idxVarOnes.size() >= opts.maxVarOnes)
                    std::cout << " (maxVarOnes reached)" << std::endl;
            }

            // make sure we return a feasible gamma (only necessary if we didn't clean up after each iteration)
            if (opts.verbose) std::cout << outputPrefix << "Cleaning up Gamma and ensuring Gamma is feasible" << std::endl;
            tsl::robin_set<Change> changes; changes.reserve(candidates.size());
            while (true) {
                if (changesInGamma.size() == 0) {
                    if (opts.verbose) std::cout << outputPrefix << "Couldn't assemble feasible Gamma." << std::endl;
                    break;
                }
                if (!cleanUpVarsInGammaWithConstraints(coreStatus, changes, true, false)) {
                    if (opts.verbose) std::cout << outputPrefix << "  Removing changes of last level" << std::endl;
                    // this undos the changes we made while cleaning up
                    undoChanges(coreStatus, changes);
                    // this undos the changes we made at the last depth (pops from back of changesInGamma)
                    if (changesInGamma.size() > 0)
                        undoChangesAtLastLevel(coreStatus, changesInGamma);
                    else
                        break;
                }
                else {
                    break;
                }
            }
            if (opts.verbose) std::cout << outputPrefix << "Done" << std::endl;

            // extract once again GammaBar
            removeFlagsFromGamma(GammaBar, idxVarOnes);
            if (constraintsSatisfied(GammaBar.sparseView(), delPlusRm, delMinusRm, piXRm, piYRm)) {
                std::cout << outputPrefix << "Successfully completed solution" << std::endl;
                Gamma = GammaBar;
                return SOL_COMPLETE;
            }
            else if (candidates.size() == 0) {
                // we cant make a valid GAMMA anymore
                ASSERT_NEVER_REACH;
            }
            if (opts.verbose) std::cout << outputPrefix << "Solution not successfully completed yet" << std::endl;
            if (opts.debugOutput) std::cout << "# VAR_ONE = " << idxVarOnes.size() << std::endl;
            return SOL_INCOMPLETE;
        }
    }
    
    std::cout << outputPrefix << "Successfully completed solution" << std::endl;
    
    Gamma = GammaBar;
    return SOL_COMPLETE;
    
}
