//
//  primalHeuristicInitialization.cpp
//  shape-matching-dd
//
//  Created by Paul Rötzer on 30.09.21.
//

#include <stdio.h>
#include <iostream>
#include <igl/sort.h>
#include "shapeMatchModel.hpp"
#include "helper/utils.hpp"
#include "primalHeuristic.hpp"
#include <igl/principal_curvature.h>
#include <igl/gaussian_curvature.h>

Eigen::MatrixXi PrimalHeuristic::getNeighboursIdx(int f) const {
    Eigen::MatrixXi connectingEdgeIdxs(3, 1);
    return getNeighboursIdx(f, connectingEdgeIdxs);
}

Eigen::MatrixXi PrimalHeuristic::getNeighboursIdx(int f, Eigen::MatrixXi &connectingEdgeIdxs) const {
    assert(connectingEdgeIdxs.rows() == 3 && connectingEdgeIdxs.cols() == 1);
    const int numFacesX = piXRm.rows();
    const int numFacesY = piYRm.rows();
    assert(f < 3 * numFacesX * numFacesY); // f can only be a triangle to triangle product face
    const int numFacesXxY = numFacesX * numFacesY;

    // f is the current product face

    // extract to f corresponding faces in X and Y with help of projection matrices
    const int fX = utils::getFirstNonZeroIndexOfCol(piXCm, f);
    const int fY = utils::getFirstNonZeroIndexOfCol(piYCm, f);

    const int neighborsfX[] = { triangleNeighboursX(fX, 0),
                                triangleNeighboursX(fX, 1),
                                triangleNeighboursX(fX, 2)};
    const int neighborsfY[] = { triangleNeighboursY(fY, 0),
                                triangleNeighboursY(fY, 1),
                                triangleNeighboursY(fY, 2)};

    int numTri2EdgNeiFound[] = {0, 0, 0};

    int numFoundNeighbours = 0;

    Eigen::MatrixXi neighbours(3, 5); neighbours = -neighbours.setOnes();

    // search for neighbours of product edge
    for (typename SparseMatInt8Cm::InnerIterator it(delPlusCm, f); it; ++it) {
        for (typename SparseMatInt8Rm::InnerIterator iit(delMinusRm, it.index()); iit; ++iit) {
            const int potentialNeighbour = iit.index();
            const int potentialNeighProjX =
                    utils::getFirstNonZeroIndexOfCol(piXCm, potentialNeighbour);
            const int potentialNeighProjY =
                    utils::getFirstNonZeroIndexOfCol(piYCm, potentialNeighbour);

            for (int i = 0; i < 3; i++) {
                if (potentialNeighProjX == neighborsfX[i]) {
                    if (utils::isIn(potentialNeighProjY, neighborsfY, 3)) {
                        numFoundNeighbours++;
                        neighbours(i, 0) = potentialNeighbour;
                    }
                    else {
                        neighbours(i, ++numTri2EdgNeiFound[i]) = potentialNeighbour;
                    }
                }
                if (potentialNeighProjY == neighborsfY[i]) {
                    if (utils::isIn(potentialNeighProjX, neighborsfX, 3)) {
                        numFoundNeighbours++;
                        neighbours(i, 0) = potentialNeighbour;
                    }
                    else {
                        neighbours(i, ++numTri2EdgNeiFound[i]) = potentialNeighbour;
                    }
                }
                if (numTri2EdgNeiFound[i] > 4) std::cout << "[PrimHeu] ERROR: Added more tri2edge neighbours than expected" << std::endl;
            }
        }

    }

    for (typename SparseMatInt8Cm::InnerIterator it(delMinusCm, f); it; ++it) {
        for (typename SparseMatInt8Rm::InnerIterator iit(delPlusRm, it.index()); iit; ++iit) {
            const int potentialNeighbour = iit.index();
            const int potentialNeighProjX =
                    utils::getFirstNonZeroIndexOfCol(piXCm, potentialNeighbour);
            const int potentialNeighProjY =
                    utils::getFirstNonZeroIndexOfCol(piYCm, potentialNeighbour);

            for (int i = 0; i < 3; i++) {
                if (potentialNeighProjX == neighborsfX[i]) {
                    if (utils::isIn(potentialNeighProjY, neighborsfY, 3)) {
                        neighbours(i, 0) = potentialNeighbour;
                    }
                    else {
                        neighbours(i, ++numTri2EdgNeiFound[i]) = potentialNeighbour;
                    }
                }
                if (potentialNeighProjY == neighborsfY[i]) {
                    if (utils::isIn(potentialNeighProjX, neighborsfX, 3)) {
                        neighbours(i, 0) = potentialNeighbour;
                    }
                    else {
                        neighbours(i, ++numTri2EdgNeiFound[i]) = potentialNeighbour;
                    }
                }
                if (numTri2EdgNeiFound[i] > 4) std::cout << "[PrimHeu] ERROR: Added more tri2edge neighbours than expected" << std::endl;
            }

        }

    }
    return neighbours;//.block(0, 0, 3, 1);
}

tsl::robin_set<int> PrimalHeuristic::getIdxAllNeighborsOfProductTriangle(int f) const {
    tsl::robin_set<int> idxAllNeighbors; idxAllNeighbors.reserve(60);
    for (typename Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(delPlusCm, f); it; ++it) {
        const int row = it.row();
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator iit(delMinusRm, row); iit; ++iit) {
            idxAllNeighbors.insert(iit.col());
        }
    }
    for (typename Eigen::SparseMatrix<int8_t, Eigen::ColMajor>::InnerIterator it(delMinusCm, f); it; ++it) {
        const int row = it.row();
        for (typename Eigen::SparseMatrix<int8_t, Eigen::RowMajor>::InnerIterator iit(delPlusRm, row); iit; ++iit) {
            idxAllNeighbors.insert(iit.col());
        }
    }
    return idxAllNeighbors;
}


std::vector<int> PrimalHeuristic::initialization(CORE_STATUS &coreStatus,
                                     std::vector<tsl::robin_set<Change>> &changesInGamma) const {

    size_t &numVarSolCandidate = coreStatus.numVarSolCandidate = 0;
    size_t &numVarLikelySolCandidate = coreStatus.numVarLikelySolCandidate = 0;
    size_t &numVarMustBeSol = coreStatus.numVarMustBeSol = 0;
    tsl::robin_set<Candidate> &candidates = coreStatus.candidates;
    tsl::robin_set<unsigned int> &idxVarOnes = coreStatus.idxVarOnes;
    MatrixInt8 &Gamma = coreStatus.Gamma;
    
    std::vector<int> initCandidatesAscending;
    if (opts.verbose) std::cout << "[PrimHeu] Inititalization" << std::endl;
    const int row = 0;
    const int numFacesXxY = numFacesX * numFacesY;
    const int idxStartTriYToVX = 3 * numFacesX * numFacesY;
    const int numTriYToVX = numFacesY * numVerticesX;
    const int idxStartTriXToVY = idxStartTriYToVX + numTriYToVX + 3 * 2 * numEdgesX * numFacesY;
    const int numTriXToVY = numFacesX * numVerticesY;

    tsl::robin_set<Change> changesAtCurrentDepth;
    // we want to find a 4-non-degenerate-triangle patch with the lowest energy
    // => e.g. if Y has less triangles it is more likely that we do not
    //    have degenerate triangles in shape Y
    Eigen::MatrixXf patchEnergies(3 * numFacesXxY, 1);
    patchEnergies.setZero();

    // sort lambda ascending according to min marginals
    auto cprMinMargs = [this] (int i, int j) { return minMarginals(i) < minMarginals(j); };

    std::vector<int> solPatches; solPatches.reserve(0.1 * numFacesX);
    float smallestMinMargPatchSum = 0;
    int idxSmallestMinMargPatch = -1; int idxSmallestNeighbors[] = {0, 0, 0};
    float smallestMinMarg = std::numeric_limits<float>::max();;
    int idxSmallestMinMarg = -1;


    bool zeroEnergySolutionPatch = false;

    if (opts.verbose) std::cout << "[PrimHeu]   > Trying to find zero energy patch" << std::endl;
    for (int f = 0; f < 3 * numFacesXxY; f++) {
        // add cost of current product face
        patchEnergies(f) = costs(f);

        // get patch
        Eigen::MatrixXi neighbours = getNeighboursIdx(f);

        patchEnergies(f) += costs(neighbours(0));
        patchEnergies(f) += costs(neighbours(1));
        patchEnergies(f) += costs(neighbours(2));

        if (patchEnergies(f) < FLOAT_EPSI) {
            zeroEnergySolutionPatch = true;
        }

        // are all possible matchings of the patch marked as a solution?
        if (minMarginals(f) < -FLOAT_EPSI) {
            float minMargsSum = minMarginals(f);
            int idxSmallestMMNeighbors[] = {-1, -1, -1};
            for (int i = 0; i < 3; i++) {
                float smallestMMNeighbour = -FLOAT_EPSI;
                for (int j = 0; j < 5; j++) {
                    if (minMarginals(neighbours(i, j)) < smallestMMNeighbour) {
                        smallestMMNeighbour = minMarginals(neighbours(i, j));
                        idxSmallestMMNeighbors[i] = j;
                    }
                }
                if (idxSmallestMMNeighbors[i] == -1) {
                    minMargsSum = 0;
                    break;
                }
                else {
                    minMargsSum += minMarginals(idxSmallestMMNeighbors[i]);
                }
            }
            if (minMargsSum < 0) {
                solPatches.push_back(f);
                if (minMargsSum < smallestMinMargPatchSum) {
                    smallestMinMargPatchSum = smallestMinMargPatchSum;
                    idxSmallestMinMargPatch = f;
                    for (int i = 0; i < 3; i++) idxSmallestNeighbors[i] = idxSmallestMMNeighbors[i];
                }
            }
        }
        if (minMarginals(f) < smallestMinMarg) {
            smallestMinMarg = minMarginals(f);
            idxSmallestMinMarg = f;
        }
    }

    const int numSolPatches = solPatches.size();

    Eigen::MatrixXf sortedPatchEnergies;
    Eigen::MatrixXi idxInPatchEnergies;
    if ( numSolPatches!= 0) {
        if (opts.verbose) std::cout << "[PrimHeu]   # Patches with all negative minMarginals: " << numSolPatches << std::endl;
        Eigen::MatrixXf solPatchEnergies(numSolPatches, 1);
        for (int i = 0; i < numSolPatches; i++) {
            solPatchEnergies(i) = patchEnergies(solPatches[i]);
        }
        patchEnergies.resize(numSolPatches, 1);
        patchEnergies = solPatchEnergies;
    }
    igl::sort(patchEnergies,
              1,    // sort each column
              true, // -> ascending order
              sortedPatchEnergies, // => TODO: find out if inplace capable
              idxInPatchEnergies);

    if (numSolPatches > 1) {
        const float relDiffSmallestPatchEnergies = (sortedPatchEnergies(1) - sortedPatchEnergies(0)) / sortedPatchEnergies(0);
        if (relDiffSmallestPatchEnergies > 1) {
            if (opts.verbose) std::cout << "[PrimHeu]   No Zero Energy Patch found but smallest patch energy significant" << std::endl;
            zeroEnergySolutionPatch = true;
        }
    }

    if (!zeroEnergySolutionPatch && opts.useMinMarginals) {
        if (opts.verbose) std::cout << "[PrimHeu]   No Zero Energy Patch found" << std::endl;
        if (numSolPatches <= 1) {
            if (opts.verbose) {
                std::cout << "[PrimHeu]   No Patch found which minMarginals recommend" << std::endl;
                std::cout << "[PrimHeu]   > Searching in set of 3-triangle patches " << std::endl;
            }

            /*
             patch with only 3 triangles
             */
            smallestMinMargPatchSum = 0;
            idxSmallestMinMarg = -1;
            for (int f = 0; f < 3 * numFacesXxY; f++) {

                // get patch
                Eigen::MatrixXi neighbours = getNeighboursIdx(f);

                // are all possible matchings of the patch marked as a solution?
                int numNonNegativeNeighbors = 0;
                if (minMarginals(f) < -FLOAT_EPSI) {
                    float minMargsSum = minMarginals(f);
                    int idxSmallestMMNeighbors[] = {-1, -1, -1};
                    for (int i = 0; i < 3; i++) {
                        float smallestMMNeighbour = -FLOAT_EPSI;
                        for (int j = 0; j < 5; j++) {
                            if (minMarginals(neighbours(i, j)) < smallestMMNeighbour) {
                                smallestMMNeighbour = minMarginals(neighbours(i, j));
                                idxSmallestMMNeighbors[i] = j;
                            }
                        }
                        if (idxSmallestMMNeighbors[i] == -1) {
                            numNonNegativeNeighbors++;
                            if (numNonNegativeNeighbors > 1) {
                                minMargsSum = 0;
                                break;
                            }
                        }
                        else {
                            minMargsSum += minMarginals(idxSmallestMMNeighbors[i]);
                        }
                    }
                    if (minMargsSum < 0) {
                        solPatches.push_back(f);
                        if (minMargsSum < smallestMinMargPatchSum) {
                            smallestMinMargPatchSum = smallestMinMargPatchSum;
                            idxSmallestMinMargPatch = f;
                            for (int i = 0; i < 3; i++) idxSmallestNeighbors[i] = idxSmallestMMNeighbors[i];
                        }
                    }
                }
            }
            if (solPatches.size() > 0) {
                if (opts.verbose) std::cout << "[PrimHeu]   Found initial, 3-triangle patch with minMarginals" << std::endl;
                Change change = setNewSolVar(coreStatus, idxSmallestMinMargPatch);
                addNewChangeToSetOfChanges(changesAtCurrentDepth, change);
                propagate(change.idx, coreStatus, changesAtCurrentDepth);
                changesInGamma.push_back(changesAtCurrentDepth);
                if (opts.verbose) std::cout << "[PrimHeu]   Done" << std::endl;
                initCandidatesAscending.push_back(idxSmallestMinMargPatch);
                initCandidatesAscending.push_back(INIT_SURE);
                return initCandidatesAscending;
            }

            if (opts.verbose) {
                std::cout << "[PrimHeu]   No 3-triangle Patch found which minMarginals recommend" << std::endl;
                std::cout << "[PrimHeu]   > Searching in set of 2-triangle patches " << std::endl;
            }
            /*
             patch with only 2 triangles
             */
            smallestMinMargPatchSum = 0;
            idxSmallestMinMarg = -1;
            for (int f = 0; f < 3 * numFacesXxY; f++) {

                // get patch
                Eigen::MatrixXi neighbours = getNeighboursIdx(f);

                // are all possible matchings of the patch marked as a solution?
                int numNonNegativeNeighbors = 0;
                if (minMarginals(f) < -FLOAT_EPSI) {
                    float minMargsSum = minMarginals(f);
                    int idxSmallestMMNeighbors[] = {-1, -1, -1};
                    for (int i = 0; i < 3; i++) {
                        float smallestMMNeighbour = -FLOAT_EPSI;
                        for (int j = 0; j < 5; j++) {
                            if (minMarginals(neighbours(i, j)) < smallestMMNeighbour) {
                                smallestMMNeighbour = minMarginals(neighbours(i, j));
                                idxSmallestMMNeighbors[i] = j;
                            }
                        }
                        if (idxSmallestMMNeighbors[i] == -1) {
                            numNonNegativeNeighbors++;
                            if (numNonNegativeNeighbors > 2) {
                                minMargsSum = 0;
                                break;
                            }
                        }
                        else {
                            minMargsSum += minMarginals(idxSmallestMMNeighbors[i]);
                        }
                    }
                    if (minMargsSum < 0) {
                        solPatches.push_back(f);
                        if (minMargsSum < smallestMinMargPatchSum) {
                            smallestMinMargPatchSum = smallestMinMargPatchSum;
                            idxSmallestMinMargPatch = f;
                            for (int i = 0; i < 3; i++) idxSmallestNeighbors[i] = idxSmallestMMNeighbors[i];
                        }
                    }
                }
            }
            if (solPatches.size() > 0) {
                if (opts.verbose) std::cout << "[PrimHeu]   Found initial, 2-triangle patch with minMarginals" << std::endl;
                Change change = setNewSolVar(coreStatus, idxSmallestMinMargPatch);
                addNewChangeToSetOfChanges(changesAtCurrentDepth, change);
                propagate(change.idx, coreStatus, changesAtCurrentDepth);
                changesInGamma.push_back(changesAtCurrentDepth);
                if (opts.verbose) std::cout << "[PrimHeu]   Done" << std::endl;
                initCandidatesAscending.push_back(idxSmallestMinMargPatch);
                initCandidatesAscending.push_back(INIT_SURE);
                return initCandidatesAscending;
            }
            if (opts.verbose) {
                std::cout << "[PrimHeu]   No 2-triangle Patch found which minMarginals recommend" << std::endl;
                std::cout << "[PrimHeu]   > Searching for product face with most negative min margs neighbors " << std::endl;
            }
            int idxProductFaceMostNegativeNeighbors = -1; int maxNumNegativeNeighbors = 0; float minMM = -FLOAT_EPSI;
            int oldNumMaxNeighbors = 0;
            std::vector<int> allProductFacesHavingMinMargs;
            for (int f = 0; f < minMarginals.rows(); f++) {
                // we dont want to init with tri to point matchings => skip those indices
                if (f == idxStartTriYToVX && opts.avoidTriToVertForInit) f += numTriYToVX;
                if (f == idxStartTriXToVY && opts.avoidTriToVertForInit) f += numTriXToVY;

                if (minMarginals(f) < -FLOAT_EPSI) {
                    allProductFacesHavingMinMargs.push_back(f);
                    float minMargsSum = minMarginals(f);
                    int numNegativeNeighbors = 0;
                    for (const auto& k: getIdxAllNeighborsOfProductTriangle(f)) {
                        if (minMarginals(k) < -FLOAT_EPSI) {
                            numNegativeNeighbors++;
                            minMargsSum += minMarginals(k);
                        }
                    }
                    if (numNegativeNeighbors > maxNumNegativeNeighbors) {
                        initCandidatesAscending.clear();
                    }
                    if (numNegativeNeighbors >= oldNumMaxNeighbors) {
                        initCandidatesAscending.push_back(f);
                    }
                    if (numNegativeNeighbors >= maxNumNegativeNeighbors && minMargsSum < minMM) {
                        oldNumMaxNeighbors = maxNumNegativeNeighbors;
                        maxNumNegativeNeighbors = numNegativeNeighbors;
                        idxProductFaceMostNegativeNeighbors = f;
                        minMM = minMarginals(f);
                    }
                }
            }
            if (idxProductFaceMostNegativeNeighbors != -1) {
                Change change = setNewSolVar(coreStatus, idxProductFaceMostNegativeNeighbors);
                addNewChangeToSetOfChanges(changesAtCurrentDepth, change);
                changesInGamma.push_back(changesAtCurrentDepth);
                bool  propagtionSuccess = propagate(idxProductFaceMostNegativeNeighbors, coreStatus, changesAtCurrentDepth);
                if (propagtionSuccess) {
                    // lets hope this never happes
                    ASSERT_NEVER_REACH;
                }
                if (opts.verbose) std::cout << "[PrimHeu]   Done" << std::endl;
                std::sort(initCandidatesAscending.begin(), initCandidatesAscending.end(), cprMinMargs);
                std::sort(allProductFacesHavingMinMargs.begin(), allProductFacesHavingMinMargs.end(), cprMinMargs);
                // Remove duplicates
                for(auto it = std::begin(initCandidatesAscending); it != std::end(initCandidatesAscending); ++it) {
                    const int currentIdx = *it;
                    auto iterFound = std::find(allProductFacesHavingMinMargs.begin(), allProductFacesHavingMinMargs.end(), currentIdx);
                    if ( iterFound != allProductFacesHavingMinMargs.end()) {
                        allProductFacesHavingMinMargs.erase(iterFound);
                    }
                }
                initCandidatesAscending.insert(initCandidatesAscending.end(), allProductFacesHavingMinMargs.begin(), allProductFacesHavingMinMargs.end());
                initCandidatesAscending.push_back(INIT_UNSURE);
                return initCandidatesAscending;
            }
            if (opts.verbose) {
                std::cout << "[PrimHeu]   No product face with negative neighbors found" << std::endl;
                std::cout << "[PrimHeu]   > Searching for smallest min marginal product face " << std::endl;
            }
            idxSmallestMinMarg = 0; float smallestMinMarginal = minMarginals(0);
            for (int f = 1; f < minMarginals.rows(); f++) {
                // we dont want to init with tri to point matchings => skip those indices
                if (f == idxStartTriYToVX && opts.avoidTriToVertForInit) f += numTriYToVX;
                if (f == idxStartTriXToVY && opts.avoidTriToVertForInit) f += numTriXToVY;

                if (minMarginals(f) < smallestMinMarg) {
                    smallestMinMarg = minMarginals(f); idxSmallestMinMarg = f;
                }
                if (minMarginals(f) < FLOAT_EPSI) {
                    initCandidatesAscending.push_back(f);
                }
            }
            Change change = setNewSolVar(coreStatus, idxSmallestMinMarg);
            addNewChangeToSetOfChanges(changesAtCurrentDepth, change);
            changesInGamma.push_back(changesAtCurrentDepth);
            bool  propagtionSuccess = propagate(idxSmallestMinMarg, coreStatus, changesAtCurrentDepth);
            if (!propagtionSuccess) {
                // lets hope this never happes
                ASSERT_NEVER_REACH;
            }
            if (opts.verbose) std::cout << "[PrimHeu]   Done" << std::endl;

            std::sort(initCandidatesAscending.begin(), initCandidatesAscending.end(), cprMinMargs);
            initCandidatesAscending.push_back(INIT_UNSURE);
            return initCandidatesAscending;
        }
    }
    
    if (!zeroEnergySolutionPatch) {
        if (opts.verbose) std::cout << "[PrimHeu]   No Zero Energy Patch found" << std::endl;
        std::vector<int> initCandidates; initCandidates.reserve(opts.maxNumInitCandidates);
        for (int i = 0; i < idxInPatchEnergies.size(); i++) {
            initCandidates.push_back(idxInPatchEnergies(i));
        }
        initCandidates.push_back(INIT_UNSURE);
        return initCandidates;
    }
    
    /*
     Add all triangle patches with zero deformation energy to solution
     => from start on we want to have as many tri 2 tri matchings as possible
     */
    Eigen::MatrixXi bestPatch(4, 1); bestPatch = -bestPatch.setOnes();
    int patchidx = 0;
    bool noMoreLowEnergyPatch = false;
    if (opts.verbose) std::cout << "[PrimHeu]   Using Patch Energy based Initialization" << std::endl;
    while (!noMoreLowEnergyPatch) {
        tsl::robin_set<Change> currentChanges;
        if (numSolPatches == 0) {
            if (patchidx >= numFacesXxY*3) break;
            bestPatch(0) = idxInPatchEnergies(patchidx);
        }
        else {
            if (patchidx >= numSolPatches) break;
            // correctly translate idxes to product space
            bestPatch(0) = solPatches[idxInPatchEnergies(patchidx)];
        }
        bestPatch.block(1, 0, 3, 1) = getNeighboursIdx(bestPatch(0)).block(0, 0, 3, 1);
        bool propagtionSucces = true;
        for (int i = 0; i < 4 ; i++) {
            Change change = setNewSolVar(coreStatus, bestPatch(i));
            addNewChangeToSetOfChanges(changesAtCurrentDepth, change);
            propagtionSucces &= propagate(change.idx, coreStatus, currentChanges);
        }
        if (!propagtionSucces || !cleanUpVarsInGammaWithConstraints(coreStatus, currentChanges, false, !opts.debugOutput)) {
            undoChanges(coreStatus, currentChanges);
        }
        if (!isGammaFeasible(Gamma)) {
            undoChanges(coreStatus, currentChanges);
            std::cout << "[PrimHeu] WARNING: Initilization strategy ran into conflicts. Consider aborting" << std::endl;
        }
        else {
            changesAtCurrentDepth.reserve(changesAtCurrentDepth.size() + currentChanges.size());
            for (tsl::robin_set<Change>::iterator it = currentChanges.begin(); it != currentChanges.end(); ++it) {
                const Change currentChange = *it;
                addNewChangeToSetOfChanges(changesAtCurrentDepth, currentChange);
            }
        }

        patchidx++;
        if (sortedPatchEnergies(patchidx) > FLOAT_EPSI) {
            noMoreLowEnergyPatch = true;
        }
    }
    if (changesAtCurrentDepth.size() == 0 || !cleanUpVarsInGammaWithConstraints(coreStatus, changesAtCurrentDepth, false, !opts.debugOutput)) {
        // hopefully this does never happen
        if (opts.verbose) std::cout << "[PrimHeu] Initialization strategy failed" << std::endl;
    }
    else {
        if (opts.verbose) std::cout << "[PrimHeu] Done" << std::endl;
    }
    changesInGamma.push_back(changesAtCurrentDepth);
    initCandidatesAscending.push_back(INIT_SURE);
    return initCandidatesAscending;
}

int decideWhichRowToChoose(Shape &shape, bool verbose, int numFacesShapeWithHoles, float meanTriangleAreaOtherShape, std::vector<int> &initFacesAscending) {
    if (initFacesAscending.size() != 0) {
        const int faceIdx = initFacesAscending.front();
        initFacesAscending.erase(initFacesAscending.begin());
        return faceIdx;
    }
    
    initFacesAscending.reserve(shape.getNumFaces());
    if (verbose) std::cout << "[PrimHeu]   > Finding the Triangle in the flattest area" << std::endl;
    const Eigen::MatrixXf V = shape.getV();
    const Eigen::MatrixXi F = shape.getF();
    const int numFaces = shape.getNumFaces();

    // radius in [minRadius, maxRadius] where radius = 0.03 * numFaces
    const int minRadius = 1;
    const int maxRadius = 50;
    const float radiusFactor = 0.005;
    const int radiusTemp = round(radiusFactor * numFaces);
    const int radius = radiusTemp < minRadius ?  minRadius : ( radiusTemp > maxRadius ? maxRadius : radiusTemp );
    Eigen::MatrixXf KF(numFaces, 1);
    //if (radius > 2) {
        Eigen::MatrixXf KV(numFaces, 1);
        igl::gaussian_curvature(V, F, KV);
        for (int face = 0; face < numFaces; face++) {
            KF(face) = 0;
            for (int vertex = 0; vertex < 3; vertex++) {
                const int vertexIdx = F(face, vertex);
                KF(face) += KV(vertexIdx);
            }
        }
    
    // Average
    const int numAvgIters = round(0.01 * numFaces);
    Eigen::MatrixXf KFAvg = KF;
    const Eigen::MatrixXi triNeighbors = shape.getTriangleNeighbours();
    const Eigen::MatrixXf triAreas = shape.getTriangleAreas();
    for (int avgIters = 0; avgIters < numAvgIters; avgIters++) {
        for (int face = 0; face < numFaces; face++) {
            const int idxN0 = triNeighbors(face, 0);
            const int idxN1 = triNeighbors(face, 1);
            const int idxN2 = triNeighbors(face, 2);
            const float a0 = triAreas(idxN0);
            const float a1 = triAreas(idxN1);
            const float a2 = triAreas(idxN2);
            KFAvg(face) = 1.0 / (3 * triAreas(face) + a0 + a1 + a2) * (3 * triAreas(face) * KF(face)) + a0 * KF(idxN0) + a1 * KF(idxN1) + a2 * KF(idxN2);
        }
        KF = KFAvg;
    }
    
    const float maxCurvature = KF.cwiseAbs().maxCoeff();
    
    // make sure we dont use a area as initialization which is in a creeze (no changes in curvature for the neighbors)
    for (int face = 0; face < numFaces; face++) {
        const bool isCurvaturePositive = KF(face) >= 0;
        if (isCurvaturePositive) {
            if (KF(triNeighbors(face, 0)) < -FLOAT_EPSI || KF(triNeighbors(face, 1)) < -FLOAT_EPSI ||  KF(triNeighbors(face, 2)) < -FLOAT_EPSI) {
                KF(face) += maxCurvature;
            }
        }
        else {
            if (KF(triNeighbors(face, 0)) > FLOAT_EPSI || KF(triNeighbors(face, 1)) > FLOAT_EPSI ||  KF(triNeighbors(face, 2)) > FLOAT_EPSI) {
                KF(face) -= maxCurvature;
            }
        }
    }
    
    // make sure we dont use a triangle which is not regular
    const float ninetyDegree = M_PI_2;
    const float twentyDegree = 0.3491;
    for (int face = 0; face < numFaces; face++) {
        const Eigen::MatrixXf angles = shape.getTriangleAngles(face);
        for (int i = 0; i < 3; i++) {
            if (angles(i) > ninetyDegree ||angles(i) < twentyDegree) {
                KF(face) += KF(face) > 0 ? maxCurvature : -maxCurvature;
                break;
            }
        }
    }
    
    // weight with delta of mean triangle areas of the other shape
    const float meanTriangleArea = triAreas.mean();
    for (int face = 0; face < numFaces; face++) {
        const float delta = std::abs(triAreas(face) - meanTriangleArea);
        KF(face) = delta * KF(face);
    }
    
    // make sure we dont initialize from a closed hole
    for (int face = numFacesShapeWithHoles-1; face < numFaces; face++) {
        KF(face) += KF(face) > 0 ? maxCurvature : -maxCurvature;
    }

    // find flattest area (triangle with gC ~ 0, but gC < 0)
    int idxFlattestTriangle = -1;
    Eigen::MatrixXf KFabs = KF.cwiseAbs();
    Eigen::MatrixXi idxSorted(KFabs.rows(), 1);
    igl::sort(KFabs,
              1,    // sort each column
              true, // -> ascending order
              KFabs,
              idxSorted);
    idxFlattestTriangle = idxSorted(0);
    for (int i = 1; i < KFabs.rows(); i++) {
        initFacesAscending.push_back(idxSorted(i));
    }
    if (verbose) std::cout << "[PrimHeu]   > Idx of this triangle is " << idxFlattestTriangle << std::endl;
    return idxFlattestTriangle;
}


// returns if gammaEmpty or not
bool PrimalHeuristic::initializeGamma(MatrixInt8 &Gamma, NonWatertightMeshHandler &nonWatertightMeshHandler) {
    if (!opts.useMinMarginals) {
        return true;
    }
    else {
        if (opts.verbose) std::cout << "[PrimHeu] Initializing Gamma" << std::endl;
        if (opts.verbose) std::cout << "[PrimHeu]   > Allowing one triangle only be matched non-degeneratly" << std::endl;
        initializationInProgress = true;
        const int numRowsDel = constr.getConstraintMatrix().rows() - numFacesX - numFacesY;
        // take either zeroth row of pi_X or pi_Y
        int startRowPi;
        const bool didCloseHolesInX = nonWatertightMeshHandler.filledHolesOfShapeX();
        const bool didCloseHolesInY = nonWatertightMeshHandler.filledHolesOfShapeY();
        const bool chooseShapeXAsInitialization = didCloseHolesInX || (!didCloseHolesInY && numFacesX > numFacesY);
        if (chooseShapeXAsInitialization) {
            if (opts.verbose) std::cout << "[PrimHeu]   > Choosing Triangle from Shape X since Shape X has less triangles" << std::endl;
            const int numFacesShapeWithHoles = didCloseHolesInX ? nonWatertightMeshHandler.getShapeXWithHoles().getNumFaces() : numFacesX;
            startRowPi = numRowsDel + decideWhichRowToChoose(shapeX,
                                                             opts.verbose,
                                                             numFacesShapeWithHoles,
                                                             shapeY.getTriangleAreas().mean(),
                                                             initFacesAscending);
        }
        else {
            if (opts.verbose) std::cout << "[PrimHeu]   > Choosing Triangle from Shape Y since Shape Y has less triangles" << std::endl;
            const int numFacesShapeWithHoles = didCloseHolesInY ? nonWatertightMeshHandler.getShapeYWithHoles().getNumFaces() : numFacesY;
            startRowPi = numRowsDel + numFacesX + decideWhichRowToChoose(shapeY,
                                                                         opts.verbose,
                                                                         numFacesShapeWithHoles,
                                                                         shapeX.getTriangleAreas().mean(),
                                                                         initFacesAscending);
        }
        Gamma = -Gamma.setOnes();
        initSet.reserve(constr.getConstraintMatrix().row(startRowPi).nonZeros());
        const SparseMatInt8 constrMatrix = constr.getConstraintMatrix();
        for (typename SparseMatInt8Rm::InnerIterator it(constrMatrix, startRowPi); it; ++it) {
            const int f = it.index();
            // set all degenerate product triangels refering to tringle "startRowPi" to zero
            // all other triangles are init candidates
            if (f >= 3 * numFacesX * numFacesY) {
                Gamma(f) = 0;
            }
            else {
                initSet.push_back(f);
            }
        }
        if (opts.verbose) std::cout << "[PrimHeu]   > Updating min marginals to obtain better guidance" << std::endl;
        return false;
    }
}
