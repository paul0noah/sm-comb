//
//  primalHeuristic.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 19.08.21.
//

#ifndef PrimalHeuristic_hpp
#define PrimalHeuristic_hpp

#include "helper/shape.hpp"
#include "shapeMatchModel/energyComputation/deformationEnergy.hpp"
#include "shapeMatchModel/constraintsComputation/constraints.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ILP_input.h>
#include <set>
#include <iostream>
#include <limits>
#include <tsl/robin_set.h>

typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixInt8;
typedef Eigen::SparseMatrix<int8_t, Eigen::RowMajor> SparseMatInt8Rm;
typedef Eigen::SparseMatrix<int8_t, Eigen::ColMajor> SparseMatInt8Cm;

enum PRIMAL_HEURISTIC_RETURN_FLAG {
    SOL_COMPLETE,
    SOL_INCOMPLETE,
    SOL_INFEASIBLE
};

enum INIT_RETURN_FLAG {
    INIT_SURE = -1,
    INIT_UNSURE = -2
};

enum CHANGE_REASON {
    UNKNOWN = 0,
    INIT,
    SOL_CANDIDATE,
    PROPAGATION,
    BACKTRACK,
    ROTATION,
    ELIMINATION
};

struct Change {
    unsigned int idx;
    int8_t oldValue;
    CHANGE_REASON reason;
    bool operator==(const Change& chg) const {
        return idx == chg.idx;
    }
};

struct Candidate {
    unsigned int idx;
    int8_t flag;
    Candidate(unsigned int idx, int8_t flag):
    idx(idx), flag(flag) {}
    bool operator==(const Candidate& candidate) const {
        return idx == candidate.idx;
    }
    bool operator<(const Candidate& candidate) const
    {
        return idx < candidate.idx;
    }
    friend std::ostream& operator<<(std::ostream& os, const Candidate& cd){
        os << "(" << cd.idx << ", " << (int)cd.flag << ")";
        return os;
    }
};

namespace std {
template<> struct hash<Change> {
    std::size_t operator()(Change const& chg) const noexcept {
        return chg.idx;
    }
};
template<> struct equal_to<Change>{
    constexpr bool operator()(const Change &lhs, const Change &rhs) const {
        return lhs.idx == rhs.idx;
    }
};
template<> struct hash<Candidate> {
    std::size_t operator()(Candidate const& cand) const noexcept {
        return std::hash<unsigned int>{}(cand.idx);
    }
};
template<> struct equal_to<Candidate>{
    constexpr bool operator()(const Candidate &lhs, const Candidate &rhs) const {
        return lhs.idx == rhs.idx;
    }
};
} // namespace std

struct PrimalHeuristicOpts {
    unsigned int maxIter;
    unsigned int maxBacktracks;
    unsigned int maxDepth;
    unsigned int maxVarOnes;
    bool useMinMarginals;
    bool verbose;
    bool debugOutput;
    bool allowOnlyNonDegenerateMatchings;
    bool cleanUpGammAfterEachIteration;
    bool useCleanUp;
    bool precheckVarCandidates;
    bool autoSetMaxBacktracks;
    bool autoSetMaxIter;
    bool avoidTriToVertForInit;
    unsigned int maxNumInitCandidates;
    PrimalHeuristicOpts(){
        autoSetMaxIter = true;
        maxIter = 1000;
        maxVarOnes = 10000000; // needed for parallel initialization
        autoSetMaxBacktracks = true;
        maxBacktracks = 100;
        maxDepth = std::numeric_limits<int>::max();
        useMinMarginals = true;
        verbose = true;
        debugOutput = false;
        allowOnlyNonDegenerateMatchings = false;
        cleanUpGammAfterEachIteration = true;
        useCleanUp = true;
        // prechecking is dreprecated and should not be used
        precheckVarCandidates = false;
        // if true => dont allow tri to vert product faces as init face
        avoidTriToVertForInit = true;
        // how many init candidates should primal heuristic try if initialization is unsure
        maxNumInitCandidates = 100;
    }
};

struct CORE_STATUS {
    size_t numVarSolCandidate;
    size_t numVarLikelySolCandidate;
    size_t numVarMustBeSol;
    tsl::robin_set<Candidate> candidates;
    tsl::robin_set<unsigned int> idxVarOnes;
    MatrixInt8& Gamma;
    int id;
    std::string outputPrefix;
    CORE_STATUS(MatrixInt8& GammaRef) : Gamma(GammaRef) {
        numVarSolCandidate = 0;
        numVarLikelySolCandidate = 0;
        numVarMustBeSol = 0;
        id = -1;
        outputPrefix = "[PrimHeu] ";
    }
};

class PrimalHeuristic {
private:
    PrimalHeuristicOpts opts;
    Shape& shapeX;
    Shape& shapeY;
    Combinations& combos;
    Eigen::MatrixXf& costs;
    Eigen::MatrixXf& minMarginals;
    Eigen::MatrixXi triangleNeighboursX;
    Eigen::MatrixXi triangleNeighboursY;
    Constraints& constr;
    unsigned int numFacesX;
    unsigned int numFacesY;
    unsigned int numEdgesX;
    unsigned int numEdgesY;
    unsigned int numVerticesX;
    unsigned int numVerticesY;
    unsigned int numProductEdges;
    unsigned int numProductFaces;
    SparseMatInt8Rm delPlusRm;
    SparseMatInt8Cm delPlusCm;
    SparseMatInt8Rm delMinusRm;
    SparseMatInt8Cm delMinusCm;
    SparseMatInt8Rm piXRm;
    SparseMatInt8Cm piXCm;
    SparseMatInt8Rm piYRm;
    SparseMatInt8Cm piYCm;
    bool initializationInProgress;
    std::vector<int> initFacesAscending;
    std::vector<int> initSet;
    void extractMatrices();
    void addNewChangeToSetOfChanges(tsl::robin_set<Change> &changesAtCurrentDepth, Change newChange) const;
    bool propagate(int idx,
                   CORE_STATUS &coreStatus,
                   tsl::robin_set<Change> &changesAtCurrentDepth) const;
    // TODO: REMOVE
    //bool propagateTri2TriMatching(int idx, MatrixInt8 &Gamma, std::vector<Change> &changesAtCurrentDepth);
    bool isGammaFeasible(MatrixInt8 &Gamma) const;
    std::vector<int> initialization(CORE_STATUS &coreStatus,
                                    std::vector<tsl::robin_set<Change>> &changesInGamma) const;
    //std::vector<int> getFallbackInitSet();
    Eigen::MatrixXi getNeighboursIdx(int f) const;
    Eigen::MatrixXi getNeighboursIdx(int f,
                                     Eigen::MatrixXi &connectingEdgeIdxs) const;
    bool setGammaValue(CORE_STATUS &coreStatus,
                       const int idx,
                       int8_t newVal,
                       tsl::robin_set<Change> &changes,
                       CHANGE_REASON reason) const;
    
    
    void undoChanges(CORE_STATUS &coreStatus,
                     tsl::robin_set<Change> &changes) const;
    void undoChangesAtLastLevel(CORE_STATUS &coreStatus,
                                std::vector<tsl::robin_set<Change>> &changesInGamma) const;
    void getVariabelCandidatesInAscendingOrder(CORE_STATUS &coreStatus,
                                               Eigen::MatrixXi &variableCandidates,
                                               const Eigen::MatrixXf cost) const;
    void getCandidatesAscending(CORE_STATUS &coreStatus,
                                Eigen::MatrixXi &variableCandidates,
                                const int8_t flag,
                                const int count,
                                const int offset) const;
    Change setNewSolVar(CORE_STATUS &coreStatus,
                        int idx) const;
    tsl::robin_set<int> getConstraintIdxAffectedByChanges(tsl::robin_set<Change> &changes,
                                                          int idxFirstChange) const;
    bool doCleanUp(int k, CORE_STATUS &coreStatus,
                   tsl::robin_set<Change> &changes,
                   int &numVarsEliminated,
                   int &numVarsSet,
                   const SparseMatInt8 &constrLHS,
                   const SparseVecInt8 &constrRHS) const;
    bool cleanUpVarsInGammaWithConstraints(CORE_STATUS &coreStatus,
                                           tsl::robin_set<Change> &changes,
                                           bool allGamma,
                                           bool quiet) const;
    bool preCheckSettingVar(CORE_STATUS &coreStatus,
                            const int idx,
                            const int8_t val) const;
    tsl::robin_set<int> getIdxAllNeighborsOfProductTriangle(int f) const;
    PRIMAL_HEURISTIC_RETURN_FLAG core(CORE_STATUS &coreStatus,
                                      std::vector<tsl::robin_set<Change>> &changesInGamma) const;
    
public:
    PrimalHeuristic(Shape &sX,
                    Shape &sY,
                    Combinations& cbs,
                    Eigen::MatrixXf& costs,
                    Eigen::MatrixXf& minMarginals,
                    Constraints& cstrs,
                    PrimalHeuristicOpts opts);
    PRIMAL_HEURISTIC_RETURN_FLAG apply(MatrixInt8 &Gamma, bool gammaEmpty);
    void updateOpts(PrimalHeuristicOpts newOpts);
    void updateMinMarginals(Eigen::MatrixXf &newMinMarginals);
    bool initializeGamma(MatrixInt8 &Gamma, NonWatertightMeshHandler &nonWatertightMeshHandler);

};

#endif /* PrimalHeuristic_hpp */
