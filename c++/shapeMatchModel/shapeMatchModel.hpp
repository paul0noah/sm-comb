//
//  ShapeMatchModel.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 23.05.21.
//

#ifndef ShapeMatchModel_hpp
#define ShapeMatchModel_hpp

#include "helper/shape.hpp"
#include "helper/nonWatertightMeshHandler.hpp"
#include "shapeMatchModel/energyComputation/deformationEnergy.hpp"
#include "shapeMatchModel/constraintsComputation/constraints.hpp"
#include "shapeMatchModel/primalHeuristic.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ILP_input.h>
#include <ctime>
#include <bdd_solver.h>

struct ShapeMatchModelOpts {
    bool verbose;
    bool debugOutputMinMarginalsNonEmptyGamma;
    bool employDualSolverMultipleTimes;
    int maxNumDualSolverCalls;
    bool useMinMarginals;
    bool useConstraintsGroups;
    bool fillHolesOfShapes; // we require watertight meshes, if true smm tries to close holes of shapes
    bool assignZeroEnergyToFilledHoles; // necessary for partial correspondence
    bool initGammaBeforeComputingMinMarginals;
    PrimalHeuristicOpts primalHeuristicOpts;
    LPMP::bdd_solver_options bddSolverOpts;
    bool writeModelToFileAfterSolving;
    std::string modelName;
    ShapeMatchModelOpts() {
        verbose = true;
        debugOutputMinMarginalsNonEmptyGamma = false;
        primalHeuristicOpts = PrimalHeuristicOpts();
        employDualSolverMultipleTimes = true;
        maxNumDualSolverCalls = 15;
        useMinMarginals = true;
        useConstraintsGroups = true;
        fillHolesOfShapes = true;
        assignZeroEnergyToFilledHoles = false;
        writeModelToFileAfterSolving = false;
#if defined(_OPENMP)
        bddSolverOpts.bdd_solver_impl_ = LPMP::bdd_solver_options::bdd_solver_impl::parallel_mma;
#else
        bddSolverOpts.bdd_solver_impl_ = LPMP::bdd_solver_options::bdd_solver_impl::sequential_mma;
#endif
        bddSolverOpts.bdd_solver_precision_ = LPMP::bdd_solver_options::bdd_solver_precision::single_prec;
        bddSolverOpts.tolerance = 1e-6;
        initGammaBeforeComputingMinMarginals = true;
        
        // make sure model name is unique if no name is provided
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];
        time (&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer,sizeof(buffer), "%Y-%m-%d_%H-%M-%S", timeinfo);
        modelName = std::string(buffer) + "_smm";
    }
};

class ShapeMatchModel {
private:
    Shape shapeX;
    Shape shapeY;
    Combinations combos;
    DeformationEnergy deformationEnergy;
    Constraints constr;
    void generate();
    bool checkWatertightness();
    std::string getVariableName(int idx, Eigen::MatrixXi &FaCombo, Eigen::MatrixXi &FbCombo);
    LPMP::ILP_input ilp;
    bool ilpGenerated;
    bool minMarginalsComputed;
    Eigen::MatrixXf minMarginals;
    ShapeMatchModelOpts opts;
    void writeSolutionToFile(MatrixInt8 &Gamma);
    //Eigen::MatrixXf getMinMarginalsWithNonEmptyGamma(Eigen::SparseMatrix<int8_t> &Gamma);
    Eigen::MatrixXf getMinMarginals(Eigen::SparseMatrix<int8_t> &Gamma, bool gammaEmpty);
    LPMP::bdd_solver* bddsolver;
    NonWatertightMeshHandler nonWatertightMeshHandler;
    float initialLowerBound;
    bool generationSuccessfull;
    
public:
    ShapeMatchModel(std::string modelname);
    ShapeMatchModel(std::string modelname, ShapeMatchModelOpts optsIn);
    ShapeMatchModel(Shape &sX, Shape & sY);
    ShapeMatchModel(Shape &sX, Shape & sY, ShapeMatchModelOpts optsIn);
    ShapeMatchModel(Eigen::MatrixXi FX, Eigen::MatrixXf VX, Eigen::MatrixXi FY, Eigen::MatrixXf VY);
    ShapeMatchModel(std::string filenameShapeX, std::string filenameShapeY);
    ShapeMatchModel(std::string filenameShapeX, int numFacesX, std::string filenameShapeY, int numFacesY);
    ShapeMatchModel(std::string filenameShapeX, std::string filenameShapeY, ShapeMatchModelOpts opts);
    ~ShapeMatchModel();
    void saveAsLp(const std::string& filename);
    void saveIlpAsLp(const std::string& filename);
    Combinations& getCombinations();
    Eigen::MatrixXf getDeformationEnergy();
    SparseMatInt8 getConstraintsMatrix();
    SparseVecInt8 getConstraintsVector();
    LPMP::ILP_input getIlpObj();
    void plotSolution(const SparseVecInt8 &G);
    void plotInterpolatedSolution(const SparseVecInt8 &G);
    MatrixInt8 solve();
    Eigen::MatrixXf getMinMarginals();
    void printSolutionInfo(SparseMatInt8 Gamma);
    MatrixInt8 readSolutionFromFile(std::string filename);
    MatrixInt8 readSolutionFromFile();
    void writeModelToFile();
    Eigen::MatrixXi getPointMatchesFromSolution(const SparseVecInt8 &Gamma);
    void updateEnergy(const Eigen::MatrixXf& Vx2VyCost, bool weightWithAreas, bool useMemReg, float lambda);
    float getFinalEnergy(const SparseVecInt8 &Gamma);
    float getLowerBound();
    bool smmCreatedSuccessFully();
    Eigen::MatrixXi& getFaCombo();
    Eigen::MatrixXi& getFbCombo();
    void constantPenaliseDegenerate(float addval);
    void setMaxNumDualSolverCalls(const int numcalls);
    void setMaxNumBacktracks(const int maxbacktracks);
    void setMaxPrimalHeuristicIters(const int maxiters);
};

#endif /* ShapeMatchModel_hpp */
