//
//  ShapeMatchModelPyBinds.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 27.06.22.
//

#include "shapeMatchModel.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT

PYBIND11_MODULE(shape_match_model_pb, handle) {
    handle.doc() = "ShapeMatchModel python bindings";

    py::class_<ShapeMatchModel, std::shared_ptr<ShapeMatchModel>> smm(handle, "ShapeMatchModel");
    smm.def(py::init<std::string, std::string>());
    smm.def(py::init<std::string, int, std::string, int>());
    smm.def(py::init<Eigen::MatrixXi, Eigen::MatrixXf, Eigen::MatrixXi, Eigen::MatrixXf>());
    smm.def(py::init<Eigen::MatrixXi, Eigen::MatrixXf, Eigen::MatrixXi, Eigen::MatrixXf, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi>());
    smm.def("updateEnergy", &ShapeMatchModel::updateEnergy);
    smm.def("solve", &ShapeMatchModel::solve);
    smm.def("getPointMatchesFromSolution", &ShapeMatchModel::getPointMatchesFromSolution);
    smm.def("plotSolution", &ShapeMatchModel::plotInterpolatedSolution);
    smm.def("getIlpObj", &ShapeMatchModel::getIlpObj);
    smm.def("getFinalEnergy", &ShapeMatchModel::getFinalEnergy);
    smm.def("getLowerBound", &ShapeMatchModel::getLowerBound);
    smm.def("smmCreatedSuccessFully", &ShapeMatchModel::smmCreatedSuccessFully);
    smm.def("constraintsFullfilled", &ShapeMatchModel::constraintsFullfilled);
    smm.def("getFXCombo", &ShapeMatchModel::getFaCombo);
    smm.def("getFYCombo", &ShapeMatchModel::getFbCombo);
    smm.def("saveSmmAsILP", &ShapeMatchModel::saveIlpAsLp);
    smm.def("constantPenaliseDegenerate", &ShapeMatchModel::constantPenaliseDegenerate);
    smm.def("setMaxNumDualSolverCalls", &ShapeMatchModel::setMaxNumDualSolverCalls);
    smm.def("setMaxNumBacktracks", &ShapeMatchModel::setMaxNumBacktracks);
    smm.def("setMaxPrimalHeuristicIters", &ShapeMatchModel::setMaxPrimalHeuristicIters);
    smm.def("pruneWithCoarserMatching", &ShapeMatchModel::pruneWithCoarserMatching);
    smm.def("writeModelForMatlab", &ShapeMatchModel::writeModelForMatlab);
    smm.def("saveAsLPFile", &ShapeMatchModel::saveIlpAsLp);
    smm.def("getEnergyVector", &ShapeMatchModel::getEforPython);
    smm.def("getRHSVector", &ShapeMatchModel::getRHSforPython);
    smm.def("getAMatrix", &ShapeMatchModel::getAforPython);


    py::class_<LPMP::ILP_input>(handle, "ILP_instance")
        .def(py::init<>());
}
