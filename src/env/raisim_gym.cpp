//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace nb = nanobind;
using namespace raisim;

#ifndef ENVIRONMENT_NAME
#define ENVIRONMENT_NAME RaisimGymEnv
#endif

NB_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  nb::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
          .def(nb::init<std::string, std::string>(), nb::arg("resourceDir"), nb::arg("cfg"))
          .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
          .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
          .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
          .def("valueObserve", &VectorizedEnvironment<ENVIRONMENT>::valueObserve)
          .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
          .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
          .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
          .def("isTerminal", &VectorizedEnvironment<ENVIRONMENT>::isTerminal)
          .def("setCommand", &VectorizedEnvironment<ENVIRONMENT>::setCommand)
          .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
          .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
          .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
          .def("getValueObDim", &VectorizedEnvironment<ENVIRONMENT>::getValueObDim)
          .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
          .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
          .def("getNumOfRewards", &VectorizedEnvironment<ENVIRONMENT>::getNumOfRewards)
          .def("getRewards", &VectorizedEnvironment<ENVIRONMENT>::getRewards)
          .def("getLoggingReward", &VectorizedEnvironment<ENVIRONMENT>::getLoggingReward)
          .def("turnOnVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOnVisualization)
          .def("turnOffVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOffVisualization)
          .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
          .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
          .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)
          .def("mapChange", &VectorizedEnvironment<ENVIRONMENT>::mapChange)
          .def("navObserve", &VectorizedEnvironment<ENVIRONMENT>::navObserve)
          .def("navReset", &VectorizedEnvironment<ENVIRONMENT>::navReset)
          .def("getNavObDim", &VectorizedEnvironment<ENVIRONMENT>::getNavObDim);
}
