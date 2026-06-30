//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "omp.h"
#include "Yaml.hpp"
#include <Eigen/Core>
#include "BasicEigenTypes.hpp"

namespace raisim {

    template<class ChildEnvironment>
    class VectorizedEnvironment {

    public:

        explicit VectorizedEnvironment(std::string resourceDir, std::string cfg): resourceDir_(resourceDir) {
          Yaml::Parse(cfg_, cfg);
          raisim::World::setActivationKey(raisim::Path(resourceDir + "/activation.raisim").getString());
          if(&cfg_["render"])
            render_ = cfg_["render"].template As<bool>();
          n_steps_ = int(cfg_["max_time"].template As<double>() / cfg_["control_dt"].template As<double>());
        }

        ~VectorizedEnvironment() {
          for (auto *ptr: environments_)
            delete ptr;
        }

        void init() {
          omp_set_num_threads(cfg_["num_threads"].template As<int>());
          num_envs_ = cfg_["num_envs"].template As<int>();

          for (int i = 0; i < num_envs_; i++) {
            environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
            environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
            environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
          }

          setSeed(0);

          for (int i = 0; i < num_envs_; i++) {
            // only the first environment is visualized
            environments_[i]->init();
            environments_[i]->reset();
          }

          obDim_ = environments_[0]->getObDim();
          valueObDim_ = environments_[0]->getValueObDim();
          actionDim_ = environments_[0]->getActionDim();
          RSFATAL_IF(obDim_ == 0 || actionDim_ == 0 || valueObDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")
        }

        // resets all environments and returns observation
        void reset() {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++)
            environments_[i]->reset();
        }

        void observe(Eigen::Ref<EigenRowMajorMat> ob) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++)
            environments_[i]->observe(ob.row(i));
        }

        void valueObserve(Eigen::Ref<EigenRowMajorMat> ob) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++)
            environments_[i]->valueObserve(ob.row(i));
        }

        void navObserve(Eigen::Ref<EigenRowMajorMat> ob) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++)
            environments_[i]->navObserve(ob.row(i));
        }

        void navReset() {
          for (auto *env: environments_)
            env->navReset();
        }

        int getNavObDim() { return environments_[0]->getNavObDim(); }

        void navStep(Eigen::Ref<EigenRowMajorMat> command_vel,
                     Eigen::Ref<EigenVec> reward,
                     Eigen::Ref<EigenBoolVec> done) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++)
            perAgentNavStep(i, command_vel, reward, done);
        }

        void step(Eigen::Ref<EigenRowMajorMat> action,
                  Eigen::Ref<EigenVec> reward,
                  Eigen::Ref<EigenBoolVec> done) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++)
            perAgentStep(i, action, reward, done);
        }

        void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
        void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
        void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
        void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }
        void setCommand(Eigen::Ref<EigenVec> command) {environments_[0]->setCommand(command);}

        void setSeed(int seed) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++) {
            environments_[i]->setSeed(seed + i);
          }
        }

        void close() {
          for (auto *env: environments_)
            env->close();
        }

        void isTerminal(Eigen::Ref<EigenBoolVec> terminalState) {
          for (int i = 0; i < num_envs_; i++) {
            float terminalReward;
            terminalState[i] = environments_[i]->isTerminal(terminalReward);
          }
        }

        void setSimulationTimeStep(double dt) {
          for (auto *env: environments_)
            env->setSimulationTimeStep(dt);
        }

        void setControlTimeStep(double dt) {
          for (auto *env: environments_)
            env->setControlTimeStep(dt);
        }

        int getObDim() { return obDim_; }
        int getValueObDim() { return valueObDim_; }
        int getActionDim() { return actionDim_; }
        int getNumOfEnvs() { return num_envs_; }
        int getNumOfRewards() {return environments_[0]->getNumOfRewards(); }

        void getRewards(Eigen::Ref<EigenRowMajorMat> rewards, Eigen::Ref<EigenVec> stdev) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++) {
            environments_[i]->getRewards(rewards.row(i));
          }
          stdev.setZero();
        }

        void getLoggingReward(Eigen::Ref<EigenVec> reward) {
#pragma omp parallel for schedule(auto)
          for (int i = 0; i < num_envs_; i++)
            reward[i] = environments_[i]->getLoggingReward();
        }

        void mapChange() {
          if(render_) environments_[0]->lockVisualizationServerMutex();
          environments_[0]->mapChange();
          if(render_) environments_[0]->unlockVisualizationServerMutex();
        }

        void curriculumUpdate() {
          if(render_) environments_[0]->lockVisualizationServerMutex();
          for (auto *env: environments_)
            env->curriculumUpdate();
          if(render_) environments_[0]->unlockVisualizationServerMutex();
        };

    private:

      inline void perAgentNavStep(int agentId,
                               Eigen::Ref<EigenRowMajorMat> command_vel,
                               Eigen::Ref<EigenVec> reward,
                               Eigen::Ref<EigenBoolVec> done) {
        reward[agentId] = environments_[agentId]->navStep(command_vel.row(agentId).transpose());

        float terminalReward = 0;
        done[agentId] = environments_[agentId]->isNavTerminal(terminalReward);

        if (done[agentId]) {
          environments_[agentId]->navReset();
          reward[agentId] += terminalReward;
        }
      }

      inline void perAgentStep(int agentId,
                               Eigen::Ref<EigenRowMajorMat> action,
                               Eigen::Ref<EigenVec> reward,
                               Eigen::Ref<EigenBoolVec> done) {
        reward[agentId] = environments_[agentId]->step(action.row(agentId));
      }

        std::vector<ChildEnvironment *> environments_;

        int num_envs_ = 1;
        int obDim_ = 0, actionDim_ = 0, valueObDim_ = 0;
        bool recordVideo_=false, render_=false;
        std::string resourceDir_;
        Yaml::Node cfg_;
        int n_steps_ = 0;
    };

}

#endif //SRC_RAISIMGYMVECENV_HPP
