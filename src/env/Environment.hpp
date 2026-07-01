//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

#include <stdlib.h>
#include <cstdint>
#include <set>

#include "Yaml.hpp"
#include "RandomHeightMapGenerator.hpp"
#include "BasicEigenTypes.hpp"
#include "planning/contactPlanning.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
      resourceDir_(resourceDir), visualizable_(visualizable),
      contactPlanning_(26, cfg["control_dt"].template As<double>()) {  // horizon=26, dt=control_dt
    /// add objects
    robot_ = world_.addArticulatedSystem(resourceDir_ + "/mini_cheetah/mini-cheetah-vision-v1.6.urdf");
    robot_->setName("mini-cheetah");
    robot_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    loopCount_ = int(cfg["control_dt"].template As<double>()/ cfg["simulation_dt"].template As<double>() + 1e-8);

    /// get robot data
    gcDim_ = robot_->getGeneralizedCoordinateDim();
    gvDim_ = robot_->getDOF();
    nJoints_ = gvDim_ - 6;
    nLegs_ = 4;

    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(17.0);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(0.4);
    robot_->setPdGains(jointPgain_, jointDgain_);
    robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    gc_init_from_.setZero(gcDim_); gv_init_from_.setZero(gvDim_);
    nominalJointConfig_.setZero(nJoints_);
    isContact_.setZero(nLegs_); contactPhase_.setZero(nLegs_);
    feetPos_.setZero(3, nLegs_); feetVel_.setZero(3, nLegs_);
    torqueInput_.setZero(gvDim_);
    jointFrictions_.setZero(nJoints_);

    historyLength_ = 12;
    historyTempMem_.setZero(nJoints_ * historyLength_);
    jointPosHist_.setZero(nJoints_ * historyLength_); jointVelHist_.setZero(nJoints_ * historyLength_);

    /// this is nominal configuration of robot
    nominalJointConfig_ << 0, -0.9, 1.8, 0, -0.9, 1.8, 0, -0.9, 1.8, 0, -0.9, 1.8;
    gc_init_ << 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, nominalJointConfig_;

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = cfg["ob_dim"].template As<int>();
    valueObDim_ = cfg["value_ob_dim"].template As<int>();
    actionDim_ = 12;
    obDouble_.setZero(obDim_);
    valueObDouble_.setZero(valueObDim_);
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);  // 0.3

    /// contact planning (gait scheduler from MPC)
    contactPlanning_.setStanceAndSwingTime(0.13, 0.13);  // stance=0.13s, swing=0.13s

    /// indices of links that should not make contact with ground
    footIndices_.push_back(robot_->getBodyIdx("shank_fr"));
    footIndices_.push_back(robot_->getBodyIdx("shank_fl"));
    footIndices_.push_back(robot_->getBodyIdx("shank_hr"));
    footIndices_.push_back(robot_->getBodyIdx("shank_hl"));
    footFrameIndices_.push_back(robot_->getFrameIdxByName("toe_fr_joint"));
    footFrameIndices_.push_back(robot_->getFrameIdxByName("toe_fl_joint"));
    footFrameIndices_.push_back(robot_->getFrameIdxByName("toe_hr_joint"));
    footFrameIndices_.push_back(robot_->getFrameIdxByName("toe_hl_joint"));

    READ_YAML(double, terrainCurriculumFactor_, cfg["curriculum"]["terrain_initial_factor"])
    READ_YAML(double, terrainCurriculumDecayFactor_, cfg["curriculum"]["terrain_decay_factor"])
    READ_YAML(bool, observationRandomization_, cfg["randomization"]["observation_randomization"])
    READ_YAML(bool, jointFrictionRandomization_, cfg["randomization"]["joint_friction_randomization"])
    READ_YAML(bool, delayRandomization_, cfg["randomization"]["delay_randomization"])
    READ_YAML(bool, gainRandomization_, cfg["randomization"]["gain_randomization"])

    /// create heightmap
    heightMap_ = terrainGenerator_.generateTerrain(&world_,
                                                   RandomHeightMapGenerator::GroundType((int) floor(uniDist_(gen_) * 2.0)),
                                                   0., gen_, uniDist_);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      int port = cfg["port_num"].template As<int>();
      server_->launchServer(port);
      server_->focusOn(robot_);
      std::cout<<"Launched server on port: " << port << std::endl;
    }

    /// reward coeff
    READ_YAML(int, numOfRewards_, cfg["num_of_rewards"])
    READ_YAML(double, torqueRewardCoeff_, cfg["rewardCoeff"]["torque_reward_coeff"])
    READ_YAML(double, baseMotionRewardCoeff_, cfg["rewardCoeff"]["base_motion_reward_coeff"])
    READ_YAML(double, commandTrackingRewardCoeff_, cfg["rewardCoeff"]["command_tracking_reward_coeff"])
    rewardWeightMatrix_.setZero(6, 6);
    rewardWeightMatrix_.diagonal() << 1.0, 1.0, 1.0, 0.2, 0.2, 1.0;

    /// nav config
    navGridX_ = cfg["nav_grid_x"].template As<int>();
    navGridY_ = cfg["nav_grid_y"].template As<int>();
    navGridSpacing_ = cfg["nav_grid_spacing"].template As<double>();
    navRayHeight_ = cfg["nav_ray_height"].template As<double>();
    navCmdHistoryK_ = cfg["nav_cmd_history_k"].template As<int>();
    navObDim_ = cfg["nav_ob_dim"].template As<int>();
    navObDouble_.setZero(navObDim_);
    navCmdHistory_.setZero(navCmdHistoryK_ * 3);
    navGoalReachingCoeff_ = cfg["nav_reward_coeff"]["goal_reaching"].template As<double>();
    navDeathPenaltyCoeff_ = cfg["nav_reward_coeff"]["death_penalty"].template As<double>();
    navIncrementalDistCoeff_ = cfg["nav_reward_coeff"]["incremental_dist"].template As<double>();
    goal_.setZero();
    goalSet_ = false;
    navVisualsCreated_ = false;
  }

  ~ENVIRONMENT() { if (server_) server_->killServer(); }

  void init() { }

  // Nav Suite
  float navStep(const Eigen::Ref<EigenVec>& command_vel) {
    // Shift command history: move newer commands to older positions
    navCmdHistory_.head((navCmdHistoryK_ - 1) * 3) = navCmdHistory_.tail((navCmdHistoryK_ - 1) * 3);
    // Append newest command at the end
    navCmdHistory_.segment((navCmdHistoryK_ - 1) * 3, 3) = command_vel.cast<double>();

    command_[0] = command_vel[0];
    command_[1] = command_vel[1];
    command_[2] = command_vel[2];

    updateNavReward();
    return navGoalDistReward_;
  }

  bool isNavTerminal(float& terminalReward) {
    terminalReward = 0.f;
    navGoalReachingReward_ = 0.f;
    navDeathPenalty_ = 0.f;

    // robot fell: body contact with non-foot link
    for (auto& contact : robot_->getContacts()) {
      if (contact.skip()) continue;
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end()) {
        terminalReward = navDeathPenaltyCoeff_;
        navDeathPenalty_ = navDeathPenaltyCoeff_;
        return true;
      }
    }

    // robot fell: tilt > 70 deg
    if (acos(rot_(8)) * 180 / M_PI > 70) {
      terminalReward = navDeathPenaltyCoeff_;
      navDeathPenalty_ = navDeathPenaltyCoeff_;
      return true;
    }

    // goal reached
    if (goalSet_) {
      updateObservation();
      double dx = goal_[0] - gc_[0], dy = goal_[1] - gc_[1];
      if (std::sqrt(dx * dx + dy * dy) < 0.2) {
        terminalReward = navGoalReachingCoeff_;
        navGoalReachingReward_ = navGoalReachingCoeff_;
        return true;
      }
    }

    return false;
  }

  void navObserve(Eigen::Ref<EigenVec> ob) {
    updateObservation();

    double robotX = gc_[0], robotY = gc_[1], robotZ = gc_[2];
    double yaw = std::atan2(2.0 * (gc_[3]*gc_[6] + gc_[4]*gc_[5]),
                             1.0 - 2.0 * (gc_[5]*gc_[5] + gc_[6]*gc_[6]));
    double cy = std::cos(yaw), sy = std::sin(yaw);

    int idx = 0;
    for (int i = 0; i < navGridX_; i++) {
      for (int j = 0; j < navGridY_; j++) {
        double localX = (i - (navGridX_ - 1) / 2.0) * navGridSpacing_;
        double localY = (j - (navGridY_ - 1) / 2.0) * navGridSpacing_;
        double worldX = robotX + cy * localX - sy * localY;
        double worldY = robotY + sy * localX + cy * localY;
        navObDouble_[idx++] = heightMap_->getHeight(worldX, worldY) - robotZ;
      }
    }

    if (goalSet_) {
      double dx = goal_[0] - robotX, dy = goal_[1] - robotY;
      navObDouble_[idx++] =  cy * dx + sy * dy;
      navObDouble_[idx++] = -sy * dx + cy * dy;
      navObDouble_[idx++] = std::sqrt(dx*dx + dy*dy);
      double goalYaw = std::atan2(dy, dx);
      double yawErr = goalYaw - yaw;
      while (yawErr > M_PI) yawErr -= 2*M_PI;
      while (yawErr < -M_PI) yawErr += 2*M_PI;
      navObDouble_[idx++] = yawErr;
    } else {
      navObDouble_.segment(idx, 4).setZero();
      idx += 4;
    }

    // Command history
    for (int k = 0; k < navCmdHistoryK_ * 3; k++) {
      navObDouble_[idx++] = navCmdHistory_[k];
    }

    if (visualizable_ && server_) {
      if (!navVisualsCreated_) {
        for (int k = 0; k < navGridX_ * navGridY_; k++) {
          auto* s = server_->addVisualSphere("navGrid_" + std::to_string(k), 0.02,
                                              0.0, 0.8, 1.0, 0.6);
          gridSpheres_.push_back(s);
        }
        goalSphere_ = server_->addVisualSphere("navGoal", 0.2, 0.0, 1.0, 0.0, 0.8);
        navVisualsCreated_ = true;
      }
      int sidx = 0;
      for (int i = 0; i < navGridX_; i++) {
        for (int j = 0; j < navGridY_; j++) {
          double localX = (i - (navGridX_ - 1) / 2.0) * navGridSpacing_;
          double localY = (j - (navGridY_ - 1) / 2.0) * navGridSpacing_;
          double worldX = robotX + cy * localX - sy * localY;
          double worldY = robotY + sy * localX + cy * localY;
          gridSpheres_[sidx]->setPosition(worldX, worldY, heightMap_->getHeight(worldX, worldY));
          sidx++;
        }
      }
      if (goalSet_) {
        double goalZ = heightMap_->getHeight(goal_[0], goal_[1]);
        goalSphere_->setPosition(goal_[0], goal_[1], goalZ + 0.1);
      }
    }

    ob = navObDouble_.cast<float>();
  }

  void navReset() {
    reset();
    robot_->getState(gc_, gv_);
    command_.setZero();
    navCmdHistory_.setZero();

    double mapHalfSize = 5.25;

    // Randomize start position within map
    gc_[0] = (2.0 * uniDist_(gen_) - 1.0) * mapHalfSize;
    gc_[1] = (2.0 * uniDist_(gen_) - 1.0) * mapHalfSize;
    gc_[2] = heightMap_->getHeight(gc_[0], gc_[1]) + 0.25;
    robot_->setState(gc_, gv_);

    // Foot placement adjustment (same logic as reset())
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20;
    for (auto& foot : footFrameIndices_) {
      robot_->getFramePosition(foot, footPosition);
      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0), footPosition(1)) - footPosition(2);
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_[2] += maxNecessaryShift;
    robot_->setState(gc_, gv_);

    // Randomize goal position within map
    goal_[0] = (2.0 * uniDist_(gen_) - 1.0) * mapHalfSize;
    goal_[1] = (2.0 * uniDist_(gen_) - 1.0) * mapHalfSize;
    goal_[2] = heightMap_->getHeight(goal_[0], goal_[1]);
    goalSet_ = true;

    // Reset reward state
    double dx = goal_[0] - gc_[0], dy = goal_[1] - gc_[1];
    navPrevGoalDist_ = std::sqrt(dx*dx + dy*dy);
    navGoalDistReward_ = 0;

    if (visualizable_ && goalSphere_) {
      goalSphere_->setPosition(goal_[0], goal_[1], goal_[2] + 0.1);
    }
  }

  int getNavObDim() const { return navObDim_; }

  void updateNavReward() {
    if (goalSet_) {
      updateObservation();
      double dx = goal_[0] - gc_[0], dy = goal_[1] - gc_[1];
      double dist = std::sqrt(dx*dx + dy*dy);
      double deltaDist = navPrevGoalDist_ - dist;
      navGoalDistReward_ = navIncrementalDistCoeff_ * deltaDist;
      navPrevGoalDist_ = dist;
    } else {
      navGoalDistReward_ = 0;
    }
  }

  void getNavRewards(Eigen::Ref<EigenVec> rewards) {
    Eigen::VectorXd re(4);
    re << navGoalDistReward_, navGoalReachingReward_, navDeathPenalty_,
          navGoalDistReward_ + navGoalReachingReward_ + navDeathPenalty_;
    rewards = re.cast<float>();
  }

  void resetForTest(double frictionCoeff) {
    /// randomize generalized coordinates
    /// x,y position
    gc_init_[0] = 0.;
    gc_init_[1] = 0.;
    /// orientation
    raisim::Mat<3,3> rotMat, yawRot, pitchRollMat;
    raisim::Vec<4> quaternion;
    raisim::Vec<3> axis = {normDist_(gen_), normDist_(gen_), normDist_(gen_)};
    axis /= axis.norm();
    raisim::angleAxisToRotMat(axis, normDist_(gen_) * 0.2, pitchRollMat);
    raisim::angleAxisToRotMat({0,0,1}, 2 * uniDist_(gen_) * M_PI, yawRot);
    raisim::matmul(pitchRollMat, yawRot, rotMat);
    raisim::rotMatToQuat(rotMat, quaternion);
    gc_init_.segment(3, 4) = quaternion.e();
    ///joint angles
    for(int i = 0 ; i < nJoints_; i++)
      gc_init_[i+7] = nominalJointConfig_[i] + 0.05 * normDist_(gen_);

    if(uniDist_(gen_) > 0.5 && hasInitial_) {
      gc_init_ = gc_init_from_;
    }

    world_.setDefaultMaterial(frictionCoeff, 0.0, 0.01);

    /// at least one foot is in contact with the terrain
    robot_->setGeneralizedCoordinate(gc_init_);
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; /// some arbitrary high negative value
    for(auto& foot: footFrameIndices_) {
      robot_->getFramePosition(foot, footPosition);
      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0), footPosition(1)) - footPosition(2);
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_(2) += maxNecessaryShift;

    /// randomize generalized velocities
    /// base linear velocity
    raisim::Vec<3> bodyVel_b, bodyVel_w;
    bodyVel_b[0] = 0.4 * normDist_(gen_);
    bodyVel_b[1] = 0.3 * normDist_(gen_);
    bodyVel_b[2] = 0.2 * normDist_(gen_);
    raisim::matvecmul(rotMat, bodyVel_b, bodyVel_w);
    /// base angular velocities (just define this in the world frame since it is isometric)
    raisim::Vec<3> bodyAng_w;
    for(int i = 0; i < 3; i++) bodyAng_w[i] = 0.3 * normDist_(gen_);
    /// joint velocities
    Eigen::VectorXd jointVel(12);
    for(int i = 0; i < 12; i++) jointVel[i] = 0.2 * normDist_(gen_);
    /// combine
    gv_init_ << bodyVel_w.e(), bodyAng_w.e(), jointVel;

    standingMode_ = uniDist_(gen_) > 0.9;

    if(standingMode_) {
      command_.setZero();
    }
    else {
      do {
        command_ << 2.0 * (uniDist_(gen_) - 0.5),
            2.0 * (uniDist_(gen_) - 0.5),
            2.0 * (uniDist_(gen_) - 0.5);
      } while(command_.norm() < 0.4);
    }

    if(uniDist_(gen_) > 0.5 && hasInitial_) {
      gv_init_ = gv_init_from_;
    }

    historyTempMem_.setZero();
    previousAction_.setZero();
    contactPlanning_.reset();
    for(int i = 0; i < historyLength_; i++) {
      jointPosHist_.segment(nJoints_ * i, nJoints_) = gc_init_.tail(nJoints_);
      jointVelHist_.segment(nJoints_ * i, nJoints_) = gv_init_.tail(nJoints_);
    }

    /// set the states
    robot_->setState(gc_init_, gv_init_);
  }

  void resetNoRandom(double frictionCoeff) {
    gc_init_ << 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, nominalJointConfig_;
    gv_init_.setZero(gvDim_);

    world_.setDefaultMaterial(frictionCoeff, 0.0, 0.01);

    /// at least one foot is in contact with the terrain
    robot_->setGeneralizedCoordinate(gc_init_);
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; /// some arbitrary high negative value
    for(auto& foot: footFrameIndices_) {
      robot_->getFramePosition(foot, footPosition);
      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0),footPosition(1)) - footPosition(2);
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_(2) += maxNecessaryShift;

    /// set the states
    robot_->setState(gc_init_, gv_init_);
  }

  void reset() {
    /// randomize generalized coordinates
    /// x,y position
    gc_init_[0] = 0.;
    gc_init_[1] = 0.;
    /// orientation
    raisim::Mat<3,3> rotMat, yawRot, pitchRollMat;
    raisim::Vec<4> quaternion;
    raisim::Vec<3> axis = {normDist_(gen_), normDist_(gen_), normDist_(gen_)};
    axis /= axis.norm();
    raisim::angleAxisToRotMat(axis, normDist_(gen_) * 0.2, pitchRollMat);
    raisim::angleAxisToRotMat({0,0,1}, 2 * uniDist_(gen_) * M_PI, yawRot);
    raisim::matmul(pitchRollMat, yawRot, rotMat);
    raisim::rotMatToQuat(rotMat, quaternion);
    gc_init_.segment(3, 4) = quaternion.e();
    ///joint angles
    for(int i = 0 ; i < nJoints_; i++)
      gc_init_[i+7] = nominalJointConfig_[i] + 0.05 * normDist_(gen_);

    if(uniDist_(gen_) > 0.5 && hasInitial_) {
      gc_init_ = gc_init_from_;
    }

    groundFrictionCoeff_ = 0.3 + 0.7 * uniDist_(gen_); /// c_f = 0.30 ~ 1.00
    world_.setDefaultMaterial(groundFrictionCoeff_, 0.0, 0.01);

    /// at least one foot is in contact with the terrain
    robot_->setGeneralizedCoordinate(gc_init_);
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; /// some arbitrary high negative value
    for(auto& foot: footFrameIndices_) {
      robot_->getFramePosition(foot, footPosition);
      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition(0), footPosition(1)) - footPosition(2);
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_(2) += maxNecessaryShift;

    /// randomize generalized velocities
    /// base linear velocity
    raisim::Vec<3> bodyVel_b, bodyVel_w;
    bodyVel_b[0] = 0.4 * normDist_(gen_);
    bodyVel_b[1] = 0.3 * normDist_(gen_);
    bodyVel_b[2] = 0.2 * normDist_(gen_);
    raisim::matvecmul(rotMat, bodyVel_b, bodyVel_w);
    /// base angular velocities (just define this in the world frame since it is isometric)
    raisim::Vec<3> bodyAng_w;
    for(int i = 0; i < 3; i++) bodyAng_w[i] = 0.3 * normDist_(gen_);
    /// joint velocities
    Eigen::VectorXd jointVel(12);
    for(int i = 0; i < 12; i++) jointVel[i] = 0.2 * normDist_(gen_);
    /// combine
    gv_init_ << bodyVel_w.e(), bodyAng_w.e(), jointVel;

    standingMode_ = uniDist_(gen_) > 0.9;

    if(standingMode_) {
      command_.setZero();
    }
    else {
      do {
        command_ << 2.0 * (uniDist_(gen_) - 0.5),
            2.0 * (uniDist_(gen_) - 0.5),
            2.0 * (uniDist_(gen_) - 0.5);
      } while(command_.norm() < 0.4);
    }

    if(uniDist_(gen_) > 0.5 && hasInitial_) {
      gv_init_ = gv_init_from_;
    }

    historyTempMem_.setZero();
    previousAction_ << nominalJointConfig_;
    contactPlanning_.reset();
    for(int i = 0; i < historyLength_; i++) {
      jointPosHist_.segment(nJoints_ * i, nJoints_) = gc_init_.tail(nJoints_);
      jointVelHist_.segment(nJoints_ * i, nJoints_) = gv_init_.tail(nJoints_);
    }

    /// set the states
    robot_->setState(gc_init_, gv_init_);

    if (jointFrictionRandomization_) {
      double jFrictionHAAHFE = 0.3 * uniDist_(gen_);  /// [0, 0.3]
      double jFrictionKFE = 0.6 * uniDist_(gen_) + 0.1;  /// [0.1, 0.7]
      jointFrictions_ << jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE, jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE,
          jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE, jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE;
    }

    if (gainRandomization_) {
      jointPgain_.tail(nJoints_).setConstant(17.0 + 1.7 * (uniDist_(gen_) - 0.5));
      jointDgain_.tail(nJoints_).setConstant(0.4 + 0.04 * (uniDist_(gen_) - 0.5));
      robot_->setPdGains(jointPgain_, jointDgain_);
    }
  }

  void getInitState(Eigen::Ref<EigenVec> gc_init, Eigen::Ref<EigenVec> gv_init) {
    gc_init = gc_init_.cast<float>();
    gv_init = gv_init_.cast<float>();
  }

  void resetFromAnotherSpace(const Eigen::Ref<EigenVec> &gc_init, const Eigen::Ref<EigenVec> &gv_init) {
    auto gcInit = gc_init.cast<double>();
    auto gvInit = gv_init.cast<double>();
    robot_->setState(gcInit, gvInit);
  }

  void setSeed(int seed) { gen_.seed(seed); terrainGenerator_.setSeed(seed); }

  void updateHistory() {
    historyTempMem_ = jointVelHist_;
    jointVelHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointVelHist_.tail(nJoints_) = obDouble_.segment(18, nJoints_);

    historyTempMem_ = jointPosHist_;
    jointPosHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointPosHist_.tail(nJoints_) = obDouble_.segment(6, nJoints_);
  }

  float step(const Eigen::Ref<EigenVec>& action) {
    pTarget_.tail(nJoints_) = actionMean_ + action.cast<double>().cwiseProduct(actionStd_);
    previousAction_ = pTarget_.tail(nJoints_);

    updateHistory();

    cost_ = 0.;
    torqueReward_ = 0.;
    commandTrackingReward_ = 0.;
    baseMotionReward_ = 0.;

    int delayCount = 0;
    if (delayRandomization_) {
      delayCount = int((0.01 / simulation_dt_) * uniDist_(gen_) - 1e-10); /// delay U ~ (0,9) ms
    }

    for(int i = 0; i < loopCount_; i++) {
      if (i == delayCount) {
        robot_->setPdTarget(pTarget_, vTarget_);
      }
      if (jointFrictionRandomization_) {
        simulateJointFriction();
      }
      world_.integrate();
      accumulateReward();
    }

    commandTrackingReward_ /= loopCount_;
    baseMotionReward_ /= loopCount_;
    torqueReward_ /= loopCount_;
    realCommandTrackingReward_ /= loopCount_;
    cost_ = float(realCommandTrackingReward_ * std::exp(torqueReward_));
    return cost_;
  }

  void simulateJointFriction() {
    auto gv = robot_->getGeneralizedVelocity().e().tail(nJoints_);
    Eigen::VectorXd frictionTorque;
    frictionTorque.setZero(gvDim_);

    /// simulate friction
    for (int i = 0; i < nJoints_; i++) {
      frictionTorque[6 + i] = std::copysign(jointFrictions_[i], -gv[i]);
    }
    robot_->setGeneralizedForce(frictionTorque);
  }

  void accumulateReward() {
    robot_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot_);
    bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);

    double linearCommandTrackingReward = 0., angularCommandTrackingReward = 0.;
    linearCommandTrackingReward += std::exp(-1.0 * (command_.head(2) - bodyLinearVel_.head(2)).squaredNorm());
    angularCommandTrackingReward += std::exp(-1.5 * pow((command_(2) - bodyAngularVel_(2)), 2));
    commandTrackingReward_ += (linearCommandTrackingReward + angularCommandTrackingReward) * commandTrackingRewardCoeff_;
    baseMotionReward_ += baseMotionRewardCoeff_ * (1.0 * bodyLinearVel_[2] * bodyLinearVel_[2] + 0.2 * fabs(bodyAngularVel_[0]) + 0.2 * fabs(bodyAngularVel_[1]));
    torqueReward_ += torqueRewardCoeff_ * (robot_->getGeneralizedForce().e().tail(nJoints_).squaredNorm());

    Eigen::VectorXd c(6), cBar(6);
    c << bodyLinearVel_, bodyAngularVel_;
    cBar << command_(0), command_(1), 0.0, 0.0, 0.0, command_(2);
    realCommandTrackingReward_ += 0.75 * std::exp(-(c - cBar).transpose() * rewardWeightMatrix_ * (c - cBar));
  }

  void getRewards(Eigen::Ref<EigenVec> rewards) {
    Eigen::VectorXd re(numOfRewards_);
    re.setZero();
    re << commandTrackingReward_, baseMotionReward_, torqueReward_, realCommandTrackingReward_;
    rewards = re.cast<float>();
  }

  float getLoggingReward() {
    cost_ = float(commandTrackingReward_ * std::exp(0.2 * (baseMotionReward_ + torqueReward_)));
    return cost_;
  }

  void updateObservation() {
    robot_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot_);

    bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);

    /// Contact planning from MPC gait scheduler
    contactPlanning_.updateContactSequence();
    isContact_ = contactPlanning_.getIsContact();
    contactPhase_ = contactPlanning_.getContactPhase();
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    updateObservation();

    /// body orientation
    obDouble_ << rot_.e().row(2).transpose(), bodyAngularVel_,
        gc_.tail(nJoints_), gv_.tail(nJoints_), previousAction_,
        jointPosHist_.segment((historyLength_ - 12) * nJoints_, nJoints_),
        jointPosHist_.segment((historyLength_ - 9) * nJoints_, nJoints_),
        jointPosHist_.segment((historyLength_ - 6) * nJoints_, nJoints_),
        jointPosHist_.segment((historyLength_ - 3) * nJoints_, nJoints_),
        jointVelHist_.segment((historyLength_ - 12) * nJoints_, nJoints_),
        jointVelHist_.segment((historyLength_ - 9) * nJoints_, nJoints_),
        jointVelHist_.segment((historyLength_ - 6) * nJoints_, nJoints_),
        jointVelHist_.segment((historyLength_ - 3) * nJoints_, nJoints_),
        isContact_, std::sin(contactPhase_(0)), std::cos(contactPhase_(0)),
        command_;

    if (observationRandomization_) {
      for (int i = 0; i < 3; i++) {
        obDouble_(i) += 0.03 * normDist_(gen_); /// orientation
        obDouble_(3 + i) += 0.1 * normDist_(gen_); /// body ang vel
      }
      obDouble_.head(3) /= obDouble_.head(3).norm();
      for (int i = 0; i < nJoints_; i++) {
        obDouble_(6 + i) += 0.05 * normDist_(gen_); /// joint pos
        obDouble_(18 + i) += 0.5 * normDist_(gen_); /// joint vel
      }
    }

    ob = obDouble_.cast<float>(); /// (147)
  }

  void valueObserve(Eigen::Ref<EigenVec> ob) {
    valueObDouble_ << rot_.e().row(2).transpose(), bodyAngularVel_, bodyLinearVel_, gc_(2) - heightMap_->getHeight(gc_(0), gc_(1)),
        gc_.tail(nJoints_), gv_.tail(nJoints_), previousAction_,
        isContact_, std::sin(contactPhase_(0)), std::cos(contactPhase_(0)),
        command_;

    ob = valueObDouble_.cast<float>(); /// (55)
  }

  bool isTerminal(float& terminalReward) {
    terminalReward = 0.f;
    for(auto& contact: robot_->getContacts()) {
      if (contact.skip()) continue;
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
    }

    if(acos(rot_(8)) * 180 / M_PI > 70) {
      return true;
    }

    return false;
  }

  void curriculumUpdate() {
    terrainCurriculumFactor_ = std::pow(terrainCurriculumFactor_, terrainCurriculumDecayFactor_);

    world_.removeObject(heightMap_);
    auto groundType = RandomHeightMapGenerator::GroundType((int) floor(uniDist_(gen_) * 2.0));
    heightMap_ = terrainGenerator_.generateTerrain(&world_,
                                                   groundType,
                                                   terrainCurriculumFactor_, gen_, uniDist_);
  }

  void mapChange() {
    world_.removeObject(heightMap_);
    auto groundType = RandomHeightMapGenerator::GroundType(groundType_++ % 2);
    heightMap_ = terrainGenerator_.generateTerrain(&world_,
                                                   groundType,
                                                   terrainCurriculumFactor_, gen_, uniDist_);
  }

  void mapChangeForTest(double terrainCurriculumFactor, int groundType){
    // terrain setting
    world_.removeObject(heightMap_);

    heightMap_ = terrainGenerator_.generateTerrain(&world_,
                                                (RandomHeightMapGenerator::GroundType)groundType,
                                                terrainCurriculumFactor, gen_, uniDist_);
  }

  int getNumOfRewards() {return numOfRewards_;}

  void setCommand(const Eigen::Ref<EigenVec> &command) {
    command_[0] =  command[0];
    command_[1] =  command[1];
    command_[2] =  command[2];
  }

  void close () { }
  void setSimulationTimeStep(double dt) { simulation_dt_ = dt; world_.setTimeStep(dt); }
  void setControlTimeStep(double dt) { control_dt_ = dt; }
  int getObDim() const { return obDim_; }
  int getValueObDim() const { return valueObDim_; }
  int getActionDim() const { return actionDim_; }
  double getControlTimeStep() { return control_dt_; }
  double getSimulationTimeStep() { return simulation_dt_; }
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  void lockVisualizationServerMutex() { server_->lockVisualizationServerMutex(); }
  void unlockVisualizationServerMutex() { server_->unlockVisualizationServerMutex(); }

 private:
  raisim::World world_;
  double simulation_dt_;
  double control_dt_;
  std::string resourceDir_;
  int obDim_=0, actionDim_=0, valueObDim_=0;
  std::unique_ptr<raisim::RaisimServer> server_;

  /// Initialization Variables
  int gcDim_, gvDim_, nJoints_, nLegs_, loopCount_;
  int numOfRewards_;
  int groundType_ = 0;
  bool visualizable_ = false;
  bool standingMode_ = false;
  bool hasInitial_ = false;
  raisim::ArticulatedSystem* robot_;
  Eigen::MatrixXd feetPos_, feetVel_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_;
  Eigen::VectorXd gc_init_from_, gv_init_from_;
  Eigen::VectorXd nominalJointConfig_;
  Eigen::VectorXd isContact_, contactPhase_;
  Eigen::VectorXd torqueInput_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::vector<size_t> footIndices_;
  std::vector<size_t> footFrameIndices_;
  raisim::Mat<3,3> rot_;
  raisim::Vec<3> footPos_, footVel_;
  Eigen::VectorXd jointPgain_, jointDgain_;
  Eigen::VectorXd pTarget_, vTarget_, previousAction_;
  int historyLength_;
  Eigen::VectorXd jointPosHist_, jointVelHist_, historyTempMem_;
  Eigen::VectorXd jointFrictions_;

  /// Gait clock (replaces MPC contactPlanning)
  /// Contact planning (gait scheduler from MPC)
  planning::contactPlanning contactPlanning_;

  /// Learning Parameters
  float cost_;
  double terminalRewardCoeff_ = -10.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double baseMotionReward_ = 0., baseMotionRewardCoeff_;
  double commandTrackingReward_ = 0., commandTrackingRewardCoeff_;
  double realCommandTrackingReward_ = 0., realCommandTrackingRewardCoeff_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, valueObDouble_;
  Eigen::MatrixXd rewardWeightMatrix_;
  Eigen::Vector3d command_;

  /// rough terrain parameters
  raisim::HeightMap* heightMap_;
  double groundFrictionCoeff_;
  RandomHeightMapGenerator terrainGenerator_;
  double terrainCurriculumFactor_, terrainCurriculumDecayFactor_;

  /// randomization
  bool observationRandomization_ = false;
  bool jointFrictionRandomization_ = false;
  bool delayRandomization_ = false;
  bool gainRandomization_ = false;

  /// navigation
  int navObDim_ = 0;
  int navGridX_ = 20, navGridY_ = 20;
  double navGridSpacing_ = 0.25;
  double navRayHeight_ = 5.0;
  int navCmdHistoryK_ = 5;
  Eigen::VectorXd navCmdHistory_;
  Eigen::VectorXd navObDouble_;
  raisim::Vec<3> goal_;
  bool goalSet_ = false;
  bool navVisualsCreated_ = false;
  std::vector<raisim::Visuals*> gridSpheres_;
  raisim::Visuals* goalSphere_ = nullptr;

  /// nav rewards
  double navGoalReachingCoeff_ = 1.0;
  double navDeathPenaltyCoeff_ = -0.5;
  double navIncrementalDistCoeff_ = 0.01;
  double navGoalDistReward_ = 0;
  double navPrevGoalDist_ = 0;
  double navGoalReachingReward_ = 0;
  double navDeathPenalty_ = 0;

  std::mt19937 gen_;
  std::normal_distribution<double> normDist_{0., 1.};
  std::uniform_real_distribution<double> uniDist_{0., 1.};
  };
}
