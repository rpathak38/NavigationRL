//
// Created by jemin on 20. 9. 22..
//

#ifndef _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
#define _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_

#include <initializer_list>
#include <string>
#include <map>


namespace raisim {

struct RewardElement {
  float coefficient;
  float reward;
  float integral;
  float curriculum;
  float update;
};

class Reward {
 public:
  Reward (std::initializer_list<std::string> names) {
    for(auto& nm: names)
      rewards_[nm] = raisim::RewardElement();
  }

  Reward () = default;

  void initializeFromConfigurationFile(const YAML::Node& cfg) {
    for(auto rw = cfg.begin(); rw != cfg.end(); rw++) {
      std::string name = (*rw).first.as<std::string>();
      rewards_[name] = raisim::RewardElement();
      rewards_[name].coefficient = (*rw).second["coeff"].as<float>();
      if((*rw).second["update"]) {
        rewards_[name].update = (*rw).second["update"].as<float>();
        if(rewards_[name].update > 0)
          rewards_[name].curriculum = 0.05;
        else
          rewards_[name].curriculum = 1;
      } else {
        rewards_[name].curriculum = 1;
        rewards_[name].update = 1;
      }

      num_rewards_ += 1;
    }
  }

  const float& operator [] (const std::string& name) {
    return rewards_[name].reward;
  }

  void record (const std::string& name, float reward) {
    RSFATAL_IF(rewards_.find(name) == rewards_.end(), name<<" was not found in the configuration file")
    RSISNAN_MSG(reward, name<<" is nan")

    rewards_[name].reward = reward * rewards_[name].coefficient * rewards_[name].curriculum;
    rewards_[name].integral += rewards_[name].reward;
  }

  void curriculumUpdate() {
    for (auto& rw: rewards_)
      if (rw.second.update < 0)
        rw.second.curriculum *= -(rw.second.update);
      else
        rw.second.curriculum = pow(rw.second.curriculum, rw.second.update);
  }

  int getNumRewards() const { return num_rewards_; }

  float getRewardIntegral(const std::string& name) { return rewards_[name].integral; }
  float getReward(const std::string& name) { return rewards_[name].reward; }

  float sum() {
    float sum = 0.f;
    for(auto& rw: rewards_)
      sum += rw.second.reward;

    return sum;
  }

  void setZero() {
    for(auto& rw: rewards_)
      rw.second.reward = 0.f;
  }

  void reset() {
    for(auto& rw: rewards_) {
      rw.second.integral = 0.f;
      rw.second.reward = 0.f;
      if(rw.second.update > 0)
        rw.second.curriculum = 1;
      else
        rw.second.curriculum = 0.05;
    }
  }

  const std::map<std::string, float>& getStdMapOfRewardIntegral() {
    for(auto& rw: rewards_)
      costSum_[rw.first] = rw.second.integral;

    return costSum_;
  }


 private:
  std::map<std::string, raisim::RewardElement> rewards_;
  std::map<std::string, float> costSum_;

  int num_rewards_ = 0;
};

}

#endif //_RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
