//
// Created by youmdonghoon on 21. 10. 12..
//

#pragma once

namespace planning {
class contactPlanning {
 public:
  explicit contactPlanning(int horizon, double dt) :
      horizon_(horizon), dt_(dt){
    nLegs_ = 4;
    contactSeq_.setZero(nLegs_, horizon_);
    last_contact_ << true, false, false, true;
    step_elapsed_ << 1, 1, 1, 1;
  }

  void setStanceAndSwingTime(const double &stanceTime, const double &swingTime) {
    stanceTime_ = stanceTime;
    swingTime_ = swingTime;
    nStance_ = int(stanceTime_ / dt_);
    nSwing_ = int(swingTime_ / dt_);
  }

  void updateContactSequence() {

    for(int i = 0; i < nLegs_; i++) {
      if(last_contact_(i)) { // Last contact is stance
        if ( step_elapsed_(i) >= nStance_ ) {
          for( int j = 0; j < nSwing_ && j < horizon_; j++)
            contactSeq_(i,j) = false;

          int num_unit = (horizon_ - nSwing_ ) / (nStance_ + nSwing_);
          int rest = (horizon_ - nSwing_ ) % (nStance_ + nSwing_);

          for(int j = 0; j < num_unit; j++) {
            for(int k = 0; k < nStance_; k++)
              contactSeq_(i,  nSwing_ + j * (nStance_ + nSwing_) + k) = true;
            for(int k = nStance_; k < nSwing_ + nStance_; k++)
              contactSeq_(i,  nSwing_ + j * (nStance_ + nSwing_) + k) = false;
          }

          for(int j = 0; j < rest ; j++)
            contactSeq_(i, nSwing_ + num_unit * (nStance_ + nSwing_) + j) = j < nStance_;

          step_elapsed_(i) = 1;
        } else {
          for(int j = 0; j < nStance_ - step_elapsed_(i) && j < horizon_; j++)
            contactSeq_(i,j) = true;

          int num_unit = (horizon_ - (nStance_ - step_elapsed_(i)) ) / (nStance_ + nSwing_);
          int rest = (horizon_ - (nStance_ - step_elapsed_(i)) ) % (nStance_ + nSwing_);

          for(int j = 0; j < num_unit; j++) {
            for(int k = 0; k < nSwing_; k++)
              contactSeq_(i,  nStance_ - step_elapsed_(i) + j * (nStance_ + nSwing_) + k) = false;
            for(int k = nSwing_; k < nSwing_ + nStance_; k++)
              contactSeq_(i,  nStance_ - step_elapsed_(i) + j * (nStance_ + nSwing_) + k) = true;
          }

          for(int j = 0; j < rest ; j++)
            contactSeq_(i, nStance_ - step_elapsed_(i) + num_unit * (nStance_ + nSwing_) + j) = j >= nSwing_;

          step_elapsed_(i)++;
        }
      } else { // Last contact is swing
        if ( step_elapsed_(i) >= nSwing_ ) {
          for( int j = 0; j < nStance_ && j < horizon_; j++)
            contactSeq_(i,j) = true;

          int num_unit = (horizon_ - nStance_ ) / (nStance_ + nSwing_);
          int rest = (horizon_ - nStance_ ) % (nStance_ + nSwing_);

          for(int j = 0; j < num_unit; j++) {
            for(int k = 0; k < nSwing_; k++)
              contactSeq_(i,  nStance_ + j * (nStance_ + nSwing_) + k) = false;
            for(int k = nSwing_; k < nSwing_ + nStance_; k++)
              contactSeq_(i,  nStance_ + j * (nStance_ + nSwing_) + k) = true;
          }

          for(int j = 0; j < rest ; j++)
            contactSeq_(i, nStance_ + num_unit * (nStance_ + nSwing_) + j) = j >= nSwing_;

          step_elapsed_(i) = 1;
        } else {
          for(int j = 0; j < nSwing_ - step_elapsed_(i) && j < horizon_; j++)
            contactSeq_(i,j) = false;

          int num_unit = (horizon_ - (nSwing_ - step_elapsed_(i)) ) / (nStance_ + nSwing_);
          int rest = (horizon_ - (nSwing_ - step_elapsed_(i)) ) % (nStance_ + nSwing_);

          for(int j = 0; j < num_unit; j++) {
            for(int k = 0; k < nStance_; k++)
              contactSeq_(i,  nSwing_ - step_elapsed_(i) + j * (nStance_ + nSwing_) + k) = true;
            for(int k = nStance_; k < nSwing_ + nStance_; k++)
              contactSeq_(i,  nSwing_ - step_elapsed_(i) + j * (nStance_ + nSwing_) + k) = false;
          }

          for(int j = 0; j < rest ; j++)
            contactSeq_(i, nSwing_ - step_elapsed_(i) + num_unit * (nStance_ + nSwing_) + j) = j < nStance_;

          step_elapsed_(i)++;

        }
      }
    }
    last_contact_ = contactSeq_.col(0);

    for(int i = 0; i < 4; i++) {
      if(last_contact_(i)) {
        contactPhase_(i) = (step_elapsed_(i) - 1) / double(nStance_ - 1);
      }
      else {
        contactPhase_(i) = (step_elapsed_(i) - 1) / double(nSwing_ - 1);
      }

      isContact_(i) = last_contact_(i);
    }
  }

  Eigen::MatrixXi getContactSequence() {
    return contactSeq_;
  }

  Eigen::Vector4d getIsContact() {
    return isContact_;
  }
  Eigen::VectorXd getContactPhase() {
    return contactPhase_;
  }

  void reset() {
    contactSeq_.setZero(nLegs_, horizon_);
    last_contact_ << true, false, false, true;
    step_elapsed_ << 1, 1, 1, 1;
  }

 private:
  int horizon_;
  int nLegs_;
  int nStance_;
  int nSwing_;
  double dt_;
  double stanceTime_;
  double swingTime_;

  Eigen::Vector4i step_elapsed_;
  Eigen::Vector4i last_contact_;
  Eigen::Vector4d isContact_;
  Eigen::Vector4d contactPhase_;

  Eigen::MatrixXi contactSeq_;
};
}