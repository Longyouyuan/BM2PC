#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <string>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <franka/model.h>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <moveit_msgs/CartesianPoint.h>

#include <matlogger2/matlogger2.h>
#include <matlogger2/utils/mat_appender.h>

namespace hrii_bm2pc_rope {

class RopeCartesianPoseController
    : public controller_interface::MultiInterfaceController<
          franka_hw::FrankaPoseCartesianInterface,
          franka_hw::FrankaModelInterface,
          franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware,
            ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  void desiredPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void publishEndeffectorPoint();

  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;

  ros::Subscriber desired_pose_sub_;
  ros::Publisher endeffector_point_pub_;
  ros::Publisher initial_pos_pub_;

  std::array<double, 16> measured_pose_{};
  std::array<double, 16> desired_pose_{};
  std::mutex pose_mutex_;
  bool has_new_pose_{false};

  double max_delta_{0.001};
  double max_delta_vel_{0.000001};  // max velocity change per cycle (acceleration limit)
  double filter_alpha_{0.05};  // EMA filter: 0=pass-through, 0.99=heavy smoothing
  std::array<double, 3> prev_vel_{{0.0, 0.0, 0.0}};
  std::array<double, 3> filtered_target_{{0.0, 0.0, 0.0}};
  int update_count_{0};

  XBot::MatLogger2::Ptr logger_;
  XBot::MatAppender::Ptr appender_;
};

}  // namespace hrii_bm2pc_rope
