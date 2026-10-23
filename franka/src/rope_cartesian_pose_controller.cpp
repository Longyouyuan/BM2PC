#include <hrii_bm2pc_rope/rope_cartesian_pose_controller.h>

#include <cmath>
#include <algorithm>
#include <iostream>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <eigen_conversions/eigen_msg.h>

namespace hrii_bm2pc_rope {

bool RopeCartesianPoseController::init(hardware_interface::RobotHW* robot_hardware,
                                       ros::NodeHandle& node_handle) {
  auto* cartesian_pose_interface =
      robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface == nullptr) {
    ROS_ERROR("RopeCartesianPoseController: Could not get Cartesian Pose interface");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("RopeCartesianPoseController: Could not get parameter arm_id");
    return false;
  }

  auto* model_interface = robot_hardware->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR("RopeCartesianPoseController: Could not get Model interface");
    return false;
  }

  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface->getHandle(arm_id + "_robot"));
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("RopeCartesianPoseController: Exception getting Cartesian handle: "
                     << e.what());
    return false;
  }

  // Read command shaping parameters.
  node_handle.param<double>("max_delta", max_delta_, 0.001);
  node_handle.param<double>("filter_alpha", filter_alpha_, 0.05);
  ROS_INFO_STREAM("RopeCartesianPoseController: max_delta = " << max_delta_
                  << ", filter_alpha = " << filter_alpha_);

  // Subscriber for desired EE pose (PoseStamped from rope_action node)
  desired_pose_sub_ = node_handle.subscribe(
      "desired_EE_pose", 1, &RopeCartesianPoseController::desiredPoseCallback, this);

  // Publisher for EE state (CartesianPoint, consumed by rope_action node)
  endeffector_point_pub_ =
      node_handle.advertise<moveit_msgs::CartesianPoint>("endeffector_point", 1);

  initial_pos_pub_ =
      node_handle.advertise<geometry_msgs::PoseStamped>("special_initial_pos", 1, true);

  // Initialize MatLogger2
  std::string arm_id_for_log = arm_id;
  std::string log_path = std::string(::getenv("HOME")) + "/log/" + arm_id_for_log + "_rope_cartesian_pose_controller";
  logger_ = XBot::MatLogger2::MakeLogger(log_path);
  appender_ = XBot::MatAppender::MakeInstance();
  appender_->add_logger(logger_);
  appender_->start_flush_thread();
  ROS_INFO_STREAM("RopeCartesianPoseController: logging to " << log_path);

  return true;
}

void RopeCartesianPoseController::starting(const ros::Time& /* time */) {
  const auto& state = cartesian_pose_handle_->getRobotState();
  // Use O_T_EE_d (last desired pose) for init — libfranka's motion generator
  // expects continuity from this, not from O_T_EE (measured).
  measured_pose_ = state.O_T_EE_d;

  {
    std::lock_guard<std::mutex> lk(pose_mutex_);
    desired_pose_ = measured_pose_;
    has_new_pose_ = false;
  }

  filtered_target_ = {measured_pose_[12], measured_pose_[13], measured_pose_[14]};

  // Set initial command to last desired pose so libfranka sees no discontinuity
  cartesian_pose_handle_->setCommand(measured_pose_);

  // Publish initial pose
  geometry_msgs::PoseStamped init_pose;
  init_pose.header.stamp = ros::Time::now();
  init_pose.header.frame_id = "panda_link0";
  init_pose.pose.position.x = measured_pose_[12];
  init_pose.pose.position.y = measured_pose_[13];
  init_pose.pose.position.z = measured_pose_[14];
  Eigen::Matrix3d rot;
  rot << measured_pose_[0], measured_pose_[4], measured_pose_[8],
         measured_pose_[1], measured_pose_[5], measured_pose_[9],
         measured_pose_[2], measured_pose_[6], measured_pose_[10];
  Eigen::Quaterniond q(rot);
  init_pose.pose.orientation.x = q.x();
  init_pose.pose.orientation.y = q.y();
  init_pose.pose.orientation.z = q.z();
  init_pose.pose.orientation.w = q.w();
  initial_pos_pub_.publish(init_pose);

  ROS_INFO_STREAM("RopeCartesianPoseController: started at EE pose [" 
      << measured_pose_[12] << ", " << measured_pose_[13] << ", " << measured_pose_[14] << "]");
}

void RopeCartesianPoseController::update(const ros::Time& /* time */,
                                         const ros::Duration& /* period */) {
  // Update robot state (closed-loop feedback)
  const auto& robot_state = cartesian_pose_handle_->getRobotState();
  measured_pose_ = robot_state.O_T_EE;

  // Grace period: echo last desired pose back for the first N cycles to let the robot settle
  static constexpr int kGraceCycles = 1000;  // ~100ms at 1kHz
  if (update_count_ < kGraceCycles) {
    cartesian_pose_handle_->setCommand(robot_state.O_T_EE_d);
    ++update_count_;
    if (update_count_ == kGraceCycles) {
      ROS_INFO("RopeCartesianPoseController: grace period finished, accepting commands");
    }
    return;
  }

  // --- Read desired pose ---
  std::array<double, 16> target;
  {
    std::lock_guard<std::mutex> lk(pose_mutex_);
    target = desired_pose_;
  }

  // --- Saturate translational delta per cycle ---
  // Elements 12,13,14 of the column-major 4x4 matrix are tx,ty,tz
  for (int i : {12, 13, 14}) {
    double delta = target[i] - measured_pose_[i];
    delta = std::max(-max_delta_, std::min(max_delta_, delta));
    target[i] = measured_pose_[i] + delta;
  }

  // Filter the final translation command that is actually sent to libfranka.
  for (int axis = 0; axis < 3; ++axis) {
    const int index = 12 + axis;
    filtered_target_[axis] = filter_alpha_ * filtered_target_[axis] +
                             (1.0 - filter_alpha_) * target[index];
    target[index] = filtered_target_[axis];
  }

  // Copy rotation directly from desired (elements 0-11 minus the translation column)
  // The rotation part is elements 0-2, 4-6, 8-10 of the 4x4 column-major matrix.
  // We don't saturate rotation — the rope_action node keeps orientation constant.

  //   // --- Send command ---
  // std::cout << "target in update: ["
  //           << target[0] << ", " << target[1] << ", "
  //           << target[2] << ", " << target[3] << ", "
  //           << target[4] << ", " << target[5] << ", "
  //           << target[6] << ", " << target[7] << ", "
  //           << target[8] << ", " << target[9] << ", "
  //           << target[10] << ", " << target[11] << ", "
  //           << target[12] << ", " << target[13] << ", "
  //           << target[14] << ", " << target[15] << "]"
  //           << std::endl;
            
  // --- Print error between command and measured ---
  ROS_INFO_THROTTLE(0.1, "cmd-meas delta: [%.6f, %.6f, %.6f]",
      target[12] - measured_pose_[12],
      target[13] - measured_pose_[13],
      target[14] - measured_pose_[14]);

  cartesian_pose_handle_->setCommand(target);


  // --- Log target translation ---
  Eigen::Vector3d target_translation(target[12], target[13], target[14]);
  logger_->add("time", ros::Time::now().toSec());
  logger_->add("target_translation", target_translation);
  Eigen::Vector3d current_translation(measured_pose_[12], measured_pose_[13], measured_pose_[14]);
  logger_->add("current_translation", current_translation);
  Eigen::Vector3d translation_error = target_translation - current_translation;
  logger_->add("translation_error", translation_error);
  // Save the full target pose as well for reference
  Eigen::VectorXd target_pose_vector(16);
  for (int i = 0; i < 16; ++i) {
    target_pose_vector(i) = target[i];
  }
  logger_->add("target_pose", target_pose_vector);
  Eigen::VectorXd current_pose_vector(16);
  for (int i = 0; i < 16; ++i) {
    current_pose_vector(i) = measured_pose_[i];
  }
  logger_->add("current_pose", current_pose_vector);
  Eigen::VectorXd pose_error_vector(16);
  for (int i = 0; i < 16; ++i) {
    pose_error_vector(i) = target[i] - measured_pose_[i];
  }
  logger_->add("pose_error", pose_error_vector);

  // --- Publish EE state ---
  publishEndeffectorPoint();
}

void RopeCartesianPoseController::publishEndeffectorPoint() {
  const auto& state = cartesian_pose_handle_->getRobotState();

  // EE position and orientation from measured state
  Eigen::Affine3d O_T_EE;
  O_T_EE.matrix() = Eigen::Map<const Eigen::Matrix<double, 4, 4>>(state.O_T_EE.data());
  Eigen::Vector3d position(O_T_EE.translation());
  Eigen::Quaterniond orientation(O_T_EE.linear());

  // EE twist: v = J(q) * dq
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(state.dq.data());
  Eigen::Matrix<double, 6, 1> EE_twist = jacobian * dq;

  moveit_msgs::CartesianPoint msg;
  tf::pointEigenToMsg(position, msg.pose.position);
  tf::quaternionEigenToMsg(orientation, msg.pose.orientation);
  tf::twistEigenToMsg(EE_twist, msg.velocity);
  endeffector_point_pub_.publish(msg);
}

void RopeCartesianPoseController::desiredPoseCallback(
    const geometry_msgs::PoseStamped::ConstPtr& msg) {
  const auto& p = msg->pose.position;

  {
    std::lock_guard<std::mutex> lk(pose_mutex_);
    desired_pose_[12] = p.x;
    desired_pose_[13] = p.y;
    desired_pose_[14] = p.z;
    has_new_pose_ = true;
  }
}

}  // namespace hrii_bm2pc_rope

PLUGINLIB_EXPORT_CLASS(hrii_bm2pc_rope::RopeCartesianPoseController,
                       controller_interface::ControllerBase)
