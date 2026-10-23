#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <moveit_msgs/CartesianPoint.h>
#include <hrii_utils/ImpedanceValues.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/TransformStamped.h>

#include <zmq.hpp>
#include <Eigen/Dense>
#include <hrii_bm2pc_rope/trajectory_utils.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <array>
#include <boost/array.hpp>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

using Clock = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

namespace {
constexpr int kOptiN = 9;
constexpr int kGoalPoint = 8;  // which rope node to track (kOptiN-1 is max)
constexpr double kTaskTime = 4.0;  // 4s for eight; 2.5s for sin
constexpr double kDt = 0.001;
constexpr int kCtrPeriod = 15;  // 25
constexpr int kHorizon = 50;  // 30
constexpr int kVisRateHz = 20;
constexpr int kRecordTimeSec = 500;
constexpr int kOptiRateAssumed = 100;
constexpr int kRecordLen = kRecordTimeSec * kOptiRateAssumed;
constexpr float kOvertimeMsAction = 20.0f;

// === Trajectory type selection ===
enum class TrajType { EIGHT, EGG, SIN };
constexpr TrajType kTrajType = TrajType::EIGHT;  // <-- change this to select trajectory

// === Trajectory shape parameters ===
constexpr double kEightScaleX = 0.45 * 2.0 * 0.3;
constexpr double kEightScaleY = 0.65 * 2.0 * 0.35;
constexpr double kEggScaleX = 0.3/2.0 * 1.0;
constexpr double kEggScaleY = 0.6/2.0;
constexpr double kSinWidth = 0.3/2.0 * 1.2;
constexpr double kSinLength = 0.6;

constexpr bool kContinuousEight = true;
constexpr int kContinuousLoops = 3;
constexpr double kContinuousTaskTime = kContinuousLoops * kTaskTime * 0.9;
constexpr double kRampFraction = 0.3 / kContinuousLoops;

struct StateHeader {
  int32_t step;
  int64_t t_state_ns;
  uint32_t P;
  uint32_t H;
} __attribute__((packed));

struct ActionHeader {
  int32_t step;
  int64_t t_state_ns;
  float compute_ms;
} __attribute__((packed));

inline int64_t nowNsSteady() {
  return std::chrono::duration_cast<ns>(Clock::now().time_since_epoch()).count();
}

inline int64_t nowNsRos() {
  return ros::Time::now().toNSec();
}

struct PaperDataRow {
  double t_sec;
  float rope_pos[kOptiN][3];
  float goal_traj[3];
};

inline const char* trajTypeName(TrajType traj_type) {
  switch (traj_type) {
    case TrajType::EGG:
      return "egg";
    case TrajType::SIN:
      return "sin";
    case TrajType::EIGHT:
    default:
      return "eight";
  }
}

using trajectory_utils::halfDenseThenUniform;
using trajectory_utils::extendLastPoint;
using trajectory_utils::eightTrajectory;
using trajectory_utils::eightTrajectoryContinuous;
using trajectory_utils::eggTrajectory;
using trajectory_utils::sinTrajectory;

boost::array<double, 9> sqrtDamping(const boost::array<double, 9>& k) {
  boost::array<double, 9> b{};
  for (size_t i = 0; i < k.size(); ++i) b[i] = 2.0 * 0.7 * std::sqrt(k[i]);
  return b;
}

}  // namespace

class BM2PCFrankaClient {
 public:
  BM2PCFrankaClient()
      : nh_(),
        pnh_("~"),
        tf_buffer_(),
        tf_listener_(tf_buffer_),
        zmq_ctx_(1),
        sub_(zmq_ctx_, zmq::socket_type::sub),
        pub_(zmq_ctx_, zmq::socket_type::pub) {
    loadParams();
    setupRos();
  }

  ~BM2PCFrankaClient() {
    shutdown_ = true;
    try { sub_.close(); } catch (...) {}
    try { pub_.close(); } catch (...) {}
    try { zmq_ctx_.close(); } catch (...) {}
    if (recv_thread_.joinable()) recv_thread_.join();
  }

  void printDelaySummary() const {
    auto [mean_total, var_total] = computeMeanVar(delay_record_);
    auto [mean_compute, var_compute] = computeMeanVar(compute_delay_record_);
    auto [mean_network, var_network] = computeMeanVar(network_delay_record_);

    std::cout << "\n========== Delay Summary ==========\n";
    std::cout << "Total   : mean = " << mean_total
              << " ms, var = " << var_total << "\n";
    std::cout << "Compute : mean = " << mean_compute
              << " ms, var = " << var_compute << "\n";
    std::cout << "Network : mean = " << mean_network
              << " ms, var = " << var_network << "\n";
    std::cout << "Samples : " << delay_record_.size() << "\n";
    std::cout << "===================================\n";
  }

  std::string getPlotDataDir() const {
    std::string pkg_path = ros::package::getPath("hrii_bm2pc_rope");
    return pkg_path + "/plot_data";
  }

  std::string getTrajName() const {
    if (continuous_eight_) {
      return "continuous_eight";
    }
    return trajTypeName(kTrajType);
  }

  void saveData() {
    std::string output_dir = getPlotDataDir();
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    saveGoalTrajectoryCSV(output_dir);
    saveRopeStateCSV(output_dir);
    saveControlCSV(output_dir);
    saveDelayCSV(output_dir);
    saveBiasCSV(output_dir);
    savePaperDataCSV(output_dir);
    ROS_INFO_STREAM("All plot data saved to " << output_dir);
  }

  void launchPlotter() {
    std::string data_dir = getPlotDataDir();
    std::string cmd = "python " + ros::package::getPath("hrii_bm2pc_rope")
                      + "/scripts/plot_rope_data.py --data_dir " + data_dir;
    ROS_INFO_STREAM("Launching plotter: " << cmd);
    system(cmd.c_str());
  }

  void waitForFirstRopeState(double timeout_sec = 3.0) {
    ros::Time t0 = ros::Time::now();

    while (ros::ok()) {
      {
        std::lock_guard<std::mutex> lk(state_mtx_);
        if (!first_frame_ && current_time_ns_ > 0) {
          ROS_INFO("Received first OptiTrack rope state.");
          opti_ee_frame_bias_ = opti_pos_current_.row(0).transpose() - pos_goal_;
          return;
        }
      }

      if ((ros::Time::now() - t0).toSec() > timeout_sec) {
        ROS_WARN("Timeout waiting for first OptiTrack rope state. Continue anyway.");
        return;
      }

      ros::Duration(0.01).sleep();
    }
  }

  void setupTrajectory() {
    ros::Duration(0.1).sleep();

    Eigen::Matrix<float, kOptiN, 3> pos, vel;
    int64_t t_state_ns;
    getRopeState(pos, vel, t_state_ns);

    total_horizon_ = static_cast<int>(total_steps_ / kCtrPeriod) + last_point_repeat_;

    auto points = halfDenseThenUniform(
        total_horizon_ + 1 - last_point_repeat_,
        0.3,  // you need this 0.3
        2.0,
        0.0,
        2.0);

    Eigen::Vector3d bias = pos.row(kGoalPoint).cast<double>();

    ROS_INFO("==== Rope State used for setupTrajectory ====");
    for (int i = 0; i < kOptiN; ++i) {
      ROS_INFO_STREAM("P" << i << ": " << pos.row(i));
    }
    ROS_INFO_STREAM("bias point (P" << kGoalPoint << "): " << bias.transpose());
    ROS_INFO("============================================");

    if (continuous_eight_) {
      // Override total_steps_ and total_horizon_ for continuous mode
      total_steps_ = static_cast<int>(continuous_task_time_ / kDt);
      total_horizon_ = static_cast<int>(total_steps_ / kCtrPeriod) + last_point_repeat_;
      int n_points = total_horizon_ + 1 - last_point_repeat_;
      goal_traj_ = eightTrajectoryContinuous(
          n_points,
          continuous_loops_,
          ramp_fraction_,
          0.45 * 2.0 * 0.3,
          0.65 * 2.0 * 0.35,
          0.0,
          bias);
      goal_traj_ = extendLastPoint(goal_traj_, last_point_repeat_);
      ROS_INFO("Continuous eight-trajectory: %d loops, %.1fs, ramp=%.0f%%",
               continuous_loops_, continuous_task_time_, ramp_fraction_ * 100.0);
    } else {
      switch (kTrajType) {
        case TrajType::EGG:
          goal_traj_ = eggTrajectory(points, kEggScaleX, kEggScaleY, 0.3, 0.0, bias);
          break;
        case TrajType::SIN:
          goal_traj_ = sinTrajectory(points, kSinWidth, kSinLength, 0.0, bias);
          break;
        case TrajType::EIGHT:
        default:
          goal_traj_ = eightTrajectory(points, kEightScaleX, kEightScaleY, 0.0, 1, bias);
          break;
      }
      goal_traj_ = extendLastPoint(goal_traj_, last_point_repeat_);
    }

    ROS_INFO_STREAM("goal_traj size = " << goal_traj_.size());
    if (!goal_traj_.empty()) {
      ROS_INFO_STREAM("goal first = " << goal_traj_.front().transpose());
      ROS_INFO_STREAM("goal last  = " << goal_traj_.back().transpose());
      for (size_t i = 0; i < std::min<size_t>(5, goal_traj_.size()); ++i) {
        ROS_INFO_STREAM("goal[" << i << "] = " << goal_traj_[i].transpose());
      }

      double total_length = 0.0;
      double max_speed = 0.0;
      const double dt_traj = kCtrPeriod * kDt;
      for (size_t i = 1; i < goal_traj_.size(); ++i) {
        double seg = (goal_traj_[i] - goal_traj_[i - 1]).norm();
        total_length += seg;
        double spd = seg / dt_traj;
        if (spd > max_speed) max_speed = spd;
      }
      ROS_INFO_STREAM("goal traj total length = " << total_length << " m");
      ROS_INFO_STREAM("goal traj max speed    = " << max_speed << " m/s");
    }
  }

  void startRuntime() {
    setupZmq();

    // 1. start receiving action first
    recv_thread_ = std::thread(&BM2PCFrankaClient::zmqReceiveLoop, this);

    // 2. optional small wait for ZMQ link warmup
    ros::Duration(0.2).sleep();

    // 3. visualization
    if (enable_vis_) { setupVisualization();}

    // 4. publish impedance
    publishInitialImpedanceValues();

    // 5. start sending rope state to server
    state_send_timer_ = nh_.createTimer(ros::Duration(1.0 / send_rate_hz_), &BM2PCFrankaClient::stateSendTimerCb, this);

    // 6. start control loop
    control_timer_ = nh_.createTimer(ros::Duration(kDt), &BM2PCFrankaClient::controlTimerCb, this);

    // 7. visualization update timer
    vis_timer_ = nh_.createTimer(ros::Duration(1.0 / kVisRateHz), &BM2PCFrankaClient::visTimerCb, this);
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ros::Subscriber rope_state_sub_;
  ros::Subscriber robot_state_sub_;
  ros::Subscriber stop_sub_;
  ros::Publisher pose_pub_;
  ros::Publisher impedance_pub_;
  ros::Publisher marker_pub_;
  ros::Publisher marker_array_pub_;
  ros::Timer control_timer_;
  ros::Timer state_send_timer_;
  ros::Timer vis_timer_;

  zmq::context_t zmq_ctx_;
  zmq::socket_t sub_;
  zmq::socket_t pub_;
  std::thread recv_thread_;
  std::atomic<bool> shutdown_{false};

  std::string marker_link_name_;
  std::string origin_link_name_;
  std::string transf_publisher_name_;
  std::string impedance_publisher_name_;
  std::string server_ip_;
  bool enable_vis_{true};
  bool use_python_for_traj_{false};
  std::string python_traj_module_;
  bool continuous_eight_{kContinuousEight};
  int continuous_loops_{kContinuousLoops};
  double continuous_task_time_{kContinuousTaskTime};
  double ramp_fraction_{kRampFraction};

  geometry_msgs::PoseStamped target_pose_;
  std::mutex state_mtx_;
  std::mutex action_mtx_;
  std::mutex record_mtx_;
  std::mutex paper_data_mtx_;

  Eigen::Matrix<float, kOptiN, 3> opti_pos_current_ = Eigen::Matrix<float, kOptiN, 3>::Zero();
  Eigen::Matrix<float, kOptiN, 3> opti_pos_last_ = Eigen::Matrix<float, kOptiN, 3>::Zero();
  Eigen::Matrix<float, kOptiN, 3> opti_vel_ = Eigen::Matrix<float, kOptiN, 3>::Zero();
  Eigen::Matrix<float, kOptiN, 3> opti_vel_filter_ = Eigen::Matrix<float, kOptiN, 3>::Zero();
  Eigen::Matrix<float, kOptiN, 3> ab_pos_est_ = Eigen::Matrix<float, kOptiN, 3>::Zero();
  Eigen::Matrix<float, kOptiN, 3> ab_vel_est_ = Eigen::Matrix<float, kOptiN, 3>::Zero();
  Eigen::Matrix<float, kOptiN, 3> ab_residual_ = Eigen::Matrix<float, kOptiN, 3>::Zero();

  std::array<float, 3> robot_vel_{{0.f, 0.f, 0.f}};
  std::array<float, 3> robot_pos_{{0.f, 0.f, 0.f}};
  Eigen::Vector3f acc_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f acc_filter_goal_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f vel_goal_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f pos_goal_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f opti_ee_frame_bias_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f real_pos_1_ = Eigen::Vector3f::Zero();

  Eigen::Vector3f pos_min_{0.33f, -0.35f, 0.25f};
  Eigen::Vector3f pos_max_{0.63f,  0.35f, 0.55f};
  Eigen::Vector3f vel_min_{-0.3f, -1.0f, -0.3f};
  Eigen::Vector3f vel_max_{ 0.3f,  1.0f,  0.3f};

  bool first_frame_{true};
  bool done_with_record_{false};
  int temp_i_{0};
  int64_t current_time_ns_{0};
  double last_opti_time_sec_{0.0};
  int opti_count_slow_{0};

  int send_step_{-1};
  int step_franka_{0};
  int delay_count_{0};
  int control_overrun_count_{0};
  std::vector<double> delay_record_;
  std::vector<double> compute_delay_record_;
  std::vector<double> network_delay_record_;
  std::vector<Eigen::Vector3d> goal_traj_;

  int horizon_{kHorizon};
  int H_{kHorizon};
  int N_{kOptiN - 1};
  int P_{kOptiN};
  int total_steps_{static_cast<int>(kTaskTime / kDt)};
  int last_point_repeat_{0};
  int total_horizon_{0};
  double send_rate_hz_{1000.0 / kCtrPeriod};
  double alpha_{0.0};
  double opti_alpha_{0.28};
  double ab_alpha_{0.6};
  double ab_beta_{0.12};

  Eigen::MatrixXd rope_state_record_ =
      Eigen::MatrixXd::Zero(kRecordLen, 1 + 3 * kOptiN * 2);

  int record_interval_{30};
  int control_i_{0};
  int total_time_sec_action_{50};

  Eigen::MatrixXd target_traj_record_ = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::MatrixXd real_traj_record_   = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::MatrixXd target_vel_record_  = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::MatrixXd real_vel_record_    = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::VectorXd control_time_record_ = Eigen::VectorXd::Zero(100 * total_time_sec_action_);

  std::atomic<bool> recording_{false};
  std::vector<PaperDataRow> paper_data_rows_;

  void loadParams() {
    pnh_.param<std::string>("marker_link_name", marker_link_name_, "panda_hand_tcp");
    pnh_.param<std::string>("origin_link_name", origin_link_name_, "panda_link0");
    pnh_.param<std::string>("transf_publisher_name", transf_publisher_name_, "/target_pose");
    pnh_.param<std::string>("impedance_publisher_name", impedance_publisher_name_, "/impedance_values");
    pnh_.param<std::string>("server_ip", server_ip_, std::string("127.0.0.1"));
    pnh_.param<bool>("enable_vis", enable_vis_, true);
    pnh_.param<bool>("use_python_for_traj", use_python_for_traj_, false);
    pnh_.param<std::string>("python_traj_module", python_traj_module_, std::string("common.utils"));
  }

  void setupRos() {
    ros::Duration(1.0).sleep();
    target_pose_ = waitForInitialPose();

    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(transf_publisher_name_, 1);
    impedance_pub_ = nh_.advertise<hrii_utils::ImpedanceValues>(impedance_publisher_name_, 1);
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("rope_goal_vis", 1);
    marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("rope_nodes_vis", 1);

    rope_state_sub_ = nh_.subscribe(
        "optitrack/unidentified_markers", 1, &BM2PCFrankaClient::ropeStateCallback, this);
    robot_state_sub_ = nh_.subscribe(
        "endeffector_point_topic", 1,
        &BM2PCFrankaClient::robotStateCallback, this);
    stop_sub_ = nh_.subscribe("/stop_signal", 1, &BM2PCFrankaClient::stopCallback, this);

    pos_goal_ << target_pose_.pose.position.x,
                 target_pose_.pose.position.y,
                 target_pose_.pose.position.z;
    robot_pos_ = {pos_goal_.x(), pos_goal_.y(), pos_goal_.z()};

    ROS_INFO_STREAM("Initial position: " << pos_goal_.transpose());
  }

  geometry_msgs::PoseStamped waitForInitialPose() {
    geometry_msgs::PoseStamped pose;
    ROS_INFO_STREAM("Trying to get initial transformation of link \""
                    << marker_link_name_ << "\" wrt \"" << origin_link_name_ << "\"...");
    while (ros::ok()) {
      try {
        auto tf = tf_buffer_.lookupTransform(
            origin_link_name_, marker_link_name_, ros::Time(0), ros::Duration(0.5));
        pose.pose.orientation = tf.transform.rotation;
        pose.pose.position.x = tf.transform.translation.x;
        pose.pose.position.y = tf.transform.translation.y;
        pose.pose.position.z = tf.transform.translation.z;
        ROS_INFO("POSE_FOUND");
        return pose;
      } catch (const std::exception&) {
        ros::Duration(0.5).sleep();
      }
    }
    return pose;
  }

  void publishInitialImpedanceValues() {
    hrii_utils::ImpedanceValues msg;
    msg.header.stamp = ros::Time::now();

    double kx = 1200.0, ky = 1200.0, kz = 400.0;
    boost::array<double, 9> k_trans = {{kx, 0, 0, 0, ky, 0, 0, 0, kz}};
    boost::array<double, 9> k_rot   = {{50, 0, 0, 0, 50, 0, 0, 0, 50}};

    auto b_trans = sqrtDamping(k_trans);
    b_trans[0] = 2.0 * 0.6 * std::sqrt(kx) * 2.0;
    auto b_rot = sqrtDamping(k_rot);

    msg.translational_stiffness = k_trans;
    msg.rotational_stiffness = k_rot;
    msg.translational_damping = b_trans;
    msg.rotational_damping = b_rot;

    impedance_pub_.publish(msg);
  }

  void setupZmq() {
    const int hwm = 1;
    sub_.setsockopt(ZMQ_SUBSCRIBE, "", 0);
    sub_.setsockopt(ZMQ_RCVHWM, &hwm, sizeof(hwm));
    pub_.setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));

    sub_.connect("tcp://" + server_ip_ + ":6001");
    ROS_INFO_STREAM("[client] SUB connect tcp://" << server_ip_ << ":6001 (action in)");

    pub_.connect("tcp://" + server_ip_ + ":4532");
    ROS_INFO_STREAM("[client] PUB connect tcp://" << server_ip_ << ":4532 (state out)");

    ros::Duration(0.2).sleep();
  }

  void setupVisualization() {
    publishGoalTrajectoryMarker();
  }

  void robotStateCallback(const moveit_msgs::CartesianPoint::ConstPtr& msg) {
    robot_pos_[0] = static_cast<float>(msg->pose.position.x);
    robot_pos_[1] = static_cast<float>(msg->pose.position.y);
    robot_pos_[2] = static_cast<float>(msg->pose.position.z);
    robot_vel_[0] = msg->velocity.linear.x;
    robot_vel_[1] = msg->velocity.linear.y;
    robot_vel_[2] = msg->velocity.linear.z;
  }

  void stopCallback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) {
      ROS_WARN("Received stop signal. Shutting down...");
      ros::shutdown();
    }
  }

  void ropeStateCallback(const sensor_msgs::PointCloud::ConstPtr& msg) {
    auto cb_start = Clock::now();
    if (done_with_record_) return;
    if (static_cast<int>(msg->points.size()) != kOptiN) return;

    Eigen::Matrix<float, kOptiN, 3> raw_points;
    for (int i = 0; i < kOptiN; ++i) {
      raw_points(i, 0) = msg->points[i].x;
      raw_points(i, 1) = msg->points[i].y;
      raw_points(i, 2) = msg->points[i].z;
    }

    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      current_time_ns_ = msg->header.stamp.toNSec();
      double current_time_sec = msg->header.stamp.toSec();

      std::vector<int> idx(kOptiN);
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(),
                [&](int a, int b) { return raw_points(a, 2) > raw_points(b, 2); });

      Eigen::Matrix<float, kOptiN, 3> ordered;
      for (int i = 0; i < kOptiN; ++i) {
        ordered.row(i) = raw_points.row(idx[i]);
      }

      // Spike rejection: if the last marker jumped more than threshold since
      // last frame, treat this frame as a spike and drop it.
      if (!first_frame_) {
        constexpr float kSpikeThreshold = 0.1f;  // meters
        constexpr int kSpikeCheckIdx = kOptiN - 1;
        float d = (ordered.row(kSpikeCheckIdx) -
                   opti_pos_current_.row(kSpikeCheckIdx)).norm();
        if (d > kSpikeThreshold) {
          ROS_WARN_THROTTLE(
              1.0, "OptiTrack spike detected on point %d (%.3f m); dropping frame.",
              kSpikeCheckIdx, d);
          return;
        }
      }

      if (first_frame_) {
        opti_pos_current_ = ordered;
        opti_pos_last_ = ordered;
        ab_pos_est_ = ordered;
        ab_vel_est_.setZero();
        ab_residual_.setZero();
        opti_vel_filter_.setZero();
        first_frame_ = false;
        ROS_INFO("Received the first rope pos!");
      } else {
        opti_pos_last_ = opti_pos_current_;
        opti_pos_current_ = ordered;
        if (last_opti_time_sec_ > 0.0) {
          double dt = current_time_sec - last_opti_time_sec_;
          updateVelocity(dt);
        }
      }
      last_opti_time_sec_ = current_time_sec;
    }

    saveRopeRecord();

    auto cb_end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(cb_end - cb_start).count();
    if (ms >= 8.33) {
      ++opti_count_slow_;
      ROS_WARN_THROTTLE(1.0, "rope state process cannot maintain 120 Hz");
    }
  }

  void updateVelocity(double dt) {
    if (dt <= 1e-9) return;
    ab_pos_est_ = ab_pos_est_ + static_cast<float>(dt) * ab_vel_est_;
    ab_residual_ = opti_pos_current_ - ab_pos_est_;
    ab_pos_est_ = ab_pos_est_ + static_cast<float>(ab_alpha_) * ab_residual_;
    ab_vel_est_ = ab_vel_est_ + static_cast<float>(ab_beta_ / dt) * ab_residual_;
    opti_vel_filter_ = ab_vel_est_;
  }

  void getRopeState(Eigen::Matrix<float, kOptiN, 3>& pos,
                    Eigen::Matrix<float, kOptiN, 3>& vel,
                    int64_t& t_state_ns) {
    std::lock_guard<std::mutex> lk(state_mtx_);
    pos = opti_pos_current_;
    vel = opti_vel_filter_;
    t_state_ns = current_time_ns_;
  }

  void saveRopeRecord() {
    if (!recording_.load()) return;
    std::lock_guard<std::mutex> lk(record_mtx_);
    if (temp_i_ >= rope_state_record_.rows()) {
      done_with_record_ = true;
      return;
    }

    rope_state_record_(temp_i_, 0) = static_cast<double>(current_time_ns_) * 1e-9;
    for (int i = 0; i < kOptiN; ++i) {
      for (int j = 0; j < 3; ++j) {
        rope_state_record_(temp_i_, 1 + i * 3 + j) = opti_pos_current_(i, j);
        rope_state_record_(temp_i_, 1 + 3 * kOptiN + i * 3 + j) = opti_vel_filter_(i, j);
      }
    }
    ++temp_i_;
  }

  void recordPaperData(const Eigen::Matrix<float, kOptiN, 3>& pos,
                       const std::array<float, 3>& goal_traj_point) {
    PaperDataRow row{};
    row.t_sec = ros::Time::now().toSec();
    for (int i = 0; i < kOptiN; ++i) {
      for (int j = 0; j < 3; ++j) {
        row.rope_pos[i][j] = pos(i, j);
      }
    }
    for (int j = 0; j < 3; ++j) {
      row.goal_traj[j] = goal_traj_point[j];
    }

    std::lock_guard<std::mutex> lk(paper_data_mtx_);
    paper_data_rows_.push_back(row);
  }

  void stateSendTimerCb(const ros::TimerEvent&) {
    auto cb_start = Clock::now();
    if (goal_traj_.empty()) return;

    Eigen::Matrix<float, kOptiN, 3> pos, vel;
    int64_t t_state_ns;
    getRopeState(pos, vel, t_state_ns);



    Eigen::Vector3f robot_pos(robot_pos_[0], robot_pos_[1], robot_pos_[2]);
    Eigen::Vector3f robot_vel(robot_vel_[0], robot_vel_[1], robot_vel_[2]);

    // Convert all OptiTrack positions to robot frame
    for (int r = 0; r < kOptiN; ++r) {
      pos.row(r) -= opti_ee_frame_bias_.transpose();
    }

    // Row 0: override with robot's own EE state (more accurate, already in robot frame)
    pos.row(0) = robot_pos.transpose();
    vel.row(0) = robot_vel.transpose();

    // vel *= 1.1f;

    // ROS_INFO_STREAM_THROTTLE(1.0, "pos (robot frame):\n" << pos);

    int i = send_step_;
    std::vector<float> goal_flat;

    // Convert goal trajectory from OptiTrack frame to robot frame
    Eigen::Vector3f bias = opti_ee_frame_bias_;

    if (i < (total_horizon_ - horizon_)) {
      goal_flat.reserve(horizon_ * 3);
      for (int k = i + 1; k < i + 1 + horizon_; ++k) {
        const auto& g = goal_traj_[k];
        goal_flat.push_back(static_cast<float>(g.x()) - bias.x());
        goal_flat.push_back(static_cast<float>(g.y()) - bias.y());
        goal_flat.push_back(static_cast<float>(g.z()) - bias.z());
      }
    } else if (i < total_horizon_ - 1) {
      for (int k = i + 1; k < total_horizon_; ++k) {
        const auto& g = goal_traj_[k];
        goal_flat.push_back(static_cast<float>(g.x()) - bias.x());
        goal_flat.push_back(static_cast<float>(g.y()) - bias.y());
        goal_flat.push_back(static_cast<float>(g.z()) - bias.z());
      }
    } else {
      const auto& g = goal_traj_.back();
      goal_flat = {static_cast<float>(g.x()) - bias.x(),
                   static_cast<float>(g.y()) - bias.y(),
                   static_cast<float>(g.z()) - bias.z()};
      ros::shutdown();
    }

    std::array<float, 3> goal_traj_point = {goal_flat[0], goal_flat[1], goal_flat[2]};

    // ROS_INFO_STREAM("state delay = " << (nowNsRos() - t_state_ns) / 1e6 << " ms");
    StateHeader hdr{i, nowNsRos(), static_cast<uint32_t>(P_), static_cast<uint32_t>(H_)};

    zmq::message_t m0(sizeof(StateHeader));
    std::memcpy(m0.data(), &hdr, sizeof(StateHeader));

    std::vector<float> pos_flat;
    pos_flat.reserve(kOptiN * 3);
    for (int r = 0; r < pos.rows(); ++r) {
      for (int c = 0; c < pos.cols(); ++c) {
        pos_flat.push_back(pos(r, c));
      }
    }

    std::vector<float> vel_flat;
    vel_flat.reserve(kOptiN * 3);
    for (int r = 0; r < vel.rows(); ++r) {
      for (int c = 0; c < vel.cols(); ++c) {
        vel_flat.push_back(vel(r, c));
      }
    }

    zmq::message_t m1(sizeof(float) * pos_flat.size());
    std::memcpy(m1.data(), pos_flat.data(), sizeof(float) * pos_flat.size());

    zmq::message_t m2(sizeof(float) * vel_flat.size());
    std::memcpy(m2.data(), vel_flat.data(), sizeof(float) * vel_flat.size());

    zmq::message_t m3(sizeof(float) * goal_flat.size());
    if (!goal_flat.empty()) {
      std::memcpy(m3.data(), goal_flat.data(), sizeof(float) * goal_flat.size());
    }

    // frame4: goal_pos (pos_goal_ in robot frame, 3 floats)
    std::array<float, 3> goal_pos_flat = {pos_goal_.x(), pos_goal_.y(), pos_goal_.z()};
    zmq::message_t m4(sizeof(float) * 3);
    std::memcpy(m4.data(), goal_pos_flat.data(), sizeof(float) * 3);

    // frame5: goal_vel (vel_goal_ in robot frame, 3 floats)
    std::array<float, 3> goal_vel_flat = {vel_goal_.x(), vel_goal_.y(), vel_goal_.z()};
    zmq::message_t m5(sizeof(float) * 3);
    std::memcpy(m5.data(), goal_vel_flat.data(), sizeof(float) * 3);

    try {
      pub_.send(m0, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m1, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m2, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m3, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m4, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m5, zmq::send_flags::dontwait);
      recordPaperData(pos, goal_traj_point);
      ++send_step_;
    } catch (const zmq::error_t&) {
      ROS_WARN_THROTTLE(1.0, "ZMQ PUB send would block; dropping this frame");
    }

    auto cb_end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(cb_end - cb_start).count();
    if (ms >= 1000.0 / send_rate_hz_) {
      ROS_WARN_THROTTLE(1.0, "rope state sending cannot maintain target rate");
    }

    if (send_step_ <= 0) {
      ROS_INFO("Sleeping 3s for warming up");
      ros::Duration(5.0).sleep();
    }
  }

  void zmqReceiveLoop() {
    while (ros::ok() && !shutdown_) {
      try {
        // Server sends 3 frames: [ActionHeader, action(3f), real_pos_1(3f)]
        zmq::message_t part0, part1, part2;
        auto r0 = sub_.recv(part0, zmq::recv_flags::none);
        if (!r0) continue;

        // Receive remaining parts of this multipart message
        {
          int64_t more_flag = 1;
          size_t more_size = sizeof(more_flag);
          sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
          if (more_flag) sub_.recv(part1, zmq::recv_flags::none);
          sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
          if (more_flag) sub_.recv(part2, zmq::recv_flags::none);
          // Drain any extra parts
          sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
          while (more_flag) {
            zmq::message_t discard;
            sub_.recv(discard, zmq::recv_flags::none);
            sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
          }
        }

        // Drain stale messages, keep newest
        while (true) {
          zmq::message_t t0, t1, t2;
          try {
            auto rr = sub_.recv(t0, zmq::recv_flags::dontwait);
            if (!rr) break;
            int64_t more_flag = 1;
            size_t more_size = sizeof(more_flag);
            sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
            if (more_flag) sub_.recv(t1, zmq::recv_flags::none);
            sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
            if (more_flag) sub_.recv(t2, zmq::recv_flags::none);
            sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
            while (more_flag) {
              zmq::message_t discard;
              sub_.recv(discard, zmq::recv_flags::none);
              sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
            }
            part0 = std::move(t0);
            part1 = std::move(t1);
            part2 = std::move(t2);
          } catch (...) {
            break;
          }
        }

        if (part0.size() != sizeof(ActionHeader) ||
            part1.size() != sizeof(float) * 3 ||
            part2.size() != sizeof(float) * 3) {
          ROS_WARN("[client] bad reply (sizes: %zu, %zu, %zu)",
                   part0.size(), part1.size(), part2.size());
          continue;
        }

        ActionHeader hdr;
        std::memcpy(&hdr, part0.data(), sizeof(ActionHeader));

        Eigen::Vector3f action;
        std::memcpy(action.data(), part1.data(), sizeof(float) * 3);

        Eigen::Vector3f real_pos_1;
        std::memcpy(real_pos_1.data(), part2.data(), sizeof(float) * 3);

        {
          std::lock_guard<std::mutex> lk(action_mtx_);
          acc_ = action;
          real_pos_1_ = real_pos_1;
        }

        // Start recording on first ZMQ action received
        if (!recording_.load()) {
          recording_.store(true);
          ROS_INFO("First ZMQ action received — recording started.");
        }

        double total_time = (nowNsRos() - hdr.t_state_ns) / 1e6;
        double compute_time = static_cast<double>(hdr.compute_ms);
        double network_time = total_time - compute_time;

        delay_record_.push_back(total_time);
        compute_delay_record_.push_back(compute_time);
        network_delay_record_.push_back(network_time);
        
        // ROS_INFO_STREAM("step = " << hdr.step << ", action = " << action.transpose());

        if (step_franka_ != hdr.step) {
          ROS_WARN_STREAM("lost " << (hdr.step - step_franka_)
                          << " action steps! step_server=" << hdr.step
                          << " step_franka=" << step_franka_);
          step_franka_ = hdr.step + 1;
        } else {
          ++step_franka_;
        }

        if (total_time >= kOvertimeMsAction) {
          ++delay_count_;
          ROS_ERROR_STREAM("Overtime!!! Percent: "
                           << (100.0 * delay_count_ / std::max<size_t>(1, delay_record_.size()))
                           << "%, Total time: " << total_time
                           << "ms, Compute=" << hdr.compute_ms
                           << "ms, Network delay=" << (total_time - hdr.compute_ms));
        }
      } catch (const zmq::error_t& e) {
        if (!ros::ok() || shutdown_) break;
        ROS_WARN_THROTTLE(1.0, "ZMQ receive error: %s", e.what());
      }
    }
  }

  void controlTimerCb(const ros::TimerEvent& event) {
    auto start = Clock::now();

    float dt = static_cast<float>((event.current_real - event.last_real).toSec());
    if (dt <= 0.0f || dt > 0.1f) dt = static_cast<float>(kDt);

    Eigen::Vector3f acc_goal;
    {
      std::lock_guard<std::mutex> lk(action_mtx_);
      acc_goal = acc_;
    }  

    acc_filter_goal_ =
        static_cast<float>(alpha_) * acc_filter_goal_ +
        static_cast<float>(1.0 - alpha_) * acc_goal;

    vel_goal_ += acc_filter_goal_ * dt;
    vel_goal_ = vel_goal_.cwiseMin(vel_max_).cwiseMax(vel_min_);

    pos_goal_ += vel_goal_ * dt;

    // Eigen::Vector3f robot_pos(robot_pos_[0], robot_pos_[1], robot_pos_[2]);
    // pos_goal_ = robot_pos + vel_goal_ * static_cast<float>(kDt) * 50;
    // pos_goal_ = 0.999 * pos_goal_ + 0.001 * robot_pos;

    for (int j = 0; j < 3; ++j) {
      if (pos_goal_[j] >= pos_max_[j] && vel_goal_[j] > 0) {
        pos_goal_[j] = pos_max_[j];
        vel_goal_[j] = 0.0f;
      } else if (pos_goal_[j] <= pos_min_[j] && vel_goal_[j] < 0) {
        pos_goal_[j] = pos_min_[j];
        vel_goal_[j] = 0.0f;
      }
    }

    // int idx = std::max(0, std::min(send_step_, static_cast<int>(goal_traj_.size()) - 1));
    // const auto& g = goal_traj_[idx];

    // // pos_goal_.setZero();
    // pos_goal_[0] = g.x() - opti_ee_frame_bias_.x();
    // pos_goal_[1] = g.y() - opti_ee_frame_bias_.y();
    // pos_goal_[2] = g.z() - opti_ee_frame_bias_.z();
    
    target_pose_.header.frame_id = origin_link_name_;
    target_pose_.header.stamp = ros::Time::now();
    target_pose_.pose.position.x = pos_goal_.x();
    target_pose_.pose.position.y = pos_goal_.y();
    target_pose_.pose.position.z = pos_goal_.z();


    pose_pub_.publish(target_pose_);

    if (control_i_ % record_interval_ == 0 && control_i_ < total_time_sec_action_ * 1000) {
      try {
        int idx = control_i_ / record_interval_;
        control_time_record_(idx) = ros::Time::now().toSec();
        target_traj_record_.col(idx) = pos_goal_.cast<double>();
        real_traj_record_(0, idx) = robot_pos_[0];
        real_traj_record_(1, idx) = robot_pos_[1];
        real_traj_record_(2, idx) = robot_pos_[2];
        target_vel_record_.col(idx) = vel_goal_.cast<double>();
        real_vel_record_(0, idx) = robot_vel_[0];
        real_vel_record_(1, idx) = robot_vel_[1];
        real_vel_record_(2, idx) = robot_vel_[2];
      } catch (...) {
      }
    }

    ++control_i_;
    if (control_i_ >= total_time_sec_action_ * 1000) {
      ROS_WARN("Time's up!");
      ros::shutdown();
      return;
    }

    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (ms >= 1.0) {
      ++control_overrun_count_;
      ROS_ERROR_STREAM(control_overrun_count_ << ", " << ms
                       << " ms, Action: " << acc_goal.transpose()
                       << ", Action execution loop cannot maintain 1000 Hz !!!");
    }
  }

//   void controlTimerCb(const ros::TimerEvent&) {
//   auto start = Clock::now();

//   Eigen::Vector3f acc_goal;
//   {
//     std::lock_guard<std::mutex> lk(action_mtx_);
//     acc_goal = acc_;
//   }

//   Eigen::Vector3f robot_pos(robot_pos_[0], robot_pos_[1], robot_pos_[2]);
//   Eigen::Vector3f robot_vel(robot_vel_[0], robot_vel_[1], robot_vel_[2]);

//   // 1) low-pass filter remote acceleration command
//   acc_filter_goal_ =
//       static_cast<float>(alpha_) * acc_filter_goal_ +
//       static_cast<float>(1.0 - alpha_) * acc_goal;

//   // 2) integrate acceleration -> desired velocity
//   vel_goal_ += acc_filter_goal_ * static_cast<float>(kDt);
//   vel_goal_ = vel_goal_.cwiseMin(vel_max_).cwiseMax(vel_min_);

//   // 3) tracking error between internal goal and real robot
//   Eigen::Vector3f pos_err = pos_goal_ - robot_pos;
//   Eigen::Vector3f vel_err = vel_goal_ - robot_vel;

//   // 4) use a mild tracking feedback term
//   //    this helps reduce lag in the later phase
//   const float kp_track = 1.0f;   // try 0.5 ~ 3.0
//   const float kv_track = 0.2f;  // try 0.0 ~ 0.2

//   Eigen::Vector3f v_ref = vel_goal_ + kp_track * pos_err + kv_track * vel_err;
//   v_ref = v_ref.cwiseMin(vel_max_).cwiseMax(vel_min_);

//   // 5) integrate corrected velocity into position goal
//   pos_goal_ += v_ref * static_cast<float>(kDt);

//   // 6) weak anchoring to avoid runaway drift
//   const float anchor = 0.002f;   // smaller than before
//   pos_goal_ = (1.0f - anchor) * pos_goal_ + anchor * robot_pos;

//   // 7) workspace limits
//   for (int j = 0; j < 3; ++j) {
//     if (pos_goal_[j] >= pos_max_[j]) {
//       pos_goal_[j] = pos_max_[j];
//       if (vel_goal_[j] > 0.0f) vel_goal_[j] = 0.0f;
//     } else if (pos_goal_[j] <= pos_min_[j]) {
//       pos_goal_[j] = pos_min_[j];
//       if (vel_goal_[j] < 0.0f) vel_goal_[j] = 0.0f;
//     }
//   }

//   // 8) publish target pose
//   target_pose_.header.frame_id = origin_link_name_;
//   target_pose_.header.stamp = ros::Time::now();
//   target_pose_.pose.position.x = pos_goal_.x();
//   target_pose_.pose.position.y = pos_goal_.y();
//   target_pose_.pose.position.z = pos_goal_.z();
//   pose_pub_.publish(target_pose_);

//   // 9) record
//   if (control_i_ % record_interval_ == 0 && control_i_ < total_time_sec_action_ * 1000) {
//     try {
//       int idx = control_i_ / record_interval_;
//       target_traj_record_.col(idx) = pos_goal_.cast<double>();
//       real_traj_record_(0, idx) = robot_pos_[0];
//       real_traj_record_(1, idx) = robot_pos_[1];
//       real_traj_record_(2, idx) = robot_pos_[2];
//       target_vel_record_.col(idx) = v_ref.cast<double>();
//       real_vel_record_(0, idx) = robot_vel_[0];
//       real_vel_record_(1, idx) = robot_vel_[1];
//       real_vel_record_(2, idx) = robot_vel_[2];
//     } catch (...) {
//     }
//   }

//   ++control_i_;
//   if (control_i_ >= total_time_sec_action_ * 1000) {
//     ROS_WARN("Time's up!");
//     ros::shutdown();
//     return;
//   }

//   auto end = Clock::now();
//   double ms = std::chrono::duration<double, std::milli>(end - start).count();
//   if (ms >= 1.0) {
//     ++control_overrun_count_;
//     ROS_ERROR_STREAM(control_overrun_count_ << ", " << ms
//                      << " ms, Action: " << acc_goal.transpose()
//                      << ", Action execution loop cannot maintain 1000 Hz !!!");
//   }
// }

  void visTimerCb(const ros::TimerEvent&) {
    if (!enable_vis_) return;
    publishRopeMarkers();
    publishCurrentGoalPointMarker();
    publishEEPosGoalMarker();
    publishEEreal_pos_1_Marker();
  }

  void publishEEreal_pos_1_Marker() {
    visualization_msgs::Marker m;
    m.header.frame_id = origin_link_name_;
    m.header.stamp = ros::Time::now();
    m.ns = "ee_real_pos_1";
    m.id = 3;
    m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;

    m.pose.position.x = real_pos_1_.x() + opti_ee_frame_bias_.x();  
    m.pose.position.y = real_pos_1_.y() + opti_ee_frame_bias_.y();
    m.pose.position.z = real_pos_1_.z() + opti_ee_frame_bias_.z();

    // ROS_INFO_THROTTLE(1.0, "real_pos_1: [%.4f, %.4f, %.4f]",
    //                   real_pos_1_.x(), real_pos_1_.y(), real_pos_1_.z());

    m.pose.orientation.w = 1.0;

    m.scale.x = 0.02;
    m.scale.y = 0.02;
    m.scale.z = 0.02;

    m.color.r = 0.2;
    m.color.g = 0.8;
    m.color.b = 1.0;
    m.color.a = 0.8;

    marker_pub_.publish(m);
  }

  void publishGoalTrajectoryMarker() {
    visualization_msgs::Marker m;
    m.header.frame_id = origin_link_name_;
    m.header.stamp = ros::Time::now();
    m.ns = "goal_traj";
    m.id = 0;
    m.type = visualization_msgs::Marker::LINE_STRIP;
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.005;
    m.color.b = 1.0;
    m.color.a = 1.0;

    for (const auto& p : goal_traj_) {
      geometry_msgs::Point pt;
      pt.x = p.x();
      pt.y = p.y();
      pt.z = p.z();
      m.points.push_back(pt);
    }
    marker_pub_.publish(m);
  }

  void publishEEPosGoalMarker() {
    visualization_msgs::Marker m;
    m.header.frame_id = origin_link_name_;
    m.header.stamp = ros::Time::now();
    m.ns = "ee_pos_goal";
    m.id = 2;
    m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;

    m.pose.position.x = pos_goal_.x() + opti_ee_frame_bias_.x();  
    m.pose.position.y = pos_goal_.y() + opti_ee_frame_bias_.y();
    m.pose.position.z = pos_goal_.z() + opti_ee_frame_bias_.z();

    m.pose.orientation.w = 1.0;

    m.scale.x = 0.02;
    m.scale.y = 0.02;
    m.scale.z = 0.02;

    m.color.r = 1.0;
    m.color.g = 0.4;
    m.color.b = 0.7;
    m.color.a = 0.8;

    marker_pub_.publish(m);
  }

  void publishCurrentGoalPointMarker() {
    if (goal_traj_.empty()) return;

    int idx = std::max(0, std::min(send_step_, static_cast<int>(goal_traj_.size()) - 1));
    const auto& g = goal_traj_[idx];

    visualization_msgs::Marker m;
    m.header.frame_id = origin_link_name_;
    m.header.stamp = ros::Time::now();
    m.ns = "goal_point";
    m.id = 1;
    m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose.position.x = g.x();
    m.pose.position.y = g.y();
    m.pose.position.z = g.z();
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.02;
    m.color.g = 1.0;
    m.color.a = 1.0;

    marker_pub_.publish(m);
  }

  void publishRopeMarkers() {
    Eigen::Matrix<float, kOptiN, 3> pos, vel;
    int64_t t;
    getRopeState(pos, vel, t);

    visualization_msgs::Marker line;
    line.header.frame_id = origin_link_name_;
    line.header.stamp = ros::Time::now();
    line.ns = "rope";
    line.id = 0;
    line.type = visualization_msgs::Marker::LINE_STRIP;
    line.action = visualization_msgs::Marker::ADD;
    line.scale.x = 0.006;
    line.color.r = 1.0;
    line.color.g = 1.0;
    line.color.a = 1.0;

    visualization_msgs::MarkerArray ma;
    for (int i = 0; i < kOptiN; ++i) {
      geometry_msgs::Point p;
      p.x = pos(i, 0);
      p.y = pos(i, 1);
      p.z = pos(i, 2);
      line.points.push_back(p);

      visualization_msgs::Marker s;
      s.header = line.header;
      s.ns = "rope_nodes";
      s.id = i;
      s.type = visualization_msgs::Marker::SPHERE;
      s.action = visualization_msgs::Marker::ADD;
      s.pose.position = p;
      s.pose.orientation.w = 1.0;
      s.scale.x = s.scale.y = s.scale.z = 0.015;
      s.color.r = 1.0;
      s.color.a = 1.0;
      ma.markers.push_back(s);
    }

    marker_pub_.publish(line);
    marker_array_pub_.publish(ma);
  }

  std::pair<double, double> computeMeanVar(const std::vector<double>& v) const {
    if (v.empty()) return {0.0, 0.0};

    double sum = 0.0;
    for (double x : v) sum += x;
    double mean = sum / static_cast<double>(v.size());

    double var = 0.0;
    for (double x : v) {
      double d = x - mean;
      var += d * d;
    }
    var /= static_cast<double>(v.size());  // population variance

    return {mean, var};
  }

  void saveRopeStateCSV(const std::string& dir) {
    std::lock_guard<std::mutex> lk(record_mtx_);
    int rows = temp_i_;
    if (rows <= 0) {
      ROS_WARN("No rope state data to save.");
      return;
    }

    std::ofstream f(dir + "/rope_state.csv");
    f << std::setprecision(12);

    // header: time, p0_x,p0_y,p0_z, ..., p5_x,p5_y,p5_z, v0_x,v0_y,v0_z, ..., v5_x,v5_y,v5_z
    f << "time";
    for (int i = 0; i < kOptiN; ++i) f << ",p" << i << "_x,p" << i << "_y,p" << i << "_z";
    for (int i = 0; i < kOptiN; ++i) f << ",v" << i << "_x,v" << i << "_y,v" << i << "_z";
    f << "\n";

    int cols = rope_state_record_.cols();
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (c > 0) f << ",";
        f << rope_state_record_(r, c);
      }
      f << "\n";
    }
    ROS_INFO_STREAM("Saved rope_state.csv (" << rows << " rows)");
  }

  void saveGoalTrajectoryCSV(const std::string& dir) {
    if (goal_traj_.empty()) {
      ROS_WARN("No goal trajectory data to save.");
      return;
    }

    std::ofstream f(dir + "/goal_traj.csv");
    f << std::setprecision(12);
    f << "goal_x,goal_y,goal_z\n";

    for (const auto& point : goal_traj_) {
      f << point.x() << "," << point.y() << "," << point.z() << "\n";
    }

    ROS_INFO_STREAM("Saved goal_traj.csv (" << goal_traj_.size() << " rows)");
  }

  void saveControlCSV(const std::string& dir) {
    std::lock_guard<std::mutex> lk(record_mtx_);
    int cols = std::min(control_i_ / record_interval_,
                        static_cast<int>(real_traj_record_.cols()));
    if (cols <= 0) {
      ROS_WARN("No control data to save.");
      return;
    }

    std::ofstream f(dir + "/control_record.csv");
    f << std::setprecision(12);
    f << "time,goal_px,goal_py,goal_pz,real_px,real_py,real_pz,"
      << "goal_vx,goal_vy,goal_vz,real_vx,real_vy,real_vz\n";

    for (int i = 0; i < cols; ++i) {
      f << control_time_record_(i)
        << "," << target_traj_record_(0, i)
        << "," << target_traj_record_(1, i)
        << "," << target_traj_record_(2, i)
        << "," << real_traj_record_(0, i)
        << "," << real_traj_record_(1, i)
        << "," << real_traj_record_(2, i)
        << "," << target_vel_record_(0, i)
        << "," << target_vel_record_(1, i)
        << "," << target_vel_record_(2, i)
        << "," << real_vel_record_(0, i)
        << "," << real_vel_record_(1, i)
        << "," << real_vel_record_(2, i)
        << "\n";
    }
    ROS_INFO_STREAM("Saved control_record.csv (" << cols << " rows)");
  }

  void saveDelayCSV(const std::string& dir) {
    std::ofstream f(dir + "/delay_record.csv");
    f << std::setprecision(12);
    f << "total_ms,compute_ms,network_ms\n";
    size_t n = delay_record_.size();
    for (size_t i = 0; i < n; ++i) {
      f << delay_record_[i]
        << "," << (i < compute_delay_record_.size() ? compute_delay_record_[i] : 0.0)
        << "," << (i < network_delay_record_.size() ? network_delay_record_[i] : 0.0)
        << "\n";
    }
    ROS_INFO_STREAM("Saved delay_record.csv (" << n << " rows)");
  }

  void savePaperDataCSV(const std::string& dir) {
    std::vector<PaperDataRow> rows;
    {
      std::lock_guard<std::mutex> lk(paper_data_mtx_);
      if (paper_data_rows_.empty()) {
        ROS_WARN("No paper data to save.");
        return;
      }
      rows = paper_data_rows_;
    }

    const std::string path = dir + "/paper_data_" + getTrajName() + ".csv";
    std::ofstream f(path);
    if (!f) {
      ROS_ERROR_STREAM("Failed to open " << path);
      return;
    }

    f << std::setprecision(12);
    f << "time";
    for (int i = 0; i < kOptiN; ++i) {
      f << ",rope_p" << i << "_x"
        << ",rope_p" << i << "_y"
        << ",rope_p" << i << "_z";
    }
    f << ",goal_traj_x,goal_traj_y,goal_traj_z\n";

    for (const auto& row : rows) {
      f << row.t_sec;
      for (int i = 0; i < kOptiN; ++i) {
        for (int j = 0; j < 3; ++j) {
          f << "," << row.rope_pos[i][j];
        }
      }
      f << "," << row.goal_traj[0]
        << "," << row.goal_traj[1]
        << "," << row.goal_traj[2]
        << "\n";
    }

    ROS_INFO_STREAM("Saved " << path << " (" << rows.size() << " rows)");
  }

  void saveBiasCSV(const std::string& dir) {
    std::ofstream f(dir + "/bias.csv");
    f << std::setprecision(12);
    f << "bias_x,bias_y,bias_z,opti_n,goal_point\n";
    f << opti_ee_frame_bias_.x() << ","
      << opti_ee_frame_bias_.y() << ","
      << opti_ee_frame_bias_.z() << ","
      << kOptiN << ","
      << kGoalPoint << "\n";
    ROS_INFO("Saved bias.csv");
  }

};

int main(int argc, char** argv) {
  ros::init(argc, argv, "bm2pc_franka_client");

  BM2PCFrankaClient node;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  node.waitForFirstRopeState(3.0);
  node.setupTrajectory();
  node.startRuntime();

  ros::waitForShutdown();

  node.printDelaySummary();
  node.saveData();
  node.launchPlotter();
  return 0;
}
