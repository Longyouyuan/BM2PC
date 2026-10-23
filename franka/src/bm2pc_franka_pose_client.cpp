#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <moveit_msgs/CartesianPoint.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/TransformStamped.h>

#include <zmq.hpp>
#include <Eigen/Dense>

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
constexpr int kOptiN = 6;
constexpr int kGoalPoint = 0;  // kOptiN -1 or 0
constexpr double kTaskTime = 100.0;
constexpr double kDt = 0.001;
constexpr int kCtrPeriod = 25;
constexpr int kHorizon = 30;
constexpr int kVisRateHz = 20;
constexpr int kRecordTimeSec = 500;
constexpr int kOptiRateAssumed = 100;
constexpr int kRecordLen = kRecordTimeSec * kOptiRateAssumed;
constexpr float kOvertimeMsAction = 15.0f;

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

std::vector<double> halfDenseThenUniform(int N,
                                         double ratio = 0.5,
                                         double sharpness = 2.0,
                                         double start = 0.0,
                                         double end = 2.0) {
  std::vector<double> x;
  x.reserve(N);

  if (N <= 1) {
    x.push_back(start);
    return x;
  }

  int n_dense = static_cast<int>(N * ratio);
  n_dense = std::max(2, std::min(n_dense, N));
  int n_uniform = N - n_dense;

  const double base = std::exp(sharpness);

  std::vector<double> x_dense;
  x_dense.reserve(n_dense);
  for (int i = 0; i < n_dense; ++i) {
    double t = (n_dense == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(n_dense - 1);
    double val = ratio * (std::pow(base, t) - 1.0) / (base - 1.0);
    x_dense.push_back(val);
  }

  const double final_spacing =
      (n_dense >= 2) ? (x_dense[n_dense - 1] - x_dense[n_dense - 2]) : ratio;

  std::vector<double> x_uniform;
  x_uniform.reserve(n_uniform);
  for (int i = 1; i <= n_uniform; ++i) {
    x_uniform.push_back(x_dense.back() + final_spacing * static_cast<double>(i));
  }

  x.insert(x.end(), x_dense.begin(), x_dense.end());
  x.insert(x.end(), x_uniform.begin(), x_uniform.end());

  const double scale = (end - start) / x.back();
  for (double& v : x) {
    v = v * scale + start;
  }

  return x;
}

std::vector<Eigen::Vector3d> extendLastPoint(const std::vector<Eigen::Vector3d>& in, int N) {
  std::vector<Eigen::Vector3d> out = in;
  if (!out.empty()) {
    for (int i = 0; i < N; ++i) out.push_back(out.back());
  }
  return out;
}

std::vector<Eigen::Vector3d> eightTrajectory(const std::vector<double>& points,
                                             double scale_x = 0.25,
                                             double scale_y = 0.45,
                                             double z0 = 0.2,
                                             int loops = 1,
                                             const Eigen::Vector3d& bias = Eigen::Vector3d::Zero()) {
  std::vector<Eigen::Vector3d> traj;
  traj.reserve(points.size());

  if (points.empty()) return traj;

  const double last = points.back();
  const double a = -0.2;

  for (double p : points) {
    double u = p / last * (2.0 * M_PI * static_cast<double>(loops));
    double t = u + a * std::sin(2.0 * u);

    double sin_t = std::sin(t);
    double cos_t = std::cos(t);
    double denom = 1.0 + sin_t * sin_t;

    double y = -scale_x * cos_t / denom;
    double x =  scale_y * (sin_t * cos_t) / denom;
    double z = z0;

    traj.emplace_back(x, y, z);
  }

  Eigen::Vector3d start_shift = traj.front();
  start_shift.z() = 0.0;

  for (auto& p : traj) {
    p -= start_shift;
    p += bias;
  }

  return traj;
}

}  // namespace

// =========================================================================
// BM2PCFrankaPoseClient — uses joint_cartesian_position_controller
//   - No impedance publisher / subscriber
//   - Subscribes to endeffector_point (CartesianPoint) from the controller
//     for robot_pos_ / robot_vel_
// =========================================================================
class BM2PCFrankaPoseClient {
 public:
  BM2PCFrankaPoseClient()
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

  ~BM2PCFrankaPoseClient() {
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

  void saveData() {
    std::string output_dir = getPlotDataDir();
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    saveRopeStateCSV(output_dir);
    saveControlCSV(output_dir);
    saveDelayCSV(output_dir);
    saveBiasCSV(output_dir);
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
        0.2,
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

    goal_traj_ = eightTrajectory(
        points,
        0.45 * 2.0 * 0.3,
        0.65 * 2.0 * 0.35,
        0.0,
        1,
        bias);

    goal_traj_ = extendLastPoint(goal_traj_, last_point_repeat_);

    ROS_INFO_STREAM("goal_traj size = " << goal_traj_.size());
    if (!goal_traj_.empty()) {
      ROS_INFO_STREAM("goal first = " << goal_traj_.front().transpose());
      ROS_INFO_STREAM("goal last  = " << goal_traj_.back().transpose());
      for (size_t i = 0; i < std::min<size_t>(5, goal_traj_.size()); ++i) {
        ROS_INFO_STREAM("goal[" << i << "] = " << goal_traj_[i].transpose());
      }
    }
  }

  void startRuntime() {
    setupZmq();

    // 1. start receiving action first
    recv_thread_ = std::thread(&BM2PCFrankaPoseClient::zmqReceiveLoop, this);

    // 2. optional small wait for ZMQ link warmup
    ros::Duration(0.2).sleep();

    // 3. visualization
    if (enable_vis_) { setupVisualization(); }

    // 4. No impedance publication for position controller

    // 5. start sending rope state to server
    state_send_timer_ = nh_.createTimer(ros::Duration(1.0 / send_rate_hz_), &BM2PCFrankaPoseClient::stateSendTimerCb, this);

    // 6. start control loop
    control_timer_ = nh_.createTimer(ros::Duration(kDt), &BM2PCFrankaPoseClient::controlTimerCb, this);

    // 7. visualization update timer
    vis_timer_ = nh_.createTimer(ros::Duration(1.0 / kVisRateHz), &BM2PCFrankaPoseClient::visTimerCb, this);
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
  std::string server_ip_;
  bool enable_vis_{true};

  geometry_msgs::PoseStamped target_pose_;
  geometry_msgs::Quaternion initial_orientation_;  // store orientation from initial TF
  Eigen::Vector3f filtered_pos_goal_ = Eigen::Vector3f::Zero();
  float pose_filter_alpha_{0.00001f};  // low-pass filter coeff for target_pose (0=no filter, 1=no smoothing)
  std::mutex state_mtx_;
  std::mutex action_mtx_;
  std::mutex record_mtx_;

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
  double alpha_{0.99};  // careful about here
  double opti_alpha_{0.28};
  double ab_alpha_{0.6};
  double ab_beta_{0.04};

  Eigen::MatrixXd rope_state_record_ =
      Eigen::MatrixXd::Zero(kRecordLen, 1 + 3 * kOptiN * 2);

  int record_interval_{30};
  int control_i_{0};
  int total_time_sec_action_{500};

  Eigen::MatrixXd target_traj_record_ = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::MatrixXd real_traj_record_   = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::MatrixXd target_vel_record_  = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::MatrixXd real_vel_record_    = Eigen::MatrixXd::Zero(3, 100 * total_time_sec_action_);
  Eigen::VectorXd control_time_record_ = Eigen::VectorXd::Zero(100 * total_time_sec_action_);

  void loadParams() {
    pnh_.param<std::string>("marker_link_name", marker_link_name_, "panda_EE");
    pnh_.param<std::string>("origin_link_name", origin_link_name_, "panda_link0");
    pnh_.param<std::string>("transf_publisher_name", transf_publisher_name_, "/target_pose");
    pnh_.param<std::string>("server_ip", server_ip_, std::string("127.0.0.1"));
    pnh_.param<bool>("enable_vis", enable_vis_, true);
  }

  void setupRos() {
    ros::Duration(1.0).sleep();

    // Wait for initial EE pose from the controller's special_initial_pos topic
    ROS_INFO("Waiting for initial EE pose from controller (special_initial_pos)...");
    auto init_msg = ros::topic::waitForMessage<geometry_msgs::PoseStamped>(
        "special_initial_pos_topic", nh_, ros::Duration(30.0));
    if (init_msg) {
      target_pose_ = *init_msg;
      ROS_INFO("Received initial EE pose from controller");
    } else {
      ROS_WARN("Timeout waiting for special_initial_pos. Falling back to TF.");
      target_pose_ = waitForInitialPose();
    }

    // --- Send command ---
    std::cout << "(my code)target_pose_ in starting: ["
              << target_pose_.pose.position.x << ", "
              << target_pose_.pose.position.y << ", "
              << target_pose_.pose.position.z << ", "
              << target_pose_.pose.orientation.x << ", "
              << target_pose_.pose.orientation.y << ", "
              << target_pose_.pose.orientation.z << ", "
              << target_pose_.pose.orientation.w << "]\n";

    // Store initial orientation — position controller requires valid orientation every publish
    initial_orientation_ = target_pose_.pose.orientation;

    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(transf_publisher_name_, 1);
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("rope_goal_vis", 1);
    marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("rope_nodes_vis", 1);

    rope_state_sub_ = nh_.subscribe(
        "optitrack/unidentified_markers", 1, &BM2PCFrankaPoseClient::ropeStateCallback, this);

    // Subscribe to the endeffector_point topic published by joint_cartesian_position_controller
    robot_state_sub_ = nh_.subscribe(
        "endeffector_point_topic", 1,
        &BM2PCFrankaPoseClient::robotStateCallback, this);

    stop_sub_ = nh_.subscribe("/stop_signal", 1, &BM2PCFrankaPoseClient::stopCallback, this);

    pos_goal_ << target_pose_.pose.position.x,
                 target_pose_.pose.position.y,
                 target_pose_.pose.position.z;
    filtered_pos_goal_ = pos_goal_;
    robot_pos_ = {pos_goal_.x(), pos_goal_.y(), pos_goal_.z()};

    ROS_INFO_STREAM("Initial pos_goal_ (from controller current_command_): " << pos_goal_.transpose());
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

  // Subscribe to CartesianPoint from joint_cartesian_position_controller
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

  void stateSendTimerCb(const ros::TimerEvent&) {

    // filtered_pos_goal_ = Eigen::Map<Eigen::Vector3f>(robot_pos_.data());
    // pos_goal_ = filtered_pos_goal_;

    auto cb_start = Clock::now();
    if (goal_traj_.empty()) return;

    Eigen::Matrix<float, kOptiN, 3> pos, vel;
    int64_t t_state_ns;
    getRopeState(pos, vel, t_state_ns);

    Eigen::Vector3f robot_pos(robot_pos_[0], robot_pos_[1], robot_pos_[2]);
    Eigen::Vector3f robot_vel(robot_vel_[0], robot_vel_[1], robot_vel_[2]);
    pos.row(0) = (robot_pos + opti_ee_frame_bias_).transpose();
    vel.row(0) = robot_vel.transpose();

    int i = send_step_;
    std::vector<float> goal_flat;

    if (i < (total_horizon_ - horizon_)) {
      goal_flat.reserve(horizon_ * 3);
      for (int k = i + 1; k < i + 1 + horizon_; ++k) {
        const auto& g = goal_traj_[k];
        goal_flat.push_back(static_cast<float>(g.x()));
        goal_flat.push_back(static_cast<float>(g.y()));
        goal_flat.push_back(static_cast<float>(g.z()));
      }
    } else if (i < total_horizon_ - 1) {
      for (int k = i + 1; k < total_horizon_; ++k) {
        const auto& g = goal_traj_[k];
        goal_flat.push_back(static_cast<float>(g.x()));
        goal_flat.push_back(static_cast<float>(g.y()));
        goal_flat.push_back(static_cast<float>(g.z()));
      }
    } else {
      const auto& g = goal_traj_.back();
      goal_flat = {static_cast<float>(g.x()),
                   static_cast<float>(g.y()),
                   static_cast<float>(g.z())};
    }

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

    try {
      pub_.send(m0, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m1, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m2, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
      pub_.send(m3, zmq::send_flags::dontwait);
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
      ros::Duration(3.0).sleep();
    }
  }

  void zmqReceiveLoop() {
    while (ros::ok() && !shutdown_) {
      try {
        zmq::message_t part0, part1;
        auto r0 = sub_.recv(part0, zmq::recv_flags::none);
        if (!r0) continue;

        while (true) {
          zmq::recv_result_t more = sub_.recv(part1, zmq::recv_flags::none);
          if (!more) break;
          int64_t more_flag = 0;
          size_t more_size = sizeof(more_flag);
          sub_.getsockopt(ZMQ_RCVMORE, &more_flag, &more_size);
          if (!more_flag) break;
        }

        while (true) {
          zmq::message_t test0, test1;
          try {
            auto rr = sub_.recv(test0, zmq::recv_flags::dontwait);
            if (!rr) break;
            (void)sub_.recv(test1, zmq::recv_flags::none);
            part0 = std::move(test0);
            part1 = std::move(test1);
          } catch (...) {
            break;
          }
        }

        if (part0.size() != sizeof(ActionHeader) || part1.size() != sizeof(float) * 3) {
          ROS_WARN("[client] bad reply");
          continue;
        }

        ActionHeader hdr;
        std::memcpy(&hdr, part0.data(), sizeof(ActionHeader));

        Eigen::Vector3f action;
        std::memcpy(action.data(), part1.data(), sizeof(float) * 3);

        if (action.hasNaN()) {
          ROS_WARN("NaN action received, ignoring");
          continue;
        }

        {
          std::lock_guard<std::mutex> lk(action_mtx_);
          acc_ = action;
        }

        double total_time = (nowNsRos() - hdr.t_state_ns) / 1e6;
        double compute_time = static_cast<double>(hdr.compute_ms);
        double network_time = total_time - compute_time;

        delay_record_.push_back(total_time);
        compute_delay_record_.push_back(compute_time);
        network_delay_record_.push_back(network_time);

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

  void controlTimerCb(const ros::TimerEvent&) {
    auto start = Clock::now();

    Eigen::Vector3f acc_goal;
    {
      std::lock_guard<std::mutex> lk(action_mtx_);
      acc_goal = acc_;
    }
    acc_goal = 0.01f * acc_goal; 

    acc_filter_goal_ =
        static_cast<float>(alpha_) * acc_filter_goal_ +
        static_cast<float>(1.0 - alpha_) * acc_goal;

    vel_goal_ += acc_filter_goal_ * static_cast<float>(kDt);
    vel_goal_ = vel_goal_.cwiseMin(vel_max_).cwiseMax(vel_min_);

    pos_goal_ += vel_goal_ * static_cast<float>(kDt);

    for (int j = 0; j < 3; ++j) {
      if (pos_goal_[j] >= pos_max_[j] && vel_goal_[j] > 0) {
        pos_goal_[j] = pos_max_[j];
        vel_goal_[j] = 0.0f;
      } else if (pos_goal_[j] <= pos_min_[j] && vel_goal_[j] < 0) {
        pos_goal_[j] = pos_min_[j];
        vel_goal_[j] = 0.0f;
      }
    }

    target_pose_.header.frame_id = origin_link_name_;
    target_pose_.header.stamp = ros::Time::now();
    // Low-pass filter on published pose to avoid jerky commands
    filtered_pos_goal_ = pose_filter_alpha_ * pos_goal_ + (1.0f - pose_filter_alpha_) * filtered_pos_goal_;
    target_pose_.pose.position.x = filtered_pos_goal_.x();
    target_pose_.pose.position.y = filtered_pos_goal_.y();
    target_pose_.pose.position.z = filtered_pos_goal_.z();
    // Position controller requires valid orientation every publish
    target_pose_.pose.orientation = initial_orientation_;
    pose_pub_.publish(target_pose_);

    if (control_i_ % record_interval_ == 0 && control_i_ < total_time_sec_action_ * 1000) {
      try {
        int idx = control_i_ / record_interval_;
        control_time_record_(idx) = ros::Time::now().toSec();
        target_traj_record_.col(idx) = filtered_pos_goal_.cast<double>();
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

  void visTimerCb(const ros::TimerEvent&) {
    if (!enable_vis_) return;
    publishRopeMarkers();
    publishCurrentGoalPointMarker();
    publishEEPosGoalMarker();
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
    var /= static_cast<double>(v.size());

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

  void saveBiasCSV(const std::string& dir) {
    std::ofstream f(dir + "/bias.csv");
    f << std::setprecision(12);
    f << "bias_x,bias_y,bias_z,opti_n\n";
    f << opti_ee_frame_bias_.x() << ","
      << opti_ee_frame_bias_.y() << ","
      << opti_ee_frame_bias_.z() << ","
      << kOptiN << "\n";
    ROS_INFO("Saved bias.csv");
  }

};

int main(int argc, char** argv) {
  ros::init(argc, argv, "bm2pc_franka_pose_client");

  BM2PCFrankaPoseClient node;

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
