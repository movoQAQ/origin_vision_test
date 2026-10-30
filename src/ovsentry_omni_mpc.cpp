#include <fmt/core.h>

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/bool.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/ovsentry_omni_logic.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"
#include "tools/recorder.hpp"

namespace
{
struct OmniCamConfig
{
  omniperception::CameraSpec spec;
  std::string dev_name;
  cv::Scalar color;
};

struct OmniInferenceResult
{
  OmniCamConfig cam;
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Armor> top_armor;
  double delta_yaw_deg = 0.0;
  double delta_pitch_deg = 0.0;
  double infer_ms = 0.0;
};

struct OmniCandidateFrame
{
  OmniInferenceResult result;
  std::chrono::steady_clock::time_point timestamp{};
  double base_big_yaw_rad = 0.0;
  bool has_base_big_yaw = false;
  std::optional<omniperception::OmniCandidate> candidate;
};

io::GimbalState make_buff_gimbal_state(const io::ROS2GimbalState & state)
{
  return {
    static_cast<float>(state.yaw), static_cast<float>(state.yaw_vel),
    static_cast<float>(state.pitch), static_cast<float>(state.pitch_vel),
    static_cast<float>(state.bullet_speed), 0};
}

std::string normalize_dev_name(const std::string & dev)
{
  if (dev.rfind("/dev/", 0) == 0) return dev.substr(5);
  return dev;
}

std::string slot_name(omniperception::OmniCameraSlot slot)
{
  switch (slot) {
    case omniperception::OmniCameraSlot::left:
      return "LEFT";
    case omniperception::OmniCameraSlot::right:
      return "RIGHT";
    case omniperception::OmniCameraSlot::back:
      return "BACK";
    case omniperception::OmniCameraSlot::extra:
      return "EXTRA";
    default:
      return "UNKNOWN";
  }
}

bool better_armor(const auto_aim::Armor & lhs, const auto_aim::Armor & rhs)
{
  if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
  return lhs.confidence > rhs.confidence;
}

std::optional<auto_aim::Armor> pick_top_armor(const std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return std::nullopt;
  auto best_it = armors.begin();
  for (auto it = std::next(armors.begin()); it != armors.end(); ++it) {
    if (better_armor(*it, *best_it)) best_it = it;
  }
  return *best_it;
}

struct ArmorTargetMask
{
  bool enabled = false;
  std::vector<uint8_t> ignored_ids;
};

uint8_t armor_name_to_nav_id(auto_aim::ArmorName name);

std::vector<uint8_t> deserialize_ignore_ids(const rclcpp::SerializedMessage & serialized_message)
{
  const auto & raw = serialized_message.get_rcl_serialized_message();
  eprosima::fastcdr::FastBuffer buffer(reinterpret_cast<char *>(raw.buffer), raw.buffer_length);
  eprosima::fastcdr::Cdr cdr(buffer);
  cdr.read_encapsulation();

  uint32_t count = 0;
  cdr >> count;

  std::vector<uint8_t> ids;
  ids.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint8_t id = 0;
    cdr >> id;
    ids.push_back(id);
  }

  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

class ArmorIgnoreSubscriber
{
public:
  ArmorIgnoreSubscriber(
    const std::string & topic = "/request_auto_aim_ignore",
    const std::string & msg_type = "rm_interfaces/msg/RequestAutoAimIgnore")
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
      self_initialized_ = true;
    }

    node_ = std::make_shared<rclcpp::Node>("auto_aim_ignore_subscriber");
    try {
      subscription_ = node_->create_generic_subscription(
        topic, msg_type, rclcpp::SensorDataQoS(),
        [this](const std::shared_ptr<rclcpp::SerializedMessage> message) {
          this->callback(message);
        });

      executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
      executor_->add_node(node_);
      spin_thread_ = std::thread([this]() { executor_->spin(); });
      tools::logger()->info("[AutoAimIgnore] Subscribed '{}' as '{}'.", topic, msg_type);
    } catch (const std::exception & e) {
      tools::logger()->warn("[AutoAimIgnore] Failed to subscribe '{}': {}", topic, e.what());
    }
  }

  ~ArmorIgnoreSubscriber()
  {
    if (executor_) executor_->cancel();
    if (spin_thread_.joinable()) spin_thread_.join();
    if (executor_ && node_) executor_->remove_node(node_);
    if (self_initialized_ && rclcpp::ok()) rclcpp::shutdown();
  }

  ArmorTargetMask mask() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return mask_;
  }

private:
  void callback(const std::shared_ptr<rclcpp::SerializedMessage> & message)
  {
    try {
      auto ids = deserialize_ignore_ids(*message);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        mask_.enabled = !ids.empty();
        mask_.ignored_ids = ids;
      }
      for (const auto id : ids) {
        tools::logger()->info("[AutoAimIgnore] ignore armor id: {}", static_cast<int>(id));
      }
    } catch (const std::exception & e) {
      tools::logger()->warn("[AutoAimIgnore] Failed to parse ignore ids: {}", e.what());
    }
  }

  mutable std::mutex mutex_;
  ArmorTargetMask mask_;
  bool self_initialized_ = false;
  std::shared_ptr<rclcpp::Node> node_;
  std::shared_ptr<rclcpp::GenericSubscription> subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
};

class BuffRequestSubscriber
{
public:
  explicit BuffRequestSubscriber(const std::string & topic = "/request_buff")
  {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
      self_initialized_ = true;
    }

    node_ = std::make_shared<rclcpp::Node>("buff_request_subscriber");
    subscription_ = node_->create_subscription<std_msgs::msg::Bool>(
      topic, 10, [this](const std_msgs::msg::Bool::SharedPtr message) {
        request_buff_.store(message->data);
        tools::logger()->info(
          "[BuffRequest] request_buff={}", message->data ? "true" : "false");
      });

    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
    spin_thread_ = std::thread([this]() { executor_->spin(); });
    tools::logger()->info("[BuffRequest] Subscribed '{}'.", topic);
  }

  ~BuffRequestSubscriber()
  {
    if (executor_) executor_->cancel();
    if (spin_thread_.joinable()) spin_thread_.join();
    if (executor_ && node_) executor_->remove_node(node_);
    if (self_initialized_ && rclcpp::ok()) rclcpp::shutdown();
  }

  bool requested() const { return request_buff_.load(); }

private:
  std::atomic_bool request_buff_{false};
  bool self_initialized_ = false;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
};

ArmorTargetMask read_nav_armor_target_mask(const ArmorIgnoreSubscriber & subscriber)
{
  return subscriber.mask();
}

ArmorTargetMask read_nav_armor_target_mask()
{
  ArmorTargetMask mask;
  mask.enabled = true;
  mask.ignored_ids = {};
  return mask;
}

void apply_armor_target_mask(std::list<auto_aim::Armor> & armors, const ArmorTargetMask & mask)
{
  if (!mask.enabled) return;
  if (mask.ignored_ids.empty()) return;

  armors.remove_if([&](const auto_aim::Armor & armor) {
    const auto id = armor_name_to_nav_id(armor.name);
    return id != 0 && std::find(mask.ignored_ids.begin(), mask.ignored_ids.end(), id) !=
                      mask.ignored_ids.end();
  });
}

std::pair<double, double> calc_delta_angle_deg(
  const auto_aim::Armor & armor, const OmniCamConfig & cam)
{
  const double delta_yaw =
    cam.spec.center_yaw_deg + (0.5 - armor.center_norm.x) * cam.spec.fov_h_deg;
  const double delta_pitch = (armor.center_norm.y - 0.5) * cam.spec.fov_v_deg;
  return {delta_yaw, delta_pitch};
}

double angular_distance_deg(double lhs_rad, double rhs_rad)
{
  return std::abs(tools::limit_rad(lhs_rad - rhs_rad)) * 57.3;
}

uint8_t armor_name_to_nav_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;
    case auto_aim::ArmorName::two:
      return 2;
    case auto_aim::ArmorName::three:
      return 3;
    case auto_aim::ArmorName::four:
      return 4;
    case auto_aim::ArmorName::five:
      return 5;
    case auto_aim::ArmorName::sentry:
      return 6;
    case auto_aim::ArmorName::outpost:
      return 7;
    case auto_aim::ArmorName::base:
      return 8;
    default:
      return 0;
  }
}

double nearest_continuous_yaw_rad(double wrapped_yaw_rad, double reference_yaw_rad)
{
  return reference_yaw_rad + tools::limit_rad(wrapped_yaw_rad - reference_yaw_rad);
}

double target_center_big_yaw_rad(const auto_aim::Target & target, double current_big_yaw_rad)
{
  const auto & ekf_x = target.ekf_x();
  const double wrapped_center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  return nearest_continuous_yaw_rad(wrapped_center_yaw, current_big_yaw_rad);
}

bool is_unlocked_outpost_target(const auto_aim::Target & target)
{
  return target.name == auto_aim::ArmorName::outpost && !target.outpost_layer_locked();
}

void apply_sentry_tracking_yaws(
  io::Command & command, const auto_aim::Target & target, double current_big_yaw_rad)
{
  if (!command.control) return;
  command.small_yaw = command.yaw;
  command.big_yaw =
    is_unlocked_outpost_target(target) ? current_big_yaw_rad
                                      : target_center_big_yaw_rad(target, current_big_yaw_rad);
  command.has_target_yaw = true;
}

void apply_abs_yaw_target(io::Command & command, double abs_yaw_rad)
{
  command.control = true;
  command.yaw = tools::limit_rad(abs_yaw_rad);
  command.big_yaw = abs_yaw_rad;
  command.small_yaw = command.yaw;
  command.has_target_yaw = true;
}

std::optional<omniperception::OmniCandidate> build_omni_candidate(
  const OmniInferenceResult & result, std::chrono::steady_clock::time_point timestamp,
  double base_big_yaw_rad)
{
  if (!result.top_armor.has_value()) return std::nullopt;

  const auto & armor = result.top_armor.value();
  omniperception::OmniCandidate candidate;
  candidate.slot = result.cam.spec.slot;
  candidate.armor_name = armor.name;
  candidate.priority = armor.priority;
  candidate.confidence = armor.confidence;
  candidate.timestamp = timestamp;
  candidate.base_big_yaw_rad = base_big_yaw_rad;
  candidate.abs_yaw_rad = base_big_yaw_rad + result.delta_yaw_deg / 57.3;
  apply_abs_yaw_target(candidate.command, candidate.abs_yaw_rad);
  candidate.command.armor_id = armor_name_to_nav_id(armor.name);
  candidate.command.pitch = 0.26;
  return candidate;
}

omniperception::AcceptedOmniTarget make_accepted_omni_target(
  const omniperception::OmniCandidate & candidate)
{
  omniperception::AcceptedOmniTarget accepted_target;
  accepted_target.slot = candidate.slot;
  accepted_target.armor_name = candidate.armor_name;
  accepted_target.priority = candidate.priority;
  accepted_target.confidence = candidate.confidence;
  accepted_target.timestamp = candidate.timestamp;
  accepted_target.base_big_yaw_rad = candidate.base_big_yaw_rad;
  accepted_target.abs_yaw_rad = candidate.abs_yaw_rad;
  accepted_target.command = candidate.command;
  return accepted_target;
}

bool same_candidate_frame(
  const OmniCandidateFrame & frame, const omniperception::OmniCandidate & candidate)
{
  if (!frame.candidate.has_value()) return false;
  return frame.candidate->slot == candidate.slot &&
         frame.candidate->armor_name == candidate.armor_name &&
         frame.candidate->timestamp == candidate.timestamp;
}

bool same_omni_target_continuation(
  const omniperception::AcceptedOmniTarget & lhs, const omniperception::AcceptedOmniTarget & rhs,
  double retarget_min_delta_deg)
{
  if (lhs.slot != rhs.slot) return false;
  if (lhs.armor_name != rhs.armor_name) return false;
  return angular_distance_deg(lhs.abs_yaw_rad, rhs.abs_yaw_rad) < retarget_min_delta_deg;
}

double horizon_distance(const auto_aim::Target & target)
{
  const auto & x = target.ekf_x();
  return std::sqrt(x[0] * x[0] + x[2] * x[2]);
}

void fill_nav_target_info(io::Command & command, const std::list<auto_aim::Target> & targets)
{
  command.armor_id = 0;
  command.vx = 0.0;
  command.vy = 0.0;
  command.horizon_distance = 0.0;

  if (!command.control || targets.empty()) return;

  const auto & target = targets.front();
  const auto x = target.ekf_x();
  command.armor_id = armor_name_to_nav_id(target.name);
  command.vx = x[1];
  command.vy = x[3];
  command.horizon_distance = horizon_distance(target);
}

void draw_omni_overlay(cv::Mat & img, const OmniInferenceResult & result)
{
  tools::draw_text(
    img,
    fmt::format("{} ({}) {:.1f}ms", slot_name(result.cam.spec.slot), result.cam.dev_name, result.infer_ms),
    {10, 30}, result.cam.color, 0.7, 2);

  if (!result.top_armor.has_value()) {
    tools::draw_text(img, "no target", {10, 60}, {120, 120, 120}, 0.7, 2);
    return;
  }

  const auto & armor = result.top_armor.value();
  tools::draw_points(img, armor.points, result.cam.color, 2);
  tools::draw_text(
    img,
    fmt::format(
      "{} pri={} conf={:.2f}", auto_aim::ARMOR_NAMES[armor.name],
      static_cast<int>(armor.priority), armor.confidence),
    {10, 60}, result.cam.color, 0.7, 2);
  tools::draw_text(
    img, fmt::format("delta yaw={:.1f} pitch={:.1f}", result.delta_yaw_deg, result.delta_pitch_deg),
    {10, 90}, result.cam.color, 0.7, 2);
}

void draw_auto_aim_overlay(
  cv::Mat & img, const std::list<auto_aim::Target> & targets, const auto_aim::Aimer & aimer,
  const auto_aim::Solver & solver)
{
  if (targets.empty()) return;

  const auto & target = targets.front();
  for (const auto & xyza : target.armor_xyza_list()) {
    const auto image_points =
      solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
    tools::draw_points(img, image_points, {0, 255, 0});
  }

  const auto & aim_point = aimer.debug_aim_point;
  const auto aim_image_points =
    solver.reproject_armor(aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
  tools::draw_points(img, aim_image_points, aim_point.valid ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0));
}

void draw_small_buff_overlay(
  cv::Mat & img, std::optional<auto_buff::PowerRune> & power_rune,
  const auto_buff::SmallTarget & target, const auto_buff::Solver & solver,
  const auto_aim::Plan & plan)
{
  tools::draw_text(img, "SMALL_BUFF", {10, 30}, {0, 255, 255}, 0.8, 2);
  tools::draw_text(
    img,
    fmt::format(
      "buff yaw={:.2f} pitch={:.2f} fire={}", plan.yaw * 57.3, plan.pitch * 57.3,
      plan.fire ? 1 : 0),
    {10, 60}, {0, 255, 255}, 0.8, 2);

  if (!power_rune.has_value()) {
    tools::draw_text(img, "no buff", {10, 90}, {120, 120, 120}, 0.8, 2);
    return;
  }

  auto & rune = power_rune.value();
  tools::draw_points(img, rune.target().points, {0, 255, 0}, 2);
  tools::draw_point(img, rune.target().center, {0, 0, 255}, 4);
  tools::draw_point(img, rune.r_center, {255, 0, 255}, 4);

  if (target.is_unsolve()) return;
  const auto image_points =
    solver.reproject_buff(target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0)),
                          target.ekf_x()[4], target.ekf_x()[5]);
  tools::draw_points(img, image_points, {255, 0, 0}, 2);
}

cv::Mat resize_for_view(const cv::Mat & img)
{
  cv::Mat resized;
  cv::resize(img, resized, {640, 360});
  return resized;
}
}  // namespace

const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
  "{left           | __yaml__                | 左前相机设备名(相对/dev)，默认读yaml.omni_left_path }"
  "{right          | __yaml__                | 右前相机设备名(相对/dev)，默认读yaml.omni_right_path }"
  "{back           | __yaml__                | 正后相机设备名(相对/dev)，默认读yaml.omni_back_path }"
  "{left_yaw       | 60                      | 左前相机中心yaw角(deg) }"
  "{right_yaw      | -60                     | 右前相机中心yaw角(deg) }"
  "{back_yaw       | 180                     | 正后相机中心yaw角(deg) }"
  "{fov_h          | 120                     | USB相机水平视场角(deg) }"
  "{fov_v          | 67                      | USB相机垂直视场角(deg) }"
  "{no-display     |                         | 关闭画面显示 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = tools::load(config_path);
  auto read_infer_device = [&](const std::string & key) {
      if (yaml[key]) return yaml[key].as<std::string>();
      if (yaml["device"]) return yaml["device"].as<std::string>();
      return std::string("UNKNOWN");
    };
  auto read_cam_path = [&](const std::string & cli_key, const std::string & yaml_key,
                           const std::string & fallback) {
      const auto cli_value = cli.get<std::string>(cli_key);
      if (!cli_value.empty() && cli_value != "__yaml__") return normalize_dev_name(cli_value);
      if (yaml[yaml_key]) return normalize_dev_name(yaml[yaml_key].as<std::string>());
      return normalize_dev_name(fallback);
    };
  auto read_cli_or_yaml_double = [&](const std::string & cli_key, const std::string & yaml_key,
                                     double fallback) {
      if (cli.has(cli_key)) return cli.get<double>(cli_key);
      if (yaml[yaml_key]) return yaml[yaml_key].as<double>();
      return fallback;
    };

  const std::string auto_aim_device = read_infer_device("auto_aim_device");
  const std::string omni_device = read_infer_device("omni_device");
  const double omni_retarget_cooldown_s =
    yaml["omni_retarget_cooldown_s"] ? yaml["omni_retarget_cooldown_s"].as<double>() : 2.5;
  const double omni_hold_release_tolerance_deg =
    yaml["omni_hold_release_tolerance_deg"] ? yaml["omni_hold_release_tolerance_deg"].as<double>() : 3.0;
  const double omni_retarget_min_delta_deg =
    yaml["omni_retarget_min_delta_deg"] ? yaml["omni_retarget_min_delta_deg"].as<double>() : 20.0;
  const double omni_command_timeout_s =
    yaml["omni_command_timeout_s"] ? yaml["omni_command_timeout_s"].as<double>() : 0.5;
  const double buff_lost_cmd_hold_s =
    yaml["buff_lost_cmd_hold_s"] ? yaml["buff_lost_cmd_hold_s"].as<double>() : 0.3;
  const auto omni_read_timeout = std::chrono::milliseconds(
    std::max(1, yaml["omni_camera_read_timeout_ms"] ? yaml["omni_camera_read_timeout_ms"].as<int>() : 10));
  const auto omni_retarget_cooldown = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_retarget_cooldown_s));
  const auto omni_command_timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(omni_command_timeout_s));
  const auto buff_lost_cmd_hold = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(std::max(0.0, buff_lost_cmd_hold_s)));
  const std::string auto_aim_ignore_topic = yaml["auto_aim_ignore_topic"]
                                              ? yaml["auto_aim_ignore_topic"].as<std::string>()
                                              : "/request_auto_aim_ignore";
  const std::string auto_aim_ignore_msg_type =
    yaml["auto_aim_ignore_msg_type"] ? yaml["auto_aim_ignore_msg_type"].as<std::string>()
                                     : "rm_interfaces/msg/RequestAutoAimIgnore";

  const double omni_fov_h_deg = read_cli_or_yaml_double("fov_h", "omni_fov_h_deg", 120.0);
  const double omni_fov_v_deg = read_cli_or_yaml_double("fov_v", "omni_fov_v_deg", 67.0);
  const OmniCamConfig left_cam_cfg{
    {omniperception::OmniCameraSlot::left, "left", read_cam_path("left", "omni_left_path", "video0"),
     read_cli_or_yaml_double("left_yaw", "omni_left_yaw_deg", 60.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("left", "omni_left_path", "video0"), {0, 255, 0}};
  const OmniCamConfig right_cam_cfg{
    {omniperception::OmniCameraSlot::right, "right", read_cam_path("right", "omni_right_path", "video2"),
     read_cli_or_yaml_double("right_yaw", "omni_right_yaw_deg", -60.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("right", "omni_right_path", "video2"), {0, 255, 255}};
  const OmniCamConfig back_cam_cfg{
    {omniperception::OmniCameraSlot::back, "back", read_cam_path("back", "omni_back_path", "video4"),
     read_cli_or_yaml_double("back_yaw", "omni_back_yaw_deg", 180.0), omni_fov_h_deg, omni_fov_v_deg},
    read_cam_path("back", "omni_back_path", "video4"), {255, 200, 0}};

  tools::logger()->info(
    "[OVSentryOmniMPC] inference devices: auto_aim={} omni={}", auto_aim_device, omni_device);

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder(30);
  const bool display = !cli.has("no-display");
  constexpr bool yolo_debug = false;

  auto gimbal = std::make_unique<io::ROS2Gimbal>(config_path);
  ArmorIgnoreSubscriber armor_ignore_subscriber(auto_aim_ignore_topic, auto_aim_ignore_msg_type);
  BuffRequestSubscriber buff_request_subscriber;
  auto auto_aim_camera = std::make_unique<io::Camera>(config_path);

  auto_aim::YOLO yolo_auto(config_path, yolo_debug, "auto_aim_device");
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::Planner planner(config_path);
  omniperception::Decider decider(config_path);
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::Aimer buff_aimer(config_path);
  constexpr bool aimer_to_now = true;

  auto yolo_omni_left = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");
  auto yolo_omni_right = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");
  auto yolo_omni_back = std::make_unique<auto_aim::YOLO>(config_path, yolo_debug, "omni_device");

  io::USBCamera cam_left(left_cam_cfg.dev_name, config_path);
  io::USBCamera cam_right(right_cam_cfg.dev_name, config_path);
  io::USBCamera cam_back(back_cam_cfg.dev_name, config_path);
  cam_left.device_name = left_cam_cfg.spec.label;
  cam_right.device_name = right_cam_cfg.spec.label;
  cam_back.device_name = back_cam_cfg.spec.label;

  cv::Mat main_img, left_img, right_img, back_img;
  std::chrono::steady_clock::time_point main_timestamp, ts_left, ts_right, ts_back;
  std::optional<io::Command> omni_hold_command;
  std::optional<omniperception::AcceptedOmniTarget> session_accepted_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> cooldown_anchor_omni_target;
  std::optional<omniperception::AcceptedOmniTarget> active_omni_timeout_target;
  std::chrono::steady_clock::time_point omni_retarget_cooldown_deadline{};
  std::chrono::steady_clock::time_point active_omni_timeout_started_at{};
  std::optional<io::Command> buff_hold_command;
  std::chrono::steady_clock::time_point buff_last_control_at{};
  bool active_omni_timeout_running = false;
  bool prev_omni_mode = false;
  int frame_count = 0;

  while (!exiter.exit()) {
    try {
      auto_aim_camera->read(main_img, main_timestamp);
      if (main_img.empty()) continue;
    } catch (const std::exception & e) {
      tools::logger()->error("[OVSentryOmniMPC] main camera read failed: {}", e.what());
      continue;
    }

    frame_count++;
    Eigen::Quaterniond q = gimbal->imu_at_image(main_timestamp);
     recorder.record(main_img,q, main_timestamp);
    solver.set_R_gimbal2world(q);
    const auto gimbal_state = gimbal->state();
    const bool small_buff_mode = buff_request_subscriber.requested();
    
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto t0 = std::chrono::steady_clock::now();
    const auto armor_target_mask = read_nav_armor_target_mask(armor_ignore_subscriber);
    std::list<auto_aim::Armor> armors;
    std::list<auto_aim::Target> targets;
    std::string tracker_state = small_buff_mode ? "small_buff" : "idle";
    if (!small_buff_mode) {
      armors = yolo_auto.detect(main_img, frame_count);
      decider.armor_filter(armors);
      decider.set_priority(armors);
      apply_armor_target_mask(armors, armor_target_mask);
      targets = tracker.track(armors, main_timestamp);
      tracker_state = tracker.state();
    }
    auto t1 = std::chrono::steady_clock::now();
    const bool omni_mode = !small_buff_mode && tracker_state == "lost";

    std::optional<OmniInferenceResult> best_omni_result;
    std::optional<double> omni_target_abs_yaw_deg;
    std::optional<double> omni_candidate_abs_yaw_deg;
    std::optional<double> omni_candidate_base_big_yaw_deg;
    std::optional<double> omni_candidate_age_ms;
    std::optional<double> omni_candidate_delta_deg;
    std::optional<double> omni_target_error_deg;
    std::optional<double> omni_selected_confidence;
    std::optional<int> omni_selected_priority;
    std::optional<std::string> omni_selected_slot;
    bool omni_hold_applied = false;
    bool omni_retarget_blocked = false;
    bool omni_retarget_cd_active = false;
    bool omni_same_target_continuation = false;
    bool omni_target_reached = false;
    bool omni_cmd_timeout_active = false;
    bool omni_cmd_timed_out = false;
    std::string omni_block_reason = "none";
    double omni_cmd_elapsed_ms = 0.0;
    double omni_retarget_remaining_ms = 0.0;
    const auto now = std::chrono::steady_clock::now();
    io::Command command{false, false, 0.0, 0.0};
    std::optional<auto_buff::PowerRune> buff_power_rune;
    auto_aim::Plan buff_plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
    double buff_detect_ms = 0.0;
    bool buff_target_ready = false;
    bool buff_command_held = false;

    auto clear_omni_timeout_session = [&]() {
        active_omni_timeout_target.reset();
        active_omni_timeout_started_at = std::chrono::steady_clock::time_point{};
        active_omni_timeout_running = false;
      };

    auto clear_omni_redirect_state = [&]() {
        omni_hold_command.reset();
        session_accepted_omni_target.reset();
        cooldown_anchor_omni_target.reset();
        omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
        clear_omni_timeout_session();
      };

    if (cooldown_anchor_omni_target.has_value() && now >= omni_retarget_cooldown_deadline) {
      cooldown_anchor_omni_target.reset();
      omni_retarget_cooldown_deadline = std::chrono::steady_clock::time_point{};
    }

    if (omni_mode && !prev_omni_mode) {
      clear_omni_redirect_state();
    } else if (!omni_mode && prev_omni_mode) {
      clear_omni_redirect_state();
    }

    if (small_buff_mode) {
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();

      buff_solver.set_R_gimbal2world(q);
      const auto t_buff0 = std::chrono::steady_clock::now();
      buff_power_rune = buff_detector.detect(main_img);
      const auto t_buff1 = std::chrono::steady_clock::now();
      buff_detect_ms = tools::delta_time(t_buff1, t_buff0) * 1e3;
      buff_solver.solve(buff_power_rune);
      buff_small_target.get_target(buff_power_rune, main_timestamp);
      buff_target_ready = !buff_small_target.is_unsolve();

      auto buff_target_copy = buff_small_target;
      buff_plan =
        buff_aimer.mpc_aim(buff_target_copy, main_timestamp, make_buff_gimbal_state(gimbal_state), true);

      command.control = buff_plan.control;
      command.shoot = buff_plan.fire;
      command.yaw = tools::limit_rad(buff_plan.yaw);
      command.pitch = buff_plan.pitch;
      command.big_yaw = nearest_continuous_yaw_rad(command.yaw, gimbal_state.big_yaw);
      command.small_yaw = command.yaw;
      command.has_target_yaw = command.control;

      if (command.control) {
        buff_hold_command = command;
        buff_last_control_at = now;
      } else if (
        buff_hold_command.has_value() &&
        now - buff_last_control_at <= buff_lost_cmd_hold) {
        // Keep navigation from reclaiming the gimbal for a brief detector/planner dropout.
        command = buff_hold_command.value();
        command.shoot = false;
        buff_command_held = true;
      }

      gimbal->send_mpc(
        command.control, command.shoot, command.big_yaw, command.small_yaw, command.pitch,
        buff_command_held ? 0.0 : buff_plan.yaw_vel,
        buff_command_held ? 0.0 : buff_plan.pitch_vel,
        buff_command_held ? 0.0 : buff_plan.yaw_acc,
        buff_command_held ? 0.0 : buff_plan.pitch_acc, 0, 0.0,
        0.0, 0.0);
    } else if (omni_mode) {
      buff_hold_command.reset();
      auto read_omni_frame = [&](io::USBCamera & camera, cv::Mat & img,
                                 std::chrono::steady_clock::time_point & ts,
                                 const OmniCamConfig & cam_cfg) {
          OmniCandidateFrame frame;
          frame.result.cam = cam_cfg;
          const bool ok = camera.read_with_timeout(img, ts, omni_read_timeout);
          if (!ok || img.empty()) {
            img.release();
            return frame;
          }
          frame.timestamp = ts;
          frame.base_big_yaw_rad = gimbal->big_yaw_at_image(ts);
          frame.has_base_big_yaw = true;
          return frame;
        };

      auto left_frame = read_omni_frame(cam_left, left_img, ts_left, left_cam_cfg);
      auto right_frame = read_omni_frame(cam_right, right_img, ts_right, right_cam_cfg);
      auto back_frame = read_omni_frame(cam_back, back_img, ts_back, back_cam_cfg);

      auto t_omni0 = std::chrono::steady_clock::now();
      if (left_frame.has_base_big_yaw && !left_img.empty()) {
        left_frame.result.armors = yolo_omni_left->detect(left_img, frame_count);
      }
      auto t_omni1 = std::chrono::steady_clock::now();
      if (right_frame.has_base_big_yaw && !right_img.empty()) {
        right_frame.result.armors = yolo_omni_right->detect(right_img, frame_count);
      }
      auto t_omni2 = std::chrono::steady_clock::now();
      if (back_frame.has_base_big_yaw && !back_img.empty()) {
        back_frame.result.armors = yolo_omni_back->detect(back_img, frame_count);
      }
      auto t_omni3 = std::chrono::steady_clock::now();

      auto finalize_frame = [&](OmniCandidateFrame & frame, const OmniCamConfig & cam_cfg,
                                double infer_ms) {
          frame.result.infer_ms = infer_ms;
          decider.armor_filter(frame.result.armors);
          decider.set_priority(frame.result.armors);
          apply_armor_target_mask(frame.result.armors, armor_target_mask);
          frame.result.top_armor = pick_top_armor(frame.result.armors);
          if (frame.result.top_armor.has_value()) {
            auto [dyaw, dpitch] = calc_delta_angle_deg(frame.result.top_armor.value(), cam_cfg);
            frame.result.delta_yaw_deg = dyaw;
            frame.result.delta_pitch_deg = dpitch;
            frame.candidate = build_omni_candidate(frame.result, frame.timestamp, frame.base_big_yaw_rad);
          }
        };

      finalize_frame(left_frame, left_cam_cfg, tools::delta_time(t_omni1, t_omni0) * 1e3);
      finalize_frame(right_frame, right_cam_cfg, tools::delta_time(t_omni2, t_omni1) * 1e3);
      finalize_frame(back_frame, back_cam_cfg, tools::delta_time(t_omni3, t_omni2) * 1e3);

      omni_retarget_cd_active =
        cooldown_anchor_omni_target.has_value() && now < omni_retarget_cooldown_deadline;
      if (omni_retarget_cd_active) {
        omni_retarget_remaining_ms =
          std::chrono::duration<double, std::milli>(omni_retarget_cooldown_deadline - now).count();
      }
      const auto reference_omni_target = omniperception::select_omni_retarget_reference_target(
        session_accepted_omni_target, cooldown_anchor_omni_target, omni_retarget_cd_active);

      std::vector<OmniCandidateFrame> candidate_frames;
      if (left_frame.candidate.has_value()) candidate_frames.push_back(left_frame);
      if (right_frame.candidate.has_value()) candidate_frames.push_back(right_frame);
      if (back_frame.candidate.has_value()) candidate_frames.push_back(back_frame);

      std::vector<omniperception::OmniCandidate> candidates;
      candidates.reserve(candidate_frames.size());
      for (const auto & frame : candidate_frames) candidates.push_back(frame.candidate.value());

      const auto selected_candidate = omniperception::select_omni_candidate(
        candidates, reference_omni_target, gimbal_state.big_yaw, omni_retarget_min_delta_deg);

      if (selected_candidate.has_value()) {
        omni_candidate_abs_yaw_deg = selected_candidate->abs_yaw_rad * 57.3;
        omni_candidate_base_big_yaw_deg = selected_candidate->base_big_yaw_rad * 57.3;
        omni_candidate_age_ms = tools::delta_time(now, selected_candidate->timestamp) * 1e3;
        omni_selected_confidence = selected_candidate->confidence;
        omni_selected_priority = static_cast<int>(selected_candidate->priority);
        omni_selected_slot = slot_name(selected_candidate->slot);

        const auto selected_frame = std::find_if(
          candidate_frames.begin(), candidate_frames.end(), [&](const OmniCandidateFrame & frame) {
            return same_candidate_frame(frame, selected_candidate.value());
          });
        if (selected_frame != candidate_frames.end()) best_omni_result = selected_frame->result;

        const auto decision = omniperception::evaluate_omni_retarget(
          selected_candidate.value(), reference_omni_target, gimbal_state.big_yaw,
          omni_retarget_cd_active, omni_retarget_min_delta_deg);
        omni_candidate_delta_deg = decision.candidate_delta_deg;
        omni_same_target_continuation = decision.same_target_continuation;

        if (decision.accept) {
          command = selected_candidate->command;
          omni_hold_command = command;
          const auto accepted_target = make_accepted_omni_target(selected_candidate.value());
          session_accepted_omni_target = accepted_target;
          if (
            !active_omni_timeout_running || !active_omni_timeout_target.has_value() ||
            !same_omni_target_continuation(
              active_omni_timeout_target.value(), accepted_target, omni_retarget_min_delta_deg)) {
            active_omni_timeout_started_at = now;
            active_omni_timeout_running = true;
          }
          active_omni_timeout_target = accepted_target;
          omni_target_abs_yaw_deg = command.big_yaw * 57.3;
          if (omniperception::should_start_omni_retarget_cooldown(
                decision, omni_retarget_min_delta_deg)) {
            cooldown_anchor_omni_target = accepted_target;
            omni_retarget_cooldown_deadline = now + omni_retarget_cooldown;
            omni_retarget_cd_active = true;
            omni_retarget_remaining_ms = omni_retarget_cooldown_s * 1e3;
          }
        } else if (reference_omni_target.has_value()) {
          command = reference_omni_target->command;
          omni_hold_command = command;
          omni_target_abs_yaw_deg = command.big_yaw * 57.3;
          omni_retarget_blocked = true;
          omni_block_reason = decision.block_reason;
        }
      } else if (omni_hold_command.has_value()) {
        const double target_error_deg = angular_distance_deg(omni_hold_command->big_yaw, gimbal_state.big_yaw);
        if (target_error_deg > omni_hold_release_tolerance_deg) {
          command = omni_hold_command.value();
          omni_target_abs_yaw_deg = command.big_yaw * 57.3;
          omni_hold_applied = true;
        } else {
          omni_hold_command.reset();
        }
      } else {
        omni_hold_command.reset();
        clear_omni_timeout_session();
      }

      if (command.control && command.has_target_yaw) {
        if (active_omni_timeout_running && active_omni_timeout_target.has_value()) {
          omni_cmd_timeout_active = true;
          omni_cmd_elapsed_ms =
            std::chrono::duration<double, std::milli>(now - active_omni_timeout_started_at).count();
        }

        const double target_error_deg = angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
        if (target_error_deg > omni_hold_release_tolerance_deg) {
          if (
            active_omni_timeout_running &&
            (now - active_omni_timeout_started_at) > omni_command_timeout) {
            tools::logger()->warn(
              "[OVSentryOmniMPC] omni command timed out after {:.0f}ms without reaching target yaw",
              omni_cmd_elapsed_ms);
            command = io::Command{false, false, 0.0, 0.0};
            omni_target_abs_yaw_deg.reset();
            omni_hold_applied = false;
            omni_cmd_timed_out = true;
            omni_cmd_timeout_active = false;
            clear_omni_redirect_state();
          }
        } else {
          clear_omni_timeout_session();
        }
      } else {
        clear_omni_timeout_session();
      }

      if (command.control && command.has_target_yaw) {
        omni_target_error_deg = angular_distance_deg(command.big_yaw, gimbal_state.big_yaw);
        omni_target_reached = omni_target_error_deg.value() <= omni_hold_release_tolerance_deg;
        if (omni_hold_command.has_value() && omni_target_reached) {
          omni_hold_command.reset();
        }
      }

      const double omni_big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
      const double omni_small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;
      gimbal->send_mpc(
        command.control, command.shoot, omni_big_yaw, omni_small_yaw, command.pitch,
        0.0, 0.0, 0.0, 0.0, static_cast<uint8_t>(command.armor_id), 0.0, 0.0, 0.0);
    } else {
      left_img.release();
      right_img.release();
      back_img.release();
      clear_omni_redirect_state();
      buff_hold_command.reset();

      const bool armor_acquiring = tracker_state == "detecting";
      if (armor_acquiring) {
        // Do not steer from a just-initialized EKF. Hold control so navigation cannot patrol away
        // while the tracker collects min_detect_count_ observations.
        command.control = true;
        command.shoot = false;
        command.yaw = gimbal_state.yaw;
        command.pitch = -gimbal_state.pitch;
        command.big_yaw = gimbal_state.big_yaw;
        command.small_yaw = gimbal_state.yaw;
        command.has_target_yaw = true;
      } else {
        command = aimer.aim(targets, main_timestamp, gimbal->bullet_speed(), aimer_to_now);
        if (command.control && !targets.empty()) {
          apply_sentry_tracking_yaws(command, targets.front(), gimbal_state.big_yaw);
        }
        command.shoot = shooter.shoot(command, aimer, targets, ypr, tracker_state == "tracking");
        fill_nav_target_info(command, targets);
      }

      const bool outpost_convergence = 
      !targets.empty() && targets.front().name == auto_aim::ArmorName::outpost &&
      (!targets.front().convergened() || targets.front().diverged());
      if (!armor_acquiring && outpost_convergence) {
        command = io::Command{false, false, 0.0, 0.0};
      }
      
      const bool unlocked_outpost =
        !targets.empty() && is_unlocked_outpost_target(targets.front());
      double small_yaw_vel = 0.0;
      double pitch_vel = 0.0;
      double small_yaw_acc = 0.0;
      double pitch_acc = 0.0;
      if (command.control && !armor_acquiring && !targets.empty() && !unlocked_outpost) {
        const auto mpc_plan = planner.plan(targets.front(), gimbal->bullet_speed());
        if (mpc_plan.control) {
          small_yaw_vel = mpc_plan.yaw_vel;
          pitch_vel = mpc_plan.pitch_vel;
          small_yaw_acc = mpc_plan.yaw_acc;
          pitch_acc = mpc_plan.pitch_acc;
        }
      }

      const double big_yaw = command.has_target_yaw ? command.big_yaw : command.yaw;
      const double small_yaw = command.has_target_yaw ? command.small_yaw : command.yaw;
      gimbal->send_mpc(
        command.control, command.shoot, big_yaw, small_yaw, command.pitch, small_yaw_vel,
        pitch_vel, small_yaw_acc, pitch_acc, static_cast<uint8_t>(command.armor_id), command.vx,
        command.vy, command.horizon_distance);
    }

    nlohmann::json data;
    data["mode"] = small_buff_mode ? 2 : (omni_mode ? 1 : 0);
    data["gimbal_mode"] = io::MODES[static_cast<int>(gimbal->mode())];
    data["armor_num"] = armors.size();
    data["tracker_state"] = tracker_state;
    data["armor_acquiring"] = (!small_buff_mode && tracker_state == "detecting") ? 1 : 0;
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_small_yaw"] = gimbal_state.yaw * 57.3;
    data["gimbal_big_yaw"] = gimbal_state.big_yaw * 57.3;
    data["bullet_speed"] = gimbal->bullet_speed();
    data["mpc_control"] = command.control ? 1 : 0;
    data["mpc_fire"] = command.shoot ? 1 : 0;
    data["mpc_yaw"] = (command.has_target_yaw ? command.small_yaw : command.yaw) * 57.3;
    data["mpc_pitch"] = command.pitch * 57.3;
    data["target_armor_id"] = static_cast<int>(command.armor_id);
    data["target_vx"] = command.vx;
    data["target_vy"] = command.vy;
    data["horizon_distance"] = command.horizon_distance;
    data["aim_source"] = (!small_buff_mode && !omni_mode && command.control) ? aimer.debug_aim_point.source : -1;
    data["aim_armor_id"] =
      (!small_buff_mode && !omni_mode && command.control) ? aimer.debug_aim_point.armor_id : -1;
    if (small_buff_mode) {
      data["buff_detected"] = buff_power_rune.has_value() ? 1 : 0;
      data["buff_target_ready"] = buff_target_ready ? 1 : 0;
      data["buff_command_held"] = buff_command_held ? 1 : 0;
      data["buff_detect_ms"] = buff_detect_ms;
      data["buff_yaw"] = buff_plan.yaw * 57.3;
      data["buff_pitch"] = buff_plan.pitch * 57.3;
      data["buff_yaw_vel"] = buff_plan.yaw_vel * 57.3;
      data["buff_pitch_vel"] = buff_plan.pitch_vel * 57.3;
      if (buff_power_rune.has_value()) {
        data["buff_r_yaw"] = buff_power_rune->ypd_in_world[0] * 57.3;
        data["buff_r_pitch"] = buff_power_rune->ypd_in_world[1] * 57.3;
        data["buff_r_distance"] = buff_power_rune->ypd_in_world[2];
        data["buff_roll"] = buff_power_rune->ypr_in_world[2] * 57.3;
      }
      if (buff_target_ready) {
        const auto x = buff_small_target.ekf_x();
        data["buff_target_roll"] = x[5] * 57.3;
        data["buff_target_spd"] = x[6] * 57.3;
      }
    }
    if (!targets.empty()) {
      const auto & target = targets.front();
      data["target_name"] = auto_aim::ARMOR_NAMES[target.name];
      data["outpost_layer_locked"] = target.outpost_layer_locked() ? 1 : 0;
      data["outpost_preview_ready"] = target.outpost_unlocked_prediction_ready() ? 1 : 0;
      const auto & ekf_data = target.ekf().data;
      if (ekf_data.count("init_preview_ready")) {
        data["init_preview_ready"] = ekf_data.at("init_preview_ready");
      }
      if (ekf_data.count("init_omega_margin")) {
        data["init_omega_margin"] = ekf_data.at("init_omega_margin");
      }
      if (ekf_data.count("init_margin")) {
        data["init_margin"] = ekf_data.at("init_margin");
      }
      if (target.name == auto_aim::ArmorName::outpost && aimer.debug_aim_point.valid) {
        const auto x = target.ekf_x();
        const double center_yaw = std::atan2(x[2], x[0]);
        data["outpost_aim_phase_deg"] =
          std::abs(tools::limit_rad(aimer.debug_aim_point.xyza[3] - center_yaw)) * 57.3;
      }
    }
    data["omni_yaw_hold"] = omni_hold_applied ? 1 : 0;
    data["omni_target_reached"] = omni_target_reached ? 1 : 0;
    data["omni_cmd_timeout_active"] = omni_cmd_timeout_active ? 1 : 0;
    data["omni_cmd_timed_out"] = omni_cmd_timed_out ? 1 : 0;
    data["omni_cmd_elapsed_ms"] = omni_cmd_elapsed_ms;
    data["omni_retarget_cd_active"] = omni_retarget_cd_active ? 1 : 0;
    data["omni_retarget_blocked"] = omni_retarget_blocked ? 1 : 0;
    data["omni_retarget_remaining_ms"] = omni_retarget_remaining_ms;
    data["omni_same_target_continuation"] = omni_same_target_continuation ? 1 : 0;
    data["omni_block_reason"] = omni_block_reason;
    if (omni_target_abs_yaw_deg.has_value()) data["omni_target_yaw"] = omni_target_abs_yaw_deg.value();
    if (omni_candidate_abs_yaw_deg.has_value()) data["omni_candidate_abs_yaw"] = omni_candidate_abs_yaw_deg.value();
    if (omni_candidate_base_big_yaw_deg.has_value()) {
      data["omni_candidate_base_big_yaw"] = omni_candidate_base_big_yaw_deg.value();
    }
    if (omni_candidate_age_ms.has_value()) data["omni_candidate_age_ms"] = omni_candidate_age_ms.value();
    if (omni_candidate_delta_deg.has_value()) data["omni_candidate_delta_deg"] = omni_candidate_delta_deg.value();
    if (omni_target_error_deg.has_value()) data["omni_target_error_deg"] = omni_target_error_deg.value();
    if (omni_selected_confidence.has_value()) data["omni_selected_confidence"] = omni_selected_confidence.value();
    if (omni_selected_priority.has_value()) data["omni_selected_priority"] = omni_selected_priority.value();
    if (omni_selected_slot.has_value()) data["omni_selected_slot"] = omni_selected_slot.value();
    data["yolo_time"] = tools::delta_time(t1, t0) * 1e3;
    plotter.plot(data);

    prev_omni_mode = omni_mode;
    if (!display) continue;

    if (small_buff_mode) {
      draw_small_buff_overlay(main_img, buff_power_rune, buff_small_target, buff_solver, buff_plan);
    } else {
      draw_auto_aim_overlay(main_img, targets, aimer, solver);
      tools::draw_text(main_img, fmt::format("[{}] mode={}", tracker_state, omni_mode ? "OMNI" : "MPC"),
        {10, 30}, {255, 255, 255}, 0.8, 2);
      tools::draw_text(main_img,
        fmt::format("mpc yaw={:.2f} pitch={:.2f} fire={}",
          (command.has_target_yaw ? command.small_yaw : command.yaw) * 57.3,
          command.pitch * 57.3, command.shoot ? 1 : 0),
        {10, 60}, {154, 50, 205}, 0.8, 2);
    }
    if (omni_target_abs_yaw_deg.has_value()) {
      tools::draw_text(main_img, fmt::format("omni target yaw={:.2f}", omni_target_abs_yaw_deg.value()),
        {10, 90}, {0, 255, 255}, 0.8, 2);
    }
    if (omni_retarget_cd_active) {
      tools::draw_text(
        main_img, fmt::format("omni retarget cd {:.0f}ms", omni_retarget_remaining_ms),
        {10, 120}, omni_retarget_blocked ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 220, 0), 0.8,
        2);
    }
    if (omni_cmd_timeout_active || omni_cmd_timed_out) {
      tools::draw_text(
        main_img,
        fmt::format(
          "omni cmd timeout {:.0f}/{:.0f}ms hit={}", omni_cmd_elapsed_ms,
          omni_command_timeout_s * 1e3, omni_cmd_timed_out ? 1 : 0),
        {10, 150}, omni_cmd_timed_out ? cv::Scalar(0, 120, 255) : cv::Scalar(180, 255, 180), 0.8,
        2);
    }
    if (omni_target_error_deg.has_value()) {
      tools::draw_text(
        main_img,
        fmt::format(
          "omni target err={:.1f} reached={}", omni_target_error_deg.value(),
          omni_target_reached ? 1 : 0),
        {10, 180}, {180, 255, 180}, 0.8, 2);
    }

    cv::Mat left_show = left_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : left_img.clone();
    cv::Mat right_show = right_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : right_img.clone();
    cv::Mat back_show = back_img.empty() ? cv::Mat::zeros(main_img.size(), main_img.type()) : back_img.clone();

    if (omni_mode && best_omni_result.has_value()) {
      const auto & best = best_omni_result.value();
      if (best.cam.spec.slot == omniperception::OmniCameraSlot::left) {
        draw_omni_overlay(left_show, best);
      } else if (best.cam.spec.slot == omniperception::OmniCameraSlot::right) {
        draw_omni_overlay(right_show, best);
      } else {
        draw_omni_overlay(back_show, best);
      }
    }

    cv::Mat main_small = resize_for_view(main_img);
    cv::Mat left_small = resize_for_view(left_show);
    cv::Mat right_small = resize_for_view(right_show);
    cv::Mat back_small = resize_for_view(back_show);
    cv::Mat top_row, bottom_row, canvas;
    cv::hconcat(main_small, left_small, top_row);
    cv::hconcat(right_small, back_small, bottom_row);
    cv::vconcat(top_row, bottom_row, canvas);
    cv::imshow("ovsentry_omni_mpc", canvas);
    if (cv::waitKey(1) == 'q') break;
  }

  gimbal->send_mpc(false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0);

  return 0;
}
