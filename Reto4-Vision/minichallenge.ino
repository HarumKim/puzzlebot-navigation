// Include Libraries
#include <micro_ros_arduino.h> 
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32.h>
#include <stdio.h>
#include <string.h>  // Required for strcmp()

// Declare ROS 2 node
rcl_node_t node;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_subscription_t subscriber1;
std_msgs__msg__Float32 angle_msg;

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}

#define ERRORPIN 15 // Error LED

#define servoPin 18
#define ANG_CHNL 0
#define ANG_FREQ 50
#define ANG_RES 8

#define ANG_MAX 180
#define ANG_MIN 0

float angleG = 0.0;

// Error Handling Function
void error_loop() {
  while (1) {
    digitalWrite(ERRORPIN, !digitalRead(ERRORPIN));
    delay(100);
  }
}

void angle_callback(const void * msgin) {
  const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
  angleG = constrain(msg->data, ANG_MIN, ANG_MAX);

  // Map angle to PWM pulse width (500-2500 microseconds)
  int dutyCycle = map(angleG, ANG_MIN, ANG_MAX, 5, 32);

  ledcWrite(ANG_CHNL, dutyCycle);
}

void setup() {
  set_microros_transports();

  // Initialize motor control pins
  pinMode(ERRORPIN, OUTPUT);
  pinMode(servoPin, OUTPUT);

  digitalWrite(ERRORPIN, HIGH);

  ledcSetup(ANG_CHNL, ANG_FREQ, ANG_RES);
  ledcAttachPin(servoPin, ANG_CHNL);

  ledcWrite(ANG_CHNL, 0);

  // Micro-ROS Setup
  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "ESP32", "", &support));

  RCCHECK(rclc_subscription_init_default(
    &subscriber1, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
    "angle"));

  // Create executor with 2 subscriptions
  RCCHECK(rclc_executor_init(&executor, &support.context,  1, &allocator));
  RCCHECK(rclc_executor_add_subscription(
    &executor, &subscriber1, &angle_msg, &angle_callback, ON_NEW_DATA));
}

void loop() {
  //delay(100);
  RCCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100)));
}
