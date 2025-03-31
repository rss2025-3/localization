from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import rclpy

import numpy as np

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.particles = None
        self.declare_parameter("num_particles", 200)
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.initialize_particles()

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def laser_callback(self, msg):
        probabilities = self.sensor_model.evaluate(self.particles, msg.ranges)
        self.resample_particles(probabilities)

        self.publish_pose()

    def odom_callback(self, msg):
        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        theta = msg.twist.twist.angular.z

        self.particles = self.motion_model.evaluate(self.particles,[x,y,theta])
        self.resample_particles()

        self.publish_pose()

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = 0

        self.particles = np.random.uniform([x - 1, y - 1, theta - np.pi], [x + 1, y + 1, theta + np.pi], (self.num_particles, 3))
        
    def resample_particles(self, probabilities=None):
        if probabilities is None:
            probabilities = np.ones(self.num_particles) / self.num_particles

        probabilities /= np.sum(probabilities)

        index = np.random.choice(self.num_particles, size=self.num_particles, p=probabilities)
        self.particles = self.particles[index]

    def publish_pose(self):
        x_avg = np.mean(self.particles[:,0])
        y_avg = np.mean(self.particles[:,1])
        avg_theta_x = np.mean(np.cos(self.particles[:, 2]))
        avg_theta_y = np.mean(np.sin(self.particles[:, 2]))
        theta_avg = np.arctan2(avg_theta_y, avg_theta_x)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "/map"
        odom_msg.child_frame_id = self.particle_filter_frame
        odom_msg.pose.pose.position.x = x_avg
        odom_msg.pose.pose.position.y = y_avg
        odom_msg.pose.pose.orientation.z = np.sin(theta_avg / 2)
        odom_msg.pose.pose.orientation.w = np.cos(theta_avg / 2)

        self.odom_pub.publish(odom_msg)

    def initialize_particles(self):
        self.particles = np.random.uniform(-1, 1, (self.num_particles, 2))
        self.particles = np.hstack([self.particles, np.random.uniform(-3/2*np.pi, 3/2*np.pi, (self.num_particles, 1))])

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
