from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from geometry_msgs.msg import PoseArray, Pose, PoseStamped

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Pose
from nav2_msgs.msg import Particle, ParticleCloud
import tf_transformations as tf

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

        self.get_logger().info(f"{odom_topic}")

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.particle_pub = self.create_publisher(ParticleCloud, "/particlecloud", 1)

        # Initialize the models
        self.get_logger().info("before init models")
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)
        self.get_logger().info("after init models")

        # Wait for map to be received before continuing
        while self.sensor_model.map is None:
            self.get_logger().info("Waiting for map...")
            rclpy.spin_once(self, timeout_sec=1.0)
        
        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        self.cur_time = self.get_clock().now()
        self.particles = None
        self.declare_parameter("num_particles", 200)
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.probabilities = np.ones(self.num_particles) / self.num_particles
        self.get_logger().info("before init particles")

        self.laser_counter = 0

        #self.initialize_particles()
        self.particles = np.zeros((self.num_particles, 3))

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
        if len(self.particles)==0:#no particles
            return

        full_range = np.array(msg.ranges)
        if len(full_range) == 0:
            return
        mask = (np.linspace(0, len(full_range)-1, 99)).astype(int)#self.sensor_model.num_beams_per_particle)).astype(int)
        actual_range = full_range[mask]

        self.probabilities = self.sensor_model.evaluate(self.particles, actual_range)
        self.probabilities = self.probabilities ** (1/3)
        self.probabilities = self.probabilities/sum(self.probabilities)

        #self.probabilities = None

        if self.probabilities is None:
            self.get_logger().info("no probabilities")
            return
        self.probabilities/=sum(self.probabilities)

        #self.get_logger().info(f"{self.probabilities}")
        #self.get_logger().info(f"{self.particles}")
        
        if (self.laser_counter % 100):
            index = np.random.choice(self.num_particles, self.num_particles, True, self.probabilities)
            self.particles = self.particles[index]

        self.laser_counter += 1

        self.publish_pose()

    def odom_callback(self, msg):
        #self.get_logger().info("odom callback")
        if len(self.particles)==0:
            return
        
        dt = (self.get_clock().now() - self.cur_time).nanoseconds * 1e-9
        self.cur_time = self.get_clock().now()
        x = msg.twist.twist.linear.x
        dx = -x*dt
        y = msg.twist.twist.linear.y
        dy = -y*dt
        theta = msg.twist.twist.angular.z
        dtheta = -theta*dt

        self.particles = self.motion_model.evaluate(self.particles,[dx,dy,dtheta])
        self.publish_pose()

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = tf.euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))[2]
        
        noise = np.random.multivariate_normal(np.zeros(3), np.eye(3), self.num_particles)

        self.particles = np.array([x,y,theta])+noise #np.random.uniform([x - 1, y - 1, theta - np.pi], [x + 1, y + 1, theta + np.pi], (self.num_particles, 3))
        self.publish_pose()

    def publish_pose(self):
        x_avg = np.average(self.particles[:,0], weights=self.probabilities)
        y_avg = np.average(self.particles[:,1], weights=self.probabilities)
        avg_theta_x = np.average(np.cos(self.particles[:, 2]), weights=self.probabilities)
        avg_theta_y = np.average(np.sin(self.particles[:, 2]), weights=self.probabilities)
        theta_avg = np.arctan2(avg_theta_y, avg_theta_x)
        
        self.get_logger().info(f'{x_avg=} {y_avg=} {theta_avg=}')

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "/map"
        # odom_msg.child_frame_id = self.particle_filter_frame
        odom_msg.pose.pose.position.x = x_avg
        odom_msg.pose.pose.position.y = y_avg
        odom_msg.pose.pose.position.z = 0.0
        quaternion = tf.quaternion_from_euler(0,0,theta_avg)
        odom_msg.pose.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])
        # odom_msg.pose.pose.orientation.z = np.sin(theta_avg / 2)
        # odom_msg.pose.pose.orientation.w = np.cos(theta_avg / 2)

        self.odom_pub.publish(odom_msg)

        # Publish particle cloud
        particle_msg = ParticleCloud()
        particle_msg.header.stamp = self.get_clock().now().to_msg()
        particle_msg.header.frame_id = "/map"
        
        for i, particle in enumerate(self.particles):
            p = Particle()
            p.pose.position.x = particle[0]
            p.pose.position.y = particle[1]
            quaternion = tf.quaternion_from_euler(0, 0, particle[2])
            p.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], 
                                         z=quaternion[2], w=quaternion[3])
            if self.probabilities is None:
                p.weight = float(5/len(self.particles))
            else:
                p.weight = float(self.probabilities[i])
            particle_msg.particles.append(p)
        
        self.particle_pub.publish(particle_msg)

    def initialize_particles(self, message):
        stata = self.sensor_model.occupancy_map
        
        if stata is not None:
            # Get clicked point in world coordinates
            clicked_x = message.pose.position.x
            clicked_y = message.pose.position.y
            
            # Get map info
            resolution = stata.info.resolution
            origin_x = stata.info.origin.position.x
            origin_y = stata.info.origin.position.y
            
            # Create rotation matrix from origin orientation
            origin_theta = tf.euler_from_quaternion((
                stata.info.origin.orientation.x,
                stata.info.origin.orientation.y,
                stata.info.origin.orientation.z,
                stata.info.origin.orientation.w))[2]
            
            # Create inverse rotation matrix to go from world to map coordinates
            inv_rotation_matrix = np.array([
                [np.cos(-origin_theta), -np.sin(-origin_theta)],
                [np.sin(-origin_theta), np.cos(-origin_theta)]
            ])
            
            # Convert clicked point to map coordinates
            clicked_world = np.array([[clicked_x - origin_x], [clicked_y - origin_y]])
            clicked_map = np.dot(inv_rotation_matrix, clicked_world)
            map_x = int((clicked_map[0] / resolution)[0])
            map_y = int((clicked_map[1] / resolution)[0])
            
            # Sample points from Gaussian distribution around clicked point
            std_dev_meters = 1.0  # 1 meter standard deviation
            std_dev_cells = int(std_dev_meters / resolution)
            
            x_samples = np.random.normal(map_x, std_dev_cells, self.num_particles)
            y_samples = np.random.normal(map_y, std_dev_cells, self.num_particles)
            
            # Clip to ensure points are within map bounds
            x_samples = np.clip(x_samples, 0, stata.info.width - 1)
            y_samples = np.clip(y_samples, 0, stata.info.height - 1)
            
            # Convert back to world coordinates
            coords = np.vstack((x_samples * resolution, y_samples * resolution))
            rotated_coords = np.dot(np.array([
                [np.cos(origin_theta), -np.sin(origin_theta)],
                [np.sin(origin_theta), np.cos(origin_theta)]
            ]), coords)
            
            world_x = rotated_coords[0] + origin_x
            world_y = rotated_coords[1] + origin_y
            
            # Generate random orientations
            theta_samples = np.random.uniform(-np.pi, np.pi, self.num_particles)
            
            # Combine into particle array
            self.particles = np.column_stack((world_x, world_y, theta_samples))
            self.probabilities = np.ones(self.num_particles) / self.num_particles
            
            self.get_logger().info(f"Initialized particles around clicked point ({clicked_x}, {clicked_y})")
        else:
            raise Exception("Map not available")

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
