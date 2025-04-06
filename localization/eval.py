import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry
import csv
import time
from datetime import datetime

class PoseEvaluator(Node):
    def __init__(self):
        super().__init__('pose_evaluator')
        
        # Set up TF2 listener for ground truth
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribe to particle filter's pose estimate
        self.pf_sub = self.create_subscription(
            Odometry,
            '/pf/pose/odom',
            self.pf_callback,
            10
        )
        
        # Initialize data storage
        self.start_time = time.time()
        
        # Create CSV file with timestamp in name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'src/localization/logs/pose_eval_{timestamp}.csv'
        
        # Write CSV header
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'actual_x', 'actual_y', 'estimated_x', 'estimated_y', 'error_x', 'error_y'])
        
        # Create timer for periodic evaluation
        self.create_timer(0.05, self.evaluate)  # 10 Hz evaluation rate
        
        self.get_logger().info(f'Pose evaluator started. Saving to {self.csv_filename}')
        
        # Store latest PF estimate
        self.latest_pf_x = None
        self.latest_pf_y = None
    
    def pf_callback(self, msg):
        self.latest_pf_x = msg.pose.pose.position.x
        self.latest_pf_y = msg.pose.pose.position.y
    
    def evaluate(self):
        try:
            # Get actual position from TF
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
            
            # Get actual position
            actual_x = transform.transform.translation.x
            actual_y = transform.transform.translation.y
            
            # Get current timestamp
            current_time = time.time() - self.start_time
            
            # If we have a PF estimate, record the data
            if self.latest_pf_x is not None and self.latest_pf_y is not None:
                error_x = actual_x - self.latest_pf_x
                error_y = actual_y - self.latest_pf_y
                
                # Save to CSV
                with open(self.csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current_time,
                        actual_x,
                        actual_y,
                        self.latest_pf_x,
                        self.latest_pf_y
                    ])
        
        except Exception as e:
            self.get_logger().warning(f'Evaluation failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    evaluator = PoseEvaluator()
    rclpy.spin(evaluator)
    evaluator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()