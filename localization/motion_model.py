import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.noise_std = (0.01, 0.01, 0.0) # (x_noise, y_noise, theta_noise)

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
     
        # Extract the odometry values
        dx, dy, dtheta = odometry

        # Extract noise standard deviations
        std_dx, std_dy, std_dtheta = self.noise_std

        # Generate noise for each particle's movement
        noise_dx = np.random.randn(particles.shape[0]) * std_dx
        noise_dy = np.random.randn(particles.shape[0]) * std_dy
        noise_dtheta = np.random.randn(particles.shape[0]) * std_dtheta

        # Create transformation matrices for position and orientation updates
        cos_theta = np.cos(particles[:, 2])
        sin_theta = np.sin(particles[:, 2])

        # Update the particles with noise using vectorized operations
        x_new = particles[:, 0] + (dx + noise_dx) * cos_theta - (dy + noise_dy) * sin_theta
        y_new = particles[:, 1] + (dx + noise_dx) * sin_theta + (dy + noise_dy) * cos_theta
        theta_new = particles[:, 2] + (dtheta + noise_dtheta)

        # Stack the new values to form the updated particles matrix
        particles = np.column_stack((x_new, y_new, theta_new))

        return particles
        

        ####################################
