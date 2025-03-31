import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.noise_std = (0.05, 0.05, 0.05) # (x_noise, y_noise, theta_noise)

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
        delta_x = odometry[0]
        delta_y = odometry[1]
        delta_theta = odometry[2]

        updated_positions = np.zeros((len(particles), 3))
        for i in range(len(particles)):
            x = particles[i][0]
            y =  particles[i][1]
            theta = particles[i][2]

            # Compute inverse rotation matrix
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R_inv = np.array([[cos_theta, sin_theta, 0],
                            [-sin_theta, cos_theta, 0], 
                            [0, 0, 1]])
            
            # Transform displacement to body frame
            body_frame = (R_inv @ np.array([x, y, theta]))
            print(np.array([delta_x, delta_y, delta_theta]).T[0])
            print(body_frame)
            body_frame = body_frame + np.array([delta_x, delta_y, delta_theta]).T[0]

            new_position = np.linalg.inv(R_inv) @ body_frame
        
            updated_positions[i, :] = new_position

        noise = np.random.normal(scale=self.noise_std, size=updated_positions.shape)
        updated_positions += noise
        return updated_positions
        

        ####################################
