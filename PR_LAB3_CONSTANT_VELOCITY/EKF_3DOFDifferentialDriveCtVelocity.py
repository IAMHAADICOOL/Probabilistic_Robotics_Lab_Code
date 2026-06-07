from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from MapFeature import *
from Pose import *
class EKF_3DOFDifferentialDriveCtVelocity(GFLocalization, DR_3DOFDifferentialDrive, EKF):

    def __init__(self, kSteps, robot, x0, P0, *args):

        self.x0 = x0  # initial state x0=[x y z psi u v w r]^T
        self.P0 = P0  # initial covariance

        # this is required for plotting
        self.index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]
        self.robot = robot
        self.kSteps = kSteps
        # TODO: To be completed by the student
        # super().__init__(index, kSteps, robot, x0, P0, *args)
        super().__init__(self.index, self.kSteps, self.robot, self.x0, self.P0, *args)  # CORRECT
        self.zk_1_compass = None  # previous compass reading

    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        # the prediction step of the EKF is given by the motion model
        xk_bar = np.zeros((6, 1))  # predicted mean state vector

        # Predicted mean vector according to me
        # wk = np.zeros((3,1)) # velocity noise vector
        # for i in range(0, 3):
        #     cov = self.Pk_1[i+3, i+3]  # covariance of the predicted state vector
        #     wk[i, 0] = np.random.normal(0, np.sqrt(cov))  # sample noise from the covariance of the predicted state vector
        
        # xk_bar[0:3] = Pose3D.oplus(Pose3D(xk_1[0:3].reshape((3,1))), Pose3D((xk_1[3:6].reshape((3,1))*self.dt+0.5*self.dt**2*wk)))  # predicted pose
        # xk_bar[3:6] = xk_1[3:6].reshape((3,1)) + self.dt*wk  # predicted velocity

        # Predicted mean vector according to the slides
        vk_bar      = xk_1[3:6].reshape((3,1))
        uk          = vk_bar * self.dt
        etak_bar    = Pose3D.oplus(Pose3D(xk_1[0:3].reshape((3,1))), Pose3D(uk.reshape((3,1))))
        xk_bar = np.block([[etak_bar], [vk_bar]])
        # print("This is the predicted mean state vector xk_bar", xk_bar)
        # xk_bar[0:3] = Pose3D.oplus(Pose3D(xk_1[0:3].reshape((3,1))), Pose3D((xk_1[3:6].reshape((3,1))*self.dt)))  # predicted pose without noise
        # xk_bar[3:6] = xk_1[3:6].reshape((3,1))  # predicted velocity without noise
        return xk_bar

    def Jfx(self, xk_1, uk):
        # TODO: To be completed by the student
        vk_bar      = xk_1[3:6].reshape((3,1))
        uk          = vk_bar * self.dt
        J_eta_x     = np.block([Pose3D.J_1oplus(xk_1[0:3].reshape((3,1)), uk.reshape((3,1))), Pose3D.J_2oplus(xk_1[0:3].reshape((3,1)))*self.dt])
        J_v_x       = np.block([np.zeros((3,3)), np.diag(np.ones(3))])
        J           = np.block([[J_eta_x], [J_v_x]])
        return J

    def Jfw(self, xk_1, uk):
        # TODO: To be completed by the student
        J           = np.block([[Pose3D.J_2oplus(xk_1[0:3].reshape((3,1)))*self.dt*self.dt/2], 
                                [np.diag(np.ones(3))*self.dt]])
        return J
    
    def Jhx(self, xk):
        H= np.zeros((0,6))
        
        if self.headingData == True:
            H_yaw = np.array([[0,0,1,0,0,0]])
            H = np.block([[H], [H_yaw]])
        # if self.encoderData == True:
        #     H_n= np.array([[ 0,0,0,self.Kn_inv[0,0],0,self.Kn_inv[0,1]],
        #                     [ 0,0,0,self.Kn_inv[1,0],0,self.Kn_inv[1,1]]])
        #     H = np.block([[H], [H_n]])
        # return H
        return H
    
    def h(self, xk, received_encoder_readings, received_heading_data):  #:hm(self, xk):
        # TODO: To be completed by the student
        if received_heading_data == False and received_encoder_readings == True:
            h = xk[3:6].reshape((3,1))  # expected observation vector are the velocities [v_x, v_y, v_theta]^T
        elif received_heading_data == True and received_encoder_readings == True:
            h = xk[2:6].reshape((4,1))  # expected observation vector [z_compass, v_x, v_y, v_theta]^T
        elif received_heading_data == True and received_encoder_readings == False:
            h = xk[2,0].reshape((1,1))  # expected observation vector is the compass reading z_compass
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk:
        """
        # TODO: To be completed by the student
        uk = np.zeros((3,1))  # input vector
        # print("This is the input uk", uk)
        # Qk = self.Pk_1[3:6, 3:6]  # covariance matrix of the noise vector
        sigma_u_dot = 0.1
        sigma_v_dot = 0.01
        sigma_r_dot = np.deg2rad(1)
    
        Qk = np.diag([sigma_u_dot**2, sigma_v_dot**2, sigma_r_dot**2])
        # print("This is the noise covariance Qk", Qk)
        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student
        zk_compass, Rk_compass = self.robot.ReadCompass()
        uk, Qk = self.robot.ReadEncoders()  # get the wheel encoder readings
        nu_k = np.zeros((3,1)) # current velocity vector
        received_encoder_readings = False  # track if encoder readings were received
        
        # Store previous state and input for Logging purposes
        # self.etak_1 = xk_1  # store previous state
        if uk is not None:
            received_encoder_readings = True
            self.uk = uk  # store input
            # Calculate current velocities based on the wheel encoder readings
            n_L = uk[0, 0]  # left wheel encoder reading
            n_R = uk[1, 0]  # right wheel encoder reading
            v_L = ((n_L * 2 * np.pi / self.robot.pulse_x_wheelTurns) * self.robot.encoder_reading_frequency) * self.wheelRadius  # left wheel linear velocity
            v_R = ((n_R * 2 * np.pi / self.robot.pulse_x_wheelTurns) * self.robot.encoder_reading_frequency) * self.wheelRadius  # right wheel linear velocity
            
            v = (v_R + v_L) / 2  # linear velocity
            w = (v_R - v_L) / self.wheelBase  # angular velocity
            # print(f"This is the shape of nu_k {nu_k.shape} inside if")
            nu_k[0, 0] = v  # linear velocity
            nu_k[1, 0] = 0  # lateral velocity

            # Jay didn't wrtie this line of code, why?
            nu_k[2, 0] = w  # angular velocity
            self.nu_k_1 = nu_k  # store previous velocity
            received_encoder_readings = True

        else:
            # no new encoder readings using previous velocities
            # print(f"This is the shape of nu_k {nu_k.shape} inside else")
            received_encoder_readings = False
            nu_k[0, 0] = self.nu_k_1[0, 0]  # previous linear velocity
            nu_k[1, 0] = self.nu_k_1[1, 0]  # lateral velocity
            nu_k[2, 0] = self.nu_k_1[2, 0]  # previous angular velocity
        # Get encoder readings
        B_1 = ((2*np.pi/self.robot.pulse_x_wheelTurns)*self.robot.encoder_reading_frequency) * self.wheelRadius
        # The following is the mathematics involved in computing the covariances through the magic table:
        # [vr,vL]^T = [nr, nl]^T * [[B, 0],[B,0]]
        # Taking [[B,0],[B,0]] = C
        # assuming [vr, vl]^T = v
        # and [nr, nl]^T = n
        # therefore
        # v' = n * C
        # The covariance of the error in of n is Qk
        # therefore, using the magic table, the covariance of error of v' will be
        # C*Qk*C^T=Q_v'
        # where 
        C = np.array([[B_1, 0],
                        [0, B_1]])
        Q_v_rl = C@Qk@C.T
        # now, even though there is no velocity reading in y, to stay consistent
        # with matrix multiplication, we will have to add one column for velocity along y
        # and appropiately extend the Q_v_rl
        Q_vrl_modified = np.asarray([[Q_v_rl[0,0], 0, 0],
                            [0,0,0],
                            [0,0,Q_v_rl[1,1]]])
        # the velocity in x, y and angular velocity can be written as
        # matrix multiplication
        # [v_x, v_y, v_theta]^T = [[1/2, 0, 1/2],
        #                          [0, 0, 0],
        #                          [1/L, 0, 1/L]]*[v_R, 0, v_L]^
        # L is the self.wheelBase
        # the following equations implement the same, just in scalar form
        # Assuming [[1/2, 0, 1/2],
        #           [0, 0, 0],
#                       [1/L, 0, 1/L]]=D, we have
        # v = D*v'
        # now the covariance of error of v' is Q_vrl_modified
        # therefore, using the magic table, the covariance of error of v will be:
        # Q_v = D*Q_vrl_modified*D^T
        L = self.wheelBase
        D = np.asarray([[0.5, 0, 0.5],
                        [0,0,0],
                        [1/L, 0, -1/L]])
        Q_v = D @ Q_vrl_modified @ D.T
        # Here, Q_v is the covariance of error of the velocity vector [v_x, v_y, v_theta]^T
        
        # if zk is not None:
        #     if np.isscalar(zk):
        #         print("Inside if of get measurements")
        #         zk = np.array([[zk]])  # Convert scalar to (1,1) matrix
        #         self.zk_1 = zk
        # else:
        #     print("Inside else of get measurements")
        #     zk = self.zk_1
        #     zk = np.array([[zk]])
        if zk_compass is not None:
            zk_compass = np.array([[zk_compass]]) if np.isscalar(zk_compass) else np.atleast_2d(zk_compass).reshape(-1, 1)
            self.zk_1_compass = zk_compass
            self.headingData = True
        else:
            self.headingData = False
            if self.zk_1_compass is None:
                # First measurement, use initial state estimate
                self.zk_1_compass = np.array([[self.xk_1[2, 0]]])  # Initial yaw
            # zk = self.zk_1

        if np.isscalar(Rk_compass):
            Rk_compass = np.array([[Rk_compass]])  # Convert scalar to (1,1) matrix


        if zk_compass is not None and uk is not None:
            # Now, we need to combine this with the covariance of error of the compass reading to
            # get the overall covariance of error of the observation vector [z_compass, v_x, v_y, v_theta]^T
            # Assuming the covariance of error of the compass reading is Rk_compass, we can
            # combine it with Q_v to get the overall covariance of error of the observation vector
            Rk = scipy.linalg.block_diag(Rk_compass, Q_v)
            # Rk[2, 2] = 1e-9
            Rk[2, 2] = 0.01
            # Now we have the overall covariance of error of the observation vector, we can construct the
            # observation vector itself
            zk = np.block([[zk_compass], [nu_k]])  # observation vector [z_compass, v_x, v_y, v_theta]^T
            # Now we need to construct the Jacobian of the observation model with respect to the state
            Hk = np.array([[0, 0, 1, 0, 0, 0],  # Jacobian of the compass reading with respect to the state vector
                            [0, 0, 0, 1, 0, 0],  # Jacobian of the velocity reading in x with respect to the state vector
                            [0, 0, 0, 0, 1, 0],  # Jacobian of the velocity reading in y with respect to the state vector
                            [0, 0, 0, 0, 0, 1]]) # Jacobian of the angular velocity reading with respect to the state vector
            # Now we need to construct the Jacobian of the observation model with respect to the noise vector
            Vk = np.eye(4)  # Jacobian of the observation model with respect to the noise vector is an identity matrix since the noise directly affects the observations
            # not sure if the following jacobians are correct
            # Hk = np.array([[0, 0, 1, 0, 0, 0]])  # Jacobian of the observation model with respect to the state vector
            # Vk = np.array([[1]])
        elif zk_compass is not None and uk is None:
            Rk = Rk_compass
            zk = zk_compass
            Hk = np.array([[0, 0, 1, 0, 0, 0]])  # Jacobian of the compass reading with respect to the state vector
            Vk = np.eye(1)  # Jacobian of the observation model with respect to
        elif zk_compass is None and uk is not None:
            Rk = Q_v
            # Rk[1, 1] = 1e-9
            Rk[1, 1] = 0.01
            zk = nu_k
            Hk = np.array([[0, 0, 0, 1, 0, 0],  # Jacobian of the velocity reading in x with respect to the state vector
                            [0, 0, 0, 0, 1, 0],  # Jacobian of the velocity reading in y with respect to the state vector
                            [0, 0, 0, 0, 0, 1]]) # Jacobian of the angular velocity reading with respect to the state vector
            Vk = np.eye(3)  # Jacobian of the observation model with respect to
        else:
            # No measurements available (both compass and encoders are None)
            zk = None
            Rk = np.zeros((0, 0))
            Hk = np.zeros((0, 6))
            Vk = np.zeros((0, 0))
        return zk, Rk, Hk, Vk, received_encoder_readings, self.headingData


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2])) # velocities are co-related

    dd_robot = EKF_3DOFDifferentialDriveCtVelocity(kSteps, robot, x0, P0)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.0, 0.03]]).T)  # run localization loop

    exit(0)