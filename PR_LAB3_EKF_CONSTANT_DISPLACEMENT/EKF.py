from GaussianFilter import *
import numpy as np
def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
class EKF(GaussianFilter):
    """
    Extended Kalman Filter class. Implements the :class:`GaussianFilter` interface for the particular case of the Extended Kalman Filter.
    """
    def __init__(self, x0, P0, *args):
        """
        Constructor of the EKF class.

        :param x0: initial mean state vector
        :param P0: initial covariance matrix
        :param args: arguments to be passed to the parent class
        """
        
        super().__init__(x0, P0, *args)  # call parent constructor

    def f(self, xk_1, uk): # motion model
        """
        Motion model of the EKF **to be overwritten by the child class**.

        :param xk_1: previous mean state vector
        :param uk: input vector
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix
        """
        pass

    def Jfx(self, xk_1):
        """
        Jacobian of the motion model with respect to the state vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def Jfw(self, xk_1):
        """
        Jacobian of the motion model with respect to the noise vector. **Method to be overwritten by the child class**.

        :param xk_1: Linearization point. By default the linearization point is the previous state vector taken from a class attribute.
        :return: Jacobian matrix
        """
        pass

    def h(self, xk):  # observation model
        """
        The observation model of the EKF is given by:

        .. math::
            z_k=h(x_k,v_k)
            :label: eq-EKF-observation-model

        This method computes the mean of this direct observation model. Therefore it does not depend on v_k since it is
        a zero mean Gaussian noise.

        :param xk: mean of the predicted state vector. By default it is taken from the class attribute.
        :return: expected observation vector
        """
        pass

    def Prediction(self, uk, Qk, xk_1=None, Pk_1=None):
        """
        Prediction step of the EKF. It calls the motion model and its Jacobians to predict the state vector and its covariance matrix.

        :param uk: input vector
        :param Qk: covariance matrix of the noise vector
        :param xk_1: previous mean state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :param Pk_1: covariance matrix of the previous state vector. By default it is taken from the class attribute. Otherwise it updates the class attribute.
        :return xk_bar, Pk_bar: predicted mean state vector and its covariance matrix. Also updated in the class attributes.
        """
        # logging for plotting
        self.Pk_1 = Pk_1 if Pk_1 is not None else self.Pk_1
        self.xk_1 = xk_1 if xk_1 is not None else self.xk_1
        

        self.uk = uk;
        self.Qk = Qk  # store the input and noise covariance for logging
        # xk_1=np.normal(self.xk_1)
        # KF equations begin here
        # TODO: To be implemented by the student
        self.xk_bar = self.f(self.xk_1,self.uk)
        # Jfx_val = self.Jfx(self.xk_1)
        # Pk_1_val = self.Pk_1
        # Jfw_val = self.Jfw(self.xk_1)
        # Qk_val = self.Qk

        # print("Shape of Jfx:", Jfx_val.shape)
        # print("Shape of Pk_1:", Pk_1_val.shape)
        # print("Shape of Jfw:", Jfw_val.shape)
        # print("Shape of Qk:", Qk_val.shape)

        # temp1 = Jfx_val @ Pk_1_val
        # print("Shape of Jfx @ Pk_1:", temp1.shape)

        # temp2 = temp1 @ Jfx_val.T
        # print("Shape of Jfx @ Pk_1 @ Jfx.T:", temp2.shape)
        # print("This is shape of Jfw_val",Jfw_val.shape)
        # print("This is shape of Qk_val",Qk_val.shape)
        # temp3 = Jfw_val @ Qk_val
        # print("Shape of Jfw @ Qk:", temp3.shape)

        # temp4 = temp3 @ Jfw_val.T
        # print("Shape of Jfw @ Qk @ Jfw.T:", temp4.shape)

        # self.Pk_bar = temp2 + temp4
        # print("Shape of Pk_bar:", self.Pk_bar.shape)
        # print(f"This is the shape of self.xk_1 at the end of Prediction step {self.xk_1.shape}")
        # self.Pk_bar = self.Jfx(self.xk_1,)@self.Pk_1@self.Jfx(self.xk_1).T + self.Jfw(self.xk_1)@self.Qk@self.Jfw(self.xk_1).T
        
        self.Pk_bar = (self.Jfx(self.xk_1, self.uk) @ self.Pk_1 @ 
                    self.Jfx(self.xk_1, self.uk).T + 
                    self.Jfw(self.xk_1, self.uk) @ self.Qk @ 
                    self.Jfw(self.xk_1, self.uk).T)
        return self.xk_bar, self.Pk_bar

    def Update(self, zk, Rk, xk_bar, Pk_bar, Hk, Vk):
        """
        Update step of the EKF. It calls the observation model and its Jacobians to update the state vector and its covariance matrix.

        :param zk: observation vector
        :param Rk: covariance matrix of the noise vector
        :param xk_bar: predicted mean state vector.
        :param Pk_bar: covariance matrix of the predicted state vector.
        :param Hk: Jacobian of the observation model with respect to the state vector.
        :param Vk: Jacobian of the observation model with respect to the noise vector.
        :return xk,Pk: updated mean state vector and its covariance matrix. Also updated in the class attributes.
        """
        # logging for plotting
        # print("This is Pk_bar inside update call in EKF.py",Pk_bar)
        self.xk_bar = xk_bar
        self.Pk_bar = Pk_bar
        self.zk = zk;
        # in the case when the observation is just one number
        if type(zk) is int:
            self.nz = 0
        else:
            self.nz = zk.shape[0];  # store dimensionality of the observation
        self.Rk = Rk  # store the observation and noise covariance for logging

        # # KF equations begin here
        # print(f"This is Pk_bar {Pk_bar}")
        # print("Shape of Pk_bar:", self.Pk_bar.shape)
        # print("Shape of Hk:", Hk.shape)
        # print("Shape of Hk.T:", Hk.T.shape)

        # temp1 = self.Pk_bar @ Hk.T
        # print("Shape of Pk_bar @ Hk.T:", temp1.shape)

        # temp2 = Hk @ self.Pk_bar
        # print("Shape of Hk @ Pk_bar:", temp2.shape)

        # temp3 = temp2 @ Hk.T
        # print("Shape of Hk @ Pk_bar @ Hk.T:", temp3.shape)
        
        # print("Shape of Vk:", Vk.shape)
        # print("Shape of Rk:", self.Rk.shape)
        # print(f"This is type of Vk {type(Vk)} and this is type of self.Rk {type(self.Rk)}")

        # temp4 = Vk @ self.Rk
        # print("Shape of Vk @ Rk:", temp4.shape)

        # temp5 = temp4 @ Vk.T
        # print("Shape of Vk @ Rk @ Vk.T:", temp5.shape)

        # temp6 = temp3 + temp5
        # print("Shape of (Hk @ Pk_bar @ Hk.T + Vk @ Rk @ Vk.T):", temp6.shape)

        # temp7 = np.linalg.inv(temp6)
        # print("Shape of inverse:", temp7.shape)

        # K = temp1 @ temp7
        # print("Shape of Kalman Gain K:", K.shape)
        K = self.Pk_bar @ Hk.T @ np.linalg.inv(Hk @ self.Pk_bar @ Hk.T + Vk @ self.Rk @ Vk.T)  # Kalman gain
        # TODO: To be implemented by the student
        # print(f"The shape of self.xk_1, K, self.zk, self.h(self.xk_bar) is {self.xk_1.shape}, {K.shape}, {self.zk.shape}, {self.h(xk_bar).shape} respectively")
        self.xk = self.xk_bar + K@(wrap_angle(self.zk - self.h(self.xk_bar)))
        self.Pk = (np.eye(len(self.xk)) - K@Hk)@self.Pk_bar
        return self.xk, self.Pk
