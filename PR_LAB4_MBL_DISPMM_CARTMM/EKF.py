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
        

        self.uk = uk
        self.Qk = Qk  # store the input and noise covariance for logging

        self.xk_bar = self.f(self.xk_1,self.uk)
        
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
        self.zk = zk
        # in the case when the observation is just one number
        if type(zk) is int:
            self.nz = 0
        else:
            self.nz = zk.shape[0];  # store dimensionality of the observation
        self.Rk = Rk  # store the observation and noise covariance for logging
        # print("This is shape of Hk:",Hk.shape)
        # print("This is shape of Vk:",Vk.shape)
        # print("This is shape of Rk:",Rk.shape)
        # print("This is shape of Pk_bar:",Pk_bar.shape)
        # Calculate Kalman gain
        K = self.Pk_bar @ Hk.T @ np.linalg.inv(Hk @ self.Pk_bar @ Hk.T + Vk @ self.Rk @ Vk.T)  # Kalman gain
        # print("This is shape of K:",K.shape)
        # Update state and covariance matrix
        # print("This is shape of zk inside EKF update:",self.zk.shape)
        # Update state and covariance matrix
        # FIX: Calculate innovation and selectively wrap the angular component.
        # print("This is zk inside EKF update:",self.zk)
        # print("This is h(xk_bar) inside EKF update:",self.h(self.xk_bar))
        innovation = self.zk - self.h(self.xk_bar)
        # print(f"This is she shape of innovation {innovation.shape}")
        # Assuming the compass (yaw) measurement is the FIRST element in the stacked vector (index 0)
        # The following attribute is defined in the class FEKFMBL
        if getattr(self, 'zm_observed', True): 
            print("Wrapping angle inside Update in EKF")
            innovation[0, 0] = wrap_angle(innovation[0, 0])
        self.xk = self.xk_bar + K@innovation

        #TODO : The term multiplied at the end - should it be multiplied or not?
        I = np.eye(len(self.xk))
        self.Pk = (I - K @ Hk) @ self.Pk_bar @ (I - K @ Hk).T + K @ self.Rk @ K.T
        
        return self.xk, self.Pk
