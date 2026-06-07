from conversions import *
from Pose import *
import numpy as np
from conversions import *
def wrap_angle(angle):
    """
    Wraps an angle (in radians) to the range [-pi, pi].
    Works for both single values and NumPy arrays.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
class Feature:
    """
    This class implements the **interface of the pose-feature compounding operation**. This class provides the interface
    to implement the compounding operation between the robot pose (represented in the N-Frame) and the feature pose (represented in
    the B-Frame) obtaining the feature representation in the N-Frame.
    The class also provides the interface to implement the Jacobians of the pose-feature compounding operation.
    """

    def __init__(BxF, feature):
        BxF.feature = feature

    def boxplus(BxF, NxB):
        """
        Pose-Feature compounding operation:

        .. math::
            ^Nx_F=^Nx_B \\boxplus ^Bx_F
            :label: eq-boxplus

        which computes the pose of a feature in the N-Frame given the pose of the robot in the N-Frame and the pose of
        the feature in the B-Frame.
        **This is a pure virtual method that must be overwritten by the child class**.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :param BxF: Feature pose in the B-Frame (:math:`^Bx_F`)
        :return: Feature pose in the N-Frame (:math:`^Nx_F`)
        """
        pass

    def J_1boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Feature compounding operation (eq. :eq:`eq-boxplus`) with respect to the first argument :math:`^Nx_B`.

        .. math::
            J_{1\\boxplus}=\\frac{\\partial ^Nx_B \\boxplus ^Bx_F}{\\partial ^Nx_B}.
            :label: eq-J_1boxplus

        **To be overriden by the child class**.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :param BxF: Feature pose in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}`
        """
        pass

    def J_2boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Feature compounding operation (eq. :eq:`eq-boxplus`) with respect to the second argument :math:`^Bx_F`.

        .. math::
            J_{2\\boxplus}=\\frac{\\partial ^Nx_B \\boxplus ^Bx_F}{\\partial ^Bx_F}.
            :label: eq-J_2boxplus

        **To be overriden by the child class**.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :return: Jacobian matrix :math:`J_{2\\boxplus}`
        """
        pass

    def ToCartesian(self):
        """
        Translates from its internal representation to the representation used for plotting.
        **To be overriden by the child class**.

        :return: Feature in Cartesian Coordinates
        """
        pass

    def J_2c(selfself):
        """
        Jacobian of the ToCartesian method. Required for plotting non Cartesian features.
        **To be overriden by the child class**.

        :return: Jacobian of the transformation
        """
        pass

class CartesianFeature(Feature,np.ndarray):
    """
    Cartesian feature class. The class inherits from the :class:`Feature` class providing an implementation of its
    interface for a Cartesian Feature, by implementing the :math:`\\boxplus` operator as well as its Jacobians. The
    class also inherits from the ndarray numpy class allowing to be operated as a numpy ndarray.
    """

    def __new__(BxF, input_array):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a :class:`CartesianFeature class object
        """
        assert input_array.shape == (3,1) or input_array.shape == (2,1), "CartesianFeature must be of 2 or 3 DOF"

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(BxF)

        # The F matrix is used to convert from a pose to a feature in order to take profit of the oplus operator already implemented in the Pose class
        # The F matrix is (nf x np) where np is de dimension of the pose and nf the dimension of the feature
        # F is build as a list of F matrices, where the index of the list matches the dimension of the feature
        BxF.feature = obj

        super().__init__(BxF,obj)

        # Finally, we must return the newly created object:
        return obj

    def boxplus(BxF, NxB):
        """
        Pose-Cartesian Feature compounding operation:

        .. math::
            F&=\\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\end{bmatrix}\\\\
            ^Nx_F&=^Nx_B \\boxplus ^Bx_F = F ( ^Nx_B \\oplus ^Bx_F )
            :label: eq-boxplus2DCartesian

        which computes the Cartesian position of a feature in the N-Frame given the pose of the robot in the N-Frame and
        the Cartesian position of the feature in the B-Frame.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :param BxF: Cartesian feature pose in the B-Frame (:math:`^Bx_F`)
        :return: Feature pose in the N-Frame (:math:`^Nx_F`)
        """

        # TODO: To be completed by the student
        F = np.array([[1, 0, 0],
                      [0, 1, 0]])
        # print("BxF.feature=", BxF.feature)
        # print("BxF", BxF)
        # print("Type of BxF.feature=", type(BxF.feature))
        # print("Type of Bxf", type(BxF))
        # print("NxB=", NxB)
        # print("F.T @ (BxF.feature)=", F.T @ (BxF))
        NxF = F @ ((NxB).oplus(Pose3D(F.T @ (BxF))))
        return CartesianFeature(NxF)

    def J_1boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Cartesian Feature compounding operation with respect to the robot pose:

        .. math::
            J_{1\\boxplus} = F J_{1\\oplus}
            :label: eq-J1boxplus2DCartesian

        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
        :param BxF: Cartesian feature pose represented in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J1boxplus2DCartesian`) (eq. :eq:`eq-J1boxplus2DCartesian`)
        """

        # TODO: To be completed by the student

        F = np.array([[1, 0, 0],
                      [0, 1, 0]])
        J = F @ NxB.J_1oplus(Pose3D(F.T @ (BxF)))
        return J

    def J_2boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Cartesian Feature compounding operation with respect to the feature position:

        .. math::
            J_{2\\boxplus} = F J_{2oplus}
            :label: eq-J2boxplus2DCartesian

        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
        :param BxF: Cartesian feature pose represented in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J2boxplus2DCartesian`)
        """

        # TODO: To be completed by the student
        F = np.array([[1, 0, 0],
                      [0, 1, 0]])
        J = F @ NxB.J_2oplus()@F.T
        return J

    def ToCartesian(self):
        """
        Translates from its internal representation to the representation used for plotting.

        :return: Feature in Cartesian Coordinates
        """
        return self

    def J_2c(self):
        """
        Jacobian of the ToCartesian method. Required for plotting non Cartesian features.
        **To be overriden by the child class**.

        :return: Jacobian of the transformation
        """
        return np.eye(self.shape[0])
    
class PolarFeature(Feature,np.ndarray):
    """
    Polar feature class. The class inherits from the :class:`Feature` class providing an implementation of its
    interface for a Polar Feature, by implementing the :math:`\\boxplus` operator as well as its Jacobians. The
    class also inherits from the ndarray numpy class allowing to be operated as a numpy ndarray.
    """

    def __new__(BxF, input_array):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a :class:`PolarFeature class object
        """
        assert input_array.shape == (2,1), "PolarFeature must be of 2 DOF"

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(BxF)

        # The F matrix is used to convert from a pose to a feature in order to take profit of the oplus operator already implemented in the Pose class
        # The F matrix is (nf x np) where np is de dimension of the pose and nf the dimension of the feature
        # F is build as a list of F matrices, where the index of the list matches the dimension of the feature
        BxF.feature = obj

        super().__init__(BxF,obj)

        # Finally, we must return the newly created object:
        return obj

    # def boxplus(BxF, NxB):
    #     """
    #     Pose-Polar Feature compounding operation:

    #     .. math::
    #         F&=\\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\end{bmatrix}\\\\
    #         ^Nx_F&=^Nx_B \\boxplus ^Bx_F = F ( ^Nx_B \\oplus ^Bx_F )
    #         :label: eq-boxplus2DPolar

    #     which computes the Polar position of a feature in the N-Frame given the pose of the robot in the N-Frame and
    #     the Polar position of the feature in the B-Frame.

    #     :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
    #     :param BxF: Polar feature pose in the B-Frame (:math:`^Bx_F`)
    #     :return: Feature pose in the N-Frame (:math:`^Nx_F`)
    #     """

    #     psi_global = NxB[2,0] + BxF[1,0]
    #     xf = NxB[0,0] + BxF[0,0] * np.cos(psi_global)
    #     yf = NxB[1,0] + BxF[0,0] * np.sin(psi_global)
        
    #     # 2. World Cartesian -> Global Polar
    #     rho_n = np.sqrt(xf**2 + yf**2)
    #     theta_n = np.arctan2(yf, xf) # atan2 is critical for signs
    #     return PolarFeature(np.array([[rho_n], [theta_n]]))
    

    # def J_1boxplus(BxF, NxB):
    #     """
    #     Jacobian of the Pose-Polar Feature compounding operation with respect to the robot pose:

    #     .. math::
    #         J_{1\\boxplus} = F J_{1\\oplus}
    #         :label: eq-J1boxplus2DPolar

    #     :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
    #     :param BxF: Polar feature pose represented in the B-Frame (:math:`^Bx_F`)
    #     :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J1boxplus2DPolar`) (eq. :eq:`eq-J1boxplus2DPolar`)
    #     """
        
    #     # Evaluated at the result of boxplus
    #     psi_sum = NxB[2,0] + BxF[1,0]
    #     xf = NxB[0,0] + BxF[0,0] * np.cos(psi_sum)
    #     yf = NxB[1,0] + BxF[0,0] * np.sin(psi_sum)
    #     rho_n2 = xf**2 + yf**2
    #     rho_n = np.sqrt(rho_n2)

    #     # Partial derivatives w.r.t [x_B, y_B, psi_B]
    #     j11, j12 = xf/rho_n, yf/rho_n
    #     j13 = (xf * (-BxF[0,0]*np.sin(psi_sum)) + yf * (BxF[0,0]*np.cos(psi_sum))) / rho_n
        
    #     j21, j22 = -yf/rho_n2, xf/rho_n2
    #     j23 = (-yf * (-BxF[0,0]*np.sin(psi_sum)) + xf * (BxF[0,0]*np.cos(psi_sum))) / rho_n2
        
    #     return np.array([[j11, j12, j13], [j21, j22, j23]])

    # def J_2boxplus(BxF, NxB):
    #     """
    #     Jacobian of the Pose-Polar Feature compounding operation with respect to the feature position:

    #     .. math::
    #         J_{2\\boxplus} = F J_{2oplus}
    #         :label: eq-J2boxplus2DPolar

    #     :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
    #     :param BxF: Polar feature pose represented in the B-Frame (:math:`^Bx_F`)
    #     :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J2boxplus2DPolar`)
    #     """
    #     psi_sum = NxB[2,0] + BxF[1,0]
    #     xf = NxB[0,0] + BxF[0,0] * np.cos(psi_sum)
    #     yf = NxB[1,0] + BxF[0,0] * np.sin(psi_sum)
    #     rho_n_sq = xf**2 + yf**2
    #     rho_n = np.sqrt(rho_n_sq)
        
    #     # wrt [rho_b, theta_b]
    #     j11 = (xf * np.cos(psi_sum) + yf * np.sin(psi_sum)) / rho_n
    #     j12 = (xf * (-BxF[0,0] * np.sin(psi_sum)) + yf * (BxF[0,0] * np.cos(psi_sum))) / rho_n
        
    #     j21 = (-yf * np.cos(psi_sum) + xf * np.sin(psi_sum)) / rho_n_sq
    #     j22 = (-yf * (-BxF[0,0] * np.sin(psi_sum)) + xf * (BxF[0,0] * np.cos(psi_sum))) / rho_n_sq
        
    #     return np.array([[j11, j12], [j21, j22]])

    def boxplus(BxF, NxB):

        """

        Pose-Polar Feature compounding operation:



        .. math::

        F&=\\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\end{bmatrix}\\\\

        ^Nx_F&=^Nx_B \\boxplus ^Bx_F = F ( ^Nx_B \\oplus ^Bx_F )

        :label: eq-boxplus2DPolar



        which computes the Polar position of a feature in the N-Frame given the pose of the robot in the N-Frame and

        the Polar position of the feature in the B-Frame.



        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)

        :param BxF: Polar feature pose in the B-Frame (:math:`^Bx_F`)

        :return: Feature pose in the N-Frame (:math:`^Nx_F`)

        """



        BxF_cartesian = p2c(BxF)

        NxF_cartesian = BxF_cartesian.boxplus(NxB)

        return c2p(NxF_cartesian)




    def J_1boxplus(BxF, NxB):

        """

        Jacobian of the Pose-Polar Feature compounding operation with respect to the robot pose:



        .. math::

        J_{1\\boxplus} = F J_{1\\oplus}

        :label: eq-J1boxplus2DPolar



        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)

        :param BxF: Polar feature pose represented in the B-Frame (:math:`^Bx_F`)

        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J1boxplus2DPolar`) (eq. :eq:`eq-J1boxplus2DPolar`)

        """


        BxF_cartesian = p2c(BxF)

        NxF_cartesian = BxF_cartesian.boxplus(NxB)

        J_c2p_value = J_c2p(NxF_cartesian)

        J1_cart = BxF_cartesian.J_1boxplus(NxB)

        return J_c2p_value @ J1_cart



    def J_2boxplus(BxF, NxB):

        """

        Jacobian of the Pose-Polar Feature compounding operation with respect to the feature position:



        .. math::

        J_{2\\boxplus} = F J_{2oplus}

        :label: eq-J2boxplus2DPolar



        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)

        :param BxF: Polar feature pose represented in the B-Frame (:math:`^Bx_F`)

        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J2boxplus2DPolar`)

        """

        BxF_cartesian = p2c(BxF)

        NxF_cartesian = BxF_cartesian.boxplus(NxB)

        J_c2p_value = J_c2p(NxF_cartesian)

        J2_cart = BxF_cartesian.J_2boxplus(NxB)

        J_p2c_value = J_p2c(BxF)

        return J_c2p_value @ J2_cart @ J_p2c_value

    def ToCartesian(self):
        """
        Translates from its internal representation to the representation used for plotting.

        :return: Feature in Cartesian Coordinates
        """
        return p2c(self)

    def J_2c(self):
        """
        Jacobian of the ToCartesian method. Required for plotting non Cartesian features.
        **To be overriden by the child class**.

        :return: Jacobian of the transformation
        """
        return J_p2c(self)

class SphericalFeature(Feature,np.ndarray):
    """
    Spherical feature class. The class inherits from the :class:`Feature` class providing an implementation of its
    interface for a Spherical Feature, by implementing the :math:`\\boxplus` operator as well as its Jacobians. The
    class also inherits from the ndarray numpy class allowing to be operated as a numpy ndarray.
    """

    def __new__(BxF, input_array):
        """
        Constructor of the class. It is called when the class is instantiated. It is required to extend the ndarry numpy class.

        :param input_array: array used to initialize the class
        :returns: the instance of a :class:`SphericalFeature` class object
        """
        assert input_array.shape == (3,1), "SphericalFeature must be of 3 DOF"

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(BxF)

        # The F matrix is used to convert from a pose to a feature in order to take profit of the oplus operator already implemented in the Pose class
        # The F matrix is (nf x np) where np is de dimension of the pose and nf the dimension of the feature
        # F is build as a list of F matrices, where the index of the list matches the dimension of the feature
        BxF.feature = obj

        super().__init__(BxF,obj)

        # Finally, we must return the newly created object:
        return obj

    def boxplus(BxF, NxB):
        """
        Pose-Spherical Feature compounding operation:

        .. math::
            F&=\\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\end{bmatrix}\\\\
            ^Nx_F&=^Nx_B \\boxplus ^Bx_F = F ( ^Nx_B \\oplus ^Bx_F )
            :label: eq-boxplus2DSpherical

        which computes the Spherical position of a feature in the N-Frame given the pose of the robot in the N-Frame and
        the Spherical position of the feature in the B-Frame.

        :param NxB: Robot pose in the N-Frame (:math:`^Nx_B`)
        :param BxF: Spherical feature pose in the B-Frame (:math:`^Bx_F`)
        :return: Feature pose in the N-Frame (:math:`^Nx_F`)
        """

        # TODO: To be completed by the student
        F = np.array([[1, 0, 0],
                      [0, 1, 0]])
        # print("BxF.feature=", BxF.feature)
        # print("BxF", BxF)
        # print("Type of BxF.feature=", type(BxF.feature))
        # print("Type of Bxf", type(BxF))
        # print("NxB=", NxB)
        # print("F.T @ (BxF.feature)=", F.T @ (BxF))
        NxF = F @ ((NxB).oplus(Pose3D(F.T @ (BxF))))
        return SphericalFeature(NxF)

    def J_1boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Spherical Feature compounding operation with respect to the robot pose:

        .. math::
            J_{1\\boxplus} = F J_{1\\oplus}
            :label: eq-J1boxplus2DSpherical

        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
        :param BxF: Spherical feature pose represented in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J1boxplus2DSpherical`) (eq. :eq:`eq-J1boxplus2DSpherical`)
        """

        # TODO: To be completed by the student

        F = np.array([[1, 0, 0],
                      [0, 1, 0]])
        J = F @ NxB.J_1oplus(Pose3D(F.T @ (BxF)))
        return J

    def J_2boxplus(BxF, NxB):
        """
        Jacobian of the Pose-Spherical Feature compounding operation with respect to the feature position:

        .. math::
            J_{2\\boxplus} = F J_{2oplus}
            :label: eq-J2boxplus2DSpherical

        :param NxB: robot pose represented in the N-Frame (:math:`^Nx_B`)
        :param BxF: Spherical feature pose represented in the B-Frame (:math:`^Bx_F`)
        :return: Jacobian matrix :math:`J_{1\\boxplus}` (eq. :eq:`eq-J2boxplus2DSpherical`)
        """

        # TODO: To be completed by the student
        F = np.array([[1, 0, 0],
                      [0, 1, 0]])
        J = F @ NxB.J_2oplus()@F.T
        return J

    def ToCartesian(self):
        """
        Translates from its internal representation to the representation used for plotting.

        :return: Feature in Cartesian Coordinates
        """
        return s2c(self)

    def J_2c(self):
        """
        Jacobian of the ToCartesian method. Required for plotting non Cartesian features.
        **To be overriden by the child class**.

        :return: Jacobian of the transformation
        """
        return J_s2c(self)

if __name__ == '__main__':

    NxB3dof = Pose3D(np.array([[5,5,np.pi/2]]).T)
    BxF = CartesianFeature(np.array([[3,3]]).T)

    NxF = BxF.boxplus(NxB3dof)
    print("This is type of NxF:",type(NxF))
    # print("NxF=", NxF)
    # print("J_1boxplus=", BxF.J_1boxplus(NxB3dof))
    # print("J_2boxplus=", BxF.J_2boxplus(NxB3dof))

    # I believe the 4Dof case is not necessary for this lab
    
    # NxB4dof=Pose4D(np.array([[5,5,5,np.pi/2]]).T)

    # NxF = BxF.boxplus(NxB4dof)

    # print("NxF=", NxF.T)
    # print("J_1boxplus=", BxF.J_1boxplus(NxB4dof))
    # print("J_2boxplus=", BxF.J_2boxplus(NxB4dof))

    exit(0)

