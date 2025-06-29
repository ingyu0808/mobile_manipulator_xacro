#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr

# Number of joint in the panda which we are controlling
n = 7

# Make two obstacles with velocities
s0 = sg.Cuboid(scale=(0.3, 0.05, 0.4), pose=sm.SE3(0.32, 0.1, 0.2))
s0.v = [0, 0, 0, 0, 0, 0]

# s1 = sg.Cuboid(scale=(0.1, 0.1, 0.1), pose=sm.SE3(1.1, 1.35, 1.65))
# s1.v = [0, 0, 0, 0, 0, 0]

collisions = [s0]

# Make a target
target = sg.Sphere(radius=0.02, pose=sm.SE3(0.4, 0.5, 0.3))

# Add the Panda and shapes to the simulator
env.add(panda)
env.add(s0)
# env.add(s1)
env.add(target)

# Set the desired end-effector pose to the location of target
Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.T[:3, -1]
# Tep.A[2, 3] += 0.1


def step():
    # The pose of the Panda's end-effector
    Te = panda.fkine(panda.q)

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    # print("Ain  , bin") #링크의 갯수 (확인된 링크 + 2)
    # print(Ain.shape, bin.shape)

    # For each collision in the scene
    for collision in collisions:

        # Form the velocity damper inequality contraint for each collision
        # object on the robot to the collision in the scene
        c_Ain, c_bin = panda.link_collision_damper(
            collision,
            panda.q[:n],
            0.3,
            0.01,
            1.0,
            start=panda.link_dict["panda_link1"],
            end=panda.link_dict["panda_hand"],
        )
         


        # If there are any parts of the robot within the influence distance
        # to the collision in the scene
        if c_Ain is not None and c_bin is not None:

            print("c_Ain  , c_bin")
            print(c_Ain.shape, c_bin.shape)

            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 4))]     
            
            print("cal_c_Ain  , cal_c_bin")
            print(c_Ain.shape, c_bin.shape)  

            print("Ain  , bin")
            print(Ain.shape, bin.shape)


            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")

    # Apply the joint velocities to the Panda
    panda.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(0.01)

    return arrived


def run():
    arrived = False
    while not arrived:
        arrived = step()


step()
run()