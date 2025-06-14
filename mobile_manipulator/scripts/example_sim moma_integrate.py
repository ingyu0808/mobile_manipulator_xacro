import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math


def step_robot(r: rtb.ERobot, Tep):

    wTe = r.fkine(r.q)

    eTep = np.linalg.inv(wTe) @ Tep

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 6)

    # Joint velocity component of Q
    Q[: r.n, : r.n] *= Y
    Q[:3, :3] *= 1.0 / et

    # Slack component of Q
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)

    v[3:] *= 1.3

    # The equality contraints
    Aeq = np.c_[r.jacobe(r.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)


    # print("Ain  , bin") #링크의 갯수 (확인된 링크 + 2)
    # print(Ain.shape, bin.shape)

    for collision in collisions:

        # Form the velocity damper inequality contraint for each collision
        # object on the robot to the collision in the scene
        c_Ain, c_bin = moma.link_collision_damper(
            collision,
            moma.q[:r.n],
            0.1,
            0.03,
            1.0,
            start=moma.link_dict["base_link"],
            end=moma.link_dict["panda_hand"],
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
    c = np.concatenate(
        (np.zeros(3), -r.jacobm(start=r.links[4]).reshape((r.n - 3,)), np.zeros(6))
    )

    # # Get base to face end-effector
    # kε = 0.5
    # bTe = r.fkine(r.q, include_base=False).A
    # θε = math.atan2(bTe[1, -1], bTe[0, -1])
    # ε = kε * θε
    # c[0] = -ε

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[: r.n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
    qd = qd[: r.n]

    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4

    if et < 0.02:
        return True, qd
    else:
        return False, qd


env = swift.Swift()
env.launch(realtime=True)

ax_goal = sg.Axes(0.1)
env.add(ax_goal)


moma = rtb.models.Momaintegrate()
moma.q = moma.qr
env.add(moma)

s0 = sg.Cuboid(scale=(0.1, 1.0, 0.4), pose=sm.SE3(3.0, 0.0, 0.5))
s0.v = [0, 0, 0, 0, 0, 0]

collisions = [s0]

env.add(s0)

arrived = False
dt = 0.025


# Behind
env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
wTep = moma.fkine(moma.q) * sm.SE3.Rz(np.pi)
wTep.A[:3, :3] = np.diag([-1, 1, -1])
wTep.A[0, -1] += 4.0
wTep.A[2, -1] += 0.25
ax_goal.T = wTep
env.step()


while not arrived:

    arrived, moma.qd = step_robot(moma, wTep.A)
    env.step(dt)

    # # Reset bases
    # base_new = moma.fkine(moma._q, end=moma.links[2]).A
    # moma._T = base_new
    # moma.q[:2] = 0

env.hold()