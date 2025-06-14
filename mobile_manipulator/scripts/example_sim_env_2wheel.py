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
    Q[:2, :2] *= 1.0 / et

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
            0.5,             # di: 속도 감쇠기가 작동하기 시작하는 영향 거리 (influence distance)
            0.05,            # ds: 충돌 형상에 접근 가능한 최소 거리 (safety margin)
            1.5,             # xi: 감쇠 이득 (velocity damper gain)
            start=moma.link_dict["chassis_link"], 
            end=moma.link_dict["panda_hand"],      
        )


        # If there are any parts of the robot within the influence distance
        # to the collision in the scene
        if c_Ain is not None and c_bin is not None:

            # print("c_Ain  , c_bin")
            # print(c_Ain.shape, c_bin.shape)
            

            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 2))]   

            # print("cal_c_Ain  , cal_c_bin")
            # print(c_Ain.shape, c_bin.shape)  

            # print("Ain  , bin")
            # print(Ain.shape, bin.shape)

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (np.zeros(2), -r.jacobm(start=r.links[4]).reshape((r.n - 2,)), np.zeros(6))
    )

    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False).A
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

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


moma = rtb.models.Moma2()
initial_position = sm.SE3(0, 0, 0.324)  # Z축으로 0.324만큼 이동
moma.q = moma.qr
moma.base = initial_position  # 초기 위치를 설정
env.add(moma)

s0 = sg.Cuboid(scale=(20.0, 0.1, 20.0), pose=sm.SE3(0.0, 2.0, 0.0))
s0.v = [0, 0, 0, 0, 0, 0]

s1 = sg.Cuboid(scale=(20.0, 0.1, 20.0), pose=sm.SE3(0.0, -2.0, 0.0))
s1.v = [0, 0, 0, 0, 0, 0]

collisions = [s0,s1]

env.add(s0)
env.add(s1)

arrived = False
dt = 0.05


env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])

# ✅ 절대 좌표와 방향 설정
goal_position = [3, 0.5, 1.0]
goal_orientation_rpy = [0, (np.pi), -(np.pi/2)]  # world 기준 RPY

wTep = sm.SE3(goal_position) * sm.SE3.RPY(goal_orientation_rpy, order='xyz')

# 목표축 표시
ax_goal.T = wTep
env.step()


while not arrived:

    arrived, moma.qd = step_robot(moma, wTep.A)
    env.step(dt)

    # Reset bases
    base_new = moma.fkine(moma._q, end=moma.links[2]).A
    # print(moma.links[2])
    moma._T = base_new
    moma.q[:2] = 0

env.hold()