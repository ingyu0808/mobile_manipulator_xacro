import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import random
import threading
import time
import math

# QP 제어 함수
def step_robot(r: rtb.ERobot, Tep):
    wTe = r.fkine(r.q)
    eTep = np.linalg.inv(wTe) @ Tep
    et = np.sum(np.abs(eTep[:3, -1]))
    Y = 0.01
    Q = np.eye(r.n + 6)
    Q[:r.n, :r.n] *= Y
    Q[:2, :2] *= 1.0 / et
    Q[r.n:, r.n:] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)
    v[3:] *= 1.3
    Aeq = np.c_[r.jacobe(r.q), np.eye(6)]
    beq = v.reshape((6,))

    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)
    Ain[:r.n, :r.n], bin[:r.n] = r.joint_velocity_damper(0.1, 0.9, r.n)

    for collision in collisions:
        c_Ain, c_bin = r.link_collision_damper(
            collision, r.q, 0.2, 0.12, 1.0,
            start=moma.link_dict['base_link'],
            end=moma.link_dict['panda_hand']
        )
        if c_Ain is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 3))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    c = np.concatenate((np.zeros(2), -r.jacobm(start=r.links[3]).reshape((r.n - 2,)), np.zeros(6)))

    kε = 0.1
    bTe = r.fkine(r.q, include_base=False).A
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

    lb = -np.r_[r.qdlim[:r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[:r.n], 10 * np.ones(6)]
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
    qd = qd[:r.n]

    qd *= 0.7 / et if et > 0.5 else 1.4
    return (et < 0.02), qd


# 전역 변수
arrived = False
second_phase_started = False

# 시뮬레이션 스레드
def simulation_thread():
    global arrived, second_phase_started, wTep, goal_box, collisions

    start_time = time.time()
    added_obstacle = False

    while True:
        arrived_flag, qd = step_robot(moma, wTep.A)
        moma.qd = qd
        env.step(dt)

        base_new = moma.fkine(moma.q, end=moma.links[2])
        moma._T = base_new.A
        moma.q[:2] = 0

        # 4초 후 장애물 추가
        if not added_obstacle and (time.time() - start_time) >= 4.0:
            obstacle_pose = sm.SE3(float(4.5), float(0.0), float(0.375))
            new_obstacle = sg.Cuboid([0.5, 0.5, 0.75], pose=obstacle_pose, color=[1.0, 0.0, 0.0, 1.0])
            env.add(new_obstacle)
            collisions.append(new_obstacle)
            added_obstacle = True
            print("장애물 추가됨!")

        time.sleep(dt)


# 환경 초기화
env = swift.Swift()
env.launch(realtime=True)
ax_goal = sg.Axes(0.1)
env.add(ax_goal)

moma = rtb.models.Momaintegratetheta()
moma.q = moma.qr
env.add(moma)

collisions, shelf_positions = [], []
for i in range(4):
    for side in [-2.0, 2.0]:
        sx, sy = float(i + 1) * 1.0, float(side)
        shelf_positions.append((sx, sy))
        for j in [1, 2]:
            z = float(j) * (1.5 / 4)
            panel = sg.Cuboid([1.2, 0.4, 0.03], pose=sm.SE3(sx, sy, z), color=[0.6, 0.4, 0.2])
        wall = sg.Cuboid([1.2, 0.4, 0.75], pose=sm.SE3(sx, sy, 0.375), color=[0.8, 0.8, 0.8, 1.0])
        env.add(wall)
        collisions.append(wall)

rpy = [-(np.pi / 2), (np.pi), 0]
wTep = sm.SE3(6.0, 0.0, 1.0) * sm.SE3.RPY(rpy, order='xyz')
ax_goal.T = wTep
env.step()
env.set_camera_pose([-4.5, 4.5, 1.2], [0, 0, 0.6])
dt = 0.025

# 시뮬레이션 실행
sim_thread = threading.Thread(target=simulation_thread)
sim_thread.start()
sim_thread.join()
env.hold()
import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import random
import threading
import time
import math

# QP 제어 함수
def step_robot(r: rtb.ERobot, Tep):
    wTe = r.fkine(r.q)
    eTep = np.linalg.inv(wTe) @ Tep
    et = np.sum(np.abs(eTep[:3, -1]))
    Y = 0.01
    Q = np.eye(r.n + 6)
    Q[:r.n, :r.n] *= Y
    Q[:2, :2] *= 1.0 / et
    Q[r.n:, r.n:] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)
    v[3:] *= 1.3
    Aeq = np.c_[r.jacobe(r.q), np.eye(6)]
    beq = v.reshape((6,))

    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)
    Ain[:r.n, :r.n], bin[:r.n] = r.joint_velocity_damper(0.1, 0.9, r.n)

    for collision in collisions:
        c_Ain, c_bin = r.link_collision_damper(
            collision, r.q, 0.2, 0.12, 1.0,
            start=moma.link_dict['base_link'],
            end=moma.link_dict['panda_hand']
        )
        if c_Ain is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 3))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    c = np.concatenate(-r.jacobm().reshape((r.n)), np.zeros(6))

    kε = 0.1
    bTe = r.fkine(r.q, include_base=False).A
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

    lb = -np.r_[r.qdlim[:r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[:r.n], 10 * np.ones(6)]
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")
    qd = qd[:r.n]

    qd *= 0.7 / et if et > 0.5 else 1.4
    return (et < 0.02), qd


# 전역 변수
arrived = False
second_phase_started = False

# 시뮬레이션 스레드
def simulation_thread():
    global arrived, second_phase_started, wTep, goal_box, collisions

    start_time = time.time()
    added_obstacle = False

    while True:
        arrived_flag, qd = step_robot(moma, wTep.A)
        moma.qd = qd
        env.step(dt)

        base_new = moma.fkine(moma.q, end=moma.links[2])
        moma._T = base_new.A
        moma.q[:2] = 0

        # 4초 후 장애물 추가
        if not added_obstacle and (time.time() - start_time) >= 4.0:
            
            new_obstacle = sg.Cuboid([0.5, 0.5, 0.75], pose=[4.5, 1, 0.375], color=[1.0, 0.0, 0.0, 1.0])
            env.add(new_obstacle)
            collisions.append(new_obstacle)
            added_obstacle = True
            print("장애물 추가됨!")

        time.sleep(dt)


# 환경 초기화
env = swift.Swift()
env.launch(realtime=True)
ax_goal = sg.Axes(0.1)
env.add(ax_goal)

moma = rtb.models.Momaintegratetheta()
moma.q = moma.qr
env.add(moma)

collisions, shelf_positions = [], []
for i in range(4):
    for side in [-2.0, 2.0]:
        sx, sy = float(i + 1) * 1.0, float(side)
        shelf_positions.append((sx, sy))
        for j in [1, 2]:
            z = float(j) * (1.5 / 4)
            panel = sg.Cuboid([1.2, 0.4, 0.03], pose=sm.SE3(sx, sy, z), color=[0.6, 0.4, 0.2])
        wall = sg.Cuboid([1.2, 0.4, 0.75], pose=sm.SE3(sx, sy, 0.375), color=[0.8, 0.8, 0.8, 1.0])
        env.add(wall)
        collisions.append(wall)

rpy = [-(np.pi / 2), (np.pi), 0]
wTep = sm.SE3(6.0, 0.0, 1.0) * sm.SE3.RPY(rpy, order='xyz')
ax_goal.T = wTep
env.step()
env.set_camera_pose([-4.5, 4.5, 1.2], [0, 0, 0.6])
dt = 0.025

# 시뮬레이션 실행
sim_thread = threading.Thread(target=simulation_thread)
sim_thread.start()
sim_thread.join()
env.hold()
