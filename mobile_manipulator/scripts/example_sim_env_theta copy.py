import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math
import matplotlib.pyplot as plt

def step_robot(r: rtb.ERobot, Tep):
    wTe = r.fkine(r.q)
    eTep = np.linalg.inv(wTe) @ Tep
    et = np.sum(np.abs(eTep[:3, -1]))
    Y = 0.01

    Q = np.eye(r.n + 6)
    Q[: r.n, : r.n] *= Y
    Q[:2, :2] *= 1.0 / et
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)
    v[3:] *= 1.3

    Aeq = np.c_[r.jacobe(r.q), np.eye(6)]
    beq = v.reshape((6,))

    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)
    ps = 0.05
    pi = 0.9
    Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)

    for collision in collisions:
        c_Ain, c_bin = moma.link_collision_damper(
            collision,
            moma.q[:r.n],
            0.5,
            0.05,
            1.5,
            start=moma.link_dict["chassis_link"],
            end=moma.link_dict["panda_hand"],
        )
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 2))]
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    c = np.concatenate(
        (np.zeros(2), -r.jacobm(start=r.links[4]).reshape((r.n - 2,)), np.zeros(6))
    )
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False).A
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

    lb = -np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[: r.n], 10 * np.ones(6)]

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

# ---------- 시각화용 변수 및 함수 ----------
manip_log = []
time_log = []
dt = 0.05

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Manipulability")
ax.set_ylim(0, 2)
ax.set_xlim(0, 1)
plt.title("Manipulability Over Time")
plt.grid(True)

def update_plot(t, m):
    time_log.append(t)
    manip_log.append(m)
    line.set_data(time_log, manip_log)
    ax.set_xlim(0, max(0.1, t))
    ax.set_ylim(0.08, max(0.13, max(manip_log)))
    plt.draw()
    plt.pause(0.001)

# ---------- 시뮬레이션 초기화 ----------
env = swift.Swift()
env.launch(realtime=True)

ax_goal = sg.Axes(0.1)
env.add(ax_goal)

moma = rtb.models.Momatheta()
panda = rtb.models.Panda()
initial_position = sm.SE3(0, 0, 0.324)
moma.q = moma.qr
moma.base = initial_position
env.add(moma)

s0 = sg.Cuboid(scale=(20.0, 0.1, 20.0), pose=sm.SE3(0.0, 1.0, 0.0))
s1 = sg.Cuboid(scale=(20.0, 0.1, 20.0), pose=sm.SE3(0.0, -1.0, 0.0))
s0.v = [0, 0, 0, 0, 0, 0]
s1.v = [0, 0, 0, 0, 0, 0]
collisions = [s0, s1]

env.add(s0)
env.add(s1)

env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])

goal_position = [6, 0.2, 0.6]
goal_orientation_rpy = [np.pi, np.pi, 0]
wTep = sm.SE3(goal_position) * sm.SE3.RPY(goal_orientation_rpy, order='xyz')
ax_goal.T = wTep
env.step()

# ---------- 메인 루프 ----------
arrived = False
step_count = 0

while not arrived:
    arrived, moma.qd = step_robot(moma, wTep.A)

    # manipulability 계산 및 실시간 플롯
    J = panda.jacobe(moma.q[2:9])
    w = np.sqrt(np.linalg.det(J @ J.T))
    update_plot(step_count * dt, w)

    env.step(dt)

    base_new = moma.fkine(moma._q, end=moma.links[2]).A
    moma._T = base_new
    moma.q[:2] = 0
    step_count += 1

plt.ioff()
env.hold()
