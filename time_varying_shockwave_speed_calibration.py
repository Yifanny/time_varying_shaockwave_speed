import numpy as np
import pickle, copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

filename = "NGSIM/trajectories-0750am-0805am-lane1.pkl"
f = open(filename, "rb")
data = pickle.load(f)
f.close()
t_traj = {}

y_min = 999
y_max = 0
v_min = 999
v_max = 0
t_min = 999
t_max = 0
for v_id, traj in data.items():
    traj = np.array(traj)
    if y_min > np.min(traj[:, 1]):
        y_min = np.min(traj[:, 1])
    if y_max < np.max(traj[:, 1]):
        y_max = np.max(traj[:, 1])
    if v_min > np.min(traj[:, 2]):
        v_min = np.min(traj[:, 2])
    if v_max < np.max(traj[:, 2]):
        v_max = np.max(traj[:, 2])
    if t_min > np.min(traj[:, 0])/10:
        t_min = np.min(traj[:, 0])/10
    if t_max < np.max(traj[:, 0])/10:
        t_max = np.max(traj[:, 0])/10
    print(traj.shape)
    for line in traj:
        frame_id = line[0]
        if frame_id in t_traj:
            t_traj[frame_id].append(line)
        else:
            t_traj[frame_id] = [line]
print(y_min, y_max)
print(v_min, v_max)


loop_det_data = []
visited_t = []
for v_id, traj in data.items():
    traj = np.array(traj)
    traj = traj[traj[:, 0].argsort()]
    # if not (traj[0][1] < y_min + 100 and traj[-1][1] > y_max - 100):
    #     continue
    min_d = []
    for line in traj:
        min_d.append(np.abs(line[1] - 40))
    if np.min(min_d) > 10:
        continue
    # print(min_d)
    min_d = np.array(min_d)
    srt_idx = min_d.argsort()
    for i in srt_idx:
        min_p = traj[i]
        if min_p[0] in visited_t:
            continue
        else:
            visited_t.append(min_p[0])
            break
    loop_det_data.append(np.append(min_p, v_id))
loop_det_data = np.array(loop_det_data)
loop_det_data = loop_det_data[loop_det_data[:, 0].argsort()]
print(loop_det_data.shape)


veh_chord = {}
for i in range(len(loop_det_data) - 1):
    v_id = loop_det_data[i][-1]
    t_vel = np.array([loop_det_data[i::, 0] / 10, loop_det_data[i::, 2], loop_det_data[i::, 1]]).T
    veh_chord[v_id] = t_vel
    print(v_id, t_vel.shape)


def cal_tau(h_j, v_j, u_c):
    return h_j / (1 + v_j / u_c)


def cal_x(v_j, tau_j):
    return v_j * tau_j


def cal_err(traj, ts, ys):
    err = {}
    err_t = {}
    err_y = {}
    traj = np.array(traj)
    traj = traj[traj[:, 0].argsort()]
    for i in range(len(ts)):
        t = ts[i]
        y = ys[i]
        for j in range(len(traj)):
            if j < len(traj) - 1:
                if traj[j][1] <= y <= traj[j + 1][1]:
                    err[i] = np.abs(t - traj[j][0] / 10)
                    err_t[i] = traj[j][0] / 10
                    err_y[i] = traj[j][1]
                    break
            else:
                err[i] = np.abs(t - traj[j][0] / 10)
                err_t[i] = traj[j][0] / 10
                err_y[i] = traj[j][1]
    # err = np.array(err)
    # err_t = np.array(err_t)
    # err_y = np.array(err_y)
    return err, err_t, err_y


def cal_fix_err(traj, t, y):
    traj = np.array(traj)
    traj = traj[traj[:, 0].argsort()]
    err = 999

    for j in range(len(traj) - 1):
        if traj[j][1] <= y <= traj[j + 1][1]:
            # print(t)
            err = np.abs(t - traj[j][0]/10)
            break
    return err

# tau_samples, x_samples = cal_tau_x(Tau, X, chord[j][0], chord[j + 1][0], u_c, v_j, chord[0][0], chord[0][2])
def cal_tau_x(Tau, X, t_0, t_1, u_c, v_j, init_t, init_y):
    start_p = [init_t + np.sum(Tau), init_y + np.sum(X)]
    traj_seg_k = v_j
    traj_seg_b = start_p[1] - traj_seg_k * start_p[0]

    point_on_x_axis = [t_1, init_y]
    wave_line_k = -np.array(u_c)
    wave_line_b = point_on_x_axis[1] - wave_line_k * point_on_x_axis[0]

    cross_point_x = (wave_line_b - traj_seg_b) / (traj_seg_k - wave_line_k)
    cross_point_y = cross_point_x * traj_seg_k + traj_seg_b

    tau_j = cross_point_x - np.sum(Tau) - init_t
    x_j = cross_point_y - np.sum(X) - init_y

    return tau_j, x_j



threshold = 10
sample_size = 20000
u_c_fix = 20.53806
max_iter = 1
# f = open("0750am-0805am-lane1_reconstructed_traj_ada_w.pkl", "rb")
# recon_traj = pickle.load(f)
# f.close()

# f = open("0750am-0805am-lane1_reconstructed_traj_fix_w.pkl", "rb")
# fix_recon_traj = pickle.load(f)
# f.close()
recon_traj = {}
fix_recon_traj = {}
print(len(veh_chord))
for v_id, chord in veh_chord.items():
    print("v_id", v_id)
    if not (data[v_id][0][1] < y_min + 100 and data[v_id][-1][1] > y_max - 100):
        continue
    # if not 300 < int(v_id) < 600:
    #     continue
    # if int(v_id) > 866:
    #     break
    if v_id in recon_traj:
        continue
    print(v_id, chord.shape)

    """
    selected_chord = []
    gt_traj = data[v_id]
    traj_cursor = 0
    for i in range(1, len(chord)):
        if chord[i][0] > gt_traj[-1][0] / 10:
            break
        for j in range(traj_cursor, len(gt_traj)):
            if np.abs(gt_traj[j][2] - chord[i][1]) < 3:
                if i in selected_chord:
                    continue
                else:
                    selected_chord.append(i)
            else:
                traj_cursor = copy.deepcopy(j)
                break
    print(selected_chord)
    continue
    """

    X = [0]
    Tau = [0]
    fix_X = [0]
    fix_Tau = [0]
    fix_finish = False
    opt_finish = False
    adaptive_u_c = []
    for j in range(len(chord) - 1):
        if fix_finish and opt_finish:
            break
        opt_id = None
        h_j = chord[j + 1][0] - chord[j][0]
        print(len(chord), fix_finish, opt_finish)
        print("h_j", h_j)
        v_j = chord[j][1]
        print("v_j", v_j)
        if h_j == 0:
            h_j = chord[j + 2][0] - chord[j][0]

        # fix w
        if not fix_finish:
            fix_tau_j = cal_tau(h_j, v_j, u_c_fix)
            fix_x_j = cal_x(v_j, fix_tau_j)
            fix_err = cal_fix_err(data[v_id], chord[0][0] + np.sum(fix_Tau) + fix_tau_j, chord[0][2] + np.sum(fix_X) + fix_x_j)
            fix_y = chord[0][2] + np.sum(fix_X)
            fix_b = fix_y - v_j * (chord[0][0] + np.sum(fix_Tau))
            if v_id in fix_recon_traj:
                if chord[0][2] + np.sum(fix_X) + fix_x_j > data[v_id][-1][1]:
                    fix_tau_j = (data[v_id][-1][1] - fix_b) / v_j - (chord[0][0] + np.sum(fix_Tau))
                    fix_recon_traj[v_id].append([v_j, fix_b, chord[0][0] + np.sum(fix_Tau), fix_tau_j, fix_x_j, fix_err])
                    fix_finish = True
                else:
                    fix_recon_traj[v_id].append([v_j, fix_b, chord[0][0] + np.sum(fix_Tau), fix_tau_j, fix_x_j, fix_err])
            else:
                if chord[0][2] + np.sum(fix_X) + fix_x_j > data[v_id][-1][1]:
                    fix_tau_j = (data[v_id][-1][1] - fix_b) / v_j - (chord[0][0] + np.sum(fix_Tau))
                    fix_recon_traj[v_id] = [[v_j, fix_b, chord[0][0] + np.sum(fix_Tau), fix_tau_j, fix_x_j, fix_err]]
                    fix_finish = True
                else:
                    fix_recon_traj[v_id] = [[v_j, fix_b, chord[0][0] + np.sum(fix_Tau), fix_tau_j, fix_x_j, fix_err]]
            fix_Tau.append(fix_tau_j)
            fix_X.append(fix_x_j)
        else:
            fix_err = 999
            fix_tau_j = 999
            fix_x_j = 999


        # adaptive w
        if not opt_finish:
            u_c_min = 17
            u_c_max = 25
            iters = 0
            opt_u_c = 20
            opt_err = 999999
            change_v_j = 0
            opt_tau_js = []
            opt_x_js = []
            opt_errs= []
            opt_u_cs = []
            while True:
                tau_j = []
                x_j = []
                while True:
                    u_c = np.random.uniform(low=u_c_min, high=u_c_max, size=sample_size)
                    if j == 0:
                        tau_samples = cal_tau(h_j, v_j, u_c)
                        x_samples = cal_x(v_j, tau_samples)
                    else:
                        tau_samples, x_samples = cal_tau_x(Tau, X, chord[j][0], chord[j + 1][0], u_c, v_j, chord[0][0], chord[0][2])
                    for k, this_tau_j in enumerate(tau_samples):
                        if j == 0:
                            if this_tau_j > 0:
                                tau_j.append(this_tau_j)
                                x_j.append(x_samples[k])
                        else:
                            if this_tau_j > 0 and x_samples[k] > 0:
                                x_j.append(x_samples[k])
                                tau_j.append(this_tau_j)
                        if len(tau_j) == sample_size:
                            tau_j = np.array(tau_j)
                            break
                    print("tau_j", np.array(x_j).shape, np.array(tau_j).shape)
                    if len(tau_j) == sample_size:
                        break

                x_j = np.array(x_j)
                err, err_t, err_y = cal_err(data[v_id], chord[0][0] + np.sum(Tau) + tau_j, chord[0][2] + np.sum(X) + x_j)
                if len(err) == 0:
                    print("err is empty")
                    opt_x_j = 0
                    opt_u_c = 0
                    opt_tau_j = 0
                    err = {999999: 999999}
                    break
                # print(min(err.values()), np.max([fix_err, threshold]))
                # if min(err.values()) > np.max([fix_err, threshold]):
                #     opt_id = min(err, key=err.get)
                #     opt_u_c = u_c[opt_id]
                #     opt_tau_j = tau_j[opt_id]
                #     opt_x_j = x_j[opt_id]
                #     print("err is too large", min(err.values()), fix_err, u_c[opt_id], err_t[opt_id], err_y[opt_id],
                #             np.sum(Tau) + chord[0][0], np.sum(X) + chord[0][2],
                #             np.sum(Tau) + chord[0][0] + opt_tau_j, np.sum(X) + chord[0][2] + opt_x_j)
                    # break
                opt_id = min(err, key=err.get)
                print("min err", err[opt_id], u_c[opt_id], x_j[opt_id], tau_j[opt_id], v_id, j)
                print(np.sum(Tau) + chord[0][0] + tau_j[opt_id], np.sum(X) + chord[0][2] + x_j[opt_id])
                print(err_t[opt_id], err_y[opt_id])
                print("fix err", fix_err)
                # if min(err) < opt_err:
                opt_u_c = u_c[opt_id]
                opt_tau_js.append(tau_j[opt_id])
                opt_x_js.append(x_j[opt_id])
                opt_u_cs.append(u_c[opt_id])
                opt_errs.append(err[opt_id])
                if min(err.values()) > 0.5 and np.sum(Tau) + chord[0][0] + tau_j[opt_id] > err_t[opt_id] and v_j < v_max:
                    print("change larger v_j", v_j)
                    v_j += 1 # np.random.normal(chord[0][1], 1)
                    change_v_j = 1
                    print(v_j)
                elif min(err.values()) > 0.5 and np.sum(Tau) + chord[0][0] + tau_j[opt_id] < err_t[opt_id] and v_j - 1 > v_min:
                    print("change smaller v_j", v_j)
                    v_j -= 1 # np.random.normal(chord[0][1], 1)
                    change_v_j = -1
                    print(v_j)
                else:
                    opt_tau_j = tau_j[opt_id]
                    opt_x_j = x_j[opt_id]
                    opt_err = min(err.values())
                    print(opt_tau_j, opt_x_j, fix_tau_j, fix_x_j)
                    break
                iters += 1
                opt_id = np.argmin(opt_errs)
                opt_tau_j = opt_tau_js[opt_id]
                opt_x_j = opt_x_js[opt_id]
                opt_err = opt_errs[opt_id]
                opt_u_c = opt_u_cs[opt_id]
                if iters > 100:
                    break
                    

            y = chord[0][2] + np.sum(X)
            b = y - v_j * (chord[0][0] + np.sum(Tau))
            if v_id in recon_traj:
                if chord[0][2] + np.sum(X) + opt_x_j > data[v_id][-1][1]:
                    opt_tau_j = (data[v_id][-1][1] - b) / v_j - (chord[0][0] + np.sum(Tau))
                    recon_traj[v_id].append([v_j, b, chord[0][0] + np.sum(Tau), opt_tau_j, opt_x_j, opt_u_c, opt_err, change_v_j])
                    opt_finish = True
                else:
                    recon_traj[v_id].append([v_j, b, chord[0][0] + np.sum(Tau), opt_tau_j, opt_x_j, opt_u_c, opt_err, change_v_j])
            else:
                if chord[0][2] + np.sum(X) + opt_x_j > data[v_id][-1][1]:
                    opt_tau_j = (data[v_id][-1][1] - b) / v_j - (chord[0][0] + np.sum(Tau))
                    recon_traj[v_id] = [[v_j, b, chord[0][0] + np.sum(Tau), opt_tau_j, opt_x_j, opt_u_c, opt_err, change_v_j]]
                    opt_finish = True
                else:
                    recon_traj[v_id] = [[v_j, b, chord[0][0] + np.sum(Tau), opt_tau_j, opt_x_j, opt_u_c, opt_err, change_v_j]]
            Tau.append(opt_tau_j)
            X.append(opt_x_j)
            adaptive_u_c.append(opt_u_c)

    # break

    f = open("0750am-0805am-lane1_reconstructed_traj_ada_w.pkl", "wb")
    pickle.dump(recon_traj, f)
    f.close()

    f = open("0750am-0805am-lane1_reconstructed_traj_fix_w.pkl", "wb")
    pickle.dump(fix_recon_traj, f)
    f.close()

print(len(recon_traj))
