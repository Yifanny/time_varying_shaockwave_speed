import copy
import numpy as np
import pickle, traceback
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dtaidistance import dtw
from co2mpas_driver import dsp as driver
from co2mpas_driver.common import vehicle_functions as vf
from co2mpas_driver.common import reading_n_organizing as rno
from co2mpas_driver.common import gear_functions as fg
from localreg import *
from scipy.interpolate import interp1d
import pandas as pd
import warnings
from scipy.stats import cauchy
warnings.filterwarnings("ignore")


filename = "NGSIM/trajectories-0750am-0805am-lane1.pkl"
f = open(filename, "rb")
data = pickle.load(f)
f.close()
t_traj = {}


def simple_run(v_des_list, time_steps, position, car_id):
    gs_style = 1  # np.random.random() * 0.4 + 0.4   # 0.6
    db_path = 'H:/UTD19-py39/venv/Lib/site-packages/co2mpas_driver/db/EuroSegmentCar_cleaned.csv'
    sim_step = 0.1
    driver_style = 1 # np.random.random() * 0.4 + 0.4  # 0.6
    duration = time_steps[-1] - time_steps[0]
    times = np.arange(time_steps[0], time_steps[-1] + sim_step, sim_step)
    v_start = v_des_list[0]

    f_des = interp1d(time_steps, v_des_list, kind="next")
    """
    acc_tmp = []
    for i in range(len(v_des_list) - 1):
        acc_tmp.append((v_des_list[i + 1] - v_des_list[i]) / (time_steps[i + 1] - time_steps[i]))
    ds_list = []
    for i, acc in enumerate(acc_tmp):
        v = v_des_list[i]
        max_acc = final_acc_curve_f(v)[1]
        min_acc = final_dec_curve_f(v)[1]
        if acc > 0:
            ds_list.append(acc/max_acc)
        elif acc < 0:
            ds_list.append(acc/min_acc)
        else:
            ds_list.append(1)
    """

    # driver_style = np.mean(ds_list)
    # print("driver style", driver_style)

    my_veh = driver(
        dict(
            db_path=db_path,
            vehicle_id=car_id,
            inputs=dict(
                inputs=dict(
                    gear_shifting_style=gs_style,
                    starting_velocity=v_start,
                    driver_style=driver_style,
                    sim_start=0,
                    sim_step=sim_step,
                    duration=duration,
                    degree=4,
                    use_linear_gs=True,
                    use_cubic=False,
                )
            ),
        )
    )["outputs"]["driver_simulation_model"]
    # print(time_steps)
    # print(v_des_list)

    res = {}
    for i, myt in enumerate(times):
        # print(myt, f_des(myt))
        if myt > time_steps[-1] - sim_step:
            break
        if myt == times[0]:
            my_veh.reset(v_start)
            res = {
                "time_step": [myt],
                "accel": [0],
                "speed": [v_start],
                "position": [position[0]],
                "gear": [0],
                "v_des": [f_des(myt)],
                "fc": [],
            }
            continue
        res["v_des"].append(f_des(myt))
        try:
            sim_res = my_veh(sim_step, res["v_des"][-1])
        except ValueError as e:
            print(myt, f_des(myt))
            print(time_steps)
            print(v_des_list)
            print(e)
            traceback.print_exc()
            exit()
        gear = sim_res[0]
        gear_count = sim_res[1]
        next_velocity = sim_res[2]
        acc = sim_res[3]
        gear_box_speeds_in = sim_res[5]
        gear_box_powers_in = sim_res[11]
        # print(myt, "/", duration, gear, gear_count, next_velocity, acc, res["position"][-1])
        if isinstance(next_velocity, np.ndarray):
            next_velocity = next_velocity[0]
        if isinstance(acc, np.ndarray):
            acc = acc[0]

        fc = my_veh.calculate_fuel_consumption(gear, sim_step, gear_box_speeds_in, gear_box_powers_in)

        res["accel"].append(acc)
        res["speed"].append(next_velocity)
        res["gear"].append(gear)
        res["position"].append(res["position"][-1] + next_velocity * sim_step)
        res["fc"].append(list(fc))
        res["time_step"].append(myt)

    # print(res["position"][-1], position[-1])

    return res


def traj_smooth(car_id, original_time, v_list, start_y, std=0.16, driver_style=1, gs_style=1):
    db_path = 'H:/UTD19-py39/venv/Lib/site-packages/co2mpas_driver/db/EuroSegmentCar_cleaned'
    db = rno.load_db_to_dictionary(db_path)
    my_car = rno.get_vehicle_from_db(db, car_id)
    full_load_speeds, full_load_torques = vf.get_load_speed_n_torque(my_car)
    """Speed and acceleration ranges and points for each gear"""
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        my_car, full_load_speeds, full_load_torques)
    """Extract speed acceleration Splines"""
    coefs_per_gear = vf.get_tan_coefs(speed_per_gear, acc_per_gear, 4)
    poly_spline = vf.get_spline_out_of_coefs(coefs_per_gear,
                                             speed_per_gear[0][0])
    """Start/stop speed for each gear"""
    start, stop = vf.get_start_stop(my_car, speed_per_gear, acc_per_gear,
                                    poly_spline)

    sp_bins = np.arange(0, stop[-1] + 1, 0.01)
    """Get resistances"""
    car_res_curve, car_res_curve_force, Alimit = vf.get_resistances(
        my_car, sp_bins)
    """Calculate Curves"""
    curves = vf.calculate_curves_to_use(poly_spline, start, stop, Alimit,
                                        car_res_curve, sp_bins)
    dec_curves = vf.calculate_dec_curves_to_use(sp_bins)
    """Get gear limit"""
    # gs_style = np.random.random() * 0.4 + 0.4  # 0.6
    gs_limits = fg.gear_linear(speed_per_gear, gs_style)
    # from co2mpas_driver.model import define_discrete_acceleration_curves as func
    # discrete_acceleration_curves = func(curves, start, stop)
    # for d in discrete_acceleration_curves:
    #     plt.plot(d['x'], d['y'])
    final_acc_curves = []
    for d in curves:
        final_acc_curves.append(d(sp_bins))
    final_acc_curves = np.array(final_acc_curves).T
    final_acc_curve = []
    for i in range(final_acc_curves.shape[0]):
        final_acc_curve.append(np.max(final_acc_curves[i]))
    final_dec_curve = dec_curves[0](sp_bins)
    end_point = -1
    for i in range(len(final_acc_curve)):
        if final_acc_curve[i] > final_dec_curve[i]:
            continue
        else:
            end_point = i
            break
    final_acc_curve = (sp_bins[0:end_point], final_acc_curve[0:end_point])
    final_dec_curve = (sp_bins[0:end_point], final_dec_curve[0:end_point])

    final_acc_curve_f = interp1d(sp_bins[0:end_point], final_acc_curve[0:end_point])
    final_dec_curve_f = interp1d(sp_bins[0:end_point], final_dec_curve[0:end_point])

    finish = False
    time = np.arange(original_time[0], original_time[-1] + 0.1, 0.1)
    speed_noisy_bounded = np.interp(time, original_time, v_list)
    tmp_acc = np.append(np.diff(speed_noisy_bounded) * 10, 0)
    for i, acc in enumerate(tmp_acc):
        sp = speed_noisy_bounded[i]
        max_acc = final_acc_curve_f(sp)[1]
        min_acc = final_dec_curve_f(sp)[1]
        if min_acc <= acc <= max_acc:
            continue
        elif acc > max_acc:
            tmp_acc[i] = max_acc
        elif acc < min_acc:
            tmp_acc[i] = min_acc
    for i, sp in enumerate(speed_noisy_bounded):
        if i == 0:
            continue
        else:
            speed_noisy_bounded[i] = speed_noisy_bounded[i - 1] + tmp_acc[i] * 0.1
    window = 7
    freq = 10
    std_a_threshold = 100
    conv_rate_prev = 100
    perc_threshold = 0.3
    while finish == False:
        tmp_sp = localreg(time, speed_noisy_bounded, degree=6, kernel=rbf.tricube,
                          radius=window / freq)  # frac=window / len(time)
        tmp_sp = np.maximum(0, tmp_sp)
        tmp_acc = np.append(np.diff(tmp_sp) * freq, 0)


        tmp_acc_s = pd.Series(tmp_acc)
        std_a_curr = tmp_acc_s.rolling(7 * freq).std().median()

        conv_rate_curr = (std_a_threshold - std_a_curr) / std_a_threshold

        stopnow = False
        if conv_rate_curr > conv_rate_prev:
            stopnow = True
        else:
            std_a_threshold = std_a_curr
            conv_rate_prev = conv_rate_curr

        # The window is affected by the frequency - that is why we multiply by 10
        if window < 30 * freq and stopnow == False:
            window += 2
            # print(window)
            # print(conv_rate_curr)
            continue

        finish = True
        unreal_values = False
        outliers = 0
        for idx, (s, ac) in enumerate(zip(tmp_sp, tmp_acc)):
            try:
                gear, gear_count = fg.gear_for_speed_profiles(gs_limits, s, 0, 0)
            except:
                unreal_values = True
                finish = False
                break
            max_acc = np.interp([s], final_acc_curve[0], final_acc_curve[1])
            min_acc = np.interp([s], final_dec_curve[0], final_dec_curve[1])

            # max_acc = Curves1[gear - 1](s)
            # min_acc = Curves_dec1[0](s)

            # Check compliance with the driver
            # driver_style = 1 #np.random.random() * 0.4 + 0.4  # 0.6
            if ac > driver_style * max_acc:
                outliers += 1
            if ac < min_acc:
                outliers += 1
        if outliers / len(tmp_sp) > perc_threshold:
            finish == False
        if window > 30 * freq:
            finish = True
        if finish == False:
            window += 2
            # print(window, std_a_threshold)
    # The window is affected by the frequency - that is why we divide by 10
    speed_noisy_bounded_filtered = localreg(time, speed_noisy_bounded, degree=6, kernel=rbf.tricube,
                                            radius=window / freq)  # frac=window / len(time)
    speed_noisy_bounded_filtered = np.array(speed_noisy_bounded_filtered)
    # {'cauchy': {'loc': 0.006960991719462949, 'scale': 0.08280889174415673}}
    # print(cauchy.rvs(loc=0.006960991719462949, scale=0.08280889174415673, size=1), np.random.normal(0.0, 0.16, 1))
    speed_noisy_bounded_filtered = np.maximum(0, speed_noisy_bounded_filtered + np.random.normal(0.0, std, len(speed_noisy_bounded_filtered)))
    # + cauchy.rvs(loc=0.006960991719462949,scale=0.08280889174415673,size=len(speed_noisy_bounded_filtered)))
    acc_noisy_bounded_filtered = np.append(np.diff(speed_noisy_bounded_filtered) * freq, 0)

    """
    w = 20
    repeat_l_pos = interp1d(l_traj[:, 0] + delta_t, l_traj[:, 1] - w * delta_t)

    alpha = 0.2
    adjusted_speed = []
    for i, v in enumerate(speed_noisy_bounded_filtered):
        if i + 1 == len(speed_noisy_bounded_filtered):
            break
        if time[i + 1] > l_traj[:, 0][-1] + delta_t:
            adjusted_speed.append(v)
            continue
        newell_speed = (repeat_l_pos(time[i+1]) - repeat_l_pos(time[i])) * freq
        adjusted_speed.append(newell_speed * alpha + v * (1 - alpha))
    adjusted_speed.append(speed_noisy_bounded_filtered[-1])
    new_pos = []
    for i, v in enumerate(adjusted_speed):
        if i == 0:
            new_pos.append(start_y + v * (1 / freq))
        else:
            new_pos.append(new_pos[-1] + v * (1 / freq))
    print("in traj smooth", len(time), len(new_pos), len(adjusted_speed))
    """
    new_pos = []
    for i, v in enumerate(speed_noisy_bounded_filtered):
        if i == 0:
            new_pos.append(start_y + v * (1 / freq))
        else:
            new_pos.append(new_pos[-1] + v * (1 / freq))
    return (time, np.array(new_pos)), speed_noisy_bounded_filtered, acc_noisy_bounded_filtered



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


f = open("0750am-0805am-lane1_reconstructed_traj_ada_w.pkl", "rb")
recon_traj = pickle.load(f)
f.close()

f = open("0750am-0805am-lane1_reconstructed_traj_ada_w_from1400.pkl", "rb")
recon_traj_1 = pickle.load(f)
f.close()

f = open("0750am-0805am-lane1_reconstructed_traj_ada_w_from1600.pkl", "rb")
recon_traj_2 = pickle.load(f)
f.close()

for v_id, traj in recon_traj_1.items():
    if v_id in recon_traj:
        continue
    else:
        recon_traj[v_id] = traj

for v_id, traj in recon_traj_2.items():
    if v_id in recon_traj:
        continue
    else:
        recon_traj[v_id] = traj

f = open("0750am-0805am-lane1_reconstructed_traj_fix_w.pkl", "rb")
fix_recon_traj = pickle.load(f)
f.close()

f = open("0750am-0805am-lane1_reconstructed_traj_fix_w_from1400.pkl", "rb")
fix_recon_traj_1 = pickle.load(f)
f.close()

f = open("0750am-0805am-lane1_reconstructed_traj_fix_w_from1600.pkl", "rb")
fix_recon_traj_2 = pickle.load(f)
f.close()

for v_id, traj in fix_recon_traj_1.items():
    if v_id in fix_recon_traj:
        continue
    else:
        fix_recon_traj[v_id] = traj

for v_id, traj in fix_recon_traj_2.items():
    if v_id in fix_recon_traj:
        continue
    else:
        fix_recon_traj[v_id] = traj


def color_map(data, cmap, v_min, v_max):
    d_min, d_max = v_min, v_max
    # print(d_min, d_max)
    cmo = plt.colormaps.get_cmap(cmap)
    cs, k = list(), 256/cmo.N

    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i * k), int((i + 1) * k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255*(data-d_min)/(d_max-d_min))

    return cs[data]


cmap = 'rainbow'

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
        min_p = traj[srt_idx[i]]
        if min_p[0] in visited_t:
            continue
        else:
            visited_t.append(min_p[0])
            break
    loop_det_data.append(np.append(min_p, v_id))
loop_det_data = np.array(loop_det_data)
loop_det_data = loop_det_data[loop_det_data[:, 0].argsort()]
print(loop_det_data.shape)
v_ids = loop_det_data[:, -1]


def cal_err(recon_traj, gt_traj):
    recon_t, recon_y = recon_traj
    gt_t, gt_y = gt_traj
    # if recon_y[-1] > gt_y[-1]:
    #     return [0]

    spacing_err = []
    for i, y in enumerate(recon_y):
        for j in range(len(gt_y) - 1):
            if gt_y[j] <= y <= gt_y[j + 1]:
                spacing_err.append(np.abs(gt_t[j] - recon_t[i]))
                break
    return spacing_err


def cal_s_err(recon_traj, gt_traj):
    recon_t, recon_y = recon_traj
    gt_t, gt_y = gt_traj
    # if recon_y[-1] > gt_y[-1]:
    #     return [0]

    spacing_err = []
    for i, t in enumerate(recon_t):
        for j in range(len(gt_t) - 1):
            if gt_t[j] <= t <= gt_t[j + 1]:
                spacing_err.append(np.abs(gt_y[j] - recon_y[i]))
                break
    return spacing_err


def cal_v_err(recon_traj, gt_traj):
    recon_v, recon_y = recon_traj
    gt_v, gt_y = gt_traj
    # if recon_y[-1] > gt_y[-1]:
    #     return [0]

    v_err = []
    for i, y in enumerate(recon_y):
        for j in range(len(gt_y) - 1):
            if gt_y[j] < y < gt_y[j + 1]:
                v_err.append(np.abs(gt_v[j] - recon_v[i]))
                break
    return v_err


def cal_fc_co2_err(recon_res, gt_res):
    recon_fc = np.array(recon_res["fc"])[:, 0]
    recon_co2 = np.array(recon_res["fc"])[:, -2]

    gt_fc = np.array(gt_res["fc"])[:, 0]
    gt_co2 = np.array(gt_res["fc"])[:, -2]

    print("fc co2", np.sum(recon_fc), np.sum(gt_fc), np.sum(recon_co2), np.sum(gt_co2))

    return np.abs(np.sum(recon_fc) - np.sum(gt_fc)), np.abs(np.sum(recon_co2) - np.sum(gt_co2))


def get_same_part_of_gt_traj_as_recon(gt_traj, y_end, y_start):
    # print(gt_traj[0], gt_traj[-1])
    # print(y_start, y_end)
    end = len(gt_traj)
    start = 0
    for i in range(len(gt_traj) - 1):
        if gt_traj[i][1] <= y_end <= gt_traj[i + 1][1]:
            end = i + 2
        if gt_traj[i][1] <= y_start <= gt_traj[i + 1][1]:
            start = i
    return gt_traj[start:end]


def get_same_part_of_gt_traj_as_recon_fix(gt_traj, recon_traj):
    if recon_traj[-1] > gt_traj[-1][1]:
        for i in range(len(recon_traj) - 1):
            if recon_traj[i] <= gt_traj[-1][1] <= recon_traj[i + 1]:
                return gt_traj, recon_traj[0: i + 1]
    else:
        for i in range(len(gt_traj) - 1):
            if gt_traj[i][1] <= recon_traj[-1] <= gt_traj[i + 1][1]:
                return gt_traj[0: i + 1], recon_traj


# different penetration rates
save_fix_err_for_plot = []
save_ada_err_for_plot = []
save_smoothed_err_for_plot = []
save_poly_err_for_plot = []
save_two_fold_err_for_plot = []

save_fix_s_err_for_plot = []
save_ada_s_err_for_plot = []
save_smoothed_s_err_for_plot = []
save_poly_s_err_for_plot = []

save_fix_v_err_for_plot = []
save_ada_v_err_for_plot = []
save_smoothed_v_err_for_plot = []
save_poly_v_err_for_plot = []
save_two_fold_v_err_for_plot = []

save_fix_fc_err_for_plot = []
save_ada_fc_err_for_plot = []
save_smoothed_fc_err_for_plot = []
save_poly_fc_err_for_plot = []
save_two_fold_fc_err_for_plot = []

save_fix_co2_err_for_plot = []
save_ada_co2_err_for_plot = []
save_smoothed_co2_err_for_plot = []
save_poly_co2_err_for_plot = []
save_two_fold_co2_err_for_plot = []
penetration_rate = 0.05
for step in range(0, 50):
    for i, line in enumerate(loop_det_data):
        if line[-1] in fix_recon_traj:
            selected_v_id = [i]
            start_id = i
            break
    for i in range(start_id, len(loop_det_data) - int(1/penetration_rate), int(1/penetration_rate)):
        if loop_det_data[i][-1] > 2000:
            break
        while True:
            id = np.random.randint(i, i + int(1/penetration_rate))
            if loop_det_data[id][-1] in fix_recon_traj and id != start_id and id - selected_v_id[-1] > 2:
                break
        selected_v_id.append(id)

    this_save_fix_err_for_plot = []
    this_save_ada_err_for_plot = []
    this_save_smoothed_err_for_plot = []
    this_save_poly_err_for_plot = []
    this_save_two_fold_err_for_plot = []

    this_save_fix_s_err_for_plot = []
    this_save_ada_s_err_for_plot = []
    this_save_smoothed_s_err_for_plot = []
    this_save_poly_s_err_for_plot = []

    this_save_fix_v_err_for_plot = []
    this_save_ada_v_err_for_plot = []
    this_save_smoothed_v_err_for_plot = []
    this_save_poly_v_err_for_plot = []
    this_save_two_fold_v_err_for_plot = []

    this_save_fix_fc_err_for_plot = []
    this_save_ada_fc_err_for_plot = []
    this_save_smoothed_fc_err_for_plot = []
    this_save_poly_fc_err_for_plot = []
    this_save_two_fold_fc_err_for_plot = []

    this_save_fix_co2_err_for_plot = []
    this_save_ada_co2_err_for_plot = []
    this_save_smoothed_co2_err_for_plot = []
    this_save_poly_co2_err_for_plot = []
    this_save_two_fold_co2_err_for_plot = []

    std = 1
    all_err = []
    fig, ax = plt.subplots()
    start_idx = 0
    total_num_veh = 0
    # for start_idx in range(0, len(v_ids), 15):
    # car_id = np.random.choice([34271, 34265, 6378, 39723, 34092, 2592, 5635, 5630, 7661, 7683, 8709, 9769, 1872, 10328], 1)[0]
    car_id = 2592
    print("car id", car_id)  # 5635, 34265, 7683

    save_recon_traj = {}
    save_recon_poly_traj = {}
    save_recon_smoothed_traj = {}
    for idx in range(1, len(selected_v_id)):
        start_idx = selected_v_id[idx - 1]
        stop_veh = selected_v_id[idx] - selected_v_id[idx - 1]
        print(start_idx, stop_veh)
        new_recon_traj = {}
        new_smoothed_recon_traj = {}
        new_smoothed_ada_recon_traj = {}
        new_poly_recon_traj = {}
        new_two_fold_traj = {}
        new_recon_traj_points = {}
        if not loop_det_data[start_idx][-1] in fix_recon_traj:
            start_idx += 1
            continue
        for i, start_veh in enumerate(loop_det_data[start_idx::]):
            v_id = start_veh[-1]

            if i >= stop_veh:
                break
            if not v_id in fix_recon_traj:
                continue
            y = loop_det_data[i + start_idx][1]
            # [v_j, b, chord[0][0] + np.sum(Tau), opt_tau_j, opt_x_j, opt_u_c, opt_err, change_v_j]
            if i == 0:
                segs = recon_traj[v_id]
                segs = np.array(segs)
                ada_v_j = copy.deepcopy(segs[:, 0])
                ada_u_c = copy.deepcopy(segs[:, 5])
                ada_tau_j = copy.deepcopy(segs[:, 3])
                new_recon_traj[v_id] = recon_traj[v_id]
                continue
            # print("---------------------------------------------------")
            # print(len(ada_v_j))
            for j in range(i, len(ada_v_j)):
                # print(j)
                if j + start_idx + 1 == len(loop_det_data):
                    break
                if j == i:
                    v_j = loop_det_data[j + start_idx][2]
                else:
                    v_j = np.random.normal(ada_v_j[j], std)
                    # v_j = ada_v_j[j]
                # u_c = np.random.normal(ada_u_c[j], 1)
                u_c = ada_u_c[j]
                tau_j = ada_tau_j[j]
                if j == i:
                    tau1 = loop_det_data[j + start_idx][0] / 10
                traj_seg_b = y - v_j * tau1

                tau2 = loop_det_data[j + 1 + start_idx][0] / 10
                wave_line_b = loop_det_data[j + 1 + start_idx][2] + u_c * tau2
                traj_seg_k = v_j
                wave_line_k = -u_c
                # print(v_j, traj_seg_b, tau1, y)
                # print(u_c, wave_line_b, tau2, loop_det_data[j + 1 + start_idx][2])

                cross_point_x = (wave_line_b - traj_seg_b) / (traj_seg_k - wave_line_k)

                cross_point_x = (cross_point_x + tau1 + tau_j) / 2
                # cross_point_x = tau1 + tau_j
                cross_point_y = cross_point_x * traj_seg_k + traj_seg_b
                #  print(cross_point_x, cross_point_y)
                # print("-------------------")

                if cross_point_y > y and cross_point_x > tau1:
                    if v_id in new_recon_traj_points:
                        new_recon_traj_points[v_id].append([cross_point_x, cross_point_y])
                        # new_recon_traj[v_id].append([v_j, traj_seg_b, tau1, cross_point_x - tau1, cross_point_y - y, u_c])
                    else:
                        new_recon_traj_points[v_id] = [
                            [loop_det_data[j + start_idx][0] / 10, loop_det_data[j + start_idx][1]],
                            [cross_point_x, cross_point_y]]
                        # new_recon_traj[v_id] = [[v_j, traj_seg_b, tau1, cross_point_x - tau1, cross_point_y - y, u_c]]
                    y = cross_point_y
                    tau1 = cross_point_x
                # iters += 1
                # if iters % 10000 == 0 and iters > 0:
                #     std += 1

            if v_id in new_recon_traj_points:
                x = np.array(new_recon_traj_points[v_id])[:, 0]
                y = np.array(new_recon_traj_points[v_id])[:, 1] * 0.3048
                # print(x)
                # print(y)

                z1 = np.polyfit(x, y, 5)
                p1 = np.poly1d(z1)
                # print(p1)
                x = np.array(sorted(x))
                xvals = np.arange(x[0], x[-1] + 0.1, 0.1)
                yvals = np.array(p1(xvals))
                vvals = (yvals[2::] - yvals[0:-2]) / (xvals[2::] - xvals[0:-2])
                for i, v in enumerate(vvals):
                    if v_min * 0.3048 <= v <= v_max * 0.3048:
                        continue
                    if v < v_min * 0.3048:
                        vvals[i] = v_min * 0.3048
                    if v > v_max * 0.3048:
                        vvals[i] = v_max * 0.3048
                smoothed_traj, smoothed_v, smoothed_acc = traj_smooth(car_id, xvals[1:-1], vvals, yvals[1])
                new_smoothed_recon_traj[v_id] = [smoothed_traj[0], smoothed_traj[1], smoothed_v]

                new_poly_recon_traj[v_id] = [xvals[1:-1], yvals[1:-1], vvals]

                save_recon_smoothed_traj[v_id] = [smoothed_traj[0], smoothed_traj[1], smoothed_v]
                save_recon_poly_traj[v_id] = [xvals[1:-1], yvals[1:-1], vvals]
                # print(smoothed_traj[0])

                # res = simple_run(vvals, x[1::], yvals[1::])
                vs = (y[2::] - y[0:-2]) / (x[2::] - x[0:-2])
                for i, v in enumerate(vs):
                    if v_min * 0.3048 <= v <= v_max * 0.3048:
                        continue
                    if v < v_min * 0.3048:
                        vs[i] = v_min * 0.3048
                    if v > v_max * 0.3048:
                        vs[i] = v_max * 0.3048
                # print("create recon traj", vs[0])
                res = simple_run(vs, x[1:-1], y[1:-1], car_id)
                # print(x, y, v)
                # print(res["accel"])
                # exit()
                new_recon_traj[v_id] = [res["time_step"], res["position"], res["speed"], res]
                save_recon_traj[v_id] = [res["time_step"], res["position"], res["speed"]]

                # print("create two fold traj", vvals[0])
                # print(smoothed_v)
                # res = simple_run(smoothed_v, smoothed_traj[0], smoothed_traj[1], car_id)
                """
                smoothed_traj1, smoothed_v1, smoothed_acc1 = traj_smooth(car_id, res["time_step"], res["speed"],
                                                                         res["position"][0], 0.8)
                smoothed_traj2, smoothed_v2, smoothed_acc2 = traj_smooth(car_id, res["time_step"], res["speed"],
                                                                         res["position"][0], 0.7)
                smoothed_traj3, smoothed_v3, smoothed_acc3 = traj_smooth(car_id, res["time_step"], res["speed"],
                                                                         res["position"][0], 0.6)
                # print(x, y, v)
                # print(res["accel"])
                new_two_fold_traj[v_id] = [[smoothed_traj1[0], smoothed_traj1[1], smoothed_v1],
                                           [smoothed_traj2[0], smoothed_traj2[1], smoothed_v2],
                                           [smoothed_traj3[0], smoothed_traj3[1], smoothed_v3]]
                """
        # if v_id in new_recon_traj:
        #     print(np.array(new_recon_traj[v_id]).shape)

        # random_color = ["red", "green", "blue", "yellow", "brown"]
        # continue

        n = 0
        fix_err = []
        ada_err = []
        smoothed_err = []
        poly_ada_err = []
        two_fold_ada_err = []

        fix_s_err = []
        ada_s_err = []
        smoothed_s_err = []
        poly_ada_s_err = []

        fix_v_err = []
        ada_v_err = []
        smoothed_v_err = []
        poly_ada_v_err = []
        two_fold_ada_v_err = []

        fix_fc_err = []
        fix_co2_err = []
        ada_fc_err = []
        ada_co2_err = []
        smoothed_fc_err = []

        smoothed_co2_err = []
        poly_ada_fc_err = []
        poly_ada_co2_err = []
        two_fold_ada_fc_err = []
        two_fold_ada_co2_err = []
        for v_id, segs in new_recon_traj.items():
            # if v_id == 137 or v_id == 423:
            #     continue
            # c = random_color[n % 5]
            n += 1
            # print(np.array(segs).shape)
            if not v_id in fix_recon_traj:
                continue
            total_num_veh += 1
            gt_traj = np.array(data[v_id])
            # plt.plot(gt_traj[:, 0] / 10, gt_traj[:, 1], color="black")

            fix_segs = fix_recon_traj[v_id]
            fix_plot_y = []
            fix_plot_t = []
            fix_plot_v = []
            if True: # n != 1:
                for i, seg in enumerate(fix_segs):
                    a = seg[0]
                    b = seg[1]
                    t = np.linspace(seg[2], seg[2] + seg[3], 10)
                    y = a * t + b
                    v = (y[2::] - y[0:-2]) / (t[2::] - t[0:-2])
                    fix_plot_y.extend(list(y[1:-1]))
                    fix_plot_t.extend(list(t[1:-1]))
                    fix_plot_v.extend(list(v))
                    # plt.plot(t, y, color="blue")
                fix_err.extend(cal_err((fix_plot_t, fix_plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                fix_s_err.extend(cal_s_err((fix_plot_t, fix_plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                fix_v_err.extend(cal_v_err((fix_plot_v, fix_plot_y), (gt_traj[:, 2], gt_traj[:, 1])))
                cut_gt_traj, fix_plot_y = get_same_part_of_gt_traj_as_recon_fix(gt_traj, fix_plot_y)
                fix_plot_t = fix_plot_t[0: len(fix_plot_y)]
                fix_plot_v = fix_plot_v[0: len(fix_plot_y)]
                # print("fix", fix_plot_v[0] * 0.3048)
                fix_res = simple_run(np.array(fix_plot_v) * 0.3048, fix_plot_t, np.array(fix_plot_y) * 0.3048, car_id)
                # print("fix gt", cut_gt_traj[0][2] * 0.3048)
                gt_res = simple_run(cut_gt_traj[:, 2] * 0.3048, cut_gt_traj[:, 0] / 10, cut_gt_traj[:, 1] * 0.3048,
                                        car_id)
                print("fix")
                fc_err, co2_err = cal_fc_co2_err(fix_res, gt_res)
                fix_fc_err.append(fc_err)
                fix_co2_err.append(co2_err)

            if n == 1:
                xs = []
                ys = []
                for i, seg in enumerate(segs):
                    a = seg[0]
                    b = seg[1]
                    t = np.linspace(seg[2], seg[2] + seg[3], 10)
                    y = a * t + b
                    # ax.plot(t, y, color="black", linewidth=2)
                    xs.extend(list(t))
                    ys.extend(list(y))
                x = np.array(xs)
                y = np.array(ys)
                z1 = np.polyfit(x, y, 5)
                p1 = np.poly1d(z1)
                #     print(p1)
                x = np.array(sorted(x))
                yvals = np.array(p1(x))
                vvals = (yvals[1::] - yvals[0:-1]) / (x[1::] - x[0:-1])
                plot_x = []
                plot_y = []
                plot_v = []
                for i, v in enumerate(vvals):
                    if v_min <= v <= v_max:
                        plot_x.append(x[i])
                        plot_y.append(yvals[i])
                        plot_v.append(v)
                    elif v > v_max:
                        plot_x.append(x[i])
                        plot_y.append(yvals[i])
                        plot_v.append(v_max)
                    elif v < v_min:
                        plot_x.append(x[i])
                        plot_y.append(yvals[i])
                        plot_v.append(v_min)

                colors = color_map(plot_v, cmap, v_min, v_max)
                # ada_err.extend(cal_err((plot_x, plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                # poly_ada_err.extend(cal_err((plot_x, plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                # two_fold_ada_err.extend(cal_err((plot_x, plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                # plt.annotate(str(i), xy=(t[5], y[5]))
                # plt.annotate(str(a), xy=(t[5] + 0.1, y[5]))
                ps = np.stack((plot_x, plot_y), axis=1)
                segments = np.stack((ps[:-1], ps[1:]), axis=1)
                line_segments = LineCollection(segments, linestyles='solid', colors="black") # colors, cmap=cmap, linewidth=5)
                ax.add_collection(line_segments)
            else:
                # t = segs[0]
                # y = segs[1]
                # plt.plot(t, y, color="red")
                # for i, seg in enumerate(segs):
                    # a = seg[0]
                    # b = seg[1]
                # print(segs[0], segs[1], segs[2])
                t = np.array(segs[0])
                y = np.array(segs[1]) / 0.3048
                v = np.array(segs[2]) / 0.3048
                plot_x = []
                plot_y = []
                plot_v = []
                for i in range(len(v)):
                    if v_min <= v[i] <= v_max:
                        plot_x.append(t[i])
                        plot_y.append(y[i])
                        plot_v.append(v[i])
                    elif v[i] > v_max:
                        plot_x.append(t[i])
                        plot_y.append(y[i])
                        plot_v.append(v_max)
                    elif v[i] < v_min:
                        plot_x.append(t[i])
                        plot_y.append(y[i])
                        plot_v.append(v_min)

                poly_traj = new_poly_recon_traj[v_id]
                poly_t = np.array(poly_traj[0])
                poly_y = np.array(poly_traj[1]) / 0.3048
                poly_v = np.array(poly_traj[2]) / 0.3048
                poly_plot_x = []
                poly_plot_y = []
                poly_plot_v = []
                for i in range(len(poly_v)):
                    if v_min <= poly_v[i] <= v_max:
                        poly_plot_x.append(poly_t[i])
                        poly_plot_y.append(poly_y[i])
                        poly_plot_v.append(poly_v[i])
                    elif v[i] > v_max:
                        poly_plot_x.append(poly_t[i])
                        poly_plot_y.append(poly_y[i])
                        poly_plot_v.append(v_max)
                    elif v[i] < v_min:
                        poly_plot_x.append(poly_t[i])
                        poly_plot_y.append(poly_y[i])
                        poly_plot_v.append(v_min)

                smoothed_traj = new_smoothed_recon_traj[v_id]
                smoothed_x = np.array(smoothed_traj[0])
                smoothed_y = np.array(smoothed_traj[1]) / 0.3048
                smoothed_v = np.array(smoothed_traj[2]) / 0.3048

                """
                two_fold_plot_x = []
                two_fold_plot_y = []
                two_fold_plot_v = []
                for two_fold_traj in new_two_fold_traj[v_id]:
                    two_fold_t = np.array(two_fold_traj[0])
                    two_fold_y = np.array(two_fold_traj[1]) / 0.3048
                    two_fold_v = np.array(two_fold_traj[2]) / 0.3048
                    this_two_fold_plot_x = []
                    this_two_fold_plot_y = []
                    this_two_fold_plot_v = []
                    for i in range(len(two_fold_v)):
                        if v_min <= two_fold_v[i] <= v_max:
                            this_two_fold_plot_x.append(two_fold_t[i])
                            this_two_fold_plot_y.append(two_fold_y[i])
                            this_two_fold_plot_v.append(two_fold_v[i])
                        elif v[i] > v_max:
                            this_two_fold_plot_x.append(two_fold_t[i])
                            this_two_fold_plot_y.append(two_fold_y[i])
                            this_two_fold_plot_v.append(v_max)
                        elif v[i] < v_min:
                            this_two_fold_plot_x.append(two_fold_t[i])
                            this_two_fold_plot_y.append(two_fold_y[i])
                            this_two_fold_plot_v.append(v_min)
                    two_fold_plot_x.append(this_two_fold_plot_x)
                    two_fold_plot_y.append(this_two_fold_plot_y)
                    two_fold_plot_v.append(this_two_fold_plot_v)
                # print("!!!!!!!!", np.array(two_fold_plot_x).shape)
                """

                # colors = color_map([a] * 10, cmap, v_min, v_max)
                colors = color_map(plot_v, cmap, v_min, v_max)
                # t = np.linspace(seg[2], seg[2] + seg[3], 10)
                # y = a * t + b
                ps = np.stack((plot_x, plot_y), axis=1)
                segments = np.stack((ps[:-1], ps[1:]), axis=1)
                line_segments = LineCollection(segments, colors=colors, linestyles='solid', cmap=cmap)
                ax.add_collection(line_segments)
                # print(len(t), len(y))
                cut_gt_traj = get_same_part_of_gt_traj_as_recon(gt_traj, plot_y[-1], plot_y[0])  # in feet
                # print("1", cut_gt_traj[0][2] * 0.3048)
                gt_res = simple_run(cut_gt_traj[:, 2] * 0.3048, cut_gt_traj[:, 0] / 10, cut_gt_traj[:, 1] * 0.3048, car_id)  # in meter
                # print("gt_res", gt_res["accel"])
                # print(len(plot_x), len(smoothed_x), len(poly_plot_x))
                # exit()

                ada_err.extend(cal_err((plot_x, plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                ada_s_err.extend(cal_s_err((plot_x, plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                smoothed_err.extend(cal_err((smoothed_x, smoothed_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                smoothed_s_err.extend(cal_s_err((smoothed_x, smoothed_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                poly_ada_err.extend(cal_err((poly_plot_x, poly_plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                poly_ada_s_err.extend(cal_s_err((poly_plot_x, poly_plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                ada_v_err.extend(cal_v_err((plot_v, plot_y), (gt_traj[:, 2], gt_traj[:, 1])))
                smoothed_v_err.extend(cal_v_err((smoothed_v, smoothed_y), (gt_traj[:, 2], gt_traj[:, 1])))
                poly_ada_v_err.extend(cal_v_err((poly_plot_v, poly_plot_y), (gt_traj[:, 2], gt_traj[:, 1])))
                # plt.annotate(str(i + 1), xy=(t[5], y[5]))
                # plt.annotate(str(a), xy=(t[5] + 0.1, y[5]))
                print("ada")
                fc_err, co2_err = cal_fc_co2_err(segs[-1], gt_res)  # in meter
                ada_fc_err.append(fc_err)
                ada_co2_err.append(co2_err)

                poly_plot_x = np.array(poly_plot_x)
                poly_plot_y = np.array(poly_plot_y)
                poly_plot_v = np.array(poly_plot_v)
                cut_gt_traj = get_same_part_of_gt_traj_as_recon(gt_traj, poly_plot_y[-1], poly_plot_y[0])  # in feet
                # print("2", cut_gt_traj[0][2] * 0.3048)
                gt_res = simple_run(cut_gt_traj[:, 2] * 0.3048, cut_gt_traj[:, 0] / 10, cut_gt_traj[:, 1] * 0.3048, car_id)  # in meter
                # print("3", poly_plot_v[0] * 0.3048)
                poly_res = simple_run(poly_plot_v * 0.3048, poly_plot_x, poly_plot_y * 0.3048, car_id)
                print("poly")
                fc_err, co2_err = cal_fc_co2_err(poly_res, gt_res)  # in meter
                poly_ada_fc_err.append(fc_err)
                poly_ada_co2_err.append(co2_err)

                smoothed_x = np.array(smoothed_x)
                smoothed_y = np.array(smoothed_y)
                smoothed_v = np.array(smoothed_v)
                cut_gt_traj = get_same_part_of_gt_traj_as_recon(gt_traj, smoothed_y[-1], smoothed_y[0])  # in feet
                # print("2", cut_gt_traj[0][2] * 0.3048)
                gt_res = simple_run(cut_gt_traj[:, 2] * 0.3048, cut_gt_traj[:, 0] / 10, cut_gt_traj[:, 1] * 0.3048,
                                    car_id)  # in meter
                # print("3", poly_plot_v[0] * 0.3048)
                smoothed_res = simple_run(smoothed_v * 0.3048, smoothed_x, smoothed_y * 0.3048, car_id)
                print("smoothed")
                fc_err, co2_err = cal_fc_co2_err(smoothed_res, gt_res)  # in meter
                smoothed_fc_err.append(fc_err)
                smoothed_co2_err.append(co2_err)

                """
                best_fc_err = 999999
                best_co2_err = 999999
                for c in range(3):
                    this_two_fold_plot_x = np.array(two_fold_plot_x[c])
                    this_two_fold_plot_y = np.array(two_fold_plot_y[c]) * 0.3048
                    this_two_fold_plot_v = np.array(two_fold_plot_v[c]) * 0.3048
                    cut_gt_traj = get_same_part_of_gt_traj_as_recon(gt_traj, this_two_fold_plot_y[-1]/0.3048, this_two_fold_plot_y[0]/0.3048)  # in feet
                    # print("4", cut_gt_traj[0][2] * 0.3048)
                    gt_res = simple_run(cut_gt_traj[:, 2] * 0.3048, cut_gt_traj[:, 0] / 10,
                                        cut_gt_traj[:, 1] * 0.3048, car_id)  # in meter
                    print("two_fold", c)
                    res = simple_run(this_two_fold_plot_v, this_two_fold_plot_x, this_two_fold_plot_y, car_id)
                    fc_err, co2_err = cal_fc_co2_err(res, gt_res)  # in meter
                    if fc_err < best_fc_err:
                        best_fc_err = fc_err
                        best_co2_err = co2_err
                        best_two_fold_plot_x = this_two_fold_plot_x
                        best_two_fold_plot_y = this_two_fold_plot_y / 0.3048
                        best_two_fold_plot_v = this_two_fold_plot_v / 0.3048
                two_fold_ada_fc_err.append(best_fc_err)
                two_fold_ada_co2_err.append(best_co2_err)
                two_fold_ada_err.extend(cal_err((best_two_fold_plot_x, best_two_fold_plot_y), (gt_traj[:, 0] / 10, gt_traj[:, 1])))
                two_fold_ada_v_err.extend(
                    cal_v_err((best_two_fold_plot_v, best_two_fold_plot_y), (gt_traj[:, 2], gt_traj[:, 1])))
                """

                # plt.figure()
                # plt.plot(gt_traj[:, 0]/10, gt_traj[:, 2], color="blue")
                # plt.plot(plot_x, plot_v, color="green")
                # plt.plot(poly_plot_x, poly_plot_v, color="red")
                # plt.plot(fix_plot_t, fix_plot_v, color="orange")
                # plt.show()

                # print(v_ids[start_idx], stop_veh, np.mean(fix_err), np.mean(ada_err), np.mean(poly_ada_err), np.mean(two_fold_ada_err))
                # print(ada_fc_err, poly_ada_fc_err, two_fold_ada_fc_err)
                # print(ada_co2_err, poly_ada_co2_err, two_fold_ada_co2_err)
                # exit()
            if n == stop_veh:
                break
        if np.mean(fix_fc_err) == 0:
            break
        if len(fix_err) > 1:
            print(v_ids[start_idx], stop_veh)
            print(np.mean(fix_err), np.mean(ada_err), np.mean(smoothed_err), np.mean(poly_ada_err))
            print(np.mean(fix_s_err), np.mean(ada_s_err), np.mean(smoothed_s_err), np.mean(poly_ada_s_err))
            print(np.mean(fix_v_err), np.mean(ada_v_err), np.mean(smoothed_v_err), np.mean(poly_ada_v_err))
            print(np.mean(fix_fc_err), np.mean(ada_fc_err), np.mean(smoothed_fc_err), np.mean(poly_ada_fc_err))
            print(np.mean(fix_co2_err), np.mean(ada_co2_err), np.mean(smoothed_co2_err), np.mean(poly_ada_co2_err))
            this_save_ada_err_for_plot.extend(ada_err)
            this_save_smoothed_err_for_plot.extend(smoothed_err)
            this_save_poly_err_for_plot.extend(poly_ada_err)
            this_save_fix_err_for_plot.extend(fix_err)
            this_save_two_fold_err_for_plot.extend(two_fold_ada_err)

            this_save_ada_s_err_for_plot.extend(ada_s_err)
            this_save_smoothed_s_err_for_plot.extend(smoothed_s_err)
            this_save_poly_s_err_for_plot.extend(poly_ada_s_err)
            this_save_fix_s_err_for_plot.extend(fix_s_err)

            this_save_ada_v_err_for_plot.extend(ada_v_err)
            this_save_smoothed_v_err_for_plot.extend(smoothed_v_err)
            this_save_poly_v_err_for_plot.extend(poly_ada_v_err)
            this_save_fix_v_err_for_plot.extend(fix_v_err)
            this_save_two_fold_v_err_for_plot.extend(two_fold_ada_v_err)

            this_save_ada_fc_err_for_plot.extend(ada_fc_err)
            this_save_smoothed_fc_err_for_plot.extend(smoothed_fc_err)
            this_save_poly_fc_err_for_plot.extend(poly_ada_fc_err)
            this_save_fix_fc_err_for_plot.extend(fix_fc_err)
            this_save_two_fold_fc_err_for_plot.extend(two_fold_ada_fc_err)

            this_save_ada_co2_err_for_plot.extend(ada_co2_err)
            this_save_smoothed_co2_err_for_plot.extend(smoothed_co2_err)
            this_save_poly_co2_err_for_plot.extend(poly_ada_co2_err)
            this_save_fix_co2_err_for_plot.extend(fix_co2_err)
            this_save_two_fold_fc_err_for_plot.extend(two_fold_ada_co2_err)
            all_err.append([v_ids[start_idx], np.mean(fix_err), np.mean(ada_err), np.mean(smoothed_err), np.mean(poly_ada_err),
                            np.mean(fix_s_err), np.mean(ada_s_err), np.mean(smoothed_s_err), np.mean(poly_ada_s_err),
                            np.mean(fix_fc_err), np.mean(fix_co2_err), np.mean(ada_fc_err), np.mean(ada_co2_err),
                            np.mean(smoothed_fc_err), np.mean(smoothed_co2_err), np.mean(poly_ada_fc_err), np.mean(poly_ada_co2_err)])
            print(np.array(all_err).shape)
            print("-----------------------------------------------------------------")
    if np.mean(fix_fc_err) == 0:
        continue
    ax.set_xlim(t_min - 10, t_max + 10)
    ax.set_ylim(120, 2000)
    # plt.title("reconstructed by partial ada w coifman")
    plt.savefig("ada_err_penetration_rate_5_for_violin_plot/"+str(int(step))+".png")
    all_err = np.array(all_err)
    print(np.mean(all_err, 0))
    # np.savetxt("10_trans_recon_veh.csv", all_err, delimiter=',')
    print(total_num_veh)
    f = open("poly_recon_traj_pr_5_"+str(int(step))+".pkl", "wb")
    pickle.dump(save_recon_poly_traj, f)
    f.close()

    f = open("smoothed_recon_traj_pr_5_" + str(int(step)) + ".pkl", "wb")
    pickle.dump(save_recon_smoothed_traj, f)
    f.close()

    f = open("ada_recon_traj_pr_5_"+str(int(step))+".pkl", "wb")
    pickle.dump(save_recon_traj, f)
    f.close()
    # exit()

    this_save_fix_err_for_plot = np.array(this_save_fix_err_for_plot)
    this_save_ada_err_for_plot = np.array(this_save_ada_err_for_plot)
    this_save_smoothed_err_for_plot = np.array(this_save_smoothed_err_for_plot)
    this_save_poly_err_for_plot = np.array(this_save_poly_err_for_plot)

    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/ada_"+str(int(step))+".csv", this_save_ada_err_for_plot, delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/smoothed_" + str(int(step)) + ".csv", this_save_smoothed_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/fix_" + str(int(step)) + ".csv", this_save_fix_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/poly_" + str(int(step)) + ".csv", this_save_poly_err_for_plot,
               delimiter=',')

    this_save_fix_s_err_for_plot = np.array(this_save_fix_s_err_for_plot)
    this_save_ada_s_err_for_plot = np.array(this_save_ada_s_err_for_plot)
    this_save_smoothed_s_err_for_plot = np.array(this_save_smoothed_s_err_for_plot)
    this_save_poly_s_err_for_plot = np.array(this_save_poly_s_err_for_plot)

    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/ada_s_" + str(int(step)) + ".csv", this_save_ada_s_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/smoothed_s_" + str(int(step)) + ".csv",
               this_save_smoothed_s_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/fix_s_" + str(int(step)) + ".csv", this_save_fix_s_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/poly_s_" + str(int(step)) + ".csv",
               this_save_poly_s_err_for_plot,
               delimiter=',')

    this_save_fix_v_err_for_plot = np.array(this_save_fix_v_err_for_plot)
    this_save_ada_v_err_for_plot = np.array(this_save_ada_v_err_for_plot)
    this_save_smoothed_v_err_for_plot = np.array(this_save_smoothed_v_err_for_plot)
    this_save_poly_v_err_for_plot = np.array(this_save_poly_v_err_for_plot)

    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/ada_v_" + str(int(step)) + ".csv", this_save_ada_v_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/smoothed_v_" + str(int(step)) + ".csv",
               this_save_smoothed_v_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/fix_v_" + str(int(step)) + ".csv", this_save_fix_v_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/poly_v_" + str(int(step)) + ".csv",
               this_save_poly_v_err_for_plot,
               delimiter=',')

    this_save_fix_fc_err_for_plot = np.array(this_save_fix_fc_err_for_plot)
    this_save_ada_fc_err_for_plot = np.array(this_save_ada_fc_err_for_plot)
    this_save_smoothed_fc_err_for_plot = np.array(this_save_smoothed_fc_err_for_plot)
    this_save_poly_fc_err_for_plot = np.array(this_save_poly_fc_err_for_plot)

    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/ada_fc_" + str(int(step)) + ".csv", this_save_ada_fc_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/smoothed_fc_" + str(int(step)) + ".csv",
               this_save_smoothed_fc_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/fix_fc_" + str(int(step)) + ".csv", this_save_fix_fc_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/poly_fc_" + str(int(step)) + ".csv",
               this_save_poly_fc_err_for_plot,
               delimiter=',')

    this_save_fix_co2_err_for_plot = np.array(this_save_fix_co2_err_for_plot)
    this_save_ada_co2_err_for_plot = np.array(this_save_ada_co2_err_for_plot)
    this_save_smoothed_co2_err_for_plot = np.array(this_save_smoothed_co2_err_for_plot)
    this_save_poly_co2_err_for_plot = np.array(this_save_poly_co2_err_for_plot)

    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/ada_co2_" + str(int(step)) + ".csv", this_save_ada_co2_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/smoothed_co2_" + str(int(step)) + ".csv",
               this_save_smoothed_co2_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/fix_co2_" + str(int(step)) + ".csv", this_save_fix_co2_err_for_plot,
               delimiter=',')
    np.savetxt("ada_err_penetration_rate_5_for_violin_plot/poly_co2_" + str(int(step)) + ".csv",
               this_save_poly_co2_err_for_plot,
               delimiter=',')

    save_fix_err_for_plot.append(np.mean(this_save_fix_err_for_plot))
    save_ada_err_for_plot.append(np.mean(this_save_ada_err_for_plot))
    save_smoothed_err_for_plot.append(np.mean(this_save_smoothed_err_for_plot))
    save_poly_err_for_plot.append(np.mean(this_save_poly_err_for_plot))

    save_fix_s_err_for_plot.append(np.mean(this_save_fix_s_err_for_plot))
    save_ada_s_err_for_plot.append(np.mean(this_save_ada_s_err_for_plot))
    save_smoothed_s_err_for_plot.append(np.mean(this_save_smoothed_s_err_for_plot))
    save_poly_s_err_for_plot.append(np.mean(this_save_poly_s_err_for_plot))

    save_fix_v_err_for_plot.append(np.mean(this_save_fix_v_err_for_plot))
    save_ada_v_err_for_plot.append(np.mean(this_save_ada_v_err_for_plot))
    save_smoothed_v_err_for_plot.append(np.mean(this_save_smoothed_v_err_for_plot))
    save_poly_v_err_for_plot.append(np.mean(this_save_poly_v_err_for_plot))

    save_fix_fc_err_for_plot.append(np.mean(this_save_fix_fc_err_for_plot))
    save_ada_fc_err_for_plot.append(np.mean(this_save_ada_fc_err_for_plot))
    save_smoothed_fc_err_for_plot.append(np.mean(this_save_smoothed_fc_err_for_plot))
    save_poly_fc_err_for_plot.append(np.mean(this_save_poly_fc_err_for_plot))

    save_fix_co2_err_for_plot.append(np.mean(this_save_fix_co2_err_for_plot))
    save_ada_co2_err_for_plot.append(np.mean(this_save_ada_co2_err_for_plot))
    save_smoothed_co2_err_for_plot.append(np.mean(this_save_smoothed_co2_err_for_plot))
    save_poly_co2_err_for_plot.append(np.mean(this_save_poly_co2_err_for_plot))
save_fix_err_for_plot = np.array(save_fix_err_for_plot)
save_ada_err_for_plot = np.array(save_ada_err_for_plot)
save_smoothed_err_for_plot = np.array(save_smoothed_err_for_plot)
save_poly_err_for_plot = np.array(save_poly_err_for_plot)
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_fix_err.csv", save_fix_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_ada_err.csv", save_ada_err_for_plot, delimiter=',') # best energy consumption
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_smoothed_err.csv", save_smoothed_err_for_plot, delimiter=',') # best traj accuracy
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_poly_err.csv", save_poly_err_for_plot, delimiter=',')

save_fix_s_err_for_plot = np.array(save_fix_s_err_for_plot)
save_ada_s_err_for_plot = np.array(save_ada_s_err_for_plot)
save_smoothed_s_err_for_plot = np.array(save_smoothed_s_err_for_plot)
save_poly_s_err_for_plot = np.array(save_poly_s_err_for_plot)
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_fix_s_err.csv", save_fix_s_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_ada_s_err.csv", save_ada_s_err_for_plot, delimiter=',') # best energy consumption
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_smoothed_s_err.csv", save_smoothed_s_err_for_plot, delimiter=',') # best traj accuracy
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_poly_s_err.csv", save_poly_s_err_for_plot, delimiter=',')

save_fix_v_err_for_plot = np.array(save_fix_v_err_for_plot)
save_ada_v_err_for_plot = np.array(save_ada_v_err_for_plot)
save_smoothed_v_err_for_plot = np.array(save_smoothed_v_err_for_plot)
save_poly_v_err_for_plot = np.array(save_poly_v_err_for_plot)
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_fix_v_err.csv", save_fix_v_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_ada_v_err.csv", save_ada_v_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_smoothed_v_err.csv", save_smoothed_v_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_poly_v_err.csv", save_poly_v_err_for_plot, delimiter=',')

save_fix_fc_err_for_plot = np.array(save_fix_fc_err_for_plot)
save_ada_fc_err_for_plot = np.array(save_ada_fc_err_for_plot)
save_smoothed_fc_err_for_plot = np.array(save_smoothed_fc_err_for_plot)
save_poly_fc_err_for_plot = np.array(save_poly_fc_err_for_plot)
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_fix_fc_err.csv", save_fix_fc_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_ada_fc_err.csv", save_ada_fc_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_smoothed_fc_err.csv", save_smoothed_fc_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_poly_fc_err.csv", save_poly_fc_err_for_plot, delimiter=',')

save_fix_co2_err_for_plot = np.array(save_fix_co2_err_for_plot)
save_ada_co2_err_for_plot = np.array(save_ada_co2_err_for_plot)
save_smoothed_co2_err_for_plot = np.array(save_smoothed_co2_err_for_plot)
save_poly_co2_err_for_plot = np.array(save_poly_co2_err_for_plot)
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_fix_co2_err.csv", save_fix_co2_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_ada_co2_err.csv", save_ada_co2_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_smoothed_co2_err.csv", save_smoothed_co2_err_for_plot, delimiter=',')
np.savetxt("ada_err_penetration_rate_5_for_violin_plot/all_mean_poly_co2_err.csv", save_poly_co2_err_for_plot, delimiter=',')












