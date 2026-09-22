import numpy as np
import pandas as pd
from fractal_analysis.simulator.wood_chan.wood_chan_multi_fractal_simulator import WoodChanMbmSimulator, WoodChanFbmSimulator
import matplotlib.pyplot as plt
from DFA import detrended_fluctuation_analysis
from break_points import break_point_detect

# Simutions 
n = 6000
width = 400
step = 200

trad, cus_h, cus_s, pelt_h, pelt_s = [], [], [], [], []
cus_h_S1, cus_s_S1, pelt_h_S1, pelt_s_S1 = [], [], [], []
for i in range(100):
    segs = []
    # b1, b2 = n // 3, 2 * n // 3
    # seg_bounds = [0, b1, b2, n]
    b = n // 2
    seg_bounds = [0, b, n]
    h_list = []
    for (a, b), h in zip(zip(seg_bounds[:-1], seg_bounds[1:]), [0.3, 0.7]):
        sim = WoodChanFbmSimulator(sample_size=b - a + 1, hurst_parameter=h)
        fbm = sim.get_fbm(seed=a + 10*i)
        h_list+= list(np.ones(len(fbm))*h)
        inc = np.diff(fbm)
        segs.append((inc - inc.mean()) / inc.std())
    h_list= np.array(h_list).reshape(-1,)
    fbm_path_fgn = np.concatenate(segs)
    x = fbm_path_fgn
    centers = np.arange(width // 2, len(x) - width // 2, step)
    q_list = [2]
    ff = break_point_detect(x, 300, 100)
    b3 = ff.det_cusum()
    b4 = ff.pelt_segmentation()

    a1 = detrended_fluctuation_analysis(x, q_list, centers, width)
    a1.fixed_DFA()
    trad.append(np.sqrt(np.sum((a1.Hq_list.reshape(-1,) - h_list[centers])**2)/len(centers)))
    a2 = detrended_fluctuation_analysis(x, q_list, centers, width)
    a2.hard_split_DFA(b3)
    cus_h.append(np.sqrt(np.sum((a2.Hq_list.reshape(-1,) - h_list[centers])**2)/len(centers)))
    cus_h_S1.append(a2.Hq_list.reshape(-1,))
    a2.soft_split_DFA(b3)
    cus_s.append(np.sqrt(np.sum((a2.Hq_list.reshape(-1,) - h_list[centers])**2)/len(centers)))
    cus_s_S1.append(a2.Hq_list.reshape(-1,))

    a3 = detrended_fluctuation_analysis(x, q_list, centers, width)
    a3.hard_split_DFA(b4)
    pelt_h.append(np.sqrt(np.sum((a3.Hq_list.reshape(-1,) - h_list[centers])**2)/len(centers)))
    pelt_h_S1.append(a3.Hq_list.reshape(-1,))
    a3.soft_split_DFA(b4)
    pelt_s.append(np.sqrt(np.sum((a3.Hq_list.reshape(-1,) - h_list[centers])**2)/len(centers)))
    pelt_s_S1.append(a3.Hq_list.reshape(-1,))
print(np.mean(trad))
print(np.mean(cus_h))
print(np.mean(cus_s))
print(np.mean(pelt_h))
print(np.mean(pelt_s))

cus_h_S1 = np.array(cus_h_S1)
cus_s_S1 = np.array(cus_s_S1)

pelt_h_S1 = np.array(pelt_h_S1)
pelt_s_S1 = np.array(pelt_s_S1)

cus_h_S1_low = np.quantile(cus_h_S1, 0.20, axis=0)
cus_h_S1_high = np.quantile(cus_h_S1, 0.80, axis=0)

cus_s_S1_low = np.quantile(cus_s_S1, 0.20, axis=0)
cus_s_S1_high = np.quantile(cus_s_S1, 0.80, axis=0)

pelt_h_S1_low = np.quantile(pelt_h_S1, 0.20, axis=0)
pelt_h_S1_high = np.quantile(pelt_h_S1, 0.80, axis=0)

pelt_s_S1_low = np.quantile(pelt_s_S1, 0.20, axis=0)
pelt_s_S1_high = np.quantile(pelt_s_S1, 0.80, axis=0)

time_t = np.linspace(0,1,n)
time_t = time_t[centers]

fig, ax = plt.subplots(2,2,figsize=(10, 7))


ax[0,0].plot(time_t, h_list[centers],'--',label='True')
ax[0,0].plot(time_t, np.mean(cus_h_S1,axis=0),'.-', color="#1f77b4", label='CUSUM-hard')
ax[0,0].fill_between(time_t, cus_h_S1_low, cus_h_S1_high, color="#1f77b4", alpha=0.15)
ax[0,0].grid()
ax[0,0].set_xticklabels([])
ax[0,0].set_ylabel('Hurst exponent H', fontsize=13)

ax[0,1].plot(time_t, h_list[centers],'--')
ax[0,1].plot(time_t, np.mean(cus_s_S1,axis=0),'.-', label='CUSUM-soft', color="#08a808")
ax[0,1].fill_between(time_t, cus_s_S1_low, cus_s_S1_high, color='#2ca02c', alpha=0.15)
ax[0,1].grid()
ax[0,1].set_xticklabels([])
ax[0,1].set_yticklabels([])

ax[1,0].plot(time_t, h_list[centers],'--')
ax[1,0].plot(time_t, np.mean(pelt_h_S1,axis=0),'.-', label='PELT-hard', color="#4d4a4a")
ax[1,0].fill_between(time_t, pelt_h_S1_low, pelt_h_S1_high, color="#4d4a4a", alpha=0.15)
ax[1,0].grid()
ax[1,0].set_xlabel('Time t', fontsize=13)
ax[1,0].set_ylabel('Hurst exponent H', fontsize=13)


ax[1,1].plot(time_t, h_list[centers],'--')
ax[1,1].plot(time_t, np.mean(pelt_s_S1,axis=0),'.-', label='PELT-soft', color="#f32929")
ax[1,1].fill_between(time_t, pelt_s_S1_low, pelt_s_S1_high, color="#f32929", alpha=0.15)
ax[1,1].grid()
ax[1,1].set_xlabel('Time t', fontsize=13)
ax[1,1].set_yticklabels([])
fig.legend(
    loc='upper center',
    ncol=5,
    bbox_to_anchor=(0.5, 1.04), fontsize=12
)
plt.tight_layout()
plt.savefig('S1_1.jpg', dpi=300, bbox_inches='tight')

# SP500 real-world application
data = pd.read_csv('sp500_extended_close.csv')
closed_p = data['close']
close_diff = np.diff(closed_p)
width = 750
x = (close_diff - close_diff.mean()) / close_diff.std()
centers = np.arange(width // 2, len(x) - width // 2, 30)

ff = break_point_detect(x, 750, 30)
b3 = ff.det_cusum()
b4 = ff.pelt_segmentation()

print(b3)
print(b4)
break_points = [int((1245+1275)/2), int((1785+1815)/2), int((2325+2355)/2), 
 int((3015+3045)/2), int((3615+3645)/2), int((4755+4845)/2), int((5685+5685)/2), 
 int((6255+6285)/2)]

data = pd.read_csv('sp500_extended_close.csv')
closed_p = data['close']
close_diff = np.diff(closed_p)
width = 400
centers = np.arange(width // 2, len(close_diff) - width // 2, 30)
a3 = detrended_fluctuation_analysis(close_diff, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], centers, width)
a3.soft_split_DFA(break_points)
# a3.fixed_DFA()
soft_res = a3.Hq_list[:,1]
a3.tau_q()
soft_res_alp = a3.alpha_list

data = pd.read_csv('sp500_extended_close.csv')
closed_p = data['close']
close_diff = np.diff(closed_p)
width = 400
centers = np.arange(width // 2, len(close_diff) - width // 2, 30)
a3 = detrended_fluctuation_analysis(close_diff, [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], centers, width)
a3.hard_split_DFA(break_points)
# a3.fixed_DFA()
hard_res = a3.Hq_list[:,1]
a3.tau_q()
hard_res_alp = a3.alpha_list

def hard_soft_res(break_pints, centers):
    out = []
    for i in break_pints:
        for j,c in enumerate(centers):
            if i < c:
                out.append(j)
                break
    return out

h_s_pos = hard_soft_res(break_points, centers)
time_H = pd.to_datetime(data['Date'])
time_H = time_H[centers]

print(np.mean(np.max(hard_res_alp[:h_s_pos[0]], axis=1) - np.min(hard_res_alp[:h_s_pos[0]], axis=1)))
print(np.mean(np.max(soft_res_alp[:h_s_pos[0]], axis=1) - np.min(soft_res_alp[:h_s_pos[0]], axis=1)))

fig, ax = plt.subplots(1,2,figsize=(8, 3.7))
ax[0].plot(time_H, hard_res,'-o', color="#1f77b4", markersize=3)
ax[0].grid()
# ax[0].set_xticklabels([])
ax[0].set_ylabel('H(q)', fontsize=13)
ax[0].set_title('Hard split', fontsize=15)

ax[1].plot(time_H, soft_res,'-o', color='#2ca02c', markersize=3)
ax[1].grid()
# ax[1].set_xticklabels([])
ax[1].set_title('Soft split', fontsize=15)

plt.tight_layout()
plt.savefig('sp500_H_S.jpg', dpi=300, bbox_inches='tight')
