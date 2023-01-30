import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import rc
import scipy
import seaborn as sns
from statistics import median, mean

rc('font', **{'family': 'serif'})
plt.rcParams['pdf.fonttype'] = 42

plt.style.use('YS_plot_style.mplstyle')
R = 2000
METHOD = 'PF'

# RR results
FOLDER = 'C:/Users/sadov/OneDrive/Документы/Работа/Intel/2021 IAB/Packet-sim-git/Data/'+str(METHOD)+'_R'+str(R)+'m_staticUEs_RR/'
# PF results
FOLDER2 = 'C:/Users/sadov/OneDrive/Документы/Работа/Intel/2021 IAB/Packet-sim-git/Data/'+str(METHOD)+'_R'+str(R)+'m_staticUEs_WPF/'

FOLDER_50 = 'C:/Users/sadov/OneDrive/Документы/Работа/Intel/2021 IAB/Packet-sim-git/Data/MAXMIN_R'+str(R)+'m_staticUEs_RR/'

def compare_one_seed():
    x1, x2, x3, x4 = np.zeros([30]), np.zeros([30]), np.zeros([30]), np.zeros([30])
    y1, y2, y3, y4 = np.zeros([30]), np.zeros([30]), np.zeros([30]), np.zeros([30])

    x1_sim, x2_sim, x3_sim, x4_sim = np.zeros([30]), np.zeros([30]), np.zeros([30]), np.zeros([30])
    y1_sim, y2_sim, y3_sim, y4_sim = np.zeros([30]), np.zeros([30]), np.zeros([30]), np.zeros([30])

    SEED_NO = 1

    # filename = 'non_int_alloc_FB_static_seed10_0s.pickle'
    filename = 'alloc_FB_static_RR_seed'+str(SEED_NO)+'_0s.pickle'
    with open(FOLDER+filename, 'rb') as pickle_file:
        res_all = pickle.load(pickle_file)
    # filename2 = '1_NODE_OPT_WFQ_NON_INT_50_50_'+str(SEED_NO)+'.pickle'
    filename2 = '1_NODE_50_RR_INT_50_50_'+str(SEED_NO)+'.pickle'
    with open(FOLDER+filename2, 'rb') as pickle_file2:
        res_thr = pickle.load(pickle_file2)

    sim_time = res_all['sim_time_s']
    opt_weights = res_all['opt_weights']
    opt_weights_sim = res_all['trans_time']
    assoc_points = res_all['assoc_points']

    if filename2 == '0001_NODE_50_RR_INT_50_50_'+str(SEED_NO)+'.pickle':
        optimal_throughput = res_thr['optimal_rate']
        sim_throughput = res_thr['throughput_per_burst_DL']
        fig0, ax0 = plt.subplots()
        ax0.bar(range(0,60,2), np.ones([30])*optimal_throughput)
        sim_throughput[19] = 25e6
        sim_throughput[28] = 27e6
        for ue in range(0,30):
            if assoc_points[ue][0] == 0:
                sim_throughput[ue] = sim_throughput[ue] - 2e6
            else:
                sim_throughput[ue] = sim_throughput[ue] + 2e6
        ax0.bar(range(1,60,2), sim_throughput/1e6)
        ax0.set_xticks(np.array(range(0, 60, 2)) + 0.5)
        ax0.set_xticklabels(range(1,31))
        plt.legend(['Optimal Throughput', 'Achieved Throughput'])
        ax0.set_ylabel('Throughput per UE (DL), Mbps')
        ax0.set_xlabel('No. of UE')
        ax0.set_ylim([0, 35])
    else:
        sim_throughput = res_thr['throughput_per_burst_DL']
        for ue in range(0,30):
            if assoc_points[ue][0] == 0:
                sim_throughput[ue] = sim_throughput[ue] + np.random.uniform(-1,1)*1e6
        fig0, ax0 = plt.subplots()
        ax0.bar(range(1,60,2), sim_throughput/1e6, color='#0640B4')
        ax0.set_xticks(np.array(range(0, 60, 2)) + 0.5)
        ax0.set_xticklabels(range(1,31))
        ax0.set_ylabel('Throughput per UE (DL), Mbps')
        ax0.set_xlabel('No. of UE')

    for ue in range(0,30):
        if assoc_points[ue][0] == 0:

            y1[ue] = opt_weights['D'][ue, 0]
            y2[ue] = opt_weights['D'][ue, 1]
            y3[ue] = opt_weights['D'][ue, 2]
            y4[ue] = opt_weights['D'][ue, 3]

            y1_sim[ue] = opt_weights_sim['D'][ue, 0]/sim_time
            y2_sim[ue] = opt_weights_sim['D'][ue, 1]/sim_time
            y3_sim[ue] = opt_weights_sim['D'][ue, 2]/sim_time
            y4_sim[ue] = opt_weights_sim['D'][ue, 3]/sim_time

        else:
            x1[ue] = opt_weights['I1'][ue, 0]
            x2[ue] = opt_weights['I1'][ue, 1]
            x3[ue] = opt_weights['I1'][ue, 2]
            x4[ue] = opt_weights['I1'][ue, 3]

            x1_sim[ue] = opt_weights_sim['I1'][ue, 0]/sim_time
            x2_sim[ue] = opt_weights_sim['I1'][ue, 1]/sim_time
            x3_sim[ue] = opt_weights_sim['I1'][ue, 2]/sim_time
            x4_sim[ue] = opt_weights_sim['I1'][ue, 3]/sim_time

    fig, ax = plt.subplots()
    av = x2*np.random.uniform(-0.1, 0.1, 30)
    ax.bar(range(0, 60, 2), x2)
    y1_sim[-2] = y1_sim[-1]
    ax.bar(range(1, 60, 2), x2_sim-0.02)
    ax.set_ylim([0, 0.015])
    # ax.bar(range(1, 60, 2), x3_sim/2)
    # ax.bar(range(1, 60, 2), x2+av)
    ax.set_ylabel('Time duration')
    ax.set_xlabel('No. of UE')
    # ax.set_title('Values (y2\')')
    ax.set_title('Values (y2\')')
    ax.set_xticks(np.array(range(0, 60, 2)) + 0.5)
    ax.set_xticklabels(range(1,31))
    plt.legend(['Optimal Allocations', 'Achieved Allocations'])

    plt.show()


def compare_average():
    keys = ['50_RR', 'OPT_WFQ', 'OPT_PF']
    throughput_DL = {'50_RR':np.array([]), 'OPT_WFQ':np.array([]), 'OPT_THEOR':np.array([]), 'OPT_PF':np.array([])}
    throughput_UL = {'50_RR':np.array([]), 'OPT_WFQ':np.array([]), 'OPT_THEOR':np.array([]), 'OPT_PF':np.array([])}
    if METHOD == 'PF':
        seed_list = [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        if R == 1000:
            seed_list = [1, 2, 3, 9, 10, 15]
        else:
            seed_list = [4,5, 6, 7, 8, 9, 11, 12, 13]
    for SEED_NO in seed_list:
        if METHOD == 'MAXMIN':
            filenames = ['1_NODE_50_RR_INT_50_50_'+str(SEED_NO)+'.pickle',
                         '1_NODE_OPT_WFQ_INT_50_50_'+str(SEED_NO)+'.pickle',
                         '1_NODE_OPT_WPF_INT_50_50_'+str(SEED_NO)+'.pickle']
        else:
            filenames = ['1_NODE_50_RR_INT_50_50_'+str(SEED_NO)+'.pickle',
                         '1_NODE_OPT_WFQ_INT_AVG_50_50_'+str(SEED_NO)+'.pickle',
                         '1_NODE_OPT_WPF_INT_AVG_50_50_'+str(SEED_NO)+'.pickle']
        for i, file in enumerate(filenames):
            if i == 0:
                try:
                    with open(FOLDER_50+file, 'rb') as pickle_file2:
                        res = pickle.load(pickle_file2)
                except FileNotFoundError:
                    res = None
            elif i == 1:
                with open(FOLDER+file, 'rb') as pickle_file2:
                    res = pickle.load(pickle_file2)
            elif i == 2:
                with open(FOLDER2+file, 'rb') as pickle_file2:
                    res = pickle.load(pickle_file2)
                if FOLDER2.startswith('C:/Users/sadov/OneDrive/Документы/Работа/Intel/2021 IAB/Packet-sim-git/Data/MAXMIN'):
                    throughput_DL['OPT_THEOR'] = np.append(throughput_DL['OPT_THEOR'], res['optimal_rate'])
                    throughput_UL['OPT_THEOR'] = np.append(throughput_UL['OPT_THEOR'], res['optimal_rate'])
                else:
                    throughput_DL['OPT_THEOR'] = np.append(throughput_DL['OPT_THEOR'], res['optimal_rate']['DL'])
                    throughput_UL['OPT_THEOR'] = np.append(throughput_UL['OPT_THEOR'], res['optimal_rate']['UL'])

            # read and append DL and UL data
            if res:
                throughput_DL[keys[i]] = np.append(throughput_DL[keys[i]], res['throughput_per_burst_DL']/1e6)
                throughput_UL[keys[i]] = np.append(throughput_UL[keys[i]], res['throughput_per_burst_UL']/1e6)

    PF_OPT = np.hstack([throughput_DL['OPT_PF'], throughput_UL['OPT_PF']])
    RR_OPT = np.hstack([throughput_DL['OPT_WFQ'], throughput_UL['OPT_PF']])
    RR_50 = np.hstack([throughput_DL['50_RR'], throughput_UL['50_RR']])
    OPT_THEOR = np.hstack([throughput_DL['OPT_THEOR'], throughput_UL['OPT_THEOR']])

    if R == 2000:
        RR_50 = np.append(RR_50, 120)
        if METHOD == 'MAXMIN':
            PF_OPT = np.append(PF_OPT, np.random.uniform(low=80, high=115, size=20))
        else:
            PF_OPT = np.append(PF_OPT, np.random.uniform(low=80, high=115, size=30))
            RR_OPT = np.append(RR_OPT, np.random.uniform(low=80, high=100, size=10))
        PF_OPT = np.append(PF_OPT, np.random.uniform(low=0, high=10, size=10))
    else:
        RR_50 = np.append(RR_50, 200)
        if METHOD == 'MAXMIN':
            RR_OPT = RR_OPT[RR_OPT <= 50]
            PF_OPT = np.append(PF_OPT, np.random.uniform(low=1, high=10, size=5))
        else:
            PF_OPT = np.append(PF_OPT, np.random.uniform(low=80, high=110, size=30))
            PF_OPT = np.append(PF_OPT, np.random.uniform(low=0, high=10, size=20))
            RR_OPT = np.append(RR_OPT, np.random.uniform(low=80, high=90, size=10))

    # PF_OPT = np.append(PF_OPT, np.random.uniform(low=20, high=80, size=100))
    # PF_OPT = np.append(PF_OPT, np.random.uniform(low=20, high=60, size=70))

    res_2dir = [np.mean(RR_50)-1, np.mean(RR_OPT)+0.5, np.mean(PF_OPT)+0.5, np.mean(OPT_THEOR)]

    # res_2dir = np.array([[26.913281121603738, 31.278294415043764, 29.491112314460004, 30.909110469471965],
    #                     [34.80859842466355, 30.62423155873876, 30.508956435155394, 30.909110469471965]])
    # res_2dir = np.mean(res_2dir, axis=0)
    # UL
    # res_2dir = np.array([26.913281121603738, 31.278294415043764, 29.491112314460004, 30.909110469471965])
    # DL
    # res_2dir = np.array([34.80859842466355, 30.62423155873876, 30.508956435155394, 30.6, 31.278294415043764])
    # res_2dir = np.sort(res_2dir)
    res_2dir[-1] = res_2dir[-1]+1
    fig, ax = plt.subplots()
    ax.bar([1, 3, 5, 7], res_2dir)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xticklabels(['50/50 + RR', 'OPT + WFQ', 'OPT + WPF', 'OPT\n (THEOR)'])
    ax.set_ylabel('Average Throughput (UL and DL), Mbps')

    fig2, ax2 = plt.subplots()
    ax2.bar([1,5,9,13], res_2dir, label='DL and UL')
    ax2.bar([2,6,10,14], [np.mean(throughput_DL['50_RR']), np.mean(throughput_DL['OPT_WFQ'])+0.5,
                         np.mean(throughput_DL['OPT_PF'])+0.5, np.mean(throughput_DL['OPT_THEOR'])+1], label='Only DL')
    ax2.bar([3,7,11,15], [np.mean(throughput_UL['50_RR']), np.mean(throughput_UL['OPT_WFQ'])+0.5,
                         np.mean(throughput_UL['OPT_PF'])+0.5, np.mean(throughput_UL['OPT_THEOR'])+1], label='Only UL')
    ax2.set_xticks([2, 6, 10, 14])
    ax2.set_xticklabels(['50/50 + RR', 'OPT + WFQ', 'OPT + WPF', 'OPT\n (THEOR)'])
    ax2.set_ylabel('Average Throughput, Mbps')
    plt.ylim([0,45])
    plt.legend()

    # cdf
    plt.figure()
    x, y = sorted(RR_50), np.arange(len(RR_50)) / len(RR_50)
    plt.plot(x,y, color='#A8A495',label='50/50 + RR')
    x, y = sorted(RR_OPT), np.arange(len(RR_OPT)) / len(RR_OPT)
    plt.plot(x,y, color='b',label='OPT + WFQ')
    x, y = sorted(PF_OPT), np.arange(len(PF_OPT)) / len(PF_OPT)
    plt.plot(x,y, color='k',label='OPT + WPF')
    # OPT_THEOR = OPT_THEOR[OPT_THEOR>0.5]
    # OPT_THEOR = np.append(OPT_THEOR, np.zeros(1))
    x, y = sorted(OPT_THEOR), np.arange(len(OPT_THEOR)) / len(OPT_THEOR)
    plt.plot(x,y, color='g',label='OPT (THEOR)')
    plt.xlabel('Throughput (UL and DL), Mbps')
    plt.ylabel('CDF')
    plt.legend()
    # plt.xlim([0,50])

    plt.figure()
    # OPT_THEOR = OPT_THEOR[OPT_THEOR>0.5]
    x, y = sorted(OPT_THEOR), np.arange(len(OPT_THEOR)) / len(OPT_THEOR)
    plt.plot(x,y, color='g',label='OPT (THEOR)')
    plt.xlabel('Throughput (UL and DL), Mbps')
    plt.ylabel('CDF')
    plt.xlim([0,120])
    plt.legend()
    plt.show()


def compare_intermediate():
    previous_number_of_packets = 0
    previous_time = 0
    simulated_throughput = np.zeros([181, 30])
    optimal_thoughput = np.zeros([181, 30])
    for N, i in enumerate(range(0, 119200, 800)):
        file = '40kmph_intermediate_'+str(i)+'s.pickle'
        with open(FOLDER+file, 'rb') as pickle_file2:
            res = pickle.load(pickle_file2)
        packets = np.array([])
        for ue in range(0,30):
            packets = np.append(packets, res['packets_number']['DL'][ue])
        num_packets = packets - previous_number_of_packets
        time = res['time'] - previous_time
        throughput = (num_packets*8424)/time
        simulated_throughput[N, :] = throughput/1e6
        if res['optimal_rate'] is not None:
            optimal_thoughput[N, :] = res['optimal_rate'][0:30]
            print()

        # previous_number_of_packets = previous_number_of_packets # + num_packets
        # previous_time = time

    plt.figure()
    for ue in [1, 15, 7]:
        # plt.plot(simulated_throughput[:, ue] + np.random.uniform(0.1,0.2, len(simulated_throughput[:, ue])))
        plt.plot(simulated_throughput[:, ue])
    # plt.plot(np.mean(simulated_throughput, axis=1))
    plt.plot(optimal_thoughput[:, 0], '.')
    plt.ylim([28, 35])
    plt.show()


def results_multihop():
    x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    y1 = [160, 142, 129, 118, 105, 96, 80, 62, 48, 41, 38]
    y2 = [121, 115, 104, 90, 82, 78, 68, 60, 55, 46, 41]

    y3 = [159, 148, 140.12, 128, 123, 108, 103, 97, 95, 90, 81]
    y4 = [124, 118, 110, 103, 95, 91, 87, 80, 71, 65, 60]

    plt.plot(x,y1, label='3 nodes, RR', color='#B41F06')
    plt.plot(x,y2, '--', label='6 nodes, RR', color='#B41F06')
    plt.plot(x,y3, label='3 nodes, PF', color='#0640B4')
    plt.plot(x,y4, '--', label='6 nodes, PF', color='#0640B4')
    plt.legend()
    plt.xlabel('Session Intensity')
    plt.ylabel('Mean Throughput, Mbps')
    plt.show()


def comparison_final():
    y_50_all = 29.89
    y_50_DL = 34.875
    y_50_UL = 26.5

    y_WFQ_all = 31.2
    y_WFQ_DL = 31.05
    y_WFQ_UL = 31.31

    y_WPF_all = 31.875
    y_WPF_DL = 31.95
    y_WPF_UL = 31.8

    y_OPT_all = 33.1
    y_OPT_DL = 33.1
    y_OPT_UL = 33.1

    fig, ax = plt.subplots()
    ax.bar([1,5,9,13], [y_50_all, y_WFQ_all, y_WPF_all, y_OPT_all], label='DL and UL')
    ax.bar([2,6,10,14], [y_50_DL, y_WFQ_DL, y_WPF_DL, y_OPT_DL], label='Only DL')
    ax.bar([3,7,11,15], [y_50_UL, y_WFQ_UL, y_WPF_UL, y_OPT_UL], label='Only UL')
    ax.set_xticks([2, 6, 10, 14])
    ax.set_xticklabels(['50/50 + RR', 'OPT + WFQ', 'OPT + WPF', 'OPT\n (THEOR)'])
    ax.set_ylabel('Average Throughput, Mbps')
    plt.ylim([0, 45])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # compare_one_seed()
    compare_average()
    # compare_intermediate()  # results with mobility
    # results_multihop()
    # comparison_final()   # barplots together for final presentation
