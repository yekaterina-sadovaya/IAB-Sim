import multiprocessing
from joblib import Parallel, delayed
from main_runner import packet_sim


num_cores = multiprocessing.cpu_count()
# inputs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
# seed_list = [10]

inputs = [50]
# seed_list = range(0, 10)
seed_list = [1, 2, 3]  # 8,9,10,11,12,13
# seed_list = [3]

if __name__ == "__main__":
    for seed in seed_list:
        processed_list = Parallel(n_jobs=num_cores)(delayed(packet_sim)(i, seed) for i in inputs)


