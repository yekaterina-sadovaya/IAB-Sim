# IAB-Sim

### How to run the simulation

Launch **run_sim.py** to start the simulation with the configured parameters. The main module does not have any input arguments.
All system parameters are configured withing the global variables class located in **gl_vars.py**.

### Simulations with and without optimization

#### Optimization

The frame patter in this framework is assumed to be split in 4 parts:

![alt text](./docs/frame_split_configuration.png)

The duration of these intervals is optimized together with the individual UEs allocations. 
While the frame division coefficients are used explicitly in the simulator to determine
the number of UL and DL slots, the allocations for UEs are used as weights for schedulers.
These weights prioritize the transmission orders and there are two scheduling options 
available, i.e., weighted proportional fair (WPF) and weighted round robin. The last one
is called weighted fair queuing (WFQ) in the model.

Note that optimization is continuous and all results are rounded to the nearest integer
in the simulator, which causes deviations from the optimal solution. In addition, optimization
does not provide an answer on how to schedule UEs within the backhaul, it just tells
how much time should be allocated for backhaul links but it does not specify explicitly, 
which UEs should transmit over the backhaul links.

#### How to run simulations with optimal allocations

As it was mentioned, optimization is solved for the full buffer traffic. Therefore, the following parameters should be configured
in the simulator to correspond to the optimization assumptions.
First, to enable optimization, switch frame_division_policy = 'OPT'.
Then, switch traffic type to full-buffer, i.e., choose traffic_type = 'full' (this is needed due to different throughput calculations) 
and set the burst size (burst_size_bytes) to a large value, which will not be transmitted in the configured time interval.

#### Simulations without optimal allocations

When optimization is disabled, there are 2 heuristic options available for the frame allocation, i.e., '50/50' and 'PF'.
Moreover, when using non-optimal approach, the frame is assumed to be divided into 2 intervals in the following way:

![alt text](./docs/frame_split_configuration_heuristic.png)

This basically means that heuristic approaches only use the first and the last slots from the optmization.


### Description of the parameters

#### Generic simulation parameters
Random simulation seed and simulation duration are set up here. Note that the duration is in internal simulation tics. One tic corresponds to the
duration of one UL/DL interval of a frame, e.g., if there are multiple consequent slots in a frame, one tic duration equal to the number of these slots multiplied by
the duration of a single slot.

#### Deployment configuration
Here you can configure basic deployment configurations such as the cell radius, carrier frequency, bandwidth, heights of UEs and IAB nodes, etc.
All heights are assumed to be in meters while the frequency and bandwidth are in Hz. 

There are three different options available for establishing UE associations: 1 - best RSRP when UEs choose the strongest link (best_rsrp_maxmin); 
2 - minimum number of hops when UEs always try to connect to the DgNB and choose IAB nodes only if the direct link is not available (min_hops);
3 - random associations (rand).

#### Scheduling and Optimization

These parameters describe scheduling and optimization settings. 
- To enable optimization, frame_division_policy should be put to 'OPT'. See the previous section for other parameters, which should be
configured for the correct work of the optimization framework. 
- To use the average path loss (PL) values in the optimization use_average should be True, otherwise, instant PL values will be used.
- There are 2 cost functions available for the optimization framework, i.e, 'MAXMIN' or 'PF' (maxmin fairness and proportional fairness). With maxmin optimization, 
the algorithm will try to maximize the minimum achievable rate among UEs and equalize the rates for all UEs.
- There are two basic schedulers implemented: round robin and proportional fair. To use them without optimal coefficients, switch the scheduler 
parameter to 'PF' or 'RR'. To compliment the scheduler with the optimal coefficients choose between the 'WFQ' or 'WPF'.
