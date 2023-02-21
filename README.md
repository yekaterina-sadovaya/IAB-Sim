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