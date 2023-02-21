# IAB-Sim

### How to run the simulation

Launch **run_sim.py** to start the simulation with the configured parameters. The main module does not have any input arguments.
All system parameters are configured withing the global variables class located in **gl_vars.py**.

### Simulations with and without optimization

#### Optimization

The frame patter in this framework is assumed to be split in 4 parts:

![alt text](./docs/configurations_opt.png)

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