# IAB-Sim

### How to run the simulation

Launch **main_runner.py** to start the simulation with the configured parameters. The main module does not have any input arguments.
All system parameters are configured withing the global variables class located in **gl_vars.py**.

### Description of the parameters

#### Generic simulation parameters
Random simulation seed and simulation duration are set up here. Note that the duration is in internal simulation tics. One tic corresponds to the
duration of one UL/DL interval of a frame, e.g., if there are multiple consequent slots in a frame, one tic duration equal to the number of these slots multiplied by
the duration of a single slot.