# %%
import pong_net_dopa
import nest
from copy import copy
import numpy as np
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": 15})
from pong_net_dopa import POLL_TIME, DOPA_ISI
from generate_gif import grayscale_to_heatmap
import matplotlib.pyplot as plt

max_runs = 4500
n_cells = 20
net = pong_net_dopa.PongNet(n_cells)

spks = []
el = []

for i in range(max_runs):
    biol_time = nest.GetKernelStatus("biological_time")

    input_cell = np.random.randint(n_cells)
    net.set_input_spiketrain(input_cell, biol_time)
    nest.Simulate(POLL_TIME)
    out_index = net.poll_network()
    biol_time = nest.GetKernelStatus("biological_time")
    

    if input_cell == out_index:
        reward_spikes = 6
    elif abs(input_cell-out_index) == 1:
        reward_spikes = 2
    elif abs(input_cell-out_index) == 2:
        reward_spikes = 1
    else:
        reward_spikes = 0
    """

    """


    dopa_spiketrain = [biol_time + 1 + x * DOPA_ISI for x in range(reward_spikes)]
    net.dopa_signal.spike_times = dopa_spiketrain
    
    #print(net.input_train)
    #print(dopa_spiketrain)
    nest.Simulate(50)
    net.reset()
    
    #net.reward_by_move(i)
    
    el.append(np.median(nest.GetConnections(net.input_neurons).get("c")))
    spks.append(len(net.dopa_signal.spike_times))
    if i % 150 == 0:
        print(i)
        hm = grayscale_to_heatmap(net.get_all_weights(), 1150, 1550, (255, 0, 0))
        plt.imshow(hm)
        plt.savefig(f"imgs/{i}.png")
plt.show()
# %%

# %%

# %%
