# %%
import nest
from copy import copy
import numpy as np
nest.set_verbosity("M_WARNING")
nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": 15})
from networks import POLL_TIME, PongNetDopa
from generate_gif import grayscale_to_heatmap
import matplotlib.pyplot as plt


#%%

max_runs = 10000
n_cells = 20
net = PongNetDopa(n_cells)

spks = []
el = []

biol_time = 0
for i in range(max_runs):

    input_cell = np.random.randint(n_cells)
    net.set_input_spiketrain(input_cell, biol_time)
    nest.Simulate(POLL_TIME)
    biol_time = nest.GetKernelStatus("biological_time")
    rates = net.get_spike_counts()

    target_rate = rates[input_cell]
    mean_rate = max(sum(rates), 1)

    reward_spikes = int((target_rate/mean_rate)* 30)
    spks.append(reward_spikes)
    if reward_spikes >= 6:
        print(reward_spikes, rates)

    #dopa_spiketrain = [biol_time + 1 + x * DOPA_ISI for x in range(reward_spikes)]
    #net.dopa_signal.spike_times = dopa_spiketrain
    net.apply_synaptic_plasticity(biol_time)

    net.reset()

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
