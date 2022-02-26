import nest
from copy import copy
import logging
import numpy as np


#: int: Amount of time the network is simulated in milliseconds.
POLL_TIME = 200
#: int: Number of spikes for input spiketrain.
NO_SPIKES = 20
#: float: Inter-Spike Interval (ISI) of input spiketrain.
ISI = POLL_TIME/NO_SPIKES
#: float: Standard deviation of Gaussian current noise in picoampere.
BG_STD = 200.
#: float: Initial mean weight when applying noise to motor neurons.
MEAN_WEIGHT = 1300.
#: float: Learning rate to use in weight updates.
LEARNING_RATE = 0.7
#: dict: reward to be applied to synapses in dependence on the distance between target and prediction.
REWARDS_DICT = {0: 1., 1: 0.7, 2: 0.4, 3: 0.1}
#: float: maximum rate for the poisson_generator representing dopamine concentration
MAX_DOPA_RATE = 100
DOPA_ISI = 3.


class PongNet(object):
    def __init__(self, num_neurons=20, with_noise=True):
        """A NEST based spiking neural network that learns through reward-based STDP

        Args:
            num_neurons (int, optional): number of neurons to simulate. Changes here need to be matched in the game simulation in pong.py. Defaults to 20.
            with_noise (bool, optional): If true, noise generators are connected to the motor neurons of the network. Defaults to True.
        """
        self.num_neurons = num_neurons

        self.weight_history = []
        self.mean_reward = np.array([0. for _ in range(self.num_neurons)])
        self.mean_reward_history = []

        self.input_generators = nest.Create(
            "spike_generator", self.num_neurons)
        self.input_neurons = nest.Create("parrot_neuron", self.num_neurons)
        nest.Connect(self.input_generators, self.input_neurons,
                     {'rule': 'one_to_one'})

        self.motor_neurons = nest.Create("iaf_psc_exp", self.num_neurons)

        self.dopa_signal = nest.Create("spike_generator")
        self.dopa_parrot = nest.Create("parrot_neuron")
        nest.Connect(self.dopa_signal, self.dopa_parrot)
        self.dopa_noise = nest.Create("poisson_generator", {"rate": 5.})
        #
        #nest.Connect(self.dopa_noise, self.dopa_parrot)
        self.vt = nest.Create("volume_transmitter")
        nest.Connect(self.dopa_parrot, self.vt)
        # nest.SetDefaults("stdp_dopamine_synapse", {"vt": self.vt.get("global_id")})
        nest.SetDefaults("stdp_dopamine_synapse", {"vt": self.vt.get("global_id"), "tau_c": 80, "tau_n": 40, "tau_plus": 45., "Wmin": 1150, "Wmax": 1550, "b": 0.01})
        # nest.SetDefaults("stdp_dopamine_synapse", {"vt": self.vt.get("global_id"), "Wmin": 1000, "Wmax": 1500})

        # Defaults: {'A_minus': 1.5, 'A_plus': 1.0, 'b': 0.0, 'c': 0.0, 'delay': 1.0, 'has_delay': True, 'n': 0.0, 'num_connections': 0, 'receptor_type': 0, 
        # 'requires_symmetric': False, 'synapse_model': 'stdp_dopamine_synapse', 'tau_c': 1000.0, 'tau_n': 200.0, 'tau_plus': 20.0, 'vt': -1, 
        # 'Wmax': 200.0, 'Wmin': 0.0, 'weight': 1.0, 'weight_recorder': ()}


        nest.Connect(self.input_neurons, self.motor_neurons, {'rule': 'all_to_all'}, {
                     "synapse_model": "stdp_dopamine_synapse", "weight": nest.random.normal(MEAN_WEIGHT, 5.)})

        self.spike_recorder = nest.Create("spike_recorder", self.num_neurons)
        nest.Connect(self.motor_neurons, self.spike_recorder,
                     {'rule': 'one_to_one'})

        self.background_generator = nest.Create("noise_generator", self.num_neurons,
                                                params={"std": BG_STD})
        nest.Connect(self.background_generator,
                     self.motor_neurons, {'rule': 'one_to_one'})

    def get_all_weights(self):
        """extract all weights between input and motor neurons from the network

        Returns:
            numpy.array: 2D array, first axis represents the input neuron, second axis 
            represents the targeted motor neuron. Note that the indices start at 0, 
            and not with the NEST internal neuron ID
        """
        x_offset = self.input_neurons[0].get("global_id")
        y_offset = self.motor_neurons[0].get("global_id")
        out = np.zeros((self.num_neurons, self.num_neurons))
        conns = nest.GetConnections(self.input_neurons)
        for conn in conns:
            source, target, weight = conn.get(
                ["source", "target", "weight"]).values()
            out[source-x_offset, target-y_offset] = weight

        return out

    def set_all_weights(self, weights):
        """set weights between input and motor neurons of the network

        Args:
            weights (numpy.array): 2D array, first axis representing the input neuron number, 
            and second axis representing the target motor neuron. See get_all_weights().
        """
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                connection = nest.GetConnections(
                    self.input_neurons[i], self.motor_neurons[j])
                connection.set({"weight": weights[i, j]})

    def get_rates(self):
        """get the neuronal firin rates of the motor neurons from the spike_recorders

        Returns:
            numpy.array: array of spike frequencies from the current simulation
        """
        events = self.spike_recorder.get("n_events")
        return np.array(events)

    def reset(self):
        """reset the network for a new iteration by clearing the spike recorders
        """
        self.spike_recorder.set({"n_events": 0})

    def poll_network(self):
        """Get the grid cell the network wants to move to. Find this cell by finding 
        the winning (highest rate) motor neuron.
        """
        rates = self.get_rates()
        logging.debug(f"Got rates: {rates}")
        # If multiple neurons have the same activation, one is chosen on random
        self.winning_neuron = int(np.random.choice(
            np.flatnonzero(rates == rates.max())))
        return self.winning_neuron

    def reward_by_move(self, run):
        """ Reward network based on how close winning neuron and desired output are.
        """
        distance = np.abs(self.winning_neuron - self.target_index)

        if distance in REWARDS_DICT:
            reward = REWARDS_DICT[distance]
        else:
            reward = 0

        n_spikes = int(reward*5)
        dopa_spiketrain = [(run+1) * POLL_TIME + 1 + x *
                           DOPA_ISI for x in range(n_spikes)]

        # print(distance, reward, dopa_spiketrain)
        self.dopa_signal.spike_times = dopa_spiketrain

        self.mean_reward[self.target_index] = (
            self.mean_reward[self.target_index] + reward) / 2

        # Store performance data at every reward update
        self.weight_history.append(self.get_all_weights())
        self.mean_reward_history.append(copy(self.mean_reward))

        logging.debug(
            f"Applying reward={reward}, mean reward={self.mean_reward[self.target_index]}, success={reward}, distance={distance}")
        logging.debug(f"Average mean reward: {np.mean(self.mean_reward)}")

    def set_input_spiketrain(self, input_cell, time):
        """Set spike train encoding position of ball along y-axis.

        Args:
            input_cell (int): Input unit that corresponds to ball position.
            run (int): current iteration for correct spike time scaling
        """
        self.target_index = input_cell
        self.input_train = [time +
                            1 + x * ISI for x in range(NO_SPIKES)]
        self.input_train = [np.round(x, 1) for x in self.input_train]
        for input_neuron in range(self.num_neurons):
            nest.SetStatus(self.input_generators[input_neuron],
                           {'spike_times': []})

        nest.SetStatus(self.input_generators[input_cell],
                       {'spike_times': self.input_train})

    def get_performance_data(self):
        """retrieve the performance of the network across all simulations

        Returns:
            tuple: a Tuple of 2 numpy.arrays containing: reward history and weight history.
        """
        return (self.mean_reward_history,
                self.weight_history)
