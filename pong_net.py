import nest
from copy import copy
import logging
import numpy as np


#: float: Amplitude of STDP curve in arbitrary units.
STDP_AMPLITUDE = 36.0
#: float: Time constant of STDP curve in milliseconds.
STDP_TAU = 64.
#: int: Cutoff for accumulated STDP.
STDP_SATURATION = 128
#: int: Amount of time the network is simulated in milliseconds.
POLL_TIME = 200
#: int: Number of spikes for input spiketrain.
NO_SPIKES = 20
#: float: Inter-Spike Interval (ISI) of input spiketrain.
ISI = 10.
#: float: Standard deviation of Gaussian current noise in picoampere.
BG_STD = 200.
#: float: Initial mean weight when applying noise to motor neurons.
MEAN_WEIGHT = 1300.0
#: float: Learning rate to use in weight updates.
LEARNING_RATE = 0.7
#: dict: reward to be applied to synapses in dependence on the distance between target and prediction.
REWARDS_DICT = {0: 1., 1: 0.7, 2: 0.4, 3: 0.1}


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
        self.performance = np.array([-1. for _ in range(self.num_neurons)])
        self.mean_reward_history = []
        self.performance_history = []

        self.input_generators = nest.Create(
            "spike_generator", self.num_neurons)
        self.input_neurons = nest.Create("parrot_neuron", self.num_neurons)
        nest.Connect(self.input_generators, self.input_neurons,
                     {'rule': 'one_to_one'})

        self.motor_neurons = nest.Create("iaf_psc_exp", self.num_neurons)

        self.spike_recorder = nest.Create("spike_recorder", self.num_neurons)
        nest.Connect(self.motor_neurons, self.spike_recorder,
                     {'rule': 'one_to_one'})

        if with_noise:
            self.background_generator = nest.Create("noise_generator", self.num_neurons,
                                                    params={"std": BG_STD})
            nest.Connect(self.background_generator,
                         self.motor_neurons, {'rule': 'one_to_one'})
            nest.Connect(self.input_neurons, self.motor_neurons, {'rule': 'all_to_all'}, {
                         "weight": nest.random.normal(MEAN_WEIGHT, 5)})
        else:
            # because the noise_generators cause adtitional spikes in the motor neurons, it is
            # necessary to compensate for their absence by slightly increasing the mean of
            # the weights between input and motor neurons to achieve similar spiking rates
            nest.Connect(self.input_neurons, self.motor_neurons, {'rule': 'all_to_all'}, {
                         "weight": nest.random.normal(MEAN_WEIGHT*1.22, 5)})

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

    def calc_reward(self, index, bare_reward):
        """calculate the reward to be applied to the synapses emerging from a given input
        neuron based on a bare reward and previous performance.

        Args:
            index (int): index of the input neuron stimulated in the current iteration
            bare_reward (float): bare reward (between 0. and 1.) to be scaled with respect to previous performance

        Returns:
            float: scaled reward that can be applied to the STDP computation
        """
        scaled_reward = bare_reward - self.mean_reward[index]
        self.mean_reward[index] = float(
            self.mean_reward[index] + scaled_reward / 2.0)
        self.performance[index] = np.ceil(bare_reward)
        return scaled_reward

    def poll_network(self):
        """Get the grid cell the network wants to move to. Find this cell by finding 
        the winning (highest rate) motor neuron.
        """
        rates = self.get_rates()
        logging.debug(f"Got rates: {rates}")

        # If multiple neurons have the same activation, one is chosen on random
        self.winning_neuron = int(np.random.choice(
            np.flatnonzero(rates == rates.max())))

    def reward_by_move(self):
        """ Reward network based on how close winning neuron and desired output are.
        """
        distance = np.abs(self.winning_neuron - self.target_index)

        if distance in REWARDS_DICT:
            bare_reward = REWARDS_DICT[distance]
        else:
            bare_reward = 0

        reward = self.calc_reward(self.target_index, bare_reward)

        self.apply_reward(reward)

        # Store performance data at every reward update
        self.weight_history.append(self.get_all_weights())
        self.mean_reward_history.append(copy(self.mean_reward))
        self.performance_history.append(copy(self.performance))

        logging.debug("Applying reward=%.3f, mean reward=%.3f, success=%.3f" %
                      (reward, self.mean_reward[self.target_index], reward))
        logging.debug("Mean rewards:")
        logging.debug(self.mean_reward)
        logging.debug(f"Average mean reward: {np.mean(self.mean_reward)}")
        logging.debug("Performances:")
        logging.debug(self.performance)

    def apply_reward(self, reward):
        """apply the previously calculated reward to all relevant synapses according to R-STDP principle

        Args:
            reward (float): reward to be passed on to the synapses
        """

        post_events = {}
        offset = self.motor_neurons[0].get("global_id")
        for index, event in enumerate(self.spike_recorder.get("events")):
            post_events[offset + index] = event["times"]

        for connection in nest.GetConnections(self.input_neurons[self.target_index]):

            # iterate connections originating from input neuron
            motor_neuron = connection.get("target")
            post_spikes = post_events[motor_neuron]
            correlation = self.calculate_stdp(self.input_train, post_spikes)
            old_weight = connection.get("weight")
            new_weight = old_weight + LEARNING_RATE * correlation * reward

            connection.set({"weight": new_weight})

    def calculate_stdp(self, pre_spikes, post_spikes, only_causal=True, next_neighbor=True):
        """Calculates STDP trace for given spike trains.

        Args:
            pre_spikes (list, numpy.array): Presynaptic spike times in milliseconds.
            post_spikes (list, numpy.array): Postsynaptic spike times in milliseconds.
            only_causal (bool, optional): Use only causal part.. Defaults to True.
            next_neighbor (bool, optional): Use only next-neighbor coincidences.. Defaults to True.

        Returns:
            [float]: Scalar that corresponds to accumulated STDP trace.
        """

        pre_spikes, post_spikes = np.sort(pre_spikes), np.sort(post_spikes)
        facilitation = 0
        depression = 0
        positions = np.searchsorted(pre_spikes, post_spikes)
        last_position = -1
        for spike, position in zip(post_spikes, positions):
            if position == last_position and next_neighbor:
                continue  # only next-neighbor pairs
            if position > 0:
                before_spike = pre_spikes[position - 1]
                facilitation += STDP_AMPLITUDE * \
                    np.exp(-(spike - before_spike) / STDP_TAU)
            if position < len(pre_spikes):
                after_spike = pre_spikes[position]
                depression += STDP_AMPLITUDE * \
                    np.exp(-(after_spike - spike) / STDP_TAU)
            last_position = position
        if only_causal:
            return min(facilitation, STDP_SATURATION)
        else:
            return min(facilitation - depression, STDP_SATURATION)

    def set_input_spiketrain(self, input_cell, run):
        """Set spike train encoding position of ball along y-axis.

        Args:
            input_cell (int): Input unit that corresponds to ball position.
            run (int): current iteration for correct spike time scaling
        """
        self.target_index = input_cell
        self.input_train = [run * POLL_TIME +
                            1 + x * ISI for x in range(NO_SPIKES)]
        for input_neuron in range(self.num_neurons):
            nest.SetStatus(self.input_generators[input_neuron],
                           {'spike_times': []})

        nest.SetStatus(self.input_generators[input_cell],
                       {'spike_times': self.input_train})

    def get_performance_data(self):
        """retrieve the performance of the network across all simulations

        Returns:
            tuple: a Tuple of 3 numpy.arrays containing: reward histor, performance history and weight history.
        """
        return (self.mean_reward_history,
                self.performance_history,
                self.weight_history)
