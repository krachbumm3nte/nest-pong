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
#: float: Initial weight when using uniform initial weight distribution.
WEIGHT = 1300.0
#: float: Learning rate to use in weight updates.
LEARNING_RATE = 0.7

REWARDS_DICT = {0: 1., 1: 0.7, 2: 0.4, 3: 0.1}


class Network(object):
    """Represents the spiking neural network.

        Args:
            num_neurons (int): Number of neurons to use.
            with_noise (bool): Create and attach noise generators.
    """

    def __init__(self, num_neurons=20, with_noise=True, random_weights=False):
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
        nest.Connect(self.input_neurons, self.motor_neurons, {
                     'rule': 'all_to_all'}, {"weight": WEIGHT})

        self.spike_recorder = nest.Create("spike_recorder", self.num_neurons)
        nest.Connect(self.motor_neurons, self.spike_recorder,
                     {'rule': 'one_to_one'})

        self.mm = nest.Create("multimeter", self.num_neurons, {
                              'record_from': ["V_m"]})
        nest.Connect(self.mm, self.motor_neurons, {"rule": "one_to_one"})

        if with_noise:
            self.background_generator = nest.Create("noise_generator", self.num_neurons,
                                                    params={"std": BG_STD})
            nest.Connect(self.background_generator,
                         self.motor_neurons, {'rule': 'one_to_one'})
            for connection in nest.GetConnections(self.input_neurons, self.motor_neurons):
                connection.set({"weight": np.random.normal(WEIGHT, 5)})
        else:
            for connection in nest.GetConnections(self.input_neurons, self.motor_neurons):
                connection.set({"weight": np.random.normal(WEIGHT*1.22, 5)})


    def get_all_weights(self):
        """
        Get a matrix containing the weights between all input and motor neurons.
        
        Returns:
            numpy.array of synaptic weights.
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
        """
        Set the weights between input and motor neurons from a 2D weight matrix

        Args:
            TODO: test and document this.
        """
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                connection = nest.GetConnections(
                    self.input_neurons[i], self.motor_neurons[j])
                connection.set({"weight" : weights[i,j]})
                   
    def get_rates(self):
        """Get rates from spike detectors.

        Returns:
            numpy.array of neuronal spike rates.
        """
        events = self.spike_recorder.get("n_events")
        return np.array(events)

    def reset(self):
        """Reset network for new iteration
        """

        self.spike_recorder.set({"n_events": 0})

    def calc_reward(self, index, bare_reward):
        # TODO: initialize mean_reward to 0 and skip the if clause?
        """
        if self.mean_reward[index] == -1:
            self.mean_reward[index] = bare_reward
        """
        success = bare_reward - self.mean_reward[index]
        self.mean_reward[index] = float(
            self.mean_reward[index] + success / 2.0)
        self.performance[index] = np.ceil(bare_reward)
        return success

    def poll_network(self):
        """Get grid cell network wants to move to. Find this cell by finding the winning (highest rate) motor neuron.
        """

        rates = self.get_rates()

        logging.debug(f"Got rates: {rates}")
        self.winning_neuron = int(np.random.choice(
            np.flatnonzero(rates == rates.max())))
        self.weights = self.get_all_weights()

    def reward_by_move(self):
        """ Reward network based on whether the correct cell was targeted.
        """
        distance = np.abs(self.winning_neuron - self.target_index)

        if distance in REWARDS_DICT:
            bare_reward = REWARDS_DICT[distance]
        else:
            bare_reward = 0

        reward = self.calc_reward(self.target_index, bare_reward)

        self.apply_reward(reward)

        self.weight_history.append(copy(self.weights))
        self.mean_reward_history.append(copy(self.mean_reward))
        self.performance_history.append(copy(self.performance))

        logging.debug("Applying reward=%.3f, mean reward=%.3f, success=%.3f" %
                      (reward, self.mean_reward[self.target_index], reward))
        logging.debug("Mean rewards:")
        logging.debug(self.mean_reward)
        logging.debug(f"Average mean reward: {np.mean(self.mean_reward)}")
        logging.debug("Performances:")
        logging.debug(self.performance)

    def apply_reward(self, success):
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
            new_weight = old_weight + LEARNING_RATE * correlation * success

            connection.set({"weight": new_weight})

    def calculate_stdp(self, pre_spikes, post_spikes,
                       only_causal=True,
                       next_neighbor=True):
        """Calculates STDP trace for given spike trains.

        Args:
            pre_spikes(list, numpy.array): Presynaptic spike times in milliseconds.
            post_spikes(list, numpy.array): Postsynaptic spike times in milliseconds.
            only_causal (bool): Use only causal part.
            next_neighbor (bool): Use only next-neighbor coincidences.

        Returns:
            Scalar that corresponds to accumulated STDP trace.
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
                facilitation += STDP_AMPLITUDE * np.exp(-(spike - before_spike)
                                                        / STDP_TAU)
            if position < len(pre_spikes):
                after_spike = pre_spikes[position]
                depression += STDP_AMPLITUDE * np.exp(-(after_spike - spike) /
                                                      STDP_TAU)
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
        # Reset first
        self.target_index = input_cell
        self.input_train = [run * POLL_TIME +
                            1 + x * ISI for x in range(NO_SPIKES)]
        for input_neuron in range(self.num_neurons):
            nest.SetStatus(self.input_generators[input_neuron],
                           {'spike_times': []})
        nest.SetStatus(self.input_generators[input_cell],
                       {'spike_times': self.input_train})

    def get_performance_data(self):
        return (self.mean_reward_history,
                self.performance_history,
                self.weight_history)
