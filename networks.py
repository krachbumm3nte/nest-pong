import nest
from copy import copy
import logging
import numpy as np
from abc import ABC, abstractmethod


#: int: Simulation time per iteration in milliseconds.
POLL_TIME = 200
#: int: Number of spikes in an input spiketrain per iteration.
N_INPUT_SPIKES = 20
#: float: Inter-Spike Interval (ISI) of input spiketrain.
ISI = POLL_TIME/N_INPUT_SPIKES
#: float: Standard deviation of Gaussian current noise in picoampere.
BG_STD = 250.
#: float: Initial mean weight for synapses between input- and motor neurons.
MEAN_WEIGHT = 1300.


class PongNet(ABC):

    def __init__(self, apply_noise=True, num_neurons=20):
        """Abstract base class for network wrappers that learn to play pong.
        Args:
            num_neurons (int, optional): number of neurons to simulate. Changes here need to be matched in the game simulation in pong.py. Defaults to 20.
            with_noise (bool, optional): If true, noise generators are connected to the motor neurons of the network. Defaults to True.
        """
        self.apply_noise = apply_noise
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
        self.spike_recorder = nest.Create("spike_recorder", self.num_neurons)
        nest.Connect(self.motor_neurons, self.spike_recorder,
                     {'rule': 'one_to_one'})

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

    def get_spike_counts(self):
        """get the firing rates of the motor neurons from the spike_recorders.

        Returns:
            numpy.array: array of spike counts from the current simulation
        """
        events = self.spike_recorder.get("n_events")
        return np.array(events)

    def reset(self):
        """reset the network for a new iteration by clearing the spike recorders
        """
        self.spike_recorder.set({"n_events": 0})

    def set_input_spiketrain(self, input_cell, biological_time):
        """Set a spike train to the input neuron encoding the current ball position along y-axis.

        Args:
            input_cell (int): Index of the input neuron that corresponds to ball position.
            biological_time (float): current biological time within the NEST simulator
        """
        self.target_index = input_cell
        self.input_train = [
            biological_time + self.input_t_offset + i * ISI for i in range(N_INPUT_SPIKES)]
        # round spike timings to the first decimal to avoid conflicts with simulation timesteps
        self.input_train = [np.round(x, 1) for x in self.input_train]

        # TODO: why doesnt this work?
        """
        self.input_generators.spike_times = []
        self.input_generators[input_cell].spike_times = self.input_train
        """

        for input_neuron in range(self.num_neurons):
            nest.SetStatus(self.input_generators[input_neuron],
                           {'spike_times': []})

        nest.SetStatus(self.input_generators[input_cell],
                       {'spike_times': self.input_train})

    def find_winning_neuron(self):
        """find the motor neuron with the highest activation.

        Returns:
            int: index of the motor neuron with maximum activation
        """
        spikes = self.get_spike_counts()
        logging.debug(f"Got spike counts: {spikes}")
        # If multiple neurons have the same activation, one is chosen at random
        self.winning_neuron = int(np.random.choice(
            np.flatnonzero(spikes == spikes.max())))
        return self.winning_neuron

    def get_performance_data(self):
        """retrieve the performance of the network across all simulations

        Returns:
            tuple: a Tuple of 2 numpy.arrays containing: reward history and weight history.
        """
        return (self.mean_reward_history,
                self.weight_history)

    @abstractmethod
    def apply_synaptic_plasticity(self, biological_time):
        """apply weight changes to the synapses according to individual update rules

        Args:
            biological_time (float): current NEST simulation time
        """
        pass


class PongNetDopa(PongNet):

    #: int: Offset for input spikes in every iteration in milliseconds.
    # This offset reserves the first 25ms of every simulation step for the application of the dopaminergic
    # reward signal, avoiding interference between them and the networks response to input.
    input_t_offset = 29
    # constant scaling factor for determining the number of dopaminergic spikes to be applied
    dopa_signal_factor = 30
    # maximum number of dopaminergic spikes per iteration
    max_dopa_spikes = 10
    # ISI for the dopaminergic spike train
    dopa_isi = 3.

    def __init__(self, apply_noise=True, num_neurons=20):
        super().__init__(apply_noise, num_neurons)

        self.dopa_signal = nest.Create("spike_generator")
        self.dopa_parrot = nest.Create("parrot_neuron")
        nest.Connect(self.dopa_signal, self.dopa_parrot)

        self.vt = nest.Create("volume_transmitter")
        nest.Connect(self.dopa_parrot, self.vt)

        nest.SetDefaults("stdp_dopamine_synapse", {"vt": self.vt.get("global_id"),
                                                   "tau_c": 80, "tau_n": 30, "tau_plus": 40, "Wmin": 1150, "Wmax": 1550,
                                                   "b": 0.02, "A_plus": 0.85})

        nest.Connect(self.input_neurons, self.motor_neurons, {'rule': 'all_to_all'}, {
                     "synapse_model": "stdp_dopamine_synapse", "weight": nest.random.normal(MEAN_WEIGHT, 5.)})

        if apply_noise:
            self.background_generator = nest.Create("noise_generator", self.num_neurons,
                                                    params={"std": BG_STD})
            nest.Connect(self.background_generator,
                         self.motor_neurons, {'rule': 'one_to_one'})

    def apply_synaptic_plasticity(self, biological_time):
        """ inject dopaminergic spikes into the network based on the proportion of the motor neurons' spikes
        that stem from the desired output 
        """

        self.find_winning_neuron()

        spike_counts = self.get_spike_counts()
        target_n_spikes = spike_counts[self.target_index]
        # avoid zero division if none of the neurons fired
        total_n_spikes = max(sum(spike_counts), 1)

        n_dopa_spikes = int((target_n_spikes/total_n_spikes)
                            * self.dopa_signal_factor)

        # cap the dopaminergic signal to avoid continually increasing synaptic weights
        n_dopa_spikes = min(n_dopa_spikes, self.max_dopa_spikes)

        # set the dopaminergic spike train to be applied at the start of the next simulation step
        dopa_spiketrain = [biological_time + 1 +
                           x * self.dopa_isi for x in range(n_dopa_spikes)]
        self.dopa_signal.spike_times = dopa_spiketrain

        reward = n_dopa_spikes / self.max_dopa_spikes
        self.mean_reward[self.target_index] = (
            self.mean_reward[self.target_index] + reward) / 2

        self.weight_history.append(self.get_all_weights())
        self.mean_reward_history.append(copy(self.mean_reward))

        logging.debug(f"Applying reward={reward}")
        logging.debug(
            f"Average reward across all neurons: {np.mean(self.mean_reward)}")

    def __repr__(self) -> str:
        return "Dopaminergic synapse network" + (" with noise" if self.apply_noise else "")

class PongNetRSTDP(PongNet):

    #: int: Offset for input spikes in every iteration in milliseconds.
    input_t_offset = 1
    #: float: Learning rate to use in weight updates.
    learning_rate = 0.7
    #: dict: reward to be applied to synapses depending on the distance between target and prediction.
    rewards_dict = {0: 1., 1: 0.7, 2: 0.4, 3: 0.1}
    #: float: Amplitude of STDP curve in arbitrary units.
    stdp_amplitude = 36.0
    #: float: Time constant of STDP curve in milliseconds.
    stdp_tau = 64.
    #: int: Satuation value for accumulated STDP.
    stdp_saturation = 128

    def __init__(self, apply_noise=True, num_neurons=20):

        super().__init__(apply_noise, num_neurons)

        if apply_noise:
            self.background_generator = nest.Create("noise_generator", self.num_neurons,
                                                    params={"std": BG_STD})
            nest.Connect(self.background_generator,
                         self.motor_neurons, {'rule': 'one_to_one'})
            nest.Connect(self.input_neurons, self.motor_neurons, {'rule': 'all_to_all'}, {
                         "weight": nest.random.normal(MEAN_WEIGHT, 5)})
        else:
            # because the noise_generators cause adtitional spikes in the motor neurons, it is
            # necessary to compensate for their absence by slightly increasing the mean of
            # the weights between input and motor neurons to achieve similar spiking rates.
            nest.Connect(self.input_neurons, self.motor_neurons, {'rule': 'all_to_all'}, {
                         "weight": nest.random.normal(MEAN_WEIGHT*1.22, 5)})

    def apply_synaptic_plasticity(self, biological_time):
        """ Reward network based on how close winning neuron and desired output are.
        """

        self.find_winning_neuron()

        distance = np.abs(self.winning_neuron - self.target_index)
        if distance in self.rewards_dict:
            bare_reward = self.rewards_dict[distance]
        else:
            bare_reward = 0

        reward = bare_reward - self.mean_reward[self.target_index]
        self.mean_reward[self.target_index] = (
            self.mean_reward[self.target_index] + reward) / 2.0

        self.apply_reward(reward)

        self.weight_history.append(self.get_all_weights())
        self.mean_reward_history.append(copy(self.mean_reward))

        logging.debug(f"Applying reward={reward}")
        logging.debug(
            f"Average reward across all neurons: {np.mean(self.mean_reward)}")

    def apply_reward(self, reward):
        """apply the previously calculated reward to all relevant synapses according to R-STDP principle

        Args:
            reward (float): reward to be passed on to the synapses
        """

        # store spike timings of all motor neurons
        post_events = {}
        offset = self.motor_neurons[0].get("global_id")
        for index, event in enumerate(self.spike_recorder.get("events")):
            post_events[offset + index] = event["times"]

        # iterate over all connections from the relevant input neuron and apply STDP
        for connection in nest.GetConnections(self.input_neurons[self.target_index]):
            motor_neuron = connection.get("target")
            motor_spikes = post_events[motor_neuron]
            correlation = self.calculate_stdp(self.input_train, motor_spikes)
            old_weight = connection.get("weight")
            new_weight = old_weight + self.learning_rate * correlation * reward

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
                facilitation += self.stdp_amplitude * \
                    np.exp(-(spike - before_spike) / self.stdp_tau)
            if position < len(pre_spikes):
                after_spike = pre_spikes[position]
                depression += self.stdp_amplitude * \
                    np.exp(-(after_spike - spike) / self.stdp_tau)
            last_position = position
        if only_causal:
            return min(facilitation, self.stdp_saturation)
        else:
            return min(facilitation - depression, self.stdp_saturation)

    def __repr__(self) -> str:
        return "r-STDP network" + (" with noise" if self.apply_noise else "")
