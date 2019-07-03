import sys
import os

# Concat top Level dir to system environmental variables
sys.path += os.path.join('..', '.')

from soup import *
from experiment import *


if __name__ == '__main__':
    def run_exp(net, prints=False):
        # INFO Run_ID needs to be more than 0, so that exp stores the trajectories!
        exp.run_net(net, 100, run_id=run_id + 1)
        exp.historical_particles[run_id] = net
        if prints:
            print("Fixpoint? " + str(net.is_fixpoint()))
            print("Loss " + str(loss))

    if True:
        # WeightWise Neural Network
        with FixpointExperiment(name="weightwise_self_application") as exp:
            for run_id in tqdm(range(20)):
                net = ParticleDecorator(WeightwiseNeuralNetwork(width=2, depth=2)
                                        .with_keras_params(activation='linear'))
                run_exp(net)
                K.clear_session()
            exp.log(exp.counters)
            exp.save(trajectorys=exp.without_particles())

    if False:
        # Aggregating Neural Network
        with FixpointExperiment(name="aggregating_self_application") as exp:
            for run_id in tqdm(range(10)):
                net = ParticleDecorator(AggregatingNeuralNetwork(aggregates=4, width=2, depth=2)
                                        .with_keras_params(activation='linear'))
                run_exp(net)
                K.clear_session()
            exp.log(exp.counters)
            exp.save(trajectorys=exp.without_particles())

    if False:
        #FFT Neural Network
        with FixpointExperiment() as exp:
            for run_id in tqdm(range(10)):
                net = ParticleDecorator(FFTNeuralNetwork(aggregates=4, width=2, depth=2)
                                        .with_keras_params(activation='linear'))
                run_exp(net)
                K.clear_session()
            exp.log(exp.counters)
            exp.save(trajectorys=exp.without_particles())

    if False:
        # ok so this works quite realiably
        with FixpointExperiment(name="weightwise_learning") as exp:
            for i in range(10):
                run_count = 100
                net = TrainingNeuralNetworkDecorator(ParticleDecorator(WeightwiseNeuralNetwork(width=2, depth=2)))
                net.with_params(epsilon=0.0001).with_keras_params(activation='linear')
                exp.historical_particles[net.get_uid()] = net
                for run_id in tqdm(range(run_count+1)):
                    net.compiled()
                    loss = net.train(epoch=run_id)
                    # run_exp(net)
                    # net.save_state(time=run_id)
                K.clear_session()
            exp.save(trajectorys=exp.without_particles())

    if False:
        # ok so this works quite realiably
        with FixpointExperiment(name="aggregating_learning") as exp:
            for i in range(10):
                run_count = 100
                net = TrainingNeuralNetworkDecorator(ParticleDecorator(AggregatingNeuralNetwork(4, width=2, depth=2)))
                net.with_params(epsilon=0.0001).with_keras_params(activation='linear')
                exp.historical_particles[net.get_uid()] = net
                for run_id in tqdm(range(run_count+1)):
                    net.compiled()
                    loss = net.train(epoch=run_id)
                    # run_exp(net)
                    # net.save_state(time=run_id)
                K.clear_session()
            exp.save(trajectorys=exp.without_particles())

    if False:
        # this explodes in our faces completely... NAN everywhere
        # TODO: Wtf is happening here?
        with FixpointExperiment() as exp:
            run_count = 10000
            net = TrainingNeuralNetworkDecorator(RecurrentNeuralNetwork(width=2, depth=2))\
                .with_params(epsilon=0.1e-2).with_keras_params(optimizer='sgd', activation='linear')
            for run_id in tqdm(range(run_count+1)):
                loss = net.compiled().train()
                if run_id % 500 == 0:
                    net.print_weights()
                    # print(net.apply_to_network(net))
                    print("Fixpoint? " + str(net.is_fixpoint()))
                    print("Loss " + str(loss))
                    print()
    if False:
        # and this gets somewhat interesting... we can still achieve non-trivial fixpoints
        # over multiple applications when training enough in-between
        with MixedFixpointExperiment() as exp:
            for run_id in range(10):
                net = TrainingNeuralNetworkDecorator(FFTNeuralNetwork(2, width=2, depth=2))\
                    .with_params(epsilon=0.0001, activation='sigmoid')
                exp.run_net(net)

                net.print_weights()

                print("Fixpoint? " + str(net.is_fixpoint()))
            exp.log(exp.counters)
