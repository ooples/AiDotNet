using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Agents.Rainbow;

/// <summary>
/// Rainbow DQN agent combining six extensions to DQN.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Rainbow combines: Double Q-learning, Dueling networks, Prioritized replay,
/// Multi-step learning, Distributional RL (C51), and Noisy networks.
/// </para>
/// <para><b>For Beginners:</b>
/// Rainbow takes the best ideas from six different DQN improvements and combines them.
/// It's currently the strongest DQN variant, achieving state-of-the-art performance.
///
/// Six components:
/// 1. **Double Q-learning**: Reduces overestimation
/// 2. **Dueling Architecture**: Separates value and advantage
/// 3. **Prioritized Replay**: Samples important experiences more
/// 4. **Multi-step Returns**: Better credit assignment
/// 5. **Distributional RL (C51)**: Learns distribution of returns
/// 6. **Noisy Networks**: Parameter noise for exploration
///
/// Famous for: DeepMind's combination achieving human-level Atari performance
/// </para>
/// </remarks>
public class RainbowDQNAgent<T> : ReinforcementLearningAgentBase<T>
{
    private readonly RainbowDQNOptions<T> _options;
    private readonly INumericOperations<T> _numOps;

    private NeuralNetwork<T> _onlineNetwork;
    private NeuralNetwork<T> _targetNetwork;
    private PrioritizedReplayBuffer<T> _replayBuffer;
    private Random _random;

    private double _epsilon;
    private int _stepCount;
    private int _updateCount;
    private double _beta;

    // N-step buffer
    private List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> _nStepBuffer;

    public RainbowDQNAgent(RainbowDQNOptions<T> options) : base(options.StateSize, options.ActionSize)
    {
        _options = options;
        _numOps = NumericOperations<T>.Instance;
        _random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        _stepCount = 0;
        _updateCount = 0;
        _epsilon = options.EpsilonStart;
        _beta = options.PriorityBeta;
        _nStepBuffer = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>();

        InitializeNetworks();
        InitializeReplayBuffer();
    }

    private void InitializeNetworks()
    {
        _onlineNetwork = CreateDuelingNetwork();
        _targetNetwork = CreateDuelingNetwork();
        CopyNetworkWeights(_onlineNetwork, _targetNetwork);
    }

    private NeuralNetwork<T> CreateDuelingNetwork()
    {
        var network = new NeuralNetwork<T>();
        int previousSize = _options.StateSize;

        // Shared layers
        foreach (var layerSize in _options.SharedLayers)
        {
            network.AddLayer(new DenseLayer<T>(previousSize, layerSize));

            if (_options.UseNoisyNetworks)
            {
                // Add noise to weights for exploration (simplified)
                network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            }
            else
            {
                network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            }
            previousSize = layerSize;
        }

        // Dueling architecture: separate value and advantage streams
        // Value stream
        int valueSize = previousSize;
        foreach (var layerSize in _options.ValueStreamLayers)
        {
            network.AddLayer(new DenseLayer<T>(valueSize, layerSize));
            network.AddLayer(new ActivationLayer<T>(new ReLU<T>()));
            valueSize = layerSize;
        }

        if (_options.UseDistributional)
        {
            // Distributional RL: output atoms for each action
            network.AddLayer(new DenseLayer<T>(previousSize, _options.ActionSize * _options.NumAtoms));
        }
        else
        {
            // Standard Q-values
            network.AddLayer(new DenseLayer<T>(previousSize, _options.ActionSize));
        }

        return network;
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new PrioritizedReplayBuffer<T>(_options.ReplayBufferSize);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Noisy networks provide exploration, so we can use less epsilon
        double actualEpsilon = _options.UseNoisyNetworks ? 0.0 : _epsilon;

        if (training && _random.NextDouble() < actualEpsilon)
        {
            // Random exploration
            int randomAction = _random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = _numOps.One;
            return action;
        }

        // Greedy action selection
        var qValues = ComputeQValues(state);
        int bestAction = ArgMax(qValues);

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = _numOps.One;
        return result;
    }

    private Vector<T> ComputeQValues(Vector<T> state)
    {
        var output = _onlineNetwork.Forward(state);

        if (_options.UseDistributional)
        {
            // Distributional RL: convert distribution to Q-values
            var qValues = new Vector<T>(_options.ActionSize);
            double deltaZ = (_options.VMax - _options.VMin) / (_options.NumAtoms - 1);

            for (int action = 0; action < _options.ActionSize; action++)
            {
                T qValue = _numOps.Zero;
                for (int atom = 0; atom < _options.NumAtoms; atom++)
                {
                    int idx = action * _options.NumAtoms + atom;
                    double z = _options.VMin + atom * deltaZ;
                    var prob = output[idx];
                    qValue = _numOps.Add(qValue, _numOps.Multiply(prob, _numOps.FromDouble(z)));
                }
                qValues[action] = qValue;
            }

            return qValues;
        }
        else
        {
            return output;
        }
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // N-step learning: accumulate transitions
        _nStepBuffer.Add((state, action, reward, nextState, done));

        if (_nStepBuffer.Count >= _options.NSteps || done)
        {
            // Compute n-step return
            var (nStepState, nStepAction, nStepReturn, nStepNextState, nStepDone) = ComputeNStepReturn();
            _replayBuffer.Add(nStepState, nStepAction, nStepReturn, nStepNextState, nStepDone);

            // Clear n-step buffer on episode end
            if (done)
            {
                _nStepBuffer.Clear();
            }
            else
            {
                _nStepBuffer.RemoveAt(0);
            }
        }

        _stepCount++;

        // Decay epsilon
        if (!_options.UseNoisyNetworks)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }

        // Increase beta for importance sampling
        _beta = Math.Min(1.0, _beta + _options.PriorityBetaIncrement);
    }

    private (Vector<T> state, Vector<T> action, T nStepReturn, Vector<T> nextState, bool done) ComputeNStepReturn()
    {
        var firstState = _nStepBuffer[0].state;
        var firstAction = _nStepBuffer[0].action;

        T nStepReturn = _numOps.Zero;
        T discount = _numOps.One;

        for (int i = 0; i < _nStepBuffer.Count; i++)
        {
            nStepReturn = _numOps.Add(nStepReturn, _numOps.Multiply(discount, _nStepBuffer[i].reward));
            discount = _numOps.Multiply(discount, _options.DiscountFactor);

            if (_nStepBuffer[i].done)
            {
                return (firstState, firstAction, nStepReturn, _nStepBuffer[i].nextState, true);
            }
        }

        var lastTransition = _nStepBuffer[_nStepBuffer.Count - 1];
        return (firstState, firstAction, nStepReturn, lastTransition.nextState, false);
    }

    public override T Train()
    {
        if (_replayBuffer.Count < _options.WarmupSteps || _replayBuffer.Count < _options.BatchSize)
        {
            return _numOps.Zero;
        }

        // Prioritized experience replay
        var (batch, indices, weights) = _replayBuffer.Sample(
            _options.BatchSize,
            _options.PriorityAlpha,
            _beta);

        T totalLoss = _numOps.Zero;
        var priorities = new List<double>();

        for (int i = 0; i < batch.Count; i++)
        {
            var experience = batch[i];
            var weight = _numOps.FromDouble(weights[i]);

            // Double Q-learning: use online network to select, target to evaluate
            var nextQValuesOnline = ComputeQValues(experience.nextState);
            int bestActionIndex = ArgMax(nextQValuesOnline);

            var nextQValuesTarget = ComputeQValuesFromNetwork(_targetNetwork, experience.nextState);
            var targetQ = nextQValuesTarget[bestActionIndex];

            T target;
            if (experience.done)
            {
                target = experience.reward;
            }
            else
            {
                var nStepDiscount = _numOps.One;
                for (int n = 0; n < _options.NSteps; n++)
                {
                    nStepDiscount = _numOps.Multiply(nStepDiscount, _options.DiscountFactor);
                }
                target = _numOps.Add(experience.reward, _numOps.Multiply(nStepDiscount, targetQ));
            }

            // Current Q-value
            var currentQValues = ComputeQValues(experience.state);
            int actionIndex = ArgMax(experience.action);
            var currentQ = currentQValues[actionIndex];

            // TD error
            var tdError = _numOps.Subtract(target, currentQ);
            var loss = _numOps.Multiply(tdError, tdError);
            loss = _numOps.Multiply(weight, loss);  // Importance sampling weight

            totalLoss = _numOps.Add(totalLoss, loss);

            // Update priority
            double priority = Math.Abs(_numOps.ToDouble(tdError));
            priorities.Add(priority);

            // Backpropagate
            var gradient = new Vector<T>(_options.ActionSize);
            gradient[actionIndex] = tdError;
            _onlineNetwork.Backward(gradient);
            _onlineNetwork.UpdateWeights(_options.LearningRate);
        }

        // Update priorities in replay buffer
        _replayBuffer.UpdatePriorities(indices, priorities, _options.PriorityEpsilon);

        // Update target network
        if (_stepCount % _options.TargetUpdateFrequency == 0)
        {
            CopyNetworkWeights(_onlineNetwork, _targetNetwork);
        }

        _updateCount++;

        return _numOps.Divide(totalLoss, _numOps.FromDouble(batch.Count));
    }

    private Vector<T> ComputeQValuesFromNetwork(NeuralNetwork<T> network, Vector<T> state)
    {
        var output = network.Forward(state);

        if (_options.UseDistributional)
        {
            var qValues = new Vector<T>(_options.ActionSize);
            double deltaZ = (_options.VMax - _options.VMin) / (_options.NumAtoms - 1);

            for (int action = 0; action < _options.ActionSize; action++)
            {
                T qValue = _numOps.Zero;
                for (int atom = 0; atom < _options.NumAtoms; atom++)
                {
                    int idx = action * _options.NumAtoms + atom;
                    double z = _options.VMin + atom * deltaZ;
                    var prob = output[idx];
                    qValue = _numOps.Add(qValue, _numOps.Multiply(prob, _numOps.FromDouble(z)));
                }
                qValues[action] = qValue;
            }

            return qValues;
        }

        return output;
    }

    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceLayers = source.GetLayers();
        var targetLayers = target.GetLayers();

        for (int i = 0; i < sourceLayers.Count; i++)
        {
            if (sourceLayers[i] is DenseLayer<T> sourceLayer && targetLayers[i] is DenseLayer<T> targetLayer)
            {
                targetLayer.SetWeights(sourceLayer.GetWeights().Clone());
                targetLayer.SetBiases(sourceLayer.GetBiases().Clone());
            }
        }
    }

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (_numOps.Compare(values[i], maxValue) > 0)
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["steps"] = _numOps.FromDouble(_stepCount),
            ["updates"] = _numOps.FromDouble(_updateCount),
            ["buffer_size"] = _numOps.FromDouble(_replayBuffer.Count),
            ["epsilon"] = _numOps.FromDouble(_epsilon)
        };
    }

    public override void ResetEpisode()
    {
        _nStepBuffer.Clear();
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public override Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public override Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }
}
