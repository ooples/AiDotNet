using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

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
public class RainbowDQNAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private RainbowDQNOptions<T> _options;
    private IOptimizer<T, Vector<T>, Vector<T>> _optimizer;

    private INeuralNetwork<T> _onlineNetwork;
    private INeuralNetwork<T> _targetNetwork;
    private PrioritizedReplayBuffer<T> _replayBuffer;

    private double _epsilon;
    private int _stepCount;
    private int _updateCount;
    private double _beta;

    // N-step buffer
    private List<(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)> _nStepBuffer;

    public RainbowDQNAgent(RainbowDQNOptions<T> options, IOptimizer<T, Vector<T>, Vector<T>>? optimizer = null)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _optimizer = optimizer ?? options.Optimizer ?? new AdamOptimizer<T, Vector<T>, Vector<T>>(this, new AdamOptimizerOptions<T, Vector<T>, Vector<T>>
        {
            LearningRate = 0.0001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });

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

        // Register networks with base class
        Networks.Add(_onlineNetwork);
        Networks.Add(_targetNetwork);
    }

    private INeuralNetwork<T> CreateDuelingNetwork()
    {
        int outputSize = _options.UseDistributional
            ? _options.ActionSize * _options.NumAtoms
            : _options.ActionSize;

        var architecture = new NeuralNetworkArchitecture<T>
        {
            TaskType = NeuralNetworkTaskType.Regression
        };

        // Use LayerHelper for production-ready network
        var layers = LayerHelper<T>.CreateDefaultDeepQNetworkLayers(architecture);

        architecture.Layers = layers.ToList();
        return new NeuralNetwork<T>(architecture, LossFunction);
    }

    private void InitializeReplayBuffer()
    {
        _replayBuffer = new PrioritizedReplayBuffer<T>(_options.ReplayBufferSize);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Noisy networks provide exploration, so we can use less epsilon
        double actualEpsilon = _options.UseNoisyNetworks ? 0.0 : _epsilon;

        if (training && Random.NextDouble() < actualEpsilon)
        {
            // Random exploration
            int randomAction = Random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        // Greedy action selection
        var qValues = ComputeQValues(state);
        int bestAction = ArgMax(qValues);

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    private Vector<T> ComputeQValues(Vector<T> state)
    {
        var output = _onlineNetwork.Predict(state);

        if (_options.UseDistributional)
        {
            // Distributional RL: convert distribution to Q-values
            var qValues = new Vector<T>(_options.ActionSize);
            double deltaZ = (_options.VMax - _options.VMin) / (_options.NumAtoms - 1);

            for (int action = 0; action < _options.ActionSize; action++)
            {
                T qValue = NumOps.Zero;
                for (int atom = 0; atom < _options.NumAtoms; atom++)
                {
                    int idx = action * _options.NumAtoms + atom;
                    double z = _options.VMin + atom * deltaZ;
                    var prob = output[idx];
                    qValue = NumOps.Add(qValue, NumOps.Multiply(prob, NumOps.FromDouble(z)));
                }
                qValues[action] = qValue;
            }

            return qValues;
        }

        return output;
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

        T nStepReturn = NumOps.Zero;
        T discount = NumOps.One;

        for (int i = 0; i < _nStepBuffer.Count; i++)
        {
            nStepReturn = NumOps.Add(nStepReturn, NumOps.Multiply(discount, _nStepBuffer[i].reward));
            discount = NumOps.Multiply(discount, DiscountFactor);

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
            return NumOps.Zero;
        }

        // Prioritized experience replay
        var (batch, indices, weights) = _replayBuffer.Sample(
            _options.BatchSize,
            _options.PriorityAlpha,
            _beta);

        T totalLoss = NumOps.Zero;
        var priorities = new List<double>();

        for (int i = 0; i < batch.Count; i++)
        {
            var experience = batch[i];
            var weight = NumOps.FromDouble(weights[i]);

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
                var nStepDiscount = NumOps.One;
                for (int n = 0; n < _options.NSteps; n++)
                {
                    nStepDiscount = NumOps.Multiply(nStepDiscount, DiscountFactor);
                }
                target = NumOps.Add(experience.reward, NumOps.Multiply(nStepDiscount, targetQ));
            }

            // Current Q-value
            var currentQValues = ComputeQValues(experience.state);
            int actionIndex = ArgMax(experience.action);
            var currentQ = currentQValues[actionIndex];

            // TD error
            var tdError = NumOps.Subtract(target, currentQ);
            var loss = NumOps.Multiply(tdError, tdError);
            loss = NumOps.Multiply(weight, loss);  // Importance sampling weight

            totalLoss = NumOps.Add(totalLoss, loss);

            // Update priority
            double priority = Math.Abs(NumOps.ToDouble(tdError));
            priorities.Add(priority);

            // Backpropagate
            var gradient = new Vector<T>(_options.ActionSize);
            gradient[actionIndex] = tdError;
            _onlineNetwork.Backward(gradient);
            _onlineNetwork.UpdateWeights(LearningRate);
        }

        // Update priorities in replay buffer
        _replayBuffer.UpdatePriorities(indices, priorities, _options.PriorityEpsilon);

        // Update target network
        if (_stepCount % _options.TargetUpdateFrequency == 0)
        {
            CopyNetworkWeights(_onlineNetwork, _targetNetwork);
        }

        _updateCount++;

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batch.Count));
    }

    private Vector<T> ComputeQValuesFromNetwork(INeuralNetwork<T> network, Vector<T> state)
    {
        var output = network.Forward(state);

        if (_options.UseDistributional)
        {
            var qValues = new Vector<T>(_options.ActionSize);
            double deltaZ = (_options.VMax - _options.VMin) / (_options.NumAtoms - 1);

            for (int action = 0; action < _options.ActionSize; action++)
            {
                T qValue = NumOps.Zero;
                for (int atom = 0; atom < _options.NumAtoms; atom++)
                {
                    int idx = action * _options.NumAtoms + atom;
                    double z = _options.VMin + atom * deltaZ;
                    var prob = output[idx];
                    qValue = NumOps.Add(qValue, NumOps.Multiply(prob, NumOps.FromDouble(z)));
                }
                qValues[action] = qValue;
            }

            return qValues;
        }

        return output;
    }

    private void CopyNetworkWeights(INeuralNetwork<T> source, INeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
    }

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["steps"] = NumOps.FromDouble(_stepCount);
        baseMetrics["updates"] = NumOps.FromDouble(_updateCount);
        baseMetrics["buffer_size"] = NumOps.FromDouble(_replayBuffer.Count);
        baseMetrics["epsilon"] = NumOps.FromDouble(_epsilon);
        return baseMetrics;
    }

    public override void ResetEpisode()
    {
        _nStepBuffer.Clear();
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = "RainbowDQN",
        };
    }

    public override byte[] Serialize()
    {
        throw new NotImplementedException("RainbowDQN serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("RainbowDQN deserialization not yet implemented");
    }

    public override Vector<T> GetParameters()
    {
        var onlineParams = _onlineNetwork.GetParameters();
        var targetParams = _targetNetwork.GetParameters();

        var combinedParams = new Vector<T>(onlineParams.Length + targetParams.Length);
        for (int i = 0; i < onlineParams.Length; i++)
        {
            combinedParams[i] = onlineParams[i];
        }
        for (int i = 0; i < targetParams.Length; i++)
        {
            combinedParams[onlineParams.Length + i] = targetParams[i];
        }

        return combinedParams;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int onlineParamCount = _onlineNetwork.ParameterCount;
        var onlineParams = new Vector<T>(onlineParamCount);
        var targetParams = new Vector<T>(parameters.Length - onlineParamCount);

        for (int i = 0; i < onlineParamCount; i++)
        {
            onlineParams[i] = parameters[i];
        }
        for (int i = 0; i < targetParams.Length; i++)
        {
            targetParams[i] = parameters[onlineParamCount + i];
        }

        _onlineNetwork.UpdateParameters(onlineParams);
        _targetNetwork.UpdateParameters(targetParams);
    }

    public override int FeatureCount => _options.StateSize;

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new RainbowDQNAgent<T>(_options, _optimizer);
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));

        var gradientMatrix = usedLossFunction.CalculateDerivative(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));
        var gradient = new Vector<T>(gradientMatrix.GetRow(0));
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _onlineNetwork.Backward(gradients);
        _onlineNetwork.UpdateWeights(learningRate);
    }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
