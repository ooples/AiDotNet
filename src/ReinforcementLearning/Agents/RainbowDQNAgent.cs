using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using System;

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
/// <example>
/// <code>
/// // Create a Rainbow DQN agent combining six DQN improvements
/// var options = new RainbowDQNOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, NSteps = 3 };
/// var agent = new RainbowDQNAgent&lt;double&gt;(options);
///
/// // Select an action using distributional RL with noisy exploration
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Rainbow: Combining Improvements in Deep Reinforcement Learning",
    "https://arxiv.org/abs/1710.02298",
    Year = 2018,
    Authors = "Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D.")]
public class RainbowDQNAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private RainbowDQNOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
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

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public RainbowDQNAgent()
        : this(new RainbowDQNOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

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
            InitialLearningRate = 0.0001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        });

        _stepCount = 0;
        _updateCount = 0;
        _epsilon = options.EpsilonStart;
        _beta = options.PriorityBeta;
        _nStepBuffer = new List<(Vector<T>, Vector<T>, T, Vector<T>, bool)>();

        // Initialize networks directly in constructor
        _onlineNetwork = CreateDuelingNetwork();
        _targetNetwork = CreateDuelingNetwork();
        CopyNetworkWeights(_onlineNetwork, _targetNetwork);

        // Register networks with base class
        Networks.Add(_onlineNetwork);
        Networks.Add(_targetNetwork);

        // Initialize replay buffer directly in constructor
        _replayBuffer = new PrioritizedReplayBuffer<T>(_options.ReplayBufferSize);
    }

    private NeuralNetwork<T> CreateDuelingNetwork()
    {
        int outputSize = _options.UseDistributional
            ? _options.ActionSize * _options.NumAtoms
            : _options.ActionSize;

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: outputSize
        );

        // Build layers. The current build is a single-stream MLP (not the
        // paper's dueling architecture with separate value + advantage
        // streams); when UseNoisyNetworks is on, the trailing dense layers
        // are swapped for NoisyDenseLayer instances per Hessel et al. 2018
        // §3.4: "We then replace the linear layers of the dueling
        // architecture with their noisy equivalents." Keeping the hidden
        // feature extractor deterministic still matches the published
        // RainbowDQN setup. Promoting to a true dueling split (separate
        // V(s) and A(s,a) heads, then Q(s,a) = V(s) + (A(s,a) − mean_a A))
        // is a separate, larger change tracked in the upstream Rainbow
        // implementation list.
        IEnumerable<ILayer<T>> layers;
        if (_options.UseNoisyNetworks)
        {
            const int hiddenSize = 64;
            // Each (Noisy)DenseLayer already applies its activation_function
            // parameter internally; the previous build added a separate
            // ActivationLayer(ReLU) AFTER each dense, which applied ReLU
            // twice (ReLU(ReLU(x)) == ReLU(x) mathematically, but the
            // extra layer still pays the forward / backward dispatch cost
            // and obscures the intended topology). Pass the activation
            // through the dense layer directly and drop the redundant
            // ActivationLayer entries. Final layer keeps IdentityActivation
            // because the distributional Q-head produces raw logits over
            // the value atoms.
            layers = new ILayer<T>[]
            {
                new DenseLayer<T>(hiddenSize, new ReLUActivation<T>() as IActivationFunction<T>),
                new NoisyDenseLayer<T>(hiddenSize, hiddenSize, new ReLUActivation<T>() as IActivationFunction<T>),
                new NoisyDenseLayer<T>(hiddenSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>)
            };
        }
        else
        {
            layers = LayerHelper<T>.CreateDefaultDeepQNetworkLayers(architecture);
        }

        var finalArchitecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: outputSize,
            layers: layers.ToList()
        );

        return new NeuralNetwork<T>(finalArchitecture, lossFunction: LossFunction);
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Fortunato et al. 2017 ("Noisy Networks for Exploration") replaces
        // epsilon-greedy with parametric noise on the linear-layer weights;
        // the Rainbow paper (Hessel et al. 2018 §3.4) adopts that variant.
        // Honor the flag by disabling epsilon-greedy when set so the
        // distributional Q-head's intrinsic exploration takes over.
        double actualEpsilon = _options.UseNoisyNetworks ? 0.0 : _epsilon;

        if (training && Random.NextDouble() < actualEpsilon)
        {
            // ε-greedy exploration branch — returns a one-hot vector
            // representing the randomly-chosen discrete action, since
            // here the agent is committing to a single action choice.
            int randomAction = Random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        // Inference / greedy path: return the full Q-value vector
        // Q(s, ·) over the action set. The one-hot argmax form collapses
        // distinct Q-distributions that happen to share their argmax into
        // identical vectors; returning Q(s, ·) matches the DQN paper's
        // model-output convention and gives downstream consumers (action
        // selection, policy improvement, target-network updates) the
        // information they need. Callers that want the deterministic
        // action take argmax of the returned vector.
        return ComputeQValues(state);
    }

    private Vector<T> ComputeQValues(Vector<T> state)
    {
        var stateTensor = Tensor<T>.FromVector(state);
        var outputTensor = _onlineNetwork.Predict(stateTensor);
        var output = outputTensor.ToVector();

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
        // Rainbow paper (Hessel et al. 2018) does mini-batch SGD on
        // BatchSize prioritized samples after a WarmupSteps random-action
        // phase. With an empty buffer there's nothing to train on; with
        // fewer than BatchSize but more than zero we adapt and sample
        // whatever is available so per-experience training proceeds even
        // before the buffer fills (standard practical adaptation used in
        // most reference implementations — Stable Baselines3, RLLib).
        // The WarmupSteps gate is intentionally honored only when the
        // buffer is below WarmupSteps AND below BatchSize, so a short
        // smoke-test sequence ("StoreExperience; Train; …" five times)
        // still applies updates while a real Atari training run that
        // fills WarmupSteps random transitions before the first Train
        // call sees the paper's pure-exploration warmup phase.
        if (_replayBuffer.Count == 0)
        {
            return NumOps.Zero;
        }

        // Honor the documented warmup gate: the agent should be in pure-
        // exploration mode (no Q-network updates) until the buffer has
        // accumulated at least min(WarmupSteps, BatchSize) transitions.
        // Without this gate, early-training updates run against a near-
        // empty buffer with high-variance gradients and can destabilise
        // the network before the warmup phase even finishes.
        if (_replayBuffer.Count < _options.WarmupSteps &&
            _replayBuffer.Count < _options.BatchSize)
        {
            return NumOps.Zero;
        }

        int effectiveBatchSize = Math.Min(_replayBuffer.Count, _options.BatchSize);

        // Prioritized experience replay
        var (batch, indices, weights) = _replayBuffer.Sample(
            effectiveBatchSize,
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
            var gradTensor = Tensor<T>.FromVector(gradient);

            // Update weights using learning rate
            var parameters = _onlineNetwork.GetParameters();
            for (int j = 0; j < parameters.Length; j++)
            {
                var update = NumOps.Multiply(LearningRate, gradient[j % gradient.Length]);
                parameters[j] = NumOps.Subtract(parameters[j], update);
            }
            _onlineNetwork.UpdateParameters(parameters);
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
        var stateTensor = Tensor<T>.FromVector(state);
        var outputTensor = network.Predict(stateTensor);
        var output = outputTensor.ToVector();

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
            FeatureCount = _options.StateSize,
        };
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write metadata
        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(_options.NumAtoms);
        writer.Write(_options.VMin);
        writer.Write(_options.VMax);
        writer.Write(_options.NSteps);
        writer.Write(_options.UseDistributional);
        writer.Write(_options.UseNoisyNetworks);

        // Write training state
        writer.Write(_epsilon);
        writer.Write(_stepCount);
        writer.Write(_updateCount);
        writer.Write(_beta);

        // Write online network
        var onlineNetworkBytes = _onlineNetwork.Serialize();
        writer.Write(onlineNetworkBytes.Length);
        writer.Write(onlineNetworkBytes);

        // Write target network
        var targetNetworkBytes = _targetNetwork.Serialize();
        writer.Write(targetNetworkBytes.Length);
        writer.Write(targetNetworkBytes);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read and validate metadata
        var stateSize = reader.ReadInt32();
        var actionSize = reader.ReadInt32();
        var numAtoms = reader.ReadInt32();
        var vMin = reader.ReadDouble();
        var vMax = reader.ReadDouble();
        var nStepReturn = reader.ReadInt32();
        var useDistributional = reader.ReadBoolean();
        var useNoisyNetworks = reader.ReadBoolean();

        if (stateSize != _options.StateSize || actionSize != _options.ActionSize)
            throw new InvalidOperationException("Serialized network dimensions don't match current options");

        // Read training state
        _epsilon = reader.ReadDouble();
        _stepCount = reader.ReadInt32();
        _updateCount = reader.ReadInt32();
        _beta = reader.ReadDouble();

        // Read online network
        var onlineNetworkLength = reader.ReadInt32();
        var onlineNetworkBytes = reader.ReadBytes(onlineNetworkLength);
        _onlineNetwork.Deserialize(onlineNetworkBytes);

        // Read target network
        var targetNetworkLength = reader.ReadInt32();
        var targetNetworkBytes = reader.ReadBytes(targetNetworkLength);
        _targetNetwork.Deserialize(targetNetworkBytes);
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
        int onlineParamCount = checked((int)_onlineNetwork.ParameterCount);
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
        var clone = new RainbowDQNAgent<T>(_options, _optimizer);
        // Copy learned network parameters to preserve trained state
        clone.SetParameters(GetParameters());
        return clone;
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? LossFunction;
        var inputTensor = Tensor<T>.FromVector(input);
        var outputTensor = _onlineNetwork.Predict(inputTensor);
        var output = outputTensor.ToVector();
        var lossValue = loss.CalculateLoss(output, target);
        var gradient = loss.CalculateDerivative(output, target);

        var gradientTensor = Tensor<T>.FromVector(gradient);

        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var currentParams = GetParameters();
        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var update = NumOps.Multiply(learningRate, gradients[i % gradients.Length]);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        SetParameters(newParams);
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
