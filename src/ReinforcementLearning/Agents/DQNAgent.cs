using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Agents.DQN;

/// <summary>
/// Deep Q-Network (DQN) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DQN is a landmark algorithm that combined Q-learning with deep neural networks, enabling RL
/// to scale to high-dimensional state spaces. It introduced two key innovations:
/// 1. Experience Replay: Breaks temporal correlations by training on random past experiences
/// 2. Target Network: Provides stable Q-value targets by using a slowly-updating copy
/// </para>
/// <para><b>For Beginners:</b>
/// DQN learns to play games (or solve problems) by learning how valuable each action is in each situation.
/// It uses a neural network to estimate these "Q-values" - essentially, expected future rewards.
///
/// The agent:
/// - Sees the current state (like game screen)
/// - Evaluates each possible action using its Q-network
/// - Picks the action with highest Q-value (with some random exploration)
/// - Learns from past experiences stored in memory
///
/// Famous for: Learning to play Atari games from pixels (DeepMind, 2015)
/// </para>
/// <para><b>Reference:</b>
/// Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
/// </para>
/// </remarks>
public class DQNAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private DQNOptions<T> _dqnOptions;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;

    private NeuralNetwork<T> _qNetwork;
    private NeuralNetwork<T> _targetNetwork;
    private double _epsilon;
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _dqnOptions.StateSize;

    /// <summary>
    /// Initializes a new instance of the DQNAgent class.
    /// </summary>
    /// <param name="options">Configuration options for the DQN agent.</param>
    public DQNAgent(DQNOptions<T> options)
        : base(CreateBaseOptions(options))
    {
        _dqnOptions = options;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
        _epsilon = options.EpsilonStart;
        _steps = 0;

        // Build Q-network
        _qNetwork = BuildQNetwork();

        // Build target network (identical architecture)
        _targetNetwork = BuildQNetwork();

        // Copy initial weights to target network
        CopyNetworkWeights(_qNetwork, _targetNetwork);

        // Register networks with base class
        Networks.Add(_qNetwork);
        Networks.Add(_targetNetwork);
    }


    private static ReinforcementLearningOptions<T> CreateBaseOptions(DQNOptions<T> options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new ReinforcementLearningOptions<T>
        {
            LearningRate = options.LearningRate,
            DiscountFactor = options.DiscountFactor,
            LossFunction = options.LossFunction,
            Seed = options.Seed,
            BatchSize = options.BatchSize,
            ReplayBufferSize = options.ReplayBufferSize,
            TargetUpdateFrequency = options.TargetUpdateFrequency,
            WarmupSteps = options.WarmupSteps,
            EpsilonStart = options.EpsilonStart,
            EpsilonEnd = options.EpsilonEnd,
            EpsilonDecay = options.EpsilonDecay
        };
    }

    private NeuralNetwork<T> BuildQNetwork()
    {
        var layers = new List<ILayer<T>>();

        // Input layer
        int prevSize = _dqnOptions.StateSize;

        // Hidden layers
        foreach (var hiddenSize in _dqnOptions.HiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        // Output layer (Q-values for each action)
        layers.Add(new DenseLayer<T>(prevSize, _dqnOptions.ActionSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        // Create architecture with layers
        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _dqnOptions.StateSize,
            outputSize: _dqnOptions.ActionSize,
            layers: layers
        );

        return new NeuralNetwork<T>(architecture, _dqnOptions.LossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Epsilon-greedy action selection
        if (training && Random.NextDouble() < _epsilon)
        {
            // Random action (exploration)
            int randomAction = Random.Next(_dqnOptions.ActionSize);
            var action = new Vector<T>(_dqnOptions.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        // Greedy action (exploitation)
        var stateTensor = Tensor<T>.FromVector(state);
        var qValuesTensor = _qNetwork.Predict(stateTensor);
        var qValues = qValuesTensor.ToVector();
        int bestAction = ArgMax(qValues);

        var greedyAction = new Vector<T>(_dqnOptions.ActionSize);
        greedyAction[bestAction] = NumOps.One;
        return greedyAction;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        var experience = new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done);
        _replayBuffer.Add(experience);
    }

    /// <inheritdoc/>
    public override T Train()
    {
        _steps++;
        TrainingSteps++;

        // Wait for warmup period
        if (_steps < _dqnOptions.WarmupSteps || !_replayBuffer.CanSample(_dqnOptions.BatchSize))
        {
            return NumOps.Zero;
        }

        // Sample batch from replay buffer
        var batch = _replayBuffer.Sample(_dqnOptions.BatchSize);

        // Compute loss and update Q-network
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Compute target Q-value
            T target;
            if (experience.Done)
            {
                // Terminal state: Q-value is just the reward
                target = experience.Reward;
            }
            else
            {
                // Non-terminal: Q-value = reward + gamma * max(Q(next_state))
                var nextStateTensor = Tensor<T>.FromVector(experience.NextState);
                var nextQValuesTensor = _targetNetwork.Predict(nextStateTensor);
                var nextQValues = nextQValuesTensor.ToVector();
                var maxNextQ = Max(nextQValues);
                target = NumOps.Add(experience.Reward,
                    NumOps.Multiply(DiscountFactor, maxNextQ));
            }

            // Get current Q-value for the action taken
            var stateTensor = Tensor<T>.FromVector(experience.State);
            var currentQValuesTensor = _qNetwork.Predict(stateTensor);
            var currentQValues = currentQValuesTensor.ToVector();
            int actionIndex = ArgMax(experience.Action);

            // Create target Q-values (same as current, except for the action taken)
            var targetQValues = currentQValues.Clone();
            targetQValues[actionIndex] = target;

            // Compute loss
            var loss = LossFunction.CalculateLoss(currentQValues, targetQValues);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backpropagate
            var outputGradients = LossFunction.CalculateDerivative(currentQValues, targetQValues);
            var gradientsTensor = Tensor<T>.FromVector(outputGradients);
            _qNetwork.Backpropagate(gradientsTensor);

            // Extract parameter gradients from network layers (not output-space gradients)
            var parameterGradients = _qNetwork.GetGradients();
            var parameters = _qNetwork.GetParameters();

            for (int i = 0; i < parameters.Length; i++)
            {
                var update = NumOps.Multiply(LearningRate, parameterGradients[i]);
                parameters[i] = NumOps.Subtract(parameters[i], update);
            }

            _qNetwork.UpdateParameters(parameters);
        }

        // Average loss
        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_dqnOptions.BatchSize));
        LossHistory.Add(avgLoss);

        // Update target network periodically
        if (_steps % _dqnOptions.TargetUpdateFrequency == 0)
        {
            CopyNetworkWeights(_qNetwork, _targetNetwork);
        }

        // Decay epsilon
        _epsilon = Math.Max(_dqnOptions.EpsilonEnd, _epsilon * _dqnOptions.EpsilonDecay);

        return avgLoss;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        baseMetrics["Epsilon"] = NumOps.FromDouble(_epsilon);
        baseMetrics["ReplayBufferSize"] = NumOps.FromDouble(_replayBuffer.Count);
        baseMetrics["Steps"] = NumOps.FromDouble(_steps);
        return baseMetrics;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DeepQNetwork,
            FeatureCount = _dqnOptions.StateSize,
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write metadata
        writer.Write(_dqnOptions.StateSize);
        writer.Write(_dqnOptions.ActionSize);
        writer.Write(NumOps.ToDouble(LearningRate));
        writer.Write(NumOps.ToDouble(DiscountFactor));
        writer.Write(_epsilon);
        writer.Write(_steps);

        // Write Q-network
        var qNetworkBytes = _qNetwork.Serialize();
        writer.Write(qNetworkBytes.Length);
        writer.Write(qNetworkBytes);

        // Write target network
        var targetNetworkBytes = _targetNetwork.Serialize();
        writer.Write(targetNetworkBytes.Length);
        writer.Write(targetNetworkBytes);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read metadata
        var stateSize = reader.ReadInt32();
        var actionSize = reader.ReadInt32();
        var learningRate = reader.ReadDouble();
        var discountFactor = reader.ReadDouble();
        _epsilon = reader.ReadDouble();
        _steps = reader.ReadInt32();

        // Read Q-network
        var qNetworkLength = reader.ReadInt32();
        var qNetworkBytes = reader.ReadBytes(qNetworkLength);
        _qNetwork.Deserialize(qNetworkBytes);

        // Read target network
        var targetNetworkLength = reader.ReadInt32();
        var targetNetworkBytes = reader.ReadBytes(targetNetworkLength);
        _targetNetwork.Deserialize(targetNetworkBytes);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return _qNetwork.GetParameters();
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        _qNetwork.UpdateParameters(parameters);
        // Sync target network to match Q-network after parameter update
        CopyNetworkWeights(_qNetwork, _targetNetwork);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clonedOptions = new DQNOptions<T>
        {
            StateSize = _dqnOptions.StateSize,
            ActionSize = _dqnOptions.ActionSize,
            LearningRate = LearningRate,
            DiscountFactor = DiscountFactor,
            LossFunction = LossFunction,
            EpsilonStart = _epsilon,
            EpsilonEnd = _dqnOptions.EpsilonEnd,
            EpsilonDecay = _dqnOptions.EpsilonDecay,
            BatchSize = _dqnOptions.BatchSize,
            ReplayBufferSize = _dqnOptions.ReplayBufferSize,
            TargetUpdateFrequency = _dqnOptions.TargetUpdateFrequency,
            WarmupSteps = _dqnOptions.WarmupSteps,
            HiddenLayers = _dqnOptions.HiddenLayers,
            Seed = _dqnOptions.Seed
        };

        var clone = new DQNAgent<T>(clonedOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? LossFunction;
        var inputTensor = Tensor<T>.FromVector(input);
        var outputTensor = _qNetwork.Predict(inputTensor);
        var output = outputTensor.ToVector();
        var lossValue = loss.CalculateLoss(output, target);
        var gradient = loss.CalculateDerivative(output, target);

        var gradientTensor = Tensor<T>.FromVector(gradient);
        _qNetwork.Backpropagate(gradientTensor);

        return gradient;
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var currentParams = GetParameters();

        // Validate that gradients vector has the correct length (parameter-space, not output-space)
        if (gradients.Length != currentParams.Length)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match parameter vector length ({currentParams.Length}). " +
                $"ApplyGradients expects parameter-space gradients (w.r.t. all network weights), not output-space gradients (w.r.t. network outputs). " +
                $"Use _qNetwork.GetGradients() after backpropagation to obtain parameter-space gradients.",
                nameof(gradients));
        }

        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var update = NumOps.Multiply(learningRate, gradients[i]);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        SetParameters(newParams);
    }

    // Helper methods

    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.UpdateParameters(sourceParams);
    }

    private int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        T maxValue = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.ToDouble(vector[i]) > NumOps.ToDouble(maxValue))
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private T Max(Vector<T> vector)
    {
        T maxValue = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.ToDouble(vector[i]) > NumOps.ToDouble(maxValue))
            {
                maxValue = vector[i];
            }
        }

        return maxValue;
    }
    /// <inheritdoc/>
    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc/>
    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
