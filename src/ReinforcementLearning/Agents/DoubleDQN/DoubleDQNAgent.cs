using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Activations;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

namespace AiDotNet.ReinforcementLearning.Agents.DoubleDQN;

/// <summary>
/// Double Deep Q-Network (Double DQN) agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Double DQN addresses the overestimation bias in standard DQN by decoupling action
/// selection from action evaluation. It uses the online network to select actions and
/// the target network to evaluate them, leading to more accurate Q-value estimates.
/// </para>
/// <para><b>For Beginners:</b>
/// Standard DQN tends to overestimate Q-values because it uses the same network to both
/// select and evaluate actions (max operator causes positive bias).
///
/// Double DQN fixes this by:
/// - Using online network to SELECT the best action
/// - Using target network to EVALUATE that action's value
///
/// Think of it like getting a second opinion: one expert picks what looks best,
/// another expert judges its actual value. This reduces overoptimistic estimates.
///
/// **Key Improvement**: More stable learning, better performance, especially when
/// there's noise or stochasticity in the environment.
/// </para>
/// <para><b>Reference:</b>
/// van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", 2015.
/// </para>
/// </remarks>
public class DoubleDQNAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private DoubleDQNOptions<T> _options;
    private readonly UniformReplayBuffer<T> _replayBuffer;

    private NeuralNetwork<T> _qNetwork;
    private NeuralNetwork<T> _targetNetwork;
    private double _epsilon;
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Initializes a new instance of the DoubleDQNAgent class.
    /// </summary>
    /// <param name="options">Configuration options for the Double DQN agent.</param>
    public DoubleDQNAgent(DoubleDQNOptions<T> options)
        : base(CreateBaseOptions(options))
    {
        _options = options;
        _replayBuffer = new UniformReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
        _epsilon = options.EpsilonStart;
        _steps = 0;

        _qNetwork = BuildQNetwork();
        _targetNetwork = BuildQNetwork();
        CopyNetworkWeights(_qNetwork, _targetNetwork);

        Networks.Add(_qNetwork);
        Networks.Add(_targetNetwork);
    }


    private static ReinforcementLearningOptions<T> CreateBaseOptions(DoubleDQNOptions<T> options)
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
        int prevSize = _options.StateSize;

        foreach (var hiddenSize in _options.HiddenLayers)
        {
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        layers.Add(new DenseLayer<T>(prevSize, _options.ActionSize, new LinearActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>
        {
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            Layers = layers,
            TaskType = TaskType.Regression
        };

        return new NeuralNetwork<T>(architecture, _options.LossFunction);
    }

    /// <inheritdoc/>
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        if (training && Random.NextDouble() < _epsilon)
        {
            int randomAction = Random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        var qValues = _qNetwork.Forward(state);
        int bestAction = ArgMax(qValues);

        var greedyAction = new Vector<T>(_options.ActionSize);
        greedyAction[bestAction] = NumOps.One;
        return greedyAction;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T>(state, action, reward, nextState, done));
    }

    /// <inheritdoc/>
    public override T Train()
    {
        _steps++;
        TrainingSteps++;

        if (_steps < _options.WarmupSteps || !_replayBuffer.CanSample(_options.BatchSize))
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(_options.BatchSize);
        T totalLoss = NumOps.Zero;

        foreach (var experience in batch)
        {
            // Double DQN: Use online network to SELECT action, target network to EVALUATE
            T target;
            if (experience.Done)
            {
                target = experience.Reward;
            }
            else
            {
                // Key difference from DQN: Use online network to select best action
                var nextQValuesOnline = _qNetwork.Forward(experience.NextState);
                int bestActionIndex = ArgMax(nextQValuesOnline);

                // Use target network to evaluate that action
                var nextQValuesTarget = _targetNetwork.Forward(experience.NextState);
                var selectedQ = nextQValuesTarget[bestActionIndex];

                target = NumOps.Add(experience.Reward,
                    NumOps.Multiply(DiscountFactor, selectedQ));
            }

            var currentQValues = _qNetwork.Forward(experience.State);
            int actionIndex = ArgMax(experience.Action);

            var targetQValues = currentQValues.Clone();
            targetQValues[actionIndex] = target;

            var loss = LossFunction.ComputeLoss(currentQValues, targetQValues);
            totalLoss = NumOps.Add(totalLoss, loss);

            var gradients = LossFunction.ComputeGradient(currentQValues, targetQValues);
            _qNetwork.Backward(gradients);

            var parameters = _qNetwork.GetFlattenedParameters();
            var gradientVector = _qNetwork.GetFlattenedGradients();

            for (int i = 0; i < parameters.Length; i++)
            {
                var update = NumOps.Multiply(LearningRate, gradientVector[i]);
                parameters[i] = NumOps.Subtract(parameters[i], update);
            }

            _qNetwork.UpdateParameters(parameters);
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_options.BatchSize));
        LossHistory.Add(avgLoss);

        if (_steps % _options.TargetUpdateFrequency == 0)
        {
            CopyNetworkWeights(_qNetwork, _targetNetwork);
        }

        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);

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
            ModelType = ModelType.DoubleDQN,
            FeatureCount = _options.StateSize,
            TrainingSampleCount = _replayBuffer.Count,
            Parameters = GetParameters()
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(_options.StateSize);
        writer.Write(_options.ActionSize);
        writer.Write(NumOps.ToDouble(LearningRate));
        writer.Write(NumOps.ToDouble(DiscountFactor));
        writer.Write(_epsilon);
        writer.Write(_steps);

        var qNetworkBytes = _qNetwork.Serialize();
        writer.Write(qNetworkBytes.Length);
        writer.Write(qNetworkBytes);

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

        reader.ReadInt32(); // stateSize
        reader.ReadInt32(); // actionSize
        reader.ReadDouble(); // learningRate
        reader.ReadDouble(); // discountFactor
        _epsilon = reader.ReadDouble();
        _steps = reader.ReadInt32();

        var qNetworkLength = reader.ReadInt32();
        var qNetworkBytes = reader.ReadBytes(qNetworkLength);
        _qNetwork.Deserialize(qNetworkBytes);

        var targetNetworkLength = reader.ReadInt32();
        var targetNetworkBytes = reader.ReadBytes(targetNetworkLength);
        _targetNetwork.Deserialize(targetNetworkBytes);
    }

    /// <inheritdoc/>
    public override Matrix<T> GetParameters()
    {
        var qNetworkParams = _qNetwork.GetFlattenedParameters();
        var matrix = new Matrix<T>(qNetworkParams.Length, 1);
        for (int i = 0; i < qNetworkParams.Length; i++)
        {
            matrix[i, 0] = qNetworkParams[i];
        }
        return matrix;
    }

    /// <inheritdoc/>
    public override void SetParameters(Matrix<T> parameters)
    {
        var vector = new Vector<T>(parameters.Rows);
        for (int i = 0; i < parameters.Rows; i++)
        {
            vector[i] = parameters[i, 0];
        }
        _qNetwork.UpdateParameters(vector);
        CopyNetworkWeights(_qNetwork, _targetNetwork);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clonedOptions = new DoubleDQNOptions<T>
        {
            StateSize = _options.StateSize,
            ActionSize = _options.ActionSize,
            LearningRate = LearningRate,
            DiscountFactor = DiscountFactor,
            LossFunction = LossFunction,
            EpsilonStart = _epsilon,
            EpsilonEnd = _options.EpsilonEnd,
            EpsilonDecay = _options.EpsilonDecay,
            BatchSize = _options.BatchSize,
            ReplayBufferSize = _options.ReplayBufferSize,
            TargetUpdateFrequency = _options.TargetUpdateFrequency,
            WarmupSteps = _options.WarmupSteps,
            HiddenLayers = _options.HiddenLayers,
            Seed = _options.Seed
        };

        var clone = new DoubleDQNAgent<T>(clonedOptions);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc/>
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? LossFunction;
        var output = _qNetwork.Forward(input);
        var lossValue = loss.ComputeLoss(output, target);
        var gradient = loss.ComputeGradient(output, target);

        _qNetwork.Backward(gradient);
        var gradients = GetParameters();

        return (gradients, lossValue);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        var currentParams = GetParameters();
        var newParams = new Matrix<T>(currentParams.Rows, 1);

        for (int i = 0; i < currentParams.Rows; i++)
        {
            var update = NumOps.Multiply(learningRate, gradients[i, 0]);
            newParams[i, 0] = NumOps.Subtract(currentParams[i, 0], update);
        }

        SetParameters(newParams);
    }

    // Helper methods
    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        target.UpdateParameters(source.GetFlattenedParameters());
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
}
