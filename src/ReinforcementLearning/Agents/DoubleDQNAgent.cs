using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;

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
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
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
            layers.Add(new DenseLayer<T>(prevSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = hiddenSize;
        }

        layers.Add(new DenseLayer<T>(prevSize, _options.ActionSize, (IActivationFunction<T>)new IdentityActivation<T>()));

        var architecture = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: _options.StateSize,
            outputSize: _options.ActionSize,
            layers: layers
        );

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

        var stateTensor = Tensor<T>.FromVector(state);
        var qValuesTensor = _qNetwork.Predict(stateTensor);
        var qValues = qValuesTensor.ToVector();
        int bestAction = ArgMax(qValues);

        var greedyAction = new Vector<T>(_options.ActionSize);
        greedyAction[bestAction] = NumOps.One;
        return greedyAction;
    }

    /// <inheritdoc/>
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        _replayBuffer.Add(new Experience<T, Vector<T>, Vector<T>>(state, action, reward, nextState, done));
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
                var nextStateTensor = Tensor<T>.FromVector(experience.NextState);
                var nextQValuesOnlineTensor = _qNetwork.Predict(nextStateTensor);
                var nextQValuesOnline = nextQValuesOnlineTensor.ToVector();
                int bestActionIndex = ArgMax(nextQValuesOnline);

                // Use target network to evaluate that action
                var nextQValuesTargetTensor = _targetNetwork.Predict(nextStateTensor);
                var nextQValuesTarget = nextQValuesTargetTensor.ToVector();
                var selectedQ = nextQValuesTarget[bestActionIndex];

                target = NumOps.Add(experience.Reward,
                    NumOps.Multiply(DiscountFactor, selectedQ));
            }

            var stateTensor = Tensor<T>.FromVector(experience.State);
            var currentQValuesTensor = _qNetwork.Predict(stateTensor);
            var currentQValues = currentQValuesTensor.ToVector();
            int actionIndex = ArgMax(experience.Action);

            var targetQValues = currentQValues.Clone();
            targetQValues[actionIndex] = target;

            var loss = LossFunction.CalculateLoss(currentQValues, targetQValues);
            totalLoss = NumOps.Add(totalLoss, loss);

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
    public override Vector<T> GetParameters()
    {
        return _qNetwork.GetParameters();
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        _qNetwork.UpdateParameters(parameters);
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

    /// <summary>
    /// Not supported for DoubleDQNAgent. Use the agent's internal Train() loop instead.
    /// </summary>
    /// <param name="gradients">Not used.</param>
    /// <param name="learningRate">Not used.</param>
    /// <exception cref="NotSupportedException">
    /// Always thrown. DoubleDQN manages gradient computation and parameter updates internally through backpropagation.
    /// </exception>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        throw new NotSupportedException(
            "ApplyGradients is not supported for DoubleDQNAgent; use the agent's internal Train() loop. " +
            "DoubleDQN manages gradient computation and parameter updates internally through backpropagation.");
    }

    // Helper methods
    private void CopyNetworkWeights(NeuralNetwork<T> source, NeuralNetwork<T> target)
    {
        target.UpdateParameters(source.GetParameters());
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
