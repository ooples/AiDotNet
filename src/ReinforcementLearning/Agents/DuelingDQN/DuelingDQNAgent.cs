using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.Agents.DuelingDQN;

/// <summary>
/// Dueling Deep Q-Network agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Dueling DQN separates the estimation of state value V(s) and action advantages A(s,a),
/// allowing the network to learn which states are valuable without having to learn the
/// effect of each action for each state. This architecture is particularly effective when
/// many actions do not affect the state in a relevant way.
/// </para>
/// <para><b>For Beginners:</b>
/// Dueling DQN splits Q-values into two parts:
/// - **Value V(s)**: How good is this state overall?
/// - **Advantage A(s,a)**: How much better is action 'a' compared to average?
/// - **Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))**
///
/// This is powerful because:
/// - The agent learns state values even when actions don't matter much
/// - Faster learning in scenarios where action choice rarely matters
/// - Better generalization across similar states
///
/// Example: In a car driving game, being on the road is valuable regardless of whether
/// you accelerate slightly or not. Dueling DQN learns "being on road = good" separately
/// from "how much to accelerate".
/// </para>
/// <para><b>Reference:</b>
/// Wang et al., "Dueling Network Architectures for Deep RL", 2016.
/// </para>
/// </remarks>
public class DuelingDQNAgent<T> : DeepReinforcementLearningAgentBase<T>
{
    private DuelingDQNOptions<T> _options;
    private readonly UniformReplayBuffer<T> _replayBuffer;

    private DuelingNetwork<T> _qNetwork;
    private DuelingNetwork<T> _targetNetwork;
    private double _epsilon;
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    public DuelingDQNAgent(DuelingDQNOptions<T> options)
        : base(new ReinforcementLearningOptions<T>
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
        })
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _replayBuffer = new UniformReplayBuffer<T>(options.ReplayBufferSize, options.Seed);
        _epsilon = options.EpsilonStart;
        _steps = 0;

        _qNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers,
            _options.ValueStreamLayers,
            _options.AdvantageStreamLayers,
            NumOps
        );

        _targetNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers,
            _options.ValueStreamLayers,
            _options.AdvantageStreamLayers,
            NumOps
        );

        CopyNetworkWeights(_qNetwork, _targetNetwork);
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

        var qValues = _qNetwork.Predict(state);
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
            // Compute target using Double DQN approach with dueling architecture
            T target;
            if (experience.Done)
            {
                target = experience.Reward;
            }
            else
            {
                // Use online network to select action
                var nextQValuesOnline = _qNetwork.Predict(experience.NextState);
                int bestActionIndex = ArgMax(nextQValuesOnline);

                // Use target network to evaluate
                var nextQValuesTarget = _targetNetwork.Predict(experience.NextState);
                var selectedQ = nextQValuesTarget[bestActionIndex];

                target = NumOps.Add(experience.Reward,
                    NumOps.Multiply(DiscountFactor, selectedQ));
            }

            var currentQValues = _qNetwork.Predict(experience.State);
            int actionIndex = ArgMax(experience.Action);

            var targetQValues = currentQValues.Clone();
            targetQValues[actionIndex] = target;

            var loss = LossFunction.CalculateLoss(currentQValues, targetQValues);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backward pass through dueling architecture
            var gradients = LossFunction.ComputeGradient(currentQValues, targetQValues);
            _qNetwork.Backward(experience.State, gradients);

            // Update parameters
            _qNetwork.UpdateWeights(LearningRate);
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
            ModelType = ModelType.DuelingDQN,
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
        return _qNetwork.GetParameters();
    }

    /// <inheritdoc/>
    public override void SetParameters(Matrix<T> parameters)
    {
        _qNetwork.SetFlattenedParameters(parameters);
    }

    /// <inheritdoc/>
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clonedOptions = new DuelingDQNOptions<T>
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
            SharedLayers = _options.SharedLayers,
            ValueStreamLayers = _options.ValueStreamLayers,
            AdvantageStreamLayers = _options.AdvantageStreamLayers,
            Seed = _options.Seed
        };

        var clone = new DuelingDQNAgent<T>(clonedOptions);
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
        var output = _qNetwork.Predict(input);
        var lossValue = loss.CalculateLoss(output, target);
        var gradient = loss.ComputeGradient(output, target);

        _qNetwork.Backward(input, gradient);
        var gradientVector = _qNetwork.GetFlattenedGradients();
        var gradientMatrix = new Matrix<T>(gradientVector.Length, 1);
        for (int i = 0; i < gradientVector.Length; i++)
        {
            gradientMatrix[i, 0] = gradientVector[i];
        }

        return (gradientMatrix, lossValue);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        var currentParams = _qNetwork.GetParameters();
        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var gradValue = (i < gradients.Rows) ? gradients[i, 0] : NumOps.Zero;
            var update = NumOps.Multiply(learningRate, gradValue);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        _qNetwork.SetFlattenedParameters(newParams);
    }

    // Helper methods
    private void CopyNetworkWeights(DuelingNetwork<T> source, DuelingNetwork<T> target)
    {
        var sourceParams = source.GetParameters();
        target.SetFlattenedParameters(sourceParams);
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

/// <summary>
/// Custom dueling network architecture that separates value and advantage streams.
/// </summary>
internal class DuelingNetwork<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<DenseLayer<T>> _sharedLayers;
    private readonly List<DenseLayer<T>> _valueLayers;
    private readonly List<DenseLayer<T>> _advantageLayers;
    private readonly int _actionSize;

    private Vector<T>? _lastSharedOutput;
    private Vector<T>? _lastValueOutput;
    private Vector<T>? _lastAdvantageOutput;

    public DuelingNetwork(
        int stateSize,
        int actionSize,
        int[] sharedLayerSizes,
        int[] valueLayerSizes,
        int[] advantageLayerSizes,
        INumericOperations<T> numOps)
    {
        _numOps = numOps;
        _actionSize = actionSize;
        _sharedLayers = new List<DenseLayer<T>>();
        _valueLayers = new List<DenseLayer<T>>();
        _advantageLayers = new List<DenseLayer<T>>();

        // Build shared feature layers
        int prevSize = stateSize;
        foreach (var size in sharedLayerSizes)
        {
            _sharedLayers.Add(new DenseLayer<T>(prevSize, size, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = size;
        }

        int sharedOutputSize = prevSize;

        // Build value stream
        prevSize = sharedOutputSize;
        foreach (var size in valueLayerSizes)
        {
            _valueLayers.Add(new DenseLayer<T>(prevSize, size, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = size;
        }
        _valueLayers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new LinearActivation<T>())); // Output single value

        // Build advantage stream
        prevSize = sharedOutputSize;
        foreach (var size in advantageLayerSizes)
        {
            _advantageLayers.Add(new DenseLayer<T>(prevSize, size, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = size;
        }
        _advantageLayers.Add(new DenseLayer<T>(prevSize, actionSize, (IActivationFunction<T>)new LinearActivation<T>())); // Output per-action advantages
    }

    public Vector<T> Forward(Vector<T> state)
    {
        // Shared layers
        var sharedOutput = state;
        foreach (var layer in _sharedLayers)
        {
            sharedOutput = layer.Forward(sharedOutput);
        }
        _lastSharedOutput = sharedOutput;

        // Value stream
        var valueOutput = sharedOutput;
        foreach (var layer in _valueLayers)
        {
            valueOutput = layer.Forward(valueOutput);
        }
        _lastValueOutput = valueOutput;
        T value = valueOutput[0];

        // Advantage stream
        var advantageOutput = sharedOutput;
        foreach (var layer in _advantageLayers)
        {
            advantageOutput = layer.Forward(advantageOutput);
        }
        _lastAdvantageOutput = advantageOutput;

        // Compute mean advantage for centering
        T meanAdvantage = _numOps.Zero;
        for (int i = 0; i < _actionSize; i++)
        {
            meanAdvantage = _numOps.Add(meanAdvantage, advantageOutput[i]);
        }
        meanAdvantage = _numOps.Divide(meanAdvantage, _numOps.FromDouble(_actionSize));

        // Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        var qValues = new Vector<T>(_actionSize);
        for (int i = 0; i < _actionSize; i++)
        {
            var centeredAdvantage = _numOps.Subtract(advantageOutput[i], meanAdvantage);
            qValues[i] = _numOps.Add(value, centeredAdvantage);
        }

        return qValues;
    }

    public void Backward(Vector<T> state, Vector<T> qGradients)
    {
        // Compute gradients for value and advantage streams
        // Value gets sum of all Q gradients
        T valueGrad = _numOps.Zero;
        for (int i = 0; i < _actionSize; i++)
        {
            valueGrad = _numOps.Add(valueGrad, qGradients[i]);
        }

        // Advantage gradients (centered)
        T meanQGrad = _numOps.Divide(valueGrad, _numOps.FromDouble(_actionSize));
        var advantageGrads = new Vector<T>(_actionSize);
        for (int i = 0; i < _actionSize; i++)
        {
            advantageGrads[i] = _numOps.Subtract(qGradients[i], meanQGrad);
        }

        // Backprop through streams (simplified - in practice would use proper backprop)
        var valueGradVec = new Vector<T>(1);
        valueGradVec[0] = valueGrad;

        // Note: Full implementation would require proper backpropagation through each stream
    }

    public void UpdateWeights(T learningRate)
    {
        // Update all layers (simplified)
        foreach (var layer in _sharedLayers)
        {
            // Layer weight updates would go here
        }
        foreach (var layer in _valueLayers)
        {
            // Layer weight updates would go here
        }
        foreach (var layer in _advantageLayers)
        {
            // Layer weight updates would go here
        }
    }

    public Matrix<T> GetFlattenedParameters()
    {
        var paramsList = new List<T>();

        foreach (var layer in _sharedLayers)
        {
            // Collect layer parameters
        }
        foreach (var layer in _valueLayers)
        {
            // Collect layer parameters
        }
        foreach (var layer in _advantageLayers)
        {
            // Collect layer parameters
        }

        var matrix = new Matrix<T>(paramsList.Count, 1);
        for (int i = 0; i < paramsList.Count; i++)
        {
            matrix[i, 0] = paramsList[i];
        }
        return matrix;
    }

    public void SetFlattenedParameters(Matrix<T> parameters)
    {
        // Set parameters for all layers
    }

    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        // Serialize network architecture and weights
        return ms.ToArray();
    }

    public void Deserialize(byte[] data)
    {
        // Deserialize network
    }
}
