using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;

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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;

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
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
        _epsilon = options.EpsilonStart;
        _steps = 0;

        _qNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers.ToArray(),
            _options.ValueStreamLayers.ToArray(),
            _options.AdvantageStreamLayers.ToArray(),
            NumOps
        );

        _targetNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers.ToArray(),
            _options.ValueStreamLayers.ToArray(),
            _options.AdvantageStreamLayers.ToArray(),
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

        var qValues = _qNetwork.Forward(state);
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
            // Compute target using Double DQN approach with dueling architecture
            T target;
            if (experience.Done)
            {
                target = experience.Reward;
            }
            else
            {
                // Use online network to select action
                var nextQValuesOnline = _qNetwork.Forward(experience.NextState);
                int bestActionIndex = ArgMax(nextQValuesOnline);

                // Use target network to evaluate
                var nextQValuesTarget = _targetNetwork.Forward(experience.NextState);
                var selectedQ = nextQValuesTarget[bestActionIndex];

                target = NumOps.Add(experience.Reward,
                    NumOps.Multiply(DiscountFactor, selectedQ));
            }

            var currentQValues = _qNetwork.Forward(experience.State);
            int actionIndex = ArgMax(experience.Action);

            var targetQValues = currentQValues.Clone();
            targetQValues[actionIndex] = target;

            var loss = LossFunction.CalculateLoss(currentQValues, targetQValues);
            totalLoss = NumOps.Add(totalLoss, loss);

            // Backward pass through dueling architecture
            var gradients = LossFunction.CalculateDerivative(currentQValues, targetQValues);
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
        var flatParams = _qNetwork.GetFlattenedParameters();
        var vector = new Vector<T>(flatParams.Rows);
        for (int i = 0; i < flatParams.Rows; i++)
            vector[i] = flatParams[i, 0];
        return vector;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var matrix = new Matrix<T>(parameters.Length, 1);
        for (int i = 0; i < parameters.Length; i++)
            matrix[i, 0] = parameters[i];
        _qNetwork.SetFlattenedParameters(matrix);
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
    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            "ComputeGradients is not supported for DuelingDQNAgent; " +
            "use the agent's internal Train() loop or expose layer gradients. " +
            "DuelingNetwork stores gradients internally but does not expose them.");
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var flatParams = _qNetwork.GetFlattenedParameters();
        var currentParams = new Vector<T>(flatParams.Rows);
        for (int i = 0; i < flatParams.Rows; i++)
            currentParams[i] = flatParams[i, 0];

        var newParams = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var gradValue = (i < gradients.Length) ? gradients[i] : NumOps.Zero;
            var update = NumOps.Multiply(learningRate, gradValue);
            newParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        var matrix = new Matrix<T>(newParams.Length, 1);
        for (int i = 0; i < newParams.Length; i++)
            matrix[i, 0] = newParams[i];
        _qNetwork.SetFlattenedParameters(matrix);
    }

    // Helper methods
    private void CopyNetworkWeights(DuelingNetwork<T> source, DuelingNetwork<T> target)
    {
        var sourceParams = source.GetFlattenedParameters();
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
        _valueLayers.Add(new DenseLayer<T>(prevSize, 1, (IActivationFunction<T>)new IdentityActivation<T>())); // Output single value

        // Build advantage stream
        prevSize = sharedOutputSize;
        foreach (var size in advantageLayerSizes)
        {
            _advantageLayers.Add(new DenseLayer<T>(prevSize, size, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = size;
        }
        _advantageLayers.Add(new DenseLayer<T>(prevSize, actionSize, (IActivationFunction<T>)new IdentityActivation<T>())); // Output per-action advantages
    }

    public Vector<T> Forward(Vector<T> state)
    {
        // Shared layers
        var sharedTensor = Tensor<T>.FromVector(state);
        foreach (var layer in _sharedLayers)
        {
            sharedTensor = layer.Forward(sharedTensor);
        }
        var sharedOutput = sharedTensor.ToVector();
        _lastSharedOutput = sharedOutput;

        // Value stream
        var valueTensor = sharedTensor;
        foreach (var layer in _valueLayers)
        {
            valueTensor = layer.Forward(valueTensor);
        }
        var valueOutput = valueTensor.ToVector();
        _lastValueOutput = valueOutput;
        T value = valueOutput[0];

        // Advantage stream
        var advantageTensor = sharedTensor;
        foreach (var layer in _advantageLayers)
        {
            advantageTensor = layer.Forward(advantageTensor);
        }
        var advantageOutput = advantageTensor.ToVector();
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

    /// <summary>
    /// Predicts Q-values for the given state.
    /// </summary>
    /// <param name="input">The input state vector.</param>
    /// <returns>Q-values for all actions.</returns>
    /// <remarks>
    /// This method is an alias for Forward and is provided for interface compatibility.
    /// </remarks>
    public Vector<T> Predict(Vector<T> input)
    {
        return Forward(input);
    }

    public void Backward(Vector<T> state, Vector<T> qGradients)
    {
        // Compute gradients for value and advantage streams
        // Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        // dQ/dV = 1 for all actions
        // dQ/dA_i = 1 - 1/n (where n is action count)

        // Value gradient: sum of all Q gradients
        T valueGrad = _numOps.Zero;
        for (int i = 0; i < _actionSize; i++)
        {
            valueGrad = _numOps.Add(valueGrad, qGradients[i]);
        }

        // Advantage gradients (centered due to mean subtraction)
        T meanQGrad = _numOps.Divide(valueGrad, _numOps.FromDouble(_actionSize));
        var advantageGrads = new Vector<T>(_actionSize);
        for (int i = 0; i < _actionSize; i++)
        {
            advantageGrads[i] = _numOps.Subtract(qGradients[i], meanQGrad);
        }

        // Backprop through advantage stream
        var advantageTensor = Tensor<T>.FromVector(advantageGrads);
        for (int i = _advantageLayers.Count - 1; i >= 0; i--)
        {
            advantageTensor = _advantageLayers[i].Backward(advantageTensor);
        }

        // Backprop through value stream
        var valueGradVec = new Vector<T>(1);
        valueGradVec[0] = valueGrad;
        var valueTensor = Tensor<T>.FromVector(valueGradVec);
        for (int i = _valueLayers.Count - 1; i >= 0; i--)
        {
            valueTensor = _valueLayers[i].Backward(valueTensor);
        }

        // Both streams converge to shared layers, so we need to sum gradients
        // The gradients from both streams need to be added together for shared layers
        var sharedGradientFromAdvantage = advantageTensor.ToVector();
        var sharedGradientFromValue = valueTensor.ToVector();

        // Combine gradients from both streams
        var combinedSharedGrad = new Vector<T>(sharedGradientFromAdvantage.Length);
        for (int i = 0; i < combinedSharedGrad.Length; i++)
        {
            combinedSharedGrad[i] = _numOps.Add(sharedGradientFromAdvantage[i], sharedGradientFromValue[i]);
        }

        // Backprop through shared layers
        var sharedTensor = Tensor<T>.FromVector(combinedSharedGrad);
        for (int i = _sharedLayers.Count - 1; i >= 0; i--)
        {
            sharedTensor = _sharedLayers[i].Backward(sharedTensor);
        }
    }

    public void UpdateWeights(T learningRate)
    {
        // Update shared layers using UpdateParameters method
        foreach (var layer in _sharedLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        // Update value stream layers
        foreach (var layer in _valueLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        // Update advantage stream layers
        foreach (var layer in _advantageLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    public Matrix<T> GetFlattenedParameters()
    {
        var paramsList = new List<T>();

        // Collect parameters from shared layers
        foreach (var layer in _sharedLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                paramsList.Add(layerParams[i]);
            }
        }

        // Collect parameters from value stream layers
        foreach (var layer in _valueLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                paramsList.Add(layerParams[i]);
            }
        }

        // Collect parameters from advantage stream layers
        foreach (var layer in _advantageLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                paramsList.Add(layerParams[i]);
            }
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
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        int offset = 0;

        // Set parameters for shared layers
        foreach (var layer in _sharedLayers)
        {
            int paramCount = layer.ParameterCount;
            var layerParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                layerParams[i] = parameters[offset++, 0];
            }
            layer.SetParameters(layerParams);
        }

        // Set parameters for value stream layers
        foreach (var layer in _valueLayers)
        {
            int paramCount = layer.ParameterCount;
            var layerParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                layerParams[i] = parameters[offset++, 0];
            }
            layer.SetParameters(layerParams);
        }

        // Set parameters for advantage stream layers
        foreach (var layer in _advantageLayers)
        {
            int paramCount = layer.ParameterCount;
            var layerParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                layerParams[i] = parameters[offset++, 0];
            }
            layer.SetParameters(layerParams);
        }

        // Validate that we consumed exactly the right number of parameters
        if (offset != parameters.Rows)
        {
            throw new ArgumentException($"Parameter count mismatch: expected {offset}, got {parameters.Rows}");
        }
    }

    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write architecture information
        writer.Write(_actionSize);
        writer.Write(_sharedLayers.Count);
        writer.Write(_valueLayers.Count);
        writer.Write(_advantageLayers.Count);

        // Write layer sizes for shared layers
        foreach (var layer in _sharedLayers)
        {
            writer.Write(layer.GetInputShape()[0]);
            writer.Write(layer.GetOutputShape()[0]);
        }

        // Write layer sizes for value layers
        foreach (var layer in _valueLayers)
        {
            writer.Write(layer.GetInputShape()[0]);
            writer.Write(layer.GetOutputShape()[0]);
        }

        // Write layer sizes for advantage layers
        foreach (var layer in _advantageLayers)
        {
            writer.Write(layer.GetInputShape()[0]);
            writer.Write(layer.GetOutputShape()[0]);
        }

        // Serialize parameters
        var parameters = GetFlattenedParameters();
        writer.Write(parameters.Rows);
        for (int i = 0; i < parameters.Rows; i++)
        {
            writer.Write(_numOps.ToDouble(parameters[i, 0]));
        }

        return ms.ToArray();
    }

    public void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read architecture information (for validation)
        int actionSize = reader.ReadInt32();
        int sharedLayersCount = reader.ReadInt32();
        int valueLayersCount = reader.ReadInt32();
        int advantageLayersCount = reader.ReadInt32();

        // Validate architecture matches
        if (actionSize != _actionSize ||
            sharedLayersCount != _sharedLayers.Count ||
            valueLayersCount != _valueLayers.Count ||
            advantageLayersCount != _advantageLayers.Count)
        {
            throw new InvalidOperationException("Network architecture mismatch during deserialization");
        }

        // Skip layer size information (already validated by layer counts)
        for (int i = 0; i < sharedLayersCount; i++)
        {
            reader.ReadInt32(); // inputSize
            reader.ReadInt32(); // outputSize
        }
        for (int i = 0; i < valueLayersCount; i++)
        {
            reader.ReadInt32();
            reader.ReadInt32();
        }
        for (int i = 0; i < advantageLayersCount; i++)
        {
            reader.ReadInt32();
            reader.ReadInt32();
        }

        // Deserialize parameters
        int paramCount = reader.ReadInt32();
        var parameters = new Matrix<T>(paramCount, 1);
        for (int i = 0; i < paramCount; i++)
        {
            parameters[i, 0] = _numOps.FromDouble(reader.ReadDouble());
        }

        SetFlattenedParameters(parameters);
    }
}
