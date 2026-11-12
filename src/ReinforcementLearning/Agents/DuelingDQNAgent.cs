using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Validation;

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
/// <example>
/// <code>
/// // Create a Dueling DQN agent that separates state value from action advantages
/// var options = new DuelingDQNOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, LearningRate = 0.001 };
/// var agent = new DuelingDQNAgent&lt;double&gt;(options);
///
/// // Select an action for the current state
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
[ResearchPaper("Dueling Network Architectures for Deep Reinforcement Learning",
    "https://arxiv.org/abs/1511.06581",
    Year = 2016,
    Authors = "Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N.")]
public partial class DuelingDQNAgent<T> : DeepReinforcementLearningAgentBase<T>, IActionValueProvider<T>
{

    /// <inheritdoc />
    /// <remarks>The online dueling Q-network only. Its target network is a periodically-synced copy, and the hand-written surface excluded it -- including it now would change the vector length and invalidate every existing checkpoint. The flatten/unflatten plumbing that used to live here moved onto DuelingNetwork itself, which implements IParameterSource over the SAME shared -> value -> advantage layer order its GetFlattenedParameters always used.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_qNetwork);
    }
    private DuelingDQNOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly UniformReplayBuffer<T, Vector<T>, Vector<T>> _replayBuffer;

    private DuelingNetwork<T> _qNetwork;
    [Buffer]
    private DuelingNetwork<T> _targetNetwork;
    private double _epsilon;
    private int _steps;

    /// <inheritdoc/>
    public override int FeatureCount => _options.StateSize;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public DuelingDQNAgent()
        : this(new DuelingDQNOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

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
        Guard.NotNull(options);
        _options = options;
        _replayBuffer = new UniformReplayBuffer<T, Vector<T>, Vector<T>>(options.ReplayBufferSize, options.Seed);
        _epsilon = options.EpsilonStart;
        _steps = 0;

        _qNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers.ToArray(),
            _options.ValueStreamLayers.ToArray(),
            _options.AdvantageStreamLayers.ToArray(),
            NumOps,
            NextNetworkInitializationSeed()
        );

        _targetNetwork = new DuelingNetwork<T>(
            _options.StateSize,
            _options.ActionSize,
            _options.SharedLayers.ToArray(),
            _options.ValueStreamLayers.ToArray(),
            _options.AdvantageStreamLayers.ToArray(),
            NumOps,
            NextNetworkInitializationSeed()
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
    Vector<T> IActionValueProvider<T>.GetActionValues(Vector<T> state) => _qNetwork.Predict(state);

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

        // A supervised one-shot Train(state, target) call bypasses the autonomous-exploration warmup
        // and trains on the samples gathered so far (clamped to the buffer); autonomous stepping still
        // respects warmup.
        int effectiveBatchSize = SupervisedUpdateRequested
            ? System.Math.Min(_options.BatchSize, _replayBuffer.Count)
            : _options.BatchSize;
        if ((!SupervisedUpdateRequested && _steps < _options.WarmupSteps)
            || effectiveBatchSize <= 0
            || !_replayBuffer.CanSample(effectiveBatchSize))
        {
            return NumOps.Zero;
        }

        var batch = _replayBuffer.Sample(effectiveBatchSize);
        // Each sampled transition takes its own update, in order (the online network changes between them).
        var losses = batch.Select(TrainOnExperience).ToList();
        T totalLoss = AiDotNetEngine.Current.Sum(Vector<T>.FromEnumerable(losses));

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
            FeatureCount = _options.StateSize,
        };
    }



    // Helper methods
    private void CopyNetworkWeights(DuelingNetwork<T> source, DuelingNetwork<T> target)
    {
        source.EnsureInitialized();
        target.EnsureInitialized();
        var sourceParams = source.GetFlattenedParameters();
        target.SetFlattenedParameters(sourceParams);
    }

    /// <summary>One transition's Double-DQN update through the dueling network; returns its loss.</summary>
    private T TrainOnExperience(Experience<T, Vector<T>, Vector<T>> experience)
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

        var currentQValues = _qNetwork.Forward(experience.State);
        int actionIndex = ArgMax(experience.Action);

        var targetQValues = currentQValues.Clone();
        targetQValues[actionIndex] = target;

        var loss = LossFunction.CalculateLoss(currentQValues, targetQValues);

        // Backward pass through dueling architecture
        var gradients = LossFunction.ComputeGradient(currentQValues, targetQValues);

        // Update parameters
        _qNetwork.UpdateWeights(gradients, LearningRate);

        return loss;
    }

    private int ArgMax(Vector<T> vector)
    {
        return RLTensorOps.ArgMax(vector);
    }
    /// <inheritdoc/>
    public override void SaveModel(string filepath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            var data = Serialize();
            System.IO.File.WriteAllBytes(filepath, data);
        }
    }

    /// <inheritdoc/>
    public override void LoadModel(string filepath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            var data = System.IO.File.ReadAllBytes(filepath);
            Deserialize(data);
        }
    }

    /// <summary>
    /// Disposes the agent and both dueling networks it owns.
    /// </summary>
    /// <remarks>
    /// The online and target networks are <see cref="DuelingNetwork{T}"/> instances rather than
    /// <see cref="INeuralNetwork{T}"/>, so they cannot be registered in <c>Networks</c> and the base class
    /// never saw them: disposing this agent used to release none of their layers.
    /// </remarks>
    /// <exception cref="AggregateException">Both networks threw from <c>Dispose</c>; a single failure is rethrown unchanged.</exception>
    public override void Dispose()
    {
        try
        {
            Helpers.DisposeOnceGuard.DisposeAll(new object?[] { _qNetwork, _targetNetwork }, GetType().Name);
        }
        finally
        {
            base.Dispose();
        }
    }
}

/// <summary>
/// Custom dueling network architecture that separates value and advantage streams.
/// </summary>
/// <remarks>
/// Owns its dense layers directly (it is not a <see cref="NeuralNetworkBase{T}"/>), so disposing it releases
/// each of them through the same once-only guard a network uses for its layers.
/// </remarks>
internal class DuelingNetwork<T> : IParameterSource<T>, IDisposable
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<DenseLayer<T>> _sharedLayers;
    private readonly List<DenseLayer<T>> _valueLayers;
    private readonly List<DenseLayer<T>> _advantageLayers;
    private readonly int _stateSize;
    private readonly int _actionSize;

    [Scratch]
    private Vector<T>? _lastSharedOutput;
    [Scratch]
    private Vector<T>? _lastValueOutput;
    [Scratch]
    private Vector<T>? _lastAdvantageOutput;
    [Scratch]
    private readonly List<Vector<T>> _lastSharedInputs = new();
    [Scratch]
    private readonly List<Vector<T>> _lastSharedOutputs = new();
    [Scratch]
    private readonly List<Vector<T>> _lastValueInputs = new();
    [Scratch]
    private readonly List<Vector<T>> _lastValueOutputs = new();
    [Scratch]
    private readonly List<Vector<T>> _lastAdvantageInputs = new();
    [Scratch]
    private readonly List<Vector<T>> _lastAdvantageOutputs = new();

    public DuelingNetwork(
        int stateSize,
        int actionSize,
        int[] sharedLayerSizes,
        int[] valueLayerSizes,
        int[] advantageLayerSizes,
        INumericOperations<T> numOps,
        int? seed = null)
    {
        _numOps = numOps;
        _stateSize = stateSize;
        _actionSize = actionSize;
        _sharedLayers = new List<DenseLayer<T>>();
        _valueLayers = new List<DenseLayer<T>>();
        _advantageLayers = new List<DenseLayer<T>>();

        // Build shared feature layers
        int prevSize = stateSize;
        foreach (var size in sharedLayerSizes)
        {
            _sharedLayers.Add(new DenseLayer<T>(size, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = size;
        }

        int sharedOutputSize = prevSize;

        // Build value stream
        prevSize = sharedOutputSize;
        foreach (var size in valueLayerSizes)
        {
            _valueLayers.Add(new DenseLayer<T>(size, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = size;
        }
        _valueLayers.Add(new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>())); // Output single value

        // Build advantage stream
        prevSize = sharedOutputSize;
        foreach (var size in advantageLayerSizes)
        {
            _advantageLayers.Add(new DenseLayer<T>(size, (IActivationFunction<T>)new ReLUActivation<T>()));
            prevSize = size;
        }
        _advantageLayers.Add(new DenseLayer<T>(actionSize, (IActivationFunction<T>)new IdentityActivation<T>())); // Output per-action advantages

        // A seeded network gives each dense layer its own seed, so its lazy weights are the same on every build.
        if (seed is int baseSeed)
        {
            int index = 0;
            foreach (var layer in _sharedLayers.Concat(_valueLayers).Concat(_advantageLayers))
                layer.RandomSeed = unchecked(baseSeed + index++);
        }
    }

    /// <summary>
    /// Releases every dense layer of the shared, value and advantage streams, each at most once.
    /// </summary>
    /// <exception cref="AggregateException">Two or more layers threw from <c>Dispose</c>; a single failure is rethrown unchanged.</exception>
    public void Dispose()
    {
        AiDotNet.Helpers.DisposeOnceGuard.DisposeAll(
            _sharedLayers.Concat(_valueLayers).Concat(_advantageLayers), nameof(DuelingNetwork<T>));
    }

    public void EnsureInitialized()
    {
        if (GetFlattenedParameters().Rows > 0) return;
        Predict(new Vector<T>(_stateSize));
    }

    public Vector<T> Forward(Vector<T> state)
    {
        // The per-layer vectors below are read only by UpdateWeights' manual backward, so they are
        // copied out only when such a backward can run.
        bool cacheForBackward = ManualBackwardCache<T>.ShouldCacheActivationsForManualBackward;
        _lastSharedInputs.Clear();
        _lastSharedOutputs.Clear();
        _lastValueInputs.Clear();
        _lastValueOutputs.Clear();
        _lastAdvantageInputs.Clear();
        _lastAdvantageOutputs.Clear();

        // Shared layers
        var sharedTensor = Tensor<T>.FromVector(state);
        foreach (var layer in _sharedLayers)
        {
            if (cacheForBackward) _lastSharedInputs.Add(sharedTensor.ToVector());
            sharedTensor = layer.Forward(sharedTensor);
            if (cacheForBackward) _lastSharedOutputs.Add(sharedTensor.ToVector());
        }
        var sharedOutput = sharedTensor.ToVector();
        ManualBackwardCache<T>.SaveForBackward(ref _lastSharedOutput, sharedOutput);

        // Value stream
        var valueTensor = sharedTensor;
        foreach (var layer in _valueLayers)
        {
            if (cacheForBackward) _lastValueInputs.Add(valueTensor.ToVector());
            valueTensor = layer.Forward(valueTensor);
            if (cacheForBackward) _lastValueOutputs.Add(valueTensor.ToVector());
        }
        var valueOutput = valueTensor.ToVector();
        ManualBackwardCache<T>.SaveForBackward(ref _lastValueOutput, valueOutput);
        T value = valueOutput[0];

        // Advantage stream
        var advantageTensor = sharedTensor;
        foreach (var layer in _advantageLayers)
        {
            if (cacheForBackward) _lastAdvantageInputs.Add(advantageTensor.ToVector());
            advantageTensor = layer.Forward(advantageTensor);
            if (cacheForBackward) _lastAdvantageOutputs.Add(advantageTensor.ToVector());
        }
        var advantageOutput = advantageTensor.ToVector();
        ManualBackwardCache<T>.SaveForBackward(ref _lastAdvantageOutput, advantageOutput);

        // Compute mean advantage for centering
        var engine = AiDotNetEngine.Current;
        var advantages = RLTensorOps.Head(advantageOutput, _actionSize);
        T meanAdvantage = _numOps.Divide(engine.Sum(advantages), _numOps.FromDouble(_actionSize));

        // Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        var centeredAdvantage = engine.Subtract(advantages, Vector<T>.CreateDefault(_actionSize, meanAdvantage));
        return engine.Add(Vector<T>.CreateDefault(_actionSize, value), centeredAdvantage);
    }

    /// <summary>
    /// Predicts Q-values for the given state.
    /// </summary>
    /// <param name="input">The input state vector.</param>
    /// <returns>Q-values for all actions.</returns>
    /// <remarks>
    /// This method runs without retaining activations for a backward pass.
    /// </remarks>
    public Vector<T> Predict(Vector<T> input)
    {
        using var inference = InferenceMode.Enter();
        return Forward(input);
    }

    public void UpdateWeights(Vector<T> outputGradient, T learningRate)
    {
        if (outputGradient.Length != _actionSize)
            throw new ArgumentException(
                $"Expected {_actionSize} output gradients, got {outputGradient.Length}.",
                nameof(outputGradient));
        if (_lastSharedOutput is null || _lastValueOutput is null || _lastAdvantageOutput is null)
            throw new InvalidOperationException("Forward must be called before updating DuelingNetwork weights.");

        // dL/dV = sum_a dL/dQ_a; dL/dA_a = dL/dQ_a - mean_a' dL/dQ_a'
        var engine = AiDotNetEngine.Current;
        T valueGradient = engine.Sum(outputGradient);
        T meanOutputGradient = _numOps.Divide(valueGradient, _numOps.FromDouble(_actionSize));
        var valueStreamGradient = Vector<T>.CreateDefault(1, valueGradient);
        var advantageStreamGradient = engine.Subtract(outputGradient, Vector<T>.CreateDefault(_actionSize, meanOutputGradient));

        var sharedGradientFromValue = BackpropagateLayers(
            _valueLayers,
            _lastValueInputs,
            _lastValueOutputs,
            valueStreamGradient,
            learningRate,
            finalLayerIsIdentity: true);

        var sharedGradientFromAdvantage = BackpropagateLayers(
            _advantageLayers,
            _lastAdvantageInputs,
            _lastAdvantageOutputs,
            advantageStreamGradient,
            learningRate,
            finalLayerIsIdentity: true);

        var sharedGradient = AddVectors(sharedGradientFromValue, sharedGradientFromAdvantage);
        BackpropagateLayers(
            _sharedLayers,
            _lastSharedInputs,
            _lastSharedOutputs,
            sharedGradient,
            learningRate,
            finalLayerIsIdentity: false);
    }

    private Vector<T> BackpropagateLayers(
        IReadOnlyList<DenseLayer<T>> layers,
        IReadOnlyList<Vector<T>> inputs,
        IReadOnlyList<Vector<T>> outputs,
        Vector<T> outputGradient,
        T learningRate,
        bool finalLayerIsIdentity)
    {
        if (layers.Count == 0) return outputGradient;
        if (inputs.Count != layers.Count || outputs.Count != layers.Count)
            throw new InvalidOperationException("DuelingNetwork forward cache does not match its layer count.");

        var gradient = outputGradient;
        for (int layerIndex = layers.Count - 1; layerIndex >= 0; layerIndex--)
        {
            bool useReluDerivative = !finalLayerIsIdentity || layerIndex != layers.Count - 1;
            gradient = BackpropagateLayer(
                layers[layerIndex],
                inputs[layerIndex],
                outputs[layerIndex],
                gradient,
                learningRate,
                useReluDerivative);
        }

        return gradient;
    }

    private Vector<T> BackpropagateLayer(
        DenseLayer<T> layer,
        Vector<T> input,
        Vector<T> output,
        Vector<T> outputGradient,
        T learningRate,
        bool useReluDerivative)
    {
        if (outputGradient.Length != output.Length)
            throw new InvalidOperationException(
                $"Gradient length {outputGradient.Length} does not match layer output length {output.Length}.");

        var parameters = layer.GetParameters();
        int inputSize = input.Length;
        int outputSize = output.Length;
        int weightCount = inputSize * outputSize;
        int expectedParameterCount = weightCount + outputSize;

        if (parameters.Length != expectedParameterCount)
            throw new InvalidOperationException(
                $"Dense layer parameter count {parameters.Length} does not match expected {expectedParameterCount}.");

        // delta = g * relu'(output) (or g for the identity head)
        var engine = AiDotNetEngine.Current;
        var localGradient = useReluDerivative
            ? engine.TensorWhere(
                engine.TensorGreaterThan(output.AsTensor(), _numOps.Zero), outputGradient.AsTensor(), engine.TensorFullLike(outputGradient.AsTensor(), _numOps.Zero)).ToVector()
            : outputGradient;

        // W is laid out [input, output] row-major, then the output biases.
        var weights = engine.Reshape(parameters.GetSubVector(0, weightCount).AsTensor(), [inputSize, outputSize]);
        var deltaRows = engine.TensorTile(engine.Reshape(localGradient.AsTensor(), [1, outputSize]), [inputSize, 1]);
        var inputColumns = engine.TensorTile(engine.Reshape(input.AsTensor(), [inputSize, 1]), [1, outputSize]);

        // dL/dx_i = sum_j W_ij delta_j; W_ij <- W_ij - lr * x_i delta_j; b_j <- b_j - lr * delta_j
        var inputGradient = engine.ReduceSum(engine.TensorMultiply(weights, deltaRows), [1], keepDims: false).ToVector();
        var updatedWeights = engine.TensorSubtract(weights, engine.TensorMultiplyScalar(engine.TensorMultiply(inputColumns, deltaRows), learningRate));
        var updatedBiases = engine.Subtract(parameters.GetSubVector(weightCount, outputSize), engine.Multiply(localGradient, learningRate));
        var updatedParameters = RLTensorOps.Concat(updatedWeights.ToVector(), updatedBiases);

        layer.SetParameters(updatedParameters);
        return inputGradient;
    }

    private Vector<T> AddVectors(Vector<T> left, Vector<T> right)
    {
        if (left.Length != right.Length)
            throw new InvalidOperationException(
                $"Cannot add gradients with lengths {left.Length} and {right.Length}.");

        return AiDotNetEngine.Current.Add(left, right);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Sized from the layers rather than from <see cref="GetFlattenedParameters"/>, which would
    /// have to build the whole flattened matrix just to report its height.
    /// </remarks>
    public long ParameterCount
    {
        get
        {
            EnsureInitialized();
            long total = 0;
            foreach (var layer in _sharedLayers) total += layer.ParameterCount;
            foreach (var layer in _valueLayers) total += layer.ParameterCount;
            foreach (var layer in _advantageLayers) total += layer.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Delegates to <see cref="GetFlattenedParameters"/> so the shared -> value -> advantage order
    /// this network has always serialized is preserved exactly. This is the plumbing DuelingDQNAgent
    /// used to hand-write; it belongs on the network that owns the layers, which is what lets the
    /// agent simply register this component and inherit all three surfaces.
    /// </remarks>
    public Vector<T> GetParameters()
    {
        EnsureInitialized();
        var flat = GetFlattenedParameters();
        return flat.Rows == 0 ? new Vector<T>(0) : AiDotNetEngine.Current.GetColumn(flat, 0);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        EnsureInitialized();
        var matrix = new Matrix<T>(parameters.Length, 1);
        parameters.AsSpan().CopyTo(matrix.AsWritableSpan());
        SetFlattenedParameters(matrix);
    }

    public Matrix<T> GetFlattenedParameters()
    {
        // shared -> value -> advantage, each layer's parameters in order, as one column
        var flat = RLTensorOps.Concat(_sharedLayers.Concat(_valueLayers).Concat(_advantageLayers)
            .Select(layer => layer.GetParameters()).ToArray());
        var matrix = new Matrix<T>(flat.Length, 1);
        flat.AsSpan().CopyTo(matrix.AsWritableSpan());
        return matrix;
    }

    public void SetFlattenedParameters(Matrix<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        int offset = 0;
        var column = parameters.Rows == 0 ? new Vector<T>(0) : AiDotNetEngine.Current.GetColumn(parameters, 0);

        // Shared layers, then the value stream, then the advantage stream, each taking its slice of the column
        foreach (var layer in _sharedLayers.Concat(_valueLayers).Concat(_advantageLayers))
        {
            int paramCount = checked((int)layer.ParameterCount);
            layer.SetParameters(column.GetSubVector(offset, paramCount));
            offset += paramCount;
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
            writer.Write(layer.GetOutputLayerShape().RequireConcrete("Recording concrete layer geometry")[0]);
        }

        // Write layer sizes for value layers
        foreach (var layer in _valueLayers)
        {
            writer.Write(layer.GetInputShape()[0]);
            writer.Write(layer.GetOutputLayerShape().RequireConcrete("Recording concrete layer geometry")[0]);
        }

        // Write layer sizes for advantage layers
        foreach (var layer in _advantageLayers)
        {
            writer.Write(layer.GetInputShape()[0]);
            writer.Write(layer.GetOutputLayerShape().RequireConcrete("Recording concrete layer geometry")[0]);
        }

        // Serialize parameters
        var parameters = GetFlattenedParameters();
        writer.Write(parameters.Rows);
        var values = parameters.Rows == 0
            ? new Vector<double>(0)
            : AiDotNetEngine.Current.GetColumn(parameters, 0).AsTensor().Cast<double>().ToVector();
        foreach (double value in values)
        {
            writer.Write(value);
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
        var values = new double[paramCount];
        for (int i = 0; i < paramCount; i++)
        {
            values[i] = reader.ReadDouble();
        }

        var parameters = new Matrix<T>(paramCount, 1);
        if (paramCount > 0)
        {
            Vector<double>.FromArray(values).AsTensor().Cast<T>().ToVector().AsSpan().CopyTo(parameters.AsWritableSpan());
        }

        SetFlattenedParameters(parameters);
    }
}
