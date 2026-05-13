using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Dueling DQN combination head (Wang et al. 2016, "Dueling Network
/// Architectures for Deep Reinforcement Learning"). Splits the shared
/// feature trunk's output into a scalar state-value V(s) and an
/// action-advantage vector A(s, a), then combines them as
/// Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a))
/// to enforce the identifiability constraint Eq. 9 of the paper.
/// </summary>
/// <typeparam name="T">Numeric type used for tensor data.</typeparam>
/// <remarks>
/// <para>
/// <b>What this layer does:</b> Reads <c>[batch, featureDim]</c> shared
/// trunk features and emits <c>[batch, actionSize]</c> Q-values. Internally
/// holds two parallel linear projections — a single-unit value head and an
/// <c>actionSize</c>-unit advantage head — and combines them using the
/// mean-subtraction form (paper §3, Eq. 9). The mean form is what every
/// production Rainbow / Double-DQN implementation uses (e.g.
/// Stable-Baselines3 <c>DuelingQ</c>, RLlib <c>DuelingMixin</c>) because the
/// max-subtraction alternative is unstable under bootstrapping.
/// </para>
/// <para>
/// <b>For Beginners:</b> "Dueling" means the network learns the value of
/// being in a state separately from the relative benefit of each action.
/// V(s) answers "is this state good?", A(s, a) answers "given this state,
/// is action a better or worse than average?". Combining them gives more
/// stable training than predicting Q(s, a) directly with one big head —
/// the value head doesn't have to redundantly encode "this state is good"
/// for every action.
/// </para>
/// <para>
/// All forward arithmetic runs through <see cref="LayerBase{T}.Engine"/>
/// ops on the trainable tensor instances, so the gradient tape captures
/// gradients for both heads automatically.
/// </para>
/// </remarks>
public class DuelingCombinationLayer<T> : LayerBase<T>
{
    private readonly int _featureDim;
    private readonly int _actionSize;

    // Trainable parameters — tape tracks them by reference identity. Fields
    // are NOT readonly because NeuralNetworkBase.GetOrCreateParameterBuffer
    // rebinds them to views into the contiguous ParameterBuffer via
    // SetTrainableParameters; the tape's TapeStepContext.ValidateBufferAlignment
    // then requires every tensor it sees during forward to be the same
    // reference the buffer holds. Copying data into the old standalone
    // tensors would leave Forward() using standalone-tensor references the
    // tape rejects with "Parameter N is not a view into the provided
    // ParameterBuffer" — the supervised RainbowDQNAgent.Train(state, target)
    // path takes for offline pretraining / BC warm-start.
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;
    private Tensor<T> _advantageWeights;
    private Tensor<T> _advantageBias;

    /// <summary>
    /// Initializes a new dueling combination head.
    /// </summary>
    /// <param name="featureDim">
    /// Trunk output feature dimension (the per-sample width of the input).
    /// Both heads share this input dimensionality.
    /// </param>
    /// <param name="actionSize">Number of discrete actions (advantage-head output width).</param>
    /// <param name="seed">Optional RNG seed for weight initialisation.</param>
    public DuelingCombinationLayer(int featureDim, int actionSize, int? seed = null)
        : base(
            inputShape: [-1],
            outputShape: [actionSize],
            scalarActivation: new IdentityActivation<T>())
    {
        if (featureDim <= 0) throw new ArgumentOutOfRangeException(nameof(featureDim));
        if (actionSize <= 0) throw new ArgumentOutOfRangeException(nameof(actionSize));

        _featureDim = featureDim;
        _actionSize = actionSize;

        _valueWeights = new Tensor<T>([featureDim, 1]);
        _valueBias = new Tensor<T>([1]);
        _advantageWeights = new Tensor<T>([featureDim, actionSize]);
        _advantageBias = new Tensor<T>([actionSize]);

        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Glorot-uniform initialisation, separate range per head:
        // U(-sqrt(6 / (fan_in + fan_out)), +same).
        InitializeUniform(_valueWeights, rng, _featureDim, 1);
        InitializeUniform(_advantageWeights, rng, _featureDim, actionSize);
        // Biases zeroed (PyTorch default for linear layers).
    }

    private void InitializeUniform(Tensor<T> tensor, Random rng, int fanIn, int fanOut)
    {
        double range = Math.Sqrt(6.0 / (fanIn + fanOut));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * range);
    }

    /// <inheritdoc/>
    public override long ParameterCount =>
        (long)_featureDim * 1 + 1 + (long)_featureDim * _actionSize + _actionSize;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters() =>
        new[] { _valueWeights, _valueBias, _advantageWeights, _advantageBias };

    /// <inheritdoc/>
    /// <remarks>
    /// Replaces the field tensor references with the supplied tensors rather
    /// than copying data into the old ones. ParameterBuffer machinery in
    /// <see cref="NeuralNetworkBase{T}.GetOrCreateParameterBuffer"/> calls
    /// this with buffer-backed views; the tape's reference-identity
    /// alignment check (TapeStepContext.ValidateBufferAlignment) then
    /// requires Forward() to use those view tensors. Validate per-dim shape
    /// match first so a same-length but differently-shaped tensor doesn't
    /// silently scramble the layer's weights.
    /// </remarks>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters.Count != 4)
            throw new ArgumentException(
                "Expected exactly 4 parameter tensors (V_w, V_b, A_w, A_b).", nameof(parameters));
        ValidateShapeMatch(parameters[0], _valueWeights, nameof(_valueWeights));
        ValidateShapeMatch(parameters[1], _valueBias, nameof(_valueBias));
        ValidateShapeMatch(parameters[2], _advantageWeights, nameof(_advantageWeights));
        ValidateShapeMatch(parameters[3], _advantageBias, nameof(_advantageBias));
        _valueWeights = parameters[0];
        _valueBias = parameters[1];
        _advantageWeights = parameters[2];
        _advantageBias = parameters[3];
    }

    private static void ValidateShapeMatch(Tensor<T> incoming, Tensor<T> existing, string paramName)
    {
        if (incoming.Rank != existing.Rank || incoming.Length != existing.Length)
            throw new ArgumentException(
                $"Shape mismatch for {paramName}: incoming rank={incoming.Rank} length={incoming.Length}, " +
                $"existing rank={existing.Rank} length={existing.Length}.");
        for (int dim = 0; dim < incoming.Rank; dim++)
        {
            if (incoming.Shape[dim] != existing.Shape[dim])
                throw new ArgumentException(
                    $"Shape mismatch for {paramName} at dim {dim}: incoming={incoming.Shape[dim]}, " +
                    $"existing={existing.Shape[dim]}.");
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank == 0)
            throw new ArgumentException(
                "DuelingCombinationLayer expects an input with at least one dimension.",
                nameof(input));

        int featureSize = input.Rank == 1 ? input.Length : input.Shape[input.Rank - 1];
        if (featureSize != _featureDim)
            throw new ArgumentException(
                $"DuelingCombinationLayer expects last-dim feature size {_featureDim}, got {featureSize} " +
                $"(input shape [{string.Join(",", input.Shape)}]).",
                nameof(input));

        // Flatten to [batch, featureDim].
        int batchSize;
        Tensor<T> flatInput;
        if (input.Rank == 1)
        {
            batchSize = 1;
            flatInput = Engine.Reshape(input, [1, _featureDim]);
        }
        else if (input.Rank == 2)
        {
            batchSize = input.Shape[0];
            flatInput = input;
        }
        else
        {
            batchSize = 1;
            for (int i = 0; i < input.Rank - 1; i++) batchSize *= input.Shape[i];
            flatInput = Engine.Reshape(input, [batchSize, _featureDim]);
        }

        // V(s): [batch, 1].
        var vRaw = Engine.TensorMatMul(flatInput, _valueWeights);
        var vBiasBroadcast = Engine.Reshape(_valueBias, [1, 1]);
        var v = Engine.TensorAdd(vRaw, vBiasBroadcast);

        // A(s, a): [batch, actionSize].
        var aRaw = Engine.TensorMatMul(flatInput, _advantageWeights);
        var aBiasBroadcast = Engine.Reshape(_advantageBias, [1, _actionSize]);
        var a = Engine.TensorAdd(aRaw, aBiasBroadcast);

        // Mean of A across the action axis (axis=1), keepdims=true → [batch, 1].
        var aMean = Engine.ReduceMean(a, [1], keepDims: true);
        // A_centered = A − mean_a A. Tile mean across actionSize and subtract.
        var aMeanTiled = Engine.TensorTile(aMean, [1, _actionSize]);
        var aCentered = Engine.TensorSubtract(a, aMeanTiled);

        // Q(s, a) = V(s) + A_centered. Tile V across actionSize first.
        var vTiled = Engine.TensorTile(v, [1, _actionSize]);
        var q = Engine.TensorAdd(vTiled, aCentered);

        // Restore the caller's rank (1D in → 1D out, N-D batch in → N-D out).
        if (input.Rank == 1) return Engine.Reshape(q, [_actionSize]);
        if (input.Rank == 2) return q;
        var outShape = new int[input.Rank];
        for (int i = 0; i < input.Rank - 1; i++) outShape[i] = input.Shape[i];
        outShape[^1] = _actionSize;
        return Engine.Reshape(q, outShape);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var p = new Vector<T>((int)ParameterCount);
        int idx = 0;
        for (int i = 0; i < _valueWeights.Length; i++) p[idx++] = _valueWeights[i];
        for (int i = 0; i < _valueBias.Length; i++) p[idx++] = _valueBias[i];
        for (int i = 0; i < _advantageWeights.Length; i++) p[idx++] = _advantageWeights[i];
        for (int i = 0; i < _advantageBias.Length; i++) p[idx++] = _advantageBias[i];
        return p;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));
        int idx = 0;
        for (int i = 0; i < _valueWeights.Length; i++) _valueWeights[i] = parameters[idx++];
        for (int i = 0; i < _valueBias.Length; i++) _valueBias[i] = parameters[idx++];
        for (int i = 0; i < _advantageWeights.Length; i++) _advantageWeights[i] = parameters[idx++];
        for (int i = 0; i < _advantageBias.Length; i++) _advantageBias[i] = parameters[idx++];
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (ParameterGradients is null) return;
        if (ParameterGradients.Length != ParameterCount)
            throw new InvalidOperationException(
                $"DuelingCombinationLayer.UpdateParameters: gradient buffer length " +
                $"{ParameterGradients.Length} does not match ParameterCount {ParameterCount}.");

        int idx = 0;
        for (int i = 0; i < _valueWeights.Length; i++)
            _valueWeights[i] = NumOps.Subtract(_valueWeights[i],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
        for (int i = 0; i < _valueBias.Length; i++)
            _valueBias[i] = NumOps.Subtract(_valueBias[i],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
        for (int i = 0; i < _advantageWeights.Length; i++)
            _advantageWeights[i] = NumOps.Subtract(_advantageWeights[i],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
        for (int i = 0; i < _advantageBias.Length; i++)
            _advantageBias[i] = NumOps.Subtract(_advantageBias[i],
                NumOps.Multiply(learningRate, ParameterGradients[idx++]));
    }

    /// <inheritdoc/>
    public override void ResetState() { /* no per-forward state outside the tape */ }

    /// <summary>
    /// Constructor metadata for DeserializationHelper post-Clone reconstruction.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["FeatureDim"] = _featureDim.ToString(inv);
        metadata["ActionSize"] = _actionSize.ToString(inv);
        return metadata;
    }
}
