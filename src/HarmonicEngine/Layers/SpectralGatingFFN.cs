using AiDotNet.HarmonicEngine.Activations;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Layers;

/// <summary>
/// Per-position feed-forward network for HREBlock. Replaces the standard transformer
/// FFN (<c>Linear → GELU → Linear</c>) with two Spectral Hebbian filters separated by
/// a Spectral Gating nonlinearity. Applied position-wise across the sequence — each
/// token's embedding vector is independently passed through the same spectral filters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In a standard transformer, the FFN is two linear layers with a
/// nonlinearity in between. Each token gets transformed through the same FFN. HRE's
/// FFN does the same thing but uses spectral Hebbian filters instead of learnable
/// dense matrices:
/// </para>
/// <para>
/// <c>SpectralHebbianLayer(E) → SpectralGating → SpectralHebbianLayer(E)</c>
/// </para>
/// <para>
/// Each Hebbian layer is FFT → multiply by learned H(k) → IFFT, applied to the
/// length-E embedding vector of a single token. The filter H(k) is a complex spectral
/// filter learned via the Hebbian update rule (Theorem 3). SpectralGating is an
/// input-dependent nonlinearity that behaves like a smooth version of ReLU.
/// </para>
/// <para>
/// Parameter count is 4·E per block (two filters × E bins × 2 real numbers per complex
/// coefficient) — dramatically fewer than a transformer FFN's 8·E² (two 4E×E matrices).
/// For embed dim 64, that's 256 FFN parameters vs transformer's 32,768 — a ~128× compression.
/// </para>
/// </remarks>
public class SpectralGatingFFN<T> : LayerBase<T>
{
    private readonly SpectralHebbianLayer<T> _filter1;
    private readonly SpectralGatingActivation<T> _gating;
    private readonly SpectralHebbianLayer<T> _filter2;
    private readonly int _embedDim;

    /// <inheritdoc/>
    public override string LayerName => $"SpectralGatingFFN_{_embedDim}";

    /// <inheritdoc/>
    public override int ParameterCount => _filter1.ParameterCount + _filter2.ParameterCount;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the first spectral Hebbian filter (for external Hebbian updates).
    /// </summary>
    public SpectralHebbianLayer<T> Filter1 => _filter1;

    /// <summary>
    /// Gets the second spectral Hebbian filter (for external Hebbian updates).
    /// </summary>
    public SpectralHebbianLayer<T> Filter2 => _filter2;

    /// <summary>
    /// Creates a new SpectralGatingFFN.
    /// </summary>
    /// <param name="embedDim">The embedding dimension E. Must be a power of 2 for FFT.</param>
    /// <param name="hebbianLearningRate">Learning rate for the Hebbian updates. Default 0.01.</param>
    /// <param name="antiHebbianAlpha">Anti-Hebbian decorrelation strength. Default 0.5,
    /// giving a fixed point at H_eq = (1/α) · H_wiener = 2 · H_wiener per Theorem 3.</param>
    public SpectralGatingFFN(int embedDim, double hebbianLearningRate = 0.01, double antiHebbianAlpha = 0.5)
        : base([embedDim], [embedDim])
    {
        if (embedDim < 2 || (embedDim & (embedDim - 1)) != 0)
            throw new ArgumentException(
                $"Embedding dimension must be a power of 2 for the FFT, got {embedDim}.",
                nameof(embedDim));

        _embedDim = embedDim;
        _filter1 = new SpectralHebbianLayer<T>(embedDim, hebbianLearningRate, antiHebbianAlpha);
        _gating = new SpectralGatingActivation<T>();
        _filter2 = new SpectralHebbianLayer<T>(embedDim, hebbianLearningRate, antiHebbianAlpha);
    }

    /// <summary>
    /// Forward pass. Accepts either a single embedding [E] or a batch [B, S, E] /
    /// unbatched sequence [S, E], and applies the FFN position-wise.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Shape dispatch: support [E], [S, E], and [B, S, E]
        if (input._shape.Length == 1)
        {
            // Single embedding vector — apply directly
            return ApplySinglePosition(input);
        }

        if (input._shape.Length == 2)
        {
            // [S, E] — apply per position
            int seqLen = input._shape[0];
            int embedDim = input._shape[1];
            if (embedDim != _embedDim)
                throw new ArgumentException(
                    $"Embedding dimension ({embedDim}) must match FFN dim ({_embedDim}).");

            var output = new Tensor<T>([seqLen, embedDim]);
            for (int s = 0; s < seqLen; s++)
            {
                var slice = new Tensor<T>([embedDim]);
                for (int e = 0; e < embedDim; e++) slice[e] = input[s, e];
                var transformed = ApplySinglePosition(slice);
                for (int e = 0; e < embedDim; e++) output[s, e] = transformed[e];
            }
            return output;
        }

        if (input._shape.Length == 3)
        {
            // [B, S, E] — apply per batch, per position
            int batchSize = input._shape[0];
            int seqLen = input._shape[1];
            int embedDim = input._shape[2];
            if (embedDim != _embedDim)
                throw new ArgumentException(
                    $"Embedding dimension ({embedDim}) must match FFN dim ({_embedDim}).");

            var output = new Tensor<T>([batchSize, seqLen, embedDim]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    var slice = new Tensor<T>([embedDim]);
                    for (int e = 0; e < embedDim; e++) slice[e] = input[b, s, e];
                    var transformed = ApplySinglePosition(slice);
                    for (int e = 0; e < embedDim; e++) output[b, s, e] = transformed[e];
                }
            }
            return output;
        }

        throw new ArgumentException(
            $"SpectralGatingFFN expects input of shape [E], [S, E], or [B, S, E]; got {input._shape.Length}D.");
    }

    /// <summary>
    /// Applies the FFN to a single length-E embedding vector.
    /// Pipeline: filter1 → gating → filter2
    /// </summary>
    private Tensor<T> ApplySinglePosition(Tensor<T> embedding)
    {
        var afterFilter1 = _filter1.Forward(embedding);
        var afterGating = _gating.Activate(afterFilter1);
        var afterFilter2 = _filter2.Forward(afterGating);
        return afterFilter2;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var p1 = _filter1.GetParameters();
        var p2 = _filter2.GetParameters();
        var combined = new Vector<T>(p1.Length + p2.Length);
        for (int i = 0; i < p1.Length; i++) combined[i] = p1[i];
        for (int i = 0; i < p2.Length; i++) combined[p1.Length + i] = p2[i];
        return combined;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        // Hebbian layers don't use gradients (non-backprop learning).
        return new Vector<T>(ParameterCount);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _filter1.UpdateParameters(learningRate);
        _filter2.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int half = _filter1.ParameterCount;
        if (parameters.Length < half + _filter2.ParameterCount)
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) is less than required " +
                $"({half + _filter2.ParameterCount}).");

        var p1 = new Vector<T>(half);
        var p2 = new Vector<T>(_filter2.ParameterCount);
        for (int i = 0; i < half; i++) p1[i] = parameters[i];
        for (int i = 0; i < _filter2.ParameterCount; i++) p2[i] = parameters[half + i];
        _filter1.SetParameters(p1);
        _filter2.SetParameters(p2);
    }

    /// <inheritdoc/>
    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_embedDim);
        _filter1.Serialize(writer);
        _filter2.Serialize(writer);
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // embedDim
        _filter1.Deserialize(reader);
        _filter2.Deserialize(reader);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _filter1.ResetState();
        _filter2.ResetState();
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _filter1.SetTrainingMode(isTraining);
        _filter2.SetTrainingMode(isTraining);
    }
}
