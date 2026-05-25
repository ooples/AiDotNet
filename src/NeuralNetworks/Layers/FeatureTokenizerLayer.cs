using System;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Feature tokenizer for tabular transformers: embeds each scalar input feature into its OWN
/// learnable embedding vector, producing a <c>[features, embedding]</c> token sequence that a
/// transformer encoder can attend over.
/// </summary>
/// <remarks>
/// <para>
/// Implements the numerical feature tokenizer of FT-Transformer (Gorishniy et al. 2021,
/// "Revisiting Deep Learning Models for Tabular Data"), the same per-column embedding idea that
/// underlies TabTransformer (Huang et al. 2020): token[f] = x[f] · W[f] + b[f], where each feature
/// f has its own embedding row W[f] (shape <c>[embedding]</c>) and bias b[f].
/// </para>
/// <para>
/// This is critical for tabular models: a shared projection (a single Dense layer) maps the whole
/// feature vector to ONE vector, so self-attention runs over a length-1 sequence (no attention) —
/// and even a per-token Dense produces collinear tokens (all <c>x_f · W</c>) whose only difference
/// is a scalar that LayerNorm then removes, collapsing distinct inputs to identical outputs.
/// Per-feature embedding directions break that degeneracy and encode feature identity, so no
/// separate positional embedding is needed.
/// </para>
/// <para>
/// The feature count is resolved lazily from the first forward input (like <see cref="DenseLayer{T}"/>),
/// so the layer adapts to the actual fed input width even when a model's declared input size differs.
/// Output is always a batched <c>[batch, features, embedding]</c> tensor (batch=1 for an unbatched
/// <c>[features]</c> input) so the downstream encoder and head treat the feature axis unambiguously.
/// Forward is expressed with broadcast Engine ops on the registered weight/bias tensors so the tape
/// computes their gradients automatically.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FeatureTokenizerLayer<T> : LayerBase<T>
{
    private int _numFeatures;
    private readonly int _embeddingDim;
    private Tensor<T> _weights = new Tensor<T>(new[] { 0, 0 }); // [numFeatures, embeddingDim]
    private Tensor<T> _biases = new Tensor<T>(new[] { 0, 0 });  // [numFeatures, embeddingDim]
    private bool _initialized;

    /// <summary>
    /// Initializes a tokenizer whose feature count is resolved lazily on the first forward pass.
    /// </summary>
    /// <param name="embeddingDim">Embedding dimension per feature token.</param>
    public FeatureTokenizerLayer(int embeddingDim)
        : base(new[] { -1 }, new[] { -1, embeddingDim })
    {
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));
        _embeddingDim = embeddingDim;
        _numFeatures = -1;
    }

    /// <summary>
    /// Initializes a tokenizer with an explicit feature count. The count is still re-resolved from
    /// the first forward input if it differs (the input is authoritative).
    /// </summary>
    /// <param name="numFeatures">Expected number of input features.</param>
    /// <param name="embeddingDim">Embedding dimension per feature token.</param>
    public FeatureTokenizerLayer(int numFeatures, int embeddingDim)
        : this(embeddingDim)
    {
        if (numFeatures > 0)
        {
            _numFeatures = numFeatures;
            EnsureTokenizerInitialized();
        }
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount => _initialized ? 2L * _numFeatures * _embeddingDim : 0L;

    private void EnsureTokenizerInitialized()
    {
        if (_initialized || _numFeatures <= 0) return;

        _weights = new Tensor<T>(new[] { _numFeatures, _embeddingDim });
        _biases = new Tensor<T>(new[] { _numFeatures, _embeddingDim });
        InitializeParameters();

        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
        _initialized = true;
        ResolveShapes(new[] { _numFeatures }, new[] { _numFeatures, _embeddingDim });
    }

    private void InitializeParameters()
    {
        var rand = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Uniform(-1/sqrt(E), 1/sqrt(E)) for BOTH weights and biases, per FT-Transformer's
        // tokenizer init. The bias must be non-zero: with a zero bias each token is x_f * W[f] —
        // a pure scalar multiple of W[f] — and the encoder's LayerNorm strips that scale,
        // collapsing constant inputs of different magnitude (all-0.1 vs all-0.9) to identical
        // tokens. A learnable non-zero bias breaks that scale-invariance.
        double scale = 1.0 / Math.Sqrt(_embeddingDim);
        for (int f = 0; f < _numFeatures; f++)
        {
            for (int e = 0; e < _embeddingDim; e++)
            {
                _weights[f, e] = NumOps.FromDouble((rand.NextDouble() * 2.0 - 1.0) * scale);
                _biases[f, e] = NumOps.FromDouble((rand.NextDouble() * 2.0 - 1.0) * scale);
            }
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int features = input.Shape[input.Rank - 1];
        if (!_initialized || _numFeatures != features)
        {
            // The fed input width is authoritative — (re)size to it.
            _initialized = false;
            _numFeatures = features;
            EnsureTokenizerInitialized();
        }

        // Always emit a batched rank-3 token sequence [batch, F, E] (batch=1 for an
        // unbatched [F] input) so downstream layers treat the feature axis (1) uniformly.
        int batch = input.Rank == 1 ? 1 : input.Shape[0];

        var expanded = Engine.Reshape(input, new[] { batch, _numFeatures, 1 });
        var wB = Engine.Reshape(_weights, new[] { 1, _numFeatures, _embeddingDim });
        var bB = Engine.Reshape(_biases, new[] { 1, _numFeatures, _embeddingDim });

        var scaled = Engine.TensorBroadcastMultiply(expanded, wB);
        return Engine.TensorBroadcastAdd(scaled, bB);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Trainable parameters are updated in-place by the tape optimizer via the
        // registered weight/bias tensors; no eager SGD step is needed here.
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        if (!_initialized) return new Vector<T>(0);
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(_weights.Data),
            Vector<T>.FromMemory(_biases.Data));
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (!_initialized)
        {
            // Resolve the feature count from the parameter vector: length = 2 * F * E.
            if (parameters.Length == 0) return;
            int inferred = parameters.Length / (2 * _embeddingDim);
            if (inferred <= 0 || inferred * 2 * _embeddingDim != parameters.Length)
                throw new ArgumentException(
                    $"Cannot infer feature count from {parameters.Length} parameters with embeddingDim {_embeddingDim}.",
                    nameof(parameters));
            _numFeatures = inferred;
            EnsureTokenizerInitialized();
        }

        int wCount = _numFeatures * _embeddingDim;
        if (parameters.Length != 2 * wCount)
            throw new ArgumentException(
                $"Expected {2 * wCount} parameters, got {parameters.Length}.", nameof(parameters));

        var wSpan = _weights.Data.Span;
        var bSpan = _biases.Data.Span;
        for (int i = 0; i < wCount; i++) wSpan[i] = parameters[i];
        for (int i = 0; i < wCount; i++) bSpan[i] = parameters[wCount + i];
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        // Stateless across batches (no cached activations).
    }
}
