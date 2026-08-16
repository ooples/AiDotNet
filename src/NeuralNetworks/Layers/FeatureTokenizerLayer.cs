using System;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

using AiDotNet.Attributes;

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
// The RANK CHANGES here, which is the whole content of the declaration: this layer's own summary
// says "Output is always a batched [batch, features, embedding] tensor", and ForwardTraced ends with
// a broadcast add over a [batch, _numFeatures, _embeddingDim] shape. So a rank-2 table of feature
// values leaves as a rank-3 token sequence.
//
// The output axes are named [Batch, Time, Features] rather than [Batch, Features, Channels] because
// that is the form the downstream transformer encoder reads - one token per input feature along the
// sequence axis, the embedding width last - and it is the same naming DenseLayer's rank-3 form uses,
// so the two chain-validate against each other instead of disagreeing about which axis is which.
//
// Only rank 2 is declared. ForwardTraced accepts rank 1 and higher ranks too, but it collapses ALL
// leading axes into the batch (batch = input.Length / features), so a rank-3 input's output batch is
// a PRODUCT of two input axes and a rank-1 input's is the constant 1 - neither is a claim this
// layer's configuration supports, so they are left undescribed rather than guessed at.
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "One token per input feature; the trailing axis is the per-token embedding width.")]
[AutoParameters]
public partial class FeatureTokenizerLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Read off <c>ResolveShapes(new[] { _numFeatures }, new[] { _numFeatures, _embeddingDim })</c>
    /// and the reshape in <c>ForwardTraced</c>. The token count is the input's own feature width -
    /// <c>ForwardTraced</c> opens by taking <c>input.Shape[input.Rank - 1]</c> and RESIZES the weight
    /// table to match ("The fed input width is authoritative"), so this is <c>Same</c>, not a
    /// <c>Fixed(_numFeatures)</c>: <c>_numFeatures</c> is a cache of the last input, not a constraint
    /// on the next one.
    /// </para>
    /// <para>
    /// The embedding width is <c>Fixed(_embeddingDim)</c>, which IS a constructor argument and is the
    /// only shape this layer decides for itself.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _embeddingDim <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Features)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_embeddingDim)),
        };
    }

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
    public FeatureTokenizerLayer(
        [LayerState] int numFeatures,
        [LayerState] int embeddingDim)
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

    private void EnsureTokenizerInitialized()
    {
        if (_initialized || _numFeatures <= 0) return;

        // If a prior feature count was registered, unregister those tensors before replacing them.
        // RegisterTrainableParameter only swaps 0-length placeholders, so without this an input-width
        // change (lazy re-resolve in Forward) would leave the old [F,E] tensors in the persistent
        // registry, double-counting ParameterCount and updating stale tensors during training.
        if (_weights.Length > 0) UnregisterTrainableParameter(_weights);
        if (_biases.Length > 0) UnregisterTrainableParameter(_biases);

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int features = input.Shape[input.Rank - 1];
        if (!_initialized || _numFeatures != features)
        {
            // The fed input width is authoritative — (re)size to it.
            _initialized = false;
            _numFeatures = features;
            EnsureTokenizerInitialized();
        }

        // Always emit a batched rank-3 token sequence [batch, F, E]. Flatten ALL leading dims into
        // the batch axis (batch = total / F) so the tokenizer handles [F] (batch=1), [B,F], and
        // higher-rank [*, F] inputs uniformly — deriving batch from input.Shape[0] alone would
        // mis-size the reshape (and throw) for rank>2 inputs whose element count is B*S*F, not B*F.
        int batch = features > 0 ? input.Length / features : 1;

        var expanded = Engine.Reshape(input, new[] { batch, _numFeatures, 1 });
        var wB = Engine.Reshape(_weights, new[] { 1, _numFeatures, _embeddingDim });
        var bB = Engine.Reshape(_biases, new[] { 1, _numFeatures, _embeddingDim });

        var scaled = Engine.TensorBroadcastMultiply(expanded, wB);
        return Engine.TensorBroadcastAdd(scaled, bB);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        // Stateless across batches (no cached activations).
    }
}
