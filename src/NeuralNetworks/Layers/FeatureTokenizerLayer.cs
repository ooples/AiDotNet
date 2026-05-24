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
/// Rank-robust: <c>[F] → [F, embedding]</c> (unbatched) and <c>[batch, F] → [batch, F, embedding]</c>
/// (batched). Forward is expressed with broadcast Engine ops on the registered weight/bias tensors
/// so the tape computes their gradients automatically.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FeatureTokenizerLayer<T> : LayerBase<T>
{
    private readonly int _numFeatures;
    private readonly int _embeddingDim;
    private Tensor<T> _weights; // [numFeatures, embeddingDim]
    private Tensor<T> _biases;  // [numFeatures, embeddingDim]

    /// <summary>
    /// Initializes a new <see cref="FeatureTokenizerLayer{T}"/>.
    /// </summary>
    /// <param name="numFeatures">Number of input features (sequence length of the produced tokens).</param>
    /// <param name="embeddingDim">Embedding dimension per feature token.</param>
    public FeatureTokenizerLayer(int numFeatures, int embeddingDim)
        : base(new[] { numFeatures }, new[] { numFeatures, embeddingDim })
    {
        if (numFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(numFeatures));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        _numFeatures = numFeatures;
        _embeddingDim = embeddingDim;
        _weights = new Tensor<T>(new[] { numFeatures, embeddingDim });
        _biases = new Tensor<T>(new[] { numFeatures, embeddingDim });
        InitializeParameters();

        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount => 2L * _numFeatures * _embeddingDim;

    private void InitializeParameters()
    {
        var rand = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Uniform(-1/sqrt(E), 1/sqrt(E)) for BOTH weights and biases, per
        // FT-Transformer's tokenizer init. The bias must be non-zero: with a zero
        // bias each token is x_f * W[f] — a pure scalar multiple of W[f] — and the
        // encoder's LayerNorm strips that scale, collapsing constant inputs of
        // different magnitude (e.g. all-0.1 vs all-0.9) to identical tokens. A
        // learnable non-zero bias breaks that scale-invariance.
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
        int rank = input.Rank;

        // Expand to [..., F, 1] so the trailing singleton broadcasts to the embedding dim.
        var expShape = new int[rank + 1];
        for (int i = 0; i < rank; i++) expShape[i] = input.Shape[i];
        expShape[rank] = 1;
        var expanded = Engine.Reshape(input, expShape);

        // Reshape the [F, E] parameters to rank+1 with leading singletons so the
        // broadcast aligns with batched ([batch, F, 1]) and unbatched ([F, 1]) inputs.
        var paramShape = new int[rank + 1];
        for (int i = 0; i < rank - 1; i++) paramShape[i] = 1;
        paramShape[rank - 1] = _numFeatures;
        paramShape[rank] = _embeddingDim;
        var wB = Engine.Reshape(_weights, paramShape);
        var bB = Engine.Reshape(_biases, paramShape);

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
        => Vector<T>.Concatenate(
            Vector<T>.FromMemory(_weights.Data),
            Vector<T>.FromMemory(_biases.Data));

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
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
