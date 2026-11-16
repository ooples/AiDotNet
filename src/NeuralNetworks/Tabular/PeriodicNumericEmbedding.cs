using System;
using System.Collections.Generic;
using AiDotNet.Diffusion.VAE;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// The PLR numeric-feature embedding of Gorishniy et al. (2022), "On Embeddings for Numerical Features in Tabular Deep
/// Learning", in the form TabR's reference code uses (<c>lib.deep.PLREmbeddings</c>): every scalar feature becomes
/// <c>ReLU(Linear([cos(2 pi c x), sin(2 pi c x)]))</c> with its own learned frequencies <c>c</c>.
/// </summary>
/// <remarks>
/// <para>
/// The periodic part holds frequencies <c>c</c> of shape [features, n_frequencies], drawn from
/// <c>N(0, frequency_scale^2)</c> as the reference <c>torch.normal(0.0, frequency_scale, ...)</c> does. The linear part
/// maps the 2 n_frequencies periodic values of a feature to d_embedding values: one <c>Linear(2k, d)</c> shared by all
/// features when <c>lite</c> (TabR's choice), or one per feature (<c>NLinear</c>, the original PLR) otherwise, each
/// initialized like <c>nn.Linear</c> (uniform in +-1/sqrt(2k)). The output is the [batch, features * d_embedding]
/// concatenation of the feature embeddings, which the encoder's first linear map consumes.
/// </para>
/// <para><b>For Beginners:</b> instead of feeding a raw number such as an age to the network, this turns it into a
/// small vector of learned waves, which lets the network pick up patterns that change quickly with the value.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class PeriodicNumericEmbedding<T> : LazyWeightModule<T>
{
    private readonly LazyWeight<T> _frequencies;
    private readonly LazyWeight<T> _weight;
    private readonly LazyWeight<T> _bias;

    internal PeriodicNumericEmbedding(int numFeatures, int numFrequencies, double frequencyScale, int embeddingDimension, bool lite)
    {
        if (numFeatures < 1) throw new ArgumentOutOfRangeException(nameof(numFeatures), "A numeric embedding needs the feature count.");
        if (numFrequencies < 1) throw new ArgumentOutOfRangeException(nameof(numFrequencies), "n_frequencies must be positive.");
        if (embeddingDimension < 1) throw new ArgumentOutOfRangeException(nameof(embeddingDimension), "d_embedding must be positive.");
        if (double.IsNaN(frequencyScale) || double.IsInfinity(frequencyScale) || frequencyScale <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frequencyScale), frequencyScale, "The frequency scale must be positive and finite.");
        }

        NumFeatures = numFeatures;
        NumFrequencies = numFrequencies;
        FrequencyScale = frequencyScale;
        EmbeddingDimension = embeddingDimension;
        Lite = lite;

        // Placeholders; Initialize draws the reference initialization from the owning model's generator
        _frequencies = LazyWeight<T>.Constant(new[] { numFeatures, numFrequencies }, 0.0);
        int periodicWidth = 2 * numFrequencies;
        _weight = lite
            ? LazyWeight<T>.Constant(new[] { periodicWidth, embeddingDimension }, 0.0)
            : LazyWeight<T>.Constant(new[] { numFeatures, periodicWidth, embeddingDimension }, 0.0);
        _bias = lite
            ? LazyWeight<T>.Constant(new[] { embeddingDimension }, 0.0)
            : LazyWeight<T>.Constant(new[] { numFeatures, embeddingDimension }, 0.0);
    }

    /// <summary>The number of scalar features embedded.</summary>
    internal int NumFeatures { get; }

    /// <summary>The reference <c>n_frequencies</c>.</summary>
    internal int NumFrequencies { get; }

    /// <summary>The reference <c>frequency_scale</c>: the standard deviation of the initial frequencies.</summary>
    internal double FrequencyScale { get; }

    /// <summary>The reference <c>d_embedding</c>: the width of each feature's embedding.</summary>
    internal int EmbeddingDimension { get; }

    /// <summary>The reference <c>lite</c>: one linear map shared by every feature instead of one per feature.</summary>
    internal bool Lite { get; }

    /// <summary>The width of the embedded row, <c>features * d_embedding</c>.</summary>
    internal int OutputWidth => NumFeatures * EmbeddingDimension;

    /// <summary>The frequencies <c>c</c>, [features, n_frequencies].</summary>
    internal Tensor<T> Frequencies => _frequencies.Value;

    /// <summary>The linear weight: [2 n_frequencies, d_embedding] when lite, else [features, 2 n_frequencies, d_embedding].</summary>
    internal Tensor<T> Weight => _weight.Value;

    /// <summary>The linear bias: [d_embedding] when lite, else [features, d_embedding].</summary>
    internal Tensor<T> Bias => _bias.Value;

    /// <summary>The live trainable tensors, for a tape training step.</summary>
    internal IEnumerable<Tensor<T>> Tensors
    {
        get
        {
            yield return Frequencies;
            yield return Weight;
            yield return Bias;
        }
    }

    /// <inheritdoc />
    protected override IEnumerable<LazyWeight<T>> OwnWeights()
    {
        yield return _frequencies;
        yield return _weight;
        yield return _bias;
    }

    /// <summary>
    /// The reference initialization: frequencies from <c>N(0, frequency_scale^2)</c>, the linear weight and bias uniform in
    /// +-1/sqrt(2 n_frequencies) (<c>nn.Linear</c>'s default for a fan-in of 2 n_frequencies).
    /// </summary>
    internal void Initialize(Random random)
    {
        if (random is null) throw new ArgumentNullException(nameof(random));

        var frequencies = Engine.TensorMultiplyScalar(
            AiDotNet.Training.CompiledStepRandom<T>.StandardNormal(new[] { (int)_frequencies.Count }, random),
            NumOps.FromDouble(FrequencyScale));
        double bound = 1.0 / Math.Sqrt(2.0 * NumFrequencies);
        var linear = AiDotNet.Training.CompiledStepRandom<T>.Uniform(
            new[] { (int)(_weight.Count + _bias.Count) }, random, -bound, bound);
        SetParameters(Engine.TensorConcatenate(new[] { frequencies, linear }, 0).ToVector());
    }

    /// <summary>
    /// Embeds a [batch, features] block of numeric features as [batch, features * d_embedding]:
    /// <c>ReLU(Linear(cat(cos(2 pi c x), sin(2 pi c x))))</c> per feature.
    /// </summary>
    internal Tensor<T> Forward(Tensor<T> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        int width = features.Shape[features.Shape.Length - 1];
        if (width != NumFeatures)
        {
            throw new ArgumentException(
                $"The numeric embedding was built for {NumFeatures} features but received {width}.", nameof(features));
        }

        int batch = features.Length / NumFeatures;
        int periodicWidth = 2 * NumFrequencies;

        // v = 2 pi c x, broadcast [batch, F, 1] * [1, F, k]
        var phases = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(
                Engine.Reshape(features, new[] { batch, NumFeatures, 1 }),
                Engine.Reshape(Frequencies, new[] { 1, NumFeatures, NumFrequencies })),
            NumOps.FromDouble(2.0 * Math.PI));
        var periodic = Engine.TensorConcatenate(new[] { Engine.TensorCos(phases), Engine.TensorSin(phases) }, 2);

        Tensor<T> embedded;
        if (Lite)
        {
            // One Linear(2k, d) for every feature
            var rows = Engine.TensorMatMul(Engine.Reshape(periodic, new[] { batch * NumFeatures, periodicWidth }), Weight);
            embedded = Engine.Reshape(
                Engine.TensorAdd(rows, Engine.Reshape(Bias, new[] { 1, EmbeddingDimension })),
                new[] { batch, NumFeatures, EmbeddingDimension });
        }
        else
        {
            // NLinear: feature f has its own Linear(2k, d), as one batched product over the feature axis
            var perFeature = Engine.TensorBatchMatMul(Engine.TensorPermute(periodic, new[] { 1, 0, 2 }), Weight);
            embedded = Engine.TensorAdd(
                Engine.TensorPermute(perFeature, new[] { 1, 0, 2 }),
                Engine.Reshape(Bias, new[] { 1, NumFeatures, EmbeddingDimension }));
        }

        return Engine.Reshape(Engine.ReLU(embedded), new[] { batch, OutputWidth });
    }
}
