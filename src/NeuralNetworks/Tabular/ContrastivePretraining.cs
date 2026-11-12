using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Contrastive pretraining module for SAINT architecture.
/// </summary>
/// <remarks>
/// <para>
/// SAINT uses contrastive learning as a self-supervised pretraining objective.
/// The model learns to distinguish between original samples and corrupted versions,
/// which helps learn meaningful representations without labels.
/// </para>
/// <para>
/// <b>For Beginners:</b> Contrastive learning is like a "spot the difference" game:
/// 1. Take an original sample
/// 2. Create a corrupted version (swap some feature values with others)
/// 3. Train the model to tell them apart
///
/// This helps the model learn which features are important and how they relate,
/// without needing labels. It's especially useful when you have lots of unlabeled data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class ContrastivePretraining<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private IEngine Engine => AiDotNetEngine.Current;
    private readonly Random _random;

    private readonly int _numFeatures;
    private readonly double _corruptionRate;
    private readonly double _temperature;

    // Projection head for contrastive learning
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _projectionWeights;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _projectionBias;
    private readonly int _projectionDim;

    // Cached values
    [Scratch]
    private Tensor<T>? _originalEmbeddingsCache;
    [Scratch]
    private Tensor<T>? _corruptedEmbeddingsCache;
    private int[]? _corruptedIndicesCache;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long ParameterCount => _projectionWeights.Length + _projectionBias.Length;

    /// <summary>
    /// Initializes contrastive pretraining module.
    /// </summary>
    /// <param name="embeddingDim">Input embedding dimension.</param>
    /// <param name="numFeatures">Number of features.</param>
    /// <param name="projectionDim">Dimension of projection head output.</param>
    /// <param name="corruptionRate">Rate of feature corruption (default 0.3).</param>
    /// <param name="temperature">Temperature for contrastive loss (default 0.1).</param>
    public ContrastivePretraining(
        int embeddingDim,
        int numFeatures,
        int projectionDim = 128,
        double corruptionRate = 0.3,
        double temperature = 0.1)
    {
        if (temperature <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive.");
        }

        if (corruptionRate < 0 || corruptionRate > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(corruptionRate), "Corruption rate must be in [0, 1].");
        }

        _numFeatures = numFeatures;
        _projectionDim = projectionDim;
        _corruptionRate = corruptionRate;
        _temperature = temperature;
        _random = RandomHelper.CreateSecureRandom();

        // Initialize projection head
        _projectionWeights = new Tensor<T>([embeddingDim, projectionDim]);
        _projectionBias = new Tensor<T>([projectionDim]);

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        double scale = Math.Sqrt(2.0 / (_projectionWeights.Shape[0] + _projectionWeights.Shape[1]));

        // N(0, scale^2) in one engine call, seeded from this module's (secure) Random.
        Engine.TensorNormalInPlace(_projectionWeights, 0.0, scale, _random.Next());

        _projectionBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Creates corrupted samples for contrastive learning.
    /// </summary>
    /// <param name="input">Original input [batchSize, numFeatures].</param>
    /// <returns>Corrupted input with some features swapped.</returns>
    public Tensor<T> CorruptSamples(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int numFeatures = input.Shape[1];

        if (numFeatures != _numFeatures)
        {
            throw new ArgumentException(
                $"Input has {numFeatures} features but expected {_numFeatures}.");
        }

        // Corrupt features by swapping with values from other samples
        // Note: When batchSize == 1, swapping from the same sample produces no real corruption
        if (batchSize <= 1)
        {
            _corruptedIndicesCache = [];
            return input.Clone();
        }

        // Fresh, detached output as before. Each cell is corrupted with probability _corruptionRate,
        // taking the same feature from a uniformly chosen different sample.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var shape = new[] { batchSize, numFeatures };
            var corruptMask = new Tensor<T>(shape);
            Engine.TensorBernoulliInPlace(corruptMask, _corruptionRate, _random.Next());

            // Draw from the batchSize - 1 other rows: r in [0, batchSize - 1), shifted past the row itself.
            var draw = Engine.TensorRandint<int>(0, batchSize - 1, shape, _random.Next());
            var rowIndex = Engine.TensorBroadcastTo(
                Engine.Reshape(Engine.TensorArange<int>(0, batchSize, 1), new[] { batchSize, 1 }), shape);
            var otherSample = Engine.TensorAdd(draw, Engine.TensorGreaterEqual(draw, rowIndex));

            var swapped = Engine.TensorTakeAlongDim(input, otherSample, 0);
            var corrupted = Engine.TensorWhere(corruptMask, swapped, input);

            // Flat positions of the corrupted cells, in row-major order.
            _corruptedIndicesCache = Engine.TensorNonzero(Engine.Reshape(corruptMask, new[] { batchSize * numFeatures })).ToArray();
            return corrupted;
        }
    }

    /// <summary>
    /// Computes contrastive loss between original and corrupted embeddings.
    /// </summary>
    /// <param name="originalEmbeddings">Original sample embeddings [batchSize, embeddingDim].</param>
    /// <param name="corruptedEmbeddings">Corrupted sample embeddings [batchSize, embeddingDim].</param>
    /// <returns>Contrastive loss value.</returns>
    public T ComputeContrastiveLoss(Tensor<T> originalEmbeddings, Tensor<T> corruptedEmbeddings)
    {
        if (originalEmbeddings.Shape[0] != corruptedEmbeddings.Shape[0] ||
            originalEmbeddings.Shape[1] != corruptedEmbeddings.Shape[1])
        {
            throw new ArgumentException(
                $"Shape mismatch: original {originalEmbeddings.Shape[0]}x{originalEmbeddings.Shape[1]} " +
                $"vs corrupted {corruptedEmbeddings.Shape[0]}x{corruptedEmbeddings.Shape[1]}.");
        }

        int embDim = originalEmbeddings.Shape[1];
        if (embDim != _projectionWeights.Shape[0])
        {
            throw new ArgumentException(
                $"Embedding dimension ({embDim}) does not match projection input dimension ({_projectionWeights.Shape[0]}).");
        }

        _originalEmbeddingsCache = originalEmbeddings;
        _corruptedEmbeddingsCache = corruptedEmbeddings;

        int batchSize = originalEmbeddings.Shape[0];

        // Project embeddings
        var originalProjected = ProjectEmbeddings(originalEmbeddings, batchSize, embDim);
        var corruptedProjected = ProjectEmbeddings(corruptedEmbeddings, batchSize, embDim);

        // L2 normalize projections
        originalProjected = L2Normalize(originalProjected, batchSize, _projectionDim);
        corruptedProjected = L2Normalize(corruptedProjected, batchSize, _projectionDim);

        // Compute InfoNCE loss
        var loss = ComputeInfoNCELoss(originalProjected, corruptedProjected, batchSize);

        return loss;
    }

    private Tensor<T> ProjectEmbeddings(Tensor<T> embeddings, int batchSize, int embDim)
    {
        // Fresh, detached output as before: x @ W + b.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var projected = Engine.TensorMatMul(
                Engine.Reshape(embeddings, new[] { batchSize, embDim }),
                Engine.Reshape(_projectionWeights, new[] { embDim, _projectionDim }));
            return Engine.TensorAdd(projected, Engine.TensorBroadcastTo(
                Engine.Reshape(_projectionBias, new[] { 1, _projectionDim }), new[] { batchSize, _projectionDim }));
        }
    }

    private Tensor<T> L2Normalize(Tensor<T> embeddings, int batchSize, int dim)
    {
        var epsilon = NumOps.FromDouble(1e-8);

        // Fresh, detached output as before: x / sqrt(sum x^2 + eps) per row.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var rows = Engine.Reshape(embeddings, new[] { batchSize, dim });
            var norm = Engine.TensorSqrt(Engine.TensorAddScalar(
                Engine.ReduceSum(Engine.TensorMultiply(rows, rows), new[] { 1 }, true), epsilon));
            var normalized = Engine.TensorDivide(rows, Engine.TensorBroadcastTo(norm, new[] { batchSize, dim }));
            return Engine.Reshape(normalized, embeddings._shape);
        }
    }

    private T ComputeInfoNCELoss(Tensor<T> original, Tensor<T> corrupted, int batchSize)
    {
        var tempScale = NumOps.FromDouble(1.0 / _temperature);
        var rowShape = new[] { batchSize, 2 * batchSize };

        // Similarity matrix [anchor, (original | corrupted)] scaled by 1 / temperature.
        var o = Engine.Reshape(original, new[] { batchSize, _projectionDim });
        var c = Engine.Reshape(corrupted, new[] { batchSize, _projectionDim });
        var similarities = Engine.TensorConcatenate(new[]
        {
            Engine.TensorMultiplyScalar(Engine.TensorMatMul(o, Engine.TensorTranspose(o)), tempScale),
            Engine.TensorMultiplyScalar(Engine.TensorMatMul(o, Engine.TensorTranspose(c)), tempScale),
        }, 1);

        // Self-similarity (column i of row i) is excluded from the max and the denominator. Filling
        // it with the row's first corrupted entry keeps the max the one over the other columns.
        var self = Engine.TensorOneHot<T>(Engine.TensorArange<int>(0, batchSize, 1), 2 * batchSize);
        var firstCorrupted = Engine.TensorBroadcastTo(
            Engine.TensorSlice(similarities, new[] { 0, batchSize }, new[] { batchSize, 1 }), rowShape);
        var maskedSimilarities = Engine.TensorWhere(self, firstCorrupted, similarities);
        var maxSim = Engine.ReduceMax(maskedSimilarities, new[] { 1 }, true);

        // Compute log-sum-exp (excluding self-similarity)
        // Exclude self before exponentiation: an overflowing exp(self) times zero is still NaN.
        var shifted = Engine.TensorExp(Engine.TensorSubtract(maskedSimilarities, Engine.TensorBroadcastTo(maxSim, rowShape)));
        var keep = Engine.TensorAddScalar(Engine.TensorNegate(self), NumOps.One);
        var sumExp = Engine.ReduceSum(Engine.TensorMultiply(shifted, keep), new[] { 1 }, true);

        // Loss = -log(exp(pos) / sum(exp)) = -pos + log(sum(exp)); the positive is (i, batchSize + i).
        var positiveIndex = Engine.Reshape(Engine.TensorArange<int>(batchSize, 2 * batchSize, 1), new[] { batchSize, 1 });
        var posScore = Engine.TensorSubtract(Engine.TensorTakeAlongDim(similarities, positiveIndex, 1), maxSim);
        var loss = Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorLog(sumExp), maxSim),
            Engine.TensorAdd(posScore, maxSim));

        return NumOps.Divide(Engine.TensorSum(loss), NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes denoising loss for reconstructing corrupted features.
    /// </summary>
    /// <param name="reconstructed">Reconstructed features [batchSize, numFeatures].</param>
    /// <param name="original">Original features [batchSize, numFeatures].</param>
    /// <returns>Denoising loss value.</returns>
    public T ComputeDenoisingLoss(Tensor<T> reconstructed, Tensor<T> original)
    {
        if (_corruptedIndicesCache == null || _corruptedIndicesCache.Length == 0)
        {
            return NumOps.Zero;
        }

        if (reconstructed.Length != original.Length)
        {
            throw new ArgumentException(
                $"Shape mismatch: reconstructed length ({reconstructed.Length}) != original length ({original.Length}).");
        }

        int maxIdx = 0;
        foreach (var idx in _corruptedIndicesCache)
            if (idx > maxIdx) maxIdx = idx;
        if (maxIdx >= reconstructed.Length)
        {
            throw new ArgumentException(
                $"Cached corruption index ({maxIdx}) exceeds tensor length ({reconstructed.Length}).");
        }

        // Squared error over the corrupted cells only.
        var positions = Tensor<int>.FromVector(new Vector<int>(_corruptedIndicesCache));
        var diff = Engine.TensorSubtract(Engine.TensorTake(reconstructed, positions), Engine.TensorTake(original, positions));
        var loss = Engine.TensorSum(Engine.TensorMultiply(diff, diff));

        return NumOps.Divide(loss, NumOps.FromDouble(_corruptedIndicesCache.Length));
    }

    /// <summary>
    /// Gets the corruption mask indicating which features were corrupted.
    /// </summary>
    /// <param name="batchSize">Batch size.</param>
    /// <returns>Mask tensor [batchSize, numFeatures] with 1 for corrupted, 0 for original.</returns>
    public Tensor<T> GetCorruptionMask(int batchSize)
    {
        int maskLength = batchSize * _numFeatures;
        var mask = new Tensor<T>([batchSize, _numFeatures]);

        var inRange = _corruptedIndicesCache?.Where(idx => idx >= 0 && idx < maskLength).ToArray() ?? Array.Empty<int>();
        if (inRange.Length == 0)
        {
            return mask;
        }

        // Ones at the corrupted positions (they are distinct, so index-add sets them to exactly 1).
        var ones = new Tensor<T>(new[] { inRange.Length });
        ones.Fill(NumOps.One);
        var flat = Engine.TensorScatterAdd(
            new Tensor<T>(new[] { maskLength }), Tensor<int>.FromVector(new Vector<int>(inRange)), ones, 0);
        return Engine.Reshape(flat, new[] { batchSize, _numFeatures });
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate, Tensor<T> gradWeights, Tensor<T> gradBias)
    {
        if (gradWeights.Length != _projectionWeights.Length)
        {
            throw new ArgumentException(
                $"gradWeights length ({gradWeights.Length}) does not match projection weights length ({_projectionWeights.Length}).");
        }

        if (gradBias.Length != _projectionBias.Length)
        {
            throw new ArgumentException(
                $"gradBias length ({gradBias.Length}) does not match projection bias length ({_projectionBias.Length}).");
        }

        Engine.TensorSubtractInPlace(_projectionWeights,
            Engine.TensorMultiplyScalar(gradWeights, learningRate));
        Engine.TensorSubtractInPlace(_projectionBias,
            Engine.TensorMultiplyScalar(gradBias, learningRate));
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _originalEmbeddingsCache = null;
        _corruptedEmbeddingsCache = null;
        _corruptedIndicesCache = null;
    }
}
