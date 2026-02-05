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
public class ContrastivePretraining<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private readonly int _numFeatures;
    private readonly double _corruptionRate;
    private readonly double _temperature;

    // Projection head for contrastive learning
    private Tensor<T> _projectionWeights;
    private Tensor<T> _projectionBias;
    private readonly int _projectionDim;

    // Cached values
    private Tensor<T>? _originalEmbeddingsCache;
    private Tensor<T>? _corruptedEmbeddingsCache;
    private int[]? _corruptedIndicesCache;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount => _projectionWeights.Length + _projectionBias.Length;

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

        for (int i = 0; i < _projectionWeights.Length; i++)
        {
            _projectionWeights[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
        }

        for (int i = 0; i < _projectionBias.Length; i++)
        {
            _projectionBias[i] = NumOps.Zero;
        }
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

        var corrupted = new Tensor<T>(input.Shape);
        var corruptedIndices = new List<int>();

        // Copy original values
        for (int i = 0; i < input.Length; i++)
        {
            corrupted[i] = input[i];
        }

        // Corrupt features by swapping with values from other samples
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                if (_random.NextDouble() < _corruptionRate)
                {
                    // Pick a random sample to swap from
                    int otherSample = _random.Next(batchSize);
                    while (otherSample == b && batchSize > 1)
                    {
                        otherSample = _random.Next(batchSize);
                    }

                    corrupted[b * numFeatures + f] = input[otherSample * numFeatures + f];
                    corruptedIndices.Add(b * numFeatures + f);
                }
            }
        }

        _corruptedIndicesCache = corruptedIndices.ToArray();
        return corrupted;
    }

    /// <summary>
    /// Computes contrastive loss between original and corrupted embeddings.
    /// </summary>
    /// <param name="originalEmbeddings">Original sample embeddings [batchSize, embeddingDim].</param>
    /// <param name="corruptedEmbeddings">Corrupted sample embeddings [batchSize, embeddingDim].</param>
    /// <returns>Contrastive loss value.</returns>
    public T ComputeContrastiveLoss(Tensor<T> originalEmbeddings, Tensor<T> corruptedEmbeddings)
    {
        _originalEmbeddingsCache = originalEmbeddings;
        _corruptedEmbeddingsCache = corruptedEmbeddings;

        int batchSize = originalEmbeddings.Shape[0];
        int embDim = originalEmbeddings.Shape[1];

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
        var projected = new Tensor<T>([batchSize, _projectionDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < _projectionDim; p++)
            {
                var sum = _projectionBias[p];
                for (int d = 0; d < embDim; d++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(
                        embeddings[b * embDim + d],
                        _projectionWeights[d * _projectionDim + p]));
                }
                projected[b * _projectionDim + p] = sum;
            }
        }

        return projected;
    }

    private Tensor<T> L2Normalize(Tensor<T> embeddings, int batchSize, int dim)
    {
        var normalized = new Tensor<T>(embeddings.Shape);
        var epsilon = NumOps.FromDouble(1e-8);

        for (int b = 0; b < batchSize; b++)
        {
            // Compute L2 norm
            var norm = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                var val = embeddings[b * dim + d];
                norm = NumOps.Add(norm, NumOps.Multiply(val, val));
            }
            norm = NumOps.Sqrt(NumOps.Add(norm, epsilon));

            // Normalize
            for (int d = 0; d < dim; d++)
            {
                normalized[b * dim + d] = NumOps.Divide(embeddings[b * dim + d], norm);
            }
        }

        return normalized;
    }

    private T ComputeInfoNCELoss(Tensor<T> original, Tensor<T> corrupted, int batchSize)
    {
        // Compute similarity matrix
        var similarities = new T[batchSize, batchSize * 2];
        var tempScale = NumOps.FromDouble(1.0 / _temperature);

        for (int i = 0; i < batchSize; i++)
        {
            // Similarities with original samples
            for (int j = 0; j < batchSize; j++)
            {
                var sim = NumOps.Zero;
                for (int d = 0; d < _projectionDim; d++)
                {
                    sim = NumOps.Add(sim, NumOps.Multiply(
                        original[i * _projectionDim + d],
                        original[j * _projectionDim + d]));
                }
                similarities[i, j] = NumOps.Multiply(sim, tempScale);
            }

            // Similarities with corrupted samples
            for (int j = 0; j < batchSize; j++)
            {
                var sim = NumOps.Zero;
                for (int d = 0; d < _projectionDim; d++)
                {
                    sim = NumOps.Add(sim, NumOps.Multiply(
                        original[i * _projectionDim + d],
                        corrupted[j * _projectionDim + d]));
                }
                similarities[i, batchSize + j] = NumOps.Multiply(sim, tempScale);
            }
        }

        // Compute cross-entropy loss (positive pairs are diagonal original-original)
        var totalLoss = NumOps.Zero;
        for (int i = 0; i < batchSize; i++)
        {
            // Find max for numerical stability
            var maxSim = similarities[i, 0];
            for (int j = 1; j < batchSize * 2; j++)
            {
                if (NumOps.Compare(similarities[i, j], maxSim) > 0)
                    maxSim = similarities[i, j];
            }

            // Compute log-sum-exp
            var sumExp = NumOps.Zero;
            for (int j = 0; j < batchSize * 2; j++)
            {
                sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(similarities[i, j], maxSim)));
            }

            // Loss = -log(exp(pos) / sum(exp)) = -pos + log(sum(exp))
            var posScore = NumOps.Subtract(similarities[i, i], maxSim); // Diagonal is positive pair
            var loss = NumOps.Subtract(
                NumOps.Add(NumOps.Log(sumExp), maxSim),
                NumOps.Add(posScore, maxSim));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
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

        var loss = NumOps.Zero;
        foreach (var idx in _corruptedIndicesCache)
        {
            var diff = NumOps.Subtract(reconstructed[idx], original[idx]);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(_corruptedIndicesCache.Length));
    }

    /// <summary>
    /// Gets the corruption mask indicating which features were corrupted.
    /// </summary>
    /// <param name="batchSize">Batch size.</param>
    /// <returns>Mask tensor [batchSize, numFeatures] with 1 for corrupted, 0 for original.</returns>
    public Tensor<T> GetCorruptionMask(int batchSize)
    {
        var mask = new Tensor<T>([batchSize, _numFeatures]);

        if (_corruptedIndicesCache != null)
        {
            foreach (var idx in _corruptedIndicesCache)
            {
                mask[idx] = NumOps.One;
            }
        }

        return mask;
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate, Tensor<T> gradWeights, Tensor<T> gradBias)
    {
        for (int i = 0; i < _projectionWeights.Length; i++)
        {
            _projectionWeights[i] = NumOps.Subtract(_projectionWeights[i],
                NumOps.Multiply(learningRate, gradWeights[i]));
        }

        for (int i = 0; i < _projectionBias.Length; i++)
        {
            _projectionBias[i] = NumOps.Subtract(_projectionBias[i],
                NumOps.Multiply(learningRate, gradBias[i]));
        }
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
