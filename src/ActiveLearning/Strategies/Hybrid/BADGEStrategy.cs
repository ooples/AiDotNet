using AiDotNet.ActiveLearning.Config;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Strategies.Hybrid;

/// <summary>
/// BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BADGE is a state-of-the-art strategy that combines uncertainty
/// and diversity. It uses gradient embeddings (gradients of the loss with respect to
/// the final layer) to represent samples, then uses k-means++ to select diverse samples.</para>
///
/// <para><b>How BADGE Works:</b></para>
/// <list type="number">
/// <item><description>Compute hypothetical gradient for each unlabeled sample</description></item>
/// <item><description>Use the predicted class to form "pseudo-labels"</description></item>
/// <item><description>Gradient magnitude captures uncertainty, direction captures features</description></item>
/// <item><description>Apply k-means++ on gradient embeddings to get diverse batch</description></item>
/// </list>
///
/// <para><b>Key Insight:</b> Gradient embeddings naturally combine model uncertainty
/// (large gradients for uncertain samples) with feature diversity (different gradient
/// directions for different samples).</para>
///
/// <para><b>When to Use:</b></para>
/// <list type="bullet">
/// <item><description>Deep learning models where gradients are available</description></item>
/// <item><description>Need both uncertainty and diversity in batch selection</description></item>
/// <item><description>State-of-the-art performance is important</description></item>
/// </list>
///
/// <para><b>Reference:</b> Ash et al. "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds" (ICLR 2020)</para>
/// </remarks>
public class BADGEStrategy<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numClasses;
    private readonly ActiveLearnerConfig<T>? _config;

    /// <inheritdoc/>
    public string Name => "BADGE (Batch Active learning by Diverse Gradient Embeddings)";

    /// <inheritdoc/>
    public string Description =>
        "Combines uncertainty and diversity using gradient embeddings with k-means++ sampling";

    /// <summary>
    /// Initializes a new BADGE strategy.
    /// </summary>
    /// <param name="numClasses">Number of output classes.</param>
    public BADGEStrategy(int numClasses)
        : this(numClasses, null)
    {
    }

    /// <summary>
    /// Initializes a new BADGE strategy with configuration.
    /// </summary>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="config">Optional configuration.</param>
    public BADGEStrategy(int numClasses, ActiveLearnerConfig<T>? config)
    {
        _numClasses = numClasses > 0 ? numClasses : 2;
        _config = config;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeScores(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool)
    {
        // For BADGE, scores are the gradient magnitudes (uncertainty component)
        var scores = new T[unlabeledPool.Count];

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            var gradientEmbedding = ComputeGradientEmbedding(model, input);
            scores[i] = ComputeNorm(gradientEmbedding);
        }

        return new Vector<T>(scores);
    }

    /// <inheritdoc/>
    public int[] SelectSamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        var batchSizeToUse = Math.Min(batchSize, unlabeledPool.Count);

        // Step 1: Compute gradient embeddings for all unlabeled samples
        var gradientEmbeddings = new List<Vector<T>>();
        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            var embedding = ComputeGradientEmbedding(model, input);
            gradientEmbeddings.Add(embedding);
        }

        // Step 2: Apply k-means++ initialization to select diverse batch
        var selectedIndices = KMeansPlusPlusInit(gradientEmbeddings, batchSizeToUse);

        return selectedIndices;
    }

    /// <inheritdoc/>
    public void UpdateState(int[] newlyLabeledIndices, TOutput[] labels)
    {
        // BADGE is stateless
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // BADGE is stateless
    }

    #region Private Methods

    private Vector<T> ComputeGradientEmbedding(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        // Get prediction probabilities
        var prediction = model.Predict(input);
        var probabilities = ConvertToProbabilities(prediction);

        // Get the predicted class (pseudo-label)
        int predictedClass = GetArgMax(probabilities);

        // If model supports gradient computation, use actual gradients
        if (model is IGradientModel<T, TInput, TOutput> gradientModel)
        {
            return gradientModel.ComputeGradientEmbedding(input, predictedClass);
        }

        // Fallback: approximate gradient embedding
        // g = (p - e_y) ⊗ φ(x)
        // where e_y is one-hot of predicted class, φ(x) is feature embedding
        return ApproximateGradientEmbedding(model, input, probabilities, predictedClass);
    }

    private Vector<T> ApproximateGradientEmbedding(
        IFullModel<T, TInput, TOutput> model,
        TInput input,
        Vector<T> probabilities,
        int predictedClass)
    {
        // Create one-hot vector for predicted class
        var oneHot = new T[probabilities.Length];
        for (int i = 0; i < oneHot.Length; i++)
        {
            oneHot[i] = i == predictedClass ? NumOps.One : NumOps.Zero;
        }

        // Gradient indicator: p - e_y
        var gradientIndicator = new T[probabilities.Length];
        for (int i = 0; i < probabilities.Length; i++)
        {
            gradientIndicator[i] = NumOps.Subtract(probabilities[i], oneHot[i]);
        }

        // Get feature embedding
        Vector<T> features;
        if (model is IFeatureExtractor<T, TInput> featureExtractor)
        {
            features = featureExtractor.ExtractFeatures(input);
        }
        else if (input is Vector<T> vectorInput)
        {
            features = vectorInput;
        }
        else
        {
            // Use prediction as features
            features = probabilities;
        }

        // Outer product: gradient embedding = gradientIndicator ⊗ features
        // This creates a vector of length (numClasses * numFeatures)
        var embedding = new T[gradientIndicator.Length * features.Length];
        int idx = 0;
        for (int i = 0; i < gradientIndicator.Length; i++)
        {
            for (int j = 0; j < features.Length; j++)
            {
                embedding[idx++] = NumOps.Multiply(gradientIndicator[i], features[j]);
            }
        }

        return new Vector<T>(embedding);
    }

    private int[] KMeansPlusPlusInit(List<Vector<T>> embeddings, int k)
    {
        if (embeddings.Count <= k)
        {
            return Enumerable.Range(0, embeddings.Count).ToArray();
        }

        var selected = new List<int>();
        var minDistances = new T[embeddings.Count];

        // Initialize all distances to infinity
        for (int i = 0; i < embeddings.Count; i++)
        {
            minDistances[i] = NumOps.MaxValue;
        }

        // Choose first center randomly (weighted by gradient magnitude for BADGE)
        int firstCenter = SelectFirstCenter(embeddings);
        selected.Add(firstCenter);

        // Update distances
        for (int i = 0; i < embeddings.Count; i++)
        {
            var dist = ComputeSquaredDistance(embeddings[i], embeddings[firstCenter]);
            minDistances[i] = dist;
        }

        // Choose remaining k-1 centers
        while (selected.Count < k)
        {
            // Sample proportional to squared distance
            int nextCenter = SampleProportionalToDistance(minDistances, selected);
            selected.Add(nextCenter);

            // Update minimum distances
            for (int i = 0; i < embeddings.Count; i++)
            {
                if (!selected.Contains(i))
                {
                    var dist = ComputeSquaredDistance(embeddings[i], embeddings[nextCenter]);
                    if (NumOps.Compare(dist, minDistances[i]) < 0)
                    {
                        minDistances[i] = dist;
                    }
                }
            }
        }

        return selected.ToArray();
    }

    private int SelectFirstCenter(List<Vector<T>> embeddings)
    {
        // For BADGE, select first center weighted by gradient magnitude (uncertainty)
        var magnitudes = new double[embeddings.Count];
        double totalMag = 0;

        for (int i = 0; i < embeddings.Count; i++)
        {
            magnitudes[i] = NumOps.ToDouble(ComputeNorm(embeddings[i]));
            totalMag += magnitudes[i];
        }

        if (totalMag <= 0)
        {
            return ThreadRandom.Next(embeddings.Count);
        }

        // Sample proportional to magnitude
        double threshold = ThreadRandom.NextDouble() * totalMag;
        double cumSum = 0;

        for (int i = 0; i < embeddings.Count; i++)
        {
            cumSum += magnitudes[i];
            if (cumSum >= threshold)
            {
                return i;
            }
        }

        return embeddings.Count - 1;
    }

    private int SampleProportionalToDistance(T[] distances, List<int> excluded)
    {
        double totalDist = 0;
        var validIndices = new List<int>();

        for (int i = 0; i < distances.Length; i++)
        {
            if (!excluded.Contains(i))
            {
                var d = NumOps.ToDouble(distances[i]);
                if (d > 0 && !double.IsInfinity(d))
                {
                    totalDist += d;
                    validIndices.Add(i);
                }
            }
        }

        if (totalDist <= 0 || validIndices.Count == 0)
        {
            // Fallback: random selection from remaining
            var remaining = Enumerable.Range(0, distances.Length)
                .Where(i => !excluded.Contains(i))
                .ToList();
            return remaining[ThreadRandom.Next(remaining.Count)];
        }

        // Sample proportional to distance
        double threshold = ThreadRandom.NextDouble() * totalDist;
        double cumSum = 0;

        foreach (var i in validIndices)
        {
            cumSum += NumOps.ToDouble(distances[i]);
            if (cumSum >= threshold)
            {
                return i;
            }
        }

        return validIndices[validIndices.Count - 1];
    }

    private T ComputeSquaredDistance(Vector<T> a, Vector<T> b)
    {
        int length = Math.Min(a.Length, b.Length);
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return sum;
    }

    private T ComputeNorm(Vector<T> v)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(v[i], v[i]));
        }
        return NumOps.Sqrt(sum);
    }

    private int GetArgMax(Vector<T> v)
    {
        if (v.Length == 0) return 0;

        int maxIdx = 0;
        T maxVal = v[0];

        for (int i = 1; i < v.Length; i++)
        {
            if (NumOps.Compare(v[i], maxVal) > 0)
            {
                maxVal = v[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    private Vector<T> ConvertToProbabilities(TOutput prediction)
    {
        if (prediction is Vector<T> vectorPred)
        {
            return Softmax(vectorPred);
        }

        var value = ConvertToNumeric(prediction);
        var p = Sigmoid(value);
        var oneMinusP = NumOps.Subtract(NumOps.One, p);
        return new Vector<T>(new[] { oneMinusP, p });
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.Compare(logits[i], maxLogit) > 0)
            {
                maxLogit = logits[i];
            }
        }

        var expValues = new T[logits.Length];
        T sum = NumOps.Zero;

        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.Subtract(logits[i], maxLogit);
            expValues[i] = NumOps.Exp(shifted);
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < expValues.Length; i++)
        {
            expValues[i] = NumOps.Divide(expValues[i], sum);
        }

        return new Vector<T>(expValues);
    }

    private T Sigmoid(T x)
    {
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        var denominator = NumOps.Add(NumOps.One, expNegX);
        return NumOps.Divide(NumOps.One, denominator);
    }

    private T ConvertToNumeric(TOutput output)
    {
        if (output is T typedValue) return typedValue;
        if (output is double doubleValue) return NumOps.FromDouble(doubleValue);
        if (output is float floatValue) return NumOps.FromDouble(floatValue);
        if (output is int intValue) return NumOps.FromDouble(intValue);
        return NumOps.Zero;
    }

    #endregion
}

/// <summary>
/// Interface for models that can compute gradient embeddings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IGradientModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Computes the gradient embedding for a sample.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <param name="predictedClass">The predicted class (for pseudo-labeling).</param>
    /// <returns>The gradient embedding vector.</returns>
    Vector<T> ComputeGradientEmbedding(TInput input, int predictedClass);

    /// <summary>
    /// Computes the gradient of the loss with respect to model parameters.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <param name="target">The target label.</param>
    /// <returns>Gradient vector.</returns>
    Vector<T> ComputeGradient(TInput input, TOutput target);
}
