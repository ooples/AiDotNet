using AiDotNet.ActiveLearning.Config;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds", "https://arxiv.org/abs/1906.03671", Year = 2020, Authors = "Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, Alekh Agarwal")]
[ComponentType(ComponentType.ActiveLearner)]
[PipelineStage(PipelineStage.Training)]
public class BADGEStrategy<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected static IEngine Engine => AiDotNetEngine.Current;

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
        var scores = new List<T>(unlabeledPool.Count);

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            var gradientEmbedding = ComputeGradientEmbedding(model, input);
            scores.Add(VectorHelper.L2Norm(gradientEmbedding));
        }

        return Vector<T>.FromList(scores);
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
        var engine = AiDotNetEngine.Current;

        // Create one-hot vector for predicted class
        var oneHot = new Vector<T>(probabilities.Length);
        if (predictedClass >= 0 && predictedClass < oneHot.Length)
        {
            oneHot[predictedClass] = NumOps.One;
        }

        // Gradient indicator: p - e_y
        var gradientIndicator = engine.Subtract(probabilities, oneHot);

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
        // laid out row-major as embedding[i * numFeatures + j] = gradientIndicator[i] * features[j]
        if (gradientIndicator.Length == 0 || features.Length == 0)
        {
            return new Vector<T>(0);
        }

        var outer = engine.TensorOuter(Tensor<T>.FromVector(gradientIndicator), Tensor<T>.FromVector(features));
        return new Vector<T>(outer.ToArray());
    }

    private int[] KMeansPlusPlusInit(List<Vector<T>> embeddings, int k)
    {
        if (embeddings.Count <= k)
        {
            return Enumerable.Range(0, embeddings.Count).ToArray();
        }

        var engine = AiDotNetEngine.Current;
        // k-means++ is invariant to a global rescale, which keeps the squared distances finite in Half
        var rows = ActiveLearningEngineOps.UnitScale(ActiveLearningEngineOps.StackVectors(embeddings));
        int n = rows.Shape[0];
        int dim = rows.Shape[1];
        var selected = new List<int>();
        var isSelected = new bool[n];

        // Choose first center randomly (weighted by gradient magnitude for BADGE)
        int firstCenter = SelectFirstCenter(rows);
        selected.Add(firstCenter);
        isSelected[firstCenter] = true;

        // Squared distance from every embedding to its nearest center (selected entries are never sampled)
        var minDistances = SquaredDistancesToRow(rows, firstCenter, dim);

        // Choose remaining k-1 centers
        while (selected.Count < k)
        {
            // Sample proportional to squared distance
            int nextCenter = SampleProportionalToDistance(minDistances, isSelected);
            selected.Add(nextCenter);
            isSelected[nextCenter] = true;

            // Update minimum distances
            minDistances = engine.TensorMin(minDistances, SquaredDistancesToRow(rows, nextCenter, dim));
        }

        return selected.ToArray();
    }

    private static Tensor<T> SquaredDistancesToRow(Tensor<T> rows, int index, int dim)
    {
        var center = AiDotNetEngine.Current.TensorSlice(rows, new[] { index, 0 }, new[] { 1, dim });
        return ActiveLearningEngineOps.SquaredDistances(rows, center).Reshape(new[] { rows.Shape[0] });
    }

    private int SelectFirstCenter(Tensor<T> rows)
    {
        // For BADGE, select first center weighted by gradient magnitude (uncertainty)
        var engine = AiDotNetEngine.Current;
        int n = rows.Shape[0];
        var cumulative = engine.TensorCumSum(ActiveLearningEngineOps.RowNorms(rows), 0);
        T totalMag = cumulative.GetFlat(n - 1);

        if (!NumOps.GreaterThan(totalMag, NumOps.Zero))
        {
            return ThreadRandom.Next(n);
        }

        // Sample proportional to magnitude: first sample whose cumulative magnitude reaches the threshold
        var threshold = new Tensor<T>(new[] { 1 });
        threshold.Fill(NumOps.Multiply(NumOps.FromDouble(ThreadRandom.NextDouble()), totalMag));
        int index = engine.TensorSearchSorted(cumulative, threshold).GetFlat(0);

        return index < n ? index : n - 1;
    }

    private int SampleProportionalToDistance(Tensor<T> distances, bool[] excluded)
    {
        var engine = AiDotNetEngine.Current;

        // Valid candidates: not excluded, positive and finite (MaxValue is the largest finite value)
        var positiveFinite = engine.TensorMultiply(
            engine.TensorGreaterThan(distances, NumOps.Zero),
            engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(distances, NumOps.MaxValue)));
        var validMask = engine.TensorMaskedFill(positiveFinite, excluded, NumOps.Zero);
        var validIndices = engine.TensorNonzero(validMask).ToArray();

        T totalDist = NumOps.Zero;
        Tensor<T>? cumulative = null;
        if (validIndices.Length > 0)
        {
            var validDistances = engine.TensorIndexSelect(distances, new Tensor<int>(validIndices, new[] { validIndices.Length }), 0);
            cumulative = engine.TensorCumSum(validDistances, 0);
            totalDist = cumulative.GetFlat(validIndices.Length - 1);
        }

        if (cumulative is null || !NumOps.GreaterThan(totalDist, NumOps.Zero))
        {
            // Fallback: random selection from remaining
            var remaining = excluded
                .Select((isExcluded, index) => (IsExcluded: isExcluded, Index: index))
                .Where(x => !x.IsExcluded)
                .Select(x => x.Index)
                .ToList();
            return remaining[ThreadRandom.Next(remaining.Count)];
        }

        // Sample proportional to distance: first valid sample whose cumulative distance reaches the threshold
        var threshold = new Tensor<T>(new[] { 1 });
        threshold.Fill(NumOps.Multiply(NumOps.FromDouble(ThreadRandom.NextDouble()), totalDist));
        int position = engine.TensorSearchSorted(cumulative, threshold).GetFlat(0);

        return validIndices[Math.Min(position, validIndices.Length - 1)];
    }

    private int GetArgMax(Vector<T> v)
    {
        if (v.Length == 0) return 0;

        // First maximum
        var row = Tensor<T>.FromVector(v).Reshape(new[] { 1, v.Length });
        return AiDotNetEngine.Current.TensorArgMax(row, 1).GetFlat(0);
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
        if (logits.Length == 0)
        {
            throw new ArgumentException("Logits must not be empty.", nameof(logits));
        }

        var engine = AiDotNetEngine.Current;
        var values = Tensor<T>.FromVector(logits);
        var expValues = engine.TensorExp(engine.TensorSubtractScalar(values, engine.TensorMaxValue(values)));
        return engine.TensorDivideScalar(expValues, engine.TensorSum(expValues)).ToVector();
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
[AiDotNet.Configuration.YamlConfigurable("GradientModel")]
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
