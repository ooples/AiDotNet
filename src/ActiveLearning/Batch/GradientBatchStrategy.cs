using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ActiveLearning.Batch;

/// <summary>
/// Gradient-based batch selection strategy using gradient embeddings (BADGE-style).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BADGE (Batch Active learning by Diverse Gradient Embeddings)
/// is a state-of-the-art method that combines uncertainty and diversity. It uses gradient
/// embeddings - vectors derived from the model's gradients - to represent samples.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Compute gradient embeddings for all unlabeled samples</description></item>
/// <item><description>Use k-means++ initialization to select diverse embeddings</description></item>
/// <item><description>The gradient magnitude captures uncertainty</description></item>
/// <item><description>k-means++ ensures diversity in gradient space</description></item>
/// </list>
///
/// <para><b>Why Gradient Embeddings?</b></para>
/// <list type="bullet">
/// <item><description>Gradients are large for uncertain predictions (model wants to change)</description></item>
/// <item><description>Similar gradients indicate redundant information</description></item>
/// <item><description>Combines uncertainty and diversity in a principled way</description></item>
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
public class GradientBatchStrategy<T, TInput, TOutput> : IGradientBatchStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly bool _useHypotheticalGradients;
    private T _diversityTradeoff;
    [Scratch]
    private Matrix<T>? _cachedEmbeddings;

    /// <inheritdoc/>
    public string Name => "Gradient Batch Selection (BADGE-style)";

    /// <inheritdoc/>
    public T DiversityTradeoff
    {
        get => _diversityTradeoff;
        set => _diversityTradeoff = value;
    }

    /// <summary>
    /// Initializes a new GradientBatchStrategy with default settings.
    /// </summary>
    public GradientBatchStrategy()
        : this(useHypotheticalGradients: true)
    {
    }

    /// <summary>
    /// Initializes a new GradientBatchStrategy with specified parameters.
    /// </summary>
    /// <param name="useHypotheticalGradients">Whether to use hypothetical labels for gradient computation.</param>
    public GradientBatchStrategy(bool useHypotheticalGradients = true)
    {
        _useHypotheticalGradients = useHypotheticalGradients;
        _diversityTradeoff = NumOps.FromDouble(0.5);
    }

    /// <inheritdoc/>
    public int[] SelectBatch(
        int[] candidateIndices,
        Vector<T> scores,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize)
    {
        if (candidateIndices.Length == 0)
        {
            return Array.Empty<int>();
        }

        int effectiveBatchSize = Math.Min(batchSize, candidateIndices.Length);

        // Create subset for gradient computation
        var candidatePool = unlabeledPool.Subset(candidateIndices);

        // Note: We need the model to compute gradient embeddings
        // If we don't have cached embeddings, use feature-based fallback
        if (_cachedEmbeddings == null)
        {
            // Use feature-based selection as fallback
            return FeatureBasedKMeansPlusPlus(candidateIndices, candidatePool, effectiveBatchSize);
        }

        // Use k-means++ on gradient embeddings
        var selectedLocalIndices = KMeansPlusPlusSelection(_cachedEmbeddings, effectiveBatchSize);

        // Map back to pool indices
        return selectedLocalIndices.Select(i => candidateIndices[i]).ToArray();
    }

    /// <inheritdoc/>
    public Matrix<T> ComputeGradientEmbeddings(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> samples)
    {
        int n = samples.Count;

        // Get embedding dimension from model
        int embeddingDim = EstimateEmbeddingDimension(model, samples);

        var embeddings = new Matrix<T>(n, embeddingDim);

        for (int i = 0; i < n; i++)
        {
            var input = samples.GetInput(i);
            var embedding = ComputeGradientEmbedding(model, input);

            // Copy the first embeddingDim values; shorter embeddings are zero-padded
            int copied = Math.Min(embeddingDim, embedding.Length);
            var row = copied == embeddingDim
                ? embedding.GetSubVector(0, embeddingDim)
                : Vector<T>.Concatenate(new[] { embedding.GetSubVector(0, copied), new Vector<T>(embeddingDim - copied) });
            embeddings.SetRow(i, row);
        }

        _cachedEmbeddings = embeddings;
        return _cachedEmbeddings;
    }

    /// <inheritdoc/>
    public int[] KMeansPlusPlusSelection(Matrix<T> embeddings, int batchSize)
    {
        int n = embeddings.Rows;
        if (n == 0 || batchSize == 0)
        {
            return Array.Empty<int>();
        }

        int effectiveBatchSize = Math.Min(batchSize, n);
        var selected = new List<int>();
        var random = RandomHelper.Shared;

        // First center: random sample
        int firstIdx = random.Next(n);
        selected.Add(firstIdx);

        var engine = AiDotNetEngine.Current;
        var rows = Tensor<T>.FromMatrix(embeddings);
        int dim = embeddings.Columns;
        var isSelected = new bool[n];
        isSelected[firstIdx] = true;

        // Minimum squared distance from every sample to the selected centers
        var minDist = new Tensor<T>(new[] { n });
        minDist.Fill(NumOps.MaxValue);

        // Subsequent centers: k-means++ initialization
        for (int k = 1; k < effectiveBatchSize; k++)
        {
            var newest = engine.TensorSlice(rows, new[] { selected[selected.Count - 1], 0 }, new[] { 1, dim });
            minDist = engine.TensorMin(minDist, ActiveLearningEngineOps.SquaredDistances(rows, newest).Reshape(new[] { n }));

            // Selected samples contribute nothing to the sampling mass
            var distances = engine.TensorMaskedFill(minDist, isSelected, NumOps.Zero);
            var cumulative = engine.TensorCumSum(distances, 0);
            var totalDistance = engine.TensorSlice(cumulative, new[] { n - 1 }, new[] { 1 });

            // Sample proportional to distance squared
            if (engine.TensorCountNonzero(engine.TensorGreaterThan(totalDistance, NumOps.Zero)) == 0)
            {
                // All remaining points are equidistant, pick randomly
                var remaining = Enumerable.Range(0, n).Where(i => !selected.Contains(i)).ToList();
                if (remaining.Count > 0)
                {
                    var pick = remaining[random.Next(remaining.Count)];
                    selected.Add(pick);
                    isSelected[pick] = true;
                }
            }
            else
            {
                // First unselected sample whose cumulative distance reaches the threshold. Selected
                // samples add zero mass, so the next unselected sample after a selected hit is the answer.
                var threshold = engine.TensorMultiplyScalar(totalDistance, NumOps.FromDouble(random.NextDouble()));
                int selectedIdx = engine.TensorSearchSorted(cumulative, threshold).GetFlat(0);
                while (selectedIdx < n && isSelected[selectedIdx])
                {
                    selectedIdx++;
                }

                if (selectedIdx >= n)
                {
                    selectedIdx = -1;
                }

                if (selectedIdx >= 0)
                {
                    selected.Add(selectedIdx);
                    isSelected[selectedIdx] = true;
                }
                else
                {
                    // Fallback: pick last unselected
                    var remaining = Enumerable.Range(0, n).Where(i => !selected.Contains(i)).ToList();
                    if (remaining.Count > 0)
                    {
                        selected.Add(remaining[^1]);
                        isSelected[remaining[^1]] = true;
                    }
                }
            }
        }

        return selected.ToArray();
    }

    /// <inheritdoc/>
    public T ComputeDiversity(TInput sample1, TInput sample2)
    {
        // Compute diversity based on feature distance
        var vec1 = ConvertToVector(sample1);
        var vec2 = ConvertToVector(sample2);
        return ComputeDistance(vec1, vec2);
    }

    #region Private Methods

    private int[] FeatureBasedKMeansPlusPlus(
        int[] candidateIndices,
        IDataset<T, TInput, TOutput> candidatePool,
        int batchSize)
    {
        // Extract features as matrix
        var features = new List<T[]>();
        int dim = 0;

        for (int i = 0; i < candidatePool.Count; i++)
        {
            var input = candidatePool.GetInput(i);
            var vec = ConvertToVector(input);
            features.Add(vec.ToArray());
            dim = Math.Max(dim, vec.Length);
        }

        if (features.Count == 0)
        {
            return Array.Empty<int>();
        }

        // Pad to uniform dimension
        var matrix = new T[features.Count, dim];
        for (int i = 0; i < features.Count; i++)
        {
            for (int d = 0; d < features[i].Length; d++)
            {
                matrix[i, d] = features[i][d];
            }
        }

        var featureMatrix = new Matrix<T>(matrix);
        var localIndices = KMeansPlusPlusSelection(featureMatrix, batchSize);

        // Map back to pool indices
        return localIndices.Select(i => candidateIndices[i]).ToArray();
    }

    private Vector<T> ComputeGradientEmbedding(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        // For BADGE, gradient embedding = (p̂ - e_ŷ) ⊗ g_θ(x)
        // where p̂ is predicted probability, e_ŷ is one-hot of predicted class,
        // and g_θ(x) is the penultimate layer activations

        // If model supports gradient computation, use it
        if (model is IGradientComputable<T, Tensor<T>, Tensor<T>> gradModel)
        {
            try
            {
                // Get prediction
                var prediction = model.Predict(input);

                // Compute gradients with hypothetical label
                if (_useHypotheticalGradients && prediction is Vector<T> probVec)
                {
                    // Use predicted class as hypothetical label
                    int predictedClass = ArgMax(probVec);
                    var hypotheticalOutput = CreateOneHot(predictedClass, probVec.Length);

                    // Compute gradient w.r.t. loss
                    var inputTensor = ConvertToTensor(input);
                    var outputTensor = ConvertToTensor(hypotheticalOutput);

                    var gradients = gradModel.ComputeGradients(inputTensor, outputTensor);

                    // ComputeGradients returns Vector<T>, which is already the embedding
                    return gradients;
                }

                // Fallback: use prediction as embedding
                return ConvertToVector(prediction);
            }
            catch
            {
                // Fallback to feature-based embedding
            }
        }

        // Use model prediction as embedding
        var pred = model.Predict(input);
        return ConvertToVector(pred);
    }

    private int EstimateEmbeddingDimension(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> samples)
    {
        if (samples.Count == 0)
        {
            return 1;
        }

        // Estimate from first sample
        var firstInput = samples.GetInput(0);
        var embedding = ComputeGradientEmbedding(model, firstInput);
        return embedding.Length;
    }

    private Vector<T> ConvertToVector(object? obj)
    {
        if (obj == null)
        {
            return new Vector<T>(new[] { NumOps.Zero });
        }

        if (obj is Vector<T> vec)
        {
            return vec;
        }

        if (obj is T[] arr)
        {
            return new Vector<T>(arr);
        }

        if (obj is IReadOnlyList<T> list)
        {
            return new Vector<T>(list.ToArray());
        }

        if (obj is T val)
        {
            return new Vector<T>(new[] { val });
        }

        if (obj is double d)
        {
            return new Vector<T>(new[] { NumOps.FromDouble(d) });
        }

        return new Vector<T>(new[] { NumOps.Zero });
    }

    private Tensor<T> ConvertToTensor(object? obj)
    {
        var vec = ConvertToVector(obj);
        return new Tensor<T>(new[] { vec.Length }, vec);
    }

    private Vector<T> FlattenTensor(Tensor<T> tensor)
    {
        return new Vector<T>(tensor.Data.ToArray());
    }

    private int ArgMax(Vector<T> vec)
    {
        if (vec.Length == 0)
        {
            return 0;
        }

        // First maximum wins, as in a left-to-right scan
        return AiDotNetEngine.Current.TensorArgMax(Tensor<T>.FromVector(vec).Reshape(new[] { 1, vec.Length }), 1).GetFlat(0);
    }

    private Vector<T> CreateOneHot(int index, int length)
    {
        var arr = new T[length];
        for (int i = 0; i < length; i++)
        {
            arr[i] = i == index ? NumOps.One : NumOps.Zero;
        }
        return new Vector<T>(arr);
    }

    private T ComputeDistance(Vector<T> a, Vector<T> b)
    {
        return VectorHelper.EuclideanDistance(a, b);
    }

    #endregion
}
