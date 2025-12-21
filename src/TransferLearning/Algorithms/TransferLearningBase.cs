
using AiDotNet.Interfaces;
using AiDotNet.TransferLearning.DomainAdaptation;
using AiDotNet.TransferLearning.FeatureMapping;

namespace AiDotNet.TransferLearning.Algorithms;

/// <summary>
/// Base class for transfer learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Transfer learning is like using knowledge from one task to help with another.
/// Imagine learning to ride a bicycle helps you learn to ride a motorcycle faster. This base class
/// provides common functionality for all transfer learning methods in AiDotNet.
/// </para>
/// </remarks>
public abstract class TransferLearningBase<T, TInput, TOutput>
{
    protected readonly INumericOperations<T> NumOps;
    protected IFeatureMapper<T>? FeatureMapper;
    protected IDomainAdapter<T>? DomainAdapter;

    /// <summary>
    /// Initializes a new instance of the TransferLearningBase class.
    /// </summary>
    protected TransferLearningBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Sets the feature mapper to use for cross-domain transfer.
    /// </summary>
    /// <param name="mapper">The feature mapper instance.</param>
    public void SetFeatureMapper(IFeatureMapper<T> mapper)
    {
        FeatureMapper = mapper;
    }

    /// <summary>
    /// Sets the domain adapter to use for reducing distribution shift.
    /// </summary>
    /// <param name="adapter">The domain adapter instance.</param>
    public void SetDomainAdapter(IDomainAdapter<T> adapter)
    {
        DomainAdapter = adapter;
    }

    /// <summary>
    /// Transfers knowledge from a source model to a target domain (same feature space).
    /// </summary>
    /// <param name="sourceModel">The model trained on the source domain.</param>
    /// <param name="targetData">Training data from the target domain.</param>
    /// <param name="targetLabels">Labels for the target domain data.</param>
    /// <returns>A new model adapted to the target domain.</returns>
    protected abstract IFullModel<T, TInput, TOutput> TransferSameDomain(
        IFullModel<T, TInput, TOutput> sourceModel,
        TInput targetData,
        TOutput targetLabels);

    /// <summary>
    /// Transfers knowledge from a source model to a target domain (different feature space).
    /// </summary>
    /// <param name="sourceModel">The model trained on the source domain.</param>
    /// <param name="targetData">Training data from the target domain.</param>
    /// <param name="targetLabels">Labels for the target domain data.</param>
    /// <returns>A new model adapted to the target domain.</returns>
    protected abstract IFullModel<T, TInput, TOutput> TransferCrossDomain(
        IFullModel<T, TInput, TOutput> sourceModel,
        TInput targetData,
        TOutput targetLabels);

    /// <summary>
    /// Evaluates if cross-domain transfer is necessary based on feature dimensions.
    /// </summary>
    /// <param name="sourceModel">The source model.</param>
    /// <param name="targetData">The target data.</param>
    /// <returns>True if cross-domain transfer is needed; otherwise, false.</returns>
    protected bool RequiresCrossDomainTransfer(IFullModel<T, TInput, TOutput> sourceModel, TInput targetData)
    {
        // Get active features from source model
        var sourceFeatures = sourceModel.GetActiveFeatureIndices().Count();

        // Get target features based on input type
        int targetFeatures = InputHelper<T, TInput>.GetInputSize(targetData);

        return sourceFeatures != targetFeatures;
    }

    /// <summary>
    /// Computes a confidence score for transfer learning success.
    /// </summary>
    /// <param name="sourceData">Source domain data.</param>
    /// <param name="targetData">Target domain data.</param>
    /// <returns>A confidence score between 0 and 1.</returns>
    protected T ComputeTransferConfidence(TInput sourceData, TInput targetData)
    {
        T confidence = NumOps.One;

        // Factor in feature mapper confidence if available
        if (FeatureMapper != null && FeatureMapper.IsTrained)
        {
            T mapperConfidence = FeatureMapper.GetMappingConfidence();
            confidence = NumOps.Multiply(confidence, mapperConfidence);
        }

        // Factor in domain discrepancy if adapter is available
        if (DomainAdapter != null && sourceData is Matrix<T> sourceMatrix && targetData is Matrix<T> targetMatrix)
        {
            T discrepancy = DomainAdapter.ComputeDomainDiscrepancy(sourceMatrix, targetMatrix);
            // Convert discrepancy to confidence (exp(-discrepancy))
            T domainConfidence = NumOps.Exp(NumOps.Negate(discrepancy));
            confidence = NumOps.Multiply(confidence, domainConfidence);
        }

        return confidence;
    }

    /// <summary>
    /// Selects the most relevant samples from source domain for transfer.
    /// </summary>
    /// <param name="sourceData">Source domain data.</param>
    /// <param name="targetData">Target domain data.</param>
    /// <param name="sampleRatio">Ratio of samples to select (0 to 1).</param>
    /// <returns>Indices of selected source samples.</returns>
    protected int[] SelectRelevantSourceSamples(TInput sourceData, TInput targetData, double sampleRatio = 0.5)
    {
        int batchSize = InputHelper<T, TInput>.GetBatchSize(sourceData);
        int numSamples = (int)(batchSize * sampleRatio);
        numSamples = Math.Max(1, Math.Min(numSamples, batchSize));

        // Compute distances from each source sample to target distribution
        var relevanceScores = new List<(int index, T score)>();

        // Compute target centroid
        var targetCentroid = ComputeCentroid(targetData);

        for (int i = 0; i < batchSize; i++)
        {
            // Get row as vector using InputHelper
            var sourceRow = GetRowAsVector(sourceData, i);

            // Distance to target centroid (negative so closer = higher score)
            T distance = ComputeEuclideanDistance(sourceRow, targetCentroid);
            T score = NumOps.Negate(distance);
            relevanceScores.Add((i, score));
        }

        // Sort by relevance and take top samples
        relevanceScores.Sort((a, b) => Convert.ToDouble(b.score).CompareTo(Convert.ToDouble(a.score)));

        return relevanceScores.Take(numSamples).Select(x => x.index).ToArray();
    }

    /// <summary>
    /// Gets a single row from TInput as a Vector.
    /// </summary>
    private Vector<T> GetRowAsVector(TInput data, int rowIndex)
    {
        if (data is Matrix<T> matrix)
        {
            return matrix.GetRow(rowIndex);
        }
        else if (data is Tensor<T> tensor)
        {
            int cols = tensor.Shape[1];
            var row = new Vector<T>(cols);
            for (int j = 0; j < cols; j++)
            {
                row[j] = tensor[rowIndex, j];
            }
            return row;
        }
        throw new ArgumentException($"Unsupported input type: {typeof(TInput).Name}");
    }

    /// <summary>
    /// Computes the centroid (mean) of a data matrix.
    /// </summary>
    protected Vector<T> ComputeCentroid(TInput data)
    {
        int batchSize = InputHelper<T, TInput>.GetBatchSize(data);
        int inputSize = InputHelper<T, TInput>.GetInputSize(data);

        var centroid = new Vector<T>(inputSize);
        for (int j = 0; j < inputSize; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < batchSize; i++)
            {
                sum = NumOps.Add(sum, InputHelper<T, TInput>.GetElement(data, i, j));
            }
            centroid[j] = NumOps.Divide(sum, NumOps.FromDouble(batchSize));
        }
        return centroid;
    }

    /// <summary>
    /// Computes the Euclidean distance between two vectors.
    /// </summary>
    protected T ComputeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sumSquares = NumOps.Zero;
        int length = Math.Min(a.Length, b.Length);
        for (int i = 0; i < length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sumSquares);
    }
}
