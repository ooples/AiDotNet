using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Helper class for in-context learning in tabular foundation models like TabPFN.
/// </summary>
/// <remarks>
/// <para>
/// In-context learning allows models to learn from examples provided at inference time
/// without updating model parameters. The model conditions on training examples
/// to make predictions on new data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of in-context learning like this:
/// - Traditional ML: Train a model, then use it to predict
/// - In-context learning: Give the model examples AND the test data together
///
/// The model "learns" from the examples in real-time through attention mechanisms.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Training)]
public class InContextLearning<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private Tensor<T>? _contextFeatures;
    private Tensor<T>? _contextLabels;
    private int _numContextSamples;
    private int _numFeatures;
    private int _labelDimension;

    /// <summary>
    /// Gets the number of samples in the current context.
    /// </summary>
    public int NumContextSamples => _numContextSamples;

    /// <summary>
    /// Gets whether context has been set.
    /// </summary>
    public bool HasContext => _contextFeatures != null;

    /// <summary>
    /// Sets the context (training) data for in-context learning.
    /// </summary>
    /// <param name="features">Context features with shape [numSamples, numFeatures].</param>
    /// <param name="labels">Context labels with shape [numSamples, labelDim].</param>
    /// <param name="maxContextSamples">Maximum allowed context samples.</param>
    public void SetContext(Tensor<T> features, Tensor<T> labels, int maxContextSamples = 1024)
    {
        if (features.Shape[0] != labels.Shape[0])
        {
            throw new ArgumentException("Features and labels must have the same number of samples");
        }

        int numSamples = features.Shape[0];
        if (numSamples > maxContextSamples)
        {
            // Subsample if too many context samples
            (features, labels) = SubsampleContext(features, labels, maxContextSamples);
            numSamples = maxContextSamples;
        }

        _contextFeatures = features;
        _contextLabels = labels;
        _numContextSamples = numSamples;
        _numFeatures = features.Shape.Length > 1 ? features.Shape[1] : 1;
        _labelDimension = labels.Shape.Length > 1 ? labels.Shape[1] : 1;
    }

    /// <summary>
    /// Sets the context for classification tasks with integer labels.
    /// </summary>
    /// <param name="features">Context features.</param>
    /// <param name="labels">Context labels as class indices.</param>
    /// <param name="numClasses">Number of classes.</param>
    /// <param name="maxContextSamples">Maximum allowed context samples.</param>
    public void SetClassificationContext(Tensor<T> features, Vector<int> labels, int numClasses, int maxContextSamples = 1024)
    {
        int numSamples = features.Shape[0];

        // Labels outside [0, numClasses) leave an all-zero row
        var oneHotLabels = AiDotNetEngine.Current.TensorOneHot<T>(
            Tensor<int>.FromVector(labels.GetSubVector(0, numSamples)), numClasses).Reshape(numSamples, numClasses);

        SetContext(features, oneHotLabels, maxContextSamples);
    }

    /// <summary>
    /// Clears the current context.
    /// </summary>
    public void ClearContext()
    {
        _contextFeatures = null;
        _contextLabels = null;
        _numContextSamples = 0;
    }

    /// <summary>
    /// Creates the combined input for in-context learning by concatenating
    /// context samples with query samples.
    /// </summary>
    /// <param name="queryFeatures">Query features to predict.</param>
    /// <returns>Combined tensor with context followed by query.</returns>
    public Tensor<T> CreateInContextInput(Tensor<T> queryFeatures)
    {
        if (_contextFeatures == null)
        {
            return queryFeatures;
        }

        int querySize = queryFeatures.Shape[0];
        int numFeatures = queryFeatures.Shape.Length > 1 ? queryFeatures.Shape[1] : _numFeatures;
        int totalSize = _numContextSamples + querySize;

        // Context samples first, then the queries
        var combined = AiDotNetEngine.Current.TensorConcatenate(
            new[]
            {
                TabularMetrics.Flat(_contextFeatures, _numContextSamples * numFeatures).Reshape(_numContextSamples, numFeatures),
                TabularMetrics.Flat(queryFeatures, querySize * numFeatures).Reshape(querySize, numFeatures),
            },
            0);

        return combined;
    }

    /// <summary>
    /// Gets the context labels for use in attention mechanisms.
    /// </summary>
    public Tensor<T>? GetContextLabels() => _contextLabels;

    /// <summary>
    /// Creates a mask indicating which positions are context vs query.
    /// </summary>
    /// <param name="querySize">Number of query samples.</param>
    /// <returns>Boolean mask where true = context, false = query.</returns>
    public bool[] CreateContextMask(int querySize)
    {
        int totalSize = _numContextSamples + querySize;
        var mask = new bool[totalSize];

        for (int i = 0; i < _numContextSamples; i++)
        {
            mask[i] = true;
        }

        return mask;
    }

    /// <summary>
    /// Extracts query predictions from combined output.
    /// </summary>
    /// <param name="combinedOutput">Output from model on combined input.</param>
    /// <param name="outputDim">Output dimension per sample.</param>
    /// <returns>Predictions for query samples only.</returns>
    public Tensor<T> ExtractQueryOutput(Tensor<T> combinedOutput, int outputDim)
    {
        int totalSize = combinedOutput.Shape[0];
        int querySize = totalSize - _numContextSamples;

        int offset = _numContextSamples * outputDim;
        var queryOutput = AiDotNetEngine.Current.TensorNarrow(
            TabularMetrics.Flat(combinedOutput, combinedOutput.Length), 0, offset, querySize * outputDim)
            .Reshape(querySize, outputDim);

        return queryOutput;
    }

    /// <summary>
    /// Subsamples context to fit within maximum size.
    /// </summary>
    private (Tensor<T> features, Tensor<T> labels) SubsampleContext(
        Tensor<T> features, Tensor<T> labels, int maxSamples)
    {
        int numSamples = features.Shape[0];
        int numFeatures = features.Shape.Length > 1 ? features.Shape[1] : 1;
        int labelDim = labels.Shape.Length > 1 ? labels.Shape[1] : 1;

        // Create random indices for subsampling
        var random = RandomHelper.CreateSecureRandom();
        var indices = Enumerable.Range(0, numSamples).OrderBy(_ => random.Next()).Take(maxSamples).ToArray();
        Array.Sort(indices);

        var engine = AiDotNetEngine.Current;
        var chosen = new Tensor<int>(indices, new[] { maxSamples });
        var subFeatures = engine.TensorIndexSelect(
            TabularMetrics.Flat(features, numSamples * numFeatures).Reshape(numSamples, numFeatures), chosen, 0);
        var subLabels = engine.TensorIndexSelect(
            TabularMetrics.Flat(labels, numSamples * labelDim).Reshape(numSamples, labelDim), chosen, 0);

        return (subFeatures, subLabels);
    }
}
