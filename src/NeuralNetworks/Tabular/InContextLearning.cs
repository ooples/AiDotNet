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
        var oneHotLabels = new Tensor<T>([numSamples, numClasses]);

        for (int i = 0; i < numSamples; i++)
        {
            int classIdx = labels[i];
            if (classIdx >= 0 && classIdx < numClasses)
            {
                oneHotLabels[i * numClasses + classIdx] = NumOps.One;
            }
        }

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

        var combined = new Tensor<T>([totalSize, numFeatures]);

        // Copy context features
        for (int i = 0; i < _numContextSamples * numFeatures; i++)
        {
            combined[i] = _contextFeatures[i];
        }

        // Copy query features
        int offset = _numContextSamples * numFeatures;
        for (int i = 0; i < querySize * numFeatures; i++)
        {
            combined[offset + i] = queryFeatures[i];
        }

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

        var queryOutput = new Tensor<T>([querySize, outputDim]);
        int offset = _numContextSamples * outputDim;

        for (int i = 0; i < querySize * outputDim; i++)
        {
            queryOutput[i] = combinedOutput[offset + i];
        }

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

        var subFeatures = new Tensor<T>([maxSamples, numFeatures]);
        var subLabels = new Tensor<T>([maxSamples, labelDim]);

        for (int i = 0; i < maxSamples; i++)
        {
            int srcIdx = indices[i];
            for (int f = 0; f < numFeatures; f++)
            {
                subFeatures[i * numFeatures + f] = features[srcIdx * numFeatures + f];
            }
            for (int l = 0; l < labelDim; l++)
            {
                subLabels[i * labelDim + l] = labels[srcIdx * labelDim + l];
            }
        }

        return (subFeatures, subLabels);
    }
}
