using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// ANIL model for few-shot classification with head-only adaptation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of ANIL after inner-loop adaptation.
/// It contains the frozen feature extractor (body) and the adapted classification head.
/// </para>
/// <para><b>For Beginners:</b> After ANIL adapts to a new task by training only
/// the classification head on support examples, this model stores:
/// </para>
/// <list type="bullet">
/// <item>The frozen body (feature extractor) from meta-training</item>
/// <item>The adapted head weights specific to this task</item>
/// <item>The adapted head bias (if used)</item>
/// </list>
/// <para>
/// When making predictions, the model extracts features using the frozen body
/// and classifies using the adapted head.
/// </para>
/// </remarks>
public class ANILModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _featureExtractor;
    private readonly Vector<T> _headWeights;
    private readonly Vector<T>? _headBias;
    private readonly ANILOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the ANILModel.
    /// </summary>
    /// <param name="featureExtractor">The feature extractor (frozen body).</param>
    /// <param name="headWeights">The adapted head weight parameters.</param>
    /// <param name="headBias">The adapted head bias parameters (optional).</param>
    /// <param name="options">The ANIL options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public ANILModel(
        IFullModel<T, TInput, TOutput> featureExtractor,
        Vector<T> headWeights,
        Vector<T>? headBias,
        ANILOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(featureExtractor);
        _featureExtractor = featureExtractor;
        Guard.NotNull(headWeights);
        _headWeights = headWeights;
        _headBias = headBias;
        Guard.NotNull(options);
        _options = options;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the adapted head weights.
    /// </summary>
    public Vector<T> HeadWeights => _headWeights;

    /// <summary>
    /// Gets the adapted head bias (may be null if not used).
    /// </summary>
    public Vector<T>? HeadBias => _headBias;

    /// <summary>
    /// Gets the number of classes this model is adapted for.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <summary>
    /// Gets the feature dimension expected by the head.
    /// </summary>
    public int FeatureDimension => _options.FeatureDimension;

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Extract features using frozen body
        var features = ExtractFeatures(input);

        // Apply adapted head
        var logits = ComputeLogits(features);

        // Convert to output type
        return ConvertToOutput(logits);
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the ANIL algorithm to train the model.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("ANIL model parameters are set during adaptation.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Return combined body + head parameters
        var bodyParams = _featureExtractor.GetParameters();
        int totalSize = bodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < bodyParams.Length; i++)
        {
            combined[idx++] = bodyParams[i];
        }
        for (int i = 0; i < _headWeights.Length; i++)
        {
            combined[idx++] = _headWeights[i];
        }
        if (_headBias != null)
        {
            for (int i = 0; i < _headBias.Length; i++)
            {
                combined[idx++] = _headBias[i];
            }
        }

        return combined;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }

    /// <summary>
    /// Extracts features from input using the frozen body.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the feature extractor returns an unsupported output type.
    /// </exception>
    private Vector<T> ExtractFeatures(TInput input)
    {
        var output = _featureExtractor.Predict(input);

        if (output is Vector<T> vec)
        {
            return vec;
        }

        if (output is Tensor<T> tensor)
        {
            return tensor.ToVector();
        }

        // Handle Matrix<T> by flattening the first row (for single examples)
        // or the entire matrix (for batch processing)
        if (output is Matrix<T> matrix)
        {
            if (matrix.Rows == 1)
            {
                // Single example: return the row as a vector
                var result = new Vector<T>(matrix.Columns);
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[j] = matrix[0, j];
                }
                return result;
            }
            else
            {
                // Multiple examples: flatten the entire matrix
                var result = new Vector<T>(matrix.Rows * matrix.Columns);
                int idx = 0;
                for (int i = 0; i < matrix.Rows; i++)
                {
                    for (int j = 0; j < matrix.Columns; j++)
                    {
                        result[idx++] = matrix[i, j];
                    }
                }
                return result;
            }
        }

        // Handle T[] array
        if (output is T[] array)
        {
            return new Vector<T>(array);
        }

        // Throw an informative exception instead of silently returning zeros
        throw new InvalidOperationException(
            $"Feature extractor returned unsupported output type '{output?.GetType().Name ?? "null"}'. " +
            $"Expected Vector<{typeof(T).Name}>, Tensor<{typeof(T).Name}>, Matrix<{typeof(T).Name}>, or {typeof(T).Name}[]. " +
            "Ensure the feature extractor model produces compatible output.");
    }

    /// <summary>
    /// Computes logits from features using adapted head parameters.
    /// </summary>
    private Vector<T> ComputeLogits(Vector<T> features)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var logits = new Vector<T>(_options.NumClasses);

        int featureDim = Math.Min(features.Length, _options.FeatureDimension);

        for (int c = 0; c < _options.NumClasses; c++)
        {
            T sum = numOps.Zero;

            for (int f = 0; f < featureDim; f++)
            {
                int weightIdx = c * _options.FeatureDimension + f;
                if (weightIdx < _headWeights.Length)
                {
                    sum = numOps.Add(sum, numOps.Multiply(features[f], _headWeights[weightIdx]));
                }
            }

            if (_headBias != null && c < _headBias.Length)
            {
                sum = numOps.Add(sum, _headBias[c]);
            }

            logits[c] = sum;
        }

        return logits;
    }

    /// <summary>
    /// Converts logits to the expected output type.
    /// </summary>
    private TOutput ConvertToOutput(Vector<T> logits)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)logits;
        }

        // Handle Tensor<T>
        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)Tensor<T>.FromVector(logits);
        }

        // Handle T[]
        if (typeof(TOutput) == typeof(T[]))
        {
            return (TOutput)(object)logits.ToArray();
        }

        throw new InvalidOperationException(
            $"Cannot convert Vector<{typeof(T).Name}> to {typeof(TOutput).Name}. " +
            $"Supported types: Vector<T>, Tensor<T>, T[]");
    }
}
