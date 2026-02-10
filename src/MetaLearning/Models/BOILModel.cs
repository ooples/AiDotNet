using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// BOIL model for few-shot classification with body-only adaptation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of BOIL after inner-loop adaptation.
/// It contains the adapted feature extractor (body) and the frozen classification head.
/// </para>
/// <para><b>For Beginners:</b> After BOIL adapts to a new task by training only
/// the feature extractor (body) on support examples, this model stores:
/// </para>
/// <list type="bullet">
/// <item>The adapted body (feature extractor) specific to this task</item>
/// <item>The frozen head weights from meta-training</item>
/// <item>The frozen head bias (if used)</item>
/// </list>
/// <para>
/// When making predictions, the model extracts features using the adapted body
/// and classifies using the frozen head. This is the opposite of ANIL which
/// freezes the body and adapts the head.
/// </para>
/// </remarks>
public class BOILModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _baseModel;
    private readonly Vector<T> _adaptedBodyParams;
    private readonly Vector<T> _headWeights;
    private readonly Vector<T>? _headBias;
    private readonly BOILOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the BOILModel.
    /// </summary>
    /// <param name="baseModel">The base model to use for feature extraction.</param>
    /// <param name="adaptedBodyParams">The adapted body parameters for this task.</param>
    /// <param name="headWeights">The frozen head weight parameters.</param>
    /// <param name="headBias">The frozen head bias parameters (optional).</param>
    /// <param name="options">The BOIL options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public BOILModel(
        IFullModel<T, TInput, TOutput> baseModel,
        Vector<T> adaptedBodyParams,
        Vector<T> headWeights,
        Vector<T>? headBias,
        BOILOptions<T, TInput, TOutput> options)
    {
        _baseModel = baseModel ?? throw new ArgumentNullException(nameof(baseModel));
        _adaptedBodyParams = adaptedBodyParams ?? throw new ArgumentNullException(nameof(adaptedBodyParams));
        _headWeights = headWeights ?? throw new ArgumentNullException(nameof(headWeights));
        _headBias = headBias;
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the adapted body parameters.
    /// </summary>
    public Vector<T> AdaptedBodyParams => _adaptedBodyParams;

    /// <summary>
    /// Gets the frozen head weights.
    /// </summary>
    public Vector<T> HeadWeights => _headWeights;

    /// <summary>
    /// Gets the frozen head bias (may be null if not used).
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
        // Apply adapted body parameters to model
        ApplyAdaptedBodyParameters();

        // Extract features using adapted body
        var features = ExtractFeatures(input);

        // Apply frozen head
        var logits = ComputeLogits(features);

        // Convert to output type
        return ConvertToOutput(logits);
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the BOIL algorithm to train the model.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("BOIL model parameters are set during adaptation.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Return combined body + head parameters
        int totalSize = _adaptedBodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < _adaptedBodyParams.Length; i++)
        {
            combined[idx++] = _adaptedBodyParams[i];
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
    /// Applies the adapted body parameters to the base model.
    /// </summary>
    private void ApplyAdaptedBodyParameters()
    {
        var currentParams = _baseModel.GetParameters();
        var updatedParams = new Vector<T>(currentParams.Length);

        // Apply adapted body parameters
        int copyLen = Math.Min(_adaptedBodyParams.Length, currentParams.Length);
        for (int i = 0; i < copyLen; i++)
        {
            updatedParams[i] = _adaptedBodyParams[i];
        }

        // Keep head parameters from the original model
        for (int i = copyLen; i < currentParams.Length; i++)
        {
            updatedParams[i] = currentParams[i];
        }

        _baseModel.SetParameters(updatedParams);
    }

    /// <summary>
    /// Extracts features from input using the adapted body.
    /// </summary>
    private Vector<T> ExtractFeatures(TInput input)
    {
        var output = _baseModel.Predict(input);

        if (output is Vector<T> vec)
        {
            return vec;
        }

        if (output is Tensor<T> tensor)
        {
            return tensor.ToVector();
        }

        // Return a default feature vector
        return new Vector<T>(_options.FeatureDimension);
    }

    /// <summary>
    /// Computes logits from features using frozen head parameters.
    /// </summary>
    private Vector<T> ComputeLogits(Vector<T> features)
    {
        var logits = new Vector<T>(_options.NumClasses);
        int featureDim = Math.Min(features.Length, _options.FeatureDimension);

        for (int c = 0; c < _options.NumClasses; c++)
        {
            T sum = NumOps.Zero;

            for (int f = 0; f < featureDim; f++)
            {
                int weightIdx = c * _options.FeatureDimension + f;
                if (weightIdx < _headWeights.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(features[f], _headWeights[weightIdx]));
                }
            }

            if (_headBias != null && c < _headBias.Length)
            {
                sum = NumOps.Add(sum, _headBias[c]);
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

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)Tensor<T>.FromVector(logits);
        }

        if (typeof(TOutput) == typeof(T[]))
        {
            return (TOutput)(object)logits.ToArray();
        }

        throw new InvalidOperationException(
            $"Cannot convert Vector<{typeof(T).Name}> to {typeof(TOutput).Name}. " +
            $"Supported types: Vector<T>, Tensor<T>, T[]");
    }
}
