using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// LEO model for few-shot classification with latent space optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of LEO after latent space optimization.
/// It contains the feature encoder, adapted classifier parameters, and the
/// optimized latent code.
/// </para>
/// <para><b>For Beginners:</b> After LEO adapts to a new task by optimizing
/// in latent space, this model stores:
/// </para>
/// <list type="bullet">
/// <item>The feature encoder for extracting embeddings</item>
/// <item>The classifier parameters decoded from the adapted latent code</item>
/// <item>The adapted latent code itself (useful for further fine-tuning)</item>
/// </list>
/// </remarks>
public class LEOModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly Vector<T> _classifierParams;
    private readonly Vector<T> _latentCode;
    private readonly LEOOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the LEOModel.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network.</param>
    /// <param name="classifierParams">The adapted classifier parameters.</param>
    /// <param name="latentCode">The optimized latent code.</param>
    /// <param name="options">The LEO options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public LEOModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Vector<T> classifierParams,
        Vector<T> latentCode,
        LEOOptions<T, TInput, TOutput> options)
    {
        _featureEncoder = featureEncoder ?? throw new ArgumentNullException(nameof(featureEncoder));
        _classifierParams = classifierParams ?? throw new ArgumentNullException(nameof(classifierParams));
        _latentCode = latentCode ?? throw new ArgumentNullException(nameof(latentCode));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the adapted classifier parameters.
    /// </summary>
    public Vector<T> ClassifierParams => _classifierParams;

    /// <summary>
    /// Gets the optimized latent code.
    /// </summary>
    public Vector<T> LatentCode => _latentCode;

    /// <summary>
    /// Gets the latent dimension.
    /// </summary>
    public int LatentDimension => _options.LatentDimension;

    /// <summary>
    /// Gets the number of classes.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Extract features using the encoder
        var embeddings = ExtractEmbeddings(input);

        // Apply classifier
        var logits = ComputeLogits(embeddings);

        return ConvertToOutput(logits);
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the LEO algorithm to train the model.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("LEO model parameters are set during adaptation.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Return combined feature encoder + classifier parameters + latent code
        var encoderParams = _featureEncoder.GetParameters();
        int totalSize = encoderParams.Length + _classifierParams.Length + _latentCode.Length;
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < encoderParams.Length; i++)
        {
            combined[idx++] = encoderParams[i];
        }
        for (int i = 0; i < _classifierParams.Length; i++)
        {
            combined[idx++] = _classifierParams[i];
        }
        for (int i = 0; i < _latentCode.Length; i++)
        {
            combined[idx++] = _latentCode[i];
        }

        return combined;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }

    /// <summary>
    /// Extracts embeddings from input using the feature encoder.
    /// </summary>
    private Vector<T> ExtractEmbeddings(TInput input)
    {
        var output = _featureEncoder.Predict(input);

        if (output is Vector<T> vec)
        {
            return vec;
        }

        if (output is Tensor<T> tensor)
        {
            return tensor.ToVector();
        }

        return new Vector<T>(_options.EmbeddingDimension);
    }

    /// <summary>
    /// Computes logits from embeddings using classifier parameters.
    /// </summary>
    private Vector<T> ComputeLogits(Vector<T> embeddings)
    {
        var logits = new Vector<T>(_options.NumClasses);
        int embDim = Math.Min(embeddings.Length, _options.EmbeddingDimension);

        for (int c = 0; c < _options.NumClasses; c++)
        {
            T sum = NumOps.Zero;
            for (int e = 0; e < embDim; e++)
            {
                int paramIdx = c * _options.EmbeddingDimension + e;
                if (paramIdx < _classifierParams.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(embeddings[e], _classifierParams[paramIdx]));
                }
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
