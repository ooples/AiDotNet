using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// MetaOptNet model for few-shot classification with convex optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of MetaOptNet after solving the convex
/// optimization problem on the support set.
/// </para>
/// <para><b>For Beginners:</b> After MetaOptNet sees the support examples and
/// solves for the optimal classifier, this model stores:
/// </para>
/// <list type="bullet">
/// <item>The feature encoder for extracting embeddings</item>
/// <item>The classifier weights from the convex solver</item>
/// <item>The temperature parameter for scaling predictions</item>
/// </list>
/// </remarks>
public class MetaOptNetModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly Matrix<T> _classifierWeights;
    private readonly T _temperature;
    private readonly MetaOptNetOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the MetaOptNetModel.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network.</param>
    /// <param name="classifierWeights">The classifier weights from convex optimization.</param>
    /// <param name="temperature">The temperature parameter for scaling.</param>
    /// <param name="options">The MetaOptNet options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public MetaOptNetModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Matrix<T> classifierWeights,
        T temperature,
        MetaOptNetOptions<T, TInput, TOutput> options)
    {
        _featureEncoder = featureEncoder ?? throw new ArgumentNullException(nameof(featureEncoder));
        _classifierWeights = classifierWeights ?? throw new ArgumentNullException(nameof(classifierWeights));
        _temperature = temperature;
        _options = options ?? throw new ArgumentNullException(nameof(options));

        // Validate embedding dimension
        if (options.EmbeddingDimension <= 0)
        {
            throw new ArgumentException(
                $"EmbeddingDimension must be positive, but was {options.EmbeddingDimension}.",
                nameof(options));
        }

        // Validate classifier weights match expected dimensions
        if (classifierWeights.Rows != options.NumClasses)
        {
            throw new ArgumentException(
                $"Classifier weights rows ({classifierWeights.Rows}) must match NumClasses ({options.NumClasses}).",
                nameof(classifierWeights));
        }

        if (classifierWeights.Columns != options.EmbeddingDimension)
        {
            throw new ArgumentException(
                $"Classifier weights columns ({classifierWeights.Columns}) must match EmbeddingDimension ({options.EmbeddingDimension}).",
                nameof(classifierWeights));
        }
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the classifier weights.
    /// </summary>
    public Matrix<T> ClassifierWeights => _classifierWeights;

    /// <summary>
    /// Gets the temperature parameter.
    /// </summary>
    public T Temperature => _temperature;

    /// <summary>
    /// Gets the number of classes.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Extract features
        var embeddings = ExtractEmbeddings(input);

        // Normalize if configured
        if (_options.NormalizeEmbeddings)
        {
            embeddings = NormalizeEmbeddings(embeddings);
        }

        // Compute logits
        var logits = ComputeLogits(embeddings);

        // Apply temperature scaling
        if (_options.UseLearnedTemperature)
        {
            logits = ScaleByTemperature(logits);
        }

        return ConvertToOutput(logits);
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the MetaOptNet algorithm to train the model.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("MetaOptNet model parameters are set during adaptation.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Return combined encoder + classifier parameters
        var encoderParams = _featureEncoder.GetParameters();
        int classifierSize = _classifierWeights.Rows * _classifierWeights.Columns;
        int totalSize = encoderParams.Length + classifierSize + 1; // +1 for temperature

        var combined = new Vector<T>(totalSize);
        int idx = 0;

        for (int i = 0; i < encoderParams.Length; i++)
        {
            combined[idx++] = encoderParams[i];
        }
        for (int i = 0; i < _classifierWeights.Rows; i++)
        {
            for (int j = 0; j < _classifierWeights.Columns; j++)
            {
                combined[idx++] = _classifierWeights[i, j];
            }
        }
        combined[idx] = _temperature;

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
    private Matrix<T> ExtractEmbeddings(TInput input)
    {
        var output = _featureEncoder.Predict(input);

        if (output is Vector<T> vec)
        {
            // Convert vector to matrix
            int numSamples = Math.Max(1, vec.Length / _options.EmbeddingDimension);
            var matrix = new Matrix<T>(numSamples, _options.EmbeddingDimension);
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < _options.EmbeddingDimension; j++)
                {
                    int idx = i * _options.EmbeddingDimension + j;
                    matrix[i, j] = idx < vec.Length ? vec[idx] : NumOps.Zero;
                }
            }
            return matrix;
        }

        if (output is Tensor<T> tensor)
        {
            var vec2 = tensor.ToVector();
            int numSamples = Math.Max(1, vec2.Length / _options.EmbeddingDimension);
            var matrix = new Matrix<T>(numSamples, _options.EmbeddingDimension);
            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < _options.EmbeddingDimension; j++)
                {
                    int idx = i * _options.EmbeddingDimension + j;
                    matrix[i, j] = idx < vec2.Length ? vec2[idx] : NumOps.Zero;
                }
            }
            return matrix;
        }

        return new Matrix<T>(1, _options.EmbeddingDimension);
    }

    /// <summary>
    /// Normalizes embeddings to unit norm.
    /// </summary>
    private Matrix<T> NormalizeEmbeddings(Matrix<T> embeddings)
    {
        var normalized = new Matrix<T>(embeddings.Rows, embeddings.Columns);

        for (int i = 0; i < embeddings.Rows; i++)
        {
            T normSq = NumOps.Zero;
            for (int j = 0; j < embeddings.Columns; j++)
            {
                normSq = NumOps.Add(normSq, NumOps.Multiply(embeddings[i, j], embeddings[i, j]));
            }
            double norm = Math.Sqrt(Math.Max(NumOps.ToDouble(normSq), 1e-8));

            for (int j = 0; j < embeddings.Columns; j++)
            {
                normalized[i, j] = NumOps.Divide(embeddings[i, j], NumOps.FromDouble(norm));
            }
        }

        return normalized;
    }

    /// <summary>
    /// Computes logits using classifier weights.
    /// </summary>
    private Vector<T> ComputeLogits(Matrix<T> embeddings)
    {
        var logits = new Vector<T>(embeddings.Rows * _classifierWeights.Columns);

        int idx = 0;
        for (int i = 0; i < embeddings.Rows; i++)
        {
            for (int c = 0; c < _classifierWeights.Columns; c++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < Math.Min(embeddings.Columns, _classifierWeights.Rows); j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(embeddings[i, j], _classifierWeights[j, c]));
                }
                logits[idx++] = sum;
            }
        }

        return logits;
    }

    /// <summary>
    /// Scales logits by temperature.
    /// </summary>
    private Vector<T> ScaleByTemperature(Vector<T> logits)
    {
        var scaled = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            scaled[i] = NumOps.Divide(logits[i], _temperature);
        }
        return scaled;
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
