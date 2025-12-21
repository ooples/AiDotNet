using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// TADAM model for few-shot classification with task conditioning and metric scaling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of TADAM after computing task-conditioned prototypes.
/// It uses learned metric scaling and temperature to classify new query examples.
/// </para>
/// <para><b>For Beginners:</b> After TADAM sees the support examples and computes
/// task-conditioned prototypes, this model stores those prototypes along with the
/// learned metric scaling parameters. It can then classify new examples by measuring
/// scaled distances to these prototypes.
/// </para>
/// </remarks>
public class TADAMModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly Dictionary<int, Tensor<T>> _prototypes;
    private readonly Vector<T> _metricScale;
    private readonly T _temperature;
    private readonly TADAMOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the TADAMModel.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network.</param>
    /// <param name="prototypes">The computed class prototypes.</param>
    /// <param name="metricScale">The learned metric scaling parameters.</param>
    /// <param name="temperature">The learned temperature parameter.</param>
    /// <param name="options">The TADAM options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public TADAMModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Dictionary<int, Tensor<T>> prototypes,
        Vector<T> metricScale,
        T temperature,
        TADAMOptions<T, TInput, TOutput> options)
    {
        _featureEncoder = featureEncoder ?? throw new ArgumentNullException(nameof(featureEncoder));
        _prototypes = prototypes ?? throw new ArgumentNullException(nameof(prototypes));
        _metricScale = metricScale ?? throw new ArgumentNullException(nameof(metricScale));
        _temperature = temperature;
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Encode the input using the feature encoder
        var encoderOutput = _featureEncoder.Predict(input);
        var queryEmbedding = ConversionsHelper.ConvertToVector<T, TOutput>(encoderOutput);

        // Optionally normalize the embedding
        if (_options.NormalizeEmbeddings)
        {
            queryEmbedding = NormalizeVector(queryEmbedding);
        }

        // Compute scaled distances to each prototype
        var distances = ComputeScaledDistances(queryEmbedding);

        // Convert distances to logits (negative distances scaled by temperature)
        var logits = ComputeLogits(distances);

        // Apply softmax to get class probabilities
        var probabilities = ApplySoftmax(logits);

        // Convert to output type
        return ConvertToOutput(probabilities);
    }

    /// <summary>
    /// Normalizes a vector to unit length.
    /// </summary>
    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        T norm = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            norm = NumOps.Add(norm, NumOps.Multiply(vector[i], vector[i]));
        }

        double normValue = Math.Sqrt(NumOps.ToDouble(norm));
        if (normValue < 1e-10)
        {
            return vector; // Avoid division by zero
        }

        var normalized = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = NumOps.Divide(vector[i], NumOps.FromDouble(normValue));
        }

        return normalized;
    }

    /// <summary>
    /// Computes scaled distances from the query to each class prototype.
    /// </summary>
    private Vector<T> ComputeScaledDistances(Vector<T> query)
    {
        var distances = new Vector<T>(_options.NumClasses);

        foreach (var kvp in _prototypes)
        {
            int classIdx = kvp.Key;
            var prototype = kvp.Value;

            if (classIdx >= 0 && classIdx < _options.NumClasses)
            {
                T distance = ComputeScaledDistance(query, prototype);
                distances[classIdx] = distance;
            }
        }

        return distances;
    }

    /// <summary>
    /// Computes the scaled Euclidean distance between query and prototype.
    /// </summary>
    private T ComputeScaledDistance(Vector<T> query, Tensor<T> prototype)
    {
        T distanceSum = NumOps.Zero;
        int dim = Math.Min(query.Length, prototype.Length);

        for (int i = 0; i < dim; i++)
        {
            T diff = NumOps.Subtract(query[i], prototype.GetFlat(i));
            T scaledDiff = diff;

            // Apply metric scaling if available
            if (_options.UseMetricScaling && i < _metricScale.Length)
            {
                scaledDiff = NumOps.Multiply(diff, _metricScale[i]);
            }

            distanceSum = NumOps.Add(distanceSum, NumOps.Multiply(scaledDiff, scaledDiff));
        }

        return distanceSum;
    }

    /// <summary>
    /// Converts distances to logits using temperature scaling.
    /// </summary>
    private Vector<T> ComputeLogits(Vector<T> distances)
    {
        var logits = new Vector<T>(distances.Length);
        double temp = NumOps.ToDouble(_temperature);
        if (temp < 1e-10)
        {
            temp = 1.0; // Avoid division by zero
        }

        for (int i = 0; i < distances.Length; i++)
        {
            // Logit = -distance / temperature (negative because closer = higher probability)
            double distValue = NumOps.ToDouble(distances[i]);
            logits[i] = NumOps.FromDouble(-distValue / temp);
        }

        return logits;
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    private Vector<T> ApplySoftmax(Vector<T> logits)
    {
        var probabilities = new Vector<T>(logits.Length);

        // Find max for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.ToDouble(logits[i]) > NumOps.ToDouble(maxLogit))
            {
                maxLogit = logits[i];
            }
        }

        // Compute exp(logit - max) and sum
        T sum = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            double expValue = Math.Exp(NumOps.ToDouble(logits[i]) - NumOps.ToDouble(maxLogit));
            probabilities[i] = NumOps.FromDouble(expValue);
            sum = NumOps.Add(sum, probabilities[i]);
        }

        // Normalize
        if (NumOps.ToDouble(sum) > 0)
        {
            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] = NumOps.Divide(probabilities[i], sum);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Converts probability vector to the expected output type.
    /// </summary>
    private TOutput ConvertToOutput(Vector<T> probabilities)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)probabilities;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)ConversionsHelper.VectorToTensor(probabilities, new int[] { probabilities.Length });
        }

        // Fallback: return the argmax as a scalar
        int predictedClass = 0;
        T maxProb = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++)
        {
            if (NumOps.ToDouble(probabilities[i]) > NumOps.ToDouble(maxProb))
            {
                maxProb = probabilities[i];
                predictedClass = i;
            }
        }

        var result = new Vector<T>(1);
        result[0] = NumOps.FromDouble(predictedClass);
        if (result is TOutput output)
        {
            return output;
        }

        if (probabilities is TOutput prob)
        {
            return prob;
        }

        throw new InvalidOperationException(
            $"Cannot convert prediction result to output type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<T> and Tensor<T>.");
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the training algorithm to train TADAM.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("TADAM parameters are updated during training.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        return _featureEncoder.GetParameters();
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
