using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Image;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Results;

/// <summary>
/// Partial class containing Test-Time Augmentation (TTA) prediction methods.
/// </summary>
public partial class PredictionModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Makes a prediction using Test-Time Augmentation for improved accuracy.
    /// </summary>
    /// <typeparam name="TAugData">The augmentation data type (e.g., ImageTensor&lt;T&gt;).</typeparam>
    /// <param name="data">The input data to predict on.</param>
    /// <param name="convertToModelInput">Function to convert augmented data to model input type.</param>
    /// <param name="convertFromModelOutput">Function to convert model output to a format suitable for aggregation.</param>
    /// <returns>A TTA result containing the aggregated prediction and individual predictions.</returns>
    /// <exception cref="InvalidOperationException">Thrown if TTA is not configured or disabled.</exception>
    /// <remarks>
    /// <para>
    /// Test-Time Augmentation improves prediction accuracy by:
    /// 1. Creating multiple augmented versions of the input
    /// 2. Making predictions on each version
    /// 3. Aggregating predictions using the configured method (mean, median, vote, etc.)
    /// </para>
    /// <para><b>For Beginners:</b> Instead of making one prediction, TTA makes several predictions
    /// on variations of your input (flipped, rotated, etc.) and combines them for a more robust answer.
    /// This typically improves accuracy by 1-3% at the cost of slower inference.
    /// </para>
    /// </remarks>
    public TestTimeAugmentationResult<TOutput> PredictWithTestTimeAugmentation<TAugData>(
        TAugData data,
        Func<TAugData, TInput> convertToModelInput,
        Func<TOutput, Vector<T>> convertFromModelOutput)
    {
        if (Options?.TTAConfiguration == null)
        {
            throw new InvalidOperationException(
                "Test-Time Augmentation is not configured. Use ConfigureTestTimeAugmentation() when building the model.");
        }

        var ttaConfig = Options.TTAConfiguration as TestTimeAugmentationConfiguration<T, TAugData>;
        if (ttaConfig == null)
        {
            throw new InvalidOperationException(
                $"TTA configuration type mismatch. Expected TestTimeAugmentationConfiguration<{typeof(T).Name}, {typeof(TAugData).Name}>.");
        }

        if (!ttaConfig.IsEnabled)
        {
            throw new InvalidOperationException("Test-Time Augmentation is disabled in the configuration.");
        }

        var predictions = new List<TOutput>();
        var context = new AugmentationContext<T>(isTraining: false, seed: ttaConfig.Seed);

        // Include original prediction if configured
        if (ttaConfig.IncludeOriginal)
        {
            var originalInput = convertToModelInput(data);
            predictions.Add(Predict(originalInput));
        }

        // Generate augmented predictions
        if (ttaConfig.Pipeline != null)
        {
            for (int i = 0; i < ttaConfig.NumberOfAugmentations; i++)
            {
                context.SampleIndex = i;

                // Apply augmentation
                var augmented = ttaConfig.Pipeline.Apply(data, context);
                var augmentedInput = convertToModelInput(augmented);

                // Make prediction
                var prediction = Predict(augmentedInput);
                predictions.Add(prediction);
            }
        }

        if (predictions.Count == 0)
        {
            throw new InvalidOperationException(
                "No predictions were made. Ensure IncludeOriginal is true or Pipeline is configured.");
        }

        // Aggregate predictions
        var aggregated = AggregatePredictions(
            predictions,
            ttaConfig.AggregationMethod,
            convertFromModelOutput,
            ttaConfig.ConfidenceThreshold);

        // Calculate statistics
        double? confidence = null;
        double? stdDev = null;

        if (predictions.Count > 1)
        {
            var vectors = predictions.Select(convertFromModelOutput).ToList();
            stdDev = CalculateStandardDeviation(vectors);
            confidence = CalculateConfidence(vectors, ttaConfig.AggregationMethod);
        }

        return new TestTimeAugmentationResult<TOutput>(
            aggregated,
            predictions.AsReadOnly(),
            confidence,
            stdDev);
    }

    /// <summary>
    /// Makes a prediction using Test-Time Augmentation when input/output types match augmentation types.
    /// </summary>
    /// <param name="data">The input data to predict on (must be augmentable).</param>
    /// <returns>A TTA result containing the aggregated prediction and individual predictions.</returns>
    /// <remarks>
    /// This is a simplified overload for when TInput is directly augmentable (e.g., ImageTensor).
    /// For complex type conversions, use the full overload with converter functions.
    /// </remarks>
    public TestTimeAugmentationResult<TOutput> PredictWithTestTimeAugmentation(TInput data)
    {
        // This overload requires TInput to be the same as TAugData
        // Use the generic version for type conversions
        return PredictWithTestTimeAugmentation(
            data,
            d => d,
            output => ConvertOutputToVector(output));
    }

    /// <summary>
    /// Aggregates multiple predictions into a single prediction based on the specified method.
    /// </summary>
    private TOutput AggregatePredictions(
        List<TOutput> predictions,
        PredictionAggregationMethod method,
        Func<TOutput, Vector<T>> convertFromOutput,
        double? confidenceThreshold)
    {
        if (predictions.Count == 1)
        {
            return predictions[0];
        }

        // Convert predictions to vectors for aggregation
        var vectors = predictions.Select(convertFromOutput).ToList();

        // Filter by confidence threshold if specified
        if (confidenceThreshold.HasValue)
        {
            var filtered = new List<Vector<T>>();
            for (int i = 0; i < vectors.Count; i++)
            {
                var maxConfidence = vectors[i].ToArray().Select(v => Convert.ToDouble(v)).Max();
                if (maxConfidence >= confidenceThreshold.Value)
                {
                    filtered.Add(vectors[i]);
                }
            }

            if (filtered.Count > 0)
            {
                vectors = filtered;
            }
        }

        Vector<T> aggregatedVector;

        switch (method)
        {
            case PredictionAggregationMethod.Mean:
                aggregatedVector = ComputeMean(vectors);
                break;

            case PredictionAggregationMethod.Median:
                aggregatedVector = ComputeMedian(vectors);
                break;

            case PredictionAggregationMethod.Max:
                aggregatedVector = ComputeMax(vectors);
                break;

            case PredictionAggregationMethod.Min:
                aggregatedVector = ComputeMin(vectors);
                break;

            case PredictionAggregationMethod.Vote:
                aggregatedVector = ComputeVote(vectors);
                break;

            case PredictionAggregationMethod.WeightedMean:
                aggregatedVector = ComputeWeightedMean(vectors);
                break;

            case PredictionAggregationMethod.GeometricMean:
                aggregatedVector = ComputeGeometricMean(vectors);
                break;

            default:
                aggregatedVector = ComputeMean(vectors);
                break;
        }

        return ConvertVectorToOutput(aggregatedVector);
    }

    /// <summary>
    /// Computes the element-wise mean of vectors.
    /// </summary>
    private Vector<T> ComputeMean(List<Vector<T>> vectors)
    {
        int length = vectors[0].Length;
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            double sum = 0;
            foreach (var vector in vectors)
            {
                sum += Convert.ToDouble(vector[i]);
            }

            result[i] = (T)Convert.ChangeType(sum / vectors.Count, typeof(T));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes the element-wise median of vectors.
    /// </summary>
    private Vector<T> ComputeMedian(List<Vector<T>> vectors)
    {
        int length = vectors[0].Length;
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            var values = vectors.Select(v => Convert.ToDouble(v[i])).OrderBy(v => v).ToList();
            int mid = values.Count / 2;
            double median = values.Count % 2 == 0
                ? (values[mid - 1] + values[mid]) / 2.0
                : values[mid];
            result[i] = (T)Convert.ChangeType(median, typeof(T));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes the element-wise maximum of vectors.
    /// </summary>
    private Vector<T> ComputeMax(List<Vector<T>> vectors)
    {
        int length = vectors[0].Length;
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            double max = vectors.Max(v => Convert.ToDouble(v[i]));
            result[i] = (T)Convert.ChangeType(max, typeof(T));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes the element-wise minimum of vectors.
    /// </summary>
    private Vector<T> ComputeMin(List<Vector<T>> vectors)
    {
        int length = vectors[0].Length;
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            double min = vectors.Min(v => Convert.ToDouble(v[i]));
            result[i] = (T)Convert.ChangeType(min, typeof(T));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes majority voting for classification predictions.
    /// </summary>
    private Vector<T> ComputeVote(List<Vector<T>> vectors)
    {
        int length = vectors[0].Length;
        var votes = new Dictionary<int, int>();

        // Each vector votes for its argmax class
        foreach (var vector in vectors)
        {
            int predictedClass = 0;
            double maxValue = Convert.ToDouble(vector[0]);

            for (int i = 1; i < length; i++)
            {
                double value = Convert.ToDouble(vector[i]);
                if (value > maxValue)
                {
                    maxValue = value;
                    predictedClass = i;
                }
            }

            votes.TryGetValue(predictedClass, out int count);
            votes[predictedClass] = count + 1;
        }

        // Find winning class
        int winningClass = votes.OrderByDescending(kvp => kvp.Value).First().Key;

        // Create one-hot result
        var result = new T[length];
        for (int i = 0; i < length; i++)
        {
            result[i] = i == winningClass
                ? (T)Convert.ChangeType(1.0, typeof(T))
                : (T)Convert.ChangeType(0.0, typeof(T));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes weighted mean based on confidence scores.
    /// </summary>
    private Vector<T> ComputeWeightedMean(List<Vector<T>> vectors)
    {
        int length = vectors[0].Length;
        var result = new T[length];

        // Weight by max confidence in each vector
        var weights = vectors.Select(v => v.ToArray().Select(x => Convert.ToDouble(x)).Max()).ToList();
        double totalWeight = weights.Sum();

        if (Math.Abs(totalWeight) < 1e-10)
        {
            return ComputeMean(vectors); // Fallback to simple mean
        }

        for (int i = 0; i < length; i++)
        {
            double weightedSum = 0;
            for (int j = 0; j < vectors.Count; j++)
            {
                weightedSum += Convert.ToDouble(vectors[j][i]) * weights[j];
            }

            result[i] = (T)Convert.ChangeType(weightedSum / totalWeight, typeof(T));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes the geometric mean of vectors.
    /// </summary>
    private Vector<T> ComputeGeometricMean(List<Vector<T>> vectors)
    {
        int length = vectors[0].Length;
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            double product = 1.0;
            foreach (var vector in vectors)
            {
                double value = Math.Max(Convert.ToDouble(vector[i]), 1e-10); // Avoid log(0)
                product *= value;
            }

            double geoMean = Math.Pow(product, 1.0 / vectors.Count);
            result[i] = (T)Convert.ChangeType(geoMean, typeof(T));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Calculates the standard deviation across predictions.
    /// </summary>
    private double CalculateStandardDeviation(List<Vector<T>> vectors)
    {
        if (vectors.Count < 2)
        {
            return 0;
        }

        var mean = ComputeMean(vectors);
        double sumSquaredDiff = 0;
        int count = 0;

        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                double diff = Convert.ToDouble(vector[i]) - Convert.ToDouble(mean[i]);
                sumSquaredDiff += diff * diff;
                count++;
            }
        }

        return Math.Sqrt(sumSquaredDiff / count);
    }

    /// <summary>
    /// Calculates confidence based on prediction agreement.
    /// </summary>
    private double CalculateConfidence(List<Vector<T>> vectors, PredictionAggregationMethod method)
    {
        if (method == PredictionAggregationMethod.Vote)
        {
            // For voting, confidence is the proportion of votes for the winning class
            var votes = new Dictionary<int, int>();

            foreach (var vector in vectors)
            {
                int predictedClass = 0;
                double maxValue = Convert.ToDouble(vector[0]);

                for (int i = 1; i < vector.Length; i++)
                {
                    double value = Convert.ToDouble(vector[i]);
                    if (value > maxValue)
                    {
                        maxValue = value;
                        predictedClass = i;
                    }
                }

                votes.TryGetValue(predictedClass, out int count);
                votes[predictedClass] = count + 1;
            }

            int maxVotes = votes.Values.Max();
            return (double)maxVotes / vectors.Count;
        }
        else
        {
            // For other methods, confidence is inversely related to variance
            double stdDev = CalculateStandardDeviation(vectors);
            return Math.Max(0, 1.0 - stdDev);
        }
    }

    /// <summary>
    /// Converts model output to a vector for aggregation.
    /// </summary>
    private Vector<T> ConvertOutputToVector(TOutput output)
    {
        if (output is Vector<T> vector)
        {
            return vector;
        }

        if (output is Matrix<T> matrix)
        {
            // Flatten matrix to vector
            var data = new T[matrix.Rows * matrix.Columns];
            int idx = 0;
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    data[idx++] = matrix[i, j];
                }
            }

            return new Vector<T>(data);
        }

        if (output is Tensor<T> tensor)
        {
            // Use tensor's ToVector if available
            var data = tensor.ToArray();
            return new Vector<T>(data);
        }

        throw new InvalidOperationException(
            $"Cannot convert output type {typeof(TOutput).Name} to Vector<{typeof(T).Name}> for aggregation. " +
            "Provide a custom converter function.");
    }

    /// <summary>
    /// Converts an aggregated vector back to the output type.
    /// </summary>
    private TOutput ConvertVectorToOutput(Vector<T> vector)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)vector;
        }

        // For other types, this would need custom handling
        // Most common case is Vector<T> output, so this should work for most scenarios
        throw new InvalidOperationException(
            $"Cannot convert aggregated Vector<{typeof(T).Name}> back to {typeof(TOutput).Name}. " +
            "For non-vector outputs, use a custom conversion approach.");
    }
}
