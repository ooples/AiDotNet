using AiDotNet.Augmentation;
using AiDotNet.Augmentation.Image;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Results;

/// <summary>
/// Partial class containing Test-Time Augmentation (TTA) prediction methods.
/// </summary>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Makes a prediction using Test-Time Augmentation for improved accuracy.
    /// </summary>
    /// <typeparam name="TAugData">The augmentation data type (e.g., ImageTensor&lt;T&gt;).</typeparam>
    /// <param name="data">The input data to predict on.</param>
    /// <param name="convertToModelInput">Function to convert augmented data to model input type.</param>
    /// <param name="convertFromModelOutput">Function to convert model output to a format suitable for aggregation.</param>
    /// <param name="augmentationPipeline">
    /// The augmentation policy applied <c>TTANumAugmentations</c> times (from the configured augmentation options) to
    /// produce the augmented variants, or <c>null</c> to predict on the original input only, which then requires
    /// <c>TTAIncludeOriginal</c> to be enabled.
    /// </param>
    /// <returns>A TTA result containing the aggregated prediction and individual predictions.</returns>
    /// <exception cref="InvalidOperationException">Thrown if TTA is not configured or disabled, or if no prediction
    /// was made (no augmentation policy and <c>TTAIncludeOriginal</c> disabled).</exception>
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
        Func<TOutput, Vector<T>> convertFromModelOutput,
        IAugmentationPolicy<T, TAugData>? augmentationPipeline = null)
    {
        var augConfig = Options?.AugmentationConfig;
        if (augConfig == null)
        {
            throw new InvalidOperationException(
                "Augmentation is not configured. Use ConfigureAugmentation() when building the model.");
        }

        if (!augConfig.EnableTTA)
        {
            throw new InvalidOperationException("Test-Time Augmentation is disabled in the configuration.");
        }

        var predictions = new List<TOutput>();
        var context = new AugmentationContext<T>(isTraining: false, seed: augConfig.Seed);

        // Include original prediction if configured
        if (augConfig.TTAIncludeOriginal)
        {
            var originalInput = convertToModelInput(data);
            predictions.Add(Predict(originalInput));
        }

        // Generate augmented predictions using the provided pipeline
        if (augmentationPipeline is not null)
        {
            for (int i = 0; i < augConfig.TTANumAugmentations; i++)
            {
                context.SampleIndex = i;

                // Apply augmentation
                var augmented = augmentationPipeline.Apply(data, context);
                var augmentedInput = convertToModelInput(augmented);

                // Make prediction
                var prediction = Predict(augmentedInput);
                predictions.Add(prediction);
            }
        }

        if (predictions.Count == 0)
        {
            throw new InvalidOperationException(
                "No predictions were made. Ensure TTAIncludeOriginal is true or provide an augmentation pipeline.");
        }

        // Aggregate predictions
        var aggregated = AggregatePredictions(
            predictions,
            augConfig.TTAAggregation,
            convertFromModelOutput,
            confidenceThreshold: null); // Confidence threshold filtering not used in unified config

        // Calculate statistics
        double? confidence = null;
        double? stdDev = null;

        if (predictions.Count > 1)
        {
            var vectors = predictions.Select(convertFromModelOutput).ToList();
            stdDev = CalculateStandardDeviation(vectors);
            confidence = CalculateConfidence(vectors, augConfig.TTAAggregation);
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
    /// <param name="augmentationPipeline">
    /// The augmentation policy applied <c>TTANumAugmentations</c> times (from the configured augmentation options) to
    /// produce the augmented variants, or <c>null</c> to predict on the original input only, which then requires
    /// <c>TTAIncludeOriginal</c> to be enabled.
    /// </param>
    /// <returns>A TTA result containing the aggregated prediction and individual predictions.</returns>
    /// <remarks>
    /// This is a simplified overload for when TInput is directly augmentable (e.g., ImageTensor).
    /// For complex type conversions, use the full overload with converter functions.
    /// </remarks>
    public TestTimeAugmentationResult<TOutput> PredictWithTestTimeAugmentation(
        TInput data,
        IAugmentationPolicy<T, TInput>? augmentationPipeline = null)
    {
        // This overload requires TInput to be the same as TAugData
        // Use the generic version for type conversions
        return PredictWithTestTimeAugmentation(
            data,
            d => d,
            output => ConvertOutputToVector(output),
            augmentationPipeline);
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
            // Keep the predictions whose largest value (their confidence) reaches the threshold.
            var numOps = MathHelper.GetNumericOperations<T>();
            var confidences = ResultEngine.ReduceMax(StackPredictions(vectors), new[] { 1 }, false);
            var threshold = ResultEngine.TensorFullLike(confidences, numOps.FromDouble(confidenceThreshold.Value));
            int[] keep = ResultEngine.TensorGreaterEqual(confidences, threshold).Cast<int>().ToArray();
            var filtered = vectors.Where((_, index) => keep[index] != 0).ToList();

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
        // Each prediction is divided by the count before summing so the sum cannot overflow in low precision.
        var stacked = StackPredictions(vectors);
        var count = MathHelper.GetNumericOperations<T>().FromDouble(vectors.Count);
        return ResultEngine.ReduceSum(ResultEngine.TensorDivideScalar(stacked, count), new[] { 0 }, false).ToVector();
    }

    /// <summary>The predictions stacked into a [count, length] tensor.</summary>
    private static Tensor<T> StackPredictions(List<Vector<T>> vectors)
    {
        return ResultEngine.TensorStack(vectors.Select(vector => Tensor<T>.FromVector(vector)).ToArray(), 0);
    }

    /// <summary>The index of each prediction's largest value (the first one on ties).</summary>
    private static int[] PredictedClasses(List<Vector<T>> vectors)
    {
        return ResultEngine.TensorArgMax(StackPredictions(vectors), 1).ToArray();
    }

    /// <summary>
    /// Votes per predicted class, with the winner being the most voted class and ties going to the class that
    /// was voted for first.
    /// </summary>
    private static (int WinningClass, int Votes) CountVotes(List<Vector<T>> vectors)
    {
        var votes = new Dictionary<int, int>();
        var order = new List<int>();
        foreach (int predictedClass in PredictedClasses(vectors))
        {
            if (votes.TryGetValue(predictedClass, out int count))
            {
                votes[predictedClass] = count + 1;
            }
            else
            {
                votes[predictedClass] = 1;
                order.Add(predictedClass);
            }
        }

        int winningClass = order[0];
        foreach (int candidate in order)
        {
            if (votes[candidate] > votes[winningClass])
            {
                winningClass = candidate;
            }
        }

        return (winningClass, votes[winningClass]);
    }

    /// <summary>
    /// Computes the element-wise median of vectors.
    /// </summary>
    private Vector<T> ComputeMedian(List<Vector<T>> vectors)
    {
        int count = vectors.Count;
        int length = vectors[0].Length;
        var numOps = MathHelper.GetNumericOperations<T>();
        var stacked = StackPredictions(vectors);
        var sorted = ResultEngine.TensorSort(stacked, 0).Values;
        int mid = count / 2;
        var median = ResultEngine.Reshape(ResultEngine.TensorSlice(sorted, new[] { mid, 0 }, new[] { 1, length }), new[] { length });
        if (count % 2 == 0)
        {
            // The two middle values are halved before adding so their sum cannot overflow in low precision.
            var lower = ResultEngine.Reshape(ResultEngine.TensorSlice(sorted, new[] { mid - 1, 0 }, new[] { 1, length }), new[] { length });
            var two = numOps.FromDouble(2.0);
            median = ResultEngine.TensorAdd(ResultEngine.TensorDivideScalar(lower, two), ResultEngine.TensorDivideScalar(median, two));
        }

        // numpy.median: an element whose column holds any NaN is NaN (NaN is the only value unequal to itself).
        var hasNaN = ResultEngine.ReduceMax(ResultEngine.TensorNotEquals(stacked, stacked), new[] { 0 }, false);
        var nan = ResultEngine.TensorFullLike(median, numOps.FromDouble(double.NaN));
        return ResultEngine.TensorWhere(ResultEngine.TensorGreaterThan(hasNaN, numOps.Zero), nan, median).ToVector();
    }

    /// <summary>
    /// Computes the element-wise maximum of vectors.
    /// </summary>
    private Vector<T> ComputeMax(List<Vector<T>> vectors)
    {
        return ResultEngine.ReduceMax(StackPredictions(vectors), new[] { 0 }, false).ToVector();
    }

    /// <summary>
    /// Computes the element-wise minimum of vectors.
    /// </summary>
    private Vector<T> ComputeMin(List<Vector<T>> vectors)
    {
        // min(x) = -max(-x).
        var negated = ResultEngine.TensorNegate(StackPredictions(vectors));
        return ResultEngine.TensorNegate(ResultEngine.ReduceMax(negated, new[] { 0 }, false)).ToVector();
    }

    /// <summary>
    /// Computes majority voting for classification predictions.
    /// </summary>
    private Vector<T> ComputeVote(List<Vector<T>> vectors)
    {
        // Each vector votes for its argmax class; the result is the winning class one-hot encoded.
        int length = vectors[0].Length;
        var (winningClass, _) = CountVotes(vectors);
        var oneHot = ResultEngine.TensorOneHot<T>(new Tensor<int>(new[] { winningClass }, new[] { 1 }), length);
        return ResultEngine.Reshape(oneHot, new[] { length }).ToVector();
    }

    /// <summary>
    /// Computes weighted mean based on confidence scores.
    /// </summary>
    private Vector<T> ComputeWeightedMean(List<Vector<T>> vectors)
    {
        // Each prediction is weighted by its largest value (its confidence).
        var numOps = MathHelper.GetNumericOperations<T>();
        var stacked = StackPredictions(vectors);
        var weights = ResultEngine.ReduceMax(stacked, new[] { 1 }, false);

        // The weights are divided by their largest magnitude before summing, so the total cannot overflow in low
        // precision (65504 in Half) when the weights are large; the normalised weights w / sum(w) are unchanged.
        T peak = ResultEngine.TensorMaxValue(ResultEngine.TensorAbs(weights));
        if (numOps.Equals(peak, numOps.Zero))
        {
            return ComputeMean(vectors); // Fallback to simple mean
        }

        var relative = ResultEngine.TensorDivideScalar(weights, peak);
        T relativeTotal = ResultEngine.TensorSum(relative);

        // |sum(w)| below 1e-10 falls back to the simple mean; compared in double as |sum(w / peak)| * peak.
        if (Math.Abs(numOps.ToDouble(relativeTotal) * numOps.ToDouble(peak)) < 1e-10)
        {
            return ComputeMean(vectors); // Fallback to simple mean
        }

        // Normalising the weights first keeps the weighted sum within range in low precision.
        var normalized = ResultEngine.Reshape(ResultEngine.TensorDivideScalar(relative, relativeTotal), new[] { 1, vectors.Count });
        return ResultEngine.Reshape(ResultEngine.TensorMatMul(normalized, stacked), new[] { vectors[0].Length }).ToVector();
    }

    /// <summary>
    /// Computes the geometric mean of vectors using log-sum-exp for numerical stability.
    /// </summary>
    private Vector<T> ComputeGeometricMean(List<Vector<T>> vectors)
    {
        // exp(mean(log(max(x, eps)))), with eps = 1e-10 or the smallest normal value of a narrower type.
        var numOps = MathHelper.GetNumericOperations<T>();
        var floored = ResultEngine.TensorMax(StackPredictions(vectors), MatrixExtensions.SafeTolerance<T>(1e-10));
        var logs = ResultEngine.TensorDivideScalar(ResultEngine.TensorLog(floored), numOps.FromDouble(vectors.Count));
        return ResultEngine.TensorExp(ResultEngine.ReduceSum(logs, new[] { 0 }, false)).ToVector();
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

        // Population standard deviation of every value around its element-wise mean.
        var numOps = MathHelper.GetNumericOperations<T>();
        int count = vectors.Count;
        int length = vectors[0].Length;
        var stacked = StackPredictions(vectors);
        var mean = ResultEngine.Reshape(Tensor<T>.FromVector(ComputeMean(vectors)), new[] { 1, length });
        var deviations = ResultEngine.TensorSubtract(stacked, ResultEngine.TensorBroadcastTo(mean, new[] { count, length }));

        // The deviations are divided by their largest magnitude before squaring so the squares cannot overflow,
        // and the mean of squares is taken per row then across rows so no partial sum exceeds the row length.
        T scale = ResultEngine.TensorMaxValue(ResultEngine.TensorAbs(deviations));
        if (numOps.Equals(scale, numOps.Zero))
        {
            return 0;
        }

        var squares = ResultEngine.TensorSquare(ResultEngine.TensorDivideScalar(deviations, scale));
        var rowMeans = ResultEngine.ReduceSum(ResultEngine.TensorDivideScalar(squares, numOps.FromDouble(length)), new[] { 1 }, false);
        T meanSquare = ResultEngine.TensorSum(ResultEngine.TensorDivideScalar(rowMeans, numOps.FromDouble(count)));
        return numOps.ToDouble(scale) * Math.Sqrt(numOps.ToDouble(meanSquare));
    }

    /// <summary>
    /// Calculates confidence based on prediction agreement.
    /// </summary>
    private double CalculateConfidence(List<Vector<T>> vectors, PredictionAggregationMethod method)
    {
        if (method == PredictionAggregationMethod.Vote)
        {
            // For voting, confidence is the proportion of votes for the winning class
            var (_, maxVotes) = CountVotes(vectors);
            return (double)maxVotes / vectors.Count;
        }
        else
        {
            // For other methods, confidence is inversely related to variance
            // Use normalized formula: 1/(1+stdDev) to properly map to [0,1] range
            double stdDev = CalculateStandardDeviation(vectors);
            return 1.0 / (1.0 + stdDev);
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
            // Flatten matrix to vector (row-major)
            return matrix.ToRowVector();
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
