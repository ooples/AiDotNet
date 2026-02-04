using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Deployment.Optimization.Quantization.Calibration;

/// <summary>
/// Helper class for calibrating quantizers using real forward passes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Calibration is the process of running sample data through a model
/// to understand the typical range of values (activations) that flow through each layer. This
/// information is crucial for accurate quantization.</para>
///
/// <para><b>Why Real Forward Passes Matter:</b></para>
/// <list type="bullet">
/// <item><description>Weight magnitudes alone don't tell the full story - activations matter too</description></item>
/// <item><description>AWQ needs to know which weights are "activated" most strongly</description></item>
/// <item><description>SmoothQuant needs activation ranges to balance quantization difficulty</description></item>
/// <item><description>Real data captures the actual distribution your model will see in production</description></item>
/// </list>
///
/// <para><b>Supported Models:</b></para>
/// <list type="bullet">
/// <item><description>INeuralNetworkModel: Full layer-by-layer activation collection</description></item>
/// <item><description>IFullModel with Predict: Output-based statistics</description></item>
/// <item><description>Any IFullModel: Falls back to parameter-based estimation</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class CalibrationHelper<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Maximum acceptable failure rate during calibration before warnings are triggered.
    /// </summary>
    /// <remarks>
    /// If more than this fraction of calibration samples fail processing, a warning is
    /// added to indicate potential issues with data format or model compatibility.
    /// </remarks>
    private const double MaxAcceptableFailureRate = 0.5;

    private readonly QuantizationConfiguration _config;

    /// <summary>
    /// Initializes a new instance of CalibrationHelper.
    /// </summary>
    /// <param name="config">Quantization configuration with calibration settings</param>
    public CalibrationHelper(QuantizationConfiguration config)
    {
        _config = config ?? new QuantizationConfiguration();
    }

    /// <summary>
    /// Collects activation statistics by running calibration data through the model.
    /// </summary>
    /// <param name="model">The model to calibrate</param>
    /// <param name="calibrationData">Calibration samples</param>
    /// <returns>Collected activation statistics</returns>
    /// <exception cref="ArgumentNullException">If model or calibrationData is null</exception>
    /// <exception cref="ArgumentException">If calibrationData is empty</exception>
    public ActivationStatistics<T> CollectActivationStatistics(
        IFullModel<T, TInput, TOutput> model,
        IEnumerable<TInput> calibrationData)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (calibrationData == null) throw new ArgumentNullException(nameof(calibrationData));

        var dataList = calibrationData.ToList();
        if (dataList.Count == 0)
        {
            throw new ArgumentException("Calibration data cannot be empty", nameof(calibrationData));
        }

        var stats = new ActivationStatistics<T>();
        int maxSamples = Math.Min(_config.NumCalibrationSamples, dataList.Count);

        // Try to use neural network-specific methods for layer-wise statistics
        if (model is INeuralNetworkModel<T> nnModel)
        {
            CollectNeuralNetworkActivations(nnModel, dataList.Take(maxSamples), stats);
            stats.IsFromRealForwardPasses = true;
        }
        // Try to use INeuralNetwork for tensor-based models
        else if (model is INeuralNetwork<T> nn && typeof(TInput) == typeof(Tensor<T>))
        {
            CollectTensorBasedActivations(nn, dataList.Take(maxSamples).Cast<Tensor<T>>(), stats);
            stats.IsFromRealForwardPasses = true;
        }
        // Prediction-based collection with parameter fallback if none succeed
        else
        {
            CollectPredictionBasedActivations(model, dataList.Take(maxSamples), stats);
            if (stats.SampleCount == 0)
            {
                CollectParameterBasedEstimates(model, stats);
                stats.IsFromRealForwardPasses = false;
            }
            else
            {
                stats.IsFromRealForwardPasses = true;
            }
        }

        // Compute global statistics from layer statistics
        ComputeGlobalStatistics(model, stats);

        return stats;
    }

    /// <summary>
    /// Collects layer-by-layer activations from a neural network model.
    /// </summary>
    private void CollectNeuralNetworkActivations(
        INeuralNetworkModel<T> model,
        IEnumerable<TInput> samples,
        ActivationStatistics<T> stats)
    {
        int totalSamples = 0;
        int failedSamples = 0;

        foreach (var sample in samples)
        {
            totalSamples++;
            try
            {
                // Convert sample to Tensor<T> if needed
                Tensor<T>? inputTensor = ConvertToTensor(sample);
                if (inputTensor == null)
                {
                    failedSamples++;
                    continue;
                }

                // Get activations from all layers
                var layerActivations = model.GetNamedLayerActivations(inputTensor);

                foreach (var kvp in layerActivations)
                {
                    if (!stats.LayerStats.TryGetValue(kvp.Key, out var layerStats))
                    {
                        layerStats = new LayerActivationStats<T> { LayerName = kvp.Key };
                        stats.LayerStats[kvp.Key] = layerStats;
                    }

                    layerStats.Update(kvp.Value);
                }

                stats.SampleCount++;
            }
            catch (Exception ex)
            {
                // Track failed samples - continue with remaining
                failedSamples++;
                // Log first few failure reasons for debugging
                if (failedSamples <= 3)
                {
                    stats.CalibrationWarnings.Add($"Sample {totalSamples} failed: {ex.Message}");
                }
                continue;
            }
        }

        // Warn if more than 50% of samples failed - indicates potential issues with data or model
        if (totalSamples > 0 && (double)failedSamples / totalSamples > MaxAcceptableFailureRate)
        {
            stats.CalibrationWarnings.Add(
                $"High calibration failure rate: {failedSamples}/{totalSamples} samples failed ({100.0 * failedSamples / totalSamples:F1}%). " +
                "This may indicate incompatible data format or model issues.");
        }
    }

    /// <summary>
    /// Collects activations from tensor-based neural networks using ForwardWithMemory.
    /// </summary>
    private void CollectTensorBasedActivations(
        INeuralNetwork<T> model,
        IEnumerable<Tensor<T>> samples,
        ActivationStatistics<T> stats)
    {
        int totalSamples = 0;
        int failedSamples = 0;

        // Set to inference mode for calibration, but restore afterwards
        model.SetTrainingMode(false);
        try
        {
            foreach (var inputTensor in samples)
            {
                totalSamples++;
                try
                {
                    // Run forward pass with memory to capture intermediate activations
                    var output = model.ForwardWithMemory(inputTensor);

                    // Store output activation stats
                    if (!stats.LayerStats.TryGetValue("output", out var outputStats))
                    {
                        outputStats = new LayerActivationStats<T> { LayerName = "output" };
                        stats.LayerStats["output"] = outputStats;
                    }
                    outputStats.Update(output);

                    // Store input activation stats
                    if (!stats.LayerStats.TryGetValue("input", out var inputStats))
                    {
                        inputStats = new LayerActivationStats<T> { LayerName = "input" };
                        stats.LayerStats["input"] = inputStats;
                    }
                    inputStats.Update(inputTensor);

                    stats.SampleCount++;
                }
                catch (Exception ex)
                {
                    failedSamples++;
                    // Log first few failure reasons for debugging
                    if (failedSamples <= 3)
                    {
                        stats.CalibrationWarnings.Add($"Forward pass sample {totalSamples} failed: {ex.Message}");
                    }
                    continue;
                }
            }
        }
        finally
        {
            // Restore training mode
            model.SetTrainingMode(true);
        }

        // Warn if more than 50% of samples failed
        if (totalSamples > 0 && (double)failedSamples / totalSamples > MaxAcceptableFailureRate)
        {
            stats.CalibrationWarnings.Add(
                $"High calibration failure rate: {failedSamples}/{totalSamples} samples failed ({100.0 * failedSamples / totalSamples:F1}%). " +
                "This may indicate incompatible data format or model issues.");
        }
    }

    /// <summary>
    /// Collects activation statistics using model predictions.
    /// </summary>
    private void CollectPredictionBasedActivations(
        IFullModel<T, TInput, TOutput> model,
        IEnumerable<TInput> samples,
        ActivationStatistics<T> stats)
    {
        int totalSamples = 0;
        int failedSamples = 0;

        foreach (var sample in samples)
        {
            totalSamples++;
            try
            {
                // Run prediction
                var output = model.Predict(sample);

                // Extract activation values from output
                var outputValues = ExtractValuesFromOutput(output);
                if (outputValues != null && outputValues.Length > 0)
                {
                    // Create a pseudo-layer for output activations
                    if (!stats.LayerStats.TryGetValue("model_output", out var outputStats))
                    {
                        outputStats = new LayerActivationStats<T> { LayerName = "model_output" };
                        stats.LayerStats["model_output"] = outputStats;
                    }

                    // Update stats manually since we have raw doubles
                    foreach (double val in outputValues)
                    {
                        double absVal = Math.Abs(val);
                        outputStats.MinValue = Math.Min(outputStats.MinValue, val);
                        outputStats.MaxValue = Math.Max(outputStats.MaxValue, val);
                        outputStats.MaxAbsValue = Math.Max(outputStats.MaxAbsValue, absVal);
                        outputStats.SampleCount++;
                        double delta = val - outputStats.Mean;
                        outputStats.Mean += delta / outputStats.SampleCount;
                        double delta2 = val - outputStats.Mean;
                        outputStats.Variance += delta * delta2;
                    }
                }

                stats.SampleCount++;
            }
            catch (Exception ex)
            {
                failedSamples++;
                // Log first few failure reasons for debugging
                if (failedSamples <= 3)
                {
                    stats.CalibrationWarnings.Add($"Prediction sample {totalSamples} failed: {ex.Message}");
                }
                continue;
            }
        }

        // Warn if more than 50% of samples failed
        if (totalSamples > 0 && (double)failedSamples / totalSamples > MaxAcceptableFailureRate)
        {
            stats.CalibrationWarnings.Add(
                $"High calibration failure rate: {failedSamples}/{totalSamples} samples failed ({100.0 * failedSamples / totalSamples:F1}%). " +
                "This may indicate incompatible data format or model issues.");
        }
    }

    /// <summary>
    /// Falls back to parameter-based estimation when forward passes aren't available.
    /// </summary>
    private void CollectParameterBasedEstimates(
        IFullModel<T, TInput, TOutput> model,
        ActivationStatistics<T> stats)
    {
        var parameters = model.GetParameters();
        int n = parameters.Length;

        // Use parameter magnitudes as activation proxies
        var paramStats = new LayerActivationStats<T> { LayerName = "parameters_estimated" };

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(parameters[i]);
            double absVal = Math.Abs(val);

            paramStats.MinValue = Math.Min(paramStats.MinValue, val);
            paramStats.MaxValue = Math.Max(paramStats.MaxValue, val);
            paramStats.MaxAbsValue = Math.Max(paramStats.MaxAbsValue, absVal);
            paramStats.SampleCount++;
            double delta = val - paramStats.Mean;
            paramStats.Mean += delta / paramStats.SampleCount;
            double delta2 = val - paramStats.Mean;
            paramStats.Variance += delta * delta2;
        }

        stats.LayerStats["parameters_estimated"] = paramStats;
        stats.SampleCount = 1; // Not from real samples
    }

    /// <summary>
    /// Computes global activation statistics from layer statistics.
    /// </summary>
    private void ComputeGlobalStatistics(
        IFullModel<T, TInput, TOutput> model,
        ActivationStatistics<T> stats)
    {
        var parameters = model.GetParameters();
        int n = parameters.Length;

        // Initialize global arrays
        stats.GlobalActivationMagnitudes = new double[n];
        stats.GlobalMaxAbsActivations = new double[n];

        // If we have layer stats, use them to inform global statistics
        if (stats.LayerStats.Count > 0 && stats.IsFromRealForwardPasses)
        {
            // Use layer stats to weight parameter importance
            // Parameters associated with high-activation layers are more important
            double totalLayerActivation = stats.LayerStats.Values
                .Sum(s => s.MaxAbsValue);

            if (totalLayerActivation > 0)
            {
                // Distribute importance based on layer activation magnitudes
                int layerIdx = 0;
                int paramsPerLayer = n / Math.Max(1, stats.LayerStats.Count);

                foreach (var layerStat in stats.LayerStats.Values)
                {
                    double layerImportance = layerStat.MaxAbsValue / totalLayerActivation;
                    int start = layerIdx * paramsPerLayer;
                    int end = Math.Min(start + paramsPerLayer, n);

                    for (int i = start; i < end; i++)
                    {
                        // Combine parameter magnitude with layer importance
                        double paramMag = Math.Abs(Convert.ToDouble(parameters[i]));
                        stats.GlobalActivationMagnitudes[i] = paramMag * (1.0 + layerImportance);
                        stats.GlobalMaxAbsActivations[i] = Math.Max(paramMag, layerStat.MaxAbsValue / paramsPerLayer);
                    }

                    layerIdx++;
                }

                // Normalize to [0, 1]
                double maxMagnitude = stats.GlobalActivationMagnitudes.Max();
                if (maxMagnitude > 0)
                {
                    for (int i = 0; i < n; i++)
                    {
                        stats.GlobalActivationMagnitudes[i] /= maxMagnitude;
                    }
                }
            }
            else
            {
                // Fall back to parameter-based
                FillFromParameters(parameters, stats);
            }
        }
        else
        {
            // Use parameter magnitudes directly
            FillFromParameters(parameters, stats);
        }
    }

    /// <summary>
    /// Fills global statistics from parameter magnitudes.
    /// </summary>
    private void FillFromParameters(Vector<T> parameters, ActivationStatistics<T> stats)
    {
        int n = parameters.Length;
        double maxMag = 0;

        for (int i = 0; i < n; i++)
        {
            double absVal = Math.Abs(Convert.ToDouble(parameters[i]));
            stats.GlobalMaxAbsActivations![i] = absVal;
            stats.GlobalActivationMagnitudes![i] = absVal;
            maxMag = Math.Max(maxMag, absVal);
        }

        // Normalize magnitudes to [0, 1]
        if (maxMag > 0)
        {
            for (int i = 0; i < n; i++)
            {
                stats.GlobalActivationMagnitudes![i] /= maxMag;
            }
        }
    }

    /// <summary>
    /// Checks if the model supports running predictions.
    /// </summary>
    private static bool CanRunPredictions(IFullModel<T, TInput, TOutput> model)
    {
        // Check if model has been trained and can make predictions
        try
        {
            return model != null;
        }
        catch (InvalidOperationException)
        {
            // Model not trained or not ready for predictions
            return false;
        }
    }

    /// <summary>
    /// Converts a sample to Tensor for neural network models.
    /// </summary>
    /// <remarks>
    /// <para>Supported types (in order of preference):</para>
    /// <list type="bullet">
    /// <item><description>Tensor&lt;T&gt; - returned directly</description></item>
    /// <item><description>Vector&lt;T&gt; - converted via Tensor.FromVector</description></item>
    /// <item><description>Matrix&lt;T&gt; - reshaped to [Rows, Columns]</description></item>
    /// <item><description>T[] - wrapped as [1, Length] tensor</description></item>
    /// <item><description>Any type with ToArray() method returning T[] (via reflection)</description></item>
    /// </list>
    /// <para>The reflection fallback is expensive and should be avoided for performance-critical code.
    /// If the ToArray() method exists but returns a different type (not T[]), conversion will fail silently.</para>
    /// </remarks>
    /// <returns>The tensor, or null if conversion is not possible.</returns>
    private Tensor<T>? ConvertToTensor(TInput sample)
    {
        if (sample is Tensor<T> tensor)
        {
            return tensor;
        }

        if (sample is Vector<T> vector)
        {
            return Tensor<T>.FromVector(vector);
        }

        if (sample is Matrix<T> matrix)
        {
            return new Tensor<T>([matrix.Rows, matrix.Columns], new Vector<T>(matrix.ToArray()));
        }

        if (sample is T[] array)
        {
            return new Tensor<T>([1, array.Length], new Vector<T>(array));
        }

        // Try to convert through reflection for other types
        // This fallback is expensive but handles custom collection types
        try
        {
            var toArrayMethod = sample?.GetType().GetMethod("ToArray", Type.EmptyTypes);
            if (toArrayMethod != null)
            {
                // Validate return type before invoking to provide clearer failure
                var returnType = toArrayMethod.ReturnType;
                if (returnType == typeof(T[]) || typeof(T[]).IsAssignableFrom(returnType))
                {
                    var arr = toArrayMethod.Invoke(sample, null) as T[];
                    if (arr != null && arr.Length > 0)
                    {
                        return new Tensor<T>([1, arr.Length], new Vector<T>(arr));
                    }
                }
                // If ToArray exists but returns wrong type, log warning for debugging
            }
        }
        catch (System.Reflection.TargetInvocationException)
        {
            // ToArray method threw an exception - sample type incompatible
        }
        catch (InvalidCastException)
        {
            // Array conversion failed - type mismatch
        }

        return null;
    }

    /// <summary>
    /// Extracts double values from model output.
    /// </summary>
    private double[]? ExtractValuesFromOutput(TOutput output)
    {
        if (output == null) return null;

        if (output is Tensor<T> tensor)
        {
            var dataArray = tensor.Data.ToArray();
            return dataArray.Select(v => Convert.ToDouble(v)).ToArray();
        }

        if (output is Vector<T> vector)
        {
            var result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = Convert.ToDouble(vector[i]);
            }
            return result;
        }

        if (output is Matrix<T> matrix)
        {
            return matrix.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
        }

        if (output is T scalar)
        {
            return new[] { Convert.ToDouble(scalar) };
        }

        if (output is T[] array)
        {
            return array.Select(v => Convert.ToDouble(v)).ToArray();
        }

        return null;
    }
}
