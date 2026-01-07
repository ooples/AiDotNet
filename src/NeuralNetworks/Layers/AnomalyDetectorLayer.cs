using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that detects anomalies by comparing predictions with actual inputs.
/// </summary>
/// <remarks>
/// <para>
/// The AnomalyDetectionLayer compares the predicted state with the actual state to calculate an anomaly score.
/// This score represents how unexpected or surprising the current input is, given the model's predictions.
/// Higher scores indicate more anomalous (unexpected) inputs.
/// </para>
/// <para><b>For Beginners:</b> This layer identifies patterns that don't match what the network expected.
/// 
/// Think of anomaly detection like this:
/// - The network learns what "normal" looks like from the data
/// - This layer compares new inputs to what the network expected to see
/// - If the actual input is very different from the prediction, it's flagged as anomalous
/// - The output is an "anomaly score" between 0 and 1 (higher means more unusual)
/// 
/// For example, in monitoring network traffic, the system might learn normal patterns
/// and then use this layer to alert when unusual activity is detected that might
/// indicate a security breach.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class AnomalyDetectorLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The threshold for determining anomalous inputs based on the anomaly score.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the threshold value for classifying inputs as anomalous. If an anomaly score exceeds this
    /// threshold, the input is considered anomalous.
    /// </para>
    /// <para><b>For Beginners:</b> This is the cutoff point for what's considered "unusual enough" to flag.
    /// 
    /// The anomaly threshold works like this:
    /// - It's a value between 0 and 1
    /// - If the anomaly score is above this threshold, the input is flagged as anomalous
    /// - Lower values make the detector more sensitive (more alerts)
    /// - Higher values make it more selective (fewer alerts)
    /// 
    /// Setting this value requires balance:
    /// - Too low, and you'll get too many false alarms
    /// - Too high, and you might miss real anomalies
    /// </para>
    /// </remarks>
    private readonly double _anomalyThreshold;

    /// <summary>
    /// The history of recent anomaly scores for adaptive thresholding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores a history of recent anomaly scores, which can be used for adaptive thresholding
    /// or trend analysis of anomaly patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This keeps track of recent anomaly scores to establish a baseline.
    /// 
    /// The anomaly history:
    /// - Stores the most recent anomaly scores (like the last 100)
    /// - Helps distinguish "new unusual" from "consistently unusual"
    /// - Can be used to detect trends in anomaly patterns
    /// 
    /// This is important because "normal" might change over time, and what's
    /// anomalous should be judged relative to recent patterns.
    /// </para>
    /// </remarks>
    private Queue<double> _anomalyHistory;

    /// <summary>
    /// The maximum number of anomaly scores to keep in history.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies how many recent anomaly scores to maintain in the history queue.
    /// A larger history provides more context for adaptive thresholding but requires more memory.
    /// </para>
    /// <para><b>For Beginners:</b> This sets how far back the layer remembers anomaly scores.
    /// 
    /// The history capacity:
    /// - Defines how many recent scores are stored
    /// - Larger values provide more stable baseline detection
    /// - Smaller values make the layer adapt more quickly to changing conditions
    /// 
    /// For example, with a capacity of 100, the layer looks at the last 100 inputs
    /// to determine what's "normal" versus "unusual".
    /// </para>
    /// </remarks>
    private readonly int _historyCapacity;

    /// <summary>
    /// The smoothing factor for exponential moving average of anomaly scores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the alpha parameter for calculating an exponential moving average of anomaly scores.
    /// This helps to smooth out noise in the anomaly detection process.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the system responds to changes in anomaly patterns.
    /// 
    /// The smoothing factor works like this:
    /// - It's a value between 0 and 1
    /// - Lower values create more smoothing (slower response to changes)
    /// - Higher values create less smoothing (faster response to changes)
    /// 
    /// For example, with a value of 0.1, the current anomaly score only contributes
    /// 10% to the smoothed score, while the previous smoothed value contributes 90%.
    /// </para>
    /// </remarks>
    private readonly double _smoothingFactor;

    /// <summary>
    /// The current smoothed anomaly score.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the current smoothed anomaly score, which is updated with each new input
    /// using an exponential moving average.
    /// </para>
    /// <para><b>For Beginners:</b> This is the current "averaged" anomaly score that reduces noise.
    /// 
    /// The smoothed anomaly score:
    /// - Combines recent anomaly scores using a weighted average
    /// - Reduces the impact of brief spikes or drops
    /// - Provides a more stable measure of how unusual the current input is
    /// 
    /// This helps prevent false alarms from temporary fluctuations in the data.
    /// </para>
    /// </remarks>
    private double _smoothedAnomalyScore;

    /// <summary>
    /// Stores the most recent input shape for any-rank tensor support.
    /// </summary>
    private int[]? _lastInputShape;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>false</c> for this layer, as it doesn't have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the anomaly detection layer has trainable parameters.
    /// Since this layer simply calculates an anomaly score based on the input and doesn't have
    /// weights or biases to update during training, it returns false.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer needs training.
    /// 
    /// A value of false means:
    /// - The layer doesn't have weights or biases that need to be learned
    /// - It performs calculations using fixed algorithms rather than learned parameters
    /// - It operates based on the statistics of the data it sees
    /// 
    /// This layer works automatically without needing a training phase.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="AnomalyDetectorLayer{T}"/> class.
    /// </summary>
    /// <param name="inputSize">The size of the input vector.</param>
    /// <param name="anomalyThreshold">The threshold for determining anomalous inputs.</param>
    /// <param name="historyCapacity">The maximum number of anomaly scores to keep in history. Default is 100.</param>
    /// <param name="smoothingFactor">The smoothing factor for the exponential moving average. Default is 0.1.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an anomaly detection layer with the specified parameters.
    /// The layer will calculate anomaly scores by comparing predicted states with actual states.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new anomaly detection layer with your settings.
    /// 
    /// The parameters you provide determine:
    /// - inputSize: How many values the layer expects in its input vector
    /// - anomalyThreshold: The cutoff point for flagging anomalies (0-1)
    /// - historyCapacity: How many recent scores to remember
    /// - smoothingFactor: How quickly to respond to changes (0-1)
    /// 
    /// These settings let you customize how sensitive the anomaly detection is
    /// and how it adapts to changing patterns over time.
    /// </para>
    /// </remarks>
    public AnomalyDetectorLayer(
        int inputSize,
        double anomalyThreshold,
        int historyCapacity = 100,
        double smoothingFactor = 0.1,
        IEngine? engine = null)
        : base([inputSize], [1])
    {
        _anomalyThreshold = anomalyThreshold;
        _historyCapacity = historyCapacity;
        _smoothingFactor = smoothingFactor;
        _anomalyHistory = new Queue<double>(_historyCapacity);
        _smoothedAnomalyScore = 0.0;
    }

    /// <summary>
    /// Performs the forward pass of the anomaly detection layer.
    /// </summary>
    /// <param name="input">The input tensor containing both predicted and actual states.</param>
    /// <returns>A tensor containing the anomaly score.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the anomaly detection layer.
    /// It calculates an anomaly score by comparing the predicted states with the actual states.
    /// The method assumes that the first half of the input represents actual states and the
    /// second half represents predicted states.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how unusual the current input is.
    /// 
    /// The forward pass:
    /// - Takes an input that contains both actual values and predicted values
    /// - Compares them to see how different they are
    /// - Calculates an anomaly score based on this difference
    /// - Updates the smoothed score and history
    /// - Returns the anomaly score and whether it exceeds the threshold
    /// 
    /// The output is a tensor with just one value: the anomaly score between 0 and 1.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInputShape = input.Shape;
        int rank = input.Shape.Length;
        int featureSize = input.Shape[^1];
        if (featureSize % 2 != 0)
            throw new ArgumentException("Input feature dimension must be even (actual + predicted).", nameof(input));

        int halfSize = featureSize / 2;
        int[] startActual = new int[rank];
        int[] lengthActual = (int[])input.Shape.Clone();
        lengthActual[rank - 1] = halfSize;

        int[] startPred = new int[rank];
        startPred[rank - 1] = halfSize;
        int[] lengthPred = (int[])input.Shape.Clone();
        lengthPred[rank - 1] = halfSize;

        var actual = Engine.TensorSlice(input, startActual, lengthActual);
        var predicted = Engine.TensorSlice(input, startPred, lengthPred);

        var anomalyScores = CalculateAnomalyScores(actual, predicted);

        double meanScore = 0.0;
        for (int i = 0; i < anomalyScores.Length; i++)
        {
            meanScore += NumOps.ToDouble(anomalyScores.GetFlat(i));
        }
        if (anomalyScores.Length > 0)
        {
            meanScore /= anomalyScores.Length;
        }

        _smoothedAnomalyScore = (_smoothingFactor * meanScore) + ((1 - _smoothingFactor) * _smoothedAnomalyScore);
        UpdateAnomalyHistory(meanScore);

        var output = new Tensor<T>([1]);
        output[0] = NumOps.FromDouble(meanScore);
        return output;
    }

    /// <summary>
    /// Performs the forward pass using GPU acceleration.
    /// </summary>
    /// <param name="inputs">The input GPU tensor containing both predicted and actual states.</param>
    /// <returns>A GPU tensor containing the anomaly score.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("No GPU backend available.");

        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var input = inputs[0];
        var shape = input.Shape;
        _lastInputShape = shape;

        int rank = shape.Length;
        int featureSize = shape[^1];
        if (featureSize % 2 != 0)
            throw new ArgumentException("Input feature dimension must be even (actual + predicted).", nameof(inputs));

        int halfSize = featureSize / 2;

        // Download data for processing (this layer has stateful operations that need CPU-side updates)
        var inputData = backend.DownloadBuffer(input.Buffer);

        // Calculate batch dimensions
        int batchSize = 1;
        for (int i = 0; i < rank - 1; i++)
            batchSize *= shape[i];

        // Calculate anomaly scores for each sample
        double totalScore = 0.0;
        int count = 0;

        for (int b = 0; b < batchSize; b++)
        {
            int baseIdx = b * featureSize;
            double mismatchCount = 0;
            double activeCount = 0;

            for (int i = 0; i < halfSize; i++)
            {
                float actual = inputData[baseIdx + i];
                float predicted = inputData[baseIdx + halfSize + i];

                // Check if either value is "active" (close to 1)
                bool actualActive = Math.Abs(actual - 1.0f) < 0.5f;
                bool predictedActive = Math.Abs(predicted - 1.0f) < 0.5f;

                if (actualActive || predictedActive)
                {
                    activeCount++;
                    // Check if values mismatch
                    if (Math.Abs(actual - predicted) > 0.5f)
                        mismatchCount++;
                }
            }

            double score = activeCount > 0 ? mismatchCount / activeCount : 0.0;
            totalScore += score;
            count++;
        }

        double meanScore = count > 0 ? totalScore / count : 0.0;

        // Update stateful elements (CPU-side)
        _smoothedAnomalyScore = (_smoothingFactor * meanScore) + ((1 - _smoothingFactor) * _smoothedAnomalyScore);
        UpdateAnomalyHistory(meanScore);

        // Create output tensor with single anomaly score
        var outputData = new float[] { (float)meanScore };
        var outputBuffer = backend.AllocateBuffer(outputData);
        var outputShape = new int[] { 1 };

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }


    /// <summary>
    /// Calculates the anomaly score based on the difference between actual and predicted states.
    /// </summary>
    /// <param name="actual">The actual state tensor.</param>
    /// <param name="predicted">The predicted state tensor.</param>
    /// <returns>The anomaly score as a value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates an anomaly score by comparing the actual state with the predicted state.
    /// The score is computed as the ratio of mismatched bits to the total number of bits, resulting
    /// in a value between 0 (perfect match) and 1 (complete mismatch).
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how different the actual input is from what was expected.
    /// 
    /// The anomaly score calculation:
    /// - Counts how many values are different between actual and predicted
    /// - Divides by the total number of values to get a percentage
    /// - Returns a value between 0 and 1
    /// 
    /// A score of 0 means perfect prediction (not anomalous at all),
    /// while 1 means completely wrong prediction (highly anomalous).
    /// </para>
    /// </remarks>
    private Tensor<T> CalculateAnomalyScores(Tensor<T> actual, Tensor<T> predicted)
    {
        var isActiveActual = Engine.TensorEquals(actual, NumOps.One);
        var isActivePredicted = Engine.TensorEquals(predicted, NumOps.One);
        var anyActive = Engine.TensorMax(isActiveActual, isActivePredicted);

        int rank = anyActive.Shape.Length;
        var axes = new[] { rank - 1 };

        var totalActiveSum = Engine.ReduceSum(anyActive, axes, keepDims: true);

        var notEqual = Engine.TensorNotEquals(actual, predicted);
        var mismatchAndActive = Engine.TensorMultiply(notEqual, anyActive);
        var mismatchSum = Engine.ReduceSum(mismatchAndActive, axes, keepDims: true);

        var scores = new Tensor<T>(totalActiveSum.Shape);
        for (int i = 0; i < totalActiveSum.Length; i++)
        {
            double totalCount = NumOps.ToDouble(totalActiveSum.GetFlat(i));
            double mismatchCount = NumOps.ToDouble(mismatchSum.GetFlat(i));
            double score = totalCount == 0.0 ? 0.0 : mismatchCount / totalCount;
            scores.SetFlat(i, NumOps.FromDouble(score));
        }

        return scores;
    }


    /// <summary>
    /// Updates the history of anomaly scores.
    /// </summary>
    /// <param name="anomalyScore">The current anomaly score.</param>
    /// <remarks>
    /// <para>
    /// This method adds the current anomaly score to the history queue and removes the oldest score
    /// if the queue exceeds its capacity.
    /// </para>
    /// <para><b>For Beginners:</b> This method keeps track of recent anomaly scores.
    /// 
    /// The history update:
    /// - Adds the new anomaly score to the history
    /// - Removes the oldest score if the history is full
    /// - Maintains a sliding window of recent scores
    /// 
    /// This history helps the layer understand what's "normal" by looking at
    /// patterns of anomaly scores over time.
    /// </para>
    /// </remarks>
    private void UpdateAnomalyHistory(double anomalyScore)
    {
        _anomalyHistory.Enqueue(anomalyScore);

        // Remove oldest score if we exceed capacity
        while (_anomalyHistory.Count > _historyCapacity)
        {
            _anomalyHistory.Dequeue();
        }
    }

    /// <summary>
    /// Determines if the current input is anomalous based on the anomaly score.
    /// </summary>
    /// <returns>True if the input is anomalous; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method determines whether the current input should be considered anomalous based on
    /// the smoothed anomaly score and the anomaly threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides if the current input is unusual enough to flag.
    /// 
    /// The anomaly decision:
    /// - Compares the smoothed anomaly score to the threshold
    /// - Returns true if the score is above the threshold
    /// - Returns false if the score is below the threshold
    /// 
    /// This binary decision can be used to trigger alerts or special handling
    /// for unusual inputs that might indicate problems or interesting events.
    /// </para>
    /// </remarks>
    public bool IsAnomaly()
    {
        return _smoothedAnomalyScore > _anomalyThreshold;
    }

    /// <summary>
    /// Gets the current anomaly score.
    /// </summary>
    /// <returns>The current smoothed anomaly score.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the current smoothed anomaly score, which represents how unexpected
    /// or surprising the most recent input was.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you how unusual the current input is.
    /// 
    /// The anomaly score:
    /// - Is a value between 0 and 1
    /// - Higher values mean more unusual inputs
    /// - Is smoothed to reduce the impact of momentary spikes
    /// 
    /// This score can be used for monitoring, visualization, or custom anomaly
    /// detection logic beyond simple thresholding.
    /// </para>
    /// </remarks>
    public double GetAnomalyScore()
    {
        return _smoothedAnomalyScore;
    }

    /// <summary>
    /// Gets the statistical properties of recent anomaly scores.
    /// </summary>
    /// <returns>A dictionary containing statistical properties (mean, standard deviation, min, max).</returns>
    /// <remarks>
    /// <para>
    /// This method calculates and returns statistical properties of the recent anomaly scores stored in the history.
    /// These statistics can be useful for adaptive thresholding or monitoring trends in anomaly patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides statistics about recent anomaly scores.
    /// 
    /// The statistics include:
    /// - Mean (average) of recent anomaly scores
    /// - Standard deviation (measure of variability)
    /// - Minimum and maximum scores
    /// - Current score and whether it's an anomaly
    /// 
    /// These statistics help understand the pattern of anomalies over time,
    /// which can be useful for adjusting thresholds or analyzing trends.
    /// </para>
    /// </remarks>
    public Dictionary<string, double> GetAnomalyStatistics()
    {
        if (_anomalyHistory.Count == 0)
        {
            return new Dictionary<string, double>
            {
                { "mean", 0.0 },
                { "stdDev", 0.0 },
                { "min", 0.0 },
                { "max", 0.0 },
                { "current", _smoothedAnomalyScore },
                { "isAnomaly", IsAnomaly() ? 1.0 : 0.0 }
            };
        }

        // Calculate statistics
        double sum = 0.0;
        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (var score in _anomalyHistory)
        {
            sum += score;
            min = Math.Min(min, score);
            max = Math.Max(max, score);
        }

        double mean = sum / _anomalyHistory.Count;

        // Calculate standard deviation
        double sumSquaredDiff = 0.0;
        foreach (var score in _anomalyHistory)
        {
            double diff = score - mean;
            sumSquaredDiff += diff * diff;
        }

        double variance = sumSquaredDiff / _anomalyHistory.Count;
        double stdDev = Math.Sqrt(variance);

        return new Dictionary<string, double>
        {
            { "mean", mean },
            { "stdDev", stdDev },
            { "min", min },
            { "max", max },
            { "current", _smoothedAnomalyScore },
            { "isAnomaly", IsAnomaly() ? 1.0 : 0.0 }
        };
    }

    /// <summary>
    /// Performs the backward pass of the anomaly detection layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the anomaly detection layer, which is used during training
    /// to propagate error gradients back through the network. Since the anomaly detection layer doesn't have
    /// trainable parameters, it simply passes the gradient through to the previous layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method passes error information backward during training.
    ///
    /// The backward pass:
    /// - Takes an error gradient from the next layer
    /// - Propagates it back to the previous layer
    /// - Doesn't modify any parameters since this layer doesn't learn
    ///
    /// This method exists to maintain compatibility with the neural network
    /// backpropagation mechanism, but it doesn't do much in this layer
    /// since there are no weights to adjust.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        // Since this layer doesn't have trainable parameters, we just propagate the gradient
        // back to the input. For anomaly detection, this is primarily a pass-through operation.

        return CreateZeroInputGradient();
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. Since this layer
    /// has no trainable parameters and serves as a monitoring layer, it returns zero gradients.
    /// This matches the manual implementation behavior.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        return CreateZeroInputGradient();
    }

    private Tensor<T> CreateZeroInputGradient()
    {
        if (_lastInputShape == null)
            throw new InvalidOperationException("Cannot run backward before forward.");

        var zeroGradient = new Tensor<T>(_lastInputShape);
        zeroGradient.Fill(NumOps.Zero);
        return zeroGradient;
    }

    /// <summary>
    /// Updates the parameters of the layer.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is empty in the current implementation as the layer does not have trainable parameters
    /// updated through gradient descent.
    /// </para>
    /// <para><b>For Beginners:</b> This method is included for compatibility but doesn't do anything in this layer.
    /// 
    /// The reason this method is empty:
    /// - This layer doesn't have weights or biases to update
    /// - It performs calculations based on fixed formulas rather than learned parameters
    /// - This method is included only to satisfy the requirements of the LayerBase class
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all parameters of the layer as a single vector.
    /// </summary>
    /// <returns>An empty vector as this layer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector since the anomaly detection layer doesn't have trainable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list since this layer doesn't learn parameters.
    /// 
    /// Since this layer:
    /// - Doesn't have weights or biases
    /// - Uses fixed formulas rather than learned parameters
    /// - Doesn't require training in the traditional sense
    /// 
    /// The method returns an empty vector to maintain compatibility with the layer interface.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // This layer has no trainable parameters
        return new Vector<T>(0);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the anomaly detection layer by clearing the anomaly history
    /// and resetting the smoothed anomaly score to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The history of anomaly scores is cleared
    /// - The smoothed anomaly score is set to zero
    /// - The layer forgets all past anomaly patterns
    /// 
    /// This is useful when:
    /// - Processing a new, unrelated sequence
    /// - Adapting to a significant change in data patterns
    /// - Testing the layer with fresh inputs
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear anomaly history
        _anomalyHistory.Clear();

        // Reset smoothed anomaly score
        _smoothedAnomalyScore = 0.0;
        _lastInputShape = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count < 2)
            throw new ArgumentException("AnomalyDetector requires two inputs: input and reconstruction.", nameof(inputNodes));

        // AnomalyDetectorLayer JIT computes anomaly scores from reconstruction error:
        // anomaly_score = mean((input - reconstruction)^2)
        // This is differentiable and enables training of anomaly detection models.

        var input = inputNodes[0];
        var reconstruction = inputNodes[1];

        // Compute anomaly score as mean squared error
        var anomalyScore = TensorOperations<T>.AnomalyScore(input, reconstruction);

        // Apply activation
        var output = ApplyActivationToGraph(anomalyScore);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c>. AnomalyDetector uses differentiable reconstruction error.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for AnomalyDetector computes the anomaly score as the
    /// reconstruction error (mean squared error between input and reconstruction).
    /// This enables training of anomaly detection models with gradient descent.
    /// The stateful historical tracking is not used in JIT mode.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

}
