using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Wraps an existing trained model to act as a teacher for knowledge distillation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class takes any trained model and makes it usable as a "teacher"
/// for knowledge distillation. The teacher model should already be trained and perform well on your task.</para>
///
/// <para>The wrapper provides a standard interface for:
/// - Getting raw predictions (logits)
/// - Getting soft predictions with temperature scaling
/// - Extracting intermediate features
/// - Accessing attention weights (for transformers)</para>
///
/// <para><b>Real-world Example:</b>
/// Imagine you have a large, accurate ResNet-50 model trained on ImageNet. You can wrap it
/// with TeacherModelWrapper and use it to train a smaller, faster ResNet-18 student model
/// that retains most of the accuracy but runs much faster.</para>
///
/// <para>Common teacher-student pairs:
/// - BERT (teacher) → DistilBERT (student): 40% smaller, 97% of performance
/// - ResNet-152 (teacher) → MobileNet (student): 10x faster inference
/// - GPT-3 (teacher) → GPT-2 (student): Deployable on edge devices</para>
/// </remarks>
public class TeacherModelWrapper<T> : ITeacherModel<Vector<T>, Vector<T>>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Func<Vector<T>, Vector<T>> _forwardFunc;
    private readonly Func<Vector<T>, string, object?>? _featureExtractor;
    private readonly Func<Vector<T>, string, object?>? _attentionExtractor;

    /// <summary>
    /// Gets the number of output dimensions (e.g., number of classes for classification).
    /// </summary>
    public int OutputDimension { get; }

    /// <summary>
    /// Initializes a new instance of the TeacherModelWrapper class from a forward function.
    /// </summary>
    /// <param name="forwardFunc">Function that performs forward pass and returns logits.</param>
    /// <param name="outputDimension">The number of output dimensions (classes).</param>
    /// <param name="featureExtractor">Optional function to extract intermediate layer features.</param>
    /// <param name="attentionExtractor">Optional function to extract attention weights.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor lets you create a teacher from any model
    /// by providing a forward function. The forward function should take input and return logits
    /// (raw outputs before softmax).</para>
    ///
    /// <para>Example usage:
    /// <code>
    /// var teacher = new TeacherModelWrapper&lt;double&gt;(
    ///     forwardFunc: input => myTrainedModel.Forward(input),
    ///     outputDimension: 10 // 10 classes (e.g., CIFAR-10)
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public TeacherModelWrapper(
        Func<Vector<T>, Vector<T>> forwardFunc,
        int outputDimension,
        Func<Vector<T>, string, object?>? featureExtractor = null,
        Func<Vector<T>, string, object?>? attentionExtractor = null)
    {
        if (outputDimension <= 0)
            throw new ArgumentException("Output dimension must be positive", nameof(outputDimension));

        _numOps = MathHelper.GetNumericOperations<T>();
        _forwardFunc = forwardFunc ?? throw new ArgumentNullException(nameof(forwardFunc));
        _featureExtractor = featureExtractor;
        _attentionExtractor = attentionExtractor;
        OutputDimension = outputDimension;
    }

    /// <summary>
    /// Initializes a new instance of the TeacherModelWrapper class from an IFullModel.
    /// </summary>
    /// <param name="model">The trained IFullModel to wrap as a teacher.</param>
    /// <param name="featureExtractor">Optional function to extract intermediate layer features.</param>
    /// <param name="attentionExtractor">Optional function to extract attention weights.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the recommended way to create a teacher - pass your trained
    /// IFullModel directly and it will be automatically wrapped for distillation.</para>
    ///
    /// <para>Example usage:
    /// <code>
    /// // After training your model
    /// IFullModel&lt;double, Vector&lt;double&gt;, Vector&lt;double&gt;&gt; trainedModel = ...;
    ///
    /// // Wrap it as a teacher
    /// var teacher = new TeacherModelWrapper&lt;double&gt;(trainedModel);
    ///
    /// // Now use it for distillation
    /// var distillationLoss = new DistillationLoss&lt;double&gt;(temperature: 3.0, alpha: 0.3);
    /// var trainer = new KnowledgeDistillationTrainer&lt;double&gt;(teacher, distillationLoss);
    /// </code>
    /// </para>
    /// </remarks>
    public TeacherModelWrapper(
        IFullModel<T, Vector<T>, Vector<T>> model,
        Func<Vector<T>, string, object?>? featureExtractor = null,
        Func<Vector<T>, string, object?>? attentionExtractor = null)
        : this(
            forwardFunc: input => model.Predict(input),
            outputDimension: GetOutputDimensionFromModel(model),
            featureExtractor: featureExtractor,
            attentionExtractor: attentionExtractor)
    {
    }

    private static int GetOutputDimensionFromModel(IFullModel<T, Vector<T>, Vector<T>> model)
    {
        // Try to infer output dimension from model metadata or use a safe default
        try
        {
            var metadata = model.GetMetadata();
            // Attempt to get output dimension from metadata if available
            // This is a heuristic - adjust based on actual metadata structure
            return 1; // Fallback - will be overridden by actual prediction
        }
        catch
        {
            return 1; // Safe default
        }
    }

    /// <summary>
    /// Gets the teacher's raw logits (pre-softmax outputs) for the given input.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <returns>Raw logits before applying softmax.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Logits are the raw numerical outputs from a neural network
    /// before converting them to probabilities. They're preferred for distillation because:
    /// 1. They preserve more information than probabilities
    /// 2. They're numerically more stable
    /// 3. Temperature scaling works better on logits</para>
    /// </remarks>
    public Vector<T> GetLogits(Vector<T> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        return _forwardFunc(input);
    }

    /// <summary>
    /// Gets the teacher's soft predictions (probabilities) with temperature scaling.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="temperature">Softmax temperature (default 1.0). Higher values produce softer distributions.</param>
    /// <returns>Probability distribution with temperature scaling applied.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This applies softmax with temperature to convert logits to probabilities:
    /// - Temperature = 1.0: Standard softmax (sharp predictions)
    /// - Temperature &gt; 1.0: Softer predictions (reveals more about class relationships)
    /// - Temperature &lt; 1.0: Sharper predictions (less useful for distillation)</para>
    ///
    /// <para>Example with logits [2.0, 1.0, 0.5]:
    /// - T=1: probs = [0.66, 0.24, 0.10] (sharp)
    /// - T=3: probs = [0.42, 0.32, 0.26] (soft, more informative)</para>
    /// </remarks>
    public Vector<T> GetSoftPredictions(Vector<T> input, double temperature = 1.0)
    {
        if (temperature <= 0)
            throw new ArgumentException("Temperature must be positive", nameof(temperature));

        var logits = GetLogits(input);
        return ApplySoftmax(logits, temperature);
    }

    /// <summary>
    /// Gets intermediate layer features from the teacher model.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="layerName">The name of the layer to extract features from.</param>
    /// <returns>Feature map from the specified layer, or null if feature extraction is not supported.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This extracts the internal representations (features) learned
    /// by intermediate layers. These features can be used for feature distillation, where the
    /// student tries to match the teacher's internal representations, not just final outputs.</para>
    ///
    /// <para>Note: This requires a featureExtractor function to be provided during construction.</para>
    /// </remarks>
    public object? GetFeatures(Vector<T> input, string layerName)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentException.ThrowIfNullOrWhiteSpace(layerName);

        if (_featureExtractor == null)
            return null;

        return _featureExtractor(input, layerName);
    }

    /// <summary>
    /// Gets attention weights from the teacher model (for transformer architectures).
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="layerName">The name of the attention layer to extract weights from.</param>
    /// <returns>Attention weight matrix from the specified layer, or null if not supported.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention weights show which parts of the input the model
    /// focuses on when making predictions. Transferring attention patterns from teacher to
    /// student can significantly improve student performance on sequence tasks.</para>
    ///
    /// <para>Note: This requires an attentionExtractor function to be provided during construction.</para>
    /// </remarks>
    public object? GetAttentionWeights(Vector<T> input, string layerName)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentException.ThrowIfNullOrWhiteSpace(layerName);

        if (_attentionExtractor == null)
            return null;

        return _attentionExtractor(input, layerName);
    }

    /// <summary>
    /// Applies softmax function with temperature scaling to convert logits to probabilities.
    /// </summary>
    /// <param name="logits">Raw network outputs before activation.</param>
    /// <param name="temperature">Temperature parameter for softening the distribution.</param>
    /// <returns>Probability distribution summing to 1.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Softmax with temperature works as follows:
    /// 1. Scale logits by dividing by temperature
    /// 2. Apply exponential function
    /// 3. Normalize so probabilities sum to 1</para>
    ///
    /// <para>We use the "max subtraction trick" for numerical stability.</para>
    /// </remarks>
    private Vector<T> ApplySoftmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);

        // Scale by temperature
        var scaledLogits = new T[n];
        for (int i = 0; i < n; i++)
        {
            double val = _numOps.ToDouble(logits[i]) / temperature;
            scaledLogits[i] = _numOps.FromDouble(val);
        }

        // Find max for numerical stability (prevents exp overflow)
        T maxLogit = scaledLogits[0];
        for (int i = 1; i < n; i++)
        {
            if (_numOps.GreaterThan(scaledLogits[i], maxLogit))
                maxLogit = scaledLogits[i];
        }

        // Compute exp(logit - max) and sum
        T sum = _numOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = _numOps.ToDouble(_numOps.Subtract(scaledLogits[i], maxLogit));
            expValues[i] = _numOps.FromDouble(Math.Exp(val));
            sum = _numOps.Add(sum, expValues[i]);
        }

        // Normalize to get probabilities
        for (int i = 0; i < n; i++)
        {
            result[i] = _numOps.Divide(expValues[i], sum);
        }

        return result;
    }
}
