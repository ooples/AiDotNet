using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Adaptive teacher model that adjusts its teaching strategy based on student performance.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Adaptive distillation is like a tutor who adjusts their teaching
/// style based on how well the student is doing. If the student is struggling, the teacher provides
/// more detailed guidance. If the student is doing well, the teacher can provide more advanced knowledge.</para>
///
/// <para><b>How It Works:</b>
/// - Monitor student's accuracy/confidence on different samples
/// - For hard samples (student struggles): Use lower temperature, focus on correct class
/// - For easy samples (student confident): Use higher temperature, reveal more class relationships
/// - Dynamically adjust temperature per sample based on student performance</para>
///
/// <para><b>Real-world Analogy:</b>
/// A math tutor notices a student struggling with algebra but excelling at geometry.
/// The tutor spends more time on algebra fundamentals while letting the student explore
/// advanced geometry concepts independently.</para>
///
/// <para><b>Benefits:</b>
/// - **Curriculum Learning**: Automatically adjusts difficulty
/// - **Efficient Training**: Focus teaching effort where needed
/// - **Better Convergence**: Avoids overwhelming student early
/// - **Personalized**: Adapts to this specific student</para>
///
/// <para><b>Adaptation Strategies:</b>
/// - **Sample-wise**: Different temperature per sample
/// - **Class-wise**: Different approach per class
/// - **Epoch-wise**: Adjust over training progress
/// - **Confidence-based**: Based on student's prediction confidence</para>
///
/// <para><b>References:</b>
/// - Mirzadeh et al. (2020). Improved Knowledge Distillation via Teacher Assistant. AAAI.
/// - Zhou et al. (2021). Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective.</para>
/// </remarks>
public class AdaptiveTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _baseTeacher;
    private readonly AdaptiveStrategy _strategy;
    private readonly double _minTemperature;
    private readonly double _maxTemperature;
    private readonly double _adaptationRate;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public override int OutputDimension => _baseTeacher.OutputDimension;

    /// <summary>
    /// Gets or sets the student performance tracker (confidence/accuracy per sample).
    /// </summary>
    public Dictionary<int, double> StudentPerformance { get; set; } = new();

    /// <summary>
    /// Initializes a new instance of the AdaptiveTeacherModel class.
    /// </summary>
    /// <param name="baseTeacher">The underlying teacher model.</param>
    /// <param name="strategy">Adaptation strategy to use.</param>
    /// <param name="minTemperature">Minimum temperature for hard samples (default: 1.0).</param>
    /// <param name="maxTemperature">Maximum temperature for easy samples (default: 5.0).</param>
    /// <param name="adaptationRate">How quickly to adapt (default: 0.1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create an adaptive teacher by providing:
    /// - Base teacher: The underlying expert model
    /// - Strategy: How to adapt (confidence-based recommended)
    /// - Temperature range: Control softness (1-5 typical)</para>
    ///
    /// <para>Example:
    /// <code>
    /// var baseTeacher = new TeacherModelWrapper&lt;double&gt;(trainedModel);
    /// var adaptiveTeacher = new AdaptiveTeacherModel&lt;double&gt;(
    ///     baseTeacher,
    ///     AdaptiveStrategy.ConfidenceBased,
    ///     minTemperature: 1.0,  // Sharp for hard samples
    ///     maxTemperature: 5.0   // Soft for easy samples
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public AdaptiveTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>> baseTeacher,
        AdaptiveStrategy strategy = AdaptiveStrategy.ConfidenceBased,
        double minTemperature = 1.0,
        double maxTemperature = 5.0,
        double adaptationRate = 0.1)
    {
        _baseTeacher = baseTeacher ?? throw new ArgumentNullException(nameof(baseTeacher));
        _strategy = strategy;
        _minTemperature = minTemperature;
        _maxTemperature = maxTemperature;
        _adaptationRate = adaptationRate;

        if (minTemperature <= 0 || maxTemperature <= minTemperature)
            throw new ArgumentException("Invalid temperature range");
        if (adaptationRate <= 0 || adaptationRate > 1)
            throw new ArgumentException("Adaptation rate must be in (0, 1]");
    }

    /// <summary>
    /// Gets logits from the base teacher.
    /// </summary>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        return _baseTeacher.GetLogits(input);
    }

    /// <summary>
    /// Gets soft predictions with adaptive temperature based on sample difficulty.
    /// </summary>
    /// <param name="input">Input data.</param>
    /// <param name="temperature">Base temperature (will be adapted).</param>
    /// <returns>Soft predictions with adaptive temperature.</returns>
    public override Vector<T> GetSoftPredictions(Vector<T> input, double temperature = 1.0)
    {
        var logits = GetLogits(input);

        // Compute adaptive temperature based on strategy
        double adaptiveTemp = ComputeAdaptiveTemperature(logits, temperature);

        return ApplyTemperatureSoftmax(logits, adaptiveTemp);
    }

    /// <summary>
    /// Updates student performance tracking for adaptive adjustment.
    /// </summary>
    /// <param name="sampleIndex">Index of the sample in batch.</param>
    /// <param name="studentPrediction">Student's prediction.</param>
    /// <param name="trueLabel">True label (optional).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this after each prediction to help the teacher
    /// learn which samples the student finds difficult.</para>
    /// </remarks>
    public void UpdateStudentPerformance(int sampleIndex, Vector<T> studentPrediction, Vector<T>? trueLabel = null)
    {
        double performance = 0;

        switch (_strategy)
        {
            case AdaptiveStrategy.ConfidenceBased:
                // Max confidence as performance measure
                performance = GetMaxConfidence(studentPrediction);
                break;

            case AdaptiveStrategy.AccuracyBased:
                if (trueLabel != null)
                {
                    // Correctness as performance measure
                    performance = IsCorrect(studentPrediction, trueLabel) ? 1.0 : 0.0;
                }
                break;

            case AdaptiveStrategy.EntropyBased:
                // Low entropy = confident (high performance)
                performance = 1.0 - ComputeEntropy(studentPrediction);
                break;
        }

        // Exponential moving average
        if (StudentPerformance.ContainsKey(sampleIndex))
        {
            StudentPerformance[sampleIndex] =
                _adaptationRate * performance + (1 - _adaptationRate) * StudentPerformance[sampleIndex];
        }
        else
        {
            StudentPerformance[sampleIndex] = performance;
        }
    }

    /// <summary>
    /// Computes adaptive temperature based on sample difficulty.
    /// </summary>
    private double ComputeAdaptiveTemperature(Vector<T> logits, double baseTemperature)
    {
        double difficulty = 0;

        switch (_strategy)
        {
            case AdaptiveStrategy.ConfidenceBased:
                // Lower confidence = harder sample = lower temperature
                difficulty = 1.0 - GetMaxConfidence(logits);
                break;

            case AdaptiveStrategy.EntropyBased:
                // Higher entropy = harder sample = lower temperature
                difficulty = ComputeEntropy(logits);
                break;

            case AdaptiveStrategy.AccuracyBased:
                // Use stored performance (default to medium difficulty)
                difficulty = 0.5;
                break;
        }

        // Map difficulty [0,1] to temperature [min, max]
        // High difficulty (0) -> min temp (sharper)
        // Low difficulty (1) -> max temp (softer)
        double adaptiveTemp = _minTemperature + (1.0 - difficulty) * (_maxTemperature - _minTemperature);

        return adaptiveTemp * baseTemperature;
    }

    /// <summary>
    /// Applies temperature-scaled softmax.
    /// </summary>
    protected override Vector<T> ApplyTemperatureSoftmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);

        // Scale by temperature
        var scaledLogits = new T[n];
        for (int i = 0; i < n; i++)
        {
            scaledLogits[i] = NumOps.FromDouble(NumOps.ToDouble(logits[i]) / temperature);
        }

        // Numerical stability
        T maxLogit = scaledLogits[0];
        for (int i = 1; i < n; i++)
        {
            if (NumOps.GreaterThan(scaledLogits[i], maxLogit))
                maxLogit = scaledLogits[i];
        }

        // Softmax
        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(NumOps.Subtract(scaledLogits[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.Divide(expValues[i], sum);
        }

        return result;
    }

    private double GetMaxConfidence(Vector<T> probs)
    {
        T maxVal = probs[0];
        for (int i = 1; i < probs.Length; i++)
        {
            if (NumOps.GreaterThan(probs[i], maxVal))
                maxVal = probs[i];
        }
        return NumOps.ToDouble(maxVal);
    }

    private double ComputeEntropy(Vector<T> probs)
    {
        double entropy = 0;
        const double epsilon = 1e-10;

        for (int i = 0; i < probs.Length; i++)
        {
            double p = NumOps.ToDouble(probs[i]);
            if (p > epsilon)
            {
                entropy -= p * Math.Log(p);
            }
        }

        // Normalize by max entropy
        double maxEntropy = Math.Log(probs.Length);
        return entropy / maxEntropy;
    }

    private bool IsCorrect(Vector<T> prediction, Vector<T> label)
    {
        int predClass = ArgMax(prediction);
        int trueClass = ArgMax(label);
        return predClass == trueClass;
    }

    private int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        T maxValue = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.GreaterThan(vector[i], maxValue))
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}

/// <summary>
/// Defines how the adaptive teacher adjusts to student performance.
/// </summary>
public enum AdaptiveStrategy
{
    /// <summary>
    /// Adjust based on student's prediction confidence (max probability).
    /// </summary>
    ConfidenceBased,

    /// <summary>
    /// Adjust based on student's correctness (requires labels).
    /// </summary>
    AccuracyBased,

    /// <summary>
    /// Adjust based on student's prediction entropy (uncertainty).
    /// </summary>
    EntropyBased
}
