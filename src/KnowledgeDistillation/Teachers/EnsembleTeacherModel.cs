using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Ensemble teacher model that combines predictions from multiple teacher models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Ensemble learning combines multiple models to create a stronger,
/// more robust teacher. The intuition is similar to seeking advice from multiple experts rather
/// than relying on a single expert.</para>
///
/// <para><b>Benefits of Ensemble Teachers:</b>
/// - **Higher Accuracy**: Ensemble outperforms individual models
/// - **Better Calibration**: Averaging reduces overconfidence
/// - **Robustness**: Less sensitive to individual model biases
/// - **Knowledge Diversity**: Student learns from complementary perspectives</para>
///
/// <para><b>Common Ensemble Strategies:</b>
/// - **Uniform Average**: Equal weight to all teachers (default)
/// - **Weighted Average**: More weight to better-performing teachers
/// - **Voting**: For classification, majority vote
/// - **Stacking**: Meta-model combines predictions</para>
///
/// <para><b>Real-world Analogy:</b>
/// Imagine learning to play chess from multiple grandmasters. Each has different playing styles
/// and strategies. By learning from all of them, you develop a more well-rounded understanding
/// of the game than you would from just one teacher.</para>
///
/// <para><b>Practical Example:</b>
/// Train 3-5 models with different:
/// - Initializations (different random seeds)
/// - Architectures (CNN, ResNet, Transformer)
/// - Hyperparameters (learning rates, depths)
/// Combine them to create a powerful ensemble teacher.</para>
///
/// <para><b>References:</b>
/// - You et al. (2017). Learning from Multiple Teacher Networks. KDD.
/// - Fukuda et al. (2017). Efficient Knowledge Distillation from an Ensemble of Teachers.</para>
/// </remarks>
public class EnsembleTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>>[] _teachers;
    private readonly double[]? _weights;
    private readonly EnsembleAggregationMode _aggregationMode;

    /// <summary>
    /// Gets the number of teachers in the ensemble.
    /// </summary>
    public int TeacherCount => _teachers.Length;

    /// <summary>
    /// Gets the output dimension (same for all teachers).
    /// </summary>
    public override int OutputDimension => _teachers[0].OutputDimension;

    /// <summary>
    /// Initializes a new instance of the EnsembleTeacherModel class.
    /// </summary>
    /// <param name="teachers">Array of teacher models to ensemble.</param>
    /// <param name="weights">Optional weights for each teacher (default: uniform). Must sum to 1.0.</param>
    /// <param name="aggregationMode">How to combine teacher predictions (default: WeightedAverage).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create an ensemble by providing multiple trained teacher models.
    /// If weights are not specified, all teachers contribute equally.</para>
    ///
    /// <para>Example:
    /// <code>
    /// var teacher1 = new TeacherModelWrapper&lt;double&gt;(model1);
    /// var teacher2 = new TeacherModelWrapper&lt;double&gt;(model2);
    /// var teacher3 = new TeacherModelWrapper&lt;double&gt;(model3);
    ///
    /// // Uniform ensemble (equal weights)
    /// var ensemble = new EnsembleTeacherModel&lt;double&gt;(
    ///     new[] { teacher1, teacher2, teacher3 }
    /// );
    ///
    /// // Weighted ensemble (based on validation accuracy)
    /// var ensemble2 = new EnsembleTeacherModel&lt;double&gt;(
    ///     teachers: new[] { teacher1, teacher2, teacher3 },
    ///     weights: new[] { 0.5, 0.3, 0.2 }  // Best model gets 50% weight
    /// );
    /// </code>
    /// </para>
    ///
    /// <para><b>Choosing Weights:</b>
    /// - **Uniform**: Use when teachers perform similarly
    /// - **Validation-based**: Weight by validation accuracy
    /// - **Confidence-based**: Weight by prediction confidence
    /// - **Diversity-based**: Weight to maximize diversity</para>
    /// </remarks>
    public EnsembleTeacherModel(
        ITeacherModel<Vector<T>, Vector<T>>[] teachers,
        double[]? weights = null,
        EnsembleAggregationMode aggregationMode = EnsembleAggregationMode.WeightedAverage)
    {
        if (teachers == null || teachers.Length == 0)
            throw new ArgumentException("At least one teacher must be provided", nameof(teachers));

        // Validate all teachers have the same output dimension
        int expectedDim = teachers[0].OutputDimension;
        for (int i = 1; i < teachers.Length; i++)
        {
            if (teachers[i].OutputDimension != expectedDim)
                throw new ArgumentException(
                    $"All teachers must have the same output dimension. Teacher 0: {expectedDim}, Teacher {i}: {teachers[i].OutputDimension}");
        }

        _teachers = teachers;
        _aggregationMode = aggregationMode;

        // Validate or create weights
        if (weights != null)
        {
            if (weights.Length != teachers.Length)
                throw new ArgumentException("Number of weights must match number of teachers", nameof(weights));

            double sum = weights.Sum();
            if (Math.Abs(sum - 1.0) > 1e-6)
                throw new ArgumentException($"Weights must sum to 1.0, got {sum}", nameof(weights));

            foreach (var weight in weights)
            {
                if (weight < 0 || weight > 1)
                    throw new ArgumentException("Weights must be between 0 and 1", nameof(weights));
            }

            _weights = weights;
        }
        else
        {
            // Uniform weights
            double uniformWeight = 1.0 / teachers.Length;
            _weights = Enumerable.Repeat(uniformWeight, teachers.Length).ToArray();
        }
    }

    /// <summary>
    /// Gets ensemble logits by combining predictions from all teachers.
    /// </summary>
    /// <param name="input">Input data.</param>
    /// <returns>Ensemble logits.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This combines logits from all teachers according to the
    /// aggregation mode (usually weighted average).</para>
    ///
    /// <para><b>Architecture Note:</b> Returns raw ensemble logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        // Get predictions from all teachers
        var teacherLogits = new Vector<T>[_teachers.Length];
        for (int i = 0; i < _teachers.Length; i++)
        {
            teacherLogits[i] = _teachers[i].GetLogits(input);
        }

        return AggregateLogits(teacherLogits);
    }

    /// <summary>
    /// Aggregates logits from multiple teachers according to the aggregation mode.
    /// </summary>
    private Vector<T> AggregateLogits(Vector<T>[] teacherLogits)
    {
        int n = teacherLogits[0].Length;
        var result = new Vector<T>(n);

        switch (_aggregationMode)
        {
            case EnsembleAggregationMode.WeightedAverage:
                // Weighted average of logits
                for (int i = 0; i < n; i++)
                {
                    T sum = NumOps.Zero;
                    for (int t = 0; t < _teachers.Length; t++)
                    {
                        var weighted = NumOps.Multiply(
                            teacherLogits[t][i],
                            NumOps.FromDouble(_weights![t]));
                        sum = NumOps.Add(sum, weighted);
                    }
                    result[i] = sum;
                }
                break;

            case EnsembleAggregationMode.GeometricMean:
                // Weighted geometric mean of probabilities (not raw logits)
                // Step 1: Convert each teacher's logits to probabilities via softmax
                var teacherProbs = new Vector<T>[_teachers.Length];
                for (int t = 0; t < _teachers.Length; t++)
                {
                    teacherProbs[t] = Softmax(teacherLogits[t]);
                }

                // Step 2: Compute weighted geometric mean of probabilities
                // Using log space for numerical stability: exp(sum(wi * log(pi)))
                for (int i = 0; i < n; i++)
                {
                    double logSum = 0;
                    for (int t = 0; t < _teachers.Length; t++)
                    {
                        double prob = Convert.ToDouble(teacherProbs[t][i]);
                        // Add small epsilon to avoid log(0)
                        logSum += _weights![t] * Math.Log(prob + 1e-10);
                    }
                    result[i] = NumOps.FromDouble(Math.Exp(logSum));
                }
                break;

            case EnsembleAggregationMode.Maximum:
                // Element-wise maximum
                for (int i = 0; i < n; i++)
                {
                    T maxVal = teacherLogits[0][i];
                    for (int t = 1; t < _teachers.Length; t++)
                    {
                        if (NumOps.GreaterThan(teacherLogits[t][i], maxVal))
                            maxVal = teacherLogits[t][i];
                    }
                    result[i] = maxVal;
                }
                break;

            case EnsembleAggregationMode.Median:
                // Element-wise median
                for (int i = 0; i < n; i++)
                {
                    var values = new double[_teachers.Length];
                    for (int t = 0; t < _teachers.Length; t++)
                    {
                        values[t] = Convert.ToDouble(teacherLogits[t][i]);
                    }
                    Array.Sort(values);
                    double median = _teachers.Length % 2 == 0
                        ? (values[_teachers.Length / 2 - 1] + values[_teachers.Length / 2]) / 2.0
                        : values[_teachers.Length / 2];
                    result[i] = NumOps.FromDouble(median);
                }
                break;

            default:
                throw new NotImplementedException($"Aggregation mode {_aggregationMode} not implemented");
        }

        return result;
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    private Vector<T> Softmax(Vector<T> logits)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);

        // Find max for numerical stability
        double max = Convert.ToDouble(logits[0]);
        for (int i = 1; i < n; i++)
        {
            double val = Convert.ToDouble(logits[i]);
            if (val > max) max = val;
        }

        // Compute exp(logit - max) and sum
        double sum = 0;
        var expValues = new double[n];
        for (int i = 0; i < n; i++)
        {
            expValues[i] = Math.Exp(Convert.ToDouble(logits[i]) - max);
            sum += expValues[i];
        }

        // Normalize to get probabilities
        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.FromDouble(expValues[i] / sum);
        }

        return result;
    }

    /// <summary>
    /// Updates teacher weights based on performance (for adaptive weighting).
    /// </summary>
    /// <param name="validationInputs">Validation inputs for evaluation.</param>
    /// <param name="validationLabels">Validation labels.</param>
    /// <remarks>
    /// <para><b>For Advanced Users:</b> Call this periodically to adjust weights based on
    /// each teacher's current performance. Better teachers get higher weights.</para>
    /// </remarks>
    public void UpdateWeights(Vector<T>[] validationInputs, Vector<T>[] validationLabels)
    {
        if (_weights == null)
            throw new InvalidOperationException("Cannot update weights when none were provided");

        var accuracies = new double[_teachers.Length];

        // Evaluate each teacher
        for (int t = 0; t < _teachers.Length; t++)
        {
            int correct = 0;
            for (int i = 0; i < validationInputs.Length; i++)
            {
                var prediction = _teachers[t].GetLogits(validationInputs[i]);
                if (ArgMax(prediction) == ArgMax(validationLabels[i]))
                    correct++;
            }
            accuracies[t] = (double)correct / validationInputs.Length;
        }

        // Convert accuracies to weights (softmax for smooth distribution)
        double sum = accuracies.Sum();
        if (sum > 0)
        {
            for (int t = 0; t < _teachers.Length; t++)
            {
                _weights[t] = accuracies[t] / sum;
            }
        }
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
/// Defines how ensemble predictions are aggregated.
/// </summary>
public enum EnsembleAggregationMode
{
    /// <summary>
    /// Weighted average of teacher logits (most common).
    /// </summary>
    WeightedAverage,

    /// <summary>
    /// Geometric mean of teacher logits (for multiplicative ensembles).
    /// </summary>
    GeometricMean,

    /// <summary>
    /// Element-wise maximum (for pessimistic ensembles).
    /// </summary>
    Maximum,

    /// <summary>
    /// Element-wise median (robust to outliers).
    /// </summary>
    Median
}
