using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Abstract base class for curriculum distillation strategies with progressive difficulty adjustment.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality for curriculum learning,
/// including progress tracking, sample difficulty management, and temperature progression.</para>
///
/// <para><b>For Implementers:</b> Derive from this class and implement
/// <see cref="ComputeCurriculumTemperature"/> and <see cref="ShouldIncludeSample"/> to define
/// your specific curriculum progression logic.</para>
///
/// <para><b>Shared Features:</b>
/// - Curriculum progress tracking (0.0 to 1.0)
/// - Sample difficulty scoring and management
/// - Temperature range validation
/// - Step/epoch-based progression</para>
/// </remarks>
public abstract class CurriculumDistillationStrategyBase<T>
    : DistillationStrategyBase<T>, ICurriculumDistillationStrategy<T>
{
    private readonly Dictionary<int, double> _sampleDifficulties;
    private int _currentStep;

    /// <summary>
    /// Gets the total number of steps in the curriculum.
    /// </summary>
    public int TotalSteps { get; }

    /// <summary>
    /// Gets the current curriculum progress (0.0 to 1.0).
    /// </summary>
    public double CurriculumProgress => (double)_currentStep / TotalSteps;

    /// <summary>
    /// Gets the minimum temperature for the curriculum.
    /// </summary>
    public double MinTemperature { get; }

    /// <summary>
    /// Gets the maximum temperature for the curriculum.
    /// </summary>
    public double MaxTemperature { get; }

    /// <summary>
    /// Initializes a new instance of the CurriculumDistillationStrategyBase class.
    /// </summary>
    /// <param name="baseTemperature">Base temperature for distillation (default: 3.0).</param>
    /// <param name="alpha">Balance between hard and soft loss (default: 0.3).</param>
    /// <param name="minTemperature">Minimum temperature for curriculum (default: 2.0).</param>
    /// <param name="maxTemperature">Maximum temperature for curriculum (default: 5.0).</param>
    /// <param name="totalSteps">Total training steps/epochs (default: 100).</param>
    /// <param name="sampleDifficulties">Optional pre-defined difficulty scores.</param>
    protected CurriculumDistillationStrategyBase(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 2.0,
        double maxTemperature = 5.0,
        int totalSteps = 100,
        Dictionary<int, double>? sampleDifficulties = null)
        : base(baseTemperature, alpha)
    {
        if (minTemperature <= 0 || maxTemperature <= minTemperature)
            throw new ArgumentException("Temperature range invalid: must have 0 < min < max");
        if (totalSteps <= 0)
            throw new ArgumentException("Total steps must be positive", nameof(totalSteps));

        MinTemperature = minTemperature;
        MaxTemperature = maxTemperature;
        TotalSteps = totalSteps;
        _sampleDifficulties = sampleDifficulties ?? new Dictionary<int, double>();
        _currentStep = 0;
    }

    /// <summary>
    /// Updates the current curriculum progress.
    /// </summary>
    public virtual void UpdateProgress(int step)
    {
        // Don't clamp to TotalSteps - 1; allow reaching TotalSteps for full 100% progress
        _currentStep = Math.Max(0, Math.Min(step, TotalSteps));
    }

    /// <summary>
    /// Sets the difficulty score for a specific sample.
    /// </summary>
    public virtual void SetSampleDifficulty(int sampleIndex, double difficulty)
    {
        difficulty = Math.Max(0.0, Math.Min(1.0, difficulty));
        _sampleDifficulties[sampleIndex] = difficulty;
    }

    /// <summary>
    /// Gets the difficulty score for a sample, if set.
    /// </summary>
    public virtual double? GetSampleDifficulty(int sampleIndex)
    {
        return _sampleDifficulties.TryGetValue(sampleIndex, out double difficulty) ? difficulty : null;
    }

    /// <summary>
    /// Determines if a sample should be included in current curriculum stage.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override to define strategy-specific sample filtering logic.</para>
    /// <para>Default: Includes all samples (no filtering).</para>
    /// </remarks>
    public abstract bool ShouldIncludeSample(int sampleIndex);

    /// <summary>
    /// Computes the curriculum-adjusted temperature based on current progress.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override to define strategy-specific temperature progression.</para>
    /// </remarks>
    public abstract double ComputeCurriculumTemperature();

    /// <summary>
    /// Computes distillation loss with curriculum-adjusted temperature for a batch.
    /// </summary>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.RowCount;
        T totalLoss = NumOps.Zero;

        // Compute curriculum-adjusted temperature (applies to entire batch)
        double curriculumTemp = ComputeCurriculumTemperature();

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            // Compute soft loss with curriculum temperature
            var studentSoft = Softmax(studentRow, curriculumTemp);
            var teacherSoft = Softmax(teacherRow, curriculumTemp);

            var softLoss = KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(curriculumTemp * curriculumTemp));

            // Add hard loss if labels provided
            if (labelRow != null)
            {
                var studentProbs = Softmax(studentRow, temperature: 1.0);
                var hardLoss = CrossEntropy(studentProbs, labelRow);

                var alphaT = NumOps.FromDouble(Alpha);
                var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);

                var sampleLoss = NumOps.Add(
                    NumOps.Multiply(alphaT, hardLoss),
                    NumOps.Multiply(oneMinusAlpha, softLoss));

                totalLoss = NumOps.Add(totalLoss, sampleLoss);
            }
            else
            {
                totalLoss = NumOps.Add(totalLoss, softLoss);
            }
        }

        // Return average loss over batch
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes gradient with curriculum-adjusted temperature for a batch.
    /// </summary>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.RowCount;
        int numClasses = studentBatchOutput.ColumnCount;
        var gradient = new Matrix<T>(batchSize, numClasses);

        // Compute curriculum-adjusted temperature (applies to entire batch)
        double curriculumTemp = ComputeCurriculumTemperature();

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            // Soft gradient with curriculum temperature
            var studentSoft = Softmax(studentRow, curriculumTemp);
            var teacherSoft = Softmax(teacherRow, curriculumTemp);

            for (int c = 0; c < numClasses; c++)
            {
                var diff = NumOps.Subtract(studentSoft[c], teacherSoft[c]);
                gradient[r, c] = NumOps.Multiply(diff, NumOps.FromDouble(curriculumTemp * curriculumTemp));
            }

            // Add hard gradient if labels provided
            if (labelRow != null)
            {
                var studentProbs = Softmax(studentRow, temperature: 1.0);

                for (int c = 0; c < numClasses; c++)
                {
                    var hardGrad = NumOps.Subtract(studentProbs[c], labelRow[c]);
                    var alphaWeighted = NumOps.Multiply(hardGrad, NumOps.FromDouble(Alpha));
                    var softWeighted = NumOps.Multiply(gradient[r, c], NumOps.FromDouble(1.0 - Alpha));
                    gradient[r, c] = NumOps.Add(alphaWeighted, softWeighted);
                }
            }
            else
            {
                // Scale by (1 - alpha) if no hard loss
                for (int c = 0; c < numClasses; c++)
                {
                    gradient[r, c] = NumOps.Multiply(gradient[r, c], NumOps.FromDouble(1.0 - Alpha));
                }
            }
        }

        // Average gradients over batch
        T oneOverBatchSize = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                gradient[r, c] = NumOps.Multiply(gradient[r, c], oneOverBatchSize);
            }
        }

        return gradient;
    }

    /// <summary>
    /// Clamps a value to the temperature range [MinTemperature, MaxTemperature].
    /// </summary>
    protected double ClampTemperature(double temperature)
    {
        return Math.Max(MinTemperature, Math.Min(MaxTemperature, temperature));
    }

    protected Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];

        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(Convert.ToDouble(logits[i]) / temperature);

        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.Divide(expValues[i], sum);
        }

        return result;
    }

    protected T CrossEntropy(Vector<T> predictions, Vector<T> targets)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = Math.Max(Convert.ToDouble(predictions[i]), 1e-10);
            double target = Convert.ToDouble(targets[i]);
            loss = NumOps.Add(loss, NumOps.FromDouble(-target * Math.Log(pred)));
        }
        return loss;
    }

    protected T KLDivergence(Vector<T> predictions, Vector<T> targets)
    {
        T kl = NumOps.Zero;
        for (int i = 0; i < predictions.Length; i++)
        {
            double p = Math.Max(Convert.ToDouble(predictions[i]), 1e-10);
            double q = Math.Max(Convert.ToDouble(targets[i]), 1e-10);
            kl = NumOps.Add(kl, NumOps.FromDouble(q * Math.Log(q / p)));
        }
        return kl;
    }
}
