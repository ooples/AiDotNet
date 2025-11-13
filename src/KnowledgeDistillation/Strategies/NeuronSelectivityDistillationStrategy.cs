using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Neuron selectivity distillation that transfers the activation patterns and selectivity of individual neurons.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Production Use:</b> This strategy focuses on matching how individual neurons respond to inputs.
/// Some neurons are highly selective (activate strongly for specific patterns), while others are more general.
/// Transferring this selectivity helps the student learn meaningful feature representations.</para>
///
/// <para><b>Key Concept:</b> Neuron selectivity measures how discriminative each neuron is. A highly selective
/// neuron activates strongly for certain inputs and weakly for others. The distribution of selectivity across
/// neurons is important for model performance.</para>
///
/// <para><b>Implementation:</b> We measure selectivity using:
/// 1. Activation variance (how much neuron output varies across samples)
/// 2. Sparsity (what percentage of time the neuron is active)
/// 3. Peak-to-average ratio (how peaked the activation distribution is)</para>
///
/// <para><b>IMPORTANT - Two-Step Usage Pattern:</b> This strategy requires intermediate layer activations,
/// which are not available through the standard IDistillationStrategy interface methods. Use as follows:</para>
///
/// <para><b>Step 1:</b> Call ComputeLoss/ComputeGradient for standard distillation on final outputs (this provides
/// the base distillation loss using the teacher's output distribution).</para>
///
/// <para><b>Step 2:</b> Separately collect intermediate layer activations during the forward pass, then call
/// ComputeSelectivityLoss to compute the selectivity matching loss. Combine this with the base loss using
/// the selectivityWeight parameter:
/// <code>
/// T baseLoss = strategy.ComputeLoss(studentOutput, teacherOutput, labels);
/// T selectivityLoss = strategy.ComputeSelectivityLoss(studentActivations, teacherActivations);
/// T totalLoss = baseLoss + selectivityLoss;  // selectivityLoss is already weighted
/// </code></para>
///
/// <para>The selectivityWeight and metric parameters are used by ComputeSelectivityLoss, not by the
/// standard interface methods.</para>
/// </remarks>
public class NeuronSelectivityDistillationStrategy<T> : DistillationStrategyBase<T>
{
    private readonly double _selectivityWeight;
    private readonly SelectivityMetric _metric;

    public NeuronSelectivityDistillationStrategy(
        double selectivityWeight = 0.5,
        SelectivityMetric metric = SelectivityMetric.Variance,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        if (selectivityWeight < 0 || selectivityWeight > 1)
            throw new ArgumentException("Selectivity weight must be between 0 and 1", nameof(selectivityWeight));

        _selectivityWeight = selectivityWeight;
        _metric = metric;
    }

    /// <summary>
    /// Computes the base distillation loss on final outputs.
    /// </summary>
    /// <remarks>
    /// This method implements standard distillation loss (soft + hard loss) on final outputs.
    /// It does NOT include the selectivity component, which requires intermediate activations.
    /// Use ComputeSelectivityLoss separately and combine the losses manually. See class remarks for usage pattern.
    /// </remarks>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        T totalLoss = NumOps.Zero;

        // Standard distillation loss on final outputs
        // Selectivity loss requires intermediate activations - use ComputeSelectivityLoss separately
        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentOutput = studentBatchOutput.GetRow(r);
            Vector<T> teacherOutput = teacherBatchOutput.GetRow(r);
            Vector<T>? trueLabels = trueLabelsBatch?.GetRow(r);

            // Standard distillation loss
            var studentSoft = DistillationHelper<T>.Softmax(studentOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, Temperature);
            var softLoss = DistillationHelper<T>.KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

            T sampleLoss;
            if (trueLabels != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentOutput, 1.0);
                var hardLoss = DistillationHelper<T>.CrossEntropy(studentProbs, trueLabels);
                sampleLoss = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
            }
            else
            {
                sampleLoss = softLoss;
            }

            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes the gradient of the base distillation loss on final outputs.
    /// </summary>
    /// <remarks>
    /// This method implements standard distillation gradient (soft + hard) on final outputs.
    /// It does NOT include the selectivity gradient, which requires intermediate activations.
    /// Selectivity gradients must be computed separately and backpropagated through the network.
    /// See class remarks for usage pattern.
    /// </remarks>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        int outputDim = studentBatchOutput.Columns;
        var gradientBatch = new Matrix<T>(batchSize, outputDim);

        // Standard distillation gradient on final outputs
        // Selectivity gradients require intermediate activations and must be backpropagated separately
        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentOutput = studentBatchOutput.GetRow(r);
            Vector<T> teacherOutput = teacherBatchOutput.GetRow(r);
            Vector<T>? trueLabels = trueLabelsBatch?.GetRow(r);

            var gradient = new Vector<T>(outputDim);

            var studentSoft = DistillationHelper<T>.Softmax(studentOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, Temperature);

            if (trueLabels != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentOutput, 1.0);

                for (int i = 0; i < outputDim; i++)
                {
                    // Soft gradient (temperature-scaled)
                    var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                    softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                    // Hard gradient
                    var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);

                    // Combined gradient: Alpha * hardGrad + (1 - Alpha) * softGrad
                    var combined = NumOps.Add(
                        NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                        NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softGrad));

                    gradient[i] = combined;
                }
            }
            else
            {
                for (int i = 0; i < outputDim; i++)
                {
                    // Soft gradient (temperature-scaled)
                    var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                    softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                    gradient[i] = softGrad;
                }
            }

            gradientBatch.SetRow(r, gradient);
        }

        return gradientBatch;
    }

    /// <summary>
    /// Computes neuron selectivity loss by comparing activation patterns across a batch.
    /// </summary>
    /// <param name="studentActivations">Student neuron activations for a batch [batchSize x numNeurons].</param>
    /// <param name="teacherActivations">Teacher neuron activations for a batch [batchSize x numNeurons].</param>
    /// <returns>Selectivity matching loss.</returns>
    /// <remarks>
    /// <para>This should be called with intermediate layer activations, not final outputs.
    /// Collect activations for an entire batch, then call this method.</para>
    /// </remarks>
    public T ComputeSelectivityLoss(Vector<T>[] studentActivations, Vector<T>[] teacherActivations)
    {
        if (studentActivations.Length != teacherActivations.Length)
            throw new ArgumentException("Student and teacher must have same batch size");

        if (studentActivations.Length == 0)
            return NumOps.Zero;

        int batchSize = studentActivations.Length;
        int numNeurons = studentActivations[0].Length;

        // Validate all activation vectors have consistent dimensions
        for (int i = 0; i < batchSize; i++)
        {
            if (studentActivations[i].Length != numNeurons)
                throw new ArgumentException(
                    $"Student activation vector at index {i} has dimension {studentActivations[i].Length}, " +
                    $"but expected {numNeurons} (from activation 0). All activation vectors must have the same dimension.",
                    nameof(studentActivations));

            if (teacherActivations[i].Length != numNeurons)
                throw new ArgumentException(
                    $"Teacher activation vector at index {i} has dimension {teacherActivations[i].Length}, " +
                    $"but expected {numNeurons} (from activation 0). All activation vectors must have the same dimension.",
                    nameof(teacherActivations));
        }

        // Compute selectivity for each neuron
        var studentSelectivity = ComputeSelectivityScores(studentActivations, numNeurons);
        var teacherSelectivity = ComputeSelectivityScores(teacherActivations, numNeurons);

        // MSE between selectivity scores
        T loss = NumOps.Zero;
        for (int i = 0; i < numNeurons; i++)
        {
            var diff = studentSelectivity[i] - teacherSelectivity[i];
            loss = NumOps.Add(loss, NumOps.FromDouble(diff * diff));
        }

        loss = NumOps.Divide(loss, NumOps.FromDouble(numNeurons));
        return NumOps.Multiply(loss, NumOps.FromDouble(_selectivityWeight));
    }

    private double[] ComputeSelectivityScores(Vector<T>[] activations, int numNeurons)
    {
        int batchSize = activations.Length;
        var scores = new double[numNeurons];

        for (int neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++)
        {
            // Collect activations for this neuron across all samples
            var neuronActivations = new double[batchSize];
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                neuronActivations[sampleIdx] = Convert.ToDouble(activations[sampleIdx][neuronIdx]);
            }

            scores[neuronIdx] = _metric switch
            {
                SelectivityMetric.Variance => ComputeVariance(neuronActivations),
                SelectivityMetric.Sparsity => ComputeSparsity(neuronActivations),
                SelectivityMetric.PeakToAverage => ComputePeakToAverage(neuronActivations),
                _ => throw new NotImplementedException($"Metric {_metric} not implemented")
            };
        }

        return scores;
    }

    private double ComputeVariance(double[] values)
    {
        double mean = values.Average();
        double sumSquaredDiff = values.Sum(v => (v - mean) * (v - mean));
        return sumSquaredDiff / values.Length;
    }

    private double ComputeSparsity(double[] values)
    {
        // Percentage of near-zero activations (< 0.01)
        int nearZeroCount = values.Count(v => Math.Abs(v) < 0.01);
        return (double)nearZeroCount / values.Length;
    }

    private double ComputePeakToAverage(double[] values)
    {
        double avg = values.Average();
        double max = values.Max();
        return max / (avg + 1e-10);
    }



}

public enum SelectivityMetric
{
    /// <summary>
    /// Variance of activations (how much the neuron's output varies).
    /// </summary>
    Variance,

    /// <summary>
    /// Sparsity (percentage of near-zero activations).
    /// </summary>
    Sparsity,

    /// <summary>
    /// Peak-to-average ratio (how peaked the activation distribution is).
    /// </summary>
    PeakToAverage
}


