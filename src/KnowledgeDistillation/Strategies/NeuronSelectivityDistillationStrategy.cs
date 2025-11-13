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
/// </remarks>
public class NeuronSelectivityDistillationStrategy<T> : DistillationStrategyBase<T, Vector<T>>
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

    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        // TODO: This strategy currently only implements standard distillation loss
        // The selectivity component is not implemented. Need to:
        // 1. Compute selectivity loss from intermediate activations
        // 2. Combine with standard distillation loss using _selectivityWeight
        
        // Standard distillation loss
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);
        var softLoss = KLDivergence(teacherSoft, studentSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

        T finalLoss;
        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);
            finalLoss = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
        }
        else
        {
            finalLoss = softLoss;
        }

        return finalLoss;
    }

    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        // TODO: This strategy currently only implements standard distillation gradient
        // The selectivity gradient is not implemented. Need to:
        // 1. Compute selectivity gradient from intermediate activations
        // 2. Combine with standard gradient using _selectivityWeight
        
        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);

        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);

            for (int i = 0; i < n; i++)
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
            for (int i = 0; i < n; i++)
            {
                // Soft gradient (temperature-scaled)
                var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                gradient[i] = softGrad;
            }
        }

        return gradient;
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


