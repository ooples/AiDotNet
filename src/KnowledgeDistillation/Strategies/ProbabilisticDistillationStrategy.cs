using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Probabilistic distillation that transfers distributional knowledge by matching statistical properties.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Production Use:</b> This strategy treats model outputs as samples from probability distributions
/// and transfers knowledge about the entire distribution, not just point predictions. It matches statistical
/// moments (mean, variance, higher moments) and can use measures like Maximum Mean Discrepancy (MMD).</para>
///
/// <para><b>Key Concept:</b> Standard distillation matches individual predictions, but neural networks can be
/// viewed as probabilistic models. This strategy captures uncertainty and distribution shape by matching:
/// 1. First moment (mean) - Expected predictions
/// 2. Second moment (variance) - Prediction uncertainty
/// 3. Distribution distance (MMD, Wasserstein) - Overall shape</para>
///
/// <para><b>Implementation:</b> We provide three modes:
/// - MomentMatching: Match mean and variance of predictions across batch
/// - MaximumMeanDiscrepancy: Use MMD with RBF kernel to match distributions
/// - EntropyTransfer: Match prediction entropy (uncertainty calibration)</para>
///
/// <para><b>Research Basis:</b> Based on probabilistic knowledge distillation and Bayesian neural networks.
/// Particularly useful for uncertainty quantification and ensemble distillation.</para>
/// </remarks>
public class ProbabilisticDistillationStrategy<T> : DistillationStrategyBase<T>
{
    private readonly double _distributionWeight;
    private readonly ProbabilisticMode _mode;
    private readonly double _mmdBandwidth;

    public ProbabilisticDistillationStrategy(
        double distributionWeight = 0.5,
        ProbabilisticMode mode = ProbabilisticMode.MomentMatching,
        double mmdBandwidth = 1.0,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        if (distributionWeight < 0 || distributionWeight > 1)
            throw new ArgumentException("Distribution weight must be between 0 and 1", nameof(distributionWeight));

        if (mmdBandwidth <= 0)
            throw new ArgumentException("MMD bandwidth must be positive", nameof(mmdBandwidth));

        _distributionWeight = distributionWeight;
        _mode = mode;
        _mmdBandwidth = mmdBandwidth;
    }

    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        T totalLoss = NumOps.Zero;

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

            // Apply distribution weight reduction exactly once
            sampleLoss = NumOps.Multiply(sampleLoss, NumOps.FromDouble(1.0 - _distributionWeight));
            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        int outputDim = studentBatchOutput.Columns;
        var gradientBatch = new Matrix<T>(batchSize, outputDim);

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

                    // Apply distribution weight reduction exactly once
                    gradient[i] = NumOps.Multiply(combined, NumOps.FromDouble(1.0 - _distributionWeight));
                }
            }
            else
            {
                for (int i = 0; i < outputDim; i++)
                {
                    // Soft gradient (temperature-scaled)
                    var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                    softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                    // Apply distribution weight reduction exactly once
                    gradient[i] = NumOps.Multiply(softGrad, NumOps.FromDouble(1.0 - _distributionWeight));
                }
            }

            gradientBatch.SetRow(r, gradient);
        }

        return gradientBatch;
    }

    /// <summary>
    /// Computes distributional loss by matching statistical properties across a batch.
    /// </summary>
    /// <param name="studentPredictions">Student probability distributions for a batch.</param>
    /// <param name="teacherPredictions">Teacher probability distributions for a batch.</param>
    /// <returns>Distributional matching loss.</returns>
    /// <remarks>
    /// <para>This should be called with predictions (post-softmax) for an entire batch.
    /// The method will compute distributional statistics and match them between student and teacher.</para>
    /// </remarks>
    public T ComputeDistributionalLoss(Vector<T>[] studentPredictions, Vector<T>[] teacherPredictions)
    {
        if (studentPredictions.Length != teacherPredictions.Length)
            throw new ArgumentException("Student and teacher must have same batch size");

        if (studentPredictions.Length == 0)
            return NumOps.Zero;

        T loss = _mode switch
        {
            ProbabilisticMode.MomentMatching => ComputeMomentMatchingLoss(studentPredictions, teacherPredictions),
            ProbabilisticMode.MaximumMeanDiscrepancy => ComputeMMDLoss(studentPredictions, teacherPredictions),
            ProbabilisticMode.EntropyTransfer => ComputeEntropyLoss(studentPredictions, teacherPredictions),
            _ => throw new NotImplementedException($"Mode {_mode} not implemented")
        };

        return NumOps.Multiply(loss, NumOps.FromDouble(_distributionWeight));
    }

    private T ComputeMomentMatchingLoss(Vector<T>[] studentPredictions, Vector<T>[] teacherPredictions)
    {
        int batchSize = studentPredictions.Length;
        int numClasses = studentPredictions[0].Length;

        // Compute mean and variance for each class across the batch
        var studentMeans = ComputeMeans(studentPredictions, numClasses);
        var teacherMeans = ComputeMeans(teacherPredictions, numClasses);
        var studentVars = ComputeVariances(studentPredictions, studentMeans, numClasses);
        var teacherVars = ComputeVariances(teacherPredictions, teacherMeans, numClasses);

        // MSE on means (first moment)
        T meanLoss = NumOps.Zero;
        for (int i = 0; i < numClasses; i++)
        {
            double diff = studentMeans[i] - teacherMeans[i];
            meanLoss = NumOps.Add(meanLoss, NumOps.FromDouble(diff * diff));
        }
        meanLoss = NumOps.Divide(meanLoss, NumOps.FromDouble(numClasses));

        // MSE on variances (second moment)
        T varLoss = NumOps.Zero;
        for (int i = 0; i < numClasses; i++)
        {
            double diff = studentVars[i] - teacherVars[i];
            varLoss = NumOps.Add(varLoss, NumOps.FromDouble(diff * diff));
        }
        varLoss = NumOps.Divide(varLoss, NumOps.FromDouble(numClasses));

        // Combined loss (equal weight to both moments)
        return NumOps.Add(
            NumOps.Multiply(meanLoss, NumOps.FromDouble(0.5)),
            NumOps.Multiply(varLoss, NumOps.FromDouble(0.5)));
    }

    private T ComputeMMDLoss(Vector<T>[] studentPredictions, Vector<T>[] teacherPredictions)
    {
        // Maximum Mean Discrepancy with RBF kernel
        // MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        // where x~student, y~teacher

        int n = studentPredictions.Length;

        // E[k(x,x')] - student vs student
        T studentKernel = ComputeAverageKernel(studentPredictions, studentPredictions);

        // E[k(y,y')] - teacher vs teacher
        T teacherKernel = ComputeAverageKernel(teacherPredictions, teacherPredictions);

        // E[k(x,y)] - student vs teacher (cross term)
        T crossKernel = ComputeAverageKernel(studentPredictions, teacherPredictions);

        // MMD² = k(x,x') - 2*k(x,y) + k(y,y')
        var mmd = NumOps.Add(studentKernel, teacherKernel);
        mmd = NumOps.Subtract(mmd, NumOps.Multiply(crossKernel, NumOps.FromDouble(2.0)));

        // Return squared MMD (already computed above)
        return mmd;
    }

    private T ComputeEntropyLoss(Vector<T>[] studentPredictions, Vector<T>[] teacherPredictions)
    {
        int batchSize = studentPredictions.Length;

        // Compute entropy for each prediction
        var studentEntropies = new double[batchSize];
        var teacherEntropies = new double[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            studentEntropies[i] = ComputeEntropy(studentPredictions[i]);
            teacherEntropies[i] = ComputeEntropy(teacherPredictions[i]);
        }

        // MSE on entropies (uncertainty calibration)
        T entropyLoss = NumOps.Zero;
        for (int i = 0; i < batchSize; i++)
        {
            double diff = studentEntropies[i] - teacherEntropies[i];
            entropyLoss = NumOps.Add(entropyLoss, NumOps.FromDouble(diff * diff));
        }

        return NumOps.Divide(entropyLoss, NumOps.FromDouble(batchSize));
    }

    private double[] ComputeMeans(Vector<T>[] predictions, int numClasses)
    {
        int batchSize = predictions.Length;
        var means = new double[numClasses];

        for (int classIdx = 0; classIdx < numClasses; classIdx++)
        {
            double sum = 0;
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                sum += Convert.ToDouble(predictions[sampleIdx][classIdx]);
            }
            means[classIdx] = sum / batchSize;
        }

        return means;
    }

    private double[] ComputeVariances(Vector<T>[] predictions, double[] means, int numClasses)
    {
        int batchSize = predictions.Length;
        var variances = new double[numClasses];

        for (int classIdx = 0; classIdx < numClasses; classIdx++)
        {
            double sumSquaredDiff = 0;
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                double val = Convert.ToDouble(predictions[sampleIdx][classIdx]);
                double diff = val - means[classIdx];
                sumSquaredDiff += diff * diff;
            }
            variances[classIdx] = sumSquaredDiff / batchSize;
        }

        return variances;
    }

    private T ComputeAverageKernel(Vector<T>[] set1, Vector<T>[] set2)
    {
        int n1 = set1.Length;
        int n2 = set2.Length;
        T sum = NumOps.Zero;
        int pairCount = 0;

        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
            {
                if (set1 == set2 && i == j) continue; // Skip diagonal for same set

                double kernelValue = RBFKernel(set1[i], set2[j]);
                sum = NumOps.Add(sum, NumOps.FromDouble(kernelValue));
                pairCount++;
            }
        }

        return pairCount > 0 ? NumOps.Divide(sum, NumOps.FromDouble(pairCount)) : NumOps.Zero;
    }

    private double RBFKernel(Vector<T> v1, Vector<T> v2)
    {
        // RBF kernel: k(x,y) = exp(-||x-y||²/(2*σ²))
        double squaredDistance = 0;

        for (int i = 0; i < v1.Length; i++)
        {
            double diff = Convert.ToDouble(v1[i]) - Convert.ToDouble(v2[i]);
            squaredDistance += diff * diff;
        }

        return Math.Exp(-squaredDistance / (2.0 * _mmdBandwidth * _mmdBandwidth));
    }

    private double ComputeEntropy(Vector<T> probabilities)
    {
        double entropy = 0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            double p = Convert.ToDouble(probabilities[i]);
            if (p > Epsilon)
            {
                entropy -= p * Math.Log(p);
            }
        }

        return entropy;
    }



}

public enum ProbabilisticMode
{
    /// <summary>
    /// Match mean and variance of predictions (first and second moments).
    /// </summary>
    MomentMatching,

    /// <summary>
    /// Use Maximum Mean Discrepancy with RBF kernel to match distributions.
    /// </summary>
    MaximumMeanDiscrepancy,

    /// <summary>
    /// Match prediction entropy for uncertainty calibration.
    /// </summary>
    EntropyTransfer
}


