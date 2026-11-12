
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Distilling the Knowledge in a Neural Network",
    "https://arxiv.org/abs/1503.02531",
    Year = 2015,
    Authors = "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
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

    /// <inheritdoc />
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Standard distillation loss per sample (soft T²·KL, blended with hard CE when labels are given)
        T totalLoss = Engine.Sum(DistillationHelper<T>.StandardSampleLosses(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha));

        var studentSoftBatch = RowsOf(DistillationHelper<T>.SoftmaxRows(studentBatchOutput, Temperature));
        var teacherSoftBatch = RowsOf(DistillationHelper<T>.SoftmaxRows(teacherBatchOutput, Temperature));

        var standardLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var distributionLoss = ComputeDistributionalLoss(studentSoftBatch, teacherSoftBatch);

        return NumOps.Add(
            NumOps.Multiply(standardLoss, NumOps.FromDouble(1.0 - _distributionWeight)),
            distributionLoss);
    }

    /// <inheritdoc />
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Standard gradient, averaged over the batch like the standard loss
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        var gradientBatch = Engine.MatrixMultiplyScalar(
            DistillationHelper<T>.StandardSampleGradients(studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha),
            oneOverBatch);

        if (_distributionWeight > 0)
        {
            var studentSoft = DistillationHelper<T>.SoftmaxRows(studentBatchOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.SoftmaxRows(teacherBatchOutput, Temperature);

            // The distributional loss is a function of the whole batch of softened student distributions; its
            // gradient w.r.t. those probabilities goes through the softmax(z / T) Jacobian to reach the logits:
            //   ∂L/∂z_k = p_k * (∂L/∂p_k - Σ_m (∂L/∂p_m) p_m) / T
            var probGradient = ComputeDistributionalGradient(RowsOf(studentSoft), RowsOf(teacherSoft));
            var logitGradient = SoftmaxJacobianProduct(Tensor<T>.FromMatrix(studentSoft), Tensor<T>.FromMatrix(probGradient));

            gradientBatch = Engine.MatrixAdd(
                Engine.MatrixMultiplyScalar(gradientBatch, NumOps.FromDouble(1.0 - _distributionWeight)),
                Engine.MatrixMultiplyScalar(logitGradient.ToMatrix(), NumOps.FromDouble(_distributionWeight)));
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
            _ => throw new ArgumentOutOfRangeException(nameof(_mode), _mode, $"Unrecognised ProbabilisticMode '{_mode}'. Valid modes: MomentMatching, MaximumMeanDiscrepancy, EntropyTransfer.")
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
        var meanDiff = Engine.Subtract(studentMeans, teacherMeans);
        T meanLoss = NumOps.Divide(Engine.DotProduct(meanDiff, meanDiff), NumOps.FromDouble(numClasses));

        // MSE on variances (second moment)
        var varDiff = Engine.Subtract(studentVars, teacherVars);
        T varLoss = NumOps.Divide(Engine.DotProduct(varDiff, varDiff), NumOps.FromDouble(numClasses));

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
        var studentEntropies = ComputeEntropies(studentPredictions);
        var teacherEntropies = ComputeEntropies(teacherPredictions);

        // MSE on entropies (uncertainty calibration)
        var diff = Engine.Subtract(studentEntropies, teacherEntropies);
        T entropyLoss = Engine.DotProduct(diff, diff);

        return NumOps.Divide(entropyLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Per-class mean over the batch.
    /// </summary>
    private Vector<T> ComputeMeans(Vector<T>[] predictions, int numClasses)
    {
        var values = DistillationHelper<T>.StackRows(predictions);
        return Engine.Reshape(Engine.ReduceMean(values, [0], keepDims: false), [numClasses]).ToVector();
    }

    /// <summary>
    /// Per-class (population) variance over the batch around the given means.
    /// </summary>
    private Vector<T> ComputeVariances(Vector<T>[] predictions, Vector<T> means, int numClasses)
    {
        var values = DistillationHelper<T>.StackRows(predictions);
        var centered = Engine.TensorSubtract(
            values, Engine.TensorBroadcastTo(Engine.Reshape(Tensor<T>.FromVector(means), [1, numClasses]), values.Shape.ToArray()));
        return Engine.Reshape(
            Engine.ReduceMean(Engine.TensorMultiply(centered, centered), [0], keepDims: false), [numClasses]).ToVector();
    }

    /// <summary>
    /// Average RBF kernel value over all pairs of the two sets (the diagonal is skipped when both are the
    /// same set, so a set is not compared with itself sample by sample).
    /// </summary>
    private T ComputeAverageKernel(Vector<T>[] set1, Vector<T>[] set2)
    {
        int n1 = set1.Length;
        int n2 = set2.Length;
        bool sameSet = set1 == set2;
        int pairCount = sameSet ? n1 * (n1 - 1) : n1 * n2;
        if (pairCount <= 0)
            return NumOps.Zero;

        var kernel = RBFKernelMatrix(DistillationHelper<T>.StackRows(set1), DistillationHelper<T>.StackRows(set2));
        if (sameSet)
        {
            // Skip diagonal for same set
            kernel = Engine.TensorMultiply(kernel, OffDiagonalMask(n1));
        }

        return NumOps.Divide(Engine.TensorSum(kernel), NumOps.FromDouble(pairCount));
    }

    /// <summary>
    /// RBF kernel k(x,y) = exp(-||x-y||²/(2*σ²)) for every row pair ([n, d] and [m, d] -> [n, m]), with the
    /// squared distance summed from the explicit differences.
    /// </summary>
    private Tensor<T> RBFKernelMatrix(Tensor<T> a, Tensor<T> b)
    {
        int n = a.Shape[0];
        int m = b.Shape[0];
        int d = a.Shape[1];
        var diff = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(Engine.Reshape(a, [n, 1, d]), [n, m, d]),
            Engine.TensorBroadcastTo(Engine.Reshape(b, [1, m, d]), [n, m, d]));
        var squaredDistance = Engine.ReduceSum(Engine.TensorMultiply(diff, diff), [2], keepDims: false);
        return Engine.TensorExp(Engine.TensorMultiplyScalar(
            squaredDistance, NumOps.FromDouble(-1.0 / (2.0 * _mmdBandwidth * _mmdBandwidth))));
    }

    /// <summary>
    /// Entropy -Σ p log p of every prediction (terms with p &lt;= ε are skipped).
    /// </summary>
    private Vector<T> ComputeEntropies(Vector<T>[] predictions)
    {
        var p = DistillationHelper<T>.StackRows(predictions);
        var terms = Engine.TensorWhere(
            Engine.TensorGreaterThan(p, NumOps.FromDouble(Epsilon)),
            Engine.TensorMultiply(p, Engine.TensorLog(p)),
            new Tensor<T>(p.Shape.ToArray()));
        return Engine.Negate(DistillationHelper<T>.RowSums(terms));
    }

    /// <summary>
    /// [n, n] mask with 0 on the diagonal and 1 elsewhere.
    /// </summary>
    private Tensor<T> OffDiagonalMask(int n)
    {
        return Engine.TensorSubtract(
            Tensor<T>.CreateDefault([n, n], NumOps.One), Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(n)));
    }

    /// <summary>
    /// Maps a gradient w.r.t. softmax(z / T) probabilities to a gradient w.r.t. the logits z:
    /// ∂L/∂z_k = p_k * (∂L/∂p_k - Σ_m (∂L/∂p_m) p_m) / T, row by row.
    /// </summary>
    private Tensor<T> SoftmaxJacobianProduct(Tensor<T> probabilities, Tensor<T> probabilityGradient)
    {
        var weightedSum = Engine.ReduceSum(Engine.TensorMultiply(probabilityGradient, probabilities), [1], keepDims: true);
        var centered = Engine.TensorSubtract(
            probabilityGradient, Engine.TensorBroadcastTo(weightedSum, probabilities.Shape.ToArray()));
        return Engine.TensorDivideScalar(Engine.TensorMultiply(probabilities, centered), NumOps.FromDouble(Temperature));
    }

    /// <summary>
    /// The rows of <paramref name="matrix"/> as separate vectors.
    /// </summary>
    private static Vector<T>[] RowsOf(Matrix<T> matrix)
    {
        return Enumerable.Range(0, matrix.Rows).Select(r => matrix.GetRow(r)).ToArray();
    }

    private Matrix<T> ComputeDistributionalGradient(Vector<T>[] studentSoft, Vector<T>[] teacherSoft)
    {
        return _mode switch
        {
            ProbabilisticMode.MomentMatching => ComputeMomentMatchingGradient(studentSoft, teacherSoft),
            ProbabilisticMode.MaximumMeanDiscrepancy => ComputeMMDGradient(studentSoft, teacherSoft),
            ProbabilisticMode.EntropyTransfer => ComputeEntropyGradient(studentSoft, teacherSoft),
            _ => throw new ArgumentOutOfRangeException(nameof(_mode), _mode, $"Unrecognised ProbabilisticMode '{_mode}'. Valid modes: MomentMatching, MaximumMeanDiscrepancy, EntropyTransfer.")
        };
    }

    /// <summary>
    /// Gradient of the moment matching loss w.r.t. the student probabilities:
    /// L = ½·mean_c (μs_c - μt_c)² + ½·mean_c (vs_c - vt_c)², so
    /// ∂L/∂p_bc = (μs_c - μt_c)/(B·C) + 2 (vs_c - vt_c)(p_bc - μs_c)/(B·C).
    /// </summary>
    private Matrix<T> ComputeMomentMatchingGradient(Vector<T>[] studentSoft, Vector<T>[] teacherSoft)
    {
        int batchSize = studentSoft.Length;
        int numClasses = studentSoft[0].Length;
        int[] shape = [batchSize, numClasses];

        var studentMeans = ComputeMeans(studentSoft, numClasses);
        var teacherMeans = ComputeMeans(teacherSoft, numClasses);
        var studentVars = ComputeVariances(studentSoft, studentMeans, numClasses);
        var teacherVars = ComputeVariances(teacherSoft, teacherMeans, numClasses);

        var p = DistillationHelper<T>.StackRows(studentSoft);
        T scale = NumOps.FromDouble(1.0 / ((double)batchSize * numClasses));
        var meanGrad = Engine.TensorMultiplyScalar(
            Engine.Reshape(Tensor<T>.FromVector(Engine.Subtract(studentMeans, teacherMeans)), [1, numClasses]), scale);
        var centered = Engine.TensorSubtract(
            p, Engine.TensorBroadcastTo(Engine.Reshape(Tensor<T>.FromVector(studentMeans), [1, numClasses]), shape));
        var varScale = Engine.TensorMultiplyScalar(
            Engine.Reshape(Tensor<T>.FromVector(Engine.Subtract(studentVars, teacherVars)), [1, numClasses]),
            NumOps.Multiply(NumOps.FromDouble(2.0), scale));
        var varGrad = Engine.TensorMultiply(centered, Engine.TensorBroadcastTo(varScale, shape));

        return Engine.TensorAdd(Engine.TensorBroadcastTo(meanGrad, shape), varGrad).ToMatrix();
    }

    /// <summary>
    /// Gradient of MMD² = E[k(s,s')] + E[k(t,t')] - 2E[k(s,t)] w.r.t. the student probabilities, with
    /// ∂k(x,y)/∂x = -k(x,y)(x - y)/σ² and the same pair normalization as the loss (n(n-1) for the
    /// student-student term without the diagonal, n·m for the cross term).
    /// </summary>
    private Matrix<T> ComputeMMDGradient(Vector<T>[] studentSoft, Vector<T>[] teacherSoft)
    {
        int batchSize = studentSoft.Length;
        int numClasses = studentSoft[0].Length;
        int teacherCount = teacherSoft.Length;
        int[] shape = [batchSize, numClasses];
        double sigmaSquared = _mmdBandwidth * _mmdBandwidth;

        var s = DistillationHelper<T>.StackRows(studentSoft);
        var t = DistillationHelper<T>.StackRows(teacherSoft);

        // Σ_j k_ij (x_i - y_j) = rowsum(K)_i * x_i - (K @ Y)_i
        Tensor<T> KernelWeightedDifferences(Tensor<T> kernel, Tensor<T> others) =>
            Engine.TensorSubtract(
                Engine.TensorMultiply(s, Engine.TensorBroadcastTo(Engine.ReduceSum(kernel, [1], keepDims: true), shape)),
                Engine.TensorMatMul(kernel, others));

        var gradient = new Tensor<T>(shape);

        int studentPairs = batchSize * (batchSize - 1);
        if (studentPairs > 0)
        {
            // d/ds_i of Σ_{a≠b} k(s_a, s_b) / (n(n-1)): each unordered pair appears twice
            var studentKernel = Engine.TensorMultiply(RBFKernelMatrix(s, s), OffDiagonalMask(batchSize));
            gradient = Engine.TensorMultiplyScalar(
                KernelWeightedDifferences(studentKernel, s), NumOps.FromDouble(-2.0 / (sigmaSquared * studentPairs)));
        }

        int crossPairs = batchSize * teacherCount;
        if (crossPairs > 0)
        {
            // d/ds_i of -2 Σ_{i,j} k(s_i, t_j) / (n·m)
            var crossKernel = RBFKernelMatrix(s, t);
            gradient = Engine.TensorAdd(gradient, Engine.TensorMultiplyScalar(
                KernelWeightedDifferences(crossKernel, t), NumOps.FromDouble(2.0 / (sigmaSquared * crossPairs))));
        }

        return gradient.ToMatrix();
    }

    /// <summary>
    /// Gradient of mean_b (Hs_b - Ht_b)² w.r.t. the student probabilities, using
    /// ∂H/∂p = -(1 + log(p + ε)).
    /// </summary>
    private Matrix<T> ComputeEntropyGradient(Vector<T>[] studentSoft, Vector<T>[] teacherSoft)
    {
        int batchSize = studentSoft.Length;
        int numClasses = studentSoft[0].Length;
        int[] shape = [batchSize, numClasses];

        var entropyDiff = Engine.Subtract(ComputeEntropies(studentSoft), ComputeEntropies(teacherSoft));

        var p = DistillationHelper<T>.StackRows(studentSoft);
        var entropyGrad = Engine.TensorNegate(Engine.TensorAddScalar(
            Engine.TensorLog(Engine.TensorAddScalar(p, NumOps.FromDouble(Epsilon))), NumOps.One));

        var rowScale = Engine.TensorMultiplyScalar(
            Engine.Reshape(Tensor<T>.FromVector(entropyDiff), [batchSize, 1]), NumOps.FromDouble(2.0 / batchSize));
        return Engine.TensorMultiply(entropyGrad, Engine.TensorBroadcastTo(rowScale, shape)).ToMatrix();
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


