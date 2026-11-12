
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Variational distillation based on variational inference principles and information theory.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Production Use:</b> This strategy applies variational inference to knowledge distillation,
/// treating representations as distributions rather than point estimates. It implements concepts from
/// Variational Information Bottleneck (VIB) and Variational Autoencoders (VAE) for distillation.</para>
///
/// <para><b>Key Concept:</b> Instead of matching point predictions, we model representations as probability
/// distributions (typically Gaussian) and match these distributions. This captures uncertainty and enables:
/// 1. Latent space alignment - Match distributions in hidden layers
/// 2. Information bottleneck - Compress while preserving task-relevant information
/// 3. Uncertainty quantification - Transfer confidence estimates</para>
///
/// <para><b>Implementation:</b> We provide three variational modes:
/// - ELBO: Match Evidence Lower Bound (reconstruction + KL)
/// - InformationBottleneck: Minimize I(Z;X) while maximizing I(Z;Y) where Z is representation
/// - LatentSpaceKL: Match KL divergence in latent space between teacher and student</para>
///
/// <para><b>Mathematical Foundation:</b>
/// For Gaussian distributions N(μ,σ²), the KL divergence is:
/// KL(P||Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²)/(2σ_q²) - 1/2
///
/// The VIB objective: min I(X;Z) - βI(Z;Y) where β controls the trade-off.</para>
///
/// <para><b>Research Basis:</b> Based on:
/// - Variational Information Bottleneck (Alemi et al., 2017)
/// - Variational Knowledge Distillation (Ahn et al., 2019)
/// - Bayesian Dark Knowledge (Balan et al., 2015)</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Variational Information Distillation for Knowledge Transfer",
    "https://arxiv.org/abs/1904.05835",
    Year = 2019,
    Authors = "Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, et al.")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class VariationalDistillationStrategy<T> : DistillationStrategyBase<T>
{
    private readonly double _variationalWeight;
    private readonly VariationalMode _mode;
    private readonly double _betaIB; // Information bottleneck trade-off parameter

    public VariationalDistillationStrategy(
        double variationalWeight = 0.5,
        VariationalMode mode = VariationalMode.LatentSpaceKL,
        double betaIB = 1.0,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        if (variationalWeight < 0 || variationalWeight > 1)
            throw new ArgumentException("Variational weight must be between 0 and 1", nameof(variationalWeight));

        if (betaIB <= 0)
            throw new ArgumentException("Beta IB must be positive", nameof(betaIB));

        _variationalWeight = variationalWeight;
        _mode = mode;
        _betaIB = betaIB;
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

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <inheritdoc />
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Soft gradient: T² from loss scaling × 1/T from softmax chain rule = net factor T
        var gradientBatch = DistillationHelper<T>.StandardSampleGradients(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha);

        // Average gradients over the batch
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        gradientBatch = Engine.MatrixMultiplyScalar(gradientBatch, oneOverBatch);

        return gradientBatch;
    }

    /// <summary>
    /// Computes variational loss using latent representations with mean and variance.
    /// </summary>
    /// <param name="studentMean">Student's mean vector in latent space.</param>
    /// <param name="studentLogVar">Student's log variance vector in latent space.</param>
    /// <param name="teacherMean">Teacher's mean vector in latent space.</param>
    /// <param name="teacherLogVar">Teacher's log variance vector in latent space.</param>
    /// <returns>Variational loss based on selected mode.</returns>
    /// <remarks>
    /// <para>Representations should be parameterized as Gaussian distributions with mean and log variance.
    /// Log variance is used for numerical stability (variance must be positive).</para>
    /// </remarks>
    public T ComputeVariationalLoss(
        Vector<T> studentMean,
        Vector<T> studentLogVar,
        Vector<T> teacherMean,
        Vector<T> teacherLogVar)
    {
        if (studentMean.Length != teacherMean.Length || studentLogVar.Length != teacherLogVar.Length)
            throw new ArgumentException("Student and teacher dimensions must match");

        T loss = _mode switch
        {
            VariationalMode.ELBO => ComputeELBOLoss(studentMean, studentLogVar, teacherMean, teacherLogVar),
            VariationalMode.InformationBottleneck => ComputeVIBLoss(studentMean, studentLogVar, teacherMean, teacherLogVar),
            VariationalMode.LatentSpaceKL => ComputeLatentKLLoss(studentMean, studentLogVar, teacherMean, teacherLogVar),
            _ => throw new ArgumentOutOfRangeException(nameof(_mode), _mode, $"Unrecognised VariationalMode '{_mode}'. Valid modes: ELBO, InformationBottleneck, LatentSpaceKL.")
        };

        return NumOps.Multiply(loss, NumOps.FromDouble(_variationalWeight));
    }

    /// <summary>
    /// Computes variational loss for batch of latent representations.
    /// </summary>
    public T ComputeVariationalLossBatch(
        Vector<T>[] studentMeans,
        Vector<T>[] studentLogVars,
        Vector<T>[] teacherMeans,
        Vector<T>[] teacherLogVars)
    {
        if (studentMeans.Length != teacherMeans.Length)
            throw new ArgumentException("Batch sizes must match");

        int batchSize = studentMeans.Length;

        var sampleLosses = new List<T>(batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            sampleLosses.Add(ComputeVariationalLoss(
                studentMeans[i],
                studentLogVars[i],
                teacherMeans[i],
                teacherLogVars[i]));
        }

        T totalLoss = Engine.Sum(new Vector<T>(sampleLosses));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    private T ComputeELBOLoss(
        Vector<T> studentMean,
        Vector<T> studentLogVar,
        Vector<T> teacherMean,
        Vector<T> teacherLogVar)
    {
        // ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        // For distillation: match student ELBO to teacher ELBO
        // Simplified: minimize reconstruction loss + KL between student and teacher latents

        int dim = studentMean.Length;

        // Reconstruction term: MSE between means
        Vector<T> meanDiff = Engine.Subtract(studentMean, teacherMean);
        T reconstructionLoss = Engine.DotProduct(meanDiff, meanDiff);
        reconstructionLoss = NumOps.Divide(reconstructionLoss, NumOps.FromDouble(dim));

        // KL divergence term between the two Gaussian distributions
        T klLoss = ComputeGaussianKL(studentMean, studentLogVar, teacherMean, teacherLogVar);

        // ELBO loss: reconstruction + KL (both should be minimized)
        return NumOps.Add(reconstructionLoss, klLoss);
    }

    private T ComputeVIBLoss(
        Vector<T> studentMean,
        Vector<T> studentLogVar,
        Vector<T> teacherMean,
        Vector<T> teacherLogVar)
    {
        // Variational Information Bottleneck objective:
        // min I(X;Z) - β*I(Z;Y)
        //
        // I(X;Z) ≈ KL(q(z|x) || p(z)) - approximated as KL to standard normal
        // I(Z;Y) ≈ task loss (captured in main loss)
        //
        // For distillation: match student's I(X;Z) to teacher's, scaled by β

        // Student's information content (KL to standard normal N(0,1))
        T studentInfo = ComputeKLToStandardNormal(studentMean, studentLogVar);

        // Teacher's information content
        T teacherInfo = ComputeKLToStandardNormal(teacherMean, teacherLogVar);

        // Match information levels (teacher acts as target information level)
        var diff = Convert.ToDouble(NumOps.Subtract(studentInfo, teacherInfo));
        T infoLoss = NumOps.FromDouble(diff * diff);

        // Scale by β (information bottleneck parameter)
        return NumOps.Multiply(infoLoss, NumOps.FromDouble(_betaIB));
    }

    private T ComputeLatentKLLoss(
        Vector<T> studentMean,
        Vector<T> studentLogVar,
        Vector<T> teacherMean,
        Vector<T> teacherLogVar)
    {
        // Direct KL divergence between student and teacher latent distributions
        // KL(student || teacher)
        return ComputeGaussianKL(studentMean, studentLogVar, teacherMean, teacherLogVar);
    }

    private T ComputeGaussianKL(
        Vector<T> muP,
        Vector<T> logVarP,
        Vector<T> muQ,
        Vector<T> logVarQ)
    {
        // KL(P||Q) for two Gaussians:
        // KL = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²)/(2σ_q²) - 1/2
        //
        // Using log variance: log(σ²) = logVar, so σ² = exp(logVar)
        // KL = 0.5 * [logVar_q - logVar_p + (exp(logVar_p) + (μ_p - μ_q)²)/exp(logVar_q) - 1]

        // Per-dimension 0.5 * [logVar_q - logVar_p + (var_p + (mu_p - mu_q)^2) / (var_q + eps) - 1], summed
        Vector<T> muDiff = Engine.Subtract(muP, muQ);
        Vector<T> ratio = Engine.Divide(
            Engine.Add(Engine.Exp(logVarP), Engine.Multiply(muDiff, muDiff)),
            Engine.Add(Engine.Exp(logVarQ), Vector<T>.CreateDefault(logVarQ.Length, NumOps.FromDouble(Epsilon))));
        Vector<T> inner = Engine.Add(Engine.Subtract(logVarQ, logVarP), ratio);
        T kl = NumOps.Multiply(
            NumOps.FromDouble(0.5),
            NumOps.Subtract(Engine.Sum(inner), NumOps.FromDouble(muP.Length)));

        return kl;
    }

    private T ComputeKLToStandardNormal(Vector<T> mu, Vector<T> logVar)
    {
        // KL(N(μ,σ²) || N(0,1)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)

        Vector<T> inner = Engine.Subtract(Engine.Add(Engine.Multiply(mu, mu), Engine.Exp(logVar)), logVar);
        T kl = NumOps.Multiply(
            NumOps.FromDouble(0.5),
            NumOps.Subtract(Engine.Sum(inner), NumOps.FromDouble(mu.Length)));

        return kl;
    }

    /// <summary>
    /// Reparameterization trick for sampling from latent distribution during training.
    /// </summary>
    /// <param name="mean">Mean of the distribution.</param>
    /// <param name="logVar">Log variance of the distribution.</param>
    /// <param name="epsilon">Random noise from standard normal N(0,1).</param>
    /// <returns>Sample z = μ + σ * ε</returns>
    public Vector<T> Reparameterize(Vector<T> mean, Vector<T> logVar, Vector<T> epsilon)
    {
        // z = μ + σ * ε, where σ = exp(0.5 * logVar)
        Vector<T> sigma = Engine.Exp(Engine.Multiply(logVar, NumOps.FromDouble(0.5)));
        var sample = Engine.Add(mean, Engine.Multiply(sigma, epsilon));

        return sample;
    }



}

public enum VariationalMode
{
    /// <summary>
    /// Match Evidence Lower Bound (reconstruction + KL regularization).
    /// </summary>
    ELBO,

    /// <summary>
    /// Variational Information Bottleneck - minimize I(X;Z) while preserving task performance.
    /// </summary>
    InformationBottleneck,

    /// <summary>
    /// Direct KL divergence matching in latent space between teacher and student.
    /// </summary>
    LatentSpaceKL
}


