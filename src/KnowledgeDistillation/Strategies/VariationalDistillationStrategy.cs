using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
public class VariationalDistillationStrategy<T> : DistillationStrategyBase<T, Vector<T>>
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

    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        // Standard distillation loss (weighted)
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);
        var softLoss = KLDivergence(teacherSoft, studentSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);
            var combinedLoss = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
            return combinedLoss;
        }

        return softLoss;
    }

    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);

        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);

        for (int i = 0; i < n; i++)
        {
            var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
        }

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);

            for (int i = 0; i < n; i++)
            {
                var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);
                gradient[i] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[i]));
            }
        }

        return gradient;
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
            _ => throw new NotImplementedException($"Mode {_mode} not implemented")
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
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            T sampleLoss = ComputeVariationalLoss(
                studentMeans[i],
                studentLogVars[i],
                teacherMeans[i],
                teacherLogVars[i]);
            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

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
        T reconstructionLoss = NumOps.Zero;
        for (int i = 0; i < dim; i++)
        {
            double diff = Convert.ToDouble(NumOps.Subtract(studentMean[i], teacherMean[i]));
            reconstructionLoss = NumOps.Add(reconstructionLoss, NumOps.FromDouble(diff * diff));
        }
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

        int dim = muP.Length;
        T kl = NumOps.Zero;

        for (int i = 0; i < dim; i++)
        {
            double muPVal = Convert.ToDouble(muP[i]);
            double muQVal = Convert.ToDouble(muQ[i]);
            double logVarPVal = Convert.ToDouble(logVarP[i]);
            double logVarQVal = Convert.ToDouble(logVarQ[i]);

            double varP = Math.Exp(logVarPVal);
            double varQ = Math.Exp(logVarQVal);
            double muDiff = muPVal - muQVal;

            double klDim = 0.5 * (
                logVarQVal - logVarPVal +
                (varP + muDiff * muDiff) / (varQ + Epsilon) - 1.0
            );

            kl = NumOps.Add(kl, NumOps.FromDouble(klDim));
        }

        return kl;
    }

    private T ComputeKLToStandardNormal(Vector<T> mu, Vector<T> logVar)
    {
        // KL(N(μ,σ²) || N(0,1)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)

        int dim = mu.Length;
        T kl = NumOps.Zero;

        for (int i = 0; i < dim; i++)
        {
            double muVal = Convert.ToDouble(mu[i]);
            double logVarVal = Convert.ToDouble(logVar[i]);
            double varVal = Math.Exp(logVarVal);

            double klDim = 0.5 * (muVal * muVal + varVal - logVarVal - 1.0);
            kl = NumOps.Add(kl, NumOps.FromDouble(klDim));
        }

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
        int dim = mean.Length;
        var sample = new Vector<T>(dim);

        for (int i = 0; i < dim; i++)
        {
            // z = μ + σ * ε, where σ = exp(0.5 * logVar)
            double muVal = Convert.ToDouble(mean[i]);
            double logVarVal = Convert.ToDouble(logVar[i]);
            double sigma = Math.Exp(0.5 * logVarVal);
            double epsVal = Convert.ToDouble(epsilon[i]);

            double sampleVal = muVal + sigma * epsVal;
            sample[i] = NumOps.FromDouble(sampleVal);
        }

        return sample;
    }

    private Vector<T> Softmax(Vector<T> logits, double temperature)
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
            result[i] = NumOps.Divide(expValues[i], sum);

        return result;
    }

    private T KLDivergence(Vector<T> p, Vector<T> q)
    {
        T divergence = NumOps.Zero;
        for (int i = 0; i < p.Length; i++)
        {
            double pVal = Convert.ToDouble(p[i]);
            double qVal = Convert.ToDouble(q[i]);
            if (pVal > Epsilon)
                divergence = NumOps.Add(divergence, NumOps.FromDouble(pVal * Math.Log(pVal / (qVal + Epsilon))));
        }
        return divergence;
    }

    private T CrossEntropy(Vector<T> predictions, Vector<T> trueLabels)
    {
        T entropy = NumOps.Zero;
        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = Convert.ToDouble(predictions[i]);
            double label = Convert.ToDouble(trueLabels[i]);
            if (label > Epsilon)
                entropy = NumOps.Add(entropy, NumOps.FromDouble(-label * Math.Log(pred + Epsilon)));
        }
        return entropy;
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
