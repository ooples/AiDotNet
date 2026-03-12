using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// BayesDAG — Bayesian DAG learning with gradient-based posterior inference.
/// </summary>
/// <remarks>
/// <para>
/// BayesDAG uses a DAG-constrained variational framework that maintains a continuous
/// relaxation of the DAG posterior. It parameterizes edge probabilities via logits Z
/// and uses the Gumbel-Sigmoid trick for differentiable sampling. The acyclicity
/// constraint is enforced via augmented Lagrangian on the expected adjacency.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize edge logits Z[i,j] = 0 (uniform prior on each edge)</item>
/// <item>Sample adjacency: A[i,j] ~ Gumbel-Sigmoid(Z[i,j] / tau)</item>
/// <item>Compute data likelihood: L = -0.5 * ||X - X*A||^2 / n</item>
/// <item>Compute acyclicity: h(A) = tr(e^(A*A)) - d</item>
/// <item>Compute ELBO = likelihood - KL(q || prior) - lambda*h(A)</item>
/// <item>Update Z via gradient ascent on ELBO</item>
/// <item>Anneal temperature tau from soft to hard</item>
/// <item>Threshold final sigmoid(Z) to get binary adjacency</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BayesDAG is a modern Bayesian method that efficiently explores
/// the space of possible causal graphs using gradient-based optimization, providing
/// principled uncertainty quantification about the causal structure.
/// </para>
/// <para>
/// Reference: Annadani et al. (2024), "BayesDAG: Gradient-Based Posterior Inference
/// for Causal Discovery", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("BayesDAG: Gradient-Based Posterior Inference for Causal Discovery", "https://openreview.net/forum?id=VBnYjBJLvS", Year = 2024, Authors = "Yashas Annadani, Nick Pawlowski, Joel Jennings, Stefan Bauer, Cheng Zhang, Wenbo Gong")]
public class BayesDAGAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly double _learningRate;

    /// <inheritdoc/>
    public override string Name => "BayesDAG";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public BayesDAGAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyBayesianOptions(options);
        _learningRate = options?.LearningRate ?? 0.01;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var cov = ComputeCovarianceMatrix(data);

        // Edge logits Z[i,j]: sigmoid(Z) = edge probability
        var Z = new Matrix<T>(d, d);
        T lr = NumOps.FromDouble(_learningRate);

        // Augmented Lagrangian parameters
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        T rhoMax = NumOps.FromDouble(1e+10);

        for (int outerIter = 0; outerIter < NumSamples; outerIter++)
        {
            // Temperature annealing
            double tau = Math.Max(0.1, 1.0 * Math.Pow(0.95, outerIter));
            T tauT = NumOps.FromDouble(tau);

            // Compute edge probabilities: p[i,j] = sigmoid(Z[i,j] / tau)
            var P = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T zScaled = NumOps.Divide(Z[i, j], tauT);
                    double sv = NumOps.ToDouble(zScaled);
                    double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                    P[i, j] = NumOps.FromDouble(sigVal);
                }

            // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var PSqC = new Matrix<T>(d, d);
            for (int i2 = 0; i2 < d; i2++)
                for (int j2 = 0; j2 < d; j2++)
                    PSqC[i2, j2] = NumOps.Multiply(P[i2, j2], P[i2, j2]);
            var expPSqC = MatrixExponentialTaylor(PSqC, d);
            T hValC = NumOps.Zero;
            for (int i2 = 0; i2 < d; i2++)
                hValC = NumOps.Add(hValC, expPSqC[i2, i2]);
            hValC = NumOps.Subtract(hValC, NumOps.FromDouble(d));

            // Compute expected weighted adjacency: W[i,j] = P[i,j] * cov[i,j] / cov[i,i]
            var W = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T varI = cov[i, i];
                    T eps = NumOps.FromDouble(1e-10);
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T olsWeight = NumOps.Divide(cov[i, j], varI);
                        W[i, j] = NumOps.Multiply(P[i, j], olsWeight);
                    }
                }

            // Compute L2 loss gradient: dL/dW = (W^T * cov - cov) for each column
            // Simplified: gradient w.r.t. P[i,j] = dL/dW[i,j] * olsWeight
            var gradZ = new Matrix<T>(d, d);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;

                    // Data fit gradient: d(-likelihood)/dP[i,j]
                    T varI = cov[i, i];
                    T eps = NumOps.FromDouble(1e-10);
                    if (!NumOps.GreaterThan(varI, eps)) continue;

                    T olsWeight = NumOps.Divide(cov[i, j], varI);

                    // Residual contribution
                    T residGrad = NumOps.Zero;
                    for (int k = 0; k < d; k++)
                    {
                        if (k == j) continue;
                        residGrad = NumOps.Add(residGrad, NumOps.Multiply(W[k, j], cov[i, k]));
                    }
                    residGrad = NumOps.Subtract(residGrad, cov[i, j]);
                    T dataGrad = NumOps.Multiply(residGrad, olsWeight);

                    // KL divergence gradient: log(P/(1-P)) (Bernoulli prior = 0.5)
                    T pij = P[i, j];
                    T oneMinusP = NumOps.Subtract(NumOps.One, pij);
                    T klGrad = NumOps.Zero;
                    if (NumOps.GreaterThan(pij, NumOps.FromDouble(1e-8)) &&
                        NumOps.GreaterThan(oneMinusP, NumOps.FromDouble(1e-8)))
                    {
                        klGrad = NumOps.FromDouble(
                            Math.Log(NumOps.ToDouble(pij) / NumOps.ToDouble(oneMinusP)));
                    }

                    // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
                    T acycGrad = NumOps.Multiply(
                        NumOps.Add(alpha, NumOps.Multiply(rho, hValC)),
                        NumOps.Multiply(expPSqC[j, i], NumOps.Multiply(NumOps.FromDouble(2), pij)));

                    // Total gradient w.r.t. P[i,j]
                    T totalGradP = NumOps.Add(dataGrad, NumOps.Add(klGrad, acycGrad));

                    // Chain rule: dL/dZ = dL/dP * dP/dZ = dL/dP * P*(1-P)/tau
                    T sigDeriv = NumOps.Divide(NumOps.Multiply(pij, oneMinusP), tauT);
                    gradZ[i, j] = NumOps.Multiply(totalGradP, sigDeriv);
                }

            // Update logits
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    Z[i, j] = NumOps.Subtract(Z[i, j], NumOps.Multiply(lr, gradZ[i, j]));
                }

            // Update augmented Lagrangian with NOTEARS h(P) = tr(exp(P∘P)) - d
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hValC));
            if (NumOps.GreaterThan(hValC, NumOps.FromDouble(0.25)))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
            if (NumOps.GreaterThan(rho, rhoMax))
                rho = rhoMax;
        }

        // Final: threshold sigmoid(Z) and compute OLS weights
        var result = new Matrix<T>(d, d);
        T edgeThreshold = NumOps.FromDouble(0.5);
        T weightThreshold = NumOps.FromDouble(0.1);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                double sv = NumOps.ToDouble(Z[i, j]) / 0.1; // hard sigmoid
                double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                T prob = NumOps.FromDouble(sigVal);

                if (NumOps.GreaterThan(prob, edgeThreshold))
                {
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, NumOps.FromDouble(1e-10)))
                    {
                        T weight = NumOps.Divide(cov[i, j], varI);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), weightThreshold))
                            result[i, j] = weight;
                    }
                }
            }

        return result;
    }

    private Matrix<T> MatrixExponentialTaylor(Matrix<T> M, int d, int terms = 10)
    {
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            result[i, i] = NumOps.One;
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            power[i, i] = NumOps.One;
        for (int k = 1; k <= terms; k++)
        {
            var next = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    T sum = NumOps.Zero;
                    for (int l = 0; l < d; l++)
                        sum = NumOps.Add(sum, NumOps.Multiply(power[i, l], M[l, j]));
                    next[i, j] = sum;
                }
            power = next;
            T factorial = NumOps.FromDouble(1.0);
            for (int f = 2; f <= k; f++)
                factorial = NumOps.Multiply(factorial, NumOps.FromDouble(f));
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    result[i, j] = NumOps.Add(result[i, j], NumOps.Divide(power[i, j], factorial));
        }
        return result;
    }
}
