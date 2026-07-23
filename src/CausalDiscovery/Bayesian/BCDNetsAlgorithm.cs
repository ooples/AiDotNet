using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// BCD-Nets — Bayesian Causal Discovery Networks.
/// </summary>
/// <remarks>
/// <para>
/// BCD-Nets use variational inference to approximate the joint posterior over DAG structures
/// and parameters. The graph structure is parameterized via Gumbel-Softmax continuous
/// relaxation of binary edge variables, and parameters (edge weights) are modeled via
/// a factorized Gaussian variational posterior. Both are optimized jointly via gradient
/// ascent on the ELBO.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize edge logits Z[i,j] and weight means mu[i,j], log-variances logvar[i,j]</item>
/// <item>Sample edges: E[i,j] ~ GumbelSigmoid(Z[i,j]/tau)</item>
/// <item>Sample weights: W[i,j] ~ N(mu[i,j], exp(logvar[i,j]))</item>
/// <item>Effective adjacency: A = E * W (element-wise)</item>
/// <item>Compute ELBO = E_q[log p(X|A)] - KL(q(Z)||p(Z)) - KL(q(W)||p(W)) - lambda*h(E)</item>
/// <item>Update Z, mu, logvar via gradient ascent</item>
/// <item>Anneal temperature tau and update augmented Lagrangian</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BCD-Nets learn both the graph structure AND the strength of each
/// connection simultaneously, using modern deep learning optimization techniques. They
/// provide uncertainty estimates for both.
/// </para>
/// <para>
/// Reference: Cundy et al. (2021), "BCD Nets: Scalable Variational Approaches for
/// Bayesian Causal Discovery", NeurIPS.
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
[ResearchPaper("BCD Nets: Scalable Variational Approaches for Bayesian Causal Discovery", "https://proceedings.neurips.cc/paper/2021/hash/5a378f8490c8d6af8647a753812f6e31-Abstract.html", Year = 2021, Authors = "Chris Cundy, Aditya Grover, Stefano Ermon")]
public class BCDNetsAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly double _learningRate;
    private const double RhoMax = 1e+16;
    private const double RhoMultiplier = 10.0;
    private const double RhoIncreaseThreshold = 0.25;

    /// <inheritdoc/>
    public override string Name => "BCD-Nets";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public BCDNetsAlgorithm(CausalDiscoveryOptions? options = null)
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
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        T lr = NumOps.FromDouble(_learningRate);

        // Variational parameters: edge logits Z, weight means mu, weight log-variances logvar
        var Z = new Matrix<T>(d, d);       // edge presence logits
        var mu = new Matrix<T>(d, d);      // weight means
        var logvar = new Matrix<T>(d, d);  // log-variance of weights

        // Initialize mu from OLS; initialize Z DIRECTION-AWARE per pair.
        //
        // The previous init turned BOTH directions of every correlated pair ON
        // (any |OLS slope| > 0.2 → logit +1), so the optimization started from a
        // dense graph of symmetric 2-cycles. The acyclicity penalty then pruned
        // directions essentially arbitrarily and the posterior routinely settled
        // on a REVERSED or spurious DAG (Markov-equivalent fits the covariance,
        // so the reconstruction term cannot recover the orientation once the
        // wrong basin is entered).
        //
        // For the linear-Gaussian equal-variance SEMs this variational sketch
        // models, the causal direction is identifiable by the smaller
        // conditional residual variance (Peters & Bühlmann 2014, the
        // identifiability result the NOTEARS family builds on):
        //   resVar(j|i) = cov[j,j] − cov[i,j]²/cov[i,i]   for i → j.
        // Initialize only the lower-residual direction of each pair ON so the
        // optimizer starts inside the true-DAG basin; the reverse logit starts
        // OFF and can still be revived by the reconstruction gradient if the
        // data disagrees.
        T eps = NumOps.FromDouble(1e-10);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                if (NumOps.GreaterThan(cov[i, i], eps))
                    mu[i, j] = NumOps.Divide(cov[i, j], cov[i, i]);
                Z[i, j] = NumOps.FromDouble(-1.0);
                logvar[i, j] = NumOps.FromDouble(-4); // small initial variance
            }

        for (int i = 0; i < d; i++)
            for (int j = i + 1; j < d; j++)
            {
                double vi = NumOps.ToDouble(cov[i, i]);
                double vj = NumOps.ToDouble(cov[j, j]);
                double cij = NumOps.ToDouble(cov[i, j]);
                if (vi < 1e-10 || vj < 1e-10) continue;

                double olsIJ = Math.Abs(cij / vi);
                double olsJI = Math.Abs(cij / vj);
                if (Math.Max(olsIJ, olsJI) <= 0.2) continue; // no signal in either direction

                double resJgivenI = vj - cij * cij / vi; // residual of i → j
                double resIgivenJ = vi - cij * cij / vj; // residual of j → i
                if (resJgivenI <= resIgivenJ)
                    Z[i, j] = NumOps.FromDouble(1.0);
                else
                    Z[j, i] = NumOps.FromDouble(1.0);
            }

        // Augmented Lagrangian for acyclicity. Per the NOTEARS scheme BCD-Nets
        // builds on (Zheng et al. 2018 §3.2; Cundy et al. 2021 §4), the duals are
        // updated on an OUTER loop after an inner optimization with FIXED
        // (alpha, rho): alpha <- alpha + rho*h once per round, and rho is
        // escalated only when h fails to shrink by the relative factor 0.25
        // versus the previous round (h_new > 0.25 * h_prev). The previous code
        // performed the dual ascent after EVERY gradient step against an
        // ABSOLUTE h > 0.25 test, so rho exploded 10x per step to 1e16 within
        // ~16 steps and the acyclicity term symmetrically annihilated every
        // edge logit before the reconstruction term could orient the graph —
        // the discovered structure was always empty.
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        double hPrev = double.PositiveInfinity;
        int innerSteps = Math.Max(1, Math.Min(25, NumSamples));

        for (int iter = 0; iter < NumSamples; iter++)
        {
            double tau = Math.Max(0.1, 1.0 * Math.Pow(0.95, iter));
            T tauT = NumOps.FromDouble(tau);

            // Compute edge probabilities and sample weights
            var P = new Matrix<T>(d, d);
            var W = new Matrix<T>(d, d);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;

                    // Edge probability via sigmoid
                    T zScaled = NumOps.Divide(Z[i, j], tauT);
                    double sv = NumOps.ToDouble(zScaled);
                    double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                    P[i, j] = NumOps.FromDouble(sigVal);

                    // Reparameterized weight sample: w = mu + exp(logvar/2) * epsilon
                    T stddev = NumOps.FromDouble(Math.Exp(0.5 * NumOps.ToDouble(logvar[i, j])));
                    T epsilon = NumOps.FromDouble(rng.NextDouble() * 2 - 1); // approximate N(0,1)
                    T wSample = NumOps.Add(mu[i, j], NumOps.Multiply(stddev, epsilon));

                    W[i, j] = NumOps.Multiply(P[i, j], wSample);
                }

            // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var PSqCurrent = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    PSqCurrent[i, j] = NumOps.Multiply(P[i, j], P[i, j]);
            var expPSqCurrent = MatrixExponentialTaylor(PSqCurrent, d);
            T hValCurrent = NumOps.Zero;
            for (int i = 0; i < d; i++)
                hValCurrent = NumOps.Add(hValCurrent, expPSqCurrent[i, i]);
            hValCurrent = NumOps.Subtract(hValCurrent, NumOps.FromDouble(d));

            var gradMu = new Matrix<T>(d, d);
            var gradLogvar = new Matrix<T>(d, d);
            var gradZ = new Matrix<T>(d, d);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;

                    // Reconstruction gradient
                    T reconGrad = NumOps.Negate(cov[i, j]);
                    for (int k = 0; k < d; k++)
                    {
                        if (k == j) continue;
                        reconGrad = NumOps.Add(reconGrad, NumOps.Multiply(W[k, j], cov[i, k]));
                    }

                    // Gradient w.r.t. mu: P[i,j] * reconGrad
                    gradMu[i, j] = NumOps.Multiply(P[i, j], reconGrad);

                    // KL for weight: d_KL(N(mu, sigma^2) || N(0, 1)) gradient
                    // = mu - gradient from variance term
                    T klMuGrad = NumOps.Multiply(NumOps.FromDouble(0.01), mu[i, j]);
                    gradMu[i, j] = NumOps.Add(gradMu[i, j], klMuGrad);

                    // Gradient w.r.t. logvar: through reparameterization
                    T var_ij = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logvar[i, j])));
                    T klVarGrad = NumOps.Multiply(NumOps.FromDouble(0.5),
                        NumOps.Subtract(var_ij, NumOps.One));
                    gradLogvar[i, j] = klVarGrad;

                    // Gradient w.r.t. Z: through edge probability
                    T pij = P[i, j];
                    T oneMinusP = NumOps.Subtract(NumOps.One, pij);
                    T wSample = NumOps.GreaterThan(pij, NumOps.FromDouble(1e-10))
                        ? NumOps.Divide(W[i, j], pij) : NumOps.Zero;
                    T edgeGrad = NumOps.Multiply(reconGrad, wSample);

                    // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
                    T acycGrad = NumOps.Multiply(
                        NumOps.Add(alpha, NumOps.Multiply(rho, hValCurrent)),
                        NumOps.Multiply(expPSqCurrent[j, i], NumOps.Multiply(NumOps.FromDouble(2), pij)));

                    T totalEdgeGrad = NumOps.Add(edgeGrad, acycGrad);
                    T sigDeriv = NumOps.Divide(NumOps.Multiply(pij, oneMinusP),
                        NumOps.Add(tauT, NumOps.FromDouble(1e-10)));
                    gradZ[i, j] = NumOps.Multiply(totalEdgeGrad, sigDeriv);
                }

            // Apply gradients. Z is clamped to [-4, 4]: during the early phase the
            // acyclicity penalty pushes EVERY logit down roughly symmetrically
            // (all 2-cycles violate the constraint), and an unclamped Z saturates
            // so deep that sigmoid'(Z/tau) ≈ 0 freezes ALL edge gradients — the
            // reconstruction term can then never pull the true edges back and the
            // posterior collapses to the empty graph. Clamping keeps the gate
            // responsive (standard logit clamping for Gumbel-sigmoid relaxations)
            // while sigmoid(±4) ≈ 0.018/0.982 still saturates the edge decision.
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    Z[i, j] = NumOps.Subtract(Z[i, j], NumOps.Multiply(lr, gradZ[i, j]));
                    double zClamped = Math.Max(-4.0, Math.Min(4.0, NumOps.ToDouble(Z[i, j])));
                    Z[i, j] = NumOps.FromDouble(zClamped);
                    mu[i, j] = NumOps.Subtract(mu[i, j], NumOps.Multiply(lr, gradMu[i, j]));
                    logvar[i, j] = NumOps.Subtract(logvar[i, j],
                        NumOps.Multiply(NumOps.FromDouble(_learningRate * 0.1), gradLogvar[i, j]));
                }

            // Dual ascent once per OUTER round (every innerSteps gradient steps),
            // per the NOTEARS / BCD-Nets augmented-Lagrangian schedule:
            //   alpha <- alpha + rho * h
            //   rho   <- 10 * rho   only if h did not shrink to <= 0.25 * h_prev
            if ((iter + 1) % innerSteps == 0)
            {
                double hNow = NumOps.ToDouble(hValCurrent);

                // Dual ascent only while the acyclicity constraint is genuinely
                // violated. Once h ≈ 0 (the edge probabilities form a DAG) the
                // multiplier must stop growing — alpha += rho·h is ~0 anyway, but
                // skipping the rho-stall check prevents a needless rho escalation
                // from numerical jitter around zero.
                if (hNow > 1e-8)
                {
                    alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hValCurrent));

                    T rhoMaxT = NumOps.FromDouble(RhoMax);
                    if (!double.IsPositiveInfinity(hPrev)
                        && hNow > RhoIncreaseThreshold * hPrev)
                    {
                        T newRho = NumOps.Multiply(rho, NumOps.FromDouble(RhoMultiplier));
                        rho = NumOps.GreaterThan(newRho, rhoMaxT) ? rhoMaxT : newRho;
                    }
                }

                hPrev = hNow;
            }
        }

        // Final output: threshold edge probabilities and use posterior mean weights
        var result = new Matrix<T>(d, d);
        T edgeThreshold = NumOps.FromDouble(0.5);
        T weightThreshold = NumOps.FromDouble(0.1);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                double sv = NumOps.ToDouble(Z[i, j]) / 0.1;
                double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                T prob = NumOps.FromDouble(sigVal);

                if (NumOps.GreaterThan(prob, edgeThreshold))
                {
                    T weight = mu[i, j];
                    if (NumOps.GreaterThan(NumOps.Abs(weight), weightThreshold))
                        result[i, j] = weight;
                }
            }

        return result;
    }

}
