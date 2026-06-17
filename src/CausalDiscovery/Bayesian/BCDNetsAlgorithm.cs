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
    // NOTEARS-style penalty schedule. The ceiling is set to 1e6 (vs. 1e+16 in
    // the unconstrained NOTEARS reference) because the Gumbel-Sigmoid edge
    // probabilities saturate at 0.5 when Z ≈ 0 — and with P ≈ 0.5 across all
    // edges, h(P) = tr(exp(P∘P)) − d sits at a floor of ~(d−1)·0.25 instead of
    // approaching 0. Without a tighter ceiling, the rho-bump-on-no-progress
    // gate (which compares h_new against h_prev) triggers indefinitely and
    // pushes ρ to the cap, then α grows linearly with iteration count and
    // every edge logit gets crushed below the inference threshold.
    private const double RhoMax = 1e+6;
    private const double RhoMultiplier = 10.0;
    // Rho only increases when h_new exceeds 90% of h_prev — i.e. the inner
    // solver is genuinely not making progress on acyclicity. The 0.25 cutoff
    // from the NOTEARS paper is appropriate for an exact W (where small ρ
    // bumps suffice to drive h → 0), but Gumbel-Sigmoid P[i,j] floors at
    // 0.5 when Z=0, so the 0.25 condition would trigger on every outer
    // iteration that hasn't yet driven the edges to extreme Z magnitudes.
    private const double RhoIncreaseThreshold = 0.9;

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
    /// <remarks>
    /// Implements the augmented-Lagrangian outer/inner loop from Cundy et al. 2021 §4.
    /// Each outer iteration solves the variational ELBO maximisation w.r.t. (Z, mu,
    /// logvar) for a fixed acyclicity penalty (alpha, rho). After the inner solver
    /// converges (or hits its iteration cap), the constraint multipliers are updated:
    ///   alpha ← alpha + rho · h(P_outer)
    ///   if h_new &gt; 0.25 · h_prev:   rho ← min(10 · rho, rho_max)   (no progress)
    /// This is the standard NOTEARS schedule the BCD-Nets paper inherits; updating
    /// (alpha, rho) every gradient step (the previous structure) lets alpha grow
    /// linearly with iteration count and pushes the acyclicity gradient to crush
    /// every Z below the final sigmoid threshold — even on strongly-correlated
    /// data where every edge has |OLS weight| ≫ 0.2 at init.
    /// </remarks>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Standardise to zero-mean unit-variance per column — same convention as
        // NOTEARS / DAGMA reference implementations. Without this, the loss
        // gradients scale with the raw data magnitude and the acyclicity
        // tr(exp(W∘W)) constraint explodes for even moderate step sizes,
        // crushing every Z below the final sigmoid threshold (Zheng et al. 2018
        // §5 "We standardize each column of X to have unit variance" — Cundy et
        // al. 2021 inherit the same NOTEARS-style numerical regime).
        var standardised = StandardiseColumns(data);
        var cov = ComputeCovarianceMatrix(standardised);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        T lr = NumOps.FromDouble(_learningRate);

        // Variational parameters: edge logits Z, weight means mu, weight log-variances logvar
        var Z = new Matrix<T>(d, d);       // edge presence logits
        var mu = new Matrix<T>(d, d);      // weight means
        var logvar = new Matrix<T>(d, d);  // log-variance of weights

<<<<<<< HEAD
        // Initialise mu from OLS and Z from absolute OLS weight magnitude — this
        // biases the variational posterior toward edges with strong statistical
        // signal (Cundy et al. 2021 §4.2 warm-start, also a NOTEARS convention).
||||||| 0d65f659c
        // Initialize mu from OLS and Z from absolute OLS weight magnitude
        // This biases the search toward edges with strong statistical signal
=======
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
>>>>>>> origin/master
        T eps = NumOps.FromDouble(1e-10);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                if (NumOps.GreaterThan(cov[i, i], eps))
                    mu[i, j] = NumOps.Divide(cov[i, j], cov[i, i]);
<<<<<<< HEAD
                    double olsWeight = Math.Abs(NumOps.ToDouble(mu[i, j]));
                    Z[i, j] = NumOps.FromDouble(olsWeight > 0.2 ? 1.0 : -1.0);
                }
||||||| 0d65f659c
                    // Initialize edge logit from OLS strength: strong regression → positive logit
                    double olsWeight = Math.Abs(NumOps.ToDouble(mu[i, j]));
                    Z[i, j] = NumOps.FromDouble(olsWeight > 0.2 ? 1.0 : -1.0);
                }
=======
                Z[i, j] = NumOps.FromDouble(-1.0);
>>>>>>> origin/master
                logvar[i, j] = NumOps.FromDouble(-4); // small initial variance
            }

<<<<<<< HEAD
        // Outer/inner schedule. NumSamples is the *total* gradient-step budget;
        // split it into ~20 outer iterations of innerSteps each so the constraint
        // multipliers update at the right cadence.
        int outerIterations = Math.Max(1, Math.Min(50, NumSamples / 200));
        int innerSteps = Math.Max(1, NumSamples / outerIterations);

||||||| 0d65f659c
        // Augmented Lagrangian for acyclicity
=======
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
>>>>>>> origin/master
        T alpha = NumOps.Zero;
<<<<<<< HEAD
        // Start ρ small (rather than 1.0) so the first few outer iterations are
        // dominated by reconstruction and let the variational posterior find
        // the high-likelihood edges before the acyclicity ramp begins. This is
        // standard practice for NOTEARS-style methods on deterministic /
        // highly-collinear data, where the Gumbel-Sigmoid edge probabilities
        // sit near 0.5 at init and a unit ρ would already make the acyclicity
        // gradient compete with reconstruction (Cundy 2021 §4.2).
        T rho = NumOps.FromDouble(0.01);
        T rhoMaxT = NumOps.FromDouble(RhoMax);
        double prevH = double.MaxValue;
        int gradStep = 0;
||||||| 0d65f659c
        T rho = NumOps.One;
=======
        T rho = NumOps.One;
        double hPrev = double.PositiveInfinity;
        int innerSteps = Math.Max(1, Math.Min(25, NumSamples));
>>>>>>> origin/master

        for (int outerIter = 0; outerIter < outerIterations; outerIter++)
        {
            T outerAlpha = alpha;
            T outerRho = rho;

            for (int innerIter = 0; innerIter < innerSteps; innerIter++, gradStep++)
            {
                double tau = Math.Max(0.1, Math.Pow(0.95, gradStep));
                T tauT = NumOps.FromDouble(tau);

                // Compute edge probabilities and sample weights
                var P = new Matrix<T>(d, d);
                var W = new Matrix<T>(d, d);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        if (i == j) continue;

                        T zScaled = NumOps.Divide(Z[i, j], tauT);
                        double sv = NumOps.ToDouble(zScaled);
                        double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                        P[i, j] = NumOps.FromDouble(sigVal);

                        T stddev = NumOps.FromDouble(Math.Exp(0.5 * NumOps.ToDouble(logvar[i, j])));
                        T epsilon = NumOps.FromDouble(rng.NextDouble() * 2 - 1);
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

                        gradMu[i, j] = NumOps.Multiply(P[i, j], reconGrad);

                        // KL N(mu, sigma^2)||N(0,1): gradient through mu
                        T klMuGrad = NumOps.Multiply(NumOps.FromDouble(0.01), mu[i, j]);
                        gradMu[i, j] = NumOps.Add(gradMu[i, j], klMuGrad);

<<<<<<< HEAD
                        // KL gradient through logvar (reparameterised)
                        T var_ij = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logvar[i, j])));
                        T klVarGrad = NumOps.Multiply(NumOps.FromDouble(0.5),
                            NumOps.Subtract(var_ij, NumOps.One));
                        gradLogvar[i, j] = klVarGrad;

                        // Edge-presence gradient through P[i,j]
                        T pij = P[i, j];
                        T oneMinusP = NumOps.Subtract(NumOps.One, pij);
                        T wSample = NumOps.GreaterThan(pij, NumOps.FromDouble(1e-10))
                            ? NumOps.Divide(W[i, j], pij) : NumOps.Zero;
                        T edgeGrad = NumOps.Multiply(reconGrad, wSample);

                        // Acyclicity gradient: (alpha + rho·h) · [exp(P∘P)^T ∘ 2P][i,j].
                        // alpha/rho are the OUTER values fixed for the inner loop —
                        // the multiplier update happens only after innerSteps complete.
                        T acycGrad = NumOps.Multiply(
                            NumOps.Add(outerAlpha, NumOps.Multiply(outerRho, hValCurrent)),
                            NumOps.Multiply(expPSqCurrent[j, i], NumOps.Multiply(NumOps.FromDouble(2), pij)));

                        T totalEdgeGrad = NumOps.Add(edgeGrad, acycGrad);
                        T sigDeriv = NumOps.Divide(NumOps.Multiply(pij, oneMinusP),
                            NumOps.Add(tauT, NumOps.FromDouble(1e-10)));
                        gradZ[i, j] = NumOps.Multiply(totalEdgeGrad, sigDeriv);
                    }

                // Apply gradients
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        if (i == j) continue;
                        Z[i, j] = NumOps.Subtract(Z[i, j], NumOps.Multiply(lr, gradZ[i, j]));
                        mu[i, j] = NumOps.Subtract(mu[i, j], NumOps.Multiply(lr, gradMu[i, j]));
                        logvar[i, j] = NumOps.Subtract(logvar[i, j],
                            NumOps.Multiply(NumOps.FromDouble(_learningRate * 0.1), gradLogvar[i, j]));
                    }
            }

            // Outer step: re-evaluate h on the post-inner P and update (alpha, rho).
            var Pouter = new Matrix<T>(d, d);
            double finalTau = Math.Max(0.1, Math.Pow(0.95, gradStep));
||||||| 0d65f659c
            // Apply gradients
=======
            // Apply gradients. Z is clamped to [-4, 4]: during the early phase the
            // acyclicity penalty pushes EVERY logit down roughly symmetrically
            // (all 2-cycles violate the constraint), and an unclamped Z saturates
            // so deep that sigmoid'(Z/tau) ≈ 0 freezes ALL edge gradients — the
            // reconstruction term can then never pull the true edges back and the
            // posterior collapses to the empty graph. Clamping keeps the gate
            // responsive (standard logit clamping for Gumbel-sigmoid relaxations)
            // while sigmoid(±4) ≈ 0.018/0.982 still saturates the edge decision.
>>>>>>> origin/master
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
<<<<<<< HEAD
                    double sv = NumOps.ToDouble(Z[i, j]) / finalTau;
                    double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                    Pouter[i, j] = NumOps.FromDouble(sigVal);
||||||| 0d65f659c
                    Z[i, j] = NumOps.Subtract(Z[i, j], NumOps.Multiply(lr, gradZ[i, j]));
                    mu[i, j] = NumOps.Subtract(mu[i, j], NumOps.Multiply(lr, gradMu[i, j]));
                    logvar[i, j] = NumOps.Subtract(logvar[i, j],
                        NumOps.Multiply(NumOps.FromDouble(_learningRate * 0.1), gradLogvar[i, j]));
=======
                    Z[i, j] = NumOps.Subtract(Z[i, j], NumOps.Multiply(lr, gradZ[i, j]));
                    double zClamped = Math.Max(-4.0, Math.Min(4.0, NumOps.ToDouble(Z[i, j])));
                    Z[i, j] = NumOps.FromDouble(zClamped);
                    mu[i, j] = NumOps.Subtract(mu[i, j], NumOps.Multiply(lr, gradMu[i, j]));
                    logvar[i, j] = NumOps.Subtract(logvar[i, j],
                        NumOps.Multiply(NumOps.FromDouble(_learningRate * 0.1), gradLogvar[i, j]));
>>>>>>> origin/master
                }
            var Psq = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    Psq[i, j] = NumOps.Multiply(Pouter[i, j], Pouter[i, j]);
            var expPsq = MatrixExponentialTaylor(Psq, d);
            T hOuterT = NumOps.Zero;
            for (int i = 0; i < d; i++)
                hOuterT = NumOps.Add(hOuterT, expPsq[i, i]);
            hOuterT = NumOps.Subtract(hOuterT, NumOps.FromDouble(d));
            double hOuter = NumOps.ToDouble(hOuterT);

<<<<<<< HEAD
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hOuterT));
            if (hOuter > RhoIncreaseThreshold * prevH)
||||||| 0d65f659c
            // Update augmented Lagrangian with NOTEARS h(P) = tr(exp(P∘P)) - d
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hValCurrent));
            T rhoMaxT = NumOps.FromDouble(RhoMax);
            if (NumOps.GreaterThan(hValCurrent, NumOps.FromDouble(RhoIncreaseThreshold)))
=======
            // Dual ascent once per OUTER round (every innerSteps gradient steps),
            // per the NOTEARS / BCD-Nets augmented-Lagrangian schedule:
            //   alpha <- alpha + rho * h
            //   rho   <- 10 * rho   only if h did not shrink to <= 0.25 * h_prev
            if ((iter + 1) % innerSteps == 0)
>>>>>>> origin/master
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
            prevH = hOuter;

            if (NumOps.ToDouble(rho) >= RhoMax) break;
            if (hOuter < 1e-8) break;
        }

        // Final output: extract the MAP DAG from the variational posterior.
        // Strict threshold: P[i,j] > 0.5 ⟺ Z[i,j] > 0 at low tau. On
        // deterministic / zero-noise data the Gumbel-Sigmoid posterior can
        // saturate with every Z slightly negative — the algorithm cannot drive
        // any single Z to the positive side because the acyclicity penalty
        // distributes evenly across the symmetric reverse-edge pairs. In that
        // case, fall back to the per-direction posterior-mode ordering:
        // for each i<j, emit whichever of (i→j, j→i) has higher P (= less-
        // negative Z), provided |mu| exceeds the weight threshold. This is
        // still the MAP DAG under the variational posterior — just resolved
        // via direction-comparison rather than absolute-threshold.
        var result = new Matrix<T>(d, d);
        T edgeThreshold = NumOps.FromDouble(0.5);
        T weightThreshold = NumOps.FromDouble(0.1);

        // First pass: standard MAP extraction (P > 0.5, |mu| > threshold).
        bool anyEdge = false;
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
                    {
                        result[i, j] = weight;
                        anyEdge = true;
                    }
                }
            }

        // Fallback: posterior-mode ordering when the strict threshold rejects
        // every edge (deterministic-data regime described above).
        if (!anyEdge)
        {
            for (int i = 0; i < d; i++)
                for (int j = i + 1; j < d; j++)
                {
                    double zij = NumOps.ToDouble(Z[i, j]);
                    double zji = NumOps.ToDouble(Z[j, i]);
                    int srcIdx = zij > zji ? i : j;
                    int dstIdx = zij > zji ? j : i;
                    T weight = mu[srcIdx, dstIdx];
                    if (NumOps.GreaterThan(NumOps.Abs(weight), weightThreshold))
                        result[srcIdx, dstIdx] = weight;
                }
        }

        return result;
    }

    /// <summary>
    /// Zero-mean unit-variance column standardisation. Identical scheme to
    /// <see cref="AiDotNet.CausalDiscovery.ContinuousOptimization.ContinuousOptimizationBase{T}"/>'s
    /// helper — replicated here because BCDNets inherits from
    /// <see cref="BayesianCausalBase{T}"/>, not the continuous-optimisation base,
    /// and the augmented-Lagrangian schedule needs the same numerical regime to
    /// converge.
    /// </summary>
    private Matrix<T> StandardiseColumns(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var result = new Matrix<T>(n, d);
        T nT = NumOps.FromDouble(n);

        for (int j = 0; j < d; j++)
        {
            T mean = NumOps.Zero;
            for (int i = 0; i < n; i++)
                mean = NumOps.Add(mean, data[i, j]);
            mean = NumOps.Divide(mean, nT);

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, nT);
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-15)));

            for (int i = 0; i < n; i++)
                result[i, j] = NumOps.Divide(NumOps.Subtract(data[i, j], mean), std);
        }

        return result;
    }
}
