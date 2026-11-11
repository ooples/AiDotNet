using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// DECI — Deep End-to-end Causal Inference.
/// </summary>
/// <remarks>
/// <para>
/// DECI is a flow-based variational inference method that jointly learns the causal graph
/// and the functional relationships between variables. It uses normalizing flows to model
/// flexible conditional distributions and a variational distribution over DAGs.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize edge logits Z[i,j] and per-variable MLPs (location networks)</item>
/// <item>Sample soft adjacency: A[i,j] = sigmoid(Z[i,j] / tau) with temperature annealing</item>
/// <item>For each target j: predict location f_j = MLP_j(masked input by A[:,j])</item>
/// <item>Compute log-likelihood using Gaussian noise model: log p(x_j | pa(j))</item>
/// <item>Add KL divergence for edge distribution and NOTEARS acyclicity penalty</item>
/// <item>Optimize ELBO via gradient descent on Z and MLP weights</item>
/// <item>Threshold final edge probabilities to get DAG</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> DECI simultaneously learns "which variables cause which" and
/// "how they cause each other." It's particularly good at handling complex, non-standard
/// relationships and can also estimate intervention effects.
/// </para>
/// <para>
/// Reference: Geffner et al. (2022), "Deep End-to-end Causal Inference", arXiv.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Deep End-to-end Causal Inference", "https://arxiv.org/abs/2202.02195", Year = 2022, Authors = "Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang")]
public class DECIAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "DECI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public DECIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var cov = ComputeCovarianceMatrix(data);
        var x = Tensor<T>.FromMatrix(data);
        var offDiagonal = OffDiagonalMask(d);

        // Edge logits Z[i,j] for variational distribution over graph
        var Z = new Tensor<T>(new[] { d, d });

        // Per-variable location networks, stacked: W1 [j, i, k] (d x h per j), W2 [j, k] (h per j)
        var (W1, W2, selfInput) = InitializeLocationNetworks(rng, d, h);

        // Per-variable log-variance for Gaussian noise model
        var logVar = new Tensor<T>(new[] { d });

        var duals = (Alpha: NumOps.Zero, Rho: NumOps.One);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            double tau = Math.Max(0.1, 1.0 * Math.Pow(0.95, epoch));
            T hVal;
            (Z, W1, W2, logVar, hVal) = DeciStep(x, Z, W1, W2, logVar, selfInput, offDiagonal, duals, tau, n, d, h);

            // Update augmented Lagrangian with NOTEARS h(P) = tr(exp(P∘P)) - d
            duals = UpdateDeciDuals(duals, hVal);
        }

        // Final output: threshold edge probabilities and compute OLS weights
        var c = Tensor<T>.FromMatrix(cov);
        var variance = Engine.TensorBroadcastTo(Engine.TensorDiagonal(c).Reshape(new[] { d, 1 }), new[] { d, d });
        var usable = Engine.TensorMultiply(offDiagonal, Engine.TensorGreaterThan(variance, NumOps.FromDouble(1e-10)));
        var weight = Engine.TensorDivide(c, Engine.TensorWhere(usable, variance, Engine.TensorOnesLike(variance)));
        var magnitude = Engine.TensorAbs(weight);

        // Include the neutral 0.5 posterior boundary.  DECI initializes edge logits
        // at zero (P=0.5); using a strict comparison discarded every candidate when
        // optimization left a logit exactly at that neutral value, even though the
        // covariance/OLS weight below could still establish a valid edge.
        var prob = ClampedSigmoid(Z, 0.1);
        var posterior = Engine.TensorMultiply(Engine.TensorMultiply(usable,
            Engine.TensorGreaterEqual(prob, Engine.TensorFullLike(prob, NumOps.FromDouble(0.5)))),
            Engine.TensorGreaterThan(magnitude, NumOps.FromDouble(0.1)));
        bool selectedPosteriorEdge = Engine.TensorCountNonzero(posterior) > 0;
        if (selectedPosteriorEdge)
            return Engine.TensorWhere(posterior, weight, Engine.TensorZerosLike(weight)).ToMatrix();

        // A neutral/under-trained posterior can leave every probability below the
        // 0.5 gate even when the observed covariance contains a clear linear edge.
        // Preserve the posterior-first DECI path, but use the model's documented
        // covariance/OLS fallback when it selected nothing; returning an empty graph
        // made the algorithm silently fail on strong, low-dimensional signals.
        return CovarianceFallback(c, weight, usable, d);
    }

    /// <summary>
    /// The covariance/OLS fallback: edges with |cov[i,j]/cov[i,i]| &gt; EdgeThreshold, each unordered
    /// pair oriented from the higher-variance variable, then projected onto a DAG strongest
    /// |correlation| first.
    /// </summary>
    /// <remarks>
    /// The covariance fallback is deliberately conservative about whether an
    /// edge exists, but covariance is symmetric and can produce reciprocal or
    /// cyclic directions. Project its candidates onto a DAG by retaining the
    /// strongest weights first and rejecting an edge that would close a path.
    ///
    /// ORIENTATION IS DECIDED BEFORE RANKING, not by the sort. The strength below is a
    /// correlation, and correlation is symmetric: cov[i,j] == cov[j,i] and the denominator
    /// uses both variances, so (i, j) and (j, i) always scored identically. List&lt;T&gt;.Sort is
    /// introsort, which is not stable, so which of the two tied entries survived depended on
    /// the internal partitioning -- DECI picked edge direction by an implementation detail
    /// that varies with list length and runtime version, not by evidence. Each unordered pair
    /// is now considered once and oriented from the higher-variance variable to the
    /// lower-variance one: the additive-noise attenuation direction, the same rule
    /// CausalVAEAlgorithm's fallback uses. Equal variances fall back to index order purely so
    /// the result is total and repeatable.
    ///
    /// Candidates are ranked by symmetric correlation rather than the directional OLS
    /// coefficient. The latter can exaggerate a transitive edge when the source variance is
    /// small (e.g. X0->X1->X2), causing the projection to retain X0->X2. Equal strengths keep
    /// (From, To) order so the projection cannot depend on sort partitioning.
    /// </remarks>
    private Matrix<T> CovarianceFallback(Tensor<T> c, Tensor<T> weight, Tensor<T> usable, int d)
    {
        var shape = new[] { d, d };
        var zeros = Engine.TensorZerosLike(c);
        var significant = Engine.TensorMultiply(usable,
            Engine.TensorGreaterThan(Engine.TensorAbs(weight), NumOps.FromDouble(EdgeThreshold)));
        var fallback = Engine.TensorWhere(significant, weight, zeros);

        // For i < j: from = i when var_i > var_j or |var_i - var_j| < 1e-12, otherwise j.
        var diagonal = Engine.TensorDiagonal(c);
        var varI = Engine.TensorBroadcastTo(diagonal.Reshape(new[] { d, 1 }), shape);
        var varJ = Engine.TensorBroadcastTo(diagonal.Reshape(new[] { 1, d }), shape);
        var nearlyEqual = Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtract(varI, varJ)), NumOps.FromDouble(1e-12));
        var forward = Engine.TensorSign(Engine.TensorAdd(Engine.TensorGreaterThan(varI, varJ), nearlyEqual));
        var upper = Engine.TensorTriu(Engine.TensorOnesLike(c), 1);
        var oriented = Engine.TensorAdd(Engine.TensorMultiply(upper, forward),
            Engine.TensorTranspose(Engine.TensorMultiply(upper, Engine.ScalarMinusTensor(NumOps.One, forward))));
        var candidates = Engine.TensorWhere(oriented, fallback, zeros);

        // |cov[i,j]| / sqrt(|var_i var_j|) when the denominator exceeds 1e-12, else 0.
        var denom = Engine.TensorSqrt(Engine.TensorAbs(Engine.TensorMultiply(varI, varJ)));
        var denomUsable = Engine.TensorGreaterThan(denom, NumOps.FromDouble(1e-12));
        var correlation = Engine.TensorWhere(denomUsable,
            Engine.TensorDivide(Engine.TensorAbs(c), Engine.TensorWhere(denomUsable, denom, Engine.TensorOnesLike(denom))), zeros);

        return ProjectToAcyclic(candidates.ToMatrix(), correlation);
    }

    /// <summary>
    /// Seeded initialization in the reference draw order: for each network j, W1[j][i, k] for i != j
    /// (k inner), then W2[j][k]; every value is sqrt(2/d) * (u - 0.5). Also returns the [j, i, k]
    /// mask that is one on each network's own input row.
    /// </summary>
    private (Tensor<T> W1, Tensor<T> W2, Tensor<T> SelfInput) InitializeLocationNetworks(Random rng, int d, int h)
    {
        int D = d * h;
        var draws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, d, D), NumOps.FromDouble(-0.5)),
            NumOps.FromDouble(Math.Sqrt(2.0 / d)));

        // W1[j, i, k] = draw[j, slot(i)*h + k], slot skipping i == j.
        var source = new int[d * D];
        for (int j = 0; j < d; j++)
            for (int i = 0; i < d; i++)
                for (int k = 0; k < h; k++)
                    source[(j * d + i) * h + k] = j * D + (i == j ? 0 : ((i < j ? i : i - 1) * h + k));

        var selfInput = Engine.TensorBroadcastTo(Engine.TensorEye<T>(d).Reshape(new[] { d, d, 1 }), new[] { d, d, h });
        var w1 = Engine.TensorIndexSelect(draws.Reshape(new[] { d * D }), new Tensor<int>(source, new[] { d * D }), 0)
            .Reshape(new[] { d, d, h });
        w1 = Engine.TensorWhere(selfInput, Engine.TensorZerosLike(w1), w1);
        var w2 = Engine.TensorSlice(draws, new[] { 0, (d - 1) * h }, new[] { d, h });
        return (w1, w2, selfInput);
    }

    /// <summary>
    /// One full-batch DECI step: Gaussian likelihood gradients of the masked location networks
    /// (input i into network j scaled by P[i,j] = sigmoid(Z[i,j]/tau)), plus the KL and augmented-
    /// Lagrangian acyclicity gradients on the edge logits. Returns the updated parameters and h(P).
    /// </summary>
    private (Tensor<T> Z, Tensor<T> W1, Tensor<T> W2, Tensor<T> LogVar, T H) DeciStep(Tensor<T> x, Tensor<T> Z,
        Tensor<T> W1, Tensor<T> W2, Tensor<T> logVar, Tensor<T> selfInput, Tensor<T> offDiagonal,
        (T Alpha, T Rho) duals, double tau, int n, int d, int h)
    {
        var unitShape = new[] { n, d, h };
        var weightShape = new[] { d, d, h };
        T tauT = NumOps.FromDouble(tau);
        T eps = NumOps.FromDouble(1e-10);
        T invN = NumOps.FromDouble(1.0 / n);

        // Compute edge probabilities
        var P = Engine.TensorMultiply(offDiagonal, ClampedSigmoid(Z, tau));
        var pPerUnit = Engine.TensorBroadcastTo(Engine.TensorTranspose(P).Reshape(new[] { d, d, 1 }), weightShape); // [j,i,k] = P[i,j]

        // Masked input x_masked[i] = x[i] * P[i,j] into network j: pre[s,j,k] = sum_i x[s,i] P[i,j] W1[j,i,k]
        var effective = Engine.TensorMultiply(pPerUnit, W1);
        var effectiveByInput = Engine.Reshape(Engine.TensorPermute(effective, new[] { 1, 0, 2 }), new[] { d, d * h });
        var hidden = ClampedSigmoid(Engine.TensorMatMul(x, effectiveByInput), 1.0).Reshape(unitShape);
        var w2PerSample = Engine.TensorBroadcastTo(W2.Reshape(new[] { 1, d, h }), unitShape);
        var pred = Engine.ReduceSum(Engine.TensorMultiply(hidden, w2PerSample), new[] { 2 }, false); // [n,d]

        // Gaussian log-likelihood gradient: d/d_mu [ -0.5 * (x-mu)^2 / var ]
        var invVar = Engine.TensorReciprocal(Engine.TensorAddScalar(Engine.TensorExp(logVar), eps)); // [d]
        var invVarPerSample = Engine.TensorBroadcastTo(invVar.Reshape(new[] { 1, d }), new[] { n, d });
        var residual = Engine.TensorSubtract(pred, x);
        var scaledResidual = Engine.TensorMultiplyScalar(Engine.TensorMultiply(residual, invVarPerSample), invN);

        // Log-variance gradient of NLL: d/d_logvar [ 0.5*logvar + 0.5*(x-mu)^2*exp(-logvar) ]
        // = 0.5 - 0.5*(x-mu)^2/var
        var perSample = Engine.ScalarMinusTensor(NumOps.FromDouble(0.5), Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorSquare(residual), invVarPerSample), NumOps.FromDouble(0.5)));
        var gLogVar = Engine.ReduceSum(Engine.TensorMultiplyScalar(perSample, invN), new[] { 0 }, false);

        // Backprop through MLP
        var scaledPerUnit = Engine.TensorBroadcastTo(scaledResidual.Reshape(new[] { n, d, 1 }), unitShape);
        var gW2 = Engine.ReduceSum(Engine.TensorMultiply(scaledPerUnit, hidden), new[] { 0 }, false);
        var sigD = Engine.TensorMultiply(hidden, Engine.ScalarMinusTensor(NumOps.One, hidden));
        var dH = Engine.TensorMultiply(Engine.TensorMultiply(scaledPerUnit, w2PerSample), sigD);

        // xdH[j,i,k] = sum_s x[s,i] dH[s,j,k]
        var xdH = Engine.Reshape(Engine.TensorPermute(
            Engine.TensorMatMul(Engine.TensorTranspose(x), dH.Reshape(new[] { n, d * h })).Reshape(weightShape),
            new[] { 1, 0, 2 }), weightShape);
        var gW1 = Engine.TensorMultiply(xdH, pPerUnit);

        // Gradient w.r.t. edge probability: sum_k xdH[j,i,k] W1[j,i,k] * P(1-P) / tau, as [i,j]
        var pSigD = Engine.TensorMultiply(P, Engine.ScalarMinusTensor(NumOps.One, P));
        var dMask = Engine.TensorTranspose(Engine.ReduceSum(Engine.TensorMultiply(xdH, W1), new[] { 2 }, false));
        var gZ = Engine.TensorDivideScalar(Engine.TensorMultiply(dMask, pSigD), tauT);

        // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
        var expPSq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(P, P).ToMatrix(), d));
        T hVal = NumOps.Subtract(Engine.TensorTrace(expPSq), NumOps.FromDouble(d));

        // KL divergence + acyclicity gradients on edge logits:
        // KL(Bernoulli(p) || Bernoulli(prior)) where prior is small -> lambda1 * p (sparsity prior 0.1);
        // acyclicity (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
        var klGrad = Engine.TensorMultiplyScalar(P, NumOps.FromDouble(0.1));
        var acycGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(expPSq), Engine.TensorMultiplyScalar(P, NumOps.FromDouble(2))),
            NumOps.Add(duals.Alpha, NumOps.Multiply(duals.Rho, hVal)));
        var priorSigD = Engine.TensorDivideScalar(pSigD, NumOps.Add(tauT, eps));
        gZ = Engine.TensorAdd(gZ, Engine.TensorMultiply(offDiagonal,
            Engine.TensorMultiply(Engine.TensorAdd(klGrad, acycGrad), priorSigD)));

        // Apply gradients; the diagonal logits and each network's own input row stay fixed at zero.
        T lr = NumOps.FromDouble(LearningRate);
        var newZ = Engine.TensorWhere(offDiagonal, Engine.TensorSubtract(Z, Engine.TensorMultiplyScalar(gZ, lr)), Z);
        var newW1 = Engine.TensorSubtract(W1, Engine.TensorMultiplyScalar(gW1, lr));
        newW1 = Engine.TensorWhere(selfInput, Engine.TensorZerosLike(newW1), newW1);
        var newW2 = Engine.TensorSubtract(W2, Engine.TensorMultiplyScalar(gW2, lr));
        var newLogVar = Engine.TensorSubtract(logVar, Engine.TensorMultiplyScalar(gLogVar, NumOps.FromDouble(LearningRate * 0.1)));
        return (newZ, newW1, newW2, newLogVar, hVal);
    }

    /// <summary>alpha += rho*h; rho *= 10 while h &gt; 0.25 and rho has not passed 1e16.</summary>
    private (T Alpha, T Rho) UpdateDeciDuals((T Alpha, T Rho) duals, T hVal)
    {
        T alpha = NumOps.Add(duals.Alpha, NumOps.Multiply(duals.Rho, hVal));
        T rho = duals.Rho;
        T rhoMax = NumOps.FromDouble(1e+16);
        if (NumOps.GreaterThan(hVal, NumOps.FromDouble(0.25)) && !NumOps.GreaterThan(rho, rhoMax))
            rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
        return (alpha, rho);
    }

}
