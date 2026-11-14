using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// AVICI — Amortized Variational Inference for Causal Discovery.
/// </summary>
/// <remarks>
/// <para>
/// AVICI uses a transformer-inspired architecture to perform causal discovery. The model
/// processes data via self-attention over variable pairs to produce edge probabilities.
/// It computes scaled dot-product attention between variable-pair representations derived
/// from sufficient statistics, then refines edge logits through multiple attention layers.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Compute pairwise features from data: covariance, correlation, variances</item>
/// <item>Project features to query/key/value via learned matrices</item>
/// <item>Apply multi-head self-attention over variable pairs</item>
/// <item>Output layer produces edge logits from attended representations</item>
/// <item>Apply NOTEARS acyclicity constraint via augmented Lagrangian</item>
/// <item>Threshold edge probabilities for final DAG</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> AVICI is like a "universal causal discovery engine" powered by a
/// transformer (similar to ChatGPT's architecture). It processes data statistics through
/// attention mechanisms to recognize causal patterns.
/// </para>
/// <para>
/// Reference: Lorch et al. (2023), "Amortized Inference for Causal Structure Learning", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Amortized Inference for Causal Structure Learning", "https://arxiv.org/abs/2205.12934", Year = 2023, Authors = "Lars Lorch, Scott Sussex, Jonas Rothfuss, Andreas Krause, Bernhard Scholkopf")]
public class AVICIAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "AVICI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public AVICIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        // The base defaults (lr 1e-3, 100 epochs) are far too small for the attention weights to move:
        // lr × the 1/(d(d-1)) gradient normalization over 100 epochs leaves edge probabilities essentially
        // at their initialization, so strong linear edges never cross the detection threshold. Use a larger
        // step and more epochs so the data fit actually converges P toward the correlations (still
        // overridable via options).
        LearningRate = 0.05;
        MaxEpochs = 400;
        ApplyDeepOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int headDim = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Standardize columns so the discovered structure is invariant to per-variable scaling and the
        // covariance/variance features below are on a consistent unit scale (cov becomes correlation).
        data = StandardizeColumns(data);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var cov = ComputeCovarianceMatrix(data);
        var corr = CovarianceToCorrelation(cov);

        // Feature dimension per edge pair: 4 features
        int featDim = 4;
        T initScale = NumOps.FromDouble(Math.Sqrt(2.0 / featDim));

        // Query, Key, Value projection matrices (featDim x headDim), drawn interleaved per (f, k),
        // then the output projection (headDim x 1).
        var qkvDraws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, featDim, headDim, 3), NumOps.FromDouble(-0.5)), initScale);
        var Wq = Engine.TensorSliceAxis(qkvDraws, 2, 0);
        var Wk = Engine.TensorSliceAxis(qkvDraws, 2, 1);
        var Wv = Engine.TensorSliceAxis(qkvDraws, 2, 2);
        var Wo = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, headDim, 1), NumOps.FromDouble(-0.5)), initScale);

        // Acyclicity warm-up: start with NO NOTEARS penalty (rho = 0). The first half of training is a
        // pure data fit so edge probabilities can converge toward the (standardized) correlation signal;
        // the augmented-Lagrangian penalty is only switched on for the second half (see the rho update at
        // end of epoch). Starting at rho = 1 and ramping ×10 on the initial near-complete graph made the
        // penalty dominate immediately and saturate every edge — including the true ones — to 0.
        T rho = NumOps.Zero;
        T boundedRho = NumOps.One;

        // Precompute features per variable pair (row i*d + j): cov[i,j], corr[i,j], cov[i,i], cov[j,j],
        // regrouped by target as [j, i, feature] for the attention over pairs sharing target j.
        var c = Tensor<T>.FromMatrix(cov);
        var r = Tensor<T>.FromMatrix(corr);
        var variance = Engine.TensorDiagonal(c);
        var shape = new[] { d, d };
        var features = Engine.TensorStack(new[]
        {
            c.Reshape(new[] { d * d }),
            r.Reshape(new[] { d * d }),
            Engine.TensorBroadcastTo(variance.Reshape(new[] { d, 1 }), shape).Reshape(new[] { d * d }),
            Engine.TensorBroadcastTo(variance.Reshape(new[] { 1, d }), shape).Reshape(new[] { d * d }),
        }, 1);
        var byTarget = Engine.Reshape(Engine.TensorPermute(features.Reshape(new[] { d, d, featDim }), new[] { 1, 0, 2 }),
            new[] { d * d, featDim });
        var offDiagonal = OffDiagonalMask(d);
        var absCorr = Engine.TensorAbs(r);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            (Wq, Wk, Wv, Wo) = AttentionStep(byTarget, Wq, Wk, Wv, Wo, absCorr, offDiagonal, rho, d, headDim);

            // Apply only a BOUNDED acyclicity penalty in the second half of training (warm-up; see the rho
            // initialization). The original schedule grew rho ×10 toward MaxPenaltyValue (1e16) AND
            // accumulated alpha every epoch — on the near-complete graph produced by strongly correlated
            // data that made the augmented-Lagrangian term dominate the data fit and drove EVERY edge
            // logit below -20, i.e. it collapsed the output to the empty graph (trivially acyclic) and
            // recovered no edges. A fixed, modest rho (no ×10 ramp, no alpha runaway) breaks ties toward a
            // DAG without overwhelming the data fit; the final orientation is resolved by BuildFinalAdjacency.
            if (epoch >= MaxEpochs / 2)
                rho = boundedRho;
        }

        // Final inference using trained parameters
        var final = AttendPairs(byTarget, Wq, Wk, Wv, Wo, offDiagonal, d, headDim);
        return BuildFinalAdjacency(final.P, cov, d);
    }

    /// <summary>Forward pass of the pair attention, with every [j, i, ·] intermediate kept for backprop.</summary>
    private sealed class PairAttention
    {
        public PairAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T> weights, Tensor<T> attended, Tensor<T> p)
        {
            Q = q;
            K = k;
            V = v;
            Weights = weights;
            Attended = attended;
            P = p;
        }

        /// <summary>Queries, keys, values [j, i, headDim] of pair (i, j).</summary>
        public Tensor<T> Q { get; }
        public Tensor<T> K { get; }
        public Tensor<T> V { get; }

        /// <summary>Attention weights [j, query i, key i].</summary>
        public Tensor<T> Weights { get; }

        /// <summary>Attended representation [j, i, headDim].</summary>
        public Tensor<T> Attended { get; }

        /// <summary>Edge probabilities [i, j], zero diagonal.</summary>
        public Tensor<T> P { get; }
    }

    /// <summary>
    /// Self-attention over the pairs sharing each target j: scores Q K^T / sqrt(headDim), a softmax
    /// shifted by max(score, -1e10) with exponents capped at 20 and denominator sum + 1e-10, then
    /// P[i,j] = sigmoid(attended · Wo) saturated beyond ±20.
    /// </summary>
    private PairAttention AttendPairs(Tensor<T> byTarget, Tensor<T> Wq, Tensor<T> Wk, Tensor<T> Wv, Tensor<T> Wo,
        Tensor<T> offDiagonal, int d, int headDim)
    {
        var pairShape = new[] { d, d, headDim };
        var q = Engine.TensorMatMul(byTarget, Wq).Reshape(pairShape);
        var k = Engine.TensorMatMul(byTarget, Wk).Reshape(pairShape);
        var v = Engine.TensorMatMul(byTarget, Wv).Reshape(pairShape);

        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));                           // [j, qi, ki]
        var scoreShape = new[] { d, d, d };
        var maxScore = Engine.TensorMax(Engine.ReduceMax(scores, new[] { 2 }, true), NumOps.FromDouble(-1e10));
        var shifted = Engine.TensorSubtract(scores, Engine.TensorBroadcastTo(maxScore, scoreShape));
        var cap = Engine.TensorFullLike(shifted, NumOps.FromDouble(20));
        var expScores = Engine.TensorExp(Engine.TensorWhere(Engine.TensorGreaterThan(shifted, cap), cap, shifted));
        var sumExp = Engine.TensorAddScalar(Engine.ReduceSum(expScores, new[] { 2 }, true), NumOps.FromDouble(1e-10));
        var weights = Engine.TensorDivide(expScores, Engine.TensorBroadcastTo(sumExp, scoreShape));

        // Weighted sum of values, then edge logits from the attended representation
        var attended = Engine.TensorMatMul(weights, v);                              // [j, i, headDim]
        var logits = Engine.TensorMatMul(attended.Reshape(new[] { d * d, headDim }), Wo).Reshape(new[] { d, d }); // [j, i]
        var p = Engine.TensorMultiply(offDiagonal, ClampedSigmoid(Engine.TensorTranspose(logits), 1.0));
        return new PairAttention(q, k, v, weights, attended, p);
    }

    /// <summary>
    /// One training step: data fit P - |corr| plus the bounded NOTEARS gradient rho·h·exp(P∘P)^T∘2P
    /// drives Wo and (through the attention weights) Wv; Wo is applied first, and the Wq / Wk
    /// gradients, which use the proxy acyclicity term rho·P·2P and the diagonal softmax Jacobian,
    /// are taken through the updated Wo. All gradients are averaged over the d(d-1) pairs.
    /// </summary>
    private (Tensor<T> Wq, Tensor<T> Wk, Tensor<T> Wv, Tensor<T> Wo) AttentionStep(Tensor<T> byTarget, Tensor<T> Wq,
        Tensor<T> Wk, Tensor<T> Wv, Tensor<T> Wo, Tensor<T> absCorr, Tensor<T> offDiagonal, T rho, int d, int headDim)
    {
        var a = AttendPairs(byTarget, Wq, Wk, Wv, Wo, offDiagonal, d, headDim);
        var P = a.P;
        T lr = NumOps.FromDouble(LearningRate);
        T normFactor = NumOps.FromDouble(1.0 / (d * (d - 1)));
        T two = NumOps.FromDouble(2);

        // Compute NOTEARS acyclicity constraint and gradient: h(P) = tr(exp(P∘P)) - d
        var expPSq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(P, P).ToMatrix(), d));
        T hCurrent = NumOps.Subtract(Engine.TensorTrace(expPSq), NumOps.FromDouble(d));
        // Bounded NOTEARS penalty coefficient: rho*h (no augmented-Lagrangian alpha term — this
        // schedule uses a fixed rho, not the alpha-accumulating dual ascent that collapsed edges).
        T augCoeff = NumOps.Multiply(rho, hCurrent);

        var dataGrad = Engine.TensorSubtract(P, absCorr);
        var sigDeriv = Engine.TensorMultiply(P, Engine.ScalarMinusTensor(NumOps.One, P));
        var acycGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(expPSq), Engine.TensorMultiplyScalar(P, two)), augCoeff);
        var dLogit = Engine.TensorMultiply(offDiagonal, Engine.TensorMultiply(Engine.TensorAdd(dataGrad, acycGrad), sigDeriv));
        var dLogitByTarget = Engine.TensorTranspose(dLogit);                         // [j, i]

        // Gradient w.r.t. Wo
        var gWo = Engine.TensorMatMul(Engine.TensorTranspose(a.Attended.Reshape(new[] { d * d, headDim })),
            dLogitByTarget.Reshape(new[] { d * d, 1 }));

        // Simplified gradient through attention to Wv (dominant term):
        // gWv[f,k] = Wo[k] * sum_{j,ki} (sum_i dLogit[i,j] attn[j,i,ki]) features[(ki,j), f]
        var keyWeight = Engine.TensorMatMul(dLogitByTarget.Reshape(new[] { d, 1, d }), a.Weights).Reshape(new[] { 1, d * d });
        var featureWeight = Engine.TensorMatMul(keyWeight, byTarget);                 // [1, featDim]
        var gWv = Engine.TensorMatMul(Engine.TensorTranspose(featureWeight), Engine.TensorTranspose(Wo));

        // Normalize and apply the Wo gradient first
        var newWo = Engine.TensorSubtract(Wo, Engine.TensorMultiplyScalar(Engine.TensorMultiplyScalar(gWo, normFactor), lr));

        // Gradient w.r.t. Wq and Wk through attention scores.
        // NOTEARS acyclicity gradient: ρ·h(W) · ∂h/∂W_ij
        // where ∂h/∂W_ij = 2·W_ij·[e^{W⊙W}]_ji (Zheng et al. 2018)
        // Approximation: use 2·pij as proxy for ∂h/∂W_ij (ignores matrix exponential).
        // Consistent with the main path's bounded penalty: (rho * pij) * 2 * pij
        // (fixed rho, no augmented-Lagrangian alpha term).
        var acycGrad2 = Engine.TensorMultiply(Engine.TensorMultiplyScalar(P, rho), Engine.TensorMultiplyScalar(P, two));
        var dLogit2 = Engine.TensorMultiply(offDiagonal, Engine.TensorMultiply(Engine.TensorAdd(dataGrad, acycGrad2), sigDeriv));
        var dLogit2ByTarget = Engine.TensorTranspose(dLogit2);

        // d(attended)/d(attnWeight[ki]) = V[ki]; dAttnW[j,i,ki] = dLogit2[i,j] * (V[j,ki,:] · Wo)
        var valueScore = Engine.TensorMatMul(a.V.Reshape(new[] { d * d, headDim }), newWo).Reshape(new[] { d, 1, d });
        var scoreShape = new[] { d, d, d };
        var dAttnW = Engine.TensorMultiply(Engine.TensorBroadcastTo(dLogit2ByTarget.Reshape(new[] { d, d, 1 }), scoreShape),
            Engine.TensorBroadcastTo(valueScore, scoreShape));
        // softmax jacobian (simplified): attn * (1-attn) for diagonal term, times the score scale
        var dScore = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(dAttnW, Engine.TensorMultiply(a.Weights, Engine.ScalarMinusTensor(NumOps.One, a.Weights))),
            NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));

        // dScore/dQ[h] = K[h], dScore/dK[h] = Q[h]
        var queryGrad = Engine.TensorMatMul(dScore, a.K).Reshape(new[] { d * d, headDim });
        var keyGrad = Engine.TensorMatMul(Engine.TensorPermute(dScore, new[] { 0, 2, 1 }), a.Q).Reshape(new[] { d * d, headDim });
        var featuresT = Engine.TensorTranspose(byTarget);
        var gWq = Engine.TensorMatMul(featuresT, queryGrad);
        var gWk = Engine.TensorMatMul(featuresT, keyGrad);

        return (
            Engine.TensorSubtract(Wq, Engine.TensorMultiplyScalar(Engine.TensorMultiplyScalar(gWq, normFactor), lr)),
            Engine.TensorSubtract(Wk, Engine.TensorMultiplyScalar(Engine.TensorMultiplyScalar(gWk, normFactor), lr)),
            Engine.TensorSubtract(Wv, Engine.TensorMultiplyScalar(Engine.TensorMultiplyScalar(gWv, normFactor), lr)),
            newWo);
    }

}
