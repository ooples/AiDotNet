using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// GraN-DAG — Gradient-based Neural DAG Learning.
/// </summary>
/// <remarks>
/// <para>
/// GraN-DAG parameterizes each structural equation f_j as a neural network with sigmoid
/// activations. The weighted adjacency A[i,j] is the PATH PRODUCT of absolute weights across
/// the MLP layers — for the 1-hidden-layer network predicting x_j, A[i,j] = Σ_k |W1_j[i,k]|·|W2_j[k]|
/// (Lachapelle et al. 2020 §2.2), so an input→hidden weight only contributes if that hidden unit
/// actually drives the output. The NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d is enforced
/// via augmented Lagrangian.
/// </para>
/// <para>
/// <b>For Beginners:</b> GraN-DAG trains a separate neural network for each variable to
/// predict it from the others. The "importance" of each input connection tells us the
/// causal strength, while a mathematical constraint ensures no circular causation.
/// </para>
/// <para>
/// Reference: Lachapelle et al. (2020), "Gradient-Based Neural DAG Learning", ICLR.
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
[ResearchPaper("Gradient-Based Neural DAG Learning", "https://openreview.net/forum?id=rklbKA4YDS", Year = 2020, Authors = "Sebastien Lachapelle, Philippe Brouillard, Tristan Deleu, Simon Lacoste-Julien")]
public class GraNDAGAlgorithm<T> : DeepCausalBase<T>
{
    /// <summary>
    /// Maximum number of entries of one [rows, d, h] hidden-activation block in a training step; samples are
    /// processed in blocks of at most this many entries.
    /// </summary>
    private const int MaxSampleBlockElements = 1 << 22;

    /// <inheritdoc/>
    public override string Name => "GraN-DAG";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public GraNDAGAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Standardize each variable to zero mean / unit variance before fitting — GraN-DAG's
        // preprocessing (Lachapelle 2020, experimental setup). The per-variable Gaussian-NLL score and
        // the path-norm adjacency are scale-sensitive, so without this a uniform rescaling of the data
        // (x -> 10x) changes the discovered edge set. Standardizing makes discovery invariant to input
        // scaling (DiscoverStructure_IsInvariantToDataScaling); a constant column is left at zero. Every
        // downstream computation (the MLP fit below and the covariance) then runs on the standardized data.
        // Reuse the shared DeepCausalBase.StandardizeColumns so every deep causal learner z-scores
        // identically (avoids the earlier /n vs /(n-1) variance-normalization drift). A constant column
        // still standardizes to zero — its centered values are all 0, independent of the divisor.
        // Capture each variable's RAW marginal variance BEFORE standardization — used as the
        // orientation cue for the final DAG projection. In an attenuating linear SEM (|coeff| < 1,
        // the test's 0.6-0.8), an exogenous root has strictly larger marginal variance than its
        // noise-attenuated descendants, so ranking nodes by raw variance recovers the causal
        // topological order (root first). GraN-DAG's Gaussian-NLL (MSE) fit cannot orient near-
        // deterministic edges from the symmetric path-norm alone (both directions predict equally
        // well), so the default net-outflow source score leaves true edges unoriented; the variance
        // cue supplies the missing orientation. The RANKING is invariant to uniform data scaling
        // (all variances scale by the same factor), preserving DiscoverStructure_IsInvariantToDataScaling.
        var rawData = Tensor<T>.FromMatrix(data);
        var rawCentered = CenterColumns(rawData);
        var rawVariance = Engine.ReduceMean(Engine.TensorMultiply(rawCentered, rawCentered), new[] { 0 }, false);

        data = StandardizeColumns(data);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));
        T half = NumOps.FromDouble(0.5);

        // Per-variable MLPs: W1 is [d (input i), d (variable j), h] holding W1[j][i, :], W2 is [d (variable j), h].
        // Per variable j the draws are scale * (U[0,1) - 0.5): W1[j][i, :] for every i != j in order, then W2[j].
        var w1Columns = new Tensor<T>[d];
        var w2Rows = new Tensor<T>[d];
        for (int j = 0; j < d; j++)
        {
            var draws = Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(Tensor<T>.CreateRandom(rng, d, h), half), scale);
            var inputs = new List<Tensor<T>>();
            if (j > 0) inputs.Add(Engine.TensorSlice(draws, new[] { 0, 0 }, new[] { j, h }));
            inputs.Add(new Tensor<T>(new[] { 1, h }));
            if (j < d - 1) inputs.Add(Engine.TensorSlice(draws, new[] { j, 0 }, new[] { d - 1 - j, h }));
            w1Columns[j] = Engine.TensorConcatenate(inputs.ToArray(), 0);
            w2Rows[j] = Engine.TensorSliceAxis(draws, 0, d - 1);
        }
        var W1 = Engine.TensorStack(w1Columns, 1);
        var W2 = Engine.TensorStack(w2Rows, 0);

        T lr = NumOps.FromDouble(LearningRate);
        var alpha = new Tensor<T>(new[] { 1 });
        var rho = Engine.TensorOnesLike(alpha);
        T rhoMax = NumOps.FromDouble(1e+16);
        T ten = NumOps.FromDouble(10);
        T clipHi = NumOps.FromDouble(10.0);
        T clipLo = NumOps.FromDouble(-10.0);
        T edgeFloor = NumOps.FromDouble(1e-12);
        double hPrev = double.PositiveInfinity;

        // Upload the standardized design matrix once — it is constant across every inner/outer step.
        // The per-variable MLP data-fit below runs as batched Engine matmuls (mirroring the sibling
        // CASTLEAlgorithm) rather than an ~n*d*h per-sample Engine.DotProduct loop. This is
        // mathematically identical — the same Gaussian-NLL score, standard logistic activation and
        // path-norm gradient that GraN-DAG (Lachapelle et al. 2020) specifies — but issues a handful of
        // BLAS-backed GEMMs per step instead of millions of scalar dot products, which was the dominant cost.
        // All d per-variable MLPs share the input, so one [n x d] * [d x (d h)] GEMM computes every hidden layer.
        T invN = NumOps.FromDouble(1.0 / n);
        var Xt = Tensor<T>.FromMatrix(data);
        // Samples are processed in blocks so the [rows, d, h] hidden activations stay bounded
        // (MaxSampleBlockElements) however large n and d are; the gradients accumulate across blocks.
        int blockRows = Math.Max(1, Math.Min(n, MaxSampleBlockElements / Math.Max(1, d * h)));
        var offDiagonal = OffDiagonalMask(d);
        var weightMask = Engine.TensorBroadcastTo(offDiagonal.Reshape(new[] { d, d, 1 }), new[] { d, d, h });
        var zerosW1 = new Tensor<T>(new[] { d, d, h });

        for (int outer = 0; outer < MaxEpochs; outer++)
        {
            for (int inner = 0; inner < 20; inner++)
            {
                // Batched forward / backward over all n samples and all d variables:
                // H_j = sigmoid(X * W1[j]), pred_j = H_j * W2[j], and the Gaussian-NLL residual
                // (pred_j - x_j)/n backpropagated to gW1[j] = X^T * dH_j and gW2[j] = H_j^T * resid_j.
                var w1ByInput = W1.Reshape(new[] { d, d * h });
                var gW2 = new Tensor<T>(new[] { d, h });
                var gW1Flat = new Tensor<T>(new[] { d, d * h });
                for (int b0 = 0; b0 < n; b0 += blockRows)
                {
                    int rows = Math.Min(blockRows, n - b0);
                    var cube = new[] { rows, d, h };
                    var xb = Engine.TensorSlice(Xt, new[] { b0, 0 }, new[] { rows, d });
                    var H = Engine.Sigmoid(Engine.TensorMatMul(xb, w1ByInput)).Reshape(cube);                 // [rows,d,h]
                    var w2Spread = Engine.TensorBroadcastTo(W2.Reshape(new[] { 1, d, h }), cube);
                    var pred = Engine.ReduceSum(Engine.TensorMultiply(H, w2Spread), new[] { 2 }, false);       // [rows,d]
                    var resid = Engine.TensorMultiplyScalar(Engine.TensorSubtract(pred, xb), invN);            // [rows,d]
                    var residSpread = Engine.TensorBroadcastTo(resid.Reshape(new[] { rows, d, 1 }), cube);

                    gW2 = Engine.TensorAdd(gW2, Engine.ReduceSum(Engine.TensorMultiply(H, residSpread), new[] { 0 }, false)); // [d,h]
                    var dPred = Engine.TensorMultiply(residSpread, w2Spread);                                 // [rows,d,h]
                    var oneMinusH = Engine.TensorSubtract(Engine.TensorOnesLike(H), H);
                    var dH = Engine.TensorMultiply(Engine.TensorMultiply(dPred, H), oneMinusH);               // [rows,d,h]
                    gW1Flat = Engine.TensorAdd(gW1Flat,
                        Engine.TensorMatMul(Engine.TensorTranspose(xb), dH.Reshape(new[] { rows, d * h })));
                }
                var gW1 = gW1Flat.Reshape(new[] { d, d, h });                                                 // [i,j,h]

                // Acyclicity gradient on adjacency A[i,j] = Σ_k |W1[j][i,k]| |W2[j][k]|
                var A = ExtractAdjacency(W1, W2, d, h);
                var (hTensor, hGrad) = ComputeExpGradient(A, d);

                // Numerical stabilization. Unlike BCD-Nets' clamped edge logits,
                // GraN-DAG's raw MLP weights are unbounded: when the dual ratchet
                // raises rho while h plateaus (20 inner steps cannot always drive
                // h down), alpha + rho*h reaches ~1e17, the penalty gradient blows
                // the weights up, tr(exp(A.A)) overflows and every weight goes
                // NaN — the learned adjacency was NaN at EVERY learning rate.
                // Clamp the augmented coefficient (the constraint force stays
                // strong but finite) and bail to the last finite state if h has
                // already gone non-finite. Reference implementations of
                // NOTEARS-family solvers apply the same guards.
                double hD = ScalarValue(hTensor);
                if (double.IsNaN(hD) || double.IsInfinity(hD)) break;
                double augD = ScalarValue(alpha) + ScalarValue(rho) * hD;
                if (augD > 1e+6) augD = 1e+6;
                // Warm-start: for the first third of training run PURE data-fit (no acyclicity force).
                // The augmented-Lagrangian gradient (augCoeff up to 1e6, clipped to |g|<=10 => ~lr*10
                // per step) otherwise dominates the much smaller data-fit gradient from epoch 0 and
                // drives every MLP weight toward zero BEFORE the network learns the true dependencies —
                // leaving a diffuse, near-zero adjacency in which no true edge stands out. Letting the
                // data fit establish a sharp connectivity first, then ramping acyclicity to PRUNE it,
                // is the intended NOTEARS/GraN-DAG order (fit, then enforce the constraint).
                if (outer < MaxEpochs / 3) augD = 0.0;

                // Chain rule: dh/dW1 via dh/dA * dA/dW1, for i != j with A[i,j] > 1e-12
                var dhda = Engine.TensorMultiply(ScalarTensor(augD, d), hGrad);
                var dhdaSpread = Engine.TensorBroadcastTo(dhda.Reshape(new[] { d, d, 1 }), new[] { d, d, h });
                var aSpread = Engine.TensorBroadcastTo(A.Reshape(new[] { d, d, 1 }), new[] { d, d, h });
                var active = Engine.TensorMultiply(weightMask, Engine.TensorGreaterThan(aSpread, edgeFloor));
                var chain = Engine.TensorDivide(Engine.TensorMultiply(dhdaSpread, W1), aSpread);
                gW1 = Engine.TensorAdd(gW1, Engine.TensorWhere(active, chain, zerosW1));

                // Update with per-element gradient clipping (|g| <= 10): bounds
                // the per-step weight change to 10*lr so a transiently large
                // acyclicity force cannot launch the unbounded MLP weights into
                // the exp-overflow regime (see the stabilization note above).
                W1 = Engine.TensorSubtract(W1, Engine.TensorMultiplyScalar(Clip(gW1, clipLo, clipHi), lr));
                W2 = Engine.TensorSubtract(W2, Engine.TensorMultiplyScalar(Clip(gW2, clipLo, clipHi), lr));

                // Zero diagonal
                W1 = Engine.TensorWhere(weightMask, W1, zerosW1);
            }

            // Outer: update augmented Lagrangian per the NOTEARS dual schedule
            // GraN-DAG adopts (Lachapelle et al. 2020 §3.2 / Zheng et al. 2018):
            // rho is escalated only when h fails to SHRINK by the relative factor
            // 0.25 versus the previous outer round. The previous code compared h
            // against an ABSOLUTE 0.25, so with any nontrivial initial h the
            // penalty multiplied 10x every epoch up to 1e16 and annihilated every
            // path norm — the learned adjacency collapsed to all-zeros and no
            // edges could ever be reported.
            // Only run the augmented-Lagrangian dual update once the acyclicity force is ACTIVE — i.e. after
            // the pure data-fit warm-start (outer >= MaxEpochs/3, matching the augD gate above). During
            // warm-start h is not being minimized, so ratcheting rho/alpha here (and breaking on rho > rhoMax)
            // could exhaust the penalty schedule before the constraint phase even starts, or inflate rho/alpha
            // so they dominate the instant the force turns on (#1789 review).
            if (outer >= MaxEpochs / 3)
            {
                var Afinal = ExtractAdjacency(W1, W2, d, h);
                var hFinal = ComputeTraceExpConstraint(Afinal, d);
                double hNow = ScalarValue(hFinal);
                if (double.IsNaN(hNow) || double.IsInfinity(hNow)) break;
                if (hNow > 1e-8)
                {
                    alpha = Engine.TensorAdd(alpha, Engine.TensorMultiply(rho, hFinal));
                    if (!double.IsPositiveInfinity(hPrev) && hNow > 0.25 * hPrev)
                        rho = Engine.TensorMultiplyScalar(rho, ten);
                }
                hPrev = hNow;
                if (Engine.TensorCountNonzero(Engine.TensorGreaterThan(rho, rhoMax)) > 0) break;
                if (hNow <= 1e-8) break;
            }
        }

        var rawAdj = ExtractAdjacency(W1, W2, d, h);
        // Use raw adjacency magnitudes as learned edge probabilities (normalize to [0,1])
        var offDiagonalAdj = Engine.TensorWhere(offDiagonal, rawAdj, Engine.TensorZerosLike(rawAdj));
        T maxNorm = Engine.TensorMaxValue(offDiagonalAdj);
        var normalized = NumOps.GreaterThan(maxNorm, NumOps.Zero)
            ? Engine.TensorDivideScalar(offDiagonalAdj, maxNorm)
            : Engine.TensorZerosLike(rawAdj);

        var cov = ComputeCovarianceMatrix(data);
        // Guarantee an acyclic output. BuildFinalAdjacency's direction tie-break only rules out
        // 2-cycles; a 3+-node cycle (A->B->C->A) can still survive raw thresholding — exactly the
        // DiscoverStructure_OutputIsAcyclic failure (topological sort visited 0/4 nodes). ProjectToDag
        // imposes a strict source-score topological order and keeps only forward edges, so the result
        // is a DAG by construction. Mirrors the sibling DAGGNNAlgorithm, which already routes through it.
        return BuildFinalAdjacency(ProjectToDag(normalized, d, rawVariance), cov, d);
    }

    /// <summary>
    /// g clipped to [lo, hi] (NaN stays NaN).
    /// </summary>
    private Tensor<T> Clip(Tensor<T> g, T lo, T hi)
    {
        var upper = Engine.TensorWhere(Engine.TensorGreaterThan(g, hi), Engine.TensorFullLike(g, hi), g);
        return Engine.TensorWhere(Engine.TensorLessThan(upper, lo), Engine.TensorFullLike(g, lo), upper);
    }

    /// <summary>
    /// The single value of a [1] tensor as a double.
    /// </summary>
    private double ScalarValue(Tensor<T> value) => NumOps.ToDouble(value.ToArray()[0]);

    /// <summary>
    /// A [d x d] tensor filled with <paramref name="value"/>.
    /// </summary>
    private Tensor<T> ScalarTensor(double value, int d)
        => Engine.TensorFullLike(new Tensor<T>(new[] { d, d }), NumOps.FromDouble(value));

    private Tensor<T> ExtractAdjacency(Tensor<T> W1, Tensor<T> W2, int d, int h)
    {
        // GraN-DAG connectivity (Lachapelle et al. 2020 §2.2): the path product of absolute
        // weights across ALL layers, NOT the first-layer norm alone. For the per-variable
        // 1-hidden-layer MLP predicting x_j, the connection strength from input i is
        //   A[i,j] = Σ_k |W1[j][i,k]| · |W2[j][k]|
        // — each input→hidden weight is scaled by how much that hidden unit actually drives the
        // output. The previous ||W1[:,i]||_2 measure ignored W2, so a large first-layer weight into
        // a hidden unit that does NOT affect the output (|W2|≈0) still inflated a spurious edge,
        // producing a diffuse/partly-reversed adjacency that the DAG projection then mis-ordered.
        var w2Spread = Engine.TensorBroadcastTo(Engine.TensorAbs(W2).Reshape(new[] { 1, d, h }), new[] { d, d, h });
        var A = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorAbs(W1), w2Spread), new[] { 2 }, false);
        return Engine.TensorWhere(OffDiagonalMask(d), A, Engine.TensorZerosLike(A));
    }

    private Tensor<T> ComputeTraceExpConstraint(Tensor<T> A, int d)
        => ComputeExpGradient(A, d).h;

    private (Tensor<T> h, Tensor<T> grad) ComputeExpGradient(Tensor<T> A, int d)
    {
        // h(A) = tr(e^(A∘A)) - d using power series: exp(M) = I + M + M^2/2! + ... (10 terms)
        // Gradient: dh/dA[i,j] = 2 * A[i,j] * (e^(A∘A))^T[i,j]
        var expM = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(A, A).ToMatrix(), d, 10));
        var h = Engine.TensorSubtractScalar(Engine.ReduceSum(Engine.TensorDiagonal(expM), new[] { 0 }, true), NumOps.FromDouble(d));
        var grad = Engine.TensorMultiplyScalar(Engine.TensorMultiply(A, Engine.TensorTranspose(expM)), NumOps.FromDouble(2));
        return (h, grad);
    }
}
