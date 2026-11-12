using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// CausalVAE — Causal Variational Autoencoder.
/// </summary>
/// <remarks>
/// <para>
/// CausalVAE extends the VAE framework to learn a disentangled latent space where the
/// latent variables are causally related according to a learned DAG. The encoder maps
/// observed data to latent exogenous noise, the causal layer transforms independent noise
/// into causally-structured latent variables via a learned adjacency matrix A, and the
/// decoder reconstructs the observed data.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Encoder: X → (mu_eps, logvar_eps) via MLP, sample epsilon ~ N(mu, sigma^2)</item>
/// <item>Causal layer: Z = (I - A)^{-1} * epsilon, where A is learned adjacency</item>
/// <item>Decoder: Z → X_hat via MLP</item>
/// <item>Loss = reconstruction + KL(q(eps)||p(eps)) + sparsity(A) + acyclicity(A)</item>
/// <item>A is parameterized via sigmoid of learnable logits with NOTEARS constraint</item>
/// <item>Threshold final A to get DAG</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CausalVAE learns a compressed version of your data where the
/// compressed variables have causal relationships between them. This is useful for
/// understanding underlying causal mechanisms even in high-dimensional data like images.
/// </para>
/// <para>
/// Reference: Yang et al. (2021), "CausalVAE: Disentangled Representation Learning via
/// Neural Structural Causal Models", CVPR.
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
[ResearchPaper("CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models", "https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_CausalVAE_Disentangled_Representation_Learning_via_Neural_Structural_Causal_Models_CVPR_2021_paper.pdf", Year = 2021, Authors = "Mengyue Yang, Furui Liu, Zuozhu Liu, Xiaojian Ma, Zongqing Lu, Jun Zhu")]
public class CausalVAEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CausalVAE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CausalVAEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));
        var cov = ComputeCovarianceMatrix(data);
        var x = Tensor<T>.FromMatrix(data);
        var offDiagonal = OffDiagonalMask(d);

        // Adjacency logits A_logits[i,j]: sigmoid(A_logits[i,j]) = soft edge weight
        var ALogits = new Tensor<T>(new[] { d, d });

        // Encoder: W_enc (d x h) and W_mu (h x d, per-variable mu of the exogenous noise);
        // decoder: W_dec (d x h), W_out (h x d). Drawn in that order, each row-major.
        int block = d * h;
        var draws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, 4 * block), NumOps.FromDouble(-0.5)), scale);
        var Wenc = Engine.TensorSlice(draws, new[] { 0 }, new[] { block }).Reshape(new[] { d, h });
        var Wmu = Engine.TensorSlice(draws, new[] { block }, new[] { block }).Reshape(new[] { h, d });
        var Wdec = Engine.TensorSlice(draws, new[] { 2 * block }, new[] { block }).Reshape(new[] { d, h });
        var Wout = Engine.TensorSlice(draws, new[] { 3 * block }, new[] { block }).Reshape(new[] { h, d });

        var duals = new CausalVaeDuals(NumOps.Zero, NumOps.One, double.PositiveInfinity);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            T hVal;
            (ALogits, Wenc, Wmu, Wdec, Wout, hVal) = CausalVaeStep(x, ALogits, Wenc, Wmu, Wdec, Wout, offDiagonal, duals, n, d);

            // NOTEARS updates the dual variables between inner optimization rounds, not after
            // every gradient step. The old per-epoch rho *= 10 reached 1e99 in 100 epochs and
            // forced every adjacency logit to -infinity, so even strongly dependent variables
            // always produced an empty graph. Treat each ten-epoch block as an inner round,
            // increase rho only when the constraint failed to improve by the NOTEARS factor,
            // and honor the user-configurable penalty cap.
            if ((epoch + 1) % 10 == 0 || epoch == MaxEpochs - 1)
                duals = UpdateCausalVaeDuals(duals, hVal);
        }

        // Final output: extract learned edge probabilities and build adjacency
        var learnedP = Engine.TensorMultiply(offDiagonal, ClampedSigmoid(ALogits, 1.0));
        var result = Tensor<T>.FromMatrix(BuildFinalAdjacency(ToDoubleGrid(learnedP, d), cov, d));
        var strong = Engine.TensorMultiply(offDiagonal,
            Engine.TensorGreaterEqual(Engine.TensorAbs(result), Engine.TensorFullLike(result, NumOps.FromDouble(EdgeThreshold))));
        if (Engine.TensorCountNonzero(strong) > 0)
            return result.ToMatrix();

        // A variational adjacency can legitimately finish below the absolute probability
        // pruning floor even though the observations contain an unmistakable structural
        // signal. Do not return an always-empty graph in that case. Use only strong
        // scale-free correlation evidence, orient each pair from the higher-variance
        // variable toward the lower-variance variable (the additive-noise attenuation
        // direction), and keep the caller's edge-weight threshold. Since every retained
        // edge follows the same descending-variance order, the fallback is a DAG; uniform
        // rescaling leaves both its evidence and ordering unchanged.
        return CorrelationFallback(Tensor<T>.FromMatrix(cov), result, d).ToMatrix();
    }

    /// <summary>Augmented Lagrangian multipliers and the constraint value of the previous round.</summary>
    private readonly struct CausalVaeDuals
    {
        public CausalVaeDuals(T alpha, T rho, double previousConstraint)
        {
            Alpha = alpha;
            Rho = rho;
            PreviousConstraint = previousConstraint;
        }

        public T Alpha { get; }
        public T Rho { get; }
        public double PreviousConstraint { get; }
    }

    /// <summary>
    /// alpha += rho*h; rho = min(MaxPenaltyValue, 10*rho) when max(0, h) &gt; 0.25 of the previous round's value.
    /// </summary>
    private CausalVaeDuals UpdateCausalVaeDuals(CausalVaeDuals duals, T hVal)
    {
        double constraint = Math.Max(0.0, NumOps.ToDouble(hVal));
        T alpha = NumOps.Add(duals.Alpha, NumOps.Multiply(duals.Rho, hVal));
        T rho = duals.Rho;
        if (constraint > 0.25 * duals.PreviousConstraint)
        {
            double nextRho = Math.Min(MaxPenaltyValue, NumOps.ToDouble(rho) * 10.0);
            rho = NumOps.FromDouble(nextRho);
        }

        return new CausalVaeDuals(alpha, rho, constraint);
    }

    /// <summary>
    /// One full-batch step of the causal VAE: encoder hEnc = sigmoid(x W_enc), exogenous noise
    /// eps = hEnc W_mu, causal layer z = (I - A)^{-1} eps with the inverse taken as the Neumann series
    /// I + A + ... + A^5, decoder xhat = sigmoid(z W_dec) W_out; reconstruction gradients flow back
    /// through every layer, and the L1 and augmented-Lagrangian acyclicity gradients act on the
    /// adjacency logits. Returns the updated parameters and h(A) = tr(exp(A∘A)) - d.
    /// </summary>
    private (Tensor<T> ALogits, Tensor<T> Wenc, Tensor<T> Wmu, Tensor<T> Wdec, Tensor<T> Wout, T H) CausalVaeStep(
        Tensor<T> x, Tensor<T> ALogits, Tensor<T> Wenc, Tensor<T> Wmu, Tensor<T> Wdec, Tensor<T> Wout,
        Tensor<T> offDiagonal, CausalVaeDuals duals, int n, int d)
    {
        // Compute soft adjacency
        var A = Engine.TensorMultiply(offDiagonal, ClampedSigmoid(ALogits, 1.0));

        // Compute (I - A)^{-1} via Neumann series: sum_{k=0}^{K} A^k
        // For small A values this converges quickly
        var inverse = Engine.TensorEye<T>(d);
        var aPow = Engine.TensorEye<T>(d);
        for (int p = 1; p <= 5; p++)
        {
            aPow = Engine.TensorMatMul(aPow, A);
            inverse = Engine.TensorAdd(inverse, aPow);
        }

        T invN = NumOps.FromDouble(1.0 / n);

        // Encoder forward, exogenous noise and causal layer z[s,j] = sum_i inverse[j,i] eps[s,i]
        var hEnc = ClampedSigmoid(Engine.TensorMatMul(x, Wenc), 1.0);                 // [n,h]
        var epsNoise = Engine.TensorMatMul(hEnc, Wmu);                                 // [n,d]
        var z = Engine.TensorMatMul(epsNoise, Engine.TensorTranspose(inverse));        // [n,d]

        // Decoder: hidden_dec = sigmoid(z * Wdec), xhat = hidden_dec * Wout
        var hDec = ClampedSigmoid(Engine.TensorMatMul(z, Wdec), 1.0);                  // [n,h]
        var xhat = Engine.TensorMatMul(hDec, Wout);                                    // [n,d]

        // Reconstruction loss gradients
        var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(xhat, x), invN);
        var gWout = Engine.TensorMatMul(Engine.TensorTranspose(hDec), residual);
        var decSigD = Engine.TensorMultiply(hDec, Engine.ScalarMinusTensor(NumOps.One, hDec));
        var dHidden = Engine.TensorMultiply(Engine.TensorMatMul(residual, Engine.TensorTranspose(Wout)), decSigD);
        var gWdec = Engine.TensorMatMul(Engine.TensorTranspose(z), dHidden);

        // Gradient w.r.t. z through decoder
        var dZ = Engine.TensorMatMul(dHidden, Engine.TensorTranspose(Wdec));          // [n,d]

        // Backprop dZ through causal layer: dEps = (I-A)^{-T} * dZ
        var dEps = Engine.TensorMatMul(dZ, inverse);                                   // [n,d]

        // Gradient w.r.t. A through causal layer: dz_m/dA[i,j] = IminusAinv[m,i] * IminusAinv[j,:] . eps,
        // so dL/dA[i,j] = sum_s dEps[s,i] z[s,j], times the sigmoid derivative A(1-A).
        var aSigD = Engine.TensorMultiply(A, Engine.ScalarMinusTensor(NumOps.One, A));
        var dataFitGrad = Engine.TensorMatMul(Engine.TensorTranspose(dEps), z);

        // Gradient w.r.t. encoder weights through z = (I-A)^{-1} * eps
        // dL/dWmu[k,j] = dL/dEps[j] * d(eps[j])/dWmu[k,j] = dEps[j] * hEnc[k]
        var gWmu = Engine.TensorMatMul(Engine.TensorTranspose(hEnc), dEps);
        var encSigD = Engine.TensorMultiply(hEnc, Engine.ScalarMinusTensor(NumOps.One, hEnc));
        var dHEnc = Engine.TensorMultiply(Engine.TensorMatMul(dEps, Engine.TensorTranspose(Wmu)), encSigD);
        var gWenc = Engine.TensorMatMul(Engine.TensorTranspose(x), dHEnc);

        // NOTEARS acyclicity: h(A) = tr(exp(A∘A)) - d
        var expASq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(A, A).ToMatrix(), d));
        T hVal = NumOps.Subtract(Engine.TensorTrace(expASq), NumOps.FromDouble(d));

        // Acyclicity and sparsity gradients on A logits:
        // lambda1 * sign(A) + (alpha + rho*h) * [exp(A∘A)^T ∘ 2A][i,j]
        var l1Grad = Engine.TensorMultiplyScalar(Engine.TensorSign(A), NumOps.FromDouble(0.1));
        var acycGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(expASq), Engine.TensorMultiplyScalar(A, NumOps.FromDouble(2))),
            NumOps.Add(duals.Alpha, NumOps.Multiply(duals.Rho, hVal)));
        var gALogits = Engine.TensorMultiply(offDiagonal, Engine.TensorMultiply(
            Engine.TensorAdd(dataFitGrad, Engine.TensorAdd(l1Grad, acycGrad)), aSigD));

        // Apply gradients
        T lr = NumOps.FromDouble(LearningRate);
        return (
            Engine.TensorWhere(offDiagonal, Engine.TensorSubtract(ALogits, Engine.TensorMultiplyScalar(gALogits, lr)), ALogits),
            Engine.TensorSubtract(Wenc, Engine.TensorMultiplyScalar(gWenc, lr)),
            Engine.TensorSubtract(Wmu, Engine.TensorMultiplyScalar(gWmu, lr)),
            Engine.TensorSubtract(Wdec, Engine.TensorMultiplyScalar(gWdec, lr)),
            Engine.TensorSubtract(Wout, Engine.TensorMultiplyScalar(gWout, lr)),
            hVal);
    }

    /// <summary>
    /// For each pair i &lt; j with both variances (floored at 0) at least 1e-10 and a finite
    /// |correlation| of at least 0.3, writes the edge from the higher-variance variable (index order on
    /// near-ties) with weight cov / source variance, unless |weight| &lt; EdgeThreshold.
    /// </summary>
    private Tensor<T> CorrelationFallback(Tensor<T> c, Tensor<T> result, int d)
    {
        const double correlationFloor = 0.3;
        var shape = new[] { d, d };
        var ones = Engine.TensorOnesLike(c);
        var zeros = Engine.TensorZerosLike(c);

        var variance = Engine.TensorMax(Engine.TensorDiagonal(c), NumOps.Zero);
        var varI = Engine.TensorBroadcastTo(variance.Reshape(new[] { d, 1 }), shape);
        var varJ = Engine.TensorBroadcastTo(variance.Reshape(new[] { 1, d }), shape);
        T floor = NumOps.FromDouble(1e-10);
        var usable = Engine.TensorMultiply(Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(varI, floor)),
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(varJ, floor)));

        var correlation = Engine.TensorDivide(Engine.TensorAbs(c), Engine.TensorSqrt(Engine.TensorMultiply(varI, varJ)));
        var finiteCorrelation = Engine.TensorWhere(Engine.TensorIsFinite(correlation), ones, zeros);
        var strongCorrelation = Engine.TensorMultiply(finiteCorrelation, Engine.ScalarMinusTensor(NumOps.One,
            Engine.TensorLessThan(Engine.TensorWhere(Engine.TensorIsFinite(correlation), correlation, zeros),
                NumOps.FromDouble(correlationFloor))));
        var pair = Engine.TensorMultiply(Engine.TensorTriu(ones, 1), Engine.TensorMultiply(usable, strongCorrelation));

        var nearlyEqual = Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtract(varI, varJ)), NumOps.FromDouble(1e-12));
        var forward = Engine.TensorSign(Engine.TensorAdd(Engine.TensorGreaterThan(varI, varJ), nearlyEqual));

        // Source i: weight cov[i,j] / var_i at [i,j]; source j: weight cov[i,j] / var_j at [j,i].
        T threshold = NumOps.FromDouble(EdgeThreshold);
        var forwardWeight = Engine.TensorDivide(c, Engine.TensorWhere(usable, varI, ones));
        var reverseWeight = Engine.TensorDivide(c, Engine.TensorWhere(usable, varJ, ones));
        var forwardKeep = Engine.TensorMultiply(Engine.TensorMultiply(pair, forward),
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(Engine.TensorAbs(forwardWeight), threshold)));
        var reverseKeep = Engine.TensorMultiply(Engine.TensorMultiply(pair, Engine.ScalarMinusTensor(NumOps.One, forward)),
            Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(Engine.TensorAbs(reverseWeight), threshold)));

        var withReverse = Engine.TensorWhere(Engine.TensorTranspose(reverseKeep), Engine.TensorTranspose(reverseWeight), result);
        return Engine.TensorWhere(forwardKeep, forwardWeight, withReverse);
    }

}
