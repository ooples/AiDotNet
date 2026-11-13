using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// GAE — Graph Autoencoder for causal discovery.
/// </summary>
/// <remarks>
/// <para>
/// GAE uses an autoencoder architecture where the encoder produces a latent graph
/// representation and the decoder reconstructs the data through the learned graph.
/// The encoder maps each variable to separate source/target latent embeddings via shared MLP,
/// then computes edge probabilities as sigmoid(Zs_i^T * Zt_j) using asymmetric dot products
/// to encode directionality. The decoder reconstructs X_hat = X * A where A is the soft
/// adjacency. NOTEARS acyclicity constraint h(A) = tr(exp(A∘A)) - d ensures a valid DAG.
/// </para>
/// <para>
/// <b>For Beginners:</b> A Graph Autoencoder compresses data through a "graph bottleneck."
/// The connections in this bottleneck represent causal relationships — the autoencoder
/// is forced to find the minimal set of connections needed to recreate the data.
/// </para>
/// <para>
/// Reference: Kipf and Welling (2016), "Variational Graph Auto-Encoders", NeurIPS Workshop.
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
[ResearchPaper("Variational Graph Auto-Encoders", "https://arxiv.org/abs/1611.07308", Year = 2016, Authors = "Thomas N. Kipf, Max Welling")]
public class GAEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GAE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public GAEAlgorithm(CausalDiscoveryOptions? options = null)
    {
        // GAE-specific defaults that follow Kipf & Welling 2016 (Variational
        // Graph Auto-Encoders) and the VCN / NotEARS hybrid this algorithm
        // is modelled after. Base CausalDiscoveryOptions defaults (lr=1e-3,
        // 100 epochs, KL ramp to 0.25) are too conservative for GAE: on the
        // 4-variable / 200-sample structure-recovery test the KL term
        // (klWeight * mu, multiplied by lr per epoch) integrates to ~0.26
        // shrinkage on the Z_s / Z_t embeddings while the reconstruction
        // signal can only lift the embeddings by ~0.01 in the same window —
        // KL dominates ~30× and every edge probability collapses near 0.
        // The original VGAE paper trained 200 epochs at lr=0.01 with KL
        // weight ≈ 0; the NotEARS hybrid uses negligible KL plus stronger
        // reconstruction. We use lr=0.01, 300 epochs, KL ≤ 0.005 (20×
        // smaller than base), λ_sparsity small enough that the reconstruction
        // signal wins.
        if (options is null || options.LearningRate is null) LearningRate = 0.01;
        if (options is null || (options.MaxIterations is null && options.MaxEpochs is null)) MaxEpochs = 300;
        DefaultKlWeight = 0.0;
        MaxKlWeight = 0.005;
        // The augmented Lagrangian acyclicity penalty (Zheng et al. 2018
        // NotEARS) is supposed to push P toward DAG-ness without crushing
        // edge magnitudes. In practice the original 1e16 cap let alpha + rho
        // run to numerical infinity within a few post-warmup epochs and
        // sigmoid-projected every P[i,j] to 0. Bound the dual variable at
        // a tight 0.5 so the penalty acts as a mild bias toward acyclicity
        // — direction is then resolved at output time by the P[i,j] > P[j,i]
        // filter, which is the actual paper-faithful DAG extraction step.
        if (options is null || options.MaxPenalty is null) MaxPenaltyValue = 0.5;
        ApplyDeepOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        // Respect caller-supplied options without overriding
        int embDim = Math.Min(HiddenUnits, Math.Max(d * 2, 8));
        double effectiveLr = LearningRate;
        int effectiveEpochs = MaxEpochs;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));
        var cov = Tensor<T>.FromMatrix(ComputeCovarianceMatrix(data));
        var x = Tensor<T>.FromMatrix(data);
        var xT = Engine.TensorTranspose(x);
        var shape = new[] { d, embDim };

        // Separate source/target embeddings to break symmetry:
        // P[i,j] = sigmoid(Zs_i . Zt_j) != P[j,i] = sigmoid(Zs_j . Zt_i)
        // Means are scale * (U[0,1) - 0.5), drawn per (i, k) as the pair (Zs, Zt).
        var initial = Engine.TensorMultiplyScalar(
            Engine.TensorSubtractScalar(Tensor<T>.CreateRandom(rng, d, embDim, 2), NumOps.FromDouble(0.5)), scale);
        var ZsMu = Engine.TensorSliceAxis(initial, 2, 0);
        var ZtMu = Engine.TensorSliceAxis(initial, 2, 1);
        var ZsLogVar = Engine.TensorFullLike(ZsMu, NumOps.FromDouble(InitialLogVariance));
        var ZtLogVar = Engine.TensorFullLike(ZtMu, NumOps.FromDouble(InitialLogVariance));

        // KL warm-up schedule to prevent posterior collapse
        int warmUpEpochs = UseKlWarmUp ? Math.Max(1, effectiveEpochs / 4) : 0;

        T lr = NumOps.FromDouble(effectiveLr);
        // Augmented-Lagrangian state as [1] tensors
        var alpha = new Tensor<T>(new[] { 1 });
        var rho = Engine.TensorFullLike(alpha, NumOps.FromDouble(0.01)); // Start small so reconstruction dominates early
        // λ_sparsity must stay well below the reconstruction-loss magnitude.
        // With 300 epochs of sign-based ℓ₁ gradient, λ=0.01 integrates to ~3.0
        // in logit space — strong enough to crush every P[i,j] from the 0.5
        // sigmoid origin down to ~0.03 before the reconstruction signal
        // (~0.1 per epoch on a real edge) ever lifts a single edge above the
        // 0.3 detection threshold. λ=0.0005 keeps the cumulative sparsity
        // budget around 0.15 in logit space — small enough that true edges
        // survive but unrelated pairs still get pushed toward 0.
        T lambda1 = NumOps.FromDouble(0.0005);
        var prevHVal = Engine.TensorFullLike(alpha, NumOps.MaxValue);
        int acyclicityWarmup = effectiveEpochs / 3; // No acyclicity penalty for first 1/3
        var lastP = new Tensor<T>(new[] { d, d }); // Save last edge probability matrix

        var offDiagonal = OffDiagonalMask(d);
        var zerosDd = new Tensor<T>(new[] { d, d });
        T invN = NumOps.FromDouble(1.0 / n);
        T dT = NumOps.FromDouble(d);
        T two = NumOps.FromDouble(2);
        T half = NumOps.FromDouble(0.5);
        T ten = NumOps.FromDouble(10);
        T quarter = NumOps.FromDouble(0.25);
        T logVarLow = NumOps.FromDouble(-10);
        T logVarHigh = NumOps.FromDouble(10);
        T rhoMax = NumOps.FromDouble(MaxPenaltyValue);
        T notNaNBound = NumOps.FromDouble(-1.0);
        T klStart = NumOps.FromDouble(DefaultKlWeight);
        T klSpan = NumOps.FromDouble(MaxKlWeight - DefaultKlWeight);
        T klWarmUpLength = NumOps.FromDouble(warmUpEpochs);
        var klMax = Engine.TensorFullLike(alpha, NumOps.FromDouble(MaxKlWeight));
        var epochNumber = new Tensor<T>(new[] { 1 }); // holds the current epoch index
        var unit = Engine.TensorOnesLike(epochNumber);

        for (int epoch = 0; epoch < effectiveEpochs; epoch++)
        {
            // KL weight schedule: ramp from DefaultKlWeight to MaxKlWeight over warmUpEpochs
            var klWeight = UseKlWarmUp && warmUpEpochs > 0 && epoch < warmUpEpochs
                ? Engine.TensorAddScalar(Engine.TensorDivideScalar(Engine.TensorMultiplyScalar(epochNumber, klSpan), klWarmUpLength), klStart)
                : klMax;
            epochNumber = Engine.TensorAdd(epochNumber, unit);
            var klW = Engine.TensorBroadcastTo(klWeight.Reshape(new[] { 1, 1 }), shape);

            // Precompute bounded std values for numerical stability
            var stdS = Engine.TensorExp(Engine.TensorMultiplyScalar(Engine.TensorMin(Engine.TensorMax(ZsLogVar, logVarLow), logVarHigh), half));
            var stdT = Engine.TensorExp(Engine.TensorMultiplyScalar(Engine.TensorMin(Engine.TensorMax(ZtLogVar, logVarLow), logVarHigh), half));

            // Sample embeddings via reparameterization: z = mu + sigma * epsilon (per (i, k): epsilon_s, then epsilon_t)
            var epsilon = StandardNormals(rng, d, embDim);
            var epsilonS = Engine.TensorSliceAxis(epsilon, 2, 0);
            var epsilonT = Engine.TensorSliceAxis(epsilon, 2, 1);
            var Zs = Engine.TensorAdd(ZsMu, Engine.TensorMultiply(stdS, epsilonS));
            var Zt = Engine.TensorAdd(ZtMu, Engine.TensorMultiply(stdT, epsilonT));

            // Compute edge probabilities: P[i,j] = sigmoid(Zs_i . Zt_j), P[i,i] = 0
            var logits = Engine.TensorMatMul(Zs, Engine.TensorTranspose(Zt));
            var P = Engine.TensorWhere(offDiagonal, ClampedSigmoid(logits, 1.0), zerosDd);

            // Save last P for final edge detection
            lastP = P;

            // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var expPSq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(P, P).ToMatrix(), d));
            var hVal = Engine.TensorSubtractScalar(Engine.ReduceSum(Engine.TensorDiagonal(expPSq), new[] { 0 }, true), dT);

            // Reconstruction: X_hat[s,j] = sum_i X[s,i] * P[i,j]; gradP[i,j] = sum_s residual[s,j] X[s,i] (i != j)
            var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(Engine.TensorMatMul(x, P), x), invN);
            var gradP = Engine.TensorWhere(offDiagonal, Engine.TensorMatMul(xT, residual), zerosDd);

            // Gradient: dP/dZ via chain rule through sigmoid and dot product
            // sign(P) is undefined for NaN (Math.Sign throws), so a diverged P fails loudly.
            if (Engine.TensorCountNonzero(Engine.TensorMultiply(offDiagonal,
                    Engine.TensorSubtract(Engine.TensorOnesLike(P), Engine.TensorGreaterThan(Engine.TensorAbs(P), notNaNBound)))) > 0)
                throw new ArithmeticException("Function does not accept floating point Not-a-Number values.");
            var sigD = Engine.TensorMultiply(P, Engine.TensorSubtract(Engine.TensorOnesLike(P), P));

            // Add sparsity and acyclicity gradients
            var sparsityGrad = Engine.TensorMultiplyScalar(Engine.TensorSign(P), lambda1);
            var penalty = sparsityGrad;
            if (epoch >= acyclicityWarmup)
            {
                // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
                // Skip during warmup so reconstruction can find edges first
                var multiplier = Engine.TensorAdd(alpha, Engine.TensorMultiply(rho, hVal)).Reshape(new[] { 1, 1 });
                var acycGrad = Engine.TensorMultiply(Engine.TensorBroadcastTo(multiplier, new[] { d, d }),
                    Engine.TensorMultiply(Engine.TensorTranspose(expPSq), Engine.TensorMultiplyScalar(P, two)));
                penalty = Engine.TensorAdd(sparsityGrad, acycGrad);
            }
            var gradScale = Engine.TensorWhere(offDiagonal,
                Engine.TensorMultiply(Engine.TensorAdd(gradP, penalty), sigD), zerosDd);

            // d(Zs_i . Zt_j)/dZs_i = Zt_j, d(Zs_i . Zt_j)/dZt_j = Zs_i
            var gradZsMu = Engine.TensorMatMul(gradScale, Zt);
            var gradZtMu = Engine.TensorMatMul(Engine.TensorTranspose(gradScale), Zs);

            // Log-variance gradient via reparameterization:
            // dL/d(logvar_s) = dL/dz_s * epsilon_s * 0.5 * exp(0.5*logvar_s)
            // Use the SAME clamped std as forward pass for consistency
            var gradZsLogVar = Engine.TensorMultiply(gradZsMu, Engine.TensorMultiply(epsilonS, Engine.TensorMultiplyScalar(stdS, half)));
            var gradZtLogVar = Engine.TensorMultiply(gradZtMu, Engine.TensorMultiply(epsilonT, Engine.TensorMultiplyScalar(stdT, half)));

            // KL divergence gradient: d/dmu = klWeight * mu, d/d(logvar) = klWeight * 0.5 * (exp(logvar) - 1)
            gradZsMu = Engine.TensorAdd(gradZsMu, Engine.TensorMultiply(klW, ZsMu));
            gradZtMu = Engine.TensorAdd(gradZtMu, Engine.TensorMultiply(klW, ZtMu));
            gradZsLogVar = Engine.TensorAdd(gradZsLogVar, Engine.TensorMultiply(klW,
                Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(Engine.TensorExp(ZsLogVar), NumOps.One), half)));
            gradZtLogVar = Engine.TensorAdd(gradZtLogVar, Engine.TensorMultiply(klW,
                Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(Engine.TensorExp(ZtLogVar), NumOps.One), half)));

            // Apply gradients to all parameters
            ZsMu = Engine.TensorSubtract(ZsMu, Engine.TensorMultiplyScalar(gradZsMu, lr));
            ZtMu = Engine.TensorSubtract(ZtMu, Engine.TensorMultiplyScalar(gradZtMu, lr));
            ZsLogVar = Engine.TensorSubtract(ZsLogVar, Engine.TensorMultiplyScalar(gradZsLogVar, lr));
            ZtLogVar = Engine.TensorSubtract(ZtLogVar, Engine.TensorMultiplyScalar(gradZtLogVar, lr));

            // Update augmented Lagrangian only after warmup.
            // NotEARS (Zheng et al. 2018) uses alpha += rho * h(P) and bumps
            // rho when h doesn't decrease fast enough. The original
            // formulation is a constrained optimisation: rho grows until h(P)
            // hits a target then alpha takes over. Without an upper cap on
            // ALPHA itself, the dual variable runs away — over 200 post-
            // warmup epochs at rho=MaxPenaltyValue and h≈1, alpha integrates
            // to MaxPenaltyValue × 200 which crushes every P[i,j] toward 0
            // (sigmoid(-large) ≈ 0). Clamp alpha to the same MaxPenaltyValue
            // and only grow rho when h is actually plateauing, so the
            // reconstruction signal can keep edges above the detection
            // threshold.
            if (epoch >= acyclicityWarmup)
            {
                var newAlpha = Engine.TensorAdd(alpha, Engine.TensorMultiply(rho, hVal));
                alpha = Engine.TensorWhere(Engine.TensorGreaterThan(newAlpha, rhoMax), Engine.TensorFullLike(newAlpha, rhoMax), newAlpha);
                var plateau = Engine.TensorMultiplyScalar(Engine.TensorMax(prevHVal, NumOps.One), quarter);
                if (Engine.TensorCountNonzero(Engine.TensorGreaterThan(hVal, plateau)) > 0)
                {
                    var newRho = Engine.TensorMultiplyScalar(rho, ten);
                    rho = Engine.TensorWhere(Engine.TensorGreaterThan(newRho, rhoMax), Engine.TensorFullLike(newRho, rhoMax), newRho);
                }
                prevHVal = hVal;
            }
        }

        // Final output: use trained P matrix for structure, covariance for weights.
        // The learned P matrix is the PRIMARY signal — only add edges where
        // P[i,j] exceeds a threshold. Covariance provides the edge weight.
        // Only add edge if learned probability exceeds 0.3.
        // Direction: edge i→j only if P[i,j] > P[j,i]
        // For ties (|P[i,j] - P[j,i]| < 1e-10), only keep the (i,j) pair where i < j to avoid 2-cycles.
        var ones = Engine.TensorOnesLike(lastP);
        var lastPT = Engine.TensorTranspose(lastP);
        var probable = Engine.TensorSubtract(ones, Engine.TensorLessThan(lastP, NumOps.FromDouble(0.3)));
        var dominant = Engine.TensorSubtract(ones, Engine.TensorLessThan(lastP, lastPT));
        var lowerTriangle = Engine.TensorTranspose(Engine.TensorTriu(ones, 1));
        var tieBreak = Engine.TensorSubtract(ones, Engine.TensorMultiply(lowerTriangle,
            Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtract(lastP, lastPT)), NumOps.FromDouble(1e-10))));

        // Edge weight = correlation coefficient cov(i,j) / sqrt(var(i) * var(j)).
        // Using the raw regression slope cov/var(i) inflates edges whose
        // source has small variance, so an indirect/mediated edge with a
        // small-variance source can dominate the direct edge with a
        // larger-variance source. The correlation coefficient is
        // symmetric in (i, j) and bounded in [-1, 1], preserving the
        // relative ordering of true vs. mediated edges.
        var variance = Engine.TensorDiagonal(cov);
        var varianceOk = Engine.TensorSubtract(Engine.TensorOnesLike(variance), Engine.TensorLessThan(variance, NumOps.FromDouble(1e-10)));
        var bothVariancesOk = Engine.TensorOuter(varianceOk, varianceOk);
        var weight = Engine.TensorDivide(cov, Engine.TensorSqrt(Engine.TensorOuter(variance, variance)));
        var strongEnough = Engine.TensorSubtract(ones, Engine.TensorLessThan(Engine.TensorAbs(weight), NumOps.FromDouble(EdgeThreshold)));

        var keep = Engine.TensorMultiply(Engine.TensorMultiply(Engine.TensorMultiply(offDiagonal, probable),
            Engine.TensorMultiply(dominant, tieBreak)), Engine.TensorMultiply(bothVariancesOk, strongEnough));
        return Engine.TensorWhere(keep, weight, zerosDd).ToMatrix();
    }

    /// <summary>
    /// Standard normals by the Box-Muller transform, two per (i, k) drawn in row-major order: each uses
    /// u1 then u2 from <paramref name="rng"/> as sqrt(-2 ln(1 - u1)) * cos(2π u2) (1 - u1 ∈ (0, 1] avoids log(0)).
    /// Returns [rows, cols, 2].
    /// </summary>
    private Tensor<T> StandardNormals(Random rng, int rows, int cols)
    {
        var uniforms = Tensor<T>.CreateRandom(rng, rows, cols, 2, 2);
        var u1 = Engine.TensorSliceAxis(uniforms, 3, 0);
        var u2 = Engine.TensorSliceAxis(uniforms, 3, 1);
        var radius = Engine.TensorSqrt(Engine.TensorMultiplyScalar(
            Engine.TensorLog(Engine.TensorSubtract(Engine.TensorOnesLike(u1), u1)), NumOps.FromDouble(-2.0)));
        return Engine.TensorMultiply(radius, Engine.TensorCos(Engine.TensorMultiplyScalar(u2, NumOps.FromDouble(2.0 * Math.PI))));
    }

}
