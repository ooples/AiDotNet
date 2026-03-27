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
[ModelPaper("Variational Graph Auto-Encoders", "https://arxiv.org/abs/1611.07308", Year = 2016, Authors = "Thomas N. Kipf, Max Welling")]
public class GAEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GAE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public GAEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

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
        var cov = ComputeCovarianceMatrix(data);
        T eps = NumOps.FromDouble(1e-10);

        // Separate source/target embeddings to break symmetry:
        // P[i,j] = sigmoid(Zs_i . Zt_j) != P[j,i] = sigmoid(Zs_j . Zt_i)
        var ZsMu = new Matrix<T>(d, embDim);
        var ZtMu = new Matrix<T>(d, embDim);
        var ZsLogVar = new Matrix<T>(d, embDim);
        var ZtLogVar = new Matrix<T>(d, embDim);
        for (int i = 0; i < d; i++)
            for (int k = 0; k < embDim; k++)
            {
                ZsMu[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                ZtMu[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                ZsLogVar[i, k] = NumOps.FromDouble(InitialLogVariance);
                ZtLogVar[i, k] = NumOps.FromDouble(InitialLogVariance);
            }

        // KL warm-up schedule to prevent posterior collapse
        int warmUpEpochs = UseKlWarmUp ? Math.Max(1, effectiveEpochs / 4) : 0;

        T lr = NumOps.FromDouble(effectiveLr);
        T alpha = NumOps.Zero;
        T rho = NumOps.FromDouble(0.01); // Start small so reconstruction dominates early
        T lambda1 = NumOps.FromDouble(0.01); // Small sparsity penalty
        double prevHVal = double.MaxValue;
        int acyclicityWarmup = effectiveEpochs / 3; // No acyclicity penalty for first 1/3
        Matrix<T> lastP = new Matrix<T>(d, d); // Save last edge probability matrix

        for (int epoch = 0; epoch < effectiveEpochs; epoch++)
        {
            // KL weight schedule: ramp from DefaultKlWeight to MaxKlWeight over warmUpEpochs
            double klWeight = UseKlWarmUp && warmUpEpochs > 0 && epoch < warmUpEpochs
                ? DefaultKlWeight + (MaxKlWeight - DefaultKlWeight) * epoch / warmUpEpochs
                : MaxKlWeight;

            // Precompute bounded std values for numerical stability
            var stdSArr = new double[d, embDim];
            var stdTArr = new double[d, embDim];
            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    double logVarS = Math.Max(-10, Math.Min(10, NumOps.ToDouble(ZsLogVar[i, k])));
                    double logVarT = Math.Max(-10, Math.Min(10, NumOps.ToDouble(ZtLogVar[i, k])));
                    stdSArr[i, k] = Math.Exp(0.5 * logVarS);
                    stdTArr[i, k] = Math.Exp(0.5 * logVarT);
                }

            // Sample embeddings via reparameterization: z = mu + sigma * epsilon
            var Zs = new Matrix<T>(d, embDim);
            var Zt = new Matrix<T>(d, embDim);
            var epsilonS = new Matrix<T>(d, embDim);
            var epsilonT = new Matrix<T>(d, embDim);
            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    T stdS = NumOps.FromDouble(stdSArr[i, k]);
                    epsilonS[i, k] = NumOps.FromDouble(SampleStandardNormal(rng));
                    Zs[i, k] = NumOps.Add(ZsMu[i, k], NumOps.Multiply(stdS, epsilonS[i, k]));
                    T stdT = NumOps.FromDouble(stdTArr[i, k]);
                    epsilonT[i, k] = NumOps.FromDouble(SampleStandardNormal(rng));
                    Zt[i, k] = NumOps.Add(ZtMu[i, k], NumOps.Multiply(stdT, epsilonT[i, k]));
                }

            // Compute edge probabilities: P[i,j] = sigmoid(Zs_i . Zt_j)
            var P = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T dot = NumOps.Zero;
                    for (int k = 0; k < embDim; k++)
                        dot = NumOps.Add(dot, NumOps.Multiply(Zs[i, k], Zt[j, k]));
                    double sv = NumOps.ToDouble(dot);
                    P[i, j] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

            // Save last P for final edge detection
            for (int i2 = 0; i2 < d; i2++)
                for (int j2 = 0; j2 < d; j2++)
                    lastP[i2, j2] = P[i2, j2];

            // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var PSq = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    PSq[i, j] = NumOps.Multiply(P[i, j], P[i, j]);
            var expPSq = MatrixExponentialTaylor(PSq, d);
            T hVal = NumOps.Zero;
            for (int i = 0; i < d; i++)
                hVal = NumOps.Add(hVal, expPSq[i, i]);
            hVal = NumOps.Subtract(hVal, NumOps.FromDouble(d));

            // Reconstruction: X_hat[s,j] = sum_i X[s,i] * P[i,j]
            var gradP = new Matrix<T>(d, d);
            T invN = NumOps.FromDouble(1.0 / n);

            for (int j = 0; j < d; j++)
            {
                for (int s = 0; s < n; s++)
                {
                    T xhat = NumOps.Zero;
                    for (int i = 0; i < d; i++)
                        xhat = NumOps.Add(xhat, NumOps.Multiply(data[s, i], P[i, j]));

                    T residual = NumOps.Multiply(NumOps.Subtract(xhat, data[s, j]), invN);
                    for (int i = 0; i < d; i++)
                        if (i != j)
                            gradP[i, j] = NumOps.Add(gradP[i, j],
                                NumOps.Multiply(residual, data[s, i]));
                }
            }

            // Gradient: dP/dZ via chain rule through sigmoid and dot product
            var gradZsMu = new Matrix<T>(d, embDim);
            var gradZtMu = new Matrix<T>(d, embDim);
            var gradZsLogVar = new Matrix<T>(d, embDim);
            var gradZtLogVar = new Matrix<T>(d, embDim);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T pij = P[i, j];
                    T sigD = NumOps.Multiply(pij, NumOps.Subtract(NumOps.One, pij));

                    // Add sparsity and acyclicity gradients
                    T sparsityGrad = NumOps.Multiply(lambda1,
                        NumOps.FromDouble(Math.Sign(NumOps.ToDouble(pij))));
                    // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
                    // Skip during warmup so reconstruction can find edges first
                    T acycGrad = epoch < acyclicityWarmup ? NumOps.Zero : NumOps.Multiply(
                        NumOps.Add(alpha, NumOps.Multiply(rho, hVal)),
                        NumOps.Multiply(expPSq[j, i], NumOps.Multiply(NumOps.FromDouble(2), pij)));
                    T totalGradP = NumOps.Add(gradP[i, j], NumOps.Add(sparsityGrad, acycGrad));
                    T gradScale = NumOps.Multiply(totalGradP, sigD);

                    // d(Zs_i . Zt_j)/dZs_i = Zt_j, d(Zs_i . Zt_j)/dZt_j = Zs_i
                    for (int k = 0; k < embDim; k++)
                    {
                        gradZsMu[i, k] = NumOps.Add(gradZsMu[i, k],
                            NumOps.Multiply(gradScale, Zt[j, k]));
                        gradZtMu[j, k] = NumOps.Add(gradZtMu[j, k],
                            NumOps.Multiply(gradScale, Zs[i, k]));

                        // Log-variance gradient via reparameterization:
                        // dL/d(logvar_s) = dL/dz_s * epsilon_s * 0.5 * exp(0.5*logvar_s)
                        // Chain rule: ∂L/∂logvar_s = gradScale * z_t[j,k] * ε_s[i,k] * 0.5 * std_s
                        T stdS = NumOps.FromDouble(Math.Exp(0.5 * NumOps.ToDouble(ZsLogVar[i, k])));
                        gradZsLogVar[i, k] = NumOps.Add(gradZsLogVar[i, k],
                            NumOps.Multiply(gradScale, NumOps.Multiply(Zt[j, k],
                            NumOps.Multiply(epsilonS[i, k],
                            NumOps.Multiply(NumOps.FromDouble(0.5), stdS)))));
                        T stdT = NumOps.FromDouble(Math.Exp(0.5 * NumOps.ToDouble(ZtLogVar[j, k])));
                        gradZtLogVar[j, k] = NumOps.Add(gradZtLogVar[j, k],
                            NumOps.Multiply(gradScale, NumOps.Multiply(Zs[i, k],
                            NumOps.Multiply(epsilonT[j, k],
                            NumOps.Multiply(NumOps.FromDouble(0.5), stdT)))));
                    }
                }

            // KL divergence gradient: d/dmu = klWeight * mu, d/d(logvar) = klWeight * 0.5 * (exp(logvar) - 1)
            T klW = NumOps.FromDouble(klWeight);
            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    gradZsMu[i, k] = NumOps.Add(gradZsMu[i, k],
                        NumOps.Multiply(klW, ZsMu[i, k]));
                    gradZtMu[i, k] = NumOps.Add(gradZtMu[i, k],
                        NumOps.Multiply(klW, ZtMu[i, k]));
                    gradZsLogVar[i, k] = NumOps.Add(gradZsLogVar[i, k],
                        NumOps.Multiply(klW, NumOps.FromDouble(
                            0.5 * (Math.Exp(NumOps.ToDouble(ZsLogVar[i, k])) - 1.0))));
                    gradZtLogVar[i, k] = NumOps.Add(gradZtLogVar[i, k],
                        NumOps.Multiply(klW, NumOps.FromDouble(
                            0.5 * (Math.Exp(NumOps.ToDouble(ZtLogVar[i, k])) - 1.0))));
                }

            // Apply gradients to all parameters
            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    ZsMu[i, k] = NumOps.Subtract(ZsMu[i, k], NumOps.Multiply(lr, gradZsMu[i, k]));
                    ZtMu[i, k] = NumOps.Subtract(ZtMu[i, k], NumOps.Multiply(lr, gradZtMu[i, k]));
                    ZsLogVar[i, k] = NumOps.Subtract(ZsLogVar[i, k], NumOps.Multiply(lr, gradZsLogVar[i, k]));
                    ZtLogVar[i, k] = NumOps.Subtract(ZtLogVar[i, k], NumOps.Multiply(lr, gradZtLogVar[i, k]));
                }

            // Update augmented Lagrangian only after warmup
            if (epoch >= acyclicityWarmup)
            {
                alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));
                double hValDouble = NumOps.ToDouble(hVal);
                T rhoMax = NumOps.FromDouble(MaxPenaltyValue);
                if (hValDouble > 0.25 * Math.Max(1.0, prevHVal))
                {
                    T newRho = NumOps.Multiply(rho, NumOps.FromDouble(10));
                    rho = NumOps.GreaterThan(newRho, rhoMax) ? rhoMax : newRho;
                }
                prevHVal = hValDouble;
            }
        }

        // Final output: use trained P matrix for structure, covariance for weights.
        // The learned P matrix is the PRIMARY signal — only add edges where
        // P[i,j] exceeds a threshold. Covariance provides the edge weight.
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                double pij = NumOps.ToDouble(lastP[i, j]);
                double pji = NumOps.ToDouble(lastP[j, i]);

                // Only add edge if learned probability exceeds threshold
                if (pij < 0.3) continue;

                // Direction: edge i→j only if P[i,j] > P[j,i]
                // For ties, only process the (i,j) pair where i < j to avoid 2-cycles.
                if (pij < pji) continue;
                if (Math.Abs(pij - pji) < 1e-10 && i > j) continue;

                // Use covariance for edge weight
                double varI = NumOps.ToDouble(cov[i, i]);
                if (varI < 1e-10) continue;
                double covIJ = NumOps.ToDouble(cov[i, j]);
                double weight = covIJ / varI;
                if (Math.Abs(weight) < EdgeThreshold) continue;

                result[i, j] = NumOps.FromDouble(weight);
            }

        return result;
    }

    /// <summary>
    /// Samples from a standard normal distribution N(0,1) using Box-Muller transform.
    /// </summary>
    private static double SampleStandardNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(); // (0,1] to avoid log(0)
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

}
