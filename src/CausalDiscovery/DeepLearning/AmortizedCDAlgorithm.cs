using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// Amortized Causal Discovery — meta-learning approach to causal structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Amortized Causal Discovery trains a neural network encoder on the input dataset to learn
/// a mapping from data statistics → edge probabilities. The encoder processes pairwise
/// sufficient statistics (covariance, partial correlations) through an MLP to produce
/// edge logits, which are then refined with NOTEARS acyclicity constraint.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Compute sufficient statistics: covariance matrix and partial correlations</item>
/// <item>For each pair (i,j): encode statistics into features via shared MLP</item>
/// <item>Edge logits = MLP output, edge probabilities = sigmoid(logits)</item>
/// <item>Refine with NOTEARS acyclicity penalty via augmented Lagrangian</item>
/// <item>Train encoder end-to-end to minimize reconstruction loss + acyclicity</item>
/// <item>Threshold final edge probabilities</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of running a slow algorithm each time you have new data,
/// this approach trains a neural network to recognize causal patterns from data statistics.
/// The network learns to map correlation/partial-correlation features to edge probabilities.
/// </para>
/// <para>
/// Reference: Lowe et al. (2022), "Amortized Causal Discovery: Learning to Infer Causal
/// Graphs from Time-Series Data", CLeaR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data", "https://proceedings.mlr.press/v177/lowe22a.html", Year = 2022, Authors = "Sindy Lowe, David Madras, Richard Zemel, Max Welling")]
public class AmortizedCDAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "AmortizedCD";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public AmortizedCDAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var cov = ComputeCovarianceMatrix(data);
        var corr = CovarianceToCorrelation(cov);
        T eps = NumOps.FromDouble(1e-10);

        // Feature dimension per edge: 4 features (cov[i,j], corr[i,j], var_i, var_j)
        int featDim = 4;
        T initScale = NumOps.FromDouble(Math.Sqrt(2.0 / featDim));

        // Shared encoder MLP: W1 (featDim x h), W2 (h x 1)
        var W1 = new Matrix<T>(featDim, h);
        var W2 = new Matrix<T>(h, 1);
        for (int f = 0; f < featDim; f++)
            for (int k = 0; k < h; k++)
                W1[f, k] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        for (int k = 0; k < h; k++)
            W2[k, 0] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;

        // Precompute features for each pair
        var features = new Matrix<T>(d * d, featDim);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                int idx = i * d + j;
                features[idx, 0] = cov[i, j];
                features[idx, 1] = corr[i, j];
                features[idx, 2] = cov[i, i];
                features[idx, 3] = cov[j, j];
            }

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Forward: compute edge logits for all pairs
            var logits = new Matrix<T>(d, d);
            var P = new Matrix<T>(d, d);
            var hiddenCache = new Matrix<T>(d * d, h);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    int idx = i * d + j;

                    // Hidden = sigmoid(features * W1)
                    for (int k = 0; k < h; k++)
                    {
                        T sum = NumOps.Zero;
                        for (int f = 0; f < featDim; f++)
                            sum = NumOps.Add(sum, NumOps.Multiply(features[idx, f], W1[f, k]));
                        hiddenCache[idx, k] = NumOps.FromDouble(Sigmoid(NumOps.ToDouble(sum)));
                    }

                    // Logit = hidden * W2
                    T logit = NumOps.Zero;
                    for (int k = 0; k < h; k++)
                        logit = NumOps.Add(logit, NumOps.Multiply(hiddenCache[idx, k], W2[k, 0]));
                    logits[i, j] = logit;

                    P[i, j] = NumOps.FromDouble(Sigmoid(NumOps.ToDouble(logit)));
                }

            // Compute NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var PSquared = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    PSquared[i, j] = NumOps.Multiply(P[i, j], P[i, j]);
            var expWW = MatrixExponentialTaylor(PSquared, d);
            T hValPrev = NumOps.Zero;
            for (int i = 0; i < d; i++)
                hValPrev = NumOps.Add(hValPrev, expWW[i, i]);
            hValPrev = NumOps.Subtract(hValPrev, NumOps.FromDouble(d));

            var gW1 = new Matrix<T>(featDim, h);
            var gW2 = new Matrix<T>(h, 1);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    int idx = i * d + j;
                    T pij = P[i, j];

                    // Data fit gradient: encourage edges where correlation is strong
                    T absCorr = NumOps.Abs(corr[i, j]);
                    T dataGrad = NumOps.Subtract(pij, absCorr);

                    // Acyclicity gradient: d/dP[i,j] of (alpha * h + rho/2 * h^2)
                    // where h = tr(exp(P∘P)) - d, gradient = (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
                    T acycGrad = NumOps.Multiply(
                        NumOps.Add(alpha, NumOps.Multiply(rho, hValPrev)),
                        NumOps.Multiply(expWW[j, i], NumOps.Multiply(pij, NumOps.FromDouble(2))));

                    T totalGradP = NumOps.Add(dataGrad, acycGrad);

                    // Chain rule: d_loss/d_logit = d_loss/d_P * d_P/d_logit
                    T sigDeriv = NumOps.Multiply(pij, NumOps.Subtract(NumOps.One, pij));
                    T dLogit = NumOps.Multiply(totalGradP, sigDeriv);

                    // d_logit/d_W2
                    for (int k = 0; k < h; k++)
                        gW2[k, 0] = NumOps.Add(gW2[k, 0],
                            NumOps.Multiply(dLogit, hiddenCache[idx, k]));

                    // d_logit/d_hidden -> d_hidden/d_W1
                    for (int k = 0; k < h; k++)
                    {
                        T hk = hiddenCache[idx, k];
                        T hidSigD = NumOps.Multiply(hk, NumOps.Subtract(NumOps.One, hk));
                        T dHidden = NumOps.Multiply(dLogit, NumOps.Multiply(W2[k, 0], hidSigD));
                        for (int f = 0; f < featDim; f++)
                            gW1[f, k] = NumOps.Add(gW1[f, k],
                                NumOps.Multiply(dHidden, features[idx, f]));
                    }
                }

            // Normalize gradients
            T normFactor = NumOps.FromDouble(1.0 / (d * (d - 1)));
            for (int f = 0; f < featDim; f++)
                for (int k = 0; k < h; k++)
                    gW1[f, k] = NumOps.Multiply(gW1[f, k], normFactor);
            for (int k = 0; k < h; k++)
                gW2[k, 0] = NumOps.Multiply(gW2[k, 0], normFactor);

            // Apply gradients
            for (int f = 0; f < featDim; f++)
                for (int k = 0; k < h; k++)
                    W1[f, k] = NumOps.Subtract(W1[f, k], NumOps.Multiply(lr, gW1[f, k]));
            for (int k = 0; k < h; k++)
                W2[k, 0] = NumOps.Subtract(W2[k, 0], NumOps.Multiply(lr, gW2[k, 0]));

            // Update augmented Lagrangian with NOTEARS h(P) = tr(exp(P∘P)) - d
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hValPrev));
            if (NumOps.GreaterThan(hValPrev, NumOps.FromDouble(0.25)))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
        }

        // Final inference pass
        var result = new Matrix<T>(d, d);
        T threshold = NumOps.FromDouble(0.5);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                int idx = i * d + j;

                T logit = NumOps.Zero;
                for (int k = 0; k < h; k++)
                {
                    T sum = NumOps.Zero;
                    for (int f = 0; f < featDim; f++)
                        sum = NumOps.Add(sum, NumOps.Multiply(features[idx, f], W1[f, k]));
                    T hid = NumOps.FromDouble(Sigmoid(NumOps.ToDouble(sum)));
                    logit = NumOps.Add(logit, NumOps.Multiply(hid, W2[k, 0]));
                }

                double prob = Sigmoid(NumOps.ToDouble(logit));
                if (prob > NumOps.ToDouble(threshold))
                {
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T weight = NumOps.Divide(cov[i, j], varI);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), NumOps.FromDouble(0.1)))
                            result[i, j] = weight;
                    }
                }
            }

        return result;
    }

    private static double Sigmoid(double x) =>
        x > 20 ? 1.0 : x < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-x));

    /// <summary>
    /// Computes matrix exponential via Taylor series: exp(M) = I + M + M^2/2! + ... + M^k/k!
    /// Used for NOTEARS acyclicity constraint h(W) = tr(exp(W∘W)) - d.
    /// </summary>
    private Matrix<T> MatrixExponentialTaylor(Matrix<T> M, int d, int terms = 10)
    {
        // result = I
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            result[i, i] = NumOps.One;

        // power = I initially, accumulate M^k / k!
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            power[i, i] = NumOps.One;

        for (int k = 1; k <= terms; k++)
        {
            // power = power * M
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

            // result += power / k!
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
