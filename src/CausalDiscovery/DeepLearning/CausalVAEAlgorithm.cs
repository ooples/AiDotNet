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
[ModelPaper("CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models", "https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_CausalVAE_Disentangled_Representation_Learning_via_Neural_Structural_Causal_Models_CVPR_2021_paper.pdf", Year = 2021, Authors = "Mengyue Yang, Furui Liu, Zuozhu Liu, Xiaojian Ma, Zongqing Lu, Jun Zhu")]
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
        T eps = NumOps.FromDouble(1e-10);

        // Adjacency logits A_logits[i,j]: sigmoid(A_logits[i,j]) = soft edge weight
        var ALogits = new Matrix<T>(d, d);

        // Encoder: W_enc (d x h), b_enc (h), produces mu and logvar for exogenous noise
        var Wenc = new Matrix<T>(d, h);
        var Wmu = new Matrix<T>(h, d);    // h -> d (produces per-variable mu)
        var WlogV = new Matrix<T>(h, d);  // h -> d (produces per-variable logvar)
        for (int i = 0; i < d; i++)
            for (int k = 0; k < h; k++)
                Wenc[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        for (int k = 0; k < h; k++)
            for (int j = 0; j < d; j++)
            {
                Wmu[k, j] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                WlogV[k, j] = NumOps.FromDouble(-2);
            }

        // Decoder: W_dec (d x h), W_out (h x d)
        var Wdec = new Matrix<T>(d, h);
        var Wout = new Matrix<T>(h, d);
        for (int i = 0; i < d; i++)
            for (int k = 0; k < h; k++)
                Wdec[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        for (int k = 0; k < h; k++)
            for (int j = 0; j < d; j++)
                Wout[k, j] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        T lambda1 = NumOps.FromDouble(0.1);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Compute soft adjacency
            var A = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    double sv = NumOps.ToDouble(ALogits[i, j]);
                    A[i, j] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

            // Compute (I - A)^{-1} via Neumann series: sum_{k=0}^{K} A^k
            // For small A values this converges quickly
            var IminusAinv = new Matrix<T>(d, d);
            var Apow = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
            {
                IminusAinv[i, i] = NumOps.One;
                Apow[i, i] = NumOps.One;
            }

            for (int p = 1; p <= 5; p++)
            {
                Apow = MatMul(Apow, A);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        IminusAinv[i, j] = NumOps.Add(IminusAinv[i, j], Apow[i, j]);
            }

            var gALogits = new Matrix<T>(d, d);
            var gWenc = new Matrix<T>(d, h);
            var gWmu = new Matrix<T>(h, d);
            var gWdec = new Matrix<T>(d, h);
            var gWout = new Matrix<T>(h, d);
            T invN = NumOps.FromDouble(1.0 / n);

            for (int s = 0; s < n; s++)
            {
                // Encoder forward: hidden_enc = sigmoid(x * Wenc)
                var hEnc = new T[h];
                for (int k = 0; k < h; k++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < d; i++)
                        sum = NumOps.Add(sum, NumOps.Multiply(data[s, i], Wenc[i, k]));
                    double sv = NumOps.ToDouble(sum);
                    hEnc[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

                // Exogenous noise: epsilon_j = mu_j(hEnc) + stddev * noise
                var epsNoise = new T[d];
                for (int j = 0; j < d; j++)
                {
                    T mu_j = NumOps.Zero;
                    for (int k = 0; k < h; k++)
                        mu_j = NumOps.Add(mu_j, NumOps.Multiply(hEnc[k], Wmu[k, j]));
                    epsNoise[j] = mu_j; // Use posterior mean for gradient computation
                }

                // Causal layer: z = (I - A)^{-1} * epsilon
                var z = new T[d];
                for (int j = 0; j < d; j++)
                {
                    z[j] = NumOps.Zero;
                    for (int i = 0; i < d; i++)
                        z[j] = NumOps.Add(z[j], NumOps.Multiply(IminusAinv[j, i], epsNoise[i]));
                }

                // Decoder: hidden_dec = sigmoid(z * Wdec), xhat = hidden_dec * Wout
                var hDec = new T[h];
                for (int k = 0; k < h; k++)
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < d; j++)
                        sum = NumOps.Add(sum, NumOps.Multiply(z[j], Wdec[j, k]));
                    double sv = NumOps.ToDouble(sum);
                    hDec[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

                var xhat = new T[d];
                for (int j = 0; j < d; j++)
                {
                    xhat[j] = NumOps.Zero;
                    for (int k = 0; k < h; k++)
                        xhat[j] = NumOps.Add(xhat[j], NumOps.Multiply(hDec[k], Wout[k, j]));
                }

                // Reconstruction loss gradients
                for (int j = 0; j < d; j++)
                {
                    T residual = NumOps.Multiply(NumOps.Subtract(xhat[j], data[s, j]), invN);

                    // Gradient w.r.t. Wout
                    for (int k = 0; k < h; k++)
                        gWout[k, j] = NumOps.Add(gWout[k, j], NumOps.Multiply(residual, hDec[k]));

                    // Gradient through decoder hidden
                    for (int k = 0; k < h; k++)
                    {
                        T sigD = NumOps.Multiply(hDec[k], NumOps.Subtract(NumOps.One, hDec[k]));
                        T dH = NumOps.Multiply(residual, NumOps.Multiply(Wout[k, j], sigD));
                        for (int i = 0; i < d; i++)
                            gWdec[i, k] = NumOps.Add(gWdec[i, k], NumOps.Multiply(dH, z[i]));
                    }
                }

                // Gradient w.r.t. z through decoder
                var dZ = new T[d];
                for (int j = 0; j < d; j++)
                {
                    T reconGrad = NumOps.Multiply(NumOps.Subtract(xhat[j], data[s, j]), invN);
                    for (int k = 0; k < h; k++)
                    {
                        T sigD = NumOps.Multiply(hDec[k], NumOps.Subtract(NumOps.One, hDec[k]));
                        T dH = NumOps.Multiply(reconGrad, NumOps.Multiply(Wout[k, j], sigD));
                        for (int i = 0; i < d; i++)
                            dZ[i] = NumOps.Add(dZ[i], NumOps.Multiply(dH, Wdec[i, k]));
                    }
                }

                // Gradient w.r.t. A through causal layer: dL/dA[i,j] = dL/dz * d(IminusAinv*eps)/dA
                // z_m = sum_l (I-A)^{-1}[m,l] * eps[l]
                // d(I-A)^{-1}/dA[i,j] = (I-A)^{-1} * dA/dA[i,j] * (I-A)^{-1}
                // where dA[i,j] is the matrix with 1 at [i,j] and 0 elsewhere
                // So dz_m/dA[i,j] = IminusAinv[m,i] * IminusAinv[j,:] . eps
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        if (i == j) continue;
                        T IinvEpsJ = NumOps.Zero;
                        for (int l = 0; l < d; l++)
                            IinvEpsJ = NumOps.Add(IinvEpsJ, NumOps.Multiply(IminusAinv[j, l], epsNoise[l]));
                        T dataFitGrad = NumOps.Zero;
                        for (int m = 0; m < d; m++)
                            dataFitGrad = NumOps.Add(dataFitGrad, NumOps.Multiply(dZ[m], IminusAinv[m, i]));
                        dataFitGrad = NumOps.Multiply(dataFitGrad, IinvEpsJ);
                        T aSigD = NumOps.Multiply(A[i, j], NumOps.Subtract(NumOps.One, A[i, j]));
                        gALogits[i, j] = NumOps.Add(gALogits[i, j], NumOps.Multiply(dataFitGrad, aSigD));
                    }

                // Gradient w.r.t. encoder weights
                for (int j = 0; j < d; j++)
                {
                    T reconGrad2 = NumOps.Multiply(NumOps.Subtract(xhat[j], data[s, j]), invN);
                    for (int k = 0; k < h; k++)
                    {
                        T sigD = NumOps.Multiply(hEnc[k], NumOps.Subtract(NumOps.One, hEnc[k]));
                        T dHEnc = NumOps.Multiply(reconGrad2, NumOps.Multiply(Wmu[k, j], sigD));
                        for (int i = 0; i < d; i++)
                            gWenc[i, k] = NumOps.Add(gWenc[i, k], NumOps.Multiply(dHEnc, data[s, i]));
                        gWmu[k, j] = NumOps.Add(gWmu[k, j], NumOps.Multiply(reconGrad2, hEnc[k]));
                    }
                }
            }

            // NOTEARS acyclicity: h(A) = tr(exp(A∘A)) - d
            var ASq = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    ASq[i, j] = NumOps.Multiply(A[i, j], A[i, j]);
            var expASq = MatrixExponentialTaylor(ASq, d);
            T hVal = NumOps.Zero;
            for (int i = 0; i < d; i++)
                hVal = NumOps.Add(hVal, expASq[i, i]);
            hVal = NumOps.Subtract(hVal, NumOps.FromDouble(d));

            // Acyclicity and sparsity gradients on A logits
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T aij = A[i, j];
                    T l1Grad = NumOps.Multiply(lambda1,
                        NumOps.FromDouble(Math.Sign(NumOps.ToDouble(aij))));
                    // Acyclicity gradient: (alpha + rho*h) * [exp(A∘A)^T ∘ 2A][i,j]
                    T acycGrad = NumOps.Multiply(
                        NumOps.Add(alpha, NumOps.Multiply(rho, hVal)),
                        NumOps.Multiply(expASq[j, i], NumOps.Multiply(NumOps.FromDouble(2), aij)));
                    T aSigD = NumOps.Multiply(aij, NumOps.Subtract(NumOps.One, aij));
                    gALogits[i, j] = NumOps.Add(gALogits[i, j],
                        NumOps.Multiply(NumOps.Add(l1Grad, acycGrad), aSigD));
                }

            // Apply gradients
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    if (i != j)
                        ALogits[i, j] = NumOps.Subtract(ALogits[i, j],
                            NumOps.Multiply(lr, gALogits[i, j]));

            for (int i = 0; i < d; i++)
                for (int k = 0; k < h; k++)
                {
                    Wenc[i, k] = NumOps.Subtract(Wenc[i, k], NumOps.Multiply(lr, gWenc[i, k]));
                    Wdec[i, k] = NumOps.Subtract(Wdec[i, k], NumOps.Multiply(lr, gWdec[i, k]));
                }
            for (int k = 0; k < h; k++)
                for (int j = 0; j < d; j++)
                {
                    Wmu[k, j] = NumOps.Subtract(Wmu[k, j], NumOps.Multiply(lr, gWmu[k, j]));
                    Wout[k, j] = NumOps.Subtract(Wout[k, j], NumOps.Multiply(lr, gWout[k, j]));
                }

            // Update augmented Lagrangian with NOTEARS h(A) = tr(exp(A∘A)) - d
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));
            if (NumOps.GreaterThan(hVal, NumOps.FromDouble(0.25)))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
        }

        // Final output: threshold adjacency
        var result = new Matrix<T>(d, d);
        T threshold = NumOps.FromDouble(0.5);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                double sv = NumOps.ToDouble(ALogits[i, j]);
                double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                T prob = NumOps.FromDouble(sigVal);

                if (NumOps.GreaterThan(prob, threshold))
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
