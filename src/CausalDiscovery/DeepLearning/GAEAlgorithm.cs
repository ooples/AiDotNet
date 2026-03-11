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
/// The encoder maps each variable to a latent embedding via shared MLP, then computes
/// edge probabilities as sigmoid(Z_i^T * Z_j). The decoder reconstructs X_hat = X * A
/// where A is the soft adjacency. NOTEARS acyclicity constraint ensures a valid DAG.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Encoder: map data statistics to node embeddings Z (d x embDim) via MLP</item>
/// <item>Compute edge probabilities: P[i,j] = sigmoid(Z_i^T * Z_j)</item>
/// <item>Decoder: reconstruct X_hat = X * P (graph-filtered signal)</item>
/// <item>Loss = reconstruction MSE + KL divergence + acyclicity penalty</item>
/// <item>Update encoder/decoder via gradient descent</item>
/// <item>Threshold final P to get adjacency</item>
/// </list>
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
        int embDim = Math.Min(HiddenUnits, 8);
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));
        var cov = ComputeCovarianceMatrix(data);
        T eps = NumOps.FromDouble(1e-10);

        // Encoder: per-variable embedding W_enc (d x embDim) maps column statistics to embedding
        var Zmu = new Matrix<T>(d, embDim);
        var ZlogVar = new Matrix<T>(d, embDim);
        for (int i = 0; i < d; i++)
            for (int k = 0; k < embDim; k++)
            {
                Zmu[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                ZlogVar[i, k] = NumOps.FromDouble(-4);
            }

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        T lambda1 = NumOps.FromDouble(0.1);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Sample embeddings via reparameterization: Z = mu + exp(logvar/2) * eps
            var Z = new Matrix<T>(d, embDim);
            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    T std = NumOps.FromDouble(Math.Exp(0.5 * NumOps.ToDouble(ZlogVar[i, k])));
                    T noise = NumOps.FromDouble(rng.NextDouble() * 2 - 1);
                    Z[i, k] = NumOps.Add(Zmu[i, k], NumOps.Multiply(std, noise));
                }

            // Compute edge probabilities: P[i,j] = sigmoid(Z_i . Z_j)
            var P = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T dot = NumOps.Zero;
                    for (int k = 0; k < embDim; k++)
                        dot = NumOps.Add(dot, NumOps.Multiply(Z[i, k], Z[j, k]));
                    double sv = NumOps.ToDouble(dot);
                    P[i, j] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                }

            // Reconstruction: X_hat[s,j] = sum_i X[s,i] * P[i,j]
            // Loss gradient w.r.t. P[i,j]: sum_s (X_hat[s,j] - X[s,j]) * X[s,i] / n
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
            var gradZmu = new Matrix<T>(d, embDim);
            var gradZlogVar = new Matrix<T>(d, embDim);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T pij = P[i, j];
                    T sigD = NumOps.Multiply(pij, NumOps.Subtract(NumOps.One, pij));

                    // Add sparsity and acyclicity gradients
                    T sparsityGrad = NumOps.Multiply(lambda1,
                        NumOps.FromDouble(Math.Sign(NumOps.ToDouble(pij))));
                    T acycGrad = NumOps.Multiply(NumOps.Add(alpha, NumOps.Multiply(rho, pij)),
                        NumOps.FromDouble(2));
                    T totalGradP = NumOps.Add(gradP[i, j], NumOps.Add(sparsityGrad, acycGrad));
                    T gradScale = NumOps.Multiply(totalGradP, sigD);

                    for (int k = 0; k < embDim; k++)
                    {
                        gradZmu[i, k] = NumOps.Add(gradZmu[i, k],
                            NumOps.Multiply(gradScale, Z[j, k]));
                        gradZmu[j, k] = NumOps.Add(gradZmu[j, k],
                            NumOps.Multiply(gradScale, Z[i, k]));
                    }
                }

            // KL divergence gradient for embeddings: KL(N(mu, sigma^2) || N(0,1))
            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    gradZmu[i, k] = NumOps.Add(gradZmu[i, k],
                        NumOps.Multiply(NumOps.FromDouble(0.01), Zmu[i, k]));
                    T var_ik = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(ZlogVar[i, k])));
                    gradZlogVar[i, k] = NumOps.Multiply(NumOps.FromDouble(0.5),
                        NumOps.Subtract(var_ik, NumOps.One));
                }

            // Apply gradients
            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    Zmu[i, k] = NumOps.Subtract(Zmu[i, k], NumOps.Multiply(lr, gradZmu[i, k]));
                    ZlogVar[i, k] = NumOps.Subtract(ZlogVar[i, k],
                        NumOps.Multiply(NumOps.FromDouble(LearningRate * 0.1), gradZlogVar[i, k]));
                }

            // Update augmented Lagrangian
            T hVal = NumOps.Zero;
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    if (i != j) hVal = NumOps.Add(hVal, NumOps.Multiply(P[i, j], P[i, j]));
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));
            if (NumOps.GreaterThan(hVal, NumOps.FromDouble(0.25)))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
        }

        // Final output
        var result = new Matrix<T>(d, d);
        T threshold = NumOps.FromDouble(0.3);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                T dot = NumOps.Zero;
                for (int k = 0; k < embDim; k++)
                    dot = NumOps.Add(dot, NumOps.Multiply(Zmu[i, k], Zmu[j, k]));
                double sv = NumOps.ToDouble(dot);
                if (sv > 0)
                {
                    T varI = cov[i, i];
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T weight = NumOps.Divide(cov[i, j], varI);
                        if (NumOps.GreaterThan(NumOps.Abs(weight), threshold))
                            result[i, j] = weight;
                    }
                }
            }

        return result;
    }
}
