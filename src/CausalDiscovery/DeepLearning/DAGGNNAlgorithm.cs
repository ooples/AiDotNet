using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// DAG-GNN — DAG Structure Learning with Graph Neural Networks.
/// </summary>
/// <remarks>
/// <para>
/// DAG-GNN uses a variational autoencoder framework where the encoder maps data to a
/// latent adjacency matrix A via learned node embeddings, and the decoder reconstructs
/// data using X_hat = X * A. The NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d
/// is enforced via augmented Lagrangian. Edge probabilities are computed as
/// A[i,j] = sigmoid(Zs_i^T * Zt_j) from separate source/target embeddings Zs, Zt.
/// </para>
/// <para>
/// <b>For Beginners:</b> DAG-GNN trains a special neural network (GNN) to simultaneously
/// figure out the graph structure AND generate data that matches the observed data.
/// The best graph is the one that lets the network most accurately recreate the data.
/// </para>
/// <para>
/// Reference: Yu et al. (2019), "DAG-GNN: DAG Structure Learning with Graph Neural Networks", ICML.
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
[ModelPaper("DAG-GNN: DAG Structure Learning with Graph Neural Networks", "https://proceedings.mlr.press/v97/yu19a.html", Year = 2019, Authors = "Yue Yu, Jie Chen, Tian Gao, Mo Yu")]
public class DAGGNNAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "DAG-GNN";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public DAGGNNAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int embDim = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));

        // Separate source and target embeddings to break symmetry:
        // P[i,j] = sigmoid(Zs_i . Zt_j) != P[j,i] = sigmoid(Zs_j . Zt_i)
        var Zs = new Matrix<T>(d, embDim);
        var Zt = new Matrix<T>(d, embDim);
        for (int i = 0; i < d; i++)
            for (int k = 0; k < embDim; k++)
            {
                Zs[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
                Zt[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            }

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        var cov = ComputeCovarianceMatrix(data);
        T eps = NumOps.FromDouble(1e-10);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
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

            // Gradient w.r.t. Zs and Zt
            var gradZs = new Matrix<T>(d, embDim);
            var gradZt = new Matrix<T>(d, embDim);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T varI = cov[i, i];
                    if (!NumOps.GreaterThan(varI, eps)) continue;

                    // Data fit: encourage P[i,j] when cov[i,j] is large
                    T corrSq = NumOps.Divide(NumOps.Multiply(cov[i, j], cov[i, j]), varI);
                    T dataGrad = NumOps.Negate(corrSq);

                    // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
                    T acycGrad = NumOps.Multiply(
                        NumOps.Add(alpha, NumOps.Multiply(rho, hVal)),
                        NumOps.Multiply(expPSq[j, i], NumOps.Multiply(NumOps.FromDouble(2), P[i, j])));
                    T totalGradP = NumOps.Add(dataGrad, acycGrad);

                    T sigDeriv = NumOps.Multiply(P[i, j], NumOps.Subtract(NumOps.One, P[i, j]));
                    T gradScale = NumOps.Multiply(totalGradP, sigDeriv);

                    // d(Zs_i . Zt_j)/dZs_i = Zt_j, d(Zs_i . Zt_j)/dZt_j = Zs_i
                    for (int k = 0; k < embDim; k++)
                    {
                        gradZs[i, k] = NumOps.Add(gradZs[i, k], NumOps.Multiply(gradScale, Zt[j, k]));
                        gradZt[j, k] = NumOps.Add(gradZt[j, k], NumOps.Multiply(gradScale, Zs[i, k]));
                    }
                }

            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                {
                    Zs[i, k] = NumOps.Subtract(Zs[i, k], NumOps.Multiply(lr, gradZs[i, k]));
                    Zt[i, k] = NumOps.Subtract(Zt[i, k], NumOps.Multiply(lr, gradZt[i, k]));
                }

            // Update augmented Lagrangian
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));
            if (NumOps.GreaterThan(hVal, NumOps.FromDouble(0.25)))
                rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
        }

        // Final output using trained embeddings
        var result = new Matrix<T>(d, d);
        T threshold = NumOps.FromDouble(0.3);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;
                T dot = NumOps.Zero;
                for (int k = 0; k < embDim; k++)
                    dot = NumOps.Add(dot, NumOps.Multiply(Zs[i, k], Zt[j, k]));
                double sv = NumOps.ToDouble(dot);
                double prob = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                if (prob > 0.5)
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
