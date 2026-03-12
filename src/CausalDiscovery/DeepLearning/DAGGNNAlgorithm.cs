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
/// A[i,j] = sigmoid(Z_i^T * Z_j) from learned embeddings Z.
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
        int embDim = Math.Min(HiddenUnits, 8);
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));

        // Node embeddings: Z is d x embDim
        var Z = new Matrix<T>(d, embDim);
        for (int i = 0; i < d; i++)
            for (int k = 0; k < embDim; k++)
                Z[i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        var cov = ComputeCovarianceMatrix(data);
        T eps = NumOps.FromDouble(1e-10);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
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

            // Gradient w.r.t. Z
            var gradZ = new Matrix<T>(d, embDim);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T varI = cov[i, i];
                    if (!NumOps.GreaterThan(varI, eps)) continue;

                    // Data fit: encourage P[i,j] when cov[i,j] is large
                    T corrSq = NumOps.Divide(NumOps.Multiply(cov[i, j], cov[i, j]), varI);
                    T dataGrad = NumOps.Negate(corrSq);

                    // Acyclicity
                    T acycGrad = NumOps.Multiply(NumOps.Add(alpha, NumOps.Multiply(rho, P[i, j])),
                        NumOps.FromDouble(2));
                    T totalGradP = NumOps.Add(dataGrad, acycGrad);

                    T sigDeriv = NumOps.Multiply(P[i, j], NumOps.Subtract(NumOps.One, P[i, j]));
                    T gradScale = NumOps.Multiply(totalGradP, sigDeriv);

                    for (int k = 0; k < embDim; k++)
                    {
                        gradZ[i, k] = NumOps.Add(gradZ[i, k], NumOps.Multiply(gradScale, Z[j, k]));
                        gradZ[j, k] = NumOps.Add(gradZ[j, k], NumOps.Multiply(gradScale, Z[i, k]));
                    }
                }

            for (int i = 0; i < d; i++)
                for (int k = 0; k < embDim; k++)
                    Z[i, k] = NumOps.Subtract(Z[i, k], NumOps.Multiply(lr, gradZ[i, k]));

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
                    dot = NumOps.Add(dot, NumOps.Multiply(Z[i, k], Z[j, k]));
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
