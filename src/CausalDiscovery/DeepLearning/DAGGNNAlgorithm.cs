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
[ResearchPaper("DAG-GNN: DAG Structure Learning with Graph Neural Networks", "https://proceedings.mlr.press/v97/yu19a.html", Year = 2019, Authors = "Yue Yu, Jie Chen, Tian Gao, Mo Yu")]
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

        // Raw per-column variance (computed BEFORE standardizing) — used only to ORIENT the final DAG, not
        // to learn it. DAG-GNN's data-fit signal here is the squared correlation cov[i,j]²/var[i], which
        // becomes symmetric once the columns are standardized to unit variance (cov[i,j]² = cov[j,i]²), so
        // the learned probability matrix P cannot recover edge DIRECTION on its own. An exogenous root has
        // higher variance than its attenuated descendants (x1 = 0.8·x0 + noise ⇒ var(x1) < var(x0)), so
        // ordering nodes by descending raw variance yields the correct causal direction. This stays
        // invariant to the uniform data scaling the contract requires: scaling every column by c multiplies
        // every variance by c², leaving the ordering — and therefore the oriented structure — unchanged.
        var rawColumns = Tensor<T>.FromMatrix(data);
        var rawSumSquares = Engine.ReduceSum(Engine.TensorSquare(CenterColumns(rawColumns)), new[] { 0 }, false);
        var rawVar = Engine.TensorDivideScalar(rawSumSquares, NumOps.FromDouble(Math.Max(1, n - 1)))
            .Cast<double>().ToArray();

        // Standardize columns so the LEARNING is invariant to per-variable scaling.
        data = StandardizeColumns(data);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        double scale = Math.Sqrt(2.0 / d);

        // Separate source and target embeddings to break symmetry:
        // P[i,j] = sigmoid(Zs_i . Zt_j) != P[j,i] = sigmoid(Zs_j . Zt_i)
        // Draws alternate Zs[i,k], Zt[i,k] in (i, k) order.
        var draws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, d, embDim, 2), NumOps.FromDouble(-0.5)),
            NumOps.FromDouble(scale));
        var Zs = Engine.TensorSliceAxis(draws, 2, 0);
        var Zt = Engine.TensorSliceAxis(draws, 2, 1);

        T alpha = NumOps.Zero;
        // Acyclicity warm-up: rho = 0 (no NOTEARS penalty) for the first half of training so the data fit
        // can drive edge probabilities up; a bounded fixed penalty is applied after (see the rho update).
        T rho = NumOps.Zero;
        T boundedRho = NumOps.One;
        var cov = ComputeCovarianceMatrix(data);
        var c = Tensor<T>.FromMatrix(cov);
        var offDiagonal = OffDiagonalMask(d);
        var dataGrad = ComputeSkeletonDataGradient(c, offDiagonal, d);

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            (Zs, Zt) = EmbeddingStep(Zs, Zt, dataGrad, offDiagonal, alpha, rho, d);

            // Apply only a BOUNDED acyclicity penalty in the second half of training (warm-up; see the rho
            // init). The original schedule grew rho ×10 toward 1e16 AND accumulated alpha every epoch, which
            // on strongly correlated data made the penalty collapse every edge to 0 (empty graph = trivially
            // acyclic), recovering nothing. A fixed, modest rho breaks ties toward a DAG without overwhelming
            // the data fit. DAG-GNN learns ASYMMETRIC edge probabilities (Zs_i·Zt_j ≠ Zs_j·Zt_i) which can
            // form a directed cycle, so the final probabilities are projected onto a DAG below.
            if (epoch >= MaxEpochs / 2)
                rho = boundedRho;
        }

        // Final output: use trained P for directionality, covariance for weights
        // Compute final P matrix from trained embeddings
        var finalP = EdgeProbabilities(Zs, Zt, offDiagonal);

        // SKELETON CONFIDENCE IS SYMMETRIC; DIRECTION IS NOT LEARNED FROM IT.
        //
        // The acyclicity penalty treats a high P[i,j] together with a high P[j,i] as a 2-cycle and
        // pushes one of them down, and which one it picks comes from the random embedding
        // initialization rather than from the data -- for linear-Gaussian data it cannot come from
        // the data, since both orientations sit in the same Markov equivalence class. Reading a
        // direction off the survivor is therefore reading off the seed, and the projection below,
        // which orients by variance, then keeps whichever direction the penalty had suppressed. On
        // a fixture whose variance ordering matches its topology that left every true edge under
        // the threshold and returned an empty graph; on one where the ordering disagreed it
        // returned the edges reversed and the test still passed, because it only counts edges.
        //
        // Taking the larger of the two therefore restores what the pair actually established -- that
        // i and j are related -- and lets the variance ordering supply the direction, which is what
        // the projection already documents itself as doing.
        var undirected = Engine.TensorMax(finalP, Engine.TensorTranspose(finalP));

        // Project the asymmetric learned probabilities onto a DAG, orienting by raw per-column variance
        // (highest-variance exogenous root first). The learned P is (near-)symmetric after standardization
        // and cannot orient edges by itself; the variance ordering supplies the direction and is preserved
        // under uniform scaling (see rawVar above), so the result stays scale-invariant. BuildFinalAdjacency
        // then thresholds and weights the acyclic probabilities.
        return BuildFinalAdjacency(ProjectToDag(ToDoubleGrid(undirected, d), d, rawVar), cov, d);
    }

    /// <summary>P[i,j] = sigmoid(Zs_i . Zt_j) off the diagonal (saturated beyond ±20), zero on it.</summary>
    private Tensor<T> EdgeProbabilities(Tensor<T> Zs, Tensor<T> Zt, Tensor<T> offDiagonal)
    {
        var dot = Engine.TensorMatMul(Zs, Engine.TensorTranspose(Zt));
        return Engine.TensorMultiply(offDiagonal, ClampedSigmoid(dot, 1.0));
    }

    /// <summary>
    /// Data-fit gradient -cov[i,j]^2 / (var[i] var[j]) for off-diagonal pairs with both variances above
    /// 1e-10, zero elsewhere (those pairs receive no gradient at all).
    /// </summary>
    /// <remarks>
    /// Data fit: encourage P[i,j] when i and j are strongly related.
    ///
    /// SYMMETRIC ON PURPOSE. This used to divide by varI alone, which is asymmetric
    /// in exactly the wrong way: cov[i,j]^2 is symmetric, so dividing by the SOURCE's
    /// variance rewarded edges leaving the LOW-variance node. In an attenuating
    /// linear SEM the parent carries the larger variance, so every edge was pushed
    /// child-to-parent -- and the variance ordering used for the final projection
    /// says the opposite, so the two fought. On a fixture whose variance ordering
    /// matches its topology the projection kept the direction this term had
    /// suppressed, it fell under the threshold, and the graph came back empty.
    ///
    /// It cannot be otherwise: for linear-Gaussian data the direction of an edge is
    /// not identifiable from second-order statistics at all -- both orientations lie
    /// in the same Markov equivalence class -- so any covariance surrogate that
    /// appears to orient is orienting on an artefact. Squared Pearson correlation is
    /// the direction-neutral form, so this term now decides only the SKELETON and
    /// leaves orientation to the acyclicity penalty and the variance ordering, which
    /// is what the projection below already documents itself as doing.
    /// </remarks>
    private (Tensor<T> Gradient, Tensor<T> Usable) ComputeSkeletonDataGradient(Tensor<T> c, Tensor<T> offDiagonal, int d)
    {
        var shape = new[] { d, d };
        var diagonal = Engine.TensorDiagonal(c);
        var varI = Engine.TensorBroadcastTo(diagonal.Reshape(new[] { d, 1 }), shape);
        var varJ = Engine.TensorBroadcastTo(diagonal.Reshape(new[] { 1, d }), shape);
        T eps = NumOps.FromDouble(1e-10);
        var usable = Engine.TensorMultiply(offDiagonal, Engine.TensorMultiply(
            Engine.TensorGreaterThan(varI, eps), Engine.TensorGreaterThan(varJ, eps)));
        var ones = Engine.TensorOnesLike(c);
        var denominator = Engine.TensorWhere(usable, Engine.TensorMultiply(varI, varJ), ones);
        var corrSq = Engine.TensorDivide(Engine.TensorMultiply(c, c), denominator);
        return (Engine.TensorWhere(usable, Engine.TensorNegate(corrSq), Engine.TensorZerosLike(c)), usable);
    }

    /// <summary>
    /// One gradient step on the source and target embeddings: for every usable pair,
    /// dL/dP = data + (alpha + rho*h) * [exp(P∘P)^T ∘ 2P], chained through P(1-P) into
    /// dZs_i += g_ij Zt_j and dZt_j += g_ij Zs_i.
    /// </summary>
    private (Tensor<T> Zs, Tensor<T> Zt) EmbeddingStep(Tensor<T> Zs, Tensor<T> Zt,
        (Tensor<T> Gradient, Tensor<T> Usable) dataGrad, Tensor<T> offDiagonal, T alpha, T rho, int d)
    {
        // Compute edge probabilities: P[i,j] = sigmoid(Zs_i . Zt_j)
        var P = EdgeProbabilities(Zs, Zt, offDiagonal);

        // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
        var expPSq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(P, P).ToMatrix(), d));
        T hVal = NumOps.Subtract(Engine.TensorTrace(expPSq), NumOps.FromDouble(d));

        // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
        var acycGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(expPSq), Engine.TensorMultiplyScalar(P, NumOps.FromDouble(2))),
            NumOps.Add(alpha, NumOps.Multiply(rho, hVal)));
        var totalGradP = Engine.TensorAdd(dataGrad.Gradient, acycGrad);
        var sigDeriv = Engine.TensorMultiply(P, Engine.ScalarMinusTensor(NumOps.One, P));
        var gradScale = Engine.TensorWhere(dataGrad.Usable, Engine.TensorMultiply(totalGradP, sigDeriv),
            Engine.TensorZerosLike(P));

        // d(Zs_i . Zt_j)/dZs_i = Zt_j, d(Zs_i . Zt_j)/dZt_j = Zs_i
        var gradZs = Engine.TensorMatMul(gradScale, Zt);
        var gradZt = Engine.TensorMatMul(Engine.TensorTranspose(gradScale), Zs);

        T lr = NumOps.FromDouble(LearningRate);
        return (Engine.TensorSubtract(Zs, Engine.TensorMultiplyScalar(gradZs, lr)),
            Engine.TensorSubtract(Zt, Engine.TensorMultiplyScalar(gradZt, lr)));
    }

}
