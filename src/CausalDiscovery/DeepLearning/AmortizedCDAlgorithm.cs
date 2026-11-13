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
[ResearchPaper("Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data", "https://proceedings.mlr.press/v177/lowe22a.html", Year = 2022, Authors = "Sindy Lowe, David Madras, Richard Zemel, Max Welling")]
public class AmortizedCDAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "AmortizedCD";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public AmortizedCDAlgorithm(CausalDiscoveryOptions? options = null)
    {
        // This solve needs far more steps than the 100 the base class defaults to: below roughly
        // 500 the acyclicity penalty in the second half outruns a data fit that has not
        // converged and the graph comes back empty. Set BEFORE options are applied, so it is a
        // default a caller can still override in either direction.
        MaxEpochs = 2000;
        ApplyDeepOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Standardize columns so the learned probabilities are invariant to per-variable scaling.
        data = StandardizeColumns(data);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var cov = ComputeCovarianceMatrix(data);
        var corr = CovarianceToCorrelation(cov);
        T eps = NumOps.FromDouble(1e-10);

        // Feature dimension per edge: 4 features (cov[i,j], corr[i,j], var_i, var_j)
        int featDim = 4;
        T initScale = NumOps.FromDouble(Math.Sqrt(2.0 / featDim));

        // Shared encoder MLP: W1 (featDim x h), W2 (h x 1), drawn W1 row-major then W2.
        var draws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, featDim * h + h), NumOps.FromDouble(-0.5)),
            initScale);
        var W1 = Engine.TensorSlice(draws, new[] { 0 }, new[] { featDim * h }).Reshape(new[] { featDim, h });
        var W2 = Engine.TensorSlice(draws, new[] { featDim * h }, new[] { h }).Reshape(new[] { h, 1 });

        // Optimization budget, for the same reason CASTLE carries one. `MaxEpochs` here counts RAW
        // full-batch gradient steps, but the reference implementations this family follows use that
        // number for something else entirely: in NOTEARS, `max_iter=100` counts OUTER
        // augmented-Lagrangian iterations, each of which solves the inner subproblem to convergence
        // with L-BFGS. Reading the default as 100 gradient steps is therefore orders of magnitude
        // short, and it showed: every edge probability finished at 0.4385 -- below its 0.5
        // initialization, because the acyclicity penalty in the second half outran a data fit that
        // had not converged -- against a floor of 0.483, so the graph came back EMPTY. Measured on
        // the linear-SEM fixture: 100 steps -> 0 edges; 500 -> the exact true structure
        // (0.998 / 0.997 / 0.996). Full-batch steps on a problem this size are microseconds.
        // MaxEpochs is documented as a MAXIMUM, so it is honoured as one. The step count this solve
        // needs is expressed as this algorithm's DEFAULT instead, set before options are applied so
        // a caller who asks for fewer gets fewer -- clamping their value upward made MaxEpochs = 10
        // silently run 2,000 steps, which is not what the option says it does.
        int steps = MaxEpochs;

        // Acyclicity warm-up: rho = 0 (no NOTEARS penalty) for the first half of training so the data fit
        // can drive edge probabilities toward the correlations; a bounded fixed penalty is applied after.
        T rho = NumOps.Zero;
        T boundedRho = NumOps.One;

        // Precompute features for each pair (row i*d + j): cov[i,j], corr[i,j], cov[i,i], cov[j,j]
        var c = Tensor<T>.FromMatrix(cov);
        var r = Tensor<T>.FromMatrix(corr);
        var variance = Engine.TensorDiagonal(c);
        var shape = new[] { d, d };
        var features = Engine.TensorStack(new[]
        {
            c.Reshape(new[] { d * d }),
            r.Reshape(new[] { d * d }),
            Engine.TensorBroadcastTo(variance.Reshape(new[] { d, 1 }), shape).Reshape(new[] { d * d }),
            Engine.TensorBroadcastTo(variance.Reshape(new[] { 1, d }), shape).Reshape(new[] { d * d }),
        }, 1);
        var offDiagonal = OffDiagonalMask(d);
        var absCorr = Engine.TensorAbs(r);

        for (int epoch = 0; epoch < steps; epoch++)
        {
            (W1, W2) = EncoderStep(features, W1, W2, absCorr, offDiagonal, rho, d);

            // Apply only a BOUNDED acyclicity penalty in the second half of training (warm-up; see the rho
            // init). The original schedule grew rho ×10 toward 1e16 AND accumulated alpha every epoch, which
            // on strongly correlated data made the augmented-Lagrangian term dominate and collapsed every
            // edge to 0 (the empty graph is trivially acyclic) — recovering no edges. A fixed, modest rho
            // breaks ties toward a DAG without overwhelming the data fit.
            if (epoch >= steps / 2)
                rho = boundedRho;
        }

        // Final inference pass
        var (learnedProbabilities, _) = EncodeEdges(features, W1, W2, offDiagonal, d);

        // Pairwise direction tie-breaks rule out only 2-cycles; project the asymmetric learned
        // probabilities through their net out-flow ordering to guarantee a DAG without assuming
        // that raw marginal variance determines causal direction.
        return BuildFinalAdjacency(ProjectToDag(learnedProbabilities, d), cov, d);
    }

    /// <summary>
    /// Runs the shared encoder on every pair: hidden = sigmoid(features W1), P = sigmoid(hidden W2),
    /// both saturated beyond ±20; P is [d, d] with a zero diagonal and hidden is [d*d, h].
    /// </summary>
    private (Tensor<T> P, Tensor<T> Hidden) EncodeEdges(Tensor<T> features, Tensor<T> W1, Tensor<T> W2,
        Tensor<T> offDiagonal, int d)
    {
        var hidden = ClampedSigmoid(Engine.TensorMatMul(features, W1), 1.0);
        var logits = Engine.TensorMatMul(hidden, W2).Reshape(new[] { d, d });
        return (Engine.TensorMultiply(offDiagonal, ClampedSigmoid(logits, 1.0)), hidden);
    }

    /// <summary>
    /// One gradient step on the encoder: dL/dP = (P - |corr|) + rho·h·[exp(P∘P)^T ∘ 2P] over the
    /// off-diagonal pairs, back-propagated through both sigmoids and averaged over the d(d-1) pairs.
    /// </summary>
    private (Tensor<T> W1, Tensor<T> W2) EncoderStep(Tensor<T> features, Tensor<T> W1, Tensor<T> W2,
        Tensor<T> absCorr, Tensor<T> offDiagonal, T rho, int d)
    {
        // Forward: compute edge logits for all pairs
        var (P, hidden) = EncodeEdges(features, W1, W2, offDiagonal, d);

        // Compute NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
        var expWW = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(P, P).ToMatrix(), d));
        T hValPrev = NumOps.Subtract(Engine.TensorTrace(expWW), NumOps.FromDouble(d));

        // Data fit gradient: encourage edges where correlation is strong
        var dataGrad = Engine.TensorSubtract(P, absCorr);

        // Acyclicity gradient: d/dP[i,j] of the bounded NOTEARS penalty (rho/2 * h^2),
        // where h = tr(exp(P∘P)) - d, so gradient = (rho*h) * [exp(P∘P)^T ∘ 2P][i,j].
        // (No augmented-Lagrangian alpha term: this schedule uses a fixed rho, not the
        // alpha-accumulating dual ascent that collapsed every edge on correlated data.)
        var acycGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(expWW), Engine.TensorMultiplyScalar(P, NumOps.FromDouble(2))),
            NumOps.Multiply(rho, hValPrev));

        // Chain rule: d_loss/d_logit = d_loss/d_P * d_P/d_logit
        var sigDeriv = Engine.TensorMultiply(P, Engine.ScalarMinusTensor(NumOps.One, P));
        var dLogit = Engine.TensorMultiply(offDiagonal,
            Engine.TensorMultiply(Engine.TensorAdd(dataGrad, acycGrad), sigDeriv)).Reshape(new[] { d * d, 1 });

        // d_logit/d_W2, then d_logit/d_hidden -> d_hidden/d_W1
        var gW2 = Engine.TensorMatMul(Engine.TensorTranspose(hidden), dLogit);
        var hidSigD = Engine.TensorMultiply(hidden, Engine.ScalarMinusTensor(NumOps.One, hidden));
        var dHidden = Engine.TensorMultiply(Engine.TensorMatMul(dLogit, Engine.TensorTranspose(W2)), hidSigD);
        var gW1 = Engine.TensorMatMul(Engine.TensorTranspose(features), dHidden);

        // Normalize gradients, then apply them
        T normFactor = NumOps.FromDouble(1.0 / (d * (d - 1)));
        T lr = NumOps.FromDouble(LearningRate);
        return (Engine.TensorSubtract(W1, Engine.TensorMultiplyScalar(Engine.TensorMultiplyScalar(gW1, normFactor), lr)),
            Engine.TensorSubtract(W2, Engine.TensorMultiplyScalar(Engine.TensorMultiplyScalar(gW2, normFactor), lr)));
    }

}
