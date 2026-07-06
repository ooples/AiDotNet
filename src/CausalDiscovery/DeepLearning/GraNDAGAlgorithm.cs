using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// GraN-DAG — Gradient-based Neural DAG Learning.
/// </summary>
/// <remarks>
/// <para>
/// GraN-DAG parameterizes each structural equation f_j as a neural network with sigmoid
/// activations. The weighted adjacency matrix A[i,j] = ||W1_j[:,i]||_2 is derived from
/// the first-layer input weights. Path-specific connectivity through the MLP gives a
/// refined adjacency measure. The NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d
/// is enforced via augmented Lagrangian.
/// </para>
/// <para>
/// <b>For Beginners:</b> GraN-DAG trains a separate neural network for each variable to
/// predict it from the others. The "importance" of each input connection tells us the
/// causal strength, while a mathematical constraint ensures no circular causation.
/// </para>
/// <para>
/// Reference: Lachapelle et al. (2020), "Gradient-Based Neural DAG Learning", ICLR.
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
[ResearchPaper("Gradient-Based Neural DAG Learning", "https://openreview.net/forum?id=rklbKA4YDS", Year = 2020, Authors = "Sebastien Lachapelle, Philippe Brouillard, Tristan Deleu, Simon Lacoste-Julien")]
public class GraNDAGAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "GraN-DAG";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public GraNDAGAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // Standardize each variable to zero mean / unit variance before fitting — GraN-DAG's
        // preprocessing (Lachapelle 2020, experimental setup). The per-variable Gaussian-NLL score and
        // the path-norm adjacency are scale-sensitive, so without this a uniform rescaling of the data
        // (x -> 10x) changes the discovered edge set. Standardizing makes discovery invariant to input
        // scaling (DiscoverStructure_IsInvariantToDataScaling); a constant column is left at zero. Every
        // downstream computation (the MLP fit below and the covariance) then runs on the standardized data.
        // Reuse the shared DeepCausalBase.StandardizeColumns so every deep causal learner z-scores
        // identically (avoids the earlier /n vs /(n-1) variance-normalization drift). A constant column
        // still standardizes to zero — its centered values are all 0, independent of the divisor.
        data = StandardizeColumns(data);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T scale = NumOps.FromDouble(Math.Sqrt(2.0 / d));

        // Per-variable MLPs: W1[j] is d x h, W2[j] is h x 1
        var W1 = new Matrix<T>[d];
        var W2 = new Matrix<T>[d];
        for (int j = 0; j < d; j++)
        {
            W1[j] = new Matrix<T>(d, h);
            W2[j] = new Matrix<T>(h, 1);
            for (int i = 0; i < d; i++)
                if (i != j)
                    for (int k = 0; k < h; k++)
                        W1[j][i, k] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            for (int k = 0; k < h; k++)
                W2[j][k, 0] = NumOps.Multiply(scale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        }

        T lr = NumOps.FromDouble(LearningRate);
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        T rhoMax = NumOps.FromDouble(1e+16);
        double hPrev = double.PositiveInfinity;

        for (int outer = 0; outer < MaxEpochs; outer++)
        {
            for (int inner = 0; inner < 20; inner++)
            {
                // Compute loss gradients per sample per variable
                var gW1 = new Matrix<T>[d];
                var gW2 = new Matrix<T>[d];
                for (int j = 0; j < d; j++)
                {
                    gW1[j] = new Matrix<T>(d, h);
                    gW2[j] = new Matrix<T>(h, 1);
                }

                T invN = NumOps.FromDouble(1.0 / n);

                // Pre-allocate reusable vectors outside the sample/target loops
                var xRow = new Vector<T>(d);
                var hidden = new Vector<T>(h);
                var w1Col = new Vector<T>(d);
                var w2Col = new Vector<T>(h);

                for (int s = 0; s < n; s++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        // Forward using Engine.DotProduct for vectorized matmul
                        for (int i = 0; i < d; i++) xRow[i] = data[s, i];

                        for (int k = 0; k < h; k++)
                        {
                            for (int i = 0; i < d; i++) w1Col[i] = W1[j][i, k];
                            double sv = NumOps.ToDouble(Engine.DotProduct(xRow, w1Col));
                            hidden[k] = NumOps.FromDouble(sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv)));
                        }

                        for (int k = 0; k < h; k++) w2Col[k] = W2[j][k, 0];
                        T pred = Engine.DotProduct(hidden, w2Col);

                        T residual = NumOps.Multiply(NumOps.Subtract(pred, data[s, j]), invN);

                        // Backprop
                        for (int k = 0; k < h; k++)
                        {
                            gW2[j][k, 0] = NumOps.Add(gW2[j][k, 0], NumOps.Multiply(residual, hidden[k]));

                            T sigDeriv = NumOps.Multiply(hidden[k], NumOps.Subtract(NumOps.One, hidden[k]));
                            T dHidden = NumOps.Multiply(residual, NumOps.Multiply(W2[j][k, 0], sigDeriv));
                            for (int i = 0; i < d; i++)
                                gW1[j][i, k] = NumOps.Add(gW1[j][i, k], NumOps.Multiply(dHidden, data[s, i]));
                        }
                    }
                }

                // Acyclicity gradient on adjacency A[i,j] = ||W1[j][:,i]||_2
                var A = ExtractAdjacency(W1, d, h);
                T hVal = ComputeTraceExpConstraint(A, d);

                // Numerical stabilization. Unlike BCD-Nets' clamped edge logits,
                // GraN-DAG's raw MLP weights are unbounded: when the dual ratchet
                // raises rho while h plateaus (20 inner steps cannot always drive
                // h down), alpha + rho*h reaches ~1e17, the penalty gradient blows
                // the weights up, tr(exp(A.A)) overflows and every weight goes
                // NaN — the learned adjacency was NaN at EVERY learning rate.
                // Clamp the augmented coefficient (the constraint force stays
                // strong but finite) and bail to the last finite state if h has
                // already gone non-finite. Reference implementations of
                // NOTEARS-family solvers apply the same guards.
                double hD = NumOps.ToDouble(hVal);
                if (double.IsNaN(hD) || double.IsInfinity(hD)) break;
                double augD = NumOps.ToDouble(alpha) + NumOps.ToDouble(rho) * hD;
                if (augD > 1e+6) augD = 1e+6;
                T augCoeff = NumOps.FromDouble(augD);

                // Chain rule: dh/dW1 via dh/dA * dA/dW1
                var (_, hGrad) = ComputeExpGradient(A, d);
                for (int j = 0; j < d; j++)
                    for (int i = 0; i < d; i++)
                    {
                        if (i == j) continue;
                        T aij = A[i, j];
                        if (!NumOps.GreaterThan(aij, NumOps.FromDouble(1e-12))) continue;

                        T dhda = NumOps.Multiply(augCoeff, hGrad[i, j]);
                        for (int k = 0; k < h; k++)
                        {
                            T w1val = W1[j][i, k];
                            T grad = NumOps.Divide(NumOps.Multiply(dhda, w1val), aij);
                            gW1[j][i, k] = NumOps.Add(gW1[j][i, k], grad);
                        }
                    }

                // Update with per-element gradient clipping (|g| <= 10): bounds
                // the per-step weight change to 10*lr so a transiently large
                // acyclicity force cannot launch the unbounded MLP weights into
                // the exp-overflow regime (see the stabilization note above).
                T clipHi = NumOps.FromDouble(10.0);
                T clipLo = NumOps.FromDouble(-10.0);
                for (int j = 0; j < d; j++)
                {
                    for (int i = 0; i < d; i++)
                        for (int k = 0; k < h; k++)
                        {
                            T g = gW1[j][i, k];
                            if (NumOps.GreaterThan(g, clipHi)) g = clipHi;
                            else if (NumOps.GreaterThan(clipLo, g)) g = clipLo;
                            W1[j][i, k] = NumOps.Subtract(W1[j][i, k], NumOps.Multiply(lr, g));
                        }
                    for (int k = 0; k < h; k++)
                    {
                        T g = gW2[j][k, 0];
                        if (NumOps.GreaterThan(g, clipHi)) g = clipHi;
                        else if (NumOps.GreaterThan(clipLo, g)) g = clipLo;
                        W2[j][k, 0] = NumOps.Subtract(W2[j][k, 0], NumOps.Multiply(lr, g));
                    }

                    // Zero diagonal
                    for (int k = 0; k < h; k++)
                        W1[j][j, k] = NumOps.Zero;
                }
            }

            // Outer: update augmented Lagrangian per the NOTEARS dual schedule
            // GraN-DAG adopts (Lachapelle et al. 2020 §3.2 / Zheng et al. 2018):
            // rho is escalated only when h fails to SHRINK by the relative factor
            // 0.25 versus the previous outer round. The previous code compared h
            // against an ABSOLUTE 0.25, so with any nontrivial initial h the
            // penalty multiplied 10x every epoch up to 1e16 and annihilated every
            // path norm — the learned adjacency collapsed to all-zeros and no
            // edges could ever be reported.
            var Afinal = ExtractAdjacency(W1, d, h);
            T hFinal = ComputeTraceExpConstraint(Afinal, d);
            double hNow = NumOps.ToDouble(hFinal);
            if (double.IsNaN(hNow) || double.IsInfinity(hNow)) break;
            if (hNow > 1e-8)
            {
                alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hFinal));
                if (!double.IsPositiveInfinity(hPrev) && hNow > 0.25 * hPrev)
                    rho = NumOps.Multiply(rho, NumOps.FromDouble(10));
            }
            hPrev = hNow;
            if (NumOps.GreaterThan(rho, rhoMax)) break;
            if (hNow <= 1e-8) break;
        }

        var rawAdj = ExtractAdjacency(W1, d, h);
        // Use raw adjacency magnitudes as learned edge probabilities (normalize to [0,1])
        double maxNorm = 0;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j)
                    maxNorm = Math.Max(maxNorm, NumOps.ToDouble(rawAdj[i, j]));

        var learnedP = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j && maxNorm > 0)
                    learnedP[i, j] = NumOps.ToDouble(rawAdj[i, j]) / maxNorm;

        var cov = ComputeCovarianceMatrix(data);
        // Guarantee an acyclic output. BuildFinalAdjacency's direction tie-break only rules out
        // 2-cycles; a 3+-node cycle (A->B->C->A) can still survive raw thresholding — exactly the
        // DiscoverStructure_OutputIsAcyclic failure (topological sort visited 0/4 nodes). ProjectToDag
        // imposes a strict source-score topological order and keeps only forward edges, so the result
        // is a DAG by construction. Mirrors the sibling DAGGNNAlgorithm, which already routes through it.
        return BuildFinalAdjacency(ProjectToDag(learnedP, d), cov, d);
    }

    private Matrix<T> ExtractAdjacency(Matrix<T>[] W1, int d, int h)
    {
        var A = new Matrix<T>(d, d);
        for (int j = 0; j < d; j++)
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                T norm = NumOps.Zero;
                for (int k = 0; k < h; k++)
                    norm = NumOps.Add(norm, NumOps.Multiply(W1[j][i, k], W1[j][i, k]));
                A[i, j] = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(norm)));
            }
        return A;
    }

    private T ComputeTraceExpConstraint(Matrix<T> A, int d)
    {
        // h(A) = tr(e^(A∘A)) - d using power series: exp(M) = I + M + M^2/2! + ...
        var AA = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                AA[i, j] = NumOps.Multiply(A[i, j], A[i, j]);

        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) power[i, i] = NumOps.One;
        var expM = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) expM[i, i] = NumOps.One;

        for (int p = 1; p <= 10; p++)
        {
            // power = power * AA (unscaled M^p)
            power = MatMul(power, AA);
            T fact = NumOps.FromDouble(1.0 / Factorial(p));
            // expM += power / p!
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    expM[i, j] = NumOps.Add(expM[i, j], NumOps.Multiply(power[i, j], fact));
        }

        T trace = NumOps.Zero;
        for (int i = 0; i < d; i++)
            trace = NumOps.Add(trace, expM[i, i]);
        return NumOps.Subtract(trace, NumOps.FromDouble(d));
    }

    private (T h, Matrix<T> grad) ComputeExpGradient(Matrix<T> A, int d)
    {
        // Gradient: dh/dA[i,j] = 2 * A[i,j] * (e^(A∘A))^T[i,j]
        var AA = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                AA[i, j] = NumOps.Multiply(A[i, j], A[i, j]);

        var expM = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) expM[i, i] = NumOps.One;
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) power[i, i] = NumOps.One;

        for (int p = 1; p <= 10; p++)
        {
            // power = power * AA (unscaled M^p)
            power = MatMul(power, AA);
            T fact = NumOps.FromDouble(1.0 / Factorial(p));
            // expM += power / p!
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    expM[i, j] = NumOps.Add(expM[i, j], NumOps.Multiply(power[i, j], fact));
        }

        T trace = NumOps.Zero;
        for (int i = 0; i < d; i++)
            trace = NumOps.Add(trace, expM[i, i]);
        T h = NumOps.Subtract(trace, NumOps.FromDouble(d));

        var grad = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                grad[i, j] = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(A[i, j], expM[j, i]));

        return (h, grad);
    }

    private static double Factorial(int n)
    {
        double result = 1;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }
}
