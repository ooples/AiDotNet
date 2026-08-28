using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// CASTLE — Causal Structure Learning via neural networks with shared masked architecture.
/// </summary>
/// <remarks>
/// <para>
/// CASTLE trains one neural sub-network f_j per variable to reconstruct x_j from the
/// other variables. The causal adjacency is read directly from each sub-network's
/// first-layer weights — A[i,j] = ‖Wh_j[i,:]‖, the influence of variable i on the
/// reconstruction of variable j — rather than from a separate gate. Sparsity is imposed
/// by a group-lasso penalty on those input-weight rows, and acyclicity by the NOTEARS
/// constraint h(A) = tr(exp(A∘A)) − d = 0, enforced with an augmented Lagrangian.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Standardize the data; initialize one MLP f_j per target (self-input held at 0)</item>
/// <item>Forward: H = σ(X·Wh_j), pred = H·Wo_j; reconstruction MSE against x_j</item>
/// <item>After a reconstruction-first warmup, add group-L1 on the Wh_j rows and the
/// augmented-Lagrangian acyclicity gradient on A_sq[i,j] = ‖Wh_j[i,:]‖²</item>
/// <item>Update all weights with Adam; dual-ascend α and escalate ρ to drive h→0</item>
/// <item>Read adjacency from the final weight-row norms, normalize, and threshold</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CASTLE trains a little predictor for each variable that guesses
/// it from the others. How strongly each other variable feeds into that predictor tells
/// us how likely it is a cause. A sparsity penalty drops weak links and an acyclicity
/// rule keeps the result a valid causal graph (no loops).
/// </para>
/// <para>
/// Reference: Kyono et al. (2020), "CASTLE: Regularization via Auxiliary Causal Graph Discovery", NeurIPS.
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
[ResearchPaper("CASTLE: Regularization via Auxiliary Causal Graph Discovery", "https://proceedings.neurips.cc/paper/2020/hash/1f8d87e1461a3d422a3e0eaa8e945e19-Abstract.html", Year = 2020, Authors = "Trent Kyono, Yao Zhang, Mihaela van der Schaar")]
public class CASTLEAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CASTLE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CASTLEAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int h = HiddenUnits;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        // ---- Standardize columns to zero mean / unit variance (NOTEARS, Zheng
        // et al. 2018; CASTLE, Kyono et al. 2020, both preprocess this way).
        // Vectorized via Engine: tile the per-column mean/inv-std into [n,d] with a
        // ones[n,1] @ row[1,d] matmul (no broadcasting assumptions), then subtract /
        // scale element-wise. Without standardization the method is data-scale
        // dependent (DiscoverStructure_IsInvariantToDataScaling fails) and variables
        // far down a causal chain have tiny variance, so their parent-mask gradients
        // never separate from the 0.5 init. ----
        var X = new Tensor<T>(new[] { n, d }, data);
        var onesN1 = new Tensor<T>(new[] { n, 1 });
        for (int s = 0; s < n; s++) onesN1[s, 0] = NumOps.One;
        var meanRow = Engine.ReduceMean(X, new[] { 0 }, true);                  // [1,d]
        var centered = Engine.TensorSubtract(X, Engine.TensorMatMul(onesN1, meanRow));
        var varRow = Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 0 }, true); // [1,d]
        var invStdRow = new Tensor<T>(new[] { 1, d });
        for (int j = 0; j < d; j++)
        {
            T v = varRow[0, j];
            invStdRow[0, j] = NumOps.GreaterThan(v, NumOps.FromDouble(1e-12))
                ? NumOps.Divide(NumOps.One, NumOps.Sqrt(v)) : NumOps.Zero;
        }
        var Xs = Engine.TensorMultiply(centered, Engine.TensorMatMul(onesN1, invStdRow)); // [n,d]

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        T initScale = NumOps.FromDouble(Math.Sqrt(2.0 / d));

        // ---- Per-variable sub-networks (CASTLE, Kyono et al. 2020 / NOTEARS-MLP,
        // Zheng et al. 2020). Each target variable j has its OWN MLP f_j that
        // reconstructs x_j from the OTHER variables. The causal adjacency is NOT a
        // separate learned gate — it is read directly from the first-layer weights:
        // A[i,j] = ‖Wh_j[i,:]‖. A separate sigmoid-mask gate (the earlier design) has
        // a fatal chicken-and-egg failure: the MLPs start random, so routing real
        // inputs through untrained weights HURTS reconstruction versus predicting the
        // standardized mean (0), so the gate gradient drives every edge to zero before
        // any network can learn to use its true parents (RecoversTrueEdges found 0/3,
        // all probabilities 0.000). Reading structure straight from the weight norms
        // removes the gate, so training simply grows the weights for genuine parents.
        // x_j is excluded from its own predictor by holding Wh_j[j,:] = 0 (no self-
        // edge / trivial identity map). ----
        var Wh = new Tensor<T>[d];
        var Wo = new Tensor<T>[d];
        // Adam first/second-moment state, one pair per network.
        var mWh = new Tensor<T>[d];
        var vWh = new Tensor<T>[d];
        var mWo = new Tensor<T>[d];
        var vWo = new Tensor<T>[d];
        for (int j = 0; j < d; j++)
        {
            Wh[j] = new Tensor<T>(new[] { d, h });
            Wo[j] = new Tensor<T>(new[] { h, 1 });
            mWh[j] = new Tensor<T>(new[] { d, h });
            vWh[j] = new Tensor<T>(new[] { d, h });
            mWo[j] = new Tensor<T>(new[] { h, 1 });
            vWo[j] = new Tensor<T>(new[] { h, 1 });
            for (int i = 0; i < d; i++)
                for (int k = 0; k < h; k++)
                    Wh[j][i, k] = i == j ? NumOps.Zero
                        : NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
            for (int k = 0; k < h; k++)
                Wo[j][k, 0] = NumOps.Multiply(initScale, NumOps.FromDouble(rng.NextDouble() - 0.5));
        }

        // Optimization budget. The augmented-Lagrangian solve needs enough full-batch
        // steps for each f_j to fit and for the constraint to bind; the tiny default
        // MaxEpochs (100) is a floor, not a ceiling — full-batch steps on this small
        // problem are microseconds, so we run a robust number of them (the paper uses
        // Adam over many epochs). Reconstruction-first warmup (no L1 / acyclicity)
        // lets the true-parent weights grow before sparsity + the DAG constraint prune.
        int steps = Math.Max(MaxEpochs, 2000);
        int warmupSteps = steps / 4;
        int dualEvery = Math.Max(1, (steps - warmupSteps) / 40);

        // Adam hyperparameters (CASTLE optimizes with Adam).
        double lrA = Math.Max(LearningRate, 0.01);
        T beta1 = NumOps.FromDouble(0.9);
        T beta2 = NumOps.FromDouble(0.999);
        T oneMinusB1 = NumOps.FromDouble(0.1);
        T oneMinusB2 = NumOps.FromDouble(0.001);
        T adamEps = NumOps.FromDouble(1e-8);
        T lambda1 = NumOps.FromDouble(0.02);   // group-lasso sparsity weight
        T groupEps = NumOps.FromDouble(1e-8);
        T gradScale = NumOps.FromDouble(2.0 / n);   // d/dpred of mean-squared error

        // Augmented-Lagrangian state for the acyclicity constraint h(A)=0 (NOTEARS):
        // the gradient of α·h + (ρ/2)·h² is (α + ρ·h)·∇h. Dual ascent on α plus
        // penalty escalation on ρ drives h→0 (a true DAG) and, with the group-L1 term,
        // prunes indirect / cyclic edges (e.g. X0→X2 in the chain X0→X1→X2).
        T alpha = NumOps.Zero;
        T rho = NumOps.One;
        T hPrev = NumOps.FromDouble(double.PositiveInfinity);

        // ---- Pre-allocate every per-step tensor ONCE (allocation-flat inner loop).
        // The solve runs `steps` (>= 2000) full-batch iterations over d sub-networks. Allocating
        // the forward/backward temporaries per iteration churned ~8 tensors of [n,h] per network
        // per step — at n=400, h=64, d=4 that is ~7.4 MB/step, so ~14.7 GB across the solve. The
        // generated tests run inside a TensorArena scope, which does not release rented buffers
        // until the scope exits, so that churn accumulated and blew the 16 GB heap limit (the
        // test host died with a fatal OOM inside AutoTensorCache.RentOrAllocate, aborting the
        // whole shard). Hoisting the invariants (Xs^T and the target columns never change) and
        // writing every op into a reused destination keeps the footprint flat at ~1.6 MB for the
        // entire solve. Operand order is unchanged, so the math is identical. ----
        var XsT = Engine.TensorTranspose(Xs);                     // [d,n] — loop-invariant
        var targetCols = new Tensor<T>[d];                        // x_j columns — loop-invariant
        for (int j = 0; j < d; j++)
        {
            targetCols[j] = new Tensor<T>(new[] { n, 1 });
            for (int s = 0; s < n; s++) targetCols[j][s, 0] = Xs[s, j];
        }

        // Gradient accumulators: written by phase 1, adjusted by phase 2, consumed by phase 3,
        // so they are per-network and live for the whole step (but allocated only once).
        var gWh = new Tensor<T>[d];
        var gWo = new Tensor<T>[d];
        for (int j = 0; j < d; j++)
        {
            gWh[j] = new Tensor<T>(new[] { d, h });
            gWo[j] = new Tensor<T>(new[] { h, 1 });
        }

        // Scratch reused across BOTH the network loop and the step loop: each value is consumed
        // within the iteration that writes it.
        var hPre = new Tensor<T>(new[] { n, h });
        var H = new Tensor<T>(new[] { n, h });
        var HT = new Tensor<T>(new[] { h, n });
        var negH = new Tensor<T>(new[] { n, h });
        var oneMinusH = new Tensor<T>(new[] { n, h });
        var dPred = new Tensor<T>(new[] { n, h });
        var dPredH = new Tensor<T>(new[] { n, h });
        var dH = new Tensor<T>(new[] { n, h });
        var pred = new Tensor<T>(new[] { n, 1 });
        var residRaw = new Tensor<T>(new[] { n, 1 });
        var resid = new Tensor<T>(new[] { n, 1 });
        var WoT = new Tensor<T>(new[] { 1, h });
        T negOne = NumOps.FromDouble(-1.0);
        var axes2 = new[] { 1, 0 };

        for (int step = 0; step < steps; step++)
        {
            bool constrain = step >= warmupSteps;

            // 1) Reconstruction forward/backward for every f_j (all O(n) work is
            //    vectorized via Engine matmuls). Gradients are accumulated and applied
            //    only after the cross-network acyclicity term is added below.
            for (int j = 0; j < d; j++)
            {
                Engine.MatMulInto(hPre, Xs, Wh[j]);               // [n,h]
                Engine.SigmoidInto(H, hPre);                      // [n,h]
                Engine.MatMulInto(pred, H, Wo[j]);                // [n,1]

                // resid = (2/n)·(pred − x_j)
                Engine.TensorSubtractInto(residRaw, pred, targetCols[j]);
                Engine.TensorMultiplyScalarInto(resid, residRaw, gradScale);                // [n,1]

                Engine.TransposeInto(HT, H, axes2);
                Engine.MatMulInto(gWo[j], HT, resid);                                       // [h,1]
                Engine.TransposeInto(WoT, Wo[j], axes2);
                Engine.MatMulInto(dPred, resid, WoT);                                       // [n,h]
                Engine.TensorMultiplyScalarInto(negH, H, negOne);
                Engine.TensorAddScalarInto(oneMinusH, negH, NumOps.One);
                Engine.TensorMultiplyInto(dPredH, dPred, H);
                Engine.TensorMultiplyInto(dH, dPredH, oneMinusH);                           // [n,h]
                Engine.MatMulInto(gWh[j], XsT, dH);                                         // [d,h]
            }

            // 2) Once past warmup, add the group-L1 sparsity and augmented-Lagrangian
            //    acyclicity gradients to the first-layer weights. The adjacency used by
            //    the constraint is A_sq[i,j] = ‖Wh_j[i,:]‖² (the squared influence of
            //    variable i on network j); h(A_sq) = tr(exp(A_sq)) − d, and
            //    ∂h/∂A_sq[i,j] = exp(A_sq)^T[i,j] = exp(A_sq)[j,i]. Chaining
            //    ∂A_sq[i,j]/∂Wh_j[i,k] = 2·Wh_j[i,k] gives the per-weight acyclicity
            //    gradient (α+ρh)·exp(A_sq)[j,i]·2·Wh_j[i,k].
            if (constrain)
            {
                var Asq = new Matrix<T>(d, d);
                for (int j = 0; j < d; j++)
                    for (int i = 0; i < d; i++)
                    {
                        if (i == j) continue;
                        T s2 = NumOps.Zero;
                        for (int k = 0; k < h; k++)
                            s2 = NumOps.Add(s2, NumOps.Multiply(Wh[j][i, k], Wh[j][i, k]));
                        Asq[i, j] = s2;
                    }
                var expAsq = MatrixExponentialTaylor(Asq, d);
                T hVal = NumOps.Zero;
                for (int i = 0; i < d; i++) hVal = NumOps.Add(hVal, expAsq[i, i]);
                hVal = NumOps.Subtract(hVal, NumOps.FromDouble(d));
                T alRhoH = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));   // α + ρ·h
                T twoAlRhoH = NumOps.Multiply(NumOps.FromDouble(2.0), alRhoH);

                for (int j = 0; j < d; j++)
                    for (int i = 0; i < d; i++)
                    {
                        if (i == j) continue;
                        // Row norm ‖Wh_j[i,:]‖ for the group-lasso subgradient.
                        T rowNorm = NumOps.Sqrt(NumOps.Add(Asq[i, j], groupEps));
                        T l1Coef = NumOps.Divide(lambda1, rowNorm);                       // λ1 / ‖·‖
                        T acycCoef = NumOps.Multiply(twoAlRhoH, expAsq[j, i]);            // 2(α+ρh)·exp[j,i]
                        T coef = NumOps.Add(l1Coef, acycCoef);
                        for (int k = 0; k < h; k++)
                            gWh[j][i, k] = NumOps.Add(gWh[j][i, k], NumOps.Multiply(coef, Wh[j][i, k]));
                    }

                // Dual ascent on the constraint (periodically — once the inner solve
                // has had dualEvery steps to reduce the loss at the current α,ρ).
                if ((step - warmupSteps) % dualEvery == dualEvery - 1)
                {
                    alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hVal));
                    double hAbs = Math.Abs(NumOps.ToDouble(hVal));
                    if (hAbs > 0.25 * Math.Abs(NumOps.ToDouble(hPrev)))
                        rho = NumOps.Multiply(rho, NumOps.FromDouble(10.0));
                    hPrev = hVal;
                }
            }

            // 3) Adam update for every network's weights. Bias-correction factors for
            //    this step (t = step+1). The self-row Wh_j[j,:] is held at zero.
            int t = step + 1;
            T bc1 = NumOps.FromDouble(1.0 - Math.Pow(0.9, t));
            T bc2 = NumOps.FromDouble(1.0 - Math.Pow(0.999, t));
            T lrAt = NumOps.FromDouble(lrA);
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    if (i == j) continue;   // self-row stays zero
                    for (int k = 0; k < h; k++)
                    {
                        T g = gWh[j][i, k];
                        T m = NumOps.Add(NumOps.Multiply(beta1, mWh[j][i, k]), NumOps.Multiply(oneMinusB1, g));
                        T v = NumOps.Add(NumOps.Multiply(beta2, vWh[j][i, k]), NumOps.Multiply(oneMinusB2, NumOps.Multiply(g, g)));
                        mWh[j][i, k] = m;
                        vWh[j][i, k] = v;
                        T mhat = NumOps.Divide(m, bc1);
                        T vhat = NumOps.Divide(v, bc2);
                        T stepVal = NumOps.Divide(NumOps.Multiply(lrAt, mhat),
                            NumOps.Add(NumOps.Sqrt(vhat), adamEps));
                        Wh[j][i, k] = NumOps.Subtract(Wh[j][i, k], stepVal);
                    }
                }
                for (int k = 0; k < h; k++)
                {
                    T g = gWo[j][k, 0];
                    T m = NumOps.Add(NumOps.Multiply(beta1, mWo[j][k, 0]), NumOps.Multiply(oneMinusB1, g));
                    T v = NumOps.Add(NumOps.Multiply(beta2, vWo[j][k, 0]), NumOps.Multiply(oneMinusB2, NumOps.Multiply(g, g)));
                    mWo[j][k, 0] = m;
                    vWo[j][k, 0] = v;
                    T mhat = NumOps.Divide(m, bc1);
                    T vhat = NumOps.Divide(v, bc2);
                    T stepVal = NumOps.Divide(NumOps.Multiply(lrAt, mhat),
                        NumOps.Add(NumOps.Sqrt(vhat), adamEps));
                    Wo[j][k, 0] = NumOps.Subtract(Wo[j][k, 0], stepVal);
                }
            }
        }

        // Adjacency = first-layer weight-row norms A[i,j] = ‖Wh_j[i,:]‖. CASTLE's
        // reference implementation thresholds these raw group norms at w_threshold=0.3;
        // normalizing by the largest edge would make every other edge's apparent
        // strength depend on that one edge and can promote weak rows as n changes.
        // Direction is resolved by A[i,j] vs A[j,i]; edge magnitude comes from the
        // covariance ratio.
        var cov = ComputeCovarianceMatrix(Xs.ToMatrix());
        var learnedP = new double[d, d];
        for (int j = 0; j < d; j++)
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                double s2 = 0.0;
                for (int k = 0; k < h; k++)
                {
                    double w = NumOps.ToDouble(Wh[j][i, k]);
                    s2 += w * w;
                }
                double a = Math.Sqrt(s2);
                learnedP[i, j] = a;
            }

        return BuildFinalAdjacency(learnedP, cov, d, learnedThreshold: 0.3);
    }

}
