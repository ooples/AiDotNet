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
    /// <summary>Group-lasso weight λ1 on the rows of Wh_j.</summary>
    private const double GroupLassoWeight = 0.02;

    /// <summary>Offset inside the group-norm square root of the group-lasso subgradient.</summary>
    private const double GroupNormEpsilon = 1e-8;

    /// <summary>Adam first-moment decay β1.</summary>
    private const double AdamBeta1 = 0.9;

    /// <summary>Adam second-moment decay β2.</summary>
    private const double AdamBeta2 = 0.999;

    /// <summary>Adam denominator offset ε.</summary>
    private const double AdamEpsilon = 1e-8;

    /// <inheritdoc/>
    public override string Name => "CASTLE";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CASTLEAlgorithm(CausalDiscoveryOptions? options = null)
    {
        // See AmortizedCDAlgorithm: this solve needs far more than the base class's 100 steps.
        // Set BEFORE options are applied so it stays an overridable default.
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

        // ---- Standardize columns to zero mean / unit variance (NOTEARS, Zheng
        // et al. 2018; CASTLE, Kyono et al. 2020, both preprocess this way).
        // Vectorized via Engine: tile the per-column mean/inv-std into [n,d] with a
        // ones[n,1] @ row[1,d] matmul (no broadcasting assumptions), then subtract /
        // scale element-wise. Without standardization the method is data-scale
        // dependent (DiscoverStructure_IsInvariantToDataScaling fails) and variables
        // far down a causal chain have tiny variance, so their parent-mask gradients
        // never separate from the 0.5 init. ----
        var X = new Tensor<T>(new[] { n, d }, data);
        var onesN1 = Engine.TensorOnesLike(new Tensor<T>(new[] { n, 1 }));
        var meanRow = Engine.ReduceMean(X, new[] { 0 }, true);                  // [1,d]
        var centered = Engine.TensorSubtract(X, Engine.TensorMatMul(onesN1, meanRow));
        var varRow = Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 0 }, true); // [1,d]
        var positiveVariance = Engine.TensorGreaterThan(varRow, NumOps.FromDouble(1e-12));
        var invStdRow = Engine.TensorWhere(positiveVariance,
            Engine.TensorReciprocal(Engine.TensorSqrt(Engine.TensorWhere(positiveVariance, varRow, Engine.TensorOnesLike(varRow)))),
            Engine.TensorZerosLike(varRow));
        var Xs = Engine.TensorMultiply(centered, Engine.TensorMatMul(onesN1, invStdRow)); // [n,d]

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        int D = d * h;

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
        //
        // All d networks are stored side by side: Wh [i, j*h + k] = Wh_j[i, k] and
        // Wo [j*h + k] = Wo_j[k], so one matmul runs every network at once.
        var (Wh, Wo, inputMask) = InitializeCastleNetworks(rng, d, h);
        // Adam first/second-moment state.
        var mWh = new Tensor<T>(new[] { d, D });
        var vWh = new Tensor<T>(new[] { d, D });
        var mWo = new Tensor<T>(new[] { D });
        var vWo = new Tensor<T>(new[] { D });

        // Optimization budget. The augmented-Lagrangian solve needs enough full-batch
        // steps for each f_j to fit and for the constraint to bind; the tiny default
        // MaxEpochs (100) is a floor, not a ceiling — full-batch steps on this small
        // problem are microseconds, so we run a robust number of them (the paper uses
        // Adam over many epochs). Reconstruction-first warmup (no L1 / acyclicity)
        // lets the true-parent weights grow before sparsity + the DAG constraint prune.
        // MaxEpochs is documented as a MAXIMUM, so it is honoured as one. The step count this solve
        // needs is expressed as this algorithm's DEFAULT instead, set before options are applied so
        // a caller who asks for fewer gets fewer -- clamping their value upward made MaxEpochs = 10
        // silently run 2,000 steps, which is not what the option says it does.
        int steps = MaxEpochs;
        int warmupSteps = steps / 4;

        // Adam hyperparameters (CASTLE optimizes with Adam).
        double lrA = Math.Max(LearningRate, 0.01);
        T gradScale = NumOps.FromDouble(2.0 / n);   // d/dpred of mean-squared error

        // CASTLE HOLDS rho AND alpha FIXED AT 1.0. The reference implementation
        // (github.com/trentkyono/CASTLE, CASTLE.py) sets rho_i = np.array([[1.0]]) and
        // alpha_i = np.array([[1.0]]) and never updates them: the penalty is simply
        // 0.5*rho*h*h + alpha*h with both coefficients constant for the whole run.
        //
        // This code had NOTEARS' augmented-Lagrangian schedule instead -- dual ascent on alpha plus
        // a x10 escalation of rho every dual step. That belongs to a different algorithm. Over 40
        // dual updates rho compounds to ~1e40, and because the empty graph is trivially acyclic an
        // unbounded penalty does not merely outweigh the reconstruction term, it annihilates it:
        // every first-layer group norm is driven below w_threshold and the algorithm returns NO
        // edges. The diagnostic tell was that the failure got WORSE with a larger step budget --
        // 10000 steps recovered an edge, 12000 recovered none -- which is the opposite of an
        // under-training curve and the signature of a penalty that never stops growing.
        // The gradient of α·h + (ρ/2)·h² is (α + ρ·h)·∇h.
        const double AcyclicityMultiplier = 1.0;
        const double AcyclicityPenaltyWeight = 1.0;
        T alpha = NumOps.FromDouble(AcyclicityMultiplier);
        T rho = NumOps.FromDouble(AcyclicityPenaltyWeight);

        // ---- Pre-allocate every [n, ·] per-step tensor ONCE (allocation-flat inner loop).
        // The generated tests run inside a TensorArena scope, which does not release rented buffers
        // until the scope exits, so per-step [n, d*h] temporaries accumulate across the >= 2000 steps
        // and can exhaust the heap. Every sample-sized op therefore writes into a reused destination;
        // only parameter-sized ([d, d*h]) temporaries are allocated per step. ----
        var XsT = Engine.TensorTranspose(Xs);                     // [d,n] — loop-invariant
        var networkOf = BlockDiagonalSelector(d, h);              // [d*h, d], 1 where unit j*h+k belongs to network j
        var buffers = new CastleBuffers(n, d, D);

        for (int step = 0; step < steps; step++)
        {
            bool constrain = step >= warmupSteps;

            // 1) Reconstruction forward/backward for every f_j.
            var (gWh, gWo) = ReconstructionGradients(Xs, XsT, Wh, Wo, networkOf, gradScale, buffers);

            // 2) Once past warmup, add the group-L1 sparsity and augmented-Lagrangian
            //    acyclicity gradients to the first-layer weights.
            if (constrain)
                gWh = AddSparsityAndAcyclicityGradient(gWh, Wh, alpha, rho, d, h);

            // 3) Adam update for every network's weights (t = step+1). The self-row Wh_j[j,:] is held at zero.
            int t = step + 1;
            (Wh, mWh, vWh) = CastleAdamStep(Wh, gWh, mWh, vWh, t, lrA, inputMask);
            (Wo, mWo, vWo) = CastleAdamStep(Wo, gWo, mWo, vWo, t, lrA, null);
        }

        // Adjacency = first-layer weight-row norms A[i,j] = ‖Wh_j[i,:]‖. CASTLE's
        // reference implementation thresholds these raw group norms at w_threshold=0.3;
        // normalizing by the largest edge would make every other edge's apparent
        // strength depend on that one edge and can promote weak rows as n changes.
        // Direction is resolved by A[i,j] vs A[j,i]; edge magnitude comes from the
        // covariance ratio.
        var cov = ComputeCovarianceMatrix(Xs.ToMatrix());
        var learnedP = Engine.TensorSqrt(SquaredGroupNorms(Wh, d, h));
        return BuildFinalAdjacency(learnedP, cov, d, learnedThreshold: 0.3);
    }

    /// <summary>Reused [n, ·] scratch tensors for the reconstruction pass.</summary>
    private sealed class CastleBuffers
    {
        public CastleBuffers(int n, int d, int units)
        {
            HPre = new Tensor<T>(new[] { n, units });
            H = new Tensor<T>(new[] { n, units });
            HT = new Tensor<T>(new[] { units, n });
            NegH = new Tensor<T>(new[] { n, units });
            OneMinusH = new Tensor<T>(new[] { n, units });
            DPred = new Tensor<T>(new[] { n, units });
            DPredH = new Tensor<T>(new[] { n, units });
            DH = new Tensor<T>(new[] { n, units });
            Pred = new Tensor<T>(new[] { n, d });
            ResidRaw = new Tensor<T>(new[] { n, d });
            Resid = new Tensor<T>(new[] { n, d });
            OutputGrad = new Tensor<T>(new[] { units, d });
        }

        public Tensor<T> HPre { get; }
        public Tensor<T> H { get; }
        public Tensor<T> HT { get; }
        public Tensor<T> NegH { get; }
        public Tensor<T> OneMinusH { get; }
        public Tensor<T> DPred { get; }
        public Tensor<T> DPredH { get; }
        public Tensor<T> DH { get; }
        public Tensor<T> Pred { get; }
        public Tensor<T> ResidRaw { get; }
        public Tensor<T> Resid { get; }
        public Tensor<T> OutputGrad { get; }
    }

    /// <summary>
    /// Seeded initialization in the reference draw order: for each network j, Wh_j[i, k] for i != j
    /// (k inner), then Wo_j[k]; every value is sqrt(2/d) * (u - 0.5). Returns Wh [d, d*h], Wo [d*h]
    /// and the mask [d, d*h] that is zero on each network's self-row.
    /// </summary>
    private (Tensor<T> Wh, Tensor<T> Wo, Tensor<T> InputMask) InitializeCastleNetworks(Random rng, int d, int h)
    {
        int D = d * h;
        var draws = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Tensor<T>.CreateRandom(rng, d, D), NumOps.FromDouble(-0.5)),
            NumOps.FromDouble(Math.Sqrt(2.0 / d)));

        // Wh [i, j*h + k] = draw[j, slot(i)*h + k], slot skipping i == j.
        var source = new int[d * D];
        var selfRow = new bool[d * D];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                for (int k = 0; k < h; k++)
                {
                    int position = i * D + j * h + k;
                    source[position] = j * D + (i == j ? 0 : ((i < j ? i : i - 1) * h + k));
                    selfRow[position] = i == j;
                }

        var gathered = Engine.TensorIndexSelect(draws.Reshape(new[] { d * D }),
            new Tensor<int>(source, new[] { d * D }), 0);
        var zeros = Engine.TensorZerosLike(gathered);
        var inputMask = Engine.TensorWhere(selfRow, zeros, Engine.TensorOnesLike(gathered)).Reshape(new[] { d, D });
        var wh = Engine.TensorWhere(selfRow, zeros, gathered).Reshape(new[] { d, D });

        // Wo [j*h + k] = draw[j, (d-1)*h + k]
        var wo = Engine.TensorSlice(draws, new[] { 0, (d - 1) * h }, new[] { d, h }).Reshape(new[] { D });
        return (wh, wo, inputMask);
    }

    /// <summary>[d*h, d] selector with 1 where hidden unit j*h + k belongs to network j.</summary>
    private Tensor<T> BlockDiagonalSelector(int d, int h)
    {
        var networkIndex = new int[d * h];
        for (int u = 0; u < d * h; u++) networkIndex[u] = u / h;
        return Engine.TensorIndexSelect(Engine.TensorEye<T>(d), new Tensor<int>(networkIndex, new[] { d * h }), 0);
    }

    /// <summary>
    /// Mean-squared-error gradients of every network: pred_j = sigmoid(Xs Wh_j) Wo_j,
    /// resid = (2/n)(pred - x); gWo = H^T resid (own network only), gWh = Xs^T (resid Wo^T ∘ H ∘ (1 - H)).
    /// </summary>
    private (Tensor<T> GWh, Tensor<T> GWo) ReconstructionGradients(Tensor<T> Xs, Tensor<T> XsT, Tensor<T> Wh,
        Tensor<T> Wo, Tensor<T> networkOf, T gradScale, CastleBuffers b)
    {
        int d = Wh.Shape[0];
        int D = Wh.Shape[1];
        var axes2 = new[] { 1, 0 };

        // Block-diagonal output weights: WoBlock[j*h + k, j] = Wo_j[k]
        var woBlock = Engine.TensorMultiply(networkOf, Engine.TensorBroadcastTo(Wo.Reshape(new[] { D, 1 }), new[] { D, d }));

        Engine.MatMulInto(b.HPre, Xs, Wh);                                          // [n,d*h]
        Engine.SigmoidInto(b.H, b.HPre);                                            // [n,d*h]
        Engine.MatMulInto(b.Pred, b.H, woBlock);                                    // [n,d]

        // resid = (2/n)·(pred − x)
        Engine.TensorSubtractInto(b.ResidRaw, b.Pred, Xs);
        Engine.TensorMultiplyScalarInto(b.Resid, b.ResidRaw, gradScale);            // [n,d]

        Engine.TransposeInto(b.HT, b.H, axes2);
        Engine.MatMulInto(b.OutputGrad, b.HT, b.Resid);                             // [d*h,d]
        var gWo = Engine.ReduceSum(Engine.TensorMultiply(b.OutputGrad, networkOf), new[] { 1 }, false); // [d*h]

        Engine.MatMulInto(b.DPred, b.Resid, Engine.TensorTranspose(woBlock));       // [n,d*h]
        Engine.TensorMultiplyScalarInto(b.NegH, b.H, NumOps.FromDouble(-1.0));
        Engine.TensorAddScalarInto(b.OneMinusH, b.NegH, NumOps.One);
        Engine.TensorMultiplyInto(b.DPredH, b.DPred, b.H);
        Engine.TensorMultiplyInto(b.DH, b.DPredH, b.OneMinusH);                     // [n,d*h]
        var gWh = new Tensor<T>(new[] { d, D });
        Engine.MatMulInto(gWh, XsT, b.DH);                                          // [d,d*h]
        return (gWh, gWo);
    }

    /// <summary>A_sq[i, j] = ‖Wh_j[i,:]‖² off the diagonal, zero on it.</summary>
    private Tensor<T> SquaredGroupNorms(Tensor<T> Wh, int d, int h)
    {
        var sums = Engine.ReduceSum(Engine.TensorSquare(Wh.Reshape(new[] { d, d, h })), new[] { 2 }, false);
        return Engine.TensorMultiply(sums, OffDiagonalMask(d));
    }

    /// <summary>
    /// Adds the group-lasso subgradient λ1·Wh_j[i,k]/‖Wh_j[i,:]‖ and the acyclicity gradient
    /// 2(α+ρh)·exp(A_sq)[j,i]·Wh_j[i,k] to the off-diagonal first-layer rows. The adjacency used
    /// by the constraint is A_sq[i,j] = ‖Wh_j[i,:]‖² (the squared influence of variable i on
    /// network j); h(A_sq) = tr(exp(A_sq)) − d, and ∂h/∂A_sq[i,j] = exp(A_sq)^T[i,j] = exp(A_sq)[j,i].
    /// No dual ascent and no rho escalation: the reference keeps both fixed at 1.0 for the whole run.
    /// </summary>
    private Tensor<T> AddSparsityAndAcyclicityGradient(Tensor<T> gWh, Tensor<T> Wh, T alpha, T rho, int d, int h)
    {
        var aSq = SquaredGroupNorms(Wh, d, h);
        var expAsq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(aSq.ToMatrix(), d));
        T hVal = NumOps.Subtract(Engine.TensorTrace(expAsq), NumOps.FromDouble(d));
        T twoAlRhoH = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Add(alpha, NumOps.Multiply(rho, hVal)));

        // Row norm ‖Wh_j[i,:]‖ for the group-lasso subgradient: λ1 / sqrt(A_sq + 1e-8)
        var l1Coef = Engine.TensorReciprocal(Engine.TensorSqrt(Engine.TensorAddScalar(aSq, NumOps.FromDouble(GroupNormEpsilon))));
        l1Coef = Engine.TensorMultiplyScalar(l1Coef, NumOps.FromDouble(GroupLassoWeight));
        var acycCoef = Engine.TensorMultiplyScalar(Engine.TensorTranspose(expAsq), twoAlRhoH);
        var coef = Engine.TensorMultiply(Engine.TensorAdd(l1Coef, acycCoef), OffDiagonalMask(d));

        var shape = new[] { d, d, h };
        var perWeight = Engine.TensorMultiply(Engine.TensorBroadcastTo(coef.Reshape(new[] { d, d, 1 }), shape),
            Wh.Reshape(shape));
        return Engine.TensorAdd(gWh, perWeight.Reshape(new[] { d, d * h }));
    }

    /// <summary>
    /// Adam (β1 0.9, β2 0.999, ε 1e-8): p - lr·m̂/(sqrt(v̂)+ε); entries where <paramref name="mask"/>
    /// is zero keep their parameter and moments unchanged.
    /// </summary>
    private (Tensor<T> Parameter, Tensor<T> M, Tensor<T> V) CastleAdamStep(Tensor<T> parameter, Tensor<T> g,
        Tensor<T> m, Tensor<T> v, int t, double lr, Tensor<T>? mask)
    {
        var newM = Engine.TensorAdd(Engine.TensorMultiplyScalar(m, NumOps.FromDouble(AdamBeta1)),
            Engine.TensorMultiplyScalar(g, NumOps.FromDouble(1.0 - AdamBeta1)));
        var newV = Engine.TensorAdd(Engine.TensorMultiplyScalar(v, NumOps.FromDouble(AdamBeta2)),
            Engine.TensorMultiplyScalar(Engine.TensorMultiply(g, g), NumOps.FromDouble(1.0 - AdamBeta2)));
        var mhat = Engine.TensorDivideScalar(newM, NumOps.FromDouble(1.0 - Math.Pow(AdamBeta1, t)));
        var vhat = Engine.TensorDivideScalar(newV, NumOps.FromDouble(1.0 - Math.Pow(AdamBeta2, t)));
        var stepVal = Engine.TensorDivide(Engine.TensorMultiplyScalar(mhat, NumOps.FromDouble(lr)),
            Engine.TensorAddScalar(Engine.TensorSqrt(vhat), NumOps.FromDouble(AdamEpsilon)));
        var updated = Engine.TensorSubtract(parameter, stepVal);
        if (mask is null)
            return (updated, newM, newV);

        return (Engine.TensorWhere(mask, updated, parameter), Engine.TensorWhere(mask, newM, m),
            Engine.TensorWhere(mask, newV, v));
    }

}
