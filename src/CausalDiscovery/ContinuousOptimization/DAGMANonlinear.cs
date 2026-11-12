using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// DAGMA Nonlinear — DAG learning via M-matrices and log-determinant with MLP structural equations.
/// </summary>
/// <remarks>
/// <para>
/// Extends DAGMA to nonlinear settings by parameterizing each structural equation with a small MLP.
/// Uses the same log-determinant acyclicity constraint as DAGMA Linear but applied to the adjacency
/// matrix extracted from MLP input-layer weights.
/// </para>
/// <para>
/// <b>Constraint:</b> h(theta, s) = -log det(sI - A∘A) + d*log(s), where A[i,j] = ||W1_j[:,i]||_2
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like DAGMA Linear but can discover non-linear causal relationships
/// using small neural networks. While DAGMA Linear assumes Y = a*X + b, this version can learn
/// curved relationships like Y = sin(X) or Y = X^2.
/// </para>
/// <para>
/// Reference: Bello et al. (2022), "DAGMA: Learning DAGs via M-matrices and a Log-Determinant
/// Acyclicity Characterization", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/2e6a3b0e4e43db1cbba0c5e5b7ee8a41-Abstract-Conference.html", Year = 2022, Authors = "Kevin Bello, Bryon Aragam, Pradeep Ravikumar")]
public class DAGMANonlinear<T> : ContinuousOptimizationBase<T>
{
    #region Constants

    private const double DAGMA_DEFAULT_LAMBDA1 = 0.03;
    private const int DEFAULT_T = 5;
    private const double DEFAULT_MU_INIT = 1.0;
    private const double DEFAULT_MU_FACTOR = 0.1;
    private const double DEFAULT_LEARNING_RATE = 0.001;
    private const double ADAM_BETA1 = 0.99;
    private const double ADAM_BETA2 = 0.999;
    private const int DEFAULT_WARM_ITER = 5000;
    private const int DEFAULT_MAX_ITER = 10000;
    private const double INNER_CONVERGENCE_TOL = 1e-6;
    private const int CHECKPOINT_INTERVAL = 500;
    private const int DEFAULT_HIDDEN_SIZE = 10;
    private const double EDGE_TOLERANCE = 1e-12;

    #endregion

    #region Fields

    private readonly double[] _sValues;
    private readonly int _hiddenSize;
    private readonly double _learningRate;
    // Preserve the paper defaults when unset, while allowing callers (and the
    // generated CI fixture) to bound the expensive inner optimization explicitly.
    private int? _configuredMaxIterations;
    private int _lastIterations;
    private double _lastH;
    private double _lastLoss;

    // MLP parameters, one MLP per output variable j:
    // _W1 [j, i, k] (input i to hidden k), _b1 [j, k], _W2 [j, k], _b2 [j].
    private Tensor<T> _W1 = new Tensor<T>(new[] { 1 });
    private Tensor<T> _b1 = new Tensor<T>(new[] { 1 });
    private Tensor<T> _W2 = new Tensor<T>(new[] { 1 });
    private Tensor<T> _b2 = new Tensor<T>(new[] { 1 });

    // 1 where input i may feed MLP j (i != j), broadcast over hidden units: [j, i, k].
    private Tensor<T> _inputMask = new Tensor<T>(new[] { 1 });

    // Adam state
    private Tensor<T> _mW1 = new Tensor<T>(new[] { 1 }), _vW1 = new Tensor<T>(new[] { 1 });
    private Tensor<T> _mb1 = new Tensor<T>(new[] { 1 }), _vb1 = new Tensor<T>(new[] { 1 });
    private Tensor<T> _mW2 = new Tensor<T>(new[] { 1 }), _vW2 = new Tensor<T>(new[] { 1 });
    private Tensor<T> _mb2 = new Tensor<T>(new[] { 1 }), _vb2 = new Tensor<T>(new[] { 1 });

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override string Name => "DAGMA Nonlinear";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes DAGMA Nonlinear with optional configuration.
    /// </summary>
    private int _seed = 42;

    public DAGMANonlinear(CausalDiscoveryOptions? options = null)
    {
        Lambda1 = DAGMA_DEFAULT_LAMBDA1;
        ApplyOptions(options);

        _hiddenSize = options?.HiddenUnits ?? DEFAULT_HIDDEN_SIZE;
        if (_hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(options), $"HiddenUnits must be > 0; got {_hiddenSize}.");

        _learningRate = options?.LearningRate ?? DEFAULT_LEARNING_RATE;
        if (double.IsNaN(_learningRate) || double.IsInfinity(_learningRate) || _learningRate <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(options), $"LearningRate must be a positive finite value; got {_learningRate}.");

        _configuredMaxIterations = options?.MaxIterations;
        if (options?.Seed.HasValue == true) _seed = options.Seed.Value;
        _sValues = [1.0, 0.9, 0.8, 0.7, 0.6];
    }

    #endregion

    #region Core Algorithm

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Standardize data to prevent sigmoid saturation and ensure stable gradients.
        data = StandardizeData(data);
        int n = data.Rows;
        int d = data.Columns;
        int h = _hiddenSize;

        InitializeMLPParameters(d, h);

        double mu = DEFAULT_MU_INIT;
        _lastIterations = 0;
        int adamStep = 0;
        int T = Math.Min(DEFAULT_T, _sValues.Length);

        for (int t = 0; t < T; t++)
        {
            double s = _sValues[t];
                int defaultInner = (t < T - 1) ? DEFAULT_WARM_ITER : DEFAULT_MAX_ITER;
                int maxInner = _configuredMaxIterations.HasValue
                    ? Math.Min(defaultInner, _configuredMaxIterations.Value)
                    : defaultInner;
            double prevObj = double.MaxValue;

            for (int inner = 1; inner <= maxInner; inner++)
            {
                adamStep++;
                var (obj, gW1, gB1, gW2, gB2) = ComputeObjectiveAndGradients(data, n, d, h, mu, s);

                AdamUpdate(gW1, gB1, gW2, gB2, d, h, adamStep);

                // Zero diagonal in W1
                ZeroSelfInputs();

                if (inner % CHECKPOINT_INTERVAL == 0)
                {
                    if (Math.Abs(obj - prevObj) / (Math.Abs(prevObj) + 1e-15) < INNER_CONVERGENCE_TOL)
                        break;
                    prevObj = obj;
                }
            }

            mu *= DEFAULT_MU_FACTOR;
        }

        _lastIterations = adamStep;
        var A = ExtractAdjacencyMatrix(d);
        var (finalLoss, _, _, _, _) = ComputeObjectiveAndGradients(data, n, d, h, 0, _sValues[^1]);
        _lastLoss = finalLoss;
        _lastH = ComputeLogDetConstraintFromA(A, _sValues[^1], d).H;

        // The shared base projection, not a local copy. The copy that used to live here sorted
        // edges by magnitude with no tiebreak, and List<T>.Sort is introsort, which is not stable --
        // two edges of equal magnitude were ordered arbitrarily, so the projected DAG could differ
        // between runs on identical input. EnforceAcyclic breaks ties by (From, To) for exactly that
        // reason, asks the cheaper "does `to` already reach `from`" question instead of rescanning
        // the whole graph after every insertion, and traverses iteratively rather than recursing to
        // depth d. NOTEARSNonlinear and NOTEARSSobolev already use it.
        return EnforceAcyclic(ThresholdAndClean(A, WThreshold));
    }

    #endregion

    #region MLP Operations

    private void InitializeMLPParameters(int d, int h)
    {
        var rng = RandomHelper.CreateSeededRandom(_seed);
        double scale = Math.Sqrt(2.0 / d);

        // Draw order of the reference initialization: for each output variable j, the (d-1)*h
        // first-layer weights (inputs i != j, hidden k inner) and then its h second-layer weights.
        var draws = Tensor<T>.CreateRandom(rng, d, d * h);
        var uniform = Engine.TensorSubtract(
            Engine.TensorMultiplyScalar(draws, NumOps.FromDouble(scale)),
            Engine.TensorFullLike(draws, NumOps.FromDouble(scale / 2.0)));

        // W1[j, i, k] = draw[j, slot(i) * h + k] where slot skips input j; self inputs stay zero.
        var w1Source = new int[d * d * h];
        for (int j = 0; j < d; j++)
            for (int i = 0; i < d; i++)
                for (int k = 0; k < h; k++)
                    w1Source[(j * d + i) * h + k] = j * d * h + (i == j ? 0 : ((i < j ? i : i - 1) * h + k));

        _inputMask = Engine.TensorBroadcastTo(OffDiagonalMask(d).Reshape(new[] { d, d, 1 }), new[] { d, d, h });
        var w1 = Engine.TensorIndexSelect(uniform.Reshape(new[] { d * d * h }),
            new Tensor<int>(w1Source, new[] { d * d * h }), 0).Reshape(new[] { d, d, h });
        _W1 = Engine.TensorWhere(_inputMask, w1, Engine.TensorZerosLike(w1));
        _W2 = Engine.TensorSlice(uniform, new[] { 0, (d - 1) * h }, new[] { d, h });
        _b1 = new Tensor<T>(new[] { d, h });
        _b2 = new Tensor<T>(new[] { d });

        _mW1 = new Tensor<T>(new[] { d, d, h }); _vW1 = new Tensor<T>(new[] { d, d, h });
        _mb1 = new Tensor<T>(new[] { d, h }); _vb1 = new Tensor<T>(new[] { d, h });
        _mW2 = new Tensor<T>(new[] { d, h }); _vW2 = new Tensor<T>(new[] { d, h });
        _mb2 = new Tensor<T>(new[] { d }); _vb2 = new Tensor<T>(new[] { d });
    }

    /// <summary>Sets the first-layer weights from input j into MLP j to zero.</summary>
    private void ZeroSelfInputs()
    {
        _W1 = Engine.TensorWhere(_inputMask, _W1, Engine.TensorZerosLike(_W1));
    }

    /// <summary>
    /// Runs every per-variable MLP on every sample.
    /// </summary>
    /// <returns>Predictions [n, d] and sigmoid hidden activations [n, d, h].</returns>
    private (Tensor<T> Output, Tensor<T> Hidden) ForwardMLP(Tensor<T> x, int n, int d, int h)
    {
        // pre[s, j, k] = b1[j, k] + sum_i x[s, i] * W1[j, i, k]
        var weightsByInput = Engine.Reshape(Engine.TensorPermute(_W1, new[] { 1, 0, 2 }), new[] { d, d * h });
        var pre = Engine.TensorMatMul(x, weightsByInput).Reshape(new[] { n, d, h });
        pre = Engine.TensorAdd(pre, Engine.TensorBroadcastTo(_b1.Reshape(new[] { 1, d, h }), new[] { n, d, h }));
        var hidden = Engine.TensorSigmoid(pre);

        // output[s, j] = b2[j] + sum_k hidden[s, j, k] * W2[j, k]
        var weighted = Engine.TensorMultiply(hidden,
            Engine.TensorBroadcastTo(_W2.Reshape(new[] { 1, d, h }), new[] { n, d, h }));
        var output = Engine.TensorAdd(Engine.ReduceSum(weighted, new[] { 2 }, false),
            Engine.TensorBroadcastTo(_b2.Reshape(new[] { 1, d }), new[] { n, d }));

        return (output, hidden);
    }

    /// <summary>A[i, j] = ||W1[j, i, :]||_2 for i != j, zero on the diagonal.</summary>
    private Matrix<T> ExtractAdjacencyMatrix(int d)
    {
        var norms = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(_W1), new[] { 2 }, false)); // [j, i]
        var A = Engine.TensorTranspose(norms);
        return Engine.TensorWhere(OffDiagonalMask(d), A, Engine.TensorZerosLike(A)).ToMatrix();
    }

    private (double H, Matrix<T> Gradient) ComputeLogDetConstraintFromA(Matrix<T> A, double s, int d)
    {
        // M = sI - A∘A
        var a = Tensor<T>.FromMatrix(A);
        var M = Engine.TensorSubtract(Engine.TensorMultiplyScalar(Engine.TensorEye<T>(d), NumOps.FromDouble(s)),
            Engine.TensorMultiply(a, a));

        double logDet = LogAbsDeterminant(M);
        double hVal = -logDet + d * Math.Log(s);

        var invM = InvertSmallMatrix(M.ToMatrix());
        if (invM == null)
        {
            System.Diagnostics.Trace.TraceWarning("DAGMA-NL: M matrix is singular; returning zero gradient for h(W).");
            return (hVal, new Matrix<T>(d, d));
        }

        T two = NumOps.FromDouble(2.0);
        var gradient = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(a, Engine.TensorTranspose(Tensor<T>.FromMatrix(invM))), two);
        return (hVal, gradient.ToMatrix());
    }

    #endregion

    #region Gradient Computation

    private (double obj, Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2)
        ComputeObjectiveAndGradients(Matrix<T> data, int n, int d, int h, double mu, double s)
    {
        var x = Tensor<T>.FromMatrix(data);
        var (pred, hidden) = ForwardMLP(x, n, d, h);
        var residual = Engine.TensorSubtract(pred, x);
        double totalLoss = NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(residual)));

        // Backward pass through every MLP, summed over samples
        var dOutput = Engine.TensorDivideScalar(residual, NumOps.FromDouble(n));
        var gB2 = Engine.ReduceSum(dOutput, new[] { 0 }, false);
        var dOutputPerUnit = Engine.TensorBroadcastTo(dOutput.Reshape(new[] { n, d, 1 }), new[] { n, d, h });
        var gW2 = Engine.ReduceSum(Engine.TensorMultiply(dOutputPerUnit, hidden), new[] { 0 }, false);

        // dHidden = dOutput * W2 * hidden * (1 - hidden)
        var dHidden = Engine.TensorMultiply(
            Engine.TensorMultiply(
                Engine.TensorMultiply(dOutputPerUnit,
                    Engine.TensorBroadcastTo(_W2.Reshape(new[] { 1, d, h }), new[] { n, d, h })),
                hidden),
            Engine.ScalarMinusTensor(NumOps.One, hidden));
        var gB1 = Engine.ReduceSum(dHidden, new[] { 0 }, false);

        // gW1[j, i, k] = sum_s x[s, i] * dHidden[s, j, k]
        var gradByInput = Engine.TensorMatMul(Engine.TensorTranspose(x), dHidden.Reshape(new[] { n, d * h }));
        var gW1 = Engine.Reshape(Engine.TensorPermute(gradByInput.Reshape(new[] { d, d, h }), new[] { 1, 0, 2 }),
            new[] { d, d, h });

        totalLoss *= 0.5 / n;

        // DAGMA constraint gradient
        var A = ExtractAdjacencyMatrix(d);
        var (hVal, hGrad) = ComputeLogDetConstraintFromA(A, s, d);
        double obj = totalLoss + Lambda1 * ComputeL1NormFromA(A, d) + mu * hVal;

        // For i != j with A[i, j] >= 1e-12:
        // gW1[j, i, k] += mu * hGrad[i, j] * W1[j, i, k] / A[i, j] + lambda1 * sign(W1[j, i, k])
        var adjacency = Engine.TensorTranspose(Tensor<T>.FromMatrix(A)); // [j, i]
        var dhda = Engine.TensorMultiplyScalar(Engine.TensorTranspose(Tensor<T>.FromMatrix(hGrad)), NumOps.FromDouble(mu));
        var ones = Engine.TensorOnesLike(adjacency);
        var usable = Engine.TensorMultiply(OffDiagonalMask(d),
            Engine.TensorSubtract(ones, Engine.TensorLessThan(adjacency, NumOps.FromDouble(1e-12))));
        var shape = new[] { d, d, h };
        var usablePerUnit = Engine.TensorBroadcastTo(usable.Reshape(new[] { d, d, 1 }), shape);
        var safeAdjacency = Engine.TensorWhere(usable, adjacency, ones);
        var constraintGrad = Engine.TensorDivide(
            Engine.TensorMultiply(Engine.TensorBroadcastTo(dhda.Reshape(new[] { d, d, 1 }), shape), _W1),
            Engine.TensorBroadcastTo(safeAdjacency.Reshape(new[] { d, d, 1 }), shape));
        var extra = Engine.TensorAdd(constraintGrad,
            Engine.TensorMultiplyScalar(Engine.TensorSign(_W1), NumOps.FromDouble(Lambda1)));
        gW1 = Engine.TensorAdd(gW1, Engine.TensorWhere(usablePerUnit, extra, Engine.TensorZerosLike(extra)));

        return (obj, gW1, gB1, gW2, gB2);
    }

    private void AdamUpdate(Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2,
        int d, int h, int step)
    {
        double bc1 = 1 - Math.Pow(ADAM_BETA1, step);
        double bc2 = 1 - Math.Pow(ADAM_BETA2, step);

        (_W1, _mW1, _vW1) = AdamUpdate(_W1, gW1, _mW1, _vW1, ADAM_BETA1, ADAM_BETA2, 1e-8, _learningRate, bc1, bc2);
        (_b1, _mb1, _vb1) = AdamUpdate(_b1, gB1, _mb1, _vb1, ADAM_BETA1, ADAM_BETA2, 1e-8, _learningRate, bc1, bc2);
        (_W2, _mW2, _vW2) = AdamUpdate(_W2, gW2, _mW2, _vW2, ADAM_BETA1, ADAM_BETA2, 1e-8, _learningRate, bc1, bc2);
        (_b2, _mb2, _vb2) = AdamUpdate(_b2, gB2, _mb2, _vb2, ADAM_BETA1, ADAM_BETA2, 1e-8, _learningRate, bc1, bc2);
    }

    private double ComputeL1NormFromA(Matrix<T> A, int d)
    {
        var a = Tensor<T>.FromMatrix(A);
        var offDiagonal = Engine.TensorWhere(OffDiagonalMask(d), Engine.TensorAbs(a), Engine.TensorZerosLike(a));
        return NumOps.ToDouble(Engine.TensorSum(offDiagonal));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the iteration count and convergence info from the last run.
    /// </summary>
    public (int Iterations, double FinalH, double FinalLoss) GetLastRunInfo()
    {
        return (_lastIterations, _lastH, _lastLoss);
    }

    #endregion
}
