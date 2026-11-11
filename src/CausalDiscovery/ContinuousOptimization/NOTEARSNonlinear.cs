using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// NOTEARS Nonlinear — continuous optimization for DAG structure learning with nonlinear (MLP) relationships.
/// </summary>
/// <remarks>
/// <para>
/// Extends the NOTEARS framework to handle nonlinear causal relationships by replacing
/// the linear model X = XW + noise with a nonlinear model X_j = f_j(Pa(X_j)) + noise,
/// where each f_j is parameterized by a small MLP (multi-layer perceptron).
/// </para>
/// <para>
/// <b>Key Differences from Linear NOTEARS:</b>
/// <list type="bullet">
/// <item>Each variable's structural equation is modeled by a 2-layer MLP</item>
/// <item>The adjacency matrix is derived from the input-layer weights: A[i,j] = ||W1_j[:,i]||_2</item>
/// <item>Acyclicity constraint: h(theta) = tr(e^(A∘A)) - d = 0, using the derived adjacency</item>
/// <item>Optimization over MLP parameters theta, not over a weight matrix W directly</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> The linear version assumes each variable is a weighted sum of its parents.
/// The nonlinear version uses small neural networks instead, so it can capture curved or complex
/// relationships between variables. For example, if income depends on age in a U-shaped curve,
/// the nonlinear version can learn that while the linear version cannot.
/// </para>
/// <para>
/// <b>Architecture per variable:</b> Input (d neurons) → Hidden (h neurons, sigmoid) → Output (1 neuron).
/// The "adjacency matrix" is extracted from the input weights: if the weights from variable i to
/// variable j's MLP are small, there's no edge i → j.
/// </para>
/// <para>
/// Reference: Zheng et al. (2020), "Learning Sparse Nonparametric DAGs", AISTATS.
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
[ResearchPaper("Learning Sparse Nonparametric DAGs", "https://proceedings.mlr.press/v108/zheng20a.html", Year = 2020, Authors = "Xun Zheng, Chen Dan, Bryon Aragam, Pradeep Ravikumar, Eric Xing")]
public class NOTEARSNonlinear<T> : ContinuousOptimizationBase<T>
{
    #region Constants

    private const double DEFAULT_RHO_MAX = 1e+16;
    private const double DEFAULT_RHO_INIT = 1.0;
    private const double RHO_MULTIPLY_FACTOR = 10.0;
    private const double CONSTRAINT_REDUCTION_THRESHOLD = 0.25;
    private const int INNER_MAX_ITERATIONS = 500;
    private const double INNER_TOLERANCE = 1e-6;
    private const double LEARNING_RATE = 0.001;
    private const int DEFAULT_HIDDEN_SIZE = 10;
    private const double ADAM_BETA1 = 0.9;
    private const double ADAM_BETA2 = 0.999;

    #endregion

    #region Fields

    private double _rhoMax = DEFAULT_RHO_MAX;
    private int _hiddenSize = DEFAULT_HIDDEN_SIZE;
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
    public override string Name => "NOTEARS Nonlinear";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes NOTEARS Nonlinear with optional configuration.
    /// </summary>
    public NOTEARSNonlinear(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
        if (options?.MaxPenalty.HasValue == true)
            _rhoMax = options.MaxPenalty.Value;
    }

    #endregion

    #region Core Algorithm

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Standardize data to prevent sigmoid saturation and ensure stable gradients.
        // Raw data values (e.g., 0-8) cause sigmoid(W1*x) ≈ 1 everywhere, killing gradients.
        data = StandardizeData(data);
        int n = data.Rows;
        int d = data.Columns;
        int h = _hiddenSize;

        InitializeMLPParameters(d, h);

        double alpha = 0.0;
        double rho = DEFAULT_RHO_INIT;
        double hOld = ComputeAcyclicityFromMLP(d);
        _lastIterations = 0;
        int adamStep = 0;

        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            _lastIterations = outerIter + 1;

            // Inner loop: Adam optimization of augmented Lagrangian
            for (int inner = 0; inner < INNER_MAX_ITERATIONS; inner++)
            {
                adamStep++;
                var (_, totalGradW1, totalGradB1, totalGradW2, totalGradB2) =
                    ComputeAugLagGradient(data, n, d, h, alpha, rho);

                AdamUpdate(totalGradW1, totalGradB1, totalGradW2, totalGradB2, d, h, adamStep);

                // Zero out diagonal entries in W1 (no self-loops)
                ZeroSelfInputs();

                // Convergence check
                if (inner % 100 == 0 && inner > 0)
                {
                    double gradNorm = ComputeGradientNorm(totalGradW1, totalGradB1, totalGradW2, totalGradB2, d, h);
                    if (gradNorm < INNER_TOLERANCE)
                        break;
                }
            }

            double hNew = ComputeAcyclicityFromMLP(d);

            if (hNew > CONSTRAINT_REDUCTION_THRESHOLD * hOld)
            {
                rho *= RHO_MULTIPLY_FACTOR;
            }
            else
            {
                alpha += rho * hNew;
                hOld = hNew;
            }

            _lastH = hNew;

            if (hNew < HTolerance || rho >= _rhoMax)
                break;
        }

        // Extract adjacency from MLP weights
        var W = ExtractAdjacencyMatrix(d);
        var (loss, _1, _2, _3, _4) = ComputeAugLagGradient(data, n, d, h, 0, 0);
        _lastLoss = loss;

        var result = ThresholdAndClean(W, WThreshold);

        // Fallback: if thresholding removed all edges (e.g., near-degenerate data where
        // all MLP weights converge to similar small values), use correlation-based detection.
        var offDiagonalMagnitude = Engine.TensorMultiply(
            Engine.TensorAbs(Tensor<T>.FromMatrix(result)), OffDiagonalMask(d));
        bool hasEdges = Engine.TensorCountNonzero(Engine.TensorGreaterThan(offDiagonalMagnitude, NumOps.Zero)) > 0;

        var candidate = hasEdges ? result : FallbackCorrelationGraph(data);
        return EnforceAcyclic(candidate);
    }

    #endregion

    #region MLP Operations

    private void InitializeMLPParameters(int d, int h)
    {
        const int DefaultSeed = 42;
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(DefaultSeed);
        double scale = Math.Sqrt(2.0 / d); // He initialization

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

    private double ComputeAcyclicityFromMLP(int d)
    {
        var A = ExtractAdjacencyMatrix(d);
        var (h, _) = ComputeNOTEARSConstraint(A);
        return h;
    }

    #endregion

    #region Gradient Computation

    private (double loss, Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2)
        ComputeAugLagGradient(Matrix<T> data, int n, int d, int h, double alpha, double rho)
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

        // Acyclicity gradient through adjacency
        var A = ExtractAdjacencyMatrix(d);
        var (hVal, hGrad) = ComputeNOTEARSConstraint(A);
        double augFactor = alpha + rho * hVal;

        // Chain rule: dh/dW1[j][i,k] = dh/dA[i,j] * dA[i,j]/dW1[j][i,k], with dA[i,j]/dW1[j][i,k] = W1[j][i,k] / A[i,j],
        // plus the L1 penalty on W1, for i != j with A[i, j] >= 1e-12.
        var adjacency = Engine.TensorTranspose(Tensor<T>.FromMatrix(A)); // [j, i]
        var dhda = Engine.TensorMultiplyScalar(Engine.TensorTranspose(Tensor<T>.FromMatrix(hGrad)), NumOps.FromDouble(augFactor));
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

        return (totalLoss, gW1, gB1, gW2, gB2);
    }

    private void AdamUpdate(Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2,
        int d, int h, int step)
    {
        double bc1 = 1 - Math.Pow(ADAM_BETA1, step);
        double bc2 = 1 - Math.Pow(ADAM_BETA2, step);

        (_W1, _mW1, _vW1) = AdamUpdate(_W1, gW1, _mW1, _vW1, ADAM_BETA1, ADAM_BETA2, 1e-8, LEARNING_RATE, bc1, bc2);
        (_b1, _mb1, _vb1) = AdamUpdate(_b1, gB1, _mb1, _vb1, ADAM_BETA1, ADAM_BETA2, 1e-8, LEARNING_RATE, bc1, bc2);
        (_W2, _mW2, _vW2) = AdamUpdate(_W2, gW2, _mW2, _vW2, ADAM_BETA1, ADAM_BETA2, 1e-8, LEARNING_RATE, bc1, bc2);
        (_b2, _mb2, _vb2) = AdamUpdate(_b2, gB2, _mb2, _vb2, ADAM_BETA1, ADAM_BETA2, 1e-8, LEARNING_RATE, bc1, bc2);
    }

    private double ComputeGradientNorm(Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2, int d, int h)
    {
        T sumOfSquares = NumOps.Add(
            NumOps.Add(Engine.TensorSum(Engine.TensorSquare(gW1)), Engine.TensorSum(Engine.TensorSquare(gB1))),
            NumOps.Add(Engine.TensorSum(Engine.TensorSquare(gW2)), Engine.TensorSum(Engine.TensorSquare(gB2))));
        return Math.Sqrt(NumOps.ToDouble(sumOfSquares));
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
