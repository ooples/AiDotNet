using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ContinuousOptimization;

/// <summary>
/// NOTEARS with Sobolev regularization — DAG learning with smoothness constraints.
/// </summary>
/// <remarks>
/// <para>
/// Extends NOTEARS nonlinear by adding a Sobolev-norm penalty on the functional relationships,
/// which encourages smooth causal mechanisms. The Sobolev penalty penalizes the squared L2 norm
/// of the Jacobian (first derivatives) of each MLP, preventing overfitting to noise.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize per-variable MLPs: Input(d) → Hidden(h, sigmoid) → Output(1)</item>
/// <item>Compute L2 reconstruction loss plus Sobolev penalty on Jacobian norms</item>
/// <item>Extract adjacency A[i,j] = ||W1[j][:,i]||_2 from input weights</item>
/// <item>Apply NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d</item>
/// <item>Optimize via Adam with augmented Lagrangian for acyclicity</item>
/// <item>Threshold the final adjacency matrix</item>
/// </list>
/// </para>
/// <para>
/// <b>Sobolev Penalty:</b> For each MLP f_j, the Sobolev penalty is:
/// <c>sum_i (df_j/dx_i)^2</c> averaged over data samples. This penalizes the sensitivity
/// of each function to its inputs, encouraging smooth relationships. The Jacobian
/// df_j/dx_i = sum_k W2[j][k] * sigmoid'(z_k) * W1[j][i,k] is computed analytically.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular NOTEARS with neural networks might learn very wiggly functions
/// that fit noise rather than real causal relationships. The Sobolev penalty encourages smoother
/// functions, similar to how L2 regularization prevents large weights — but it penalizes the
/// derivatives (wigglyness) of the learned functions, not just their magnitude.
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
public class NOTEARSSobolev<T> : ContinuousOptimizationBase<T>
{
    /// <summary>
    /// Maximum number of entries of one [rows, d, h] activation or [d, rows, d] Jacobian block in the
    /// gradient pass; samples are processed in blocks of at most this many entries.
    /// </summary>
    private const int MaxSampleBlockElements = 1 << 22;

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
    private const double DEFAULT_SOBOLEV_WEIGHT = 0.1;

    private double _rhoMax = DEFAULT_RHO_MAX;
    private int _hiddenSize = DEFAULT_HIDDEN_SIZE;
    private double _sobolevWeight = DEFAULT_SOBOLEV_WEIGHT;
    private readonly int _seed;

    /// <summary>
    /// Seed used when the caller does not supply one, so that a run is reproducible by default.
    /// </summary>
    /// <remarks>
    /// Previously this fell back to an unseeded secure RNG, which made the model nondeterministic
    /// and its generated model-family tests flaky: the identical test binary produced 13 failures
    /// in one run and 8 in the next over the same 42 tests, which makes regression detection on
    /// this path impossible. Callers still override via the options' Seed property. Matches the
    /// fixed-seed convention already used by DAGMANonlinear in this module.
    /// </remarks>
    private const int DefaultRandomSeed = 42;
    // MLP parameters per variable:
    // One MLP per output variable j: _W1 [j, i, k] (input i to hidden k), _b1 [j, k], _W2 [j, k], _b2 [j].
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

    /// <inheritdoc/>
    public override string Name => "NOTEARS Sobolev";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    /// <summary>
    /// Initializes NOTEARS Sobolev with optional configuration.
    /// </summary>
    public NOTEARSSobolev(CausalDiscoveryOptions? options = null)
    {
        ApplyOptions(options);
        if (options?.MaxPenalty is { } maxPenalty)
        {
            if (double.IsNaN(maxPenalty) || double.IsInfinity(maxPenalty) || maxPenalty <= 0)
                throw new ArgumentException("MaxPenalty must be a positive finite number.");
            _rhoMax = maxPenalty;
        }
        if (options?.HiddenUnits is { } hiddenUnits)
        {
            if (hiddenUnits < 1)
                throw new ArgumentException("HiddenUnits must be at least 1.");
            _hiddenSize = hiddenUnits;
        }
        _seed = options?.Seed ?? DefaultRandomSeed;
        _sobolevWeight = options?.SobolevWeight ?? DEFAULT_SOBOLEV_WEIGHT;
        if (_sobolevWeight < 0 || double.IsNaN(_sobolevWeight) || double.IsInfinity(_sobolevWeight))
            throw new ArgumentException("SobolevWeight must be a non-negative finite number.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        data = StandardizeData(data);
        int n = data.Rows;
        int d = data.Columns;
        int h = _hiddenSize;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        InitializeMLPParameters(d, h);

        double alpha = 0.0;
        double rho = DEFAULT_RHO_INIT;
        double hOld = ComputeAcyclicityFromMLP(d);
        int adamStep = 0;

        for (int outerIter = 0; outerIter < MaxIterations; outerIter++)
        {
            for (int inner = 0; inner < INNER_MAX_ITERATIONS; inner++)
            {
                adamStep++;
                var (_, gradW1, gradB1, gradW2, gradB2) =
                    ComputeAugLagGradientWithSobolev(data, n, d, h, alpha, rho);

                AdamUpdate(gradW1, gradB1, gradW2, gradB2, adamStep);

                // Zero diagonal (no self-loops)
                ZeroSelfInputs();

                if (inner % 100 == 0 && inner > 0)
                {
                    double gradNorm = ComputeGradientNorm(gradW1, gradB1, gradW2, gradB2);
                    if (gradNorm < INNER_TOLERANCE) break;
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

            if (hNew < HTolerance || rho >= _rhoMax) break;
        }

        var W = ExtractAdjacencyMatrix(d);
        var result = ThresholdAndClean(W, WThreshold);

        // Fallback if all edges removed
        var offDiagonalMagnitude = Engine.TensorMultiply(
            Engine.TensorAbs(Tensor<T>.FromMatrix(result)), OffDiagonalMask(d));
        bool hasEdges = Engine.TensorCountNonzero(Engine.TensorGreaterThan(offDiagonalMagnitude, NumOps.Zero)) > 0;

        var candidate = hasEdges ? result : FallbackCorrelationGraph(data);
        return EnforceAcyclic(candidate);
    }

    private void InitializeMLPParameters(int d, int h)
    {
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed);
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
    /// Forward pass of every per-variable MLP on every sample:
    /// output = W2[j]^T * sigmoid(W1[j]^T * x + b1[j]) + b2[j], with the sigmoid saturated to
    /// exactly 1 above 20 and 0 below -20.
    /// </summary>
    /// <returns>Predictions [n, d] and hidden activations [n, d, h].</returns>
    private (Tensor<T> Output, Tensor<T> Hidden) ForwardMLP(Tensor<T> x, int n, int d, int h)
    {
        var weightsByInput = Engine.Reshape(Engine.TensorPermute(_W1, new[] { 1, 0, 2 }), new[] { d, d * h });
        var pre = Engine.TensorMatMul(x, weightsByInput).Reshape(new[] { n, d, h });
        pre = Engine.TensorAdd(pre, Engine.TensorBroadcastTo(_b1.Reshape(new[] { 1, d, h }), new[] { n, d, h }));

        var hidden = Engine.TensorSigmoid(pre);
        hidden = Engine.TensorWhere(Engine.TensorGreaterThan(pre, NumOps.FromDouble(20)), Engine.TensorOnesLike(pre), hidden);
        hidden = Engine.TensorWhere(Engine.TensorLessThan(pre, NumOps.FromDouble(-20)), Engine.TensorZerosLike(pre), hidden);

        var weighted = Engine.TensorMultiply(hidden,
            Engine.TensorBroadcastTo(_W2.Reshape(new[] { 1, d, h }), new[] { n, d, h }));
        var output = Engine.TensorAdd(Engine.ReduceSum(weighted, new[] { 2 }, false),
            Engine.TensorBroadcastTo(_b2.Reshape(new[] { 1, d }), new[] { n, d }));

        return (output, hidden);
    }

    /// <summary>
    /// Extracts adjacency matrix: A[i,j] = ||W1[j][:,i]||_2
    /// </summary>
    private Matrix<T> ExtractAdjacencyMatrix(int d)
    {
        var norms = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(_W1), new[] { 2 }, false)); // [j, i]
        var A = Engine.TensorTranspose(norms);
        return Engine.TensorWhere(OffDiagonalMask(d), A, Engine.TensorZerosLike(A)).ToMatrix();
    }

    private double ComputeAcyclicityFromMLP(int d)
    {
        var A = ExtractAdjacencyMatrix(d);
        var (hVal, _) = ComputeNOTEARSConstraint(A);
        return hVal;
    }

    /// <summary>
    /// Computes augmented Lagrangian gradient with Sobolev penalty.
    /// The Sobolev penalty penalizes ||df_j/dx||^2 averaged over samples.
    /// Jacobian: df_j/dx_i = sum_k W2[j][k] * sigmoid'(z_k) * W1[j][i,k]
    /// </summary>
    private (double loss, Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2)
        ComputeAugLagGradientWithSobolev(Matrix<T> data, int n, int d, int h, double alpha, double rho)
    {
        var x = Tensor<T>.FromMatrix(data);
        T invN = NumOps.FromDouble(1.0 / n);
        T sobolevScale = NumOps.FromDouble(2.0 * _sobolevWeight / n);
        var w1Transposed = Engine.Reshape(Engine.TensorPermute(_W1, new[] { 0, 2, 1 }), new[] { d, h, d });
        var selfInputMask = OffDiagonalMask(d).Reshape(new[] { d, 1, d });

        // Data term, Sobolev penalty and their gradients, summed over samples. The samples are processed in
        // blocks so the [rows, d, h] activations and the [d, rows, d] Jacobian stay bounded
        // (MaxSampleBlockElements) for large n; every normalization still uses the full sample count.
        var squaredError = new Tensor<T>(new[] { d });
        var sobolevSquares = new Tensor<T>(new[] { d, d });
        var gB2 = new Tensor<T>(new[] { d });
        var gW2 = new Tensor<T>(new[] { d, h });
        var gB1 = new Tensor<T>(new[] { d, h });
        var gradByInput = new Tensor<T>(new[] { d, d * h });
        var sobGradW1 = new Tensor<T>(new[] { d, d, h });
        int blockRows = Math.Max(1, Math.Min(n, MaxSampleBlockElements / Math.Max(1, d * Math.Max(d, h))));
        for (int b0 = 0; b0 < n; b0 += blockRows)
        {
            int rows = Math.Min(blockRows, n - b0);
            var unitShape = new[] { rows, d, h };
            var xb = Engine.TensorSlice(x, new[] { b0, 0 }, new[] { rows, d });
            var (pred, hidden) = ForwardMLP(xb, rows, d, h);
            var residual = Engine.TensorSubtract(pred, xb);
            squaredError = Engine.TensorAdd(squaredError, Engine.ReduceSum(Engine.TensorSquare(residual), new[] { 0 }, false));

            // Data gradient: dL/d(pred) = residual / n
            var dOutput = Engine.TensorMultiplyScalar(residual, invN);
            gB2 = Engine.TensorAdd(gB2, Engine.ReduceSum(dOutput, new[] { 0 }, false));
            var dOutputPerUnit = Engine.TensorBroadcastTo(dOutput.Reshape(new[] { rows, d, 1 }), unitShape);

            // dL/dW2[k] = dOutput * hidden[k]
            gW2 = Engine.TensorAdd(gW2, Engine.ReduceSum(Engine.TensorMultiply(dOutputPerUnit, hidden), new[] { 0 }, false));

            // sigmoid'(z) = h(1-h); dHidden = dOutput * (W2 * sigmoid'(z))
            var sigDeriv = Engine.TensorMultiply(hidden, Engine.ScalarMinusTensor(NumOps.One, hidden));
            var w2PerSample = Engine.TensorBroadcastTo(_W2.Reshape(new[] { 1, d, h }), unitShape);
            var w2SigDeriv = Engine.TensorMultiply(w2PerSample, sigDeriv); // [s, j, k]
            var dHidden = Engine.TensorMultiply(dOutputPerUnit, w2SigDeriv);
            gB1 = Engine.TensorAdd(gB1, Engine.ReduceSum(dHidden, new[] { 0 }, false));

            // gW1[j, i, k] += sum_s dHidden[s, j, k] * x[s, i]
            gradByInput = Engine.TensorAdd(gradByInput,
                Engine.TensorMatMul(Engine.TensorTranspose(xb), dHidden.Reshape(new[] { rows, d * h })));

            // Sobolev penalty: penalize ||df_j/dx||^2 with df_j/dx_i = sum_k W2[j][k] * sigmoid'(z_k) * W1[j][i,k],
            // for i != j. Batched over j: q[j, s, k] = W2[j, k] * sigmoid'(z[s, j, k]).
            var q = Engine.Reshape(Engine.TensorPermute(w2SigDeriv, new[] { 1, 0, 2 }), new[] { d, rows, h });
            var jacobian = Engine.TensorMatMul(q, w1Transposed); // [j, s, i]
            var selfInput = Engine.TensorBroadcastTo(selfInputMask, new[] { d, rows, d });
            jacobian = Engine.TensorWhere(selfInput, jacobian, Engine.TensorZerosLike(jacobian));
            sobolevSquares = Engine.TensorAdd(sobolevSquares,
                Engine.ReduceSum(Engine.TensorSquare(jacobian), new[] { 1 }, false));

            // Gradient of Sobolev penalty: sobCoeff = (2 * weight / n) * J
            var sobCoeff = Engine.TensorMultiplyScalar(jacobian, sobolevScale); // [j, s, i]

            // d(J^2)/dW1[j][i,k] = 2*J * W2[k] * sig'(z_k), summed over samples
            sobGradW1 = Engine.TensorAdd(sobGradW1, Engine.TensorMatMul(
                Engine.Reshape(Engine.TensorPermute(sobCoeff, new[] { 0, 2, 1 }), new[] { d, d, rows }), q)); // [j, i, k]

            // d(J^2)/dW2[k] = 2*J * sig'(z_k) * W1[j][i,k], summed over inputs and samples
            var coeffTimesW1 = Engine.TensorMatMul(sobCoeff, _W1); // [j, s, k]
            var sigDerivByVariable = Engine.Reshape(Engine.TensorPermute(sigDeriv, new[] { 1, 0, 2 }), new[] { d, rows, h });
            gW2 = Engine.TensorAdd(gW2,
                Engine.ReduceSum(Engine.TensorMultiply(coeffTimesW1, sigDerivByVariable), new[] { 1 }, false));
        }

        var gW1 = Engine.TensorAdd(
            Engine.Reshape(Engine.TensorPermute(gradByInput.Reshape(new[] { d, d, h }), new[] { 1, 0, 2 }), new[] { d, d, h }),
            sobGradW1);
        double totalLoss = NumOps.ToDouble(Engine.TensorSum(squaredError));
        double totalSobolev = NumOps.ToDouble(Engine.TensorSum(sobolevSquares));

        totalLoss *= 0.5 / n;

        // Acyclicity gradient: dh/dW1[j][i,k] = augFactor * hGrad[i,j] * W1[j][i,k] / A[i,j],
        // plus the L1 penalty, for i != j with A[i, j] >= 1e-12.
        var A = ExtractAdjacencyMatrix(d);
        var (hVal, hGrad) = ComputeNOTEARSConstraint(A);
        double augFactor = alpha + rho * hVal;

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

        return (totalLoss + _sobolevWeight * totalSobolev / n, gW1, gB1, gW2, gB2);
    }

    private void AdamUpdate(Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2, int step)
    {
        double bc1 = 1 - Math.Pow(ADAM_BETA1, step);
        double bc2 = 1 - Math.Pow(ADAM_BETA2, step);

        (_W1, _mW1, _vW1) = SobolevAdamStep(_W1, gW1, _mW1, _vW1, bc1, bc2);
        (_b1, _mb1, _vb1) = SobolevAdamStep(_b1, gB1, _mb1, _vb1, bc1, bc2);
        (_W2, _mW2, _vW2) = SobolevAdamStep(_W2, gW2, _mW2, _vW2, bc1, bc2);
        (_b2, _mb2, _vb2) = SobolevAdamStep(_b2, gB2, _mb2, _vb2, bc1, bc2);
    }

    /// <summary>
    /// p - lr * (mHat / (sqrt(vHat) + eps)) with m = b1*m + (1-b1)*g and v = b2*v + (1-b2)*g².
    /// </summary>
    private (Tensor<T> Parameter, Tensor<T> M, Tensor<T> V) SobolevAdamStep(
        Tensor<T> parameter, Tensor<T> g, Tensor<T> m, Tensor<T> v, double bc1, double bc2)
    {
        var newM = Engine.TensorAdd(Engine.TensorMultiplyScalar(m, NumOps.FromDouble(ADAM_BETA1)),
            Engine.TensorMultiplyScalar(g, NumOps.FromDouble(1 - ADAM_BETA1)));
        var newV = Engine.TensorAdd(Engine.TensorMultiplyScalar(v, NumOps.FromDouble(ADAM_BETA2)),
            Engine.TensorMultiplyScalar(Engine.TensorMultiply(g, g), NumOps.FromDouble(1 - ADAM_BETA2)));
        var mHat = Engine.TensorDivideScalar(newM, NumOps.FromDouble(bc1));
        var vHat = Engine.TensorDivideScalar(newV, NumOps.FromDouble(bc2));
        var ratio = Engine.TensorDivide(mHat, Engine.TensorAddScalar(Engine.TensorSqrt(vHat), NumOps.FromDouble(1e-8)));
        return (Engine.TensorSubtract(parameter, Engine.TensorMultiplyScalar(ratio, NumOps.FromDouble(LEARNING_RATE))),
            newM, newV);
    }

    private double ComputeGradientNorm(Tensor<T> gW1, Tensor<T> gB1, Tensor<T> gW2, Tensor<T> gB2)
    {
        T sumOfSquares = NumOps.Add(
            NumOps.Add(Engine.TensorSum(Engine.TensorSquare(gW1)), Engine.TensorSum(Engine.TensorSquare(gB1))),
            NumOps.Add(Engine.TensorSum(Engine.TensorSquare(gW2)), Engine.TensorSum(Engine.TensorSquare(gB2))));
        return Math.Sqrt(NumOps.ToDouble(sumOfSquares));
    }
}
