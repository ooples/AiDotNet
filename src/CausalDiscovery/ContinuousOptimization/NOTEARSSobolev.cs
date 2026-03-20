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
[ModelPaper("Learning Sparse Nonparametric DAGs", "https://proceedings.mlr.press/v108/zheng20a.html", Year = 2020, Authors = "Xun Zheng, Chen Dan, Bryon Aragam, Pradeep Ravikumar, Eric Xing")]
public class NOTEARSSobolev<T> : ContinuousOptimizationBase<T>
{
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
    private readonly int? _seed;

    // MLP parameters per variable:
    // W1[j] is [d x h], b1[j] is [h], W2[j] is [h], b2[j] is scalar
    private Matrix<T>[] _W1 = [];
    private Matrix<T>[] _b1 = [];
    private Matrix<T>[] _W2 = [];
    private T[] _b2 = [];

    // Adam state
    private Matrix<T>[] _mW1 = [], _vW1 = [];
    private Matrix<T>[] _mb1 = [], _vb1 = [];
    private Matrix<T>[] _mW2 = [], _vW2 = [];
    private T[] _mb2 = [], _vb2 = [];

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
        _seed = options?.Seed;
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

                AdamUpdate(gradW1, gradB1, gradW2, gradB2, d, h, adamStep);

                // Zero diagonal (no self-loops)
                for (int j = 0; j < d; j++)
                    for (int k = 0; k < h; k++)
                        _W1[j][j, k] = NumOps.Zero;

                if (inner % 100 == 0 && inner > 0)
                {
                    double gradNorm = ComputeGradientNorm(gradW1, gradB1, gradW2, gradB2, d, h);
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
        bool hasEdges = false;
        for (int i = 0; i < d && !hasEdges; i++)
            for (int j = 0; j < d && !hasEdges; j++)
                if (i != j && NumOps.GreaterThan(NumOps.Abs(result[i, j]), NumOps.Zero))
                    hasEdges = true;

        return hasEdges ? result : FallbackCorrelationGraph(data);
    }

    private void InitializeMLPParameters(int d, int h)
    {
        var rng = _seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(_seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        double scale = Math.Sqrt(2.0 / d);

        _W1 = new Matrix<T>[d];
        _b1 = new Matrix<T>[d];
        _W2 = new Matrix<T>[d];
        _b2 = new T[d];

        _mW1 = new Matrix<T>[d]; _vW1 = new Matrix<T>[d];
        _mb1 = new Matrix<T>[d]; _vb1 = new Matrix<T>[d];
        _mW2 = new Matrix<T>[d]; _vW2 = new Matrix<T>[d];
        _mb2 = new T[d]; _vb2 = new T[d];

        for (int j = 0; j < d; j++)
        {
            _W1[j] = new Matrix<T>(d, h);
            _b1[j] = new Matrix<T>(1, h);
            _W2[j] = new Matrix<T>(1, h);
            _b2[j] = NumOps.Zero;

            _mW1[j] = new Matrix<T>(d, h); _vW1[j] = new Matrix<T>(d, h);
            _mb1[j] = new Matrix<T>(1, h); _vb1[j] = new Matrix<T>(1, h);
            _mW2[j] = new Matrix<T>(1, h); _vW2[j] = new Matrix<T>(1, h);
            _mb2[j] = NumOps.Zero; _vb2[j] = NumOps.Zero;

            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                for (int k = 0; k < h; k++)
                    _W1[j][i, k] = NumOps.FromDouble(rng.NextDouble() * scale - scale / 2.0);
            }

            for (int k = 0; k < h; k++)
                _W2[j][0, k] = NumOps.FromDouble(rng.NextDouble() * scale - scale / 2.0);
        }
    }

    /// <summary>
    /// Forward pass: output = W2[j]^T * sigmoid(W1[j]^T * x + b1[j]) + b2[j]
    /// Also returns hidden activations and pre-activation values for gradient computation.
    /// </summary>
    private (T output, T[] hidden, T[] preAct) ForwardMLP(Matrix<T> data, int sample, int j, int d, int h)
    {
        var hidden = new T[h];
        var preAct = new T[h];

        for (int k = 0; k < h; k++)
        {
            T sum = _b1[j][0, k];
            for (int i = 0; i < d; i++)
                sum = NumOps.Add(sum, NumOps.Multiply(data[sample, i], _W1[j][i, k]));
            preAct[k] = sum;

            // Sigmoid activation
            double sv = NumOps.ToDouble(sum);
            double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
            hidden[k] = NumOps.FromDouble(sigVal);
        }

        T output = _b2[j];
        for (int k = 0; k < h; k++)
            output = NumOps.Add(output, NumOps.Multiply(hidden[k], _W2[j][0, k]));

        return (output, hidden, preAct);
    }

    /// <summary>
    /// Extracts adjacency matrix: A[i,j] = ||W1[j][:,i]||_2
    /// </summary>
    private Matrix<T> ExtractAdjacencyMatrix(int d)
    {
        int h = _b1[0].Columns;
        var A = new Matrix<T>(d, d);
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                T norm = NumOps.Zero;
                for (int k = 0; k < h; k++)
                {
                    T w = _W1[j][i, k];
                    norm = NumOps.Add(norm, NumOps.Multiply(w, w));
                }
                A[i, j] = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(norm)));
            }
        }
        return A;
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
    private (double loss, Matrix<T>[] gW1, Matrix<T>[] gB1, Matrix<T>[] gW2, T[] gB2)
        ComputeAugLagGradientWithSobolev(Matrix<T> data, int n, int d, int h, double alpha, double rho)
    {
        var gW1 = new Matrix<T>[d];
        var gB1 = new Matrix<T>[d];
        var gW2 = new Matrix<T>[d];
        var gB2 = new T[d];
        for (int j = 0; j < d; j++)
        {
            gW1[j] = new Matrix<T>(d, h);
            gB1[j] = new Matrix<T>(1, h);
            gW2[j] = new Matrix<T>(1, h);
            gB2[j] = NumOps.Zero;
        }

        double totalLoss = 0;
        double totalSobolev = 0;
        T invN = NumOps.FromDouble(1.0 / n);

        for (int sample = 0; sample < n; sample++)
        {
            for (int j = 0; j < d; j++)
            {
                var (pred, hidden, _) = ForwardMLP(data, sample, j, d, h);
                T residual = NumOps.Subtract(pred, data[sample, j]);
                double residualD = NumOps.ToDouble(residual);
                totalLoss += residualD * residualD;

                // Data gradient: dL/d(pred) = residual / n
                T dOutput = NumOps.Multiply(residual, invN);

                // Gradient for b2
                gB2[j] = NumOps.Add(gB2[j], dOutput);

                // Gradient for W2, b1, W1 (backprop)
                for (int k = 0; k < h; k++)
                {
                    T hk = hidden[k];
                    // dL/dW2[k] = dOutput * hidden[k]
                    gW2[j][0, k] = NumOps.Add(gW2[j][0, k], NumOps.Multiply(dOutput, hk));

                    // sigmoid'(z) = h(1-h)
                    T sigDeriv = NumOps.Multiply(hk, NumOps.Subtract(NumOps.One, hk));
                    T dHidden = NumOps.Multiply(dOutput, NumOps.Multiply(_W2[j][0, k], sigDeriv));

                    gB1[j][0, k] = NumOps.Add(gB1[j][0, k], dHidden);
                    for (int i = 0; i < d; i++)
                        gW1[j][i, k] = NumOps.Add(gW1[j][i, k],
                            NumOps.Multiply(dHidden, data[sample, i]));
                }

                // Sobolev penalty: penalize ||df_j/dx||^2
                // df_j/dx_i = sum_k W2[j][k] * sigmoid'(z_k) * W1[j][i,k]
                for (int i = 0; i < d; i++)
                {
                    if (i == j) continue;
                    T jacobian_ij = NumOps.Zero;
                    for (int k = 0; k < h; k++)
                    {
                        T hk2 = hidden[k];
                        T sigD = NumOps.Multiply(hk2, NumOps.Subtract(NumOps.One, hk2));
                        jacobian_ij = NumOps.Add(jacobian_ij,
                            NumOps.Multiply(_W2[j][0, k],
                            NumOps.Multiply(sigD, _W1[j][i, k])));
                    }

                    double jVal = NumOps.ToDouble(jacobian_ij);
                    totalSobolev += jVal * jVal;

                    // Gradient of Sobolev penalty w.r.t. W1[j][i,k] and W2[j][k]
                    T sobCoeff = NumOps.Multiply(NumOps.FromDouble(2.0 * _sobolevWeight / n), jacobian_ij);

                    for (int k = 0; k < h; k++)
                    {
                        T hk3 = hidden[k];
                        T sigD = NumOps.Multiply(hk3, NumOps.Subtract(NumOps.One, hk3));

                        // d(J^2)/dW1[j][i,k] = 2*J * W2[k] * sig'(z_k)
                        T sobGradW1 = NumOps.Multiply(sobCoeff,
                            NumOps.Multiply(_W2[j][0, k], sigD));
                        gW1[j][i, k] = NumOps.Add(gW1[j][i, k], sobGradW1);

                        // d(J^2)/dW2[k] = 2*J * sig'(z_k) * W1[j][i,k]
                        T sobGradW2 = NumOps.Multiply(sobCoeff,
                            NumOps.Multiply(sigD, _W1[j][i, k]));
                        gW2[j][0, k] = NumOps.Add(gW2[j][0, k], sobGradW2);
                    }
                }
            }
        }

        totalLoss *= 0.5 / n;

        // Acyclicity gradient
        var A = ExtractAdjacencyMatrix(d);
        var (hVal, hGrad) = ComputeNOTEARSConstraint(A);
        double augFactor = alpha + rho * hVal;

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                double aij = NumOps.ToDouble(A[i, j]);
                if (aij < 1e-12) continue;

                double dhda = augFactor * NumOps.ToDouble(hGrad[i, j]);
                for (int k = 0; k < h; k++)
                {
                    double w1val = NumOps.ToDouble(_W1[j][i, k]);
                    double grad = dhda * w1val / aij;
                    // L1 penalty
                    grad += Lambda1 * Math.Sign(w1val);
                    gW1[j][i, k] = NumOps.Add(gW1[j][i, k], NumOps.FromDouble(grad));
                }
            }
        }

        return (totalLoss + _sobolevWeight * totalSobolev / n, gW1, gB1, gW2, gB2);
    }

    private void AdamUpdate(Matrix<T>[] gW1, Matrix<T>[] gB1, Matrix<T>[] gW2, T[] gB2,
        int d, int h, int step)
    {
        T bc1 = NumOps.FromDouble(1 - Math.Pow(ADAM_BETA1, step));
        T bc2 = NumOps.FromDouble(1 - Math.Pow(ADAM_BETA2, step));
        T beta1T = NumOps.FromDouble(ADAM_BETA1);
        T beta2T = NumOps.FromDouble(ADAM_BETA2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - ADAM_BETA1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - ADAM_BETA2);
        T lrT = NumOps.FromDouble(LEARNING_RATE);
        T eps = NumOps.FromDouble(1e-8);

        for (int j = 0; j < d; j++)
        {
            // W1 update
            for (int i = 0; i < d; i++)
            {
                for (int k = 0; k < h; k++)
                {
                    T g = gW1[j][i, k];
                    T m = NumOps.Add(NumOps.Multiply(beta1T, _mW1[j][i, k]),
                                     NumOps.Multiply(oneMinusBeta1, g));
                    T v = NumOps.Add(NumOps.Multiply(beta2T, _vW1[j][i, k]),
                                     NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(g, g)));
                    _mW1[j][i, k] = m;
                    _vW1[j][i, k] = v;
                    T mHat = NumOps.Divide(m, bc1);
                    T vHat = NumOps.Divide(v, bc2);
                    _W1[j][i, k] = NumOps.Subtract(_W1[j][i, k],
                        NumOps.Multiply(lrT, NumOps.Divide(mHat,
                        NumOps.Add(NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(vHat))), eps))));
                }
            }

            // b1 update
            for (int k = 0; k < h; k++)
            {
                T g = gB1[j][0, k];
                T m = NumOps.Add(NumOps.Multiply(beta1T, _mb1[j][0, k]),
                                 NumOps.Multiply(oneMinusBeta1, g));
                T v = NumOps.Add(NumOps.Multiply(beta2T, _vb1[j][0, k]),
                                 NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(g, g)));
                _mb1[j][0, k] = m;
                _vb1[j][0, k] = v;
                T mHat = NumOps.Divide(m, bc1);
                T vHat = NumOps.Divide(v, bc2);
                _b1[j][0, k] = NumOps.Subtract(_b1[j][0, k],
                    NumOps.Multiply(lrT, NumOps.Divide(mHat,
                    NumOps.Add(NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(vHat))), eps))));
            }

            // W2 update
            for (int k = 0; k < h; k++)
            {
                T g = gW2[j][0, k];
                T m = NumOps.Add(NumOps.Multiply(beta1T, _mW2[j][0, k]),
                                 NumOps.Multiply(oneMinusBeta1, g));
                T v = NumOps.Add(NumOps.Multiply(beta2T, _vW2[j][0, k]),
                                 NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(g, g)));
                _mW2[j][0, k] = m;
                _vW2[j][0, k] = v;
                T mHat = NumOps.Divide(m, bc1);
                T vHat = NumOps.Divide(v, bc2);
                _W2[j][0, k] = NumOps.Subtract(_W2[j][0, k],
                    NumOps.Multiply(lrT, NumOps.Divide(mHat,
                    NumOps.Add(NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(vHat))), eps))));
            }

            // b2 update
            {
                T g = gB2[j];
                T m = NumOps.Add(NumOps.Multiply(beta1T, _mb2[j]),
                                 NumOps.Multiply(oneMinusBeta1, g));
                T v = NumOps.Add(NumOps.Multiply(beta2T, _vb2[j]),
                                 NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(g, g)));
                _mb2[j] = m;
                _vb2[j] = v;
                T mHat = NumOps.Divide(m, bc1);
                T vHat = NumOps.Divide(v, bc2);
                _b2[j] = NumOps.Subtract(_b2[j],
                    NumOps.Multiply(lrT, NumOps.Divide(mHat,
                    NumOps.Add(NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(vHat))), eps))));
            }
        }
    }

    private double ComputeGradientNorm(Matrix<T>[] gW1, Matrix<T>[] gB1, Matrix<T>[] gW2, T[] gB2,
        int d, int h)
    {
        double norm = 0;
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
                for (int k = 0; k < h; k++)
                {
                    double g = NumOps.ToDouble(gW1[j][i, k]);
                    norm += g * g;
                }
            for (int k = 0; k < h; k++)
            {
                double g1 = NumOps.ToDouble(gB1[j][0, k]);
                norm += g1 * g1;
                double g2 = NumOps.ToDouble(gW2[j][0, k]);
                norm += g2 * g2;
            }
            double gb = NumOps.ToDouble(gB2[j]);
            norm += gb * gb;
        }

        return Math.Sqrt(norm);
    }
}
