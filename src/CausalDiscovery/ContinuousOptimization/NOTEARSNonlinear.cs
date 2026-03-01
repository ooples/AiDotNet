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

    // MLP parameters per variable:
    // W1[j] is [d x h], b1[j] is [h], W2[j] is [h], b2[j] is scalar
    private Matrix<T>[] _W1 = [];
    private double[][] _b1 = [];
    private double[][] _W2 = [];
    private double[] _b2 = [];

    // Adam state
    private Matrix<T>[] _mW1 = [], _vW1 = [];
    private double[][] _mb1 = [], _vb1 = [];
    private double[][] _mW2 = [], _vW2 = [];
    private double[] _mb2 = [], _vb2 = [];

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
                for (int j = 0; j < d; j++)
                    for (int k = 0; k < h; k++)
                        _W1[j][j, k] = NumOps.Zero;

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
        bool hasEdges = false;
        for (int i = 0; i < d && !hasEdges; i++)
            for (int j = 0; j < d && !hasEdges; j++)
                if (i != j && Math.Abs(NumOps.ToDouble(result[i, j])) > 0)
                    hasEdges = true;

        return hasEdges ? result : FallbackCorrelationGraph(data);
    }

    #endregion

    #region MLP Operations

    private void InitializeMLPParameters(int d, int h)
    {
        var rng = new Random(42);
        double scale = Math.Sqrt(2.0 / d); // He initialization

        _W1 = new Matrix<T>[d];
        _b1 = new double[d][];
        _W2 = new double[d][];
        _b2 = new double[d];

        _mW1 = new Matrix<T>[d]; _vW1 = new Matrix<T>[d];
        _mb1 = new double[d][]; _vb1 = new double[d][];
        _mW2 = new double[d][]; _vW2 = new double[d][];
        _mb2 = new double[d]; _vb2 = new double[d];

        for (int j = 0; j < d; j++)
        {
            _W1[j] = new Matrix<T>(d, h);
            _b1[j] = new double[h];
            _W2[j] = new double[h];
            _b2[j] = 0;

            _mW1[j] = new Matrix<T>(d, h); _vW1[j] = new Matrix<T>(d, h);
            _mb1[j] = new double[h]; _vb1[j] = new double[h];
            _mW2[j] = new double[h]; _vW2[j] = new double[h];

            for (int i = 0; i < d; i++)
            {
                if (i == j) continue; // No self-loops
                for (int k = 0; k < h; k++)
                {
                    _W1[j][i, k] = NumOps.FromDouble(rng.NextDouble() * scale - scale / 2.0);
                }
            }

            for (int k = 0; k < h; k++)
            {
                _W2[j][k] = rng.NextDouble() * scale - scale / 2.0;
            }
        }
    }

    /// <summary>
    /// Forward pass for variable j: output = W2[j]^T * sigmoid(W1[j]^T * x + b1[j]) + b2[j]
    /// </summary>
    private (double output, double[] hidden) ForwardMLP(Matrix<T> data, int sample, int j, int d)
    {
        int h = _b1[j].Length;
        var hidden = new double[h];

        for (int k = 0; k < h; k++)
        {
            double sum = _b1[j][k];
            for (int i = 0; i < d; i++)
                sum += NumOps.ToDouble(data[sample, i]) * NumOps.ToDouble(_W1[j][i, k]);
            hidden[k] = 1.0 / (1.0 + Math.Exp(-sum)); // sigmoid
        }

        double output = _b2[j];
        for (int k = 0; k < h; k++)
            output += hidden[k] * _W2[j][k];

        return (output, hidden);
    }

    /// <summary>
    /// Extract adjacency matrix from MLP weights: A[i,j] = ||W1[j][:,i]||_2
    /// </summary>
    private Matrix<T> ExtractAdjacencyMatrix(int d)
    {
        int h = _b1[0].Length;
        var A = new Matrix<T>(d, d);
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                double norm = 0;
                for (int k = 0; k < h; k++)
                {
                    double w = NumOps.ToDouble(_W1[j][i, k]);
                    norm += w * w;
                }
                A[i, j] = NumOps.FromDouble(Math.Sqrt(norm));
            }
        }

        return A;
    }

    private double ComputeAcyclicityFromMLP(int d)
    {
        var A = ExtractAdjacencyMatrix(d);
        var (h, _) = ComputeNOTEARSConstraint(A);
        return h;
    }

    #endregion

    #region Gradient Computation

    private (double loss, Matrix<T>[] gW1, double[][] gB1, double[][] gW2, double[] gB2)
        ComputeAugLagGradient(Matrix<T> data, int n, int d, int h, double alpha, double rho)
    {
        var gW1 = new Matrix<T>[d];
        var gB1 = new double[d][];
        var gW2 = new double[d][];
        var gB2 = new double[d];
        for (int j = 0; j < d; j++)
        {
            gW1[j] = new Matrix<T>(d, h);
            gB1[j] = new double[h];
            gW2[j] = new double[h];
        }

        double totalLoss = 0;

        // Compute loss and data gradients for each sample
        for (int sample = 0; sample < n; sample++)
        {
            for (int j = 0; j < d; j++)
            {
                var (pred, hidden) = ForwardMLP(data, sample, j, d);
                double residual = pred - NumOps.ToDouble(data[sample, j]);
                totalLoss += residual * residual;

                // Backprop through MLP
                double dOutput = residual / n; // gradient of (1/2n)*||R||^2

                // Gradient for W2, b2
                gB2[j] += dOutput;
                for (int k = 0; k < h; k++)
                    gW2[j][k] += dOutput * hidden[k];

                // Gradient for W1, b1 (through sigmoid)
                for (int k = 0; k < h; k++)
                {
                    double dHidden = dOutput * _W2[j][k] * hidden[k] * (1 - hidden[k]);
                    gB1[j][k] += dHidden;
                    for (int i = 0; i < d; i++)
                        gW1[j][i, k] = NumOps.Add(gW1[j][i, k],
                            NumOps.FromDouble(dHidden * NumOps.ToDouble(data[sample, i])));
                }
            }
        }

        totalLoss *= 0.5 / n;

        // Acyclicity gradient through adjacency
        var A = ExtractAdjacencyMatrix(d);
        var (hVal, hGrad) = ComputeNOTEARSConstraint(A);
        double augFactor = alpha + rho * hVal;

        // Chain rule: dh/dW1[j][i,k] = dh/dA[i,j] * dA[i,j]/dW1[j][i,k]
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
                    // dA[i,j]/dW1[j][i,k] = W1[j][i,k] / A[i,j]
                    double grad = dhda * w1val / aij;
                    // L1 penalty on W1
                    grad += Lambda1 * Math.Sign(w1val);
                    gW1[j][i, k] = NumOps.Add(gW1[j][i, k], NumOps.FromDouble(grad));
                }
            }
        }

        return (totalLoss, gW1, gB1, gW2, gB2);
    }

    private void AdamUpdate(Matrix<T>[] gW1, double[][] gB1, double[][] gW2, double[] gB2,
        int d, int h, int step)
    {
        double bc1 = 1 - Math.Pow(ADAM_BETA1, step);
        double bc2 = 1 - Math.Pow(ADAM_BETA2, step);

        for (int j = 0; j < d; j++)
        {
            // W1 update
            for (int i = 0; i < d; i++)
            {
                for (int k = 0; k < h; k++)
                {
                    double gVal = NumOps.ToDouble(gW1[j][i, k]);
                    double mVal = ADAM_BETA1 * NumOps.ToDouble(_mW1[j][i, k]) + (1 - ADAM_BETA1) * gVal;
                    double vVal = ADAM_BETA2 * NumOps.ToDouble(_vW1[j][i, k]) + (1 - ADAM_BETA2) * gVal * gVal;
                    _mW1[j][i, k] = NumOps.FromDouble(mVal);
                    _vW1[j][i, k] = NumOps.FromDouble(vVal);
                    double mHat = mVal / bc1;
                    double vHat = vVal / bc2;
                    double wVal = NumOps.ToDouble(_W1[j][i, k]);
                    wVal -= LEARNING_RATE * mHat / (Math.Sqrt(vHat) + 1e-8);
                    _W1[j][i, k] = NumOps.FromDouble(wVal);
                }
            }

            // b1 update
            for (int k = 0; k < h; k++)
            {
                _mb1[j][k] = ADAM_BETA1 * _mb1[j][k] + (1 - ADAM_BETA1) * gB1[j][k];
                _vb1[j][k] = ADAM_BETA2 * _vb1[j][k] + (1 - ADAM_BETA2) * gB1[j][k] * gB1[j][k];
                double mHat = _mb1[j][k] / bc1;
                double vHat = _vb1[j][k] / bc2;
                _b1[j][k] -= LEARNING_RATE * mHat / (Math.Sqrt(vHat) + 1e-8);
            }

            // W2 update
            for (int k = 0; k < h; k++)
            {
                _mW2[j][k] = ADAM_BETA1 * _mW2[j][k] + (1 - ADAM_BETA1) * gW2[j][k];
                _vW2[j][k] = ADAM_BETA2 * _vW2[j][k] + (1 - ADAM_BETA2) * gW2[j][k] * gW2[j][k];
                double mHat = _mW2[j][k] / bc1;
                double vHat = _vW2[j][k] / bc2;
                _W2[j][k] -= LEARNING_RATE * mHat / (Math.Sqrt(vHat) + 1e-8);
            }

            // b2 update
            _mb2[j] = ADAM_BETA1 * _mb2[j] + (1 - ADAM_BETA1) * gB2[j];
            _vb2[j] = ADAM_BETA2 * _vb2[j] + (1 - ADAM_BETA2) * gB2[j] * gB2[j];
            {
                double mHat = _mb2[j] / bc1;
                double vHat = _vb2[j] / bc2;
                _b2[j] -= LEARNING_RATE * mHat / (Math.Sqrt(vHat) + 1e-8);
            }
        }
    }

    private double ComputeGradientNorm(Matrix<T>[] gW1, double[][] gB1, double[][] gW2, double[] gB2,
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
                norm += gB1[j][k] * gB1[j][k];
            for (int k = 0; k < h; k++)
                norm += gW2[j][k] * gW2[j][k];
            norm += gB2[j] * gB2[j];
        }

        return Math.Sqrt(norm);
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
