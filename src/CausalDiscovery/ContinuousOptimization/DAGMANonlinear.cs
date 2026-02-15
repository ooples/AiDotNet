using AiDotNet.Enums;
using AiDotNet.Models.Options;

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

    #endregion

    #region Fields

    private readonly double[] _sValues;
    private int _hiddenSize = DEFAULT_HIDDEN_SIZE;
    private int _lastIterations;
    private double _lastH;
    private double _lastLoss;

    // MLP parameters per variable
    private double[][,] _W1 = [];
    private double[][] _b1 = [];
    private double[][] _W2 = [];
    private double[] _b2 = [];

    // Adam state
    private double[][,] _mW1 = [], _vW1 = [];
    private double[][] _mb1 = [], _vb1 = [];
    private double[][] _mW2 = [], _vW2 = [];
    private double[] _mb2 = [], _vb2 = [];

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
        if (options?.Seed.HasValue == true) _seed = options.Seed.Value;
        _sValues = [1.0, 0.9, 0.8, 0.7, 0.6];
    }

    #endregion

    #region Core Algorithm

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        double[,] X = MatrixToDoubleArray(data);
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
            int maxInner = (t < T - 1) ? DEFAULT_WARM_ITER : DEFAULT_MAX_ITER;
            double prevObj = double.MaxValue;

            for (int inner = 1; inner <= maxInner; inner++)
            {
                adamStep++;
                var (obj, gW1, gB1, gW2, gB2) = ComputeObjectiveAndGradients(X, n, d, h, mu, s);

                AdamUpdate(gW1, gB1, gW2, gB2, d, h, adamStep);

                // Zero diagonal in W1
                for (int j = 0; j < d; j++)
                    for (int k = 0; k < h; k++)
                        _W1[j][j, k] = 0;

                if (inner % CHECKPOINT_INTERVAL == 0)
                {
                    if (Math.Abs(obj - prevObj) / (Math.Abs(prevObj) + 1e-15) < INNER_CONVERGENCE_TOL)
                        break;
                    prevObj = obj;
                }
            }

            mu *= DEFAULT_MU_FACTOR;
            _lastIterations += maxInner;
        }

        var A = ExtractAdjacencyMatrix(d);
        var (finalLoss, _, _, _, _) = ComputeObjectiveAndGradients(X, n, d, h, 0, _sValues[^1]);
        _lastLoss = finalLoss;
        _lastH = ComputeLogDetConstraintFromA(A, _sValues[^1], d).H;

        var WThresholded = ThresholdAndClean(A, WThreshold);
        return DoubleArrayToMatrix(WThresholded);
    }

    #endregion

    #region MLP Operations

    private void InitializeMLPParameters(int d, int h)
    {
        var rng = new Random(_seed);
        double scale = Math.Sqrt(2.0 / d);

        _W1 = new double[d][,]; _b1 = new double[d][];
        _W2 = new double[d][]; _b2 = new double[d];
        _mW1 = new double[d][,]; _vW1 = new double[d][,];
        _mb1 = new double[d][]; _vb1 = new double[d][];
        _mW2 = new double[d][]; _vW2 = new double[d][];
        _mb2 = new double[d]; _vb2 = new double[d];

        for (int j = 0; j < d; j++)
        {
            _W1[j] = new double[d, h]; _b1[j] = new double[h];
            _W2[j] = new double[h]; _b2[j] = 0;
            _mW1[j] = new double[d, h]; _vW1[j] = new double[d, h];
            _mb1[j] = new double[h]; _vb1[j] = new double[h];
            _mW2[j] = new double[h]; _vW2[j] = new double[h];

            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                for (int k = 0; k < h; k++)
                    _W1[j][i, k] = rng.NextDouble() * scale - scale / 2.0;
            }

            for (int k = 0; k < h; k++)
                _W2[j][k] = rng.NextDouble() * scale - scale / 2.0;
        }
    }

    private (double output, double[] hidden) ForwardMLP(double[] x, int j)
    {
        int d = x.Length;
        int h = _b1[j].Length;
        var hidden = new double[h];

        for (int k = 0; k < h; k++)
        {
            double sum = _b1[j][k];
            for (int i = 0; i < d; i++)
                sum += x[i] * _W1[j][i, k];
            hidden[k] = 1.0 / (1.0 + Math.Exp(-sum));
        }

        double output = _b2[j];
        for (int k = 0; k < h; k++)
            output += hidden[k] * _W2[j][k];

        return (output, hidden);
    }

    private double[,] ExtractAdjacencyMatrix(int d)
    {
        int h = _b1[0].Length;
        var A = new double[d, d];
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                double norm = 0;
                for (int k = 0; k < h; k++)
                    norm += _W1[j][i, k] * _W1[j][i, k];
                A[i, j] = Math.Sqrt(norm);
            }
        }

        return A;
    }

    private (double H, double[,] Gradient) ComputeLogDetConstraintFromA(double[,] A, double s, int d)
    {
        var M = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            M[i, i] = s;
            for (int j = 0; j < d; j++)
                M[i, j] -= A[i, j] * A[i, j];
        }

        double logDet = ComputeLogDeterminant(M, d);
        double hVal = -logDet + d * Math.Log(s);

        var invM = InvertMatrix(M, d);
        var gradient = new double[d, d];
        if (invM != null)
        {
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    gradient[i, j] = 2.0 * A[i, j] * invM[j, i];
        }

        return (hVal, gradient);
    }

    #endregion

    #region Gradient Computation

    private (double obj, double[][,] gW1, double[][] gB1, double[][] gW2, double[] gB2)
        ComputeObjectiveAndGradients(double[,] X, int n, int d, int h, double mu, double s)
    {
        var gW1 = new double[d][,];
        var gB1 = new double[d][];
        var gW2 = new double[d][];
        var gB2 = new double[d];
        for (int j = 0; j < d; j++)
        {
            gW1[j] = new double[d, h]; gB1[j] = new double[h]; gW2[j] = new double[h];
        }

        double totalLoss = 0;

        for (int sample = 0; sample < n; sample++)
        {
            var x = new double[d];
            for (int j = 0; j < d; j++) x[j] = X[sample, j];

            for (int j = 0; j < d; j++)
            {
                var (pred, hidden) = ForwardMLP(x, j);
                double residual = pred - x[j];
                totalLoss += residual * residual;

                double dOutput = residual / n;
                gB2[j] += dOutput;
                for (int k = 0; k < h; k++)
                    gW2[j][k] += dOutput * hidden[k];

                for (int k = 0; k < h; k++)
                {
                    double dHidden = dOutput * _W2[j][k] * hidden[k] * (1 - hidden[k]);
                    gB1[j][k] += dHidden;
                    for (int i = 0; i < d; i++)
                        gW1[j][i, k] += dHidden * x[i];
                }
            }
        }

        totalLoss *= 0.5 / n;

        // DAGMA constraint gradient
        var A = ExtractAdjacencyMatrix(d);
        var (hVal, hGrad) = ComputeLogDetConstraintFromA(A, s, d);
        double obj = totalLoss + Lambda1 * ComputeL1NormFromA(A, d) + mu * hVal;

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
            {
                if (i == j) continue;
                double aij = A[i, j];
                if (aij < 1e-12) continue;

                double dhda = mu * hGrad[i, j];
                for (int k = 0; k < h; k++)
                {
                    gW1[j][i, k] += dhda * _W1[j][i, k] / aij;
                    gW1[j][i, k] += Lambda1 * Math.Sign(_W1[j][i, k]);
                }
            }
        }

        return (obj, gW1, gB1, gW2, gB2);
    }

    private void AdamUpdate(double[][,] gW1, double[][] gB1, double[][] gW2, double[] gB2,
        int d, int h, int step)
    {
        double bc1 = 1 - Math.Pow(ADAM_BETA1, step);
        double bc2 = 1 - Math.Pow(ADAM_BETA2, step);

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < d; i++)
                for (int k = 0; k < h; k++)
                {
                    _mW1[j][i, k] = ADAM_BETA1 * _mW1[j][i, k] + (1 - ADAM_BETA1) * gW1[j][i, k];
                    _vW1[j][i, k] = ADAM_BETA2 * _vW1[j][i, k] + (1 - ADAM_BETA2) * gW1[j][i, k] * gW1[j][i, k];
                    _W1[j][i, k] -= DEFAULT_LEARNING_RATE * (_mW1[j][i, k] / bc1) / (Math.Sqrt(_vW1[j][i, k] / bc2) + 1e-8);
                }

            for (int k = 0; k < h; k++)
            {
                _mb1[j][k] = ADAM_BETA1 * _mb1[j][k] + (1 - ADAM_BETA1) * gB1[j][k];
                _vb1[j][k] = ADAM_BETA2 * _vb1[j][k] + (1 - ADAM_BETA2) * gB1[j][k] * gB1[j][k];
                _b1[j][k] -= DEFAULT_LEARNING_RATE * (_mb1[j][k] / bc1) / (Math.Sqrt(_vb1[j][k] / bc2) + 1e-8);
            }

            for (int k = 0; k < h; k++)
            {
                _mW2[j][k] = ADAM_BETA1 * _mW2[j][k] + (1 - ADAM_BETA1) * gW2[j][k];
                _vW2[j][k] = ADAM_BETA2 * _vW2[j][k] + (1 - ADAM_BETA2) * gW2[j][k] * gW2[j][k];
                _W2[j][k] -= DEFAULT_LEARNING_RATE * (_mW2[j][k] / bc1) / (Math.Sqrt(_vW2[j][k] / bc2) + 1e-8);
            }

            _mb2[j] = ADAM_BETA1 * _mb2[j] + (1 - ADAM_BETA1) * gB2[j];
            _vb2[j] = ADAM_BETA2 * _vb2[j] + (1 - ADAM_BETA2) * gB2[j] * gB2[j];
            _b2[j] -= DEFAULT_LEARNING_RATE * (_mb2[j] / bc1) / (Math.Sqrt(_vb2[j] / bc2) + 1e-8);
        }
    }

    private static double ComputeL1NormFromA(double[,] A, int d)
    {
        double sum = 0;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j) sum += Math.Abs(A[i, j]);
        return sum;
    }

    #endregion

    #region Matrix Utilities

    private static double ComputeLogDeterminant(double[,] matrix, int d)
    {
        var LU = (double[,])matrix.Clone();
        for (int k = 0; k < d; k++)
        {
            int maxRow = k;
            for (int i = k + 1; i < d; i++)
                if (Math.Abs(LU[i, k]) > Math.Abs(LU[maxRow, k]))
                    maxRow = i;

            if (maxRow != k)
                for (int j = 0; j < d; j++)
                    (LU[k, j], LU[maxRow, j]) = (LU[maxRow, j], LU[k, j]);

            if (Math.Abs(LU[k, k]) < 1e-15)
                return double.NegativeInfinity;

            for (int i = k + 1; i < d; i++)
            {
                LU[i, k] /= LU[k, k];
                for (int j = k + 1; j < d; j++)
                    LU[i, j] -= LU[i, k] * LU[k, j];
            }
        }

        double logDet = 0;
        for (int i = 0; i < d; i++)
            logDet += Math.Log(Math.Abs(LU[i, i]));
        return logDet;
    }

    private static double[,]? InvertMatrix(double[,] matrix, int d)
    {
        var aug = new double[d, 2 * d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++) aug[i, j] = matrix[i, j];
            aug[i, i + d] = 1.0;
        }

        for (int col = 0; col < d; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < d; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            if (Math.Abs(aug[maxRow, col]) < 1e-12) return null;

            if (maxRow != col)
                for (int j = 0; j < 2 * d; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            double pivot = aug[col, col];
            for (int j = 0; j < 2 * d; j++) aug[col, j] /= pivot;

            for (int row = 0; row < d; row++)
            {
                if (row != col)
                {
                    double factor = aug[row, col];
                    for (int j = 0; j < 2 * d; j++)
                        aug[row, j] -= factor * aug[col, j];
                }
            }
        }

        var result = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                result[i, j] = aug[i, j + d];
        return result;
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
