using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Causal;

/// <summary>
/// PC Algorithm-based Feature Selection for causal discovery.
/// </summary>
/// <remarks>
/// <para>
/// Uses the PC (Peter-Clark) algorithm to learn a causal graph structure
/// from data, then selects features that are directly connected to the
/// target variable in the causal graph.
/// </para>
/// <para><b>For Beginners:</b> The PC algorithm tries to figure out what
/// causes what by testing if variables become independent when controlling
/// for other variables. Features that are directly connected to the target
/// in this causal web are the most important for understanding and prediction.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PCAlgorithmSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxFeatures;
    private readonly double _alpha;
    private readonly int _maxConditioningSetSize;

    private bool[,]? _adjacencyMatrix;
    private double[]? _connectionStrengths;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public double Alpha => _alpha;
    public bool[,]? AdjacencyMatrix => _adjacencyMatrix;
    public double[]? ConnectionStrengths => _connectionStrengths;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PCAlgorithmSelector(
        int maxFeatures = 20,
        double alpha = 0.05,
        int maxConditioningSetSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));
        if (alpha <= 0 || alpha >= 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));

        _maxFeatures = maxFeatures;
        _alpha = alpha;
        _maxConditioningSetSize = maxConditioningSetSize;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PCAlgorithmSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays (include target as last column)
        var X = new double[n, p + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
            X[i, p] = NumOps.ToDouble(target[i]); // Target is index p
        }

        int targetIdx = p;
        int totalVars = p + 1;

        // Initialize complete graph
        _adjacencyMatrix = new bool[totalVars, totalVars];
        for (int i = 0; i < totalVars; i++)
            for (int j = 0; j < totalVars; j++)
                _adjacencyMatrix[i, j] = (i != j);

        // PC algorithm: test conditional independence
        for (int condSize = 0; condSize <= _maxConditioningSetSize; condSize++)
        {
            for (int i = 0; i < totalVars; i++)
            {
                for (int j = i + 1; j < totalVars; j++)
                {
                    if (!_adjacencyMatrix[i, j]) continue;

                    // Get neighbors for conditioning
                    var neighbors = GetNeighbors(i, j, totalVars);
                    if (neighbors.Count < condSize) continue;

                    // Test all conditioning sets of size condSize
                    foreach (var condSet in GetCombinations(neighbors, condSize))
                    {
                        if (TestConditionalIndependence(X, n, i, j, condSet.ToList()))
                        {
                            _adjacencyMatrix[i, j] = false;
                            _adjacencyMatrix[j, i] = false;
                            break;
                        }
                    }
                }
            }
        }

        // Compute connection strengths to target
        _connectionStrengths = new double[p];
        for (int j = 0; j < p; j++)
        {
            if (_adjacencyMatrix[j, targetIdx])
            {
                // Direct connection - use partial correlation strength
                _connectionStrengths[j] = ComputePartialCorrelation(X, n, j, targetIdx, new List<int>());
            }
            else
            {
                // Indirect connection - lower score
                _connectionStrengths[j] = 0.1 * ComputePartialCorrelation(X, n, j, targetIdx, new List<int>());
            }
        }

        // Select features connected to target
        int numToSelect = Math.Min(_maxFeatures, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_connectionStrengths[j]))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private List<int> GetNeighbors(int i, int j, int totalVars)
    {
        if (_adjacencyMatrix is null)
            return new List<int>();

        var neighbors = new List<int>();
        for (int k = 0; k < totalVars; k++)
        {
            if (k != i && k != j && (_adjacencyMatrix[i, k] || _adjacencyMatrix[j, k]))
                neighbors.Add(k);
        }
        return neighbors;
    }

    private IEnumerable<IEnumerable<int>> GetCombinations(List<int> items, int size)
    {
        if (size == 0)
        {
            yield return Enumerable.Empty<int>();
            yield break;
        }

        for (int i = 0; i <= items.Count - size; i++)
        {
            foreach (var combination in GetCombinations(items.Skip(i + 1).ToList(), size - 1))
            {
                yield return new[] { items[i] }.Concat(combination);
            }
        }
    }

    private bool TestConditionalIndependence(double[,] X, int n, int i, int j, List<int> condSet)
    {
        double partialCorr = ComputePartialCorrelation(X, n, i, j, condSet);
        double z = Math.Sqrt(n - condSet.Count - 3) * 0.5 * Math.Log((1 + partialCorr) / (1 - partialCorr + 1e-10));
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));
        return pValue > _alpha;
    }

    private double ComputePartialCorrelation(double[,] X, int n, int i, int j, List<int> condSet)
    {
        if (condSet.Count == 0)
            return ComputeCorrelation(X, n, i, j);

        // Compute residuals after regressing out conditioning variables
        var residI = ComputeResiduals(X, n, i, condSet);
        var residJ = ComputeResiduals(X, n, j, condSet);

        // Compute correlation of residuals
        double meanI = residI.Average();
        double meanJ = residJ.Average();
        double sum = 0, sumSqI = 0, sumSqJ = 0;

        for (int k = 0; k < n; k++)
        {
            double diffI = residI[k] - meanI;
            double diffJ = residJ[k] - meanJ;
            sum += diffI * diffJ;
            sumSqI += diffI * diffI;
            sumSqJ += diffJ * diffJ;
        }

        double denom = Math.Sqrt(sumSqI * sumSqJ) + 1e-10;
        return sum / denom;
    }

    private double ComputeCorrelation(double[,] X, int n, int i, int j)
    {
        double meanI = 0, meanJ = 0;
        for (int k = 0; k < n; k++)
        {
            meanI += X[k, i];
            meanJ += X[k, j];
        }
        meanI /= n;
        meanJ /= n;

        double sum = 0, sumSqI = 0, sumSqJ = 0;
        for (int k = 0; k < n; k++)
        {
            double diffI = X[k, i] - meanI;
            double diffJ = X[k, j] - meanJ;
            sum += diffI * diffJ;
            sumSqI += diffI * diffI;
            sumSqJ += diffJ * diffJ;
        }

        double denom = Math.Sqrt(sumSqI * sumSqJ) + 1e-10;
        return sum / denom;
    }

    private double[] ComputeResiduals(double[,] X, int n, int target, List<int> predictors)
    {
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
            residuals[i] = X[i, target];

        if (predictors.Count == 0)
            return residuals;

        // Simple linear regression
        foreach (int pred in predictors)
        {
            double meanX = 0, meanY = 0;
            for (int i = 0; i < n; i++)
            {
                meanX += X[i, pred];
                meanY += residuals[i];
            }
            meanX /= n;
            meanY /= n;

            double sxy = 0, sxx = 0;
            for (int i = 0; i < n; i++)
            {
                sxy += (X[i, pred] - meanX) * (residuals[i] - meanY);
                sxx += (X[i, pred] - meanX) * (X[i, pred] - meanX);
            }

            double beta = sxx > 1e-10 ? sxy / sxx : 0;
            double intercept = meanY - beta * meanX;

            for (int i = 0; i < n; i++)
                residuals[i] -= beta * X[i, pred] + intercept;
        }

        return residuals;
    }

    private double NormalCDF(double x)
    {
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        double a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x) / Math.Sqrt(2);
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return 0.5 * (1.0 + sign * y);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PCAlgorithmSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PCAlgorithmSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PCAlgorithmSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
