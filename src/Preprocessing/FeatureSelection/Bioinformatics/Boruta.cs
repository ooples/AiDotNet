using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Boruta feature selection algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Boruta creates "shadow features" (shuffled copies of real features) and compares
/// feature importance against these shadows. Features consistently outperforming
/// the best shadow are confirmed as important.
/// </para>
/// <para><b>For Beginners:</b> Boruta asks: "Is this feature better than random noise?"
/// It shuffles your features to create meaningless versions (shadows), then compares
/// real features against these shadows. Features that consistently beat their shadows
/// are truly informative.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Boruta<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxIterations;
    private readonly double _pValue;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _importanceFunction;

    private double[]? _importances;
    private BorutaDecision[]? _decisions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum BorutaDecision
    {
        Confirmed,
        Rejected,
        Tentative
    }

    public int MaxIterations => _maxIterations;
    public double[]? Importances => _importances;
    public BorutaDecision[]? Decisions => _decisions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public Boruta(
        int maxIterations = 100,
        double pValue = 0.05,
        Func<Matrix<T>, Vector<T>, double[]>? importanceFunction = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxIterations < 10)
            throw new ArgumentException("Max iterations must be at least 10.", nameof(maxIterations));
        if (pValue <= 0 || pValue >= 1)
            throw new ArgumentException("P-value must be between 0 and 1.", nameof(pValue));

        _maxIterations = maxIterations;
        _pValue = pValue;
        _importanceFunction = importanceFunction;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "Boruta requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Track hits (times feature beats max shadow)
        var hits = new int[p];
        _importances = new double[p];

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Create shadow features (shuffled copies)
            var extendedData = new T[n, 2 * p];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                    extendedData[i, j] = data[i, j];
            }

            // Create shuffled shadows
            for (int j = 0; j < p; j++)
            {
                var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                for (int i = 0; i < n; i++)
                    extendedData[i, p + j] = data[indices[i], j];
            }

            var extendedMatrix = new Matrix<T>(extendedData);

            // Get importances
            var allImportances = GetImportances(extendedMatrix, target);

            // Find max shadow importance
            double maxShadow = double.MinValue;
            for (int j = p; j < 2 * p; j++)
                maxShadow = Math.Max(maxShadow, allImportances[j]);

            // Count hits
            for (int j = 0; j < p; j++)
            {
                if (allImportances[j] > maxShadow)
                    hits[j]++;
                _importances[j] += allImportances[j];
            }
        }

        // Average importances
        for (int j = 0; j < p; j++)
            _importances[j] /= _maxIterations;

        // Make decisions using binomial test
        _decisions = new BorutaDecision[p];
        double threshold = BinomialThreshold(_maxIterations, _pValue);

        for (int j = 0; j < p; j++)
        {
            if (hits[j] >= threshold)
                _decisions[j] = BorutaDecision.Confirmed;
            else if (hits[j] <= _maxIterations - threshold)
                _decisions[j] = BorutaDecision.Rejected;
            else
                _decisions[j] = BorutaDecision.Tentative;
        }

        // Select confirmed features (and tentative if no confirmed)
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_decisions[j] == BorutaDecision.Confirmed)
                selected.Add(j);
        }

        if (selected.Count == 0)
        {
            // Include tentative features
            for (int j = 0; j < p; j++)
            {
                if (_decisions[j] == BorutaDecision.Tentative)
                    selected.Add(j);
            }
        }

        if (selected.Count == 0)
        {
            // Fall back to best features by importance
            selected = _importances
                .Select((imp, idx) => (Imp: imp, Index: idx))
                .OrderByDescending(x => x.Imp)
                .Take(Math.Max(1, p / 10))
                .Select(x => x.Index)
                .ToList();
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double[] GetImportances(Matrix<T> data, Vector<T> target)
    {
        if (_importanceFunction is not null)
            return _importanceFunction(data, target);

        return DefaultImportances(data, target);
    }

    private double[] DefaultImportances(Matrix<T> data, Vector<T> target)
    {
        // Correlation-based importance
        int n = data.Rows;
        int p = data.Columns;
        var importances = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
                importances[j] = Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
        }

        return importances;
    }

    private double BinomialThreshold(int n, double alpha)
    {
        // Approximate using normal approximation to binomial
        // Under null, hits ~ Binomial(n, 0.5)
        double mean = n * 0.5;
        double std = Math.Sqrt(n * 0.25);
        double z = NormalQuantile(1 - alpha);
        return mean + z * std;
    }

    private double NormalQuantile(double p)
    {
        // Approximate inverse normal CDF
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        double t = Math.Sqrt(-2 * Math.Log(1 - p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("Boruta has not been fitted.");

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
        throw new NotSupportedException("Boruta does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("Boruta has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
