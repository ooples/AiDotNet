using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Firefly Algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Firefly Algorithm is inspired by the flashing behavior of fireflies.
/// Brighter fireflies attract dimmer ones, with attraction decreasing with
/// distance. This creates a natural clustering toward optimal solutions.
/// </para>
/// <para><b>For Beginners:</b> Imagine fireflies searching for the best spot
/// in a forest. Brighter fireflies (better solutions) attract dimmer ones,
/// so all fireflies gradually move toward the brightest areas. In feature
/// selection, "brightness" represents how good a set of features is.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FireflyAlgorithm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _alpha;
    private readonly double _beta0;
    private readonly double _gamma;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int MaxIterations => _maxIterations;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FireflyAlgorithm(
        int nFeaturesToSelect = 10,
        int populationSize = 25,
        int maxIterations = 50,
        double alpha = 0.5,
        double beta0 = 1.0,
        double gamma = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 2)
            throw new ArgumentException("Population size must be at least 2.", nameof(populationSize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _maxIterations = maxIterations;
        _alpha = alpha;
        _beta0 = beta0;
        _gamma = gamma;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FireflyAlgorithm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize fireflies
        var fireflies = new double[_populationSize, p];
        for (int i = 0; i < _populationSize; i++)
            for (int j = 0; j < p; j++)
                fireflies[i, j] = random.NextDouble();

        var intensity = new double[_populationSize];
        for (int i = 0; i < _populationSize; i++)
            intensity[i] = EvaluateSolution(data, target, fireflies, i, p, n);

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < _populationSize; j++)
                {
                    if (intensity[j] > intensity[i])
                    {
                        // Move firefly i towards j
                        double distance = ComputeDistance(fireflies, i, j, p);
                        double beta = _beta0 * Math.Exp(-_gamma * distance * distance);

                        for (int k = 0; k < p; k++)
                        {
                            double movement = beta * (fireflies[j, k] - fireflies[i, k])
                                            + _alpha * (random.NextDouble() - 0.5);
                            fireflies[i, k] = Math.Max(0, Math.Min(1, fireflies[i, k] + movement));
                        }
                    }
                }

                intensity[i] = EvaluateSolution(data, target, fireflies, i, p, n);
            }
        }

        // Find best firefly
        int bestIdx = 0;
        double bestIntensity = intensity[0];
        for (int i = 1; i < _populationSize; i++)
        {
            if (intensity[i] > bestIntensity)
            {
                bestIntensity = intensity[i];
                bestIdx = i;
            }
        }

        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = fireflies[bestIdx, j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeDistance(double[,] fireflies, int i, int j, int p)
    {
        double sum = 0;
        for (int k = 0; k < p; k++)
        {
            double diff = fireflies[i, k] - fireflies[j, k];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private double EvaluateSolution(Matrix<T> data, Vector<T> target, double[,] positions, int idx, int p, int n)
    {
        var selectedFeatures = Enumerable.Range(0, p)
            .Where(j => positions[idx, j] > 0.5)
            .ToArray();

        if (selectedFeatures.Length == 0)
            return 0;

        double totalScore = 0;
        foreach (int f in selectedFeatures)
            totalScore += ComputeCorrelation(data, target, f, n);

        int targetCount = Math.Min(_nFeaturesToSelect, p);
        double countPenalty = Math.Abs(selectedFeatures.Length - targetCount);

        return totalScore / selectedFeatures.Length - 0.1 * countPenalty;
    }

    private double ComputeCorrelation(Matrix<T> data, Vector<T> target, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, j]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FireflyAlgorithm has not been fitted.");

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
        throw new NotSupportedException("FireflyAlgorithm does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FireflyAlgorithm has not been fitted.");

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
