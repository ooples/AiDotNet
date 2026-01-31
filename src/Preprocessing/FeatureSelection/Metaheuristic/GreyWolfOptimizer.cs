using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Grey Wolf Optimizer (GWO) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// GWO mimics the leadership hierarchy and hunting mechanism of grey wolves.
/// Alpha, beta, and delta wolves guide the pack toward optimal solutions.
/// </para>
/// <para><b>For Beginners:</b> Imagine a wolf pack hunting. The leaders (alpha,
/// beta, delta) guide the others toward prey. In optimization, this translates
/// to the best solutions guiding the search. Wolves encircle prey and then
/// attack - similarly, we narrow in on good feature subsets.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GreyWolfOptimizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
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

    public GreyWolfOptimizer(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 4)
            throw new ArgumentException("Population size must be at least 4.", nameof(populationSize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _maxIterations = maxIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GreyWolfOptimizer requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize population (continuous positions)
        var positions = new double[_populationSize, p];
        for (int i = 0; i < _populationSize; i++)
            for (int j = 0; j < p; j++)
                positions[i, j] = random.NextDouble();

        var fitness = new double[_populationSize];
        for (int i = 0; i < _populationSize; i++)
            fitness[i] = EvaluateSolution(data, target, positions, i, p, n);

        // Track best wolves
        var sortedIndices = Enumerable.Range(0, _populationSize)
            .OrderByDescending(i => fitness[i])
            .ToArray();

        int alphaIdx = sortedIndices[0];
        int betaIdx = sortedIndices[1];
        int deltaIdx = sortedIndices[2];

        var alphaPos = new double[p];
        var betaPos = new double[p];
        var deltaPos = new double[p];

        for (int j = 0; j < p; j++)
        {
            alphaPos[j] = positions[alphaIdx, j];
            betaPos[j] = positions[betaIdx, j];
            deltaPos[j] = positions[deltaIdx, j];
        }

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double a = 2 - iter * 2.0 / _maxIterations;

            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    // Encircling prey
                    double r1 = random.NextDouble();
                    double r2 = random.NextDouble();
                    double A1 = 2 * a * r1 - a;
                    double C1 = 2 * r2;
                    double D_alpha = Math.Abs(C1 * alphaPos[j] - positions[i, j]);
                    double X1 = alphaPos[j] - A1 * D_alpha;

                    r1 = random.NextDouble();
                    r2 = random.NextDouble();
                    double A2 = 2 * a * r1 - a;
                    double C2 = 2 * r2;
                    double D_beta = Math.Abs(C2 * betaPos[j] - positions[i, j]);
                    double X2 = betaPos[j] - A2 * D_beta;

                    r1 = random.NextDouble();
                    r2 = random.NextDouble();
                    double A3 = 2 * a * r1 - a;
                    double C3 = 2 * r2;
                    double D_delta = Math.Abs(C3 * deltaPos[j] - positions[i, j]);
                    double X3 = deltaPos[j] - A3 * D_delta;

                    positions[i, j] = Math.Max(0, Math.Min(1, (X1 + X2 + X3) / 3));
                }

                fitness[i] = EvaluateSolution(data, target, positions, i, p, n);
            }

            // Update alpha, beta, delta
            sortedIndices = Enumerable.Range(0, _populationSize)
                .OrderByDescending(i => fitness[i])
                .ToArray();

            alphaIdx = sortedIndices[0];
            betaIdx = sortedIndices[1];
            deltaIdx = sortedIndices[2];

            for (int j = 0; j < p; j++)
            {
                alphaPos[j] = positions[alphaIdx, j];
                betaPos[j] = positions[betaIdx, j];
                deltaPos[j] = positions[deltaIdx, j];
            }
        }

        // Extract best solution
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = alphaPos[j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
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
            throw new InvalidOperationException("GreyWolfOptimizer has not been fitted.");

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
        throw new NotSupportedException("GreyWolfOptimizer does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GreyWolfOptimizer has not been fitted.");

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
