using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Firefly Algorithm for feature selection optimization.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Firefly Algorithm (FA) metaheuristic inspired by the flashing behavior
/// of fireflies. Brighter fireflies (better solutions) attract others, with
/// attraction decreasing with distance.
/// </para>
/// <para><b>For Beginners:</b> Imagine fireflies in a field at night. Brighter
/// fireflies attract dimmer ones. In feature selection, "brightness" is how good
/// a feature subset is, and fireflies move toward better solutions while also
/// randomly exploring new combinations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FireflyFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _alpha;  // Randomness parameter
    private readonly double _beta0;  // Attractiveness at r=0
    private readonly double _gamma;  // Light absorption coefficient
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _fitnessFunction;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FireflyFS(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        double alpha = 0.5,
        double beta0 = 1.0,
        double gamma = 1.0,
        Func<Matrix<T>, Vector<T>, int[], double>? fitnessFunction = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 5)
            throw new ArgumentException("Population size must be at least 5.", nameof(populationSize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _maxIterations = maxIterations;
        _alpha = alpha;
        _beta0 = beta0;
        _gamma = gamma;
        _fitnessFunction = fitnessFunction;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FireflyFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize fireflies (binary position vectors)
        var fireflies = new double[_populationSize][];
        var brightness = new double[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            fireflies[i] = new double[p];
            for (int j = 0; j < p; j++)
                fireflies[i][j] = random.NextDouble();
            brightness[i] = EvaluateSolution(fireflies[i], data, target);
        }

        // Optimization loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < _populationSize; j++)
                {
                    if (brightness[j] > brightness[i])
                    {
                        // Calculate distance
                        double r = 0;
                        for (int k = 0; k < p; k++)
                            r += Math.Pow(fireflies[i][k] - fireflies[j][k], 2);
                        r = Math.Sqrt(r);

                        // Calculate attractiveness
                        double beta = _beta0 * Math.Exp(-_gamma * r * r);

                        // Move firefly i toward j
                        for (int k = 0; k < p; k++)
                        {
                            fireflies[i][k] += beta * (fireflies[j][k] - fireflies[i][k])
                                             + _alpha * (random.NextDouble() - 0.5);
                            fireflies[i][k] = Math.Max(0, Math.Min(1, fireflies[i][k]));
                        }

                        brightness[i] = EvaluateSolution(fireflies[i], data, target);
                    }
                }
            }
        }

        // Find best firefly
        int bestIdx = 0;
        for (int i = 1; i < _populationSize; i++)
            if (brightness[i] > brightness[bestIdx])
                bestIdx = i;

        // Calculate feature importances from best solution
        _featureImportances = (double[])fireflies[bestIdx].Clone();

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureImportances
            .Select((v, idx) => (Value: v, Index: idx))
            .OrderByDescending(x => x.Value)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateSolution(double[] position, Matrix<T> data, Vector<T> target)
    {
        // Convert continuous position to binary selection
        var selectedFeatures = new List<int>();
        for (int i = 0; i < position.Length; i++)
            if (position[i] > 0.5)
                selectedFeatures.Add(i);

        if (selectedFeatures.Count == 0)
            return 0;

        if (_fitnessFunction is not null)
            return _fitnessFunction(data, target, selectedFeatures.ToArray());

        return DefaultFitness(data, target, selectedFeatures.ToArray());
    }

    private double DefaultFitness(Matrix<T> data, Vector<T> target, int[] features)
    {
        // Correlation-based fitness
        int n = data.Rows;
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double totalCorr = 0;
        foreach (int j in features)
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
                totalCorr += Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
        }

        return totalCorr / features.Length - 0.01 * features.Length;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FireflyFS has not been fitted.");

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
        throw new NotSupportedException("FireflyFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FireflyFS has not been fitted.");

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
