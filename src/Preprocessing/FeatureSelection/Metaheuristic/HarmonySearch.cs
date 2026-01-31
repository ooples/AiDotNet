using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Harmony Search for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Harmony Search is inspired by the improvisation process of musicians.
/// Just as musicians create harmonies by combining notes from memory with
/// random variations, this algorithm creates solutions by mixing previous
/// good solutions with random exploration.
/// </para>
/// <para><b>For Beginners:</b> Imagine a band trying to write the perfect song.
/// They remember good riffs from past songs (harmony memory) and sometimes try
/// completely new notes (random pitch). The best combinations are kept for
/// future improvisation. In feature selection, we're looking for the perfect
/// "harmony" of features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HarmonySearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _harmonyMemorySize;
    private readonly int _maxIterations;
    private readonly double _hmcr;
    private readonly double _par;
    private readonly double _bw;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int HarmonyMemorySize => _harmonyMemorySize;
    public int MaxIterations => _maxIterations;
    public double HMCR => _hmcr;
    public double PAR => _par;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HarmonySearch(
        int nFeaturesToSelect = 10,
        int harmonyMemorySize = 20,
        int maxIterations = 100,
        double hmcr = 0.9,
        double par = 0.3,
        double bw = 0.1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (harmonyMemorySize < 2)
            throw new ArgumentException("Harmony memory size must be at least 2.", nameof(harmonyMemorySize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _harmonyMemorySize = harmonyMemorySize;
        _maxIterations = maxIterations;
        _hmcr = hmcr;
        _par = par;
        _bw = bw;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HarmonySearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize harmony memory
        var harmonyMemory = new double[_harmonyMemorySize, p];
        for (int i = 0; i < _harmonyMemorySize; i++)
            for (int j = 0; j < p; j++)
                harmonyMemory[i, j] = random.NextDouble();

        var fitness = new double[_harmonyMemorySize];
        for (int i = 0; i < _harmonyMemorySize; i++)
            fitness[i] = EvaluateSolution(data, target, harmonyMemory, i, p, n);

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Create new harmony
            var newHarmony = new double[p];

            for (int j = 0; j < p; j++)
            {
                if (random.NextDouble() < _hmcr)
                {
                    // Memory consideration
                    int memIdx = random.Next(_harmonyMemorySize);
                    newHarmony[j] = harmonyMemory[memIdx, j];

                    // Pitch adjustment
                    if (random.NextDouble() < _par)
                    {
                        newHarmony[j] += _bw * (2 * random.NextDouble() - 1);
                        newHarmony[j] = Math.Max(0, Math.Min(1, newHarmony[j]));
                    }
                }
                else
                {
                    // Random selection
                    newHarmony[j] = random.NextDouble();
                }
            }

            // Evaluate new harmony
            var tempMemory = (double[,])harmonyMemory.Clone();
            for (int j = 0; j < p; j++)
                tempMemory[0, j] = newHarmony[j];

            double newFitness = EvaluateSolution(data, target, tempMemory, 0, p, n);

            // Find worst harmony
            int worstIdx = 0;
            double worstFitness = fitness[0];
            for (int i = 1; i < _harmonyMemorySize; i++)
            {
                if (fitness[i] < worstFitness)
                {
                    worstFitness = fitness[i];
                    worstIdx = i;
                }
            }

            // Replace if better
            if (newFitness > worstFitness)
            {
                for (int j = 0; j < p; j++)
                    harmonyMemory[worstIdx, j] = newHarmony[j];
                fitness[worstIdx] = newFitness;
            }
        }

        // Find best harmony
        int bestIdx = 0;
        double bestFitness = fitness[0];
        for (int i = 1; i < _harmonyMemorySize; i++)
        {
            if (fitness[i] > bestFitness)
            {
                bestFitness = fitness[i];
                bestIdx = i;
            }
        }

        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = harmonyMemory[bestIdx, j];

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
            throw new InvalidOperationException("HarmonySearch has not been fitted.");

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
        throw new NotSupportedException("HarmonySearch does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HarmonySearch has not been fitted.");

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
