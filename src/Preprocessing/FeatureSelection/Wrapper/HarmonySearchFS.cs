using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Harmony Search algorithm for feature selection optimization.
/// </summary>
/// <remarks>
/// <para>
/// Inspired by musical improvisation, Harmony Search maintains a "harmony
/// memory" of good solutions and creates new ones by combining and modifying
/// existing harmonies.
/// </para>
/// <para><b>For Beginners:</b> Like musicians improvising together, this
/// algorithm creates new feature combinations by blending elements from
/// known good combinations, with occasional random exploration.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HarmonySearchFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _harmonyMemorySize;
    private readonly int _nIterations;
    private readonly double _harmonyMemoryConsideringRate;
    private readonly double _pitchAdjustingRate;
    private readonly int _nFeaturesToSelect;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, bool[], double>? _fitnessFunc;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _bestFitness;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public double BestFitness => _bestFitness;
    public override bool SupportsInverseTransform => false;

    public HarmonySearchFS(
        int nFeaturesToSelect = 10,
        int harmonyMemorySize = 20,
        int nIterations = 100,
        double harmonyMemoryConsideringRate = 0.9,
        double pitchAdjustingRate = 0.3,
        Func<Matrix<T>, Vector<T>, bool[], double>? fitnessFunc = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (harmonyMemorySize < 2)
            throw new ArgumentException("Harmony memory size must be at least 2.", nameof(harmonyMemorySize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _harmonyMemorySize = harmonyMemorySize;
        _nIterations = nIterations;
        _harmonyMemoryConsideringRate = harmonyMemoryConsideringRate;
        _pitchAdjustingRate = pitchAdjustingRate;
        _fitnessFunc = fitnessFunc;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HarmonySearchFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize harmony memory
        var harmonyMemory = new List<(bool[] Harmony, double Fitness)>();
        for (int i = 0; i < _harmonyMemorySize; i++)
        {
            var harmony = new bool[p];
            int nSelected = random.Next(1, Math.Min(_nFeaturesToSelect + 3, p));
            var indices = Enumerable.Range(0, p).OrderBy(_ => random.Next()).Take(nSelected).ToList();
            foreach (int idx in indices)
                harmony[idx] = true;

            double fitness = EvaluateFitness(data, target, harmony);
            harmonyMemory.Add((harmony, fitness));
        }

        _featureImportances = new double[p];
        _bestFitness = harmonyMemory.Max(h => h.Fitness);

        // Main optimization loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Create new harmony
            var newHarmony = new bool[p];

            for (int j = 0; j < p; j++)
            {
                if (random.NextDouble() < _harmonyMemoryConsideringRate)
                {
                    // Select from harmony memory
                    int memoryIdx = random.Next(_harmonyMemorySize);
                    newHarmony[j] = harmonyMemory[memoryIdx].Harmony[j];

                    // Pitch adjustment
                    if (random.NextDouble() < _pitchAdjustingRate)
                        newHarmony[j] = !newHarmony[j];
                }
                else
                {
                    // Random selection
                    newHarmony[j] = random.NextDouble() < 0.5;
                }
            }

            // Ensure at least one feature selected
            if (!newHarmony.Any(b => b))
                newHarmony[random.Next(p)] = true;

            // Limit to approximately nFeaturesToSelect
            int currentCount = newHarmony.Count(b => b);
            while (currentCount > _nFeaturesToSelect + 2)
            {
                var selectedIdx = Enumerable.Range(0, p).Where(i => newHarmony[i]).ToList();
                int toRemove = selectedIdx[random.Next(selectedIdx.Count)];
                newHarmony[toRemove] = false;
                currentCount--;
            }

            double newFitness = EvaluateFitness(data, target, newHarmony);

            // Update feature importances
            for (int j = 0; j < p; j++)
                if (newHarmony[j])
                    _featureImportances[j] += newFitness;

            // Update harmony memory
            int worstIdx = 0;
            double worstFitness = harmonyMemory[0].Fitness;
            for (int i = 1; i < _harmonyMemorySize; i++)
            {
                if (harmonyMemory[i].Fitness < worstFitness)
                {
                    worstFitness = harmonyMemory[i].Fitness;
                    worstIdx = i;
                }
            }

            if (newFitness > worstFitness)
            {
                harmonyMemory[worstIdx] = (newHarmony, newFitness);
                if (newFitness > _bestFitness)
                    _bestFitness = newFitness;
            }
        }

        // Select best harmony
        var bestHarmony = harmonyMemory.OrderByDescending(h => h.Fitness).First().Harmony;
        var selectedList = new List<int>();
        for (int j = 0; j < p; j++)
            if (bestHarmony[j])
                selectedList.Add(j);

        // Limit to nFeaturesToSelect
        if (selectedList.Count > _nFeaturesToSelect)
        {
            selectedList = selectedList
                .OrderByDescending(j => _featureImportances[j])
                .Take(_nFeaturesToSelect)
                .ToList();
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double EvaluateFitness(Matrix<T> data, Vector<T> target, bool[] mask)
    {
        if (_fitnessFunc is not null)
            return _fitnessFunc(data, target, mask);

        int n = data.Rows;
        int nSelected = mask.Count(b => b);
        if (nSelected == 0) return double.NegativeInfinity;

        double totalCorr = 0;
        for (int j = 0; j < mask.Length; j++)
        {
            if (!mask[j]) continue;

            double xMean = 0, yMean = 0;
            for (int i = 0; i < n; i++)
            {
                xMean += NumOps.ToDouble(data[i, j]);
                yMean += NumOps.ToDouble(target[i]);
            }
            xMean /= n;
            yMean /= n;

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

        double penalty = nSelected > _nFeaturesToSelect ? 0.1 * (nSelected - _nFeaturesToSelect) : 0;
        return totalCorr / nSelected - penalty;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HarmonySearchFS has not been fitted.");

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
        throw new NotSupportedException("HarmonySearchFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HarmonySearchFS has not been fitted.");

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
