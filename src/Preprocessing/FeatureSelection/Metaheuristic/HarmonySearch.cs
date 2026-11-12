using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HarmonySearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _harmonyMemorySize;
    private readonly int _maxIterations;
    private readonly double _hmcr;
    private readonly double _par;
    private readonly double _bw;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int HarmonyMemorySize => _harmonyMemorySize;
    public int MaxIterations => _maxIterations;
    public double HMCR => _hmcr;
    public double PAR => _par;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HarmonySearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int targetCount = Math.Min(_nFeaturesToSelect, p);
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Initialize harmony memory
        var harmonyMemory = PopulationSearchOps.Uniform<T>(random, _harmonyMemorySize, p);
        var fitness = PopulationSearchOps.ThresholdFitness(harmonyMemory, correlations, targetCount, 0.1);

        T bandwidth = NumOps.FromDouble(_bw);
        var memorySource = new int[p];
        var adjustSource = new int[p];
        var memoryRows = new int[p];
        var unused = new Tensor<T>(new[] { 1 });

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Improvise a new harmony. The draws per dimension are conditional (memory consideration, then a memory
            // row and a pitch-adjustment test with its bandwidth draw, or else a fresh random value), so they are
            // taken one by one in Geem's order; every value derived from them is computed below on whole tensors.
            var drawn = new Tensor<T>[p];
            for (int j = 0; j < p; j++)
            {
                memorySource[j] = 0;
                adjustSource[j] = 0;
                memoryRows[j] = 0;
                drawn[j] = unused;
                if (random.NextDouble() < _hmcr)
                {
                    memorySource[j] = 1;
                    memoryRows[j] = random.Next(_harmonyMemorySize);
                    if (random.NextDouble() < _par)
                    {
                        adjustSource[j] = 1;
                        drawn[j] = PopulationSearchOps.Uniform<T>(random, 1);
                    }
                }
                else
                {
                    drawn[j] = PopulationSearchOps.Uniform<T>(random, 1);
                }
            }

            var draws = engine.TensorConcatenate(drawn, 0);
            var fromMemory = new Tensor<int>((int[])memorySource.Clone(), new[] { p }).Cast<T>();
            var adjusted = new Tensor<int>((int[])adjustSource.Clone(), new[] { p }).Cast<T>();

            // Memory consideration with pitch adjustment: x + bw * (2u - 1), clamped to [0, 1]
            var memoryValues = PopulationSearchOps.GatherPerColumn(harmonyMemory, memoryRows);
            var pitched = PopulationSearchOps.Clamp01(engine.TensorAdd(
                memoryValues, engine.TensorMultiplyScalar(PopulationSearchOps.Affine(draws, 2, -1), bandwidth)));
            var newHarmony = engine.TensorWhere(
                fromMemory, engine.TensorWhere(adjusted, pitched, memoryValues), draws);
            var newFitness = PopulationSearchOps.ThresholdFitness(
                engine.Reshape(newHarmony, new[] { 1, p }), correlations, targetCount, 0.1);

            // Replace the worst harmony if the new one is better
            int worstIdx = PopulationSearchOps.ArgMin(fitness);
            if (PopulationSearchOps.Greater(newFitness, PopulationSearchOps.At(fitness, worstIdx)))
            {
                harmonyMemory = PopulationSearchOps.ReplaceRow(harmonyMemory, worstIdx, newHarmony);
                fitness = PopulationSearchOps.ReplaceAt(fitness, worstIdx, newFitness);
            }
        }

        // Best harmony
        var best = PopulationSearchOps.Row(harmonyMemory, PopulationSearchOps.ArgMax(fitness));
        _featureScores = best.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(best, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HarmonySearch has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HarmonySearch does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HarmonySearch has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
