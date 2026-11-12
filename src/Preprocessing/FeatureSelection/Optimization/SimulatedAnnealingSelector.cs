using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Optimization;

/// <summary>
/// Simulated Annealing based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses simulated annealing optimization to find optimal feature subsets by
/// accepting both improvements and occasionally worse solutions to escape local optima.
/// </para>
/// <para><b>For Beginners:</b> Like cooling metal, this starts with high "temperature"
/// where it accepts almost any change, then gradually cools down and becomes pickier.
/// This helps find good feature combinations by exploring broadly first, then refining.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SimulatedAnnealingSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _initialTemperature;
    private readonly double _coolingRate;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double InitialTemperature => _initialTemperature;
    public double CoolingRate => _coolingRate;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SimulatedAnnealingSelector(
        int nFeaturesToSelect = 10,
        double initialTemperature = 100.0,
        double coolingRate = 0.95,
        int nIterations = 1000,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _initialTemperature = initialTemperature;
        _coolingRate = coolingRate;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SimulatedAnnealingSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var scores = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize current solution
        var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect).ToArray();
        var current = PopulationSearchOps.IndexMask<T>(indices, p);

        var currentEnergy = Energy(current, scores);
        var best = current;
        var bestEnergy = currentEnergy;

        var importances = new Tensor<T>(new[] { p });
        var temperature = FeatureSelectionTensorOps.Filled(new[] { 1 }, NumOps.FromDouble(_initialTemperature));
        T cooling = NumOps.FromDouble(_coolingRate);
        var zero = new Tensor<T>(new[] { 1 });

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Generate neighbor by swapping a feature
            var neighbor = current;
            var selected = FeatureSelectionTensorOps.MaskPositions(current);

            if (rand.NextDouble() < 0.5 && selected.Length > 1)
            {
                // Remove a random selected feature
                int toRemove = selected[rand.Next(selected.Length)];
                neighbor = engine.TensorSubtract(neighbor, FeatureSelectionTensorOps.OneHot<T>(toRemove, p));

                // Add a random unselected feature
                var unselected = FeatureSelectionTensorOps.MaskPositions(FeatureSelectionTensorOps.OneMinus(neighbor));
                if (unselected.Length > 0)
                {
                    int toAdd = unselected[rand.Next(unselected.Length)];
                    neighbor = engine.TensorAdd(neighbor, FeatureSelectionTensorOps.OneHot<T>(toAdd, p));
                }
            }
            else
            {
                // Flip a random feature
                var flip = FeatureSelectionTensorOps.OneHot<T>(rand.Next(p), p);
                neighbor = engine.TensorWhere(flip, FeatureSelectionTensorOps.OneMinus(neighbor), neighbor);
            }

            var neighborEnergy = Energy(neighbor, scores);
            var delta = engine.TensorSubtract(neighborEnergy, currentEnergy);

            // Metropolis criterion: accept downhill moves, uphill ones with probability exp(-delta / T)
            bool accept = PopulationSearchOps.Greater(zero, delta)
                || PopulationSearchOps.Greater(
                    engine.TensorExp(engine.TensorNegate(engine.TensorDivide(delta, temperature))),
                    PopulationSearchOps.Uniform<T>(rand, 1));
            if (accept)
            {
                current = neighbor;
                currentEnergy = neighborEnergy;

                if (PopulationSearchOps.Greater(bestEnergy, currentEnergy))
                {
                    best = current;
                    bestEnergy = currentEnergy;
                }
            }

            // Track feature importance
            importances = engine.TensorSubtract(importances, PopulationSearchOps.Scale(current, currentEnergy));

            // Cool down
            temperature = engine.TensorMultiplyScalar(temperature, cooling);
        }

        _selectedIndices = FeatureSelectionTensorOps.MaskPositions(best);

        // Normalize feature importances
        var maxImportance = ParetoOps.Max(importances);
        if (PopulationSearchOps.Greater(maxImportance, zero))
            importances = engine.TensorDivide(importances, engine.TensorBroadcastTo(maxImportance, new[] { p }));
        _featureImportances = importances.ToVector();

        IsFitted = true;
    }

    /// <summary>The energy of a 0/1 mask [p]: minus the mean absolute target correlation of its features (0 when empty).</summary>
    private Tensor<T> Energy(Tensor<T> mask, Tensor<T> scores) =>
        Engine.TensorNegate(PopulationSearchOps.SubsetFitness(
            Engine.Reshape(mask, new[] { 1, mask.Length }), scores, 0, NumOps.Zero, NumOps.Zero));

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
            throw new InvalidOperationException("SimulatedAnnealingSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SimulatedAnnealingSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SimulatedAnnealingSelector has not been fitted.");

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
