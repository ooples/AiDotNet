using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantum;

/// <summary>
/// Quantum Annealing-Inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Simulates quantum annealing for feature selection, using quantum tunneling
/// effects to escape local optima and find globally optimal feature subsets.
/// </para>
/// <para><b>For Beginners:</b> Quantum annealing is like simulated annealing
/// (slowly cooling a system) but with quantum effects that allow "tunneling"
/// through barriers to find better solutions. This method simulates this
/// process to search for the best feature combination.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class QuantumAnnealingSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSteps;
    private readonly double _initialGamma;
    private readonly double _finalGamma;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public QuantumAnnealingSelector(
        int nFeaturesToSelect = 10,
        int nSteps = 1000,
        double initialGamma = 1.0,
        double finalGamma = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSteps = nSteps;
        _initialGamma = initialGamma;
        _finalGamma = finalGamma;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "QuantumAnnealingSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        var rand = RandomHelper.CreateSecureRandom();

        // Precompute feature correlations with target
        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Initialize current solution
        var current = PopulationSearchOps.IndexMask<T>(
            PreprocessingTensorOps.TopKIndicesSorted(correlations, Math.Min(_nFeaturesToSelect, p)), p);

        var currentEnergy = ComputeEnergy(correlations, current, _nFeaturesToSelect);
        var best = current;
        var bestEnergy = currentEnergy;

        // counted in int: a 0/1 sum in T stops counting at 2048 for Half
        var selectionCounts = new Tensor<int>(new[] { p });
        var zero = new Tensor<T>(new[] { 1 });

        for (int step = 0; step < _nSteps; step++)
        {
            // Anneal gamma (quantum tunneling strength)
            double progress = (double)step / _nSteps;
            double gamma = _initialGamma * Math.Pow(_finalGamma / _initialGamma, progress);

            // Temperature decreases linearly
            double temp = (1 - progress) + 0.01;

            // Propose a move (flip a random bit)
            var flip = FeatureSelectionTensorOps.OneHot<T>(rand.Next(p), p);
            var proposal = engine.TensorWhere(flip, PreprocessingTensorOps.OneMinus(current), current);

            // Ensure valid solution
            if (!PopulationSearchOps.Any(proposal))
                continue;

            var newEnergy = ComputeEnergy(correlations, proposal, _nFeaturesToSelect);

            // Accept or reject with quantum tunneling probability: 1 downhill, else max(exp(-|d| / gamma), exp(-d / temp))
            var delta = engine.TensorSubtract(newEnergy, currentEnergy);
            var tunnelProb = engine.TensorExp(PopulationSearchOps.Affine(engine.TensorAbs(delta), -1 / gamma, 0));
            var thermalProb = engine.TensorExp(PopulationSearchOps.Affine(delta, -1 / temp, 0));
            var draw = PopulationSearchOps.Uniform<T>(rand, 1);
            bool accept = !PopulationSearchOps.Greater(delta, zero)
                || PopulationSearchOps.Greater(engine.TensorMax(tunnelProb, thermalProb), draw);

            if (accept)
            {
                current = proposal;
                currentEnergy = newEnergy;
                if (PopulationSearchOps.Greater(bestEnergy, currentEnergy))
                {
                    bestEnergy = currentEnergy;
                    best = current;
                }
            }

            // Track selection frequency
            selectionCounts = engine.TensorAdd(selectionCounts, current.Cast<int>());
        }

        // Compute feature scores: correlation + selection frequency + 1 for the best solution's features
        var featureScores = engine.TensorAdd(
            engine.TensorAdd(correlations, PopulationSearchOps.Frequency<T>(selectionCounts, _nSteps)), best);
        _featureScores = featureScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(featureScores, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Energy of a 0/1 mask [p]: minus the summed correlations of its features plus 0.1 |count - target count|.
    /// Returns [1].
    /// </summary>
    internal static Tensor<T> ComputeEnergy(Tensor<T> correlations, Tensor<T> solution, int targetCount)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var gain = engine.ReduceSum(engine.TensorMultiply(solution, correlations), new[] { 0 }, true);
        // counted in int: a 0/1 sum in T stops counting at 2048 for Half
        var count = engine.ReduceSum(solution.Cast<int>(), new[] { 0 }, true).Cast<T>();
        var penalty = engine.TensorMultiplyScalar(
            engine.TensorAbs(engine.TensorAddScalar(count, ops.FromDouble(-targetCount))), ops.FromDouble(0.1));
        return engine.TensorSubtract(penalty, gain);
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
            throw new InvalidOperationException("QuantumAnnealingSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("QuantumAnnealingSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantumAnnealingSelector has not been fitted.");

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
