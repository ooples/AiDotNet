using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantum;

/// <summary>
/// Quantum-Inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses quantum-inspired optimization with quantum bits (qubits) represented
/// as probability amplitudes to search the feature subset space.
/// </para>
/// <para><b>For Beginners:</b> This method is inspired by quantum computing
/// principles. Each feature is represented by a "qubit" that can be in a
/// superposition of selected and not-selected states. The algorithm evolves
/// these quantum states to find the best feature subset, exploring many
/// possibilities simultaneously.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class QuantumInspiredSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _rotationAngle;

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

    public QuantumInspiredSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 20,
        int nGenerations = 100,
        double rotationAngle = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _rotationAngle = rotationAngle;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "QuantumInspiredSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Initialize quantum population (qubits as angles): sin^2(theta) is the probability of observing 1,
        // and every qubit starts in the 50-50 superposition theta = pi / 4
        T quarterPi = NumOps.FromDouble(Math.PI / 4);
        T sizePenalty = NumOps.FromDouble(0.01);
        var qubits = new Tensor<T>[_populationSize];
        for (int i = 0; i < _populationSize; i++)
            qubits[i] = PreprocessingTensorOps.Filled(new[] { p }, quarterPi);

        // Best solution tracking
        var bestSolution = new Tensor<T>(new[] { p });
        Tensor<T>? bestFitness = null;
        // counted in int: a 0/1 sum in T stops counting at 2048 for Half
        var selectionCounts = new Tensor<int>(new[] { p });

        for (int gen = 0; gen < _nGenerations; gen++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Observe (collapse) qubits to a classical solution, ensuring at least one feature
                var solution = ParetoOps.EnsureNonEmpty(Observe(qubits[i], PopulationSearchOps.Uniform<T>(rand, p)), rand);

                // Evaluate fitness and update the best
                var fitness = PopulationSearchOps.SubsetFitness(
                    engine.Reshape(solution, new[] { 1, p }), correlations, _nFeaturesToSelect,
                    sizePenalty, NumOps.MinValue);
                if (bestFitness is null || PopulationSearchOps.Greater(fitness, bestFitness))
                {
                    bestFitness = fitness;
                    bestSolution = solution;
                }

                // Update selection counts
                selectionCounts = engine.TensorAdd(selectionCounts, solution.Cast<int>());

                // Quantum rotation gate update towards the best solution
                qubits[i] = Rotate(qubits[i], solution, bestSolution, _rotationAngle);
            }
        }

        // Compute feature scores from selection counts and best solution
        var featureScores = engine.TensorAdd(
            PopulationSearchOps.Frequency<T>(selectionCounts, (double)_populationSize * _nGenerations),
            bestSolution);
        _featureScores = featureScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(featureScores, numToSelect);

        IsFitted = true;
    }

    /// <summary>Observation of a qubit row: bit j is 1 when <paramref name="draws"/>[j] &lt; sin^2(theta_j).</summary>
    internal static Tensor<T> Observe(Tensor<T> theta, Tensor<T> draws)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorLessThan(draws, engine.TensorSquare(engine.TensorSin(theta)));
    }

    /// <summary>
    /// The rotation gate of Han and Kim (2002) for a solution worse than or equal to the best: where the observed
    /// bit differs from the best solution's bit, theta turns by <paramref name="angle"/> towards it, increasing
    /// (towards pi / 2, bit 1) while theta &lt; pi / 2 when the best bit is 1 and decreasing (towards 0, bit 0)
    /// while theta &gt; 0 when it is 0.
    /// </summary>
    internal static Tensor<T> Rotate(Tensor<T> theta, Tensor<T> solution, Tensor<T> best, double angle)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var differs = engine.TensorAbs(engine.TensorSubtract(solution, best));
        var up = engine.TensorMultiply(
            engine.TensorMultiply(differs, best), engine.TensorLessThan(theta, ops.FromDouble(Math.PI / 2)));
        var down = engine.TensorMultiply(
            engine.TensorMultiply(differs, PreprocessingTensorOps.OneMinus(best)), engine.TensorGreaterThan(theta, ops.Zero));
        return engine.TensorAdd(theta, engine.TensorMultiplyScalar(engine.TensorSubtract(up, down), ops.FromDouble(angle)));
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
            throw new InvalidOperationException("QuantumInspiredSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("QuantumInspiredSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantumInspiredSelector has not been fitted.");

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
