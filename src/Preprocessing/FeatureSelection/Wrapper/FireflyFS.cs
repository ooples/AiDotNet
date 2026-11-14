using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FireflyFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _alpha;  // Randomness parameter
    private readonly double _beta0;  // Attractiveness at r=0
    private readonly double _gamma;  // Light absorption coefficient
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _fitnessFunction;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FireflyFS(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        double alpha = 0.5,
        double beta0 = 1.0,
        double gamma = 1.0,
        Func<Matrix<T>, Vector<T>, int[], T>? fitnessFunction = null,
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FireflyFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize fireflies: positions uniform on [0, 1), drawn from the selector's generator in master's
        // order
        var fireflies = new Tensor<T>[_populationSize];
        var brightness = new T[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            fireflies[i] = WrapperScoringOps.Draws<T>(random, p);
            brightness[i] = EvaluateSolution(fireflies[i], data, target);
        }

        T beta0 = NumOps.FromDouble(_beta0);
        T noiseScale = NumOps.FromDouble(_alpha);
        T noiseOffset = NumOps.FromDouble(-0.5 * _alpha);
        T negativeGamma = NumOps.FromDouble(-_gamma);
        T zero = NumOps.Zero;
        T one = NumOps.One;

        // Optimization loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < _populationSize; j++)
                {
                    if (WrapperScoringOps.IsAbove(brightness[j], brightness[i]))
                    {
                        // Attractiveness beta0 exp(-gamma r^2) with r the distance between the fireflies
                        var difference = Engine.TensorSubtract(fireflies[j], fireflies[i]);
                        T squaredDistance = Engine.TensorSumOfSquares(difference);
                        T beta = WrapperScoringOps.Multiply(beta0,
                            WrapperScoringOps.Exp(WrapperScoringOps.Multiply(negativeGamma, squaredDistance)));

                        // Move firefly i toward j, plus a random step alpha u with u uniform on [-0.5, 0.5)
                        var noise = WrapperScoringOps.Affine(WrapperScoringOps.Draws<T>(random, p), noiseScale, noiseOffset);
                        fireflies[i] = Engine.TensorClamp(Engine.TensorAdd(fireflies[i],
                            Engine.TensorAdd(Engine.TensorMultiplyScalar(difference, beta), noise)), zero, one);

                        brightness[i] = EvaluateSolution(fireflies[i], data, target);
                    }
                }
            }
        }

        // Find best firefly (the first best)
        int bestIdx = PreprocessingTensorOps.FirstArgMax(new Tensor<T>(brightness, new[] { _populationSize }));

        // Calculate feature importances from best solution
        _featureImportances = fireflies[bestIdx].ToVector();

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(fireflies[bestIdx], nToSelect);

        IsFitted = true;
    }

    private T EvaluateSolution(Tensor<T> position, Matrix<T> data, Vector<T> target)
    {
        // Convert continuous position to binary selection
        var selectedFeatures = PreprocessingTensorOps.MaskPositions(Engine.TensorGreaterThan(position, NumOps.FromDouble(0.5)));

        if (selectedFeatures.Length == 0)
            return NumOps.Zero;

        if (_fitnessFunction is not null)
            return _fitnessFunction(data, target, selectedFeatures);

        return DefaultFitness(data, target, selectedFeatures);
    }

    private T DefaultFitness(Matrix<T> data, Vector<T> target, int[] features)
    {
        // Correlation-based fitness: mean |Pearson r| with the target, minus 0.01 per selected feature
        return NumOps.Subtract(WrapperScoringOps.MeanAbsTargetCorrelation(data, target, features),
            NumOps.FromDouble(0.01 * features.Length));
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
            throw new InvalidOperationException("FireflyFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FireflyFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FireflyFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
