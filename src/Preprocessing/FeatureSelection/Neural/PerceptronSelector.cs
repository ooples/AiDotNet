using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Perceptron based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their weight magnitudes in a trained single-layer
/// perceptron, identifying features with the strongest learned connections.
/// </para>
/// <para><b>For Beginners:</b> A perceptron is the simplest neural network - just
/// one layer of weights connecting inputs to output. Features with larger weights
/// (positive or negative) are more important for the prediction.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PerceptronSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int? _randomState;

    private Vector<T>? _weightImportance;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int Epochs => _epochs;
    public double LearningRate => _learningRate;
    public Vector<T>? WeightImportance => _weightImportance;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PerceptronSelector(
        int nFeaturesToSelect = 10,
        int epochs = 100,
        double learningRate = 0.01,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _epochs = epochs;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PerceptronSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Standardize data and target
        var x = NeuralSelectorOps.StandardizePopulation(Tensor<T>.FromMatrix(data));
        var y = engine.Reshape(
            NeuralSelectorOps.StandardizePopulation(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })), new[] { n });

        // Initialize weights
        var weights = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(1.0 / p), p);
        var bias = new Tensor<T>(new[] { 1 });
        T rate = NumOps.FromDouble(_learningRate);
        var rows = NeuralSelectorOps.SplitRows(x);
        var targets = NeuralSelectorOps.SplitValues(y);

        // Train the linear unit by per-sample gradient descent on the squared error (Widrow-Hoff)
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                var prediction = engine.TensorAdd(
                    engine.ReduceSum(engine.TensorMultiply(rows[i], weights), new[] { 0 }, true), bias);
                var error = engine.TensorSubtract(prediction, targets[i]);
                weights = NeuralSelectorOps.Step(weights, PopulationSearchOps.Scale(rows[i], error), rate);
                bias = NeuralSelectorOps.Step(bias, error, rate);
            }
        }

        // Feature importance based on absolute weight
        var importance = engine.TensorAbs(weights);
        _weightImportance = importance.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(importance, numToSelect);

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
            throw new InvalidOperationException("PerceptronSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PerceptronSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PerceptronSelector has not been fitted.");

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
