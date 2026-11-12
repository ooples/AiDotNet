using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Autoencoder based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their reconstruction importance in a simple autoencoder,
/// measuring how much each feature contributes to the learned representation.
/// </para>
/// <para><b>For Beginners:</b> An autoencoder compresses data and reconstructs it.
/// Features that are harder to reconstruct accurately are often more important
/// because they carry unique information not captured by other features.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int? _randomState;

    private Vector<T>? _reconstructionImportance;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int HiddenSize => _hiddenSize;
    public int Epochs => _epochs;
    public double LearningRate => _learningRate;
    public Vector<T>? ReconstructionImportance => _reconstructionImportance;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AutoencoderSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 10,
        int epochs = 100,
        double learningRate = 0.01,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _epochs = epochs;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Standardize data
        var x = NeuralSelectorOps.StandardizePopulation(Tensor<T>.FromMatrix(data));
        int h = Math.Min(_hiddenSize, p);

        // Initialize weights (Xavier initialization)
        var w1 = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / (p + h)), p, h);
        var w2 = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / (h + p)), h, p);
        var b1 = new Tensor<T>(new[] { h });
        var b2 = new Tensor<T>(new[] { p });
        T rate = NumOps.FromDouble(_learningRate);
        var rows = NeuralSelectorOps.SplitRows(x);

        // Train autoencoder: tanh hidden layer, linear output, squared reconstruction error, per-sample steps
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Forward pass
                var hidden = engine.TensorTanh(engine.TensorAdd(NeuralSelectorOps.VecMat(rows[i], w1), b1));
                var output = engine.TensorAdd(NeuralSelectorOps.VecMat(hidden, w2), b2);

                // Backward pass (with the decoder weights before their update)
                var dOutput = engine.TensorSubtract(output, rows[i]);
                var dHidden = engine.TensorMultiply(
                    NeuralSelectorOps.MatVec(w2, dOutput), PreprocessingTensorOps.OneMinus(engine.TensorSquare(hidden)));

                // Update weights
                w2 = NeuralSelectorOps.Step(w2, NeuralSelectorOps.Outer(hidden, dOutput), rate);
                b2 = NeuralSelectorOps.Step(b2, dOutput, rate);
                w1 = NeuralSelectorOps.Step(w1, NeuralSelectorOps.Outer(rows[i], dHidden), rate);
                b1 = NeuralSelectorOps.Step(b1, dHidden, rate);
            }
        }

        // Compute mean absolute reconstruction error per feature
        var hiddenAll = engine.TensorTanh(engine.TensorAdd(
            engine.TensorMatMul(x, w1), engine.TensorBroadcastTo(engine.Reshape(b1, new[] { 1, h }), new[] { n, h })));
        var outputAll = engine.TensorAdd(
            engine.TensorMatMul(hiddenAll, w2), engine.TensorBroadcastTo(engine.Reshape(b2, new[] { 1, p }), new[] { n, p }));
        var importance = engine.ReduceMean(engine.TensorAbs(engine.TensorSubtract(x, outputAll)), new[] { 0 }, false);
        _reconstructionImportance = importance.ToVector();

        // Features with higher reconstruction error are potentially more important
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importance, numToSelect);

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutoencoderSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AutoencoderSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AutoencoderSelector has not been fitted.");

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
