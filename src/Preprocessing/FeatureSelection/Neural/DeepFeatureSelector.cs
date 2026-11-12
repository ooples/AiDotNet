using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Deep Feature Selection using a multi-layer neural network.
/// </summary>
/// <remarks>
/// <para>
/// Uses a deep neural network with L1 regularization on the first layer
/// to perform feature selection while learning non-linear relationships.
/// </para>
/// <para><b>For Beginners:</b> This method trains a multi-layer neural network
/// but penalizes the input layer weights. Features with larger input weights
/// after training are considered more important. The network can capture
/// complex non-linear patterns that simpler methods might miss.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DeepFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int[] _hiddenLayers;
    private readonly int _nEpochs;
    private readonly double _l1Penalty;

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

    public DeepFeatureSelector(
        int nFeaturesToSelect = 10,
        int[]? hiddenLayers = null,
        int nEpochs = 100,
        double l1Penalty = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenLayers = hiddenLayers ?? [64, 32];
        _nEpochs = nEpochs;
        _l1Penalty = l1Penalty;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DeepFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var rand = RandomHelper.CreateSecureRandom();

        // Normalize
        var x = NeuralSelectorOps.StandardizeSample(Tensor<T>.FromMatrix(data));
        var rows = NeuralSelectorOps.SplitRows(x);
        var targets = NeuralSelectorOps.SplitValues(Tensor<T>.FromVector(target));

        // Build layer sizes
        var layerSizes = new List<int> { p };
        foreach (int h in _hiddenLayers)
            layerSizes.Add(Math.Min(h, p));
        layerSizes.Add(1); // Output layer

        // Initialize weights (He-scaled uniform) and zero biases
        var weights = new List<Tensor<T>>();
        var biases = new List<Tensor<T>>();
        for (int l = 0; l < layerSizes.Count - 1; l++)
        {
            weights.Add(NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / layerSizes[l]), layerSizes[l], layerSizes[l + 1]));
            biases.Add(new Tensor<T>(new[] { layerSizes[l + 1] }));
        }

        T rate = NumOps.FromDouble(0.001);
        T l1 = NumOps.FromDouble(_l1Penalty);
        T zero = NumOps.Zero;
        int last = weights.Count - 1;

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Forward pass: ReLU hidden layers, linear output
                var activations = new List<Tensor<T>> { rows[i] };
                var preActivations = new List<Tensor<T>>();
                for (int l = 0; l < weights.Count; l++)
                {
                    var pre = engine.TensorAdd(NeuralSelectorOps.VecMat(activations[l], weights[l]), biases[l]);
                    preActivations.Add(pre);
                    activations.Add(l < last ? engine.TensorMax(pre, zero) : pre);
                }

                // Squared-error gradient, then back-propagation through the ReLU layers
                var deltas = new Tensor<T>[weights.Count];
                deltas[last] = engine.TensorSubtract(activations[weights.Count], targets[i]);
                for (int l = last - 1; l >= 0; l--)
                {
                    deltas[l] = engine.TensorMultiply(
                        NeuralSelectorOps.MatVec(weights[l + 1], deltas[l + 1]),
                        engine.TensorGreaterThan(preActivations[l], zero));
                }

                // Update weights, with the L1 subgradient on the first layer only
                for (int l = 0; l < weights.Count; l++)
                {
                    var gradient = NeuralSelectorOps.Outer(activations[l], deltas[l]);
                    if (l == 0)
                        gradient = engine.TensorAdd(gradient, engine.TensorMultiplyScalar(engine.TensorSign(weights[0]), l1));

                    weights[l] = NeuralSelectorOps.Step(weights[l], gradient, rate);
                    biases[l] = NeuralSelectorOps.Step(biases[l], deltas[l], rate);
                }
            }
        }

        // Compute feature importance from first layer weights
        var importance = engine.ReduceSum(engine.TensorAbs(weights[0]), new[] { 1 }, false);
        _featureImportances = importance.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importance, numToSelect);

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
            throw new InvalidOperationException("DeepFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DeepFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DeepFeatureSelector has not been fitted.");

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
