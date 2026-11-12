using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Attention-based Feature Selection using neural attention mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// Uses an attention mechanism to learn feature importance weights.
/// The attention scores indicate which features the model focuses on
/// when making predictions.
/// </para>
/// <para><b>For Beginners:</b> Attention is like a spotlight that tells
/// the model where to focus. Features that get more "spotlight" (higher
/// attention) are more important for making predictions. This method
/// learns those importance weights automatically from the data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AttentionBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _nEpochs;
    private readonly double _learningRate;

    private Vector<T>? _attentionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? AttentionScores => _attentionScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AttentionBasedSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 32,
        int nEpochs = 100,
        double learningRate = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _nEpochs = nEpochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AttentionBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Convert and normalize data
        var x = NeuralSelectorOps.StandardizeSample(Tensor<T>.FromMatrix(data));
        var rows = NeuralSelectorOps.SplitRows(x);
        var targets = NeuralSelectorOps.SplitValues(Tensor<T>.FromVector(target));

        // Initialize attention network
        int hidden = Math.Min(_hiddenSize, p);
        var w1 = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / (p + hidden)), p, hidden);
        var w2 = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / (hidden + 1)), hidden);
        var wPred = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / (p + 1)), p);
        T rate = NumOps.FromDouble(_learningRate);

        // Train with attention mechanism: per-sample gradient descent on the squared error
        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                var (gradW1, gradW2, gradPred) = Gradients(rows[i], targets[i], w1, w2, wPred);
                w1 = NeuralSelectorOps.Step(w1, gradW1, rate);
                w2 = NeuralSelectorOps.Step(w2, gradW2, rate);
                wPred = NeuralSelectorOps.Step(wPred, gradPred, rate);
            }
        }

        // Final attention scores: the mean attention over the samples
        var attention = new Tensor<T>(new[] { p });
        for (int i = 0; i < n; i++)
            attention = engine.TensorAdd(attention, Forward(rows[i], w1, w2, wPred, out _, out _));
        var scores = engine.TensorDivideScalar(attention, NumOps.FromDouble(n));
        _attentionScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// The attention model for one sample x [p]: h = tanh(x W1) [hidden], scores s = W1 (h .* w2) [p],
    /// attention a = softmax(s) and prediction sum_j a_j x_j v_j with v = <paramref name="wPred"/>. Returns a.
    /// </summary>
    internal static Tensor<T> Forward(
        Tensor<T> x, Tensor<T> w1, Tensor<T> w2, Tensor<T> wPred, out Tensor<T> hidden, out Tensor<T> prediction)
    {
        var engine = AiDotNetEngine.Current;
        hidden = engine.TensorTanh(NeuralSelectorOps.VecMat(x, w1));
        var scores = NeuralSelectorOps.MatVec(w1, engine.TensorMultiply(hidden, w2));
        var attention = engine.TensorSoftmax(scores, 0);
        prediction = engine.ReduceSum(engine.TensorMultiply(attention, engine.TensorMultiply(x, wPred)), new[] { 0 }, true);
        return attention;
    }

    /// <summary>
    /// Gradients of the loss (prediction - y)^2 / 2 of one sample with respect to W1 [p, hidden], w2 [hidden] and
    /// v [p]. With e the error, g = e x .* v and ds = a .* (g - a . g) the softmax backward pass:
    /// dv = e a .* x, dw2 = (W1^T ds) .* h, and dW1 = ds (h .* w2)^T + x ((W1^T ds) .* w2 .* (1 - h^2))^T, the
    /// second term coming from h = tanh(x W1).
    /// </summary>
    internal static (Tensor<T> W1, Tensor<T> W2, Tensor<T> WPred) Gradients(
        Tensor<T> x, Tensor<T> y, Tensor<T> w1, Tensor<T> w2, Tensor<T> wPred)
    {
        var engine = AiDotNetEngine.Current;
        var attention = Forward(x, w1, w2, wPred, out var hidden, out var prediction);
        var error = engine.TensorSubtract(prediction, y);

        var gradPred = PopulationSearchOps.Scale(engine.TensorMultiply(attention, x), error);
        var g = PopulationSearchOps.Scale(engine.TensorMultiply(x, wPred), error);
        var ag = engine.ReduceSum(engine.TensorMultiply(attention, g), new[] { 0 }, true);
        var ds = engine.TensorMultiply(attention, engine.TensorSubtract(g, engine.TensorBroadcastTo(ag, new[] { x.Length })));

        var back = NeuralSelectorOps.VecMat(ds, w1); // W1^T ds [hidden]
        var gradW2 = engine.TensorMultiply(back, hidden);
        var throughTanh = engine.TensorMultiply(
            engine.TensorMultiply(back, w2), FeatureSelectionTensorOps.OneMinus(engine.TensorSquare(hidden)));
        var gradW1 = engine.TensorAdd(
            NeuralSelectorOps.Outer(ds, engine.TensorMultiply(hidden, w2)), NeuralSelectorOps.Outer(x, throughTanh));
        return (gradW1, gradW2, gradPred);
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
            throw new InvalidOperationException("AttentionBasedSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AttentionBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AttentionBasedSelector has not been fitted.");

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
