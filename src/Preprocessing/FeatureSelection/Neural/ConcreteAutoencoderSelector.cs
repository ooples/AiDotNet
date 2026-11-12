using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Concrete Autoencoder Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Concrete Autoencoder which learns a discrete feature selection
/// mask using the Concrete/Gumbel-Softmax distribution for differentiable
/// feature selection.
/// </para>
/// <para><b>For Beginners:</b> The Concrete Autoencoder learns which features
/// to select as part of its training. It uses a special trick (Gumbel-Softmax)
/// that lets it learn a "soft" selection that gradually becomes "hard" (0 or 1)
/// as training progresses.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ConcreteAutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _nEpochs;
    private readonly double _initialTemperature;
    private readonly double _minTemperature;

    private Vector<T>? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SelectionProbabilities => _selectionProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ConcreteAutoencoderSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 32,
        int nEpochs = 200,
        double initialTemperature = 10.0,
        double minTemperature = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _nEpochs = nEpochs;
        _initialTemperature = initialTemperature;
        _minTemperature = minTemperature;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        // Normalize
        var x = NeuralSelectorOps.StandardizeSample(Tensor<T>.FromMatrix(data));
        var rows = NeuralSelectorOps.SplitRows(x);

        int k = Math.Min(_nFeaturesToSelect, p);
        int h = Math.Min(_hiddenSize, p);
        var rand = RandomHelper.CreateSecureRandom();

        // Selection logits (one for each feature, for each of k selections) and the decoder k -> h -> p
        var network = new ConcreteNetwork
        {
            Logits = NeuralSelectorOps.SymmetricUniform<T>(rand, 0.1, k, p),
            HiddenWeights = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / (k + h)), k, h),
            HiddenBias = new Tensor<T>(new[] { h }),
            OutputWeights = NeuralSelectorOps.SymmetricUniform<T>(rand, Math.Sqrt(2.0 / (h + p)), h, p),
            OutputBias = new Tensor<T>(new[] { p }),
        };
        T rate = NumOps.FromDouble(0.01);

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            // Anneal temperature: T0 (T_B / T0)^(epoch / B)
            double temp = _initialTemperature * Math.Pow(_minTemperature / _initialTemperature, (double)epoch / _nEpochs);

            for (int i = 0; i < n; i++)
            {
                var gradients = network.Gradients(rows[i], PopulationSearchOps.Uniform<T>(rand, k, p), temp);
                network.Logits = NeuralSelectorOps.Step(network.Logits, gradients.Logits, rate);
                network.HiddenWeights = NeuralSelectorOps.Step(network.HiddenWeights, gradients.HiddenWeights, rate);
                network.HiddenBias = NeuralSelectorOps.Step(network.HiddenBias, gradients.HiddenBias, rate);
                network.OutputWeights = NeuralSelectorOps.Step(network.OutputWeights, gradients.OutputWeights, rate);
                network.OutputBias = NeuralSelectorOps.Step(network.OutputBias, gradients.OutputBias, rate);
            }
        }

        // Compute selection probabilities from final logits
        var probabilities = engine.ReduceSum(engine.TensorSoftmax(network.Logits, 1), new[] { 0 }, false);
        _selectionProbabilities = probabilities.ToVector();
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(probabilities, k);

        IsFitted = true;
    }

    /// <summary>
    /// The concrete autoencoder of Abid, Balin and Zou (2019): k concrete selector nodes with logits [k, p] pick
    /// m_s = softmax((logits_s + gumbel) / T) and pass v_s = m_s . x to a decoder with one LeakyReLU (0.2) hidden
    /// layer of size h and a linear output of size p, trained on the squared reconstruction error.
    /// </summary>
    internal sealed class ConcreteNetwork
    {
        public Tensor<T> Logits { get; set; } = new Tensor<T>(new[] { 0, 0 });

        public Tensor<T> HiddenWeights { get; set; } = new Tensor<T>(new[] { 0, 0 });

        public Tensor<T> HiddenBias { get; set; } = new Tensor<T>(new[] { 0 });

        public Tensor<T> OutputWeights { get; set; } = new Tensor<T>(new[] { 0, 0 });

        public Tensor<T> OutputBias { get; set; } = new Tensor<T>(new[] { 0 });

        /// <summary>
        /// The reconstruction loss |output - x|^2 / 2 of one sample x [p] for the uniforms [k, p] of the Gumbel
        /// noise -log(-log(u + 1e-20) + 1e-20) at temperature <paramref name="temperature"/>. Returns [1].
        /// </summary>
        public Tensor<T> Loss(Tensor<T> x, Tensor<T> uniforms, double temperature)
        {
            var engine = AiDotNetEngine.Current;
            Forward(x, uniforms, temperature, out _, out _, out _, out _, out var error);
            return PopulationSearchOps.Affine(engine.ReduceSum(engine.TensorSquare(error), new[] { 0 }, true), 0.5, 0);
        }

        /// <summary>
        /// Gradients of <see cref="Loss"/> with respect to every parameter. With e = output - x, the decoder
        /// gradients are the usual back-propagation ones; through the selector, dv = W_h dhidden and, since
        /// dv_s / dlogit_sj = m_sj (x_j - v_s) / T, dlogits_sj = dv_s m_sj (x_j - v_s) / T.
        /// </summary>
        public ConcreteNetwork Gradients(Tensor<T> x, Tensor<T> uniforms, double temperature)
        {
            var engine = AiDotNetEngine.Current;
            var ops = MathHelper.GetNumericOperations<T>();
            int k = Logits.Shape[0];
            int p = Logits.Shape[1];
            Forward(x, uniforms, temperature, out var selection, out var selected, out var hiddenPre, out var hidden, out var error);

            var slope = engine.TensorWhere(
                engine.TensorGreaterThan(hiddenPre, ops.Zero),
                FeatureSelectionTensorOps.Filled(new[] { hidden.Length }, ops.One),
                FeatureSelectionTensorOps.Filled(new[] { hidden.Length }, ops.FromDouble(0.2)));
            var hiddenDelta = engine.TensorMultiply(NeuralSelectorOps.MatVec(OutputWeights, error), slope);
            var selectedDelta = NeuralSelectorOps.MatVec(HiddenWeights, hiddenDelta);

            var gap = engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.Reshape(x, new[] { 1, p }), new[] { k, p }),
                engine.TensorBroadcastTo(engine.Reshape(selected, new[] { k, 1 }), new[] { k, p }));
            var logitGradient = PopulationSearchOps.Affine(
                engine.TensorMultiply(
                    engine.TensorMultiply(selection, gap),
                    engine.TensorBroadcastTo(engine.Reshape(selectedDelta, new[] { k, 1 }), new[] { k, p })),
                1 / temperature, 0);

            return new ConcreteNetwork
            {
                Logits = logitGradient,
                HiddenWeights = NeuralSelectorOps.Outer(selected, hiddenDelta),
                HiddenBias = hiddenDelta,
                OutputWeights = NeuralSelectorOps.Outer(hidden, error),
                OutputBias = error,
            };
        }

        private void Forward(
            Tensor<T> x, Tensor<T> uniforms, double temperature, out Tensor<T> selection, out Tensor<T> selected,
            out Tensor<T> hiddenPre, out Tensor<T> hidden, out Tensor<T> error)
        {
            var engine = AiDotNetEngine.Current;
            var ops = MathHelper.GetNumericOperations<T>();
            T tiny = ops.FromDouble(1e-20);
            var gumbel = engine.TensorNegate(engine.TensorLog(engine.TensorAddScalar(
                engine.TensorNegate(engine.TensorLog(engine.TensorAddScalar(uniforms, tiny))), tiny)));
            selection = engine.TensorSoftmax(PopulationSearchOps.Affine(engine.TensorAdd(Logits, gumbel), 1 / temperature, 0), 1);
            selected = NeuralSelectorOps.MatVec(selection, x);
            hiddenPre = engine.TensorAdd(NeuralSelectorOps.VecMat(selected, HiddenWeights), HiddenBias);
            hidden = engine.TensorWhere(
                engine.TensorGreaterThan(hiddenPre, ops.Zero), hiddenPre, engine.TensorMultiplyScalar(hiddenPre, ops.FromDouble(0.2)));
            var output = engine.TensorAdd(NeuralSelectorOps.VecMat(hidden, OutputWeights), OutputBias);
            error = engine.TensorSubtract(output, x);
        }
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConcreteAutoencoderSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ConcreteAutoencoderSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConcreteAutoencoderSelector has not been fitted.");

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
