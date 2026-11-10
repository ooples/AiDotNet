using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>A PyTorch-compatible LSTM step retaining independent input and recurrent biases.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>Both biases contribute to the gates and remain independently trainable. Folding them
/// into one bias preserves inference but changes optimizer updates.</remarks>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ApiShape = LayerApiShape.DualTensor, TestInputShape = "1, 4", TestConstructorArgs = "4, 2")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class TorchLstmCellLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputSize, _hiddenSize;
    [SubLayerInput("1, _inputSize")]
    private readonly DenseLayer<T> _input;
    [SubLayerInput("1, _hiddenSize")]
    private readonly DenseLayer<T> _recurrent;
    public override bool SupportsTraining => true;
    public override bool RequiresMultipleInputs => true;

    /// <summary>Creates a cell with i, f, g, o gate ordering and two distinct biases.</summary>
    public TorchLstmCellLayer([LayerState] int inputSize, [LayerState] int hiddenSize)
        : base(new[] { inputSize }, new[] { 2 * hiddenSize })
    {
        if (inputSize <= 0 || hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        _inputSize = inputSize; _hiddenSize = hiddenSize;
        _input = new DenseLayer<T>(4 * hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _recurrent = new DenseLayer<T>(4 * hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        RegisterSubLayer(_input); RegisterSubLayer(_recurrent);
    }

    internal DenseLayer<T> InputProjection => _input;
    internal DenseLayer<T> RecurrentProjection => _recurrent;
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> state) => Forward(new[] { input, state });
    public (Tensor<T> Hidden, Tensor<T> Cell) SplitState(Tensor<T> state) =>
        (Engine.TensorSlice(state, new[] { 0, 0 }, new[] { state.Shape[0], _hiddenSize }),
         Engine.TensorSlice(state, new[] { 0, _hiddenSize }, new[] { state.Shape[0], _hiddenSize }));

    /// <summary>Loads one layer of a PyTorch <c>nn.LSTM</c> (<c>weight_ih [4H, input]</c>, <c>weight_hh [4H, H]</c> and
    /// both biases <c>[4H]</c>) straight from <paramref name="source"/>.</summary>
    internal void LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string weightIhName, string weightHhName, string biasIhName,
        string biasHhName)
    {
        int gates = 4 * _hiddenSize;
        void Load(DenseLayer<T> layer, int input, string weightName, string biasName)
        {
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) layer.Forward(new Tensor<T>(new[] { 1, input }));
            var w = layer.GetWeights(); var b = layer.GetBiases();
            source.CopyTransposedTo(weightName, gates, input, w);
            source.CopyTo(biasName, new[] { gates }, b);
            Engine.InvalidatePersistentTensor(w); Engine.InvalidatePersistentTensor(b);
        }
        Load(_input, _inputSize, weightIhName, biasIhName); Load(_recurrent, _hiddenSize, weightHhName, biasHhName);
    }

    internal void LoadTorchWeights(double[] weightIh, double[] weightHh, double[] biasIh, double[] biasHh)
    {
        int gates = 4 * _hiddenSize;
        if (weightIh.Length != gates * _inputSize || weightHh.Length != gates * _hiddenSize ||
            biasIh.Length != gates || biasHh.Length != gates) throw new ArgumentException("LSTM checkpoint shape differs.");
        void Load(DenseLayer<T> layer, int input, double[] weights, double[] biases)
        {
            using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) layer.Forward(new Tensor<T>(new[] { 1, input }));
            var w = layer.GetWeights(); var b = layer.GetBiases();
            for (int g = 0; g < gates; g++)
            {
                b[g] = NumOps.FromDouble(biases[g]);
                for (int i = 0; i < input; i++) w[i, g] = NumOps.FromDouble(weights[g * input + i]);
            }
            Engine.InvalidatePersistentTensor(w); Engine.InvalidatePersistentTensor(b);
        }
        Load(_input, _inputSize, weightIh, biasIh); Load(_recurrent, _hiddenSize, weightHh, biasHh);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input) => throw new InvalidOperationException("An LSTM step requires input and state.");
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs.Length != 2) throw new ArgumentException("Expected input and [hidden;cell] state.", nameof(inputs));
        var x = inputs[0]; var state = inputs[1];
        if (x.Rank != 2 || x.Shape[1] != _inputSize || state.Rank != 2 || state.Shape[0] != x.Shape[0] || state.Shape[1] != 2 * _hiddenSize)
            throw new ArgumentException("LSTM input or state shape differs.", nameof(inputs));
        var (h, c) = SplitState(state); int batch = x.Shape[0];
        var gates = Engine.TensorAdd(_input.Forward(x), _recurrent.Forward(h));
        Tensor<T> Gate(int index) => Engine.TensorSlice(gates, new[] { 0, index * _hiddenSize }, new[] { batch, _hiddenSize });
        var next = Engine.TensorAdd(Engine.TensorMultiply(Engine.Sigmoid(Gate(1)), c),
            Engine.TensorMultiply(Engine.Sigmoid(Gate(0)), Engine.Tanh(Gate(2))));
        return Engine.TensorConcatenate(new[] { Engine.TensorMultiply(Engine.Sigmoid(Gate(3)), Engine.Tanh(next)), next }, 1);
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2 ? new[]
    {
        new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(2 * _hiddenSize))
    } : null;
    public override void ResetState() { }
}
