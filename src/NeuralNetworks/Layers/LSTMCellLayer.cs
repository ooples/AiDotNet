using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// One step of a long short-term memory cell, <c>(h', c') = LSTM(x, h, c)</c>, for recurrent decoders that run a step at
/// a time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Hochreiter &amp; Schmidhuber (1997) with forget gates (Gers et al. 2000):
/// <c>i, f, o = σ(W x + U h + b)</c>, <c>g = tanh(W_g x + U_g h + b_g)</c>, <c>c' = f ⊙ c + i ⊙ g</c>,
/// <c>h' = o ⊙ tanh(c')</c>. An optional cap clips <c>c'</c> to ±cap (Non-Attentive Tacotron, Table 6: "LSTM cell abs
/// value cap 10.0").
/// </para>
/// <para>Inputs: <c>x</c> <c>[batch, input]</c> and the state <c>[batch, 2 · hidden]</c> holding <c>h</c> then <c>c</c>;
/// the output is the next state in the same layout (use <see cref="SplitState"/>).</para>
/// <para><b>For Beginners:</b> An LSTM keeps a separate long-term memory that it adds to and erases from through gates,
/// which lets it remember things over many steps.</para>
/// </remarks>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ApiShape = LayerApiShape.DualTensor, TestInputShape = "1, 4", TestConstructorArgs = "4, 2, 0.0")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class LSTMCellLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly double _cellClip;

    [SubLayerInput("1, _inputSize")]
    private readonly DenseLayer<T> _input;
    [SubLayerInput("1, _hiddenSize")]
    private readonly BiasFreeLinearLayer<T> _recurrent;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool RequiresMultipleInputs => true;

    /// <summary>Creates an LSTM cell.</summary>
    /// <param name="inputSize">Width of the step input.</param>
    /// <param name="hiddenSize">Width of the hidden and cell states.</param>
    /// <param name="cellClip">Clip the cell state to ±this value; 0 for no clipping.</param>
    public LSTMCellLayer([LayerState] int inputSize, [LayerState] int hiddenSize, [LayerState] double cellClip = 0.0)
        : base(new[] { inputSize }, new[] { 2 * hiddenSize })
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (cellClip < 0) throw new ArgumentOutOfRangeException(nameof(cellClip));
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _cellClip = cellClip;
        _input = new DenseLayer<T>(4 * hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _recurrent = new BiasFreeLinearLayer<T>(hiddenSize, 4 * hiddenSize);
        RegisterSubLayer(_input);
        RegisterSubLayer(_recurrent);
    }

    /// <summary>Width of the hidden and cell states.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Width of the step input.</summary>
    public int InputSize => _inputSize;

    /// <summary>
    /// Loads one layer of a PyTorch <c>nn.LSTM</c>: <c>weight_ih_l{k}</c> <c>[4H, input]</c>, <c>weight_hh_l{k}</c>
    /// <c>[4H, H]</c> and the two biases <c>[4H]</c>, whose gates are ordered i, f, g, o as here; the two biases add.
    /// </summary>
    internal void LoadTorchWeights(double[] weightIh, double[] weightHh, double[] biasIh, double[] biasHh)
    {
        int gates = 4 * _hiddenSize;
        if (weightIh.Length != gates * _inputSize || weightHh.Length != gates * _hiddenSize || biasIh.Length != gates || biasHh.Length != gates)
            throw new ArgumentException($"Expected weights [{gates}, {_inputSize}] and [{gates}, {_hiddenSize}] and biases [{gates}].");
        // The input projection sizes itself on its first forward; one zero step materializes it.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            _input.Forward(new Tensor<T>(new[] { 1, _inputSize }));
        var input = _input.GetWeights();                                                    // [input, 4H]
        var bias = _input.GetBiases();
        var recurrent = _recurrent.Weights;                                                 // [H, 4H]
        for (int g = 0; g < gates; g++)
        {
            for (int i = 0; i < _inputSize; i++) input[i, g] = NumOps.FromDouble(weightIh[g * _inputSize + i]);
            for (int h = 0; h < _hiddenSize; h++) recurrent[h, g] = NumOps.FromDouble(weightHh[g * _hiddenSize + h]);
            bias[g] = NumOps.FromDouble(biasIh[g] + biasHh[g]);
        }
        Engine.InvalidatePersistentTensor(input);
        Engine.InvalidatePersistentTensor(bias);
        Engine.InvalidatePersistentTensor(recurrent);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 2
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(2 * _hiddenSize)),
            }
            : null;

    /// <summary>One step: the next state <c>[h'; c']</c> from the step input and the state <c>[h; c]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> state) => Forward(new[] { input, state });

    /// <summary>Splits a step output <c>[batch, 2 · hidden]</c> into the hidden and cell states.</summary>
    public (Tensor<T> Hidden, Tensor<T> Cell) SplitState(Tensor<T> state)
    {
        int batch = state.Shape[0];
        return (Engine.TensorSlice(state, new[] { 0, 0 }, new[] { batch, _hiddenSize }),
                Engine.TensorSlice(state, new[] { 0, _hiddenSize }, new[] { batch, _hiddenSize }));
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => throw new InvalidOperationException($"{nameof(LSTMCellLayer<T>)} needs the current state; call Forward(input, state).");

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length != 2)
            throw new ArgumentException("Expected the step input and the state [h; c].", nameof(inputs));
        var x = inputs[0];
        var state = inputs[1];
        int batch = x.Shape[0], n = _hiddenSize;
        if (x.Rank != 2 || x.Shape[1] != _inputSize)
            throw new ArgumentException($"Expected input [batch, {_inputSize}], got [{string.Join(", ", x.Shape)}].", nameof(inputs));
        if (state.Rank != 2 || state.Shape[1] != 2 * n || state.Shape[0] != batch)
            throw new ArgumentException($"Expected the state [{batch}, {2 * n}].", nameof(inputs));
        var (h, c) = SplitState(state);

        var gates = Engine.TensorAdd(_input.Forward(x), _recurrent.Forward(h));          // [B, 4H]: i | f | g | o
        Tensor<T> Gate(int k) => Engine.TensorSlice(gates, new[] { 0, k * n }, new[] { batch, n });
        var i = Engine.Sigmoid(Gate(0));
        var f = Engine.Sigmoid(Gate(1));
        var g = Engine.Tanh(Gate(2));
        var o = Engine.Sigmoid(Gate(3));
        var cNext = Engine.TensorAdd(Engine.TensorMultiply(f, c), Engine.TensorMultiply(i, g));
        if (_cellClip > 0)
            cNext = Engine.TensorClamp(cNext, NumOps.FromDouble(-_cellClip), NumOps.FromDouble(_cellClip));
        var hNext = Engine.TensorMultiply(o, Engine.Tanh(cNext));
        return Engine.TensorConcatenate(new[] { hNext, cNext }, 1);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputSize"] = _inputSize.ToString(inv);
        metadata["HiddenSize"] = _hiddenSize.ToString(inv);
        metadata["CellClip"] = _cellClip.ToString("R", inv);
        return metadata;
    }
}
