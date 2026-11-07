using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>The recurrent cell a <see cref="BidirectionalRecurrentLayer{T}"/> runs in each direction.</summary>
public enum RecurrentCellType
{
    /// <summary>Long short-term memory (<see cref="LSTMLayer{T}"/>).</summary>
    Lstm = 0,

    /// <summary>Gated recurrent unit (<see cref="GRULayer{T}"/>).</summary>
    Gru = 1,
}

/// <summary>
/// A bidirectional recurrent layer whose forward and backward outputs are concatenated: <c>[time, 2 · hidden]</c>, the
/// convention of PyTorch's <c>bidirectional=True</c> LSTM and GRU.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>One recurrent layer reads the sequence forwards, a second reads it reversed and its output is reversed back;
/// both outputs are concatenated per position. <see cref="BidirectionalLayer{T}"/> adds or stacks the two directions
/// instead. The input width is declared, so a restored layer sizes both directions before any forward.</para>
/// <para>Input <c>[time, input]</c> or <c>[batch, time, input]</c>; output <c>[.., time, 2 · hidden]</c>.</para>
/// <para><b>For Beginners:</b> Reads a sequence both left-to-right and right-to-left, so every position knows about what
/// came before and after it.</para>
/// </remarks>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 5, 4",
    TestConstructorArgs = "4, 3, AiDotNet.NeuralNetworks.Layers.RecurrentCellType.Lstm")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class BidirectionalRecurrentLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly RecurrentCellType _cell;

    [SubLayerInput("1, 1, _inputSize")]
    private readonly LSTMLayer<T>? _forwardLstm;
    [SubLayerInput("1, 1, _inputSize")]
    private readonly LSTMLayer<T>? _backwardLstm;
    [SubLayerInput("1, 1, _inputSize")]
    private readonly GRULayer<T>? _forwardGru;
    [SubLayerInput("1, 1, _inputSize")]
    private readonly GRULayer<T>? _backwardGru;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="inputSize">Width of each input step.</param>
    /// <param name="hiddenSize">Width of each direction's output.</param>
    /// <param name="cell">The recurrent cell.</param>
    public BidirectionalRecurrentLayer([LayerState] int inputSize, [LayerState] int hiddenSize, [LayerState] RecurrentCellType cell)
        : base(new[] { inputSize }, new[] { 2 * hiddenSize })
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _cell = cell;
        if (cell == RecurrentCellType.Lstm)
        {
            _forwardLstm = new LSTMLayer<T>(hiddenSize);
            _backwardLstm = new LSTMLayer<T>(hiddenSize);
            RegisterSubLayer(_forwardLstm);
            RegisterSubLayer(_backwardLstm);
        }
        else
        {
            _forwardGru = new GRULayer<T>(hiddenSize, returnSequences: true);
            _backwardGru = new GRULayer<T>(hiddenSize, returnSequences: true);
            RegisterSubLayer(_forwardGru);
            RegisterSubLayer(_backwardGru);
        }
    }

    /// <summary>Width of the output (both directions).</summary>
    public int OutputSize => 2 * _hiddenSize;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(2 * _hiddenSize));
        var time = new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time));
        return inputRank switch
        {
            2 => new[] { time, features },
            3 => new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), time, features },
            _ => null,
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _inputSize)
            throw new ArgumentException($"Expected [time, {_inputSize}] or [batch, time, {_inputSize}], got [{string.Join(", ", input.Shape)}].", nameof(input));
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], _inputSize }) : input;
        int batch = x.Shape[0], time = x.Shape[1];
        LayerBase<T> forwardLayer = _cell == RecurrentCellType.Lstm ? _forwardLstm! : _forwardGru!;
        LayerBase<T> backwardLayer = _cell == RecurrentCellType.Lstm ? _backwardLstm! : _backwardGru!;
        var forward = forwardLayer.Forward(x);
        var backward = Reverse(backwardLayer.Forward(Reverse(x)));
        var output = Engine.TensorConcatenate(new[] { forward, backward }, 2);
        return unbatched ? Engine.Reshape(output, new[] { time, 2 * _hiddenSize }) : output;
    }

    // Reverses the time axis of [batch, time, features] (index selection takes 2-D input, so time is moved to the front).
    private Tensor<T> Reverse(Tensor<T> x)
    {
        int batch = x.Shape[0], time = x.Shape[1], features = x.Shape[2];
        var index = new Tensor<int>(new[] { time });
        for (int t = 0; t < time; t++) index[t] = time - 1 - t;
        var timeMajor = Engine.Reshape(Engine.TensorPermute(x, new[] { 1, 0, 2 }).Contiguous(), new[] { time, batch * features });
        var reversed = Engine.Reshape(Engine.TensorIndexSelect(timeMajor, index, 0), new[] { time, batch, features });
        return Engine.TensorPermute(reversed, new[] { 1, 0, 2 }).Contiguous();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputSize"] = _inputSize.ToString(inv);
        metadata["HiddenSize"] = _hiddenSize.ToString(inv);
        metadata["Cell"] = _cell.ToString();
        return metadata;
    }
}
