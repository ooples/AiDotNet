using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Tacotron's CBHG module: a 1-D Convolution Bank, Highway network and bidirectional GRU.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Tacotron (Wang et al. 2017, §3.1, Fig. 2, Table 1): the input is convolved with <c>K</c> sets of 1-D filters of widths
/// 1..K, the outputs are stacked and max-pooled along time (stride 1, width 2), passed through two fixed-width 1-D
/// projection convolutions, added to the input (residual), fed to a 4-layer highway network, and read by a
/// bidirectional GRU whose forward and backward outputs are concatenated. Batch normalization is used for every
/// convolution.
/// </para>
/// <para>
/// What the paper leaves unstated follows the reference implementation (keithito/tacotron, <c>models/modules.py</c>):
/// convolutions use "same" padding (for an even width, ⌊(k−1)/2⌋ frames in front and the rest behind), apply their
/// activation and then batch normalization; the max pooling pads at the end; when the residual sum's width differs
/// from the highway width (the post-processing CBHG reads 80 mel bins into 128-wide highways) a linear layer maps it
/// first; highway gates start biased to −1.
/// </para>
/// <para>Input <c>[time, channels]</c> or <c>[batch, time, channels]</c>; output <c>[.., time, 2 · gruUnits]</c>.</para>
/// <para><b>For Beginners:</b> The convolution bank looks at the sequence through windows of many sizes at once, like
/// reading a sentence by its letters, syllables and words together; the GRUs then read the result in both
/// directions.</para>
/// </remarks>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, ChangesShape = true, TestInputShape = "1, 6, 8",
    TestConstructorArgs = "8, 4, 8, new[] { 8, 8 }, 8, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class CbhgLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputChannels;
    private readonly int _bankSize;
    private readonly int _bankChannels;
    private readonly int[] _projections;
    private readonly int _highwayWidth;
    private readonly int _gruUnits;

    [SubLayerInput("1, _inputChannels, 1")]
    private readonly List<Conv1DLayer<T>> _bank = new();
    [SubLayerInput("_bankChannels")]
    private readonly List<BatchNormalizationLayer<T>> _bankNorms = new();
    private readonly Conv1DLayer<T> _projection1;
    private readonly BatchNormalizationLayer<T> _projection1Norm;
    private readonly Conv1DLayer<T> _projection2;
    private readonly BatchNormalizationLayer<T> _projection2Norm;
    private readonly DenseLayer<T>? _highwayInput;
    [SubLayerInput("_highwayWidth")]
    private readonly List<HighwayLayer<T>> _highways = new();
    private readonly GRULayer<T> _forwardGru;
    private readonly GRULayer<T> _backwardGru;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates a CBHG module.</summary>
    /// <param name="inputChannels">Channels of the input sequence.</param>
    /// <param name="bankSize">Number of bank filter widths, 1..K (16 encoder, 8 post-processing).</param>
    /// <param name="bankChannels">Channels per bank convolution (128).</param>
    /// <param name="projections">Channels of the two projection convolutions; the second must equal
    /// <paramref name="inputChannels"/> for the residual (encoder 128, 128; post-processing 256, 80).</param>
    /// <param name="highwayWidth">Width of the 4 highway layers (128).</param>
    /// <param name="gruUnits">Units of each GRU direction (128).</param>
    public CbhgLayer(
        [LayerState] int inputChannels,
        [LayerState] int bankSize,
        [LayerState] int bankChannels,
        [LayerState] int[] projections,
        [LayerState] int highwayWidth,
        [LayerState] int gruUnits)
        : base(new[] { inputChannels }, new[] { 2 * gruUnits })
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (bankSize <= 0) throw new ArgumentOutOfRangeException(nameof(bankSize));
        if (bankChannels <= 0) throw new ArgumentOutOfRangeException(nameof(bankChannels));
        if (projections is null || projections.Length != 2 || projections[0] <= 0)
            throw new ArgumentException("Expected two projection widths.", nameof(projections));
        if (projections[1] != inputChannels)
            throw new ArgumentException(
                $"The second projection ({projections[1]}) must match the input channels ({inputChannels}) for the residual connection.",
                nameof(projections));
        if (highwayWidth <= 0) throw new ArgumentOutOfRangeException(nameof(highwayWidth));
        if (gruUnits <= 0) throw new ArgumentOutOfRangeException(nameof(gruUnits));

        _inputChannels = inputChannels;
        _bankSize = bankSize;
        _bankChannels = bankChannels;
        _projections = (int[])projections.Clone();
        _highwayWidth = highwayWidth;
        _gruUnits = gruUnits;

        for (int k = 1; k <= bankSize; k++)
        {
            var conv = new Conv1DLayer<T>(inputChannels: inputChannels, outputChannels: bankChannels, kernelSize: k,
                padding: 0, activation: new ReLUActivation<T>());
            var norm = new BatchNormalizationLayer<T>(bankChannels, epsilon: 1e-3, momentum: 0.99);
            _bank.Add(conv);
            _bankNorms.Add(norm);
            RegisterSubLayer(conv);
            RegisterSubLayer(norm);
        }
        _projection1 = new Conv1DLayer<T>(inputChannels: bankSize * bankChannels, outputChannels: projections[0],
            kernelSize: 3, padding: 0, activation: new ReLUActivation<T>());
        _projection1Norm = new BatchNormalizationLayer<T>(projections[0], epsilon: 1e-3, momentum: 0.99);
        _projection2 = new Conv1DLayer<T>(inputChannels: projections[0], outputChannels: projections[1], kernelSize: 3,
            padding: 0);
        _projection2Norm = new BatchNormalizationLayer<T>(projections[1], epsilon: 1e-3, momentum: 0.99);
        RegisterSubLayer(_projection1);
        RegisterSubLayer(_projection1Norm);
        RegisterSubLayer(_projection2);
        RegisterSubLayer(_projection2Norm);
        if (inputChannels != highwayWidth)
        {
            _highwayInput = new DenseLayer<T>(highwayWidth, new IdentityActivation<T>() as IActivationFunction<T>);
            RegisterSubLayer(_highwayInput);
        }
        for (int i = 0; i < 4; i++)
        {
            var highway = new HighwayLayer<T>(highwayWidth, new ReLUActivation<T>() as IActivationFunction<T>,
                new SigmoidActivation<T>() as IActivationFunction<T>);
            _highways.Add(highway);
            RegisterSubLayer(highway);
        }
        _forwardGru = new GRULayer<T>(gruUnits, returnSequences: true);
        _backwardGru = new GRULayer<T>(gruUnits, returnSequences: true);
        RegisterSubLayer(_forwardGru);
        RegisterSubLayer(_backwardGru);
    }

    /// <summary>Width of the output (both GRU directions).</summary>
    public int OutputChannels => 2 * _gruUnits;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(2 * _gruUnits));
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
        if (input.Rank is not (2 or 3) || input.Shape[input.Rank - 1] != _inputChannels)
            throw new ArgumentException(
                $"Expected [time, {_inputChannels}] or [batch, time, {_inputChannels}], got [{string.Join(", ", input.Shape)}].",
                nameof(input));
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;

        var bank = new Tensor<T>[_bankSize];
        for (int k = 0; k < _bankSize; k++)
            bank[k] = Normalize(_bankNorms[k], SameConv(_bank[k], x, k + 1));
        var stacked = _bankSize == 1 ? bank[0] : Engine.TensorConcatenate(bank, 2);

        var pooled = MaxPoolWidth2(stacked);
        var projected = Normalize(_projection1Norm, SameConv(_projection1, pooled, 3));
        projected = Normalize(_projection2Norm, SameConv(_projection2, projected, 3));

        var highway = Engine.TensorAdd(projected, x);
        int batch = highway.Shape[0], time = highway.Shape[1];
        var rows = Engine.Reshape(highway, new[] { batch * time, highway.Shape[2] });
        if (_highwayInput is not null) rows = _highwayInput.Forward(rows);
        foreach (var layer in _highways) rows = layer.Forward(rows);
        var sequence = Engine.Reshape(rows, new[] { batch, time, _highwayWidth });

        var forward = _forwardGru.Forward(sequence);
        var backward = Reverse(_backwardGru.Forward(Reverse(sequence)));
        var output = Engine.TensorConcatenate(new[] { forward, backward }, 2);
        return unbatched ? Engine.Reshape(output, new[] { time, 2 * _gruUnits }) : output;
    }

    // TF "same" convolution: pad floor((k-1)/2) frames in front and the rest behind, then an unpadded convolution.
    private Tensor<T> SameConv(Conv1DLayer<T> conv, Tensor<T> x, int kernel)
    {
        int batch = x.Shape[0], channels = x.Shape[2];
        int front = (kernel - 1) / 2, back = kernel - 1 - front;
        var parts = new List<Tensor<T>>(3);
        if (front > 0) parts.Add(new Tensor<T>(new[] { batch, front, channels }));
        parts.Add(x);
        if (back > 0) parts.Add(new Tensor<T>(new[] { batch, back, channels }));
        var padded = parts.Count == 1 ? x : Engine.TensorConcatenate(parts.ToArray(), 1);
        var channelsFirst = Engine.TensorPermute(padded, new[] { 0, 2, 1 }).Contiguous();
        return Engine.TensorPermute(conv.Forward(channelsFirst), new[] { 0, 2, 1 }).Contiguous();
    }

    // Batch normalization of each channel over batch and time.
    private Tensor<T> Normalize(BatchNormalizationLayer<T> norm, Tensor<T> x)
    {
        int batch = x.Shape[0], time = x.Shape[1], channels = x.Shape[2];
        return Engine.Reshape(norm.Forward(Engine.Reshape(x, new[] { batch * time, channels })), new[] { batch, time, channels });
    }

    // Max pooling, width 2, stride 1, TF "same": y[t] = max(x[t], x[t + 1]), the last frame pooled with itself.
    private Tensor<T> MaxPoolWidth2(Tensor<T> x)
    {
        int batch = x.Shape[0], time = x.Shape[1], channels = x.Shape[2];
        if (time == 1) return x;
        var next = Engine.TensorConcatenate(new[]
        {
            Engine.TensorSlice(x, new[] { 0, 1, 0 }, new[] { batch, time - 1, channels }),
            Engine.TensorSlice(x, new[] { 0, time - 1, 0 }, new[] { batch, 1, channels }),
        }, 1);
        // max(a, b) = (a + b + |a - b|) / 2
        var sum = Engine.TensorAdd(x, next);
        var gap = Engine.TensorAbs(Engine.TensorSubtract(x, next));
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(sum, gap), NumOps.FromDouble(0.5));
    }

    // Reverses the time axis of [batch, time, channels]. Index selection takes a 2-D tensor, so time is moved to the
    // front and the other axes flattened around the selection.
    private Tensor<T> Reverse(Tensor<T> x)
    {
        int batch = x.Shape[0], time = x.Shape[1], channels = x.Shape[2];
        var index = new Tensor<int>(new[] { time });
        for (int t = 0; t < time; t++) index[t] = time - 1 - t;
        var timeMajor = Engine.Reshape(Engine.TensorPermute(x, new[] { 1, 0, 2 }).Contiguous(), new[] { time, batch * channels });
        var reversed = Engine.Reshape(Engine.TensorIndexSelect(timeMajor, index, 0), new[] { time, batch, channels });
        return Engine.TensorPermute(reversed, new[] { 1, 0, 2 }).Contiguous();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments so deserialization can rebuild the sublayers before loading weights.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputChannels"] = _inputChannels.ToString(inv);
        metadata["BankSize"] = _bankSize.ToString(inv);
        metadata["BankChannels"] = _bankChannels.ToString(inv);
        metadata["Projections"] = string.Join(",", _projections);
        metadata["HighwayWidth"] = _highwayWidth.ToString(inv);
        metadata["GruUnits"] = _gruUnits.ToString(inv);
        return metadata;
    }
}
