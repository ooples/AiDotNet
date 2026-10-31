using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// One step of a gated recurrent unit: <c>h' = GRU(x, h)</c>, for recurrent decoders that run a step at a time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The gated recurrent unit of Cho et al. (2014) and Chung et al. (2014), with the reset gate applied to the state
/// before its projection (as TensorFlow's <c>GRUCell</c>, which Tacotron's implementations use):
/// </para>
/// <list type="bullet">
/// <item><c>r = σ(W_r x + U_r h + b_r)</c>, <c>z = σ(W_z x + U_z h + b_z)</c></item>
/// <item><c>n = tanh(W_n x + U_n (r ⊙ h) + b_n)</c></item>
/// <item><c>h' = z ⊙ h + (1 − z) ⊙ n</c></item>
/// </list>
/// <para>Inputs: <c>x</c> <c>[batch, input]</c> and <c>h</c> <c>[batch, hidden]</c>; output <c>[batch, hidden]</c>.
/// <see cref="GRULayer{T}"/> runs a whole sequence; an autoregressive decoder (Tacotron's attention and decoder RNNs)
/// needs each step's state between steps, which this cell exposes.</para>
/// <para><b>For Beginners:</b> A GRU keeps a running memory. Each step it decides how much of the old memory to keep
/// and how much to overwrite with something computed from the new input.</para>
/// </remarks>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ApiShape = LayerApiShape.DualTensor, TestInputShape = "1, 4", TestConstructorArgs = "4, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GRUCellLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;

    [SubLayerInput("1, _inputSize")]
    private readonly DenseLayer<T> _input;
    [SubLayerInput("1, _hiddenSize")]
    private readonly BiasFreeLinearLayer<T> _recurrentGates;
    [SubLayerInput("1, _hiddenSize")]
    private readonly BiasFreeLinearLayer<T> _recurrentCandidate;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool RequiresMultipleInputs => true;

    /// <summary>Creates a GRU cell.</summary>
    /// <param name="inputSize">Width of the step input.</param>
    /// <param name="hiddenSize">Width of the state.</param>
    public GRUCellLayer([LayerState] int inputSize, [LayerState] int hiddenSize)
        : base(new[] { inputSize }, new[] { hiddenSize })
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _input = new DenseLayer<T>(3 * hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _recurrentGates = new BiasFreeLinearLayer<T>(hiddenSize, 2 * hiddenSize);
        _recurrentCandidate = new BiasFreeLinearLayer<T>(hiddenSize, hiddenSize);
        RegisterSubLayer(_input);
        RegisterSubLayer(_recurrentGates);
        RegisterSubLayer(_recurrentCandidate);
    }

    /// <summary>Width of the state.</summary>
    public int HiddenSize => _hiddenSize;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 2
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenSize)),
            }
            : null;

    /// <summary>One step: the next state from the step input and the current state.</summary>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> state) => Forward(new[] { input, state });

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => throw new InvalidOperationException($"{nameof(GRUCellLayer<T>)} needs the current state; call Forward(input, state).");

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length != 2)
            throw new ArgumentException("Expected the step input and the current state.", nameof(inputs));
        var x = inputs[0];
        var h = inputs[1];
        if (x.Rank != 2 || x.Shape[1] != _inputSize)
            throw new ArgumentException($"Expected input [batch, {_inputSize}], got [{string.Join(", ", x.Shape)}].", nameof(inputs));
        if (h.Rank != 2 || h.Shape[1] != _hiddenSize || h.Shape[0] != x.Shape[0])
            throw new ArgumentException($"Expected state [{x.Shape[0]}, {_hiddenSize}], got [{string.Join(", ", h.Shape)}].", nameof(inputs));
        int batch = x.Shape[0], n = _hiddenSize;

        var projected = _input.Forward(x);              // [B, 3H]: r | z | n
        var recurrent = _recurrentGates.Forward(h);     // [B, 2H]: r | z
        var r = Engine.Sigmoid(Engine.TensorAdd(
            Engine.TensorSlice(projected, new[] { 0, 0 }, new[] { batch, n }),
            Engine.TensorSlice(recurrent, new[] { 0, 0 }, new[] { batch, n })));
        var z = Engine.Sigmoid(Engine.TensorAdd(
            Engine.TensorSlice(projected, new[] { 0, n }, new[] { batch, n }),
            Engine.TensorSlice(recurrent, new[] { 0, n }, new[] { batch, n })));
        var candidate = Engine.Tanh(Engine.TensorAdd(
            Engine.TensorSlice(projected, new[] { 0, 2 * n }, new[] { batch, n }),
            _recurrentCandidate.Forward(Engine.TensorMultiply(r, h))));

        // h' = z * h + (1 - z) * n = n + z * (h - n)
        return Engine.TensorAdd(candidate, Engine.TensorMultiply(z, Engine.TensorSubtract(h, candidate)));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments so deserialization can rebuild the sublayers before loading weights.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputSize"] = _inputSize.ToString(inv);
        metadata["HiddenSize"] = _hiddenSize.ToString(inv);
        return metadata;
    }
}
