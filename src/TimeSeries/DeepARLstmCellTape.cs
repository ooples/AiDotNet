using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Validation;

namespace AiDotNet.TimeSeries;

/// <summary>
/// A tape-trainable LSTM cell over column-major <c>[features, batch]</c> activations, used by recurrent agents.
/// </summary>
[TensorLayout(TensorAxis.Features, TensorAxis.Batch, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, TensorAxis.Batch, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal partial class DeepARLstmCellTape<T> : NeuralNetworks.Layers.LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// HAND-WRITTEN because a recurrent cell replaces the feature axis rather than carrying it
    /// through: ForwardTraced returns the <c>h</c> of <c>Step</c>, and <c>Step</c> builds it as
    /// <c>oGate * tanh(cNew)</c> out of gates narrowed to <c>_hiddenSize</c> rows each - so the
    /// width is the hidden size, whatever the input width was. Only the batch column count survives,
    /// which is why the second axis is Same. The constructor states the same pair:
    /// <c>base(new[] { inputSize }, new[] { hiddenSize })</c>.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _hiddenSize <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenSize)),
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        };
    }

    private readonly int _inputSize;
    private readonly int _hiddenSize;

    // Input-to-gate and hidden-to-gate weights, stacked over the 4 gates in the order
    // (input, forget, cell, output): _wx is [4H, inputSize], _wh is [4H, H], _bias is [4H].
    //
    // NOT readonly: these are registered trainable parameters, and TrainableParameterGenerator emits a
    // SetTrainableParameters that REPLACES them with ParameterBuffer views. readonly forbade that
    // assignment (CS0191), so the buffer-backed path could never have worked for this cell. The
    // conflict stayed invisible only because the generator skips non-partial types and this class was
    // not partial until it declared a shape contract.
    private Tensor<T> _wx;
    private Tensor<T> _wh;
    private Tensor<T> _bias;

    public int InputSize => _inputSize;
    public int HiddenSize => _hiddenSize;

    public override bool SupportsTraining => true;
    public override void ResetState() { }
    /// <summary>
    /// Single-step forward from a zero initial state (satisfies the <c>ILayer</c> contract).
    /// The model drives the recurrence via <see cref="Step"/>; this convenience overload runs
    /// one timestep on <paramref name="input"/> <c>[inputSize, B]</c> and returns the hidden
    /// state <c>[H, B]</c>.
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int batch = input.Shape.Length > 1 ? input.Shape[1] : 1;
        var h0 = new Tensor<T>(new[] { _hiddenSize, batch });
        var c0 = new Tensor<T>(new[] { _hiddenSize, batch });
        var (h, _) = Step(input, h0, c0);
        return h;
    }

    public DeepARLstmCellTape(int inputSize, int hiddenSize, int seed = 42)
        : base(new[] { inputSize }, new[] { hiddenSize })
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;

        var random = RandomHelper.CreateSeededRandom(seed);
        double stddevX = Math.Sqrt(2.0 / (inputSize + hiddenSize));
        double stddevH = Math.Sqrt(2.0 / (hiddenSize + hiddenSize));

        _wx = CreateRandomTensor(new[] { 4 * hiddenSize, inputSize }, stddevX, random);
        _wh = CreateRandomTensor(new[] { 4 * hiddenSize, hiddenSize }, stddevH, random);
        _bias = new Tensor<T>(new[] { 4 * hiddenSize });

        // Forget-gate bias initialized to 1 (a standard LSTM trick that helps gradients flow
        // early in training). Gate order is (i, f, g, o), so forget occupies [H, 2H).
        for (int i = _hiddenSize; i < 2 * _hiddenSize; i++)
            _bias[i] = NumOps.One;

        RegisterTrainableParameter(_wx, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_wh, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private Tensor<T> CreateRandomTensor(int[] shape, double stddev, Random random)
    {
        var tensor = new Tensor<T>(shape);
        int total = tensor.Length;
        for (int i = 0; i < total; i++)
            tensor[i] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * stddev);
        return tensor;
    }

    /// <summary>
    /// Computes one LSTM step. Inputs are column-major: <paramref name="xt"/> is
    /// <c>[inputSize, B]</c>, <paramref name="hPrev"/> and <paramref name="cPrev"/> are
    /// <c>[H, B]</c>. Returns the new hidden and cell states, both <c>[H, B]</c>. The same
    /// method serves training (under a tape) and inference (eager) — identical ops, so there
    /// is no scalar/tape divergence.
    /// </summary>
    public (Tensor<T> h, Tensor<T> c) Step(Tensor<T> xt, Tensor<T> hPrev, Tensor<T> cPrev)
    {
        // Pre-activations for all 4 gates: [4H, B].
        var fromX = Engine.TensorMatMul(_wx, xt);      // [4H, B]
        var fromH = Engine.TensorMatMul(_wh, hPrev);   // [4H, B]
        var preact = Engine.TensorAdd(fromX, fromH);
        var biasCol = Engine.Reshape(_bias, new[] { 4 * _hiddenSize, 1 });
        preact = Engine.TensorAdd(preact, biasCol);

        // Slice the stacked pre-activations into the four gates along axis 0.
        var iPre = Engine.TensorNarrow(preact, 0, 0, _hiddenSize);
        var fPre = Engine.TensorNarrow(preact, 0, _hiddenSize, _hiddenSize);
        var gPre = Engine.TensorNarrow(preact, 0, 2 * _hiddenSize, _hiddenSize);
        var oPre = Engine.TensorNarrow(preact, 0, 3 * _hiddenSize, _hiddenSize);

        var iGate = Engine.Sigmoid(iPre);
        var fGate = Engine.Sigmoid(fPre);
        var gGate = Engine.Tanh(gPre);
        var oGate = Engine.Sigmoid(oPre);

        // c = f ⊙ c_prev + i ⊙ g ; h = o ⊙ tanh(c).
        var cNew = Engine.TensorAdd(
            Engine.TensorMultiply(fGate, cPrev),
            Engine.TensorMultiply(iGate, gGate));
        var hNew = Engine.TensorMultiply(oGate, Engine.Tanh(cNew));

        return (hNew, cNew);
    }

    private static void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor._shape)
            writer.Write(dim);
        for (int i = 0; i < tensor.Length; i++)
            writer.Write(Convert.ToDouble(tensor[i]));
    }

    private void ReadTensorInto(BinaryReader reader, Tensor<T> tensor)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int d = 0; d < rank; d++)
            shape[d] = reader.ReadInt32();
        int total = shape.Aggregate(1, (a, bb) => a * bb);
        for (int i = 0; i < total; i++)
        {
            double v = reader.ReadDouble();
            if (i < tensor.Length)
                tensor[i] = NumOps.FromDouble(v);
        }
    }
}
