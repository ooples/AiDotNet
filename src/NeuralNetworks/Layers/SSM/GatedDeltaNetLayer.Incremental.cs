using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// The per-sequence decoding state of a <see cref="GatedDeltaNetLayer{T}"/>: the last <c>convKernelSize - 1</c> inputs
/// the causal depthwise convolution reads, and the per-head delta-rule memory. Create it with
/// <see cref="GatedDeltaNetLayer{T}.CreateStepState"/>; one state belongs to one layer and one batch of sequences.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class GatedDeltaNetStepState<T>
{
    internal GatedDeltaNetStepState(int batchSize, int kernelSize, int numHeads, int headDimension, int modelDimension)
    {
        BatchSize = batchSize;
        History = new Tensor<T>?[kernelSize - 1];
        Memory = new Tensor<T>(new[] { batchSize * numHeads, headDimension, headDimension });
        ModelDimension = modelDimension;
    }

    /// <summary>The number of sequences this state decodes.</summary>
    public int BatchSize { get; }

    /// <summary>The number of positions consumed so far.</summary>
    public int Position { get; internal set; }

    internal int ModelDimension { get; }

    // History[m - 1] = the input m positions back, [B, D]; null before the sequence start (the forward's zero padding).
    internal Tensor<T>?[] History { get; }

    // [B·H, headDim (value), headDim (key)], the scan state S of GatedDeltaNetScanForward.
    internal Tensor<T> Memory { get; set; }
}

public partial class GatedDeltaNetLayer<T>
{
    /// <summary>
    /// Creates an empty decoding state for <paramref name="batchSize"/> sequences, for use with <see cref="Step"/>.
    /// </summary>
    /// <param name="batchSize">The number of sequences decoded together.</param>
    /// <returns>A zero state, equivalent to the start of a sequence.</returns>
    public GatedDeltaNetStepState<T> CreateStepState(int batchSize = 1)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        return new GatedDeltaNetStepState<T>(batchSize, _convKernelSize, _numHeads, _headDimension, _modelDimension);
    }

    /// <summary>
    /// Runs the layer on ONE new position per sequence and advances <paramref name="state"/>, giving the same output
    /// the full causal <c>Forward</c> produces at that position, in O(1) time and memory per token.
    /// </summary>
    /// <param name="input">This position's input, <c>[batch, 1, modelDim]</c> or <c>[batch, modelDim]</c>.</param>
    /// <param name="state">The state from <see cref="CreateStepState"/>, advanced in place.</param>
    /// <returns>The output at this position, in the input's shape.</returns>
    /// <remarks>
    /// <para>The same math as the forward, one position at a time: the causal depthwise convolution reads the carried
    /// inputs, and the delta rule of <c>GatedDeltaNetScanForward</c> updates the memory S per head:
    /// <c>k' = k/√d</c>, <c>S ← α·S + β·(v − S k') k'ᵀ</c>, <c>o = S q</c>.</para>
    /// <para><b>For Beginners:</b> Generating text one token at a time should not re-read the whole text each time.
    /// This layer's memory is a fixed-size matrix per head, so each new token only updates that matrix.</para>
    /// </remarks>
    public Tensor<T> Step(Tensor<T> input, GatedDeltaNetStepState<T> state)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (state is null) throw new ArgumentNullException(nameof(state));
        int b = state.BatchSize, d = _modelDimension, h = _numHeads, hd = _headDimension, rows = b * h;
        bool rank3 = input.Shape.Length == 3;
        if ((rank3 && (input.Shape[0] != b || input.Shape[1] != 1 || input.Shape[2] != d))
            || (!rank3 && (input.Shape.Length != 2 || input.Shape[0] != b || input.Shape[1] != d)))
            throw new ArgumentException(
                $"Step expects one position per sequence, [{b}, 1, {d}] or [{b}, {d}]; got [{string.Join(",", input.Shape)}].", nameof(input));
        if (state.ModelDimension != d || state.History.Length != _convKernelSize - 1 || state.Memory.Shape[0] != rows || state.Memory.Shape[1] != hd)
            throw new ArgumentException("The state was created by a layer of a different shape.", nameof(state));

        var x = Engine.Reshape(input, new[] { b, d });

        // Causal depthwise conv: w[:, 0] weights this position, w[:, m] the input m positions back.
        var conv = Engine.TensorMultiply(x, Engine.Reshape(Engine.TensorSlice(_convWeights, new[] { 0, 0 }, new[] { d, 1 }), new[] { 1, d }));
        for (int m = 1; m < _convKernelSize; m++)
        {
            if (state.History[m - 1] is not { } past) continue;
            var tap = Engine.Reshape(Engine.TensorSlice(_convWeights, new[] { 0, m }, new[] { d, 1 }), new[] { 1, d });
            conv = Engine.TensorAdd(conv, Engine.TensorMultiply(past, tap));
        }
        conv = Engine.TensorAdd(conv, Engine.Reshape(_convBias, new[] { 1, d }));
        for (int m = state.History.Length - 1; m > 0; m--) state.History[m] = state.History[m - 1];
        if (state.History.Length > 0) state.History[0] = x;

        var silu = Engine.Swish(conv);
        var q = Engine.Reshape(Engine.TensorMatMul(silu, _queryWeights), new[] { rows, hd, 1 });
        var k = Engine.TensorMultiplyScalar(Engine.Reshape(Engine.TensorMatMul(silu, _keyWeights), new[] { rows, 1, hd }),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var v = Engine.Reshape(Engine.TensorMatMul(silu, _valueWeights), new[] { rows, hd, 1 });
        var beta = Engine.Reshape(Engine.Sigmoid(Engine.TensorAdd(Engine.TensorMatMul(silu, _betaWeights),
            Engine.Reshape(_betaBias, new[] { 1, h }))), new[] { rows, 1, 1 });
        var alpha = Engine.Reshape(Engine.Sigmoid(Engine.TensorAdd(Engine.TensorMatMul(silu, _alphaWeights),
            Engine.Reshape(_alphaBias, new[] { 1, h }))), new[] { rows, 1, 1 });
        var gate = Engine.Swish(Engine.TensorAdd(Engine.TensorMatMul(silu, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, d })));

        // Delta rule, all heads at once: S k' and the correction read the memory BEFORE this position's decay.
        var predicted = Engine.BatchMatMul(state.Memory, Engine.Reshape(k, new[] { rows, hd, 1 }));
        var delta = Engine.TensorMultiply(Engine.TensorSubtract(v, predicted), beta);
        state.Memory = Engine.TensorAdd(Engine.TensorMultiply(state.Memory, alpha), Engine.BatchMatMul(delta, k));
        var o = Engine.Reshape(Engine.BatchMatMul(state.Memory, q), new[] { b, d });

        var projected = Engine.TensorAdd(Engine.TensorMatMul(Engine.TensorMultiply(o, gate), _outputProjectionWeights),
            Engine.Reshape(_outputProjectionBias, new[] { 1, d }));
        var result = ApplyActivation(projected);
        state.Position++;
        return Engine.Reshape(result, input._shape);
    }
}
