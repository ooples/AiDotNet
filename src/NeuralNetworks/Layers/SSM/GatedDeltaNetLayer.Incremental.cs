using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines.Autodiff;

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
        History = kernelSize > 1 ? new Tensor<T>(new[] { kernelSize - 1, batchSize, modelDimension }) : null;
        Memory = new Tensor<T>(new[] { batchSize * numHeads, headDimension, headDimension });
        ModelDimension = modelDimension;
    }

    /// <summary>The number of sequences this state decodes.</summary>
    public int BatchSize { get; }

    /// <summary>The number of positions consumed so far.</summary>
    public int Position { get; internal set; }

    internal int ModelDimension { get; }

    // [K - 1, B, D]: History[m - 1] = the input m positions back, zero before the sequence start (the forward's zero
    // padding); null when the kernel size is 1. Shifted in place by IEngine.CausalConv1dStep, so a compiled decode plan
    // keeps it in one persistent buffer.
    internal Tensor<T>? History { get; }

    // [B·H, headDim (value), headDim (key)], the scan state S of GatedDeltaNetScanForward, updated in place by
    // IEngine.GatedDeltaNetStep.
    internal Tensor<T> Memory { get; }
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
        // One position per sequence: [B, 1, D] or [B, D].
        bool onePositionRank3 = input.Shape.Length == 3 && input.Shape[1] == 1 && input.Shape[2] == d;
        bool onePositionRank2 = input.Shape.Length == 2 && input.Shape[1] == d;
        if (input.Shape[0] != b || !(onePositionRank3 || onePositionRank2))
            throw new ArgumentException(
                $"Step expects one position per sequence, [{b}, 1, {d}] or [{b}, {d}]; got [{string.Join(",", input.Shape)}].", nameof(input));
        if (state.ModelDimension != d || (state.History?.Shape[0] ?? 0) != _convKernelSize - 1 || state.Memory.Shape[0] != rows || state.Memory.Shape[1] != hd)
            throw new ArgumentException("The state was created by a layer of a different shape.", nameof(state));

        var x = Engine.Reshape(input, new[] { b, d });

        // Causal depthwise conv + SiLU in one step: w[:, 0] weights this position, w[:, m] the input m positions back;
        // the carried inputs shift in place.
        var silu = state.History is { } history
            ? Engine.CausalConv1dStep(history, x, _convWeights, _convBias, silu: true)
            : Engine.Swish(Engine.TensorAdd(Engine.TensorMultiply(x, Engine.Reshape(_convWeights, new[] { 1, d })),
                Engine.Reshape(_convBias, new[] { 1, d })));

        var q = Engine.Reshape(Engine.TensorMatMul(silu, _queryWeights), new[] { rows, hd });
        var k = Engine.TensorMultiplyScalar(Engine.Reshape(Engine.TensorMatMul(silu, _keyWeights), new[] { rows, hd }),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var v = Engine.Reshape(Engine.TensorMatMul(silu, _valueWeights), new[] { rows, hd });
        var beta = Engine.Reshape(Engine.Sigmoid(Engine.TensorAdd(Engine.TensorMatMul(silu, _betaWeights),
            Engine.Reshape(_betaBias, new[] { 1, h }))), new[] { rows });
        var alpha = Engine.Reshape(Engine.Sigmoid(Engine.TensorAdd(Engine.TensorMatMul(silu, _alphaWeights),
            Engine.Reshape(_alphaBias, new[] { 1, h }))), new[] { rows });
        var gate = Engine.Swish(Engine.TensorAdd(Engine.TensorMatMul(silu, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, d })));

        // Delta rule, all heads at once, the memory updated in place: S k' and the correction read the memory BEFORE
        // this position's decay; o = S_new q.
        var o = Engine.Reshape(Engine.GatedDeltaNetStep(state.Memory, q, k, v, alpha, beta), new[] { b, d });

        var projected = Engine.TensorAdd(Engine.TensorMatMul(Engine.TensorMultiply(o, gate), _outputProjectionWeights),
            Engine.Reshape(_outputProjectionBias, new[] { 1, d }));
        var result = StepActivation(projected);
        state.Position++;
        return Engine.Reshape(result, input._shape);
    }

    // A step is never followed by the eager backward that drains the layer's pre-activation cache, and a new layer is in
    // training mode by default, so a decode loop would keep one projected tensor per token for the layer's life.
    // Outside a tape the activation therefore runs in a no-grad scope, which does not retain it. Under a tape nothing
    // is retained either way, and the step stays differentiable.
    private Tensor<T> StepActivation(Tensor<T> projected)
    {
        if (GradientTape<T>.Current is not null) return ApplyActivation(projected);
        using var noGrad = new NoGradScope<T>();
        return ApplyActivation(projected);
    }
}
