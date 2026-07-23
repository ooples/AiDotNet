using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Carries the recurrent state a <see cref="MambaBlock{T}"/> needs to generate one token at a time without
/// reprocessing the prefix: the causal-conv input window and the SSM hidden state. This is the "KV cache"
/// equivalent for a selective state-space block.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class MambaStepState<T>
{
    internal MambaStepState(int batchSize, int kernelSize, int innerDimension)
    {
        // Zero-initialized window of the last `kernelSize` x-branch vectors (matches the causal conv's
        // implicit left zero-padding for the first positions).
        ConvWindow = new Tensor<T>(new[] { batchSize, kernelSize, innerDimension });
        HiddenState = null;
    }

    /// <summary>The last K x-branch vectors, oldest first: [batch, kernelSize, innerDim].</summary>
    internal Tensor<T> ConvWindow { get; set; }

    /// <summary>The SSM hidden state [batch, innerDim, stateDim], or null before the first step.</summary>
    internal Tensor<T>? HiddenState { get; set; }
}

internal partial class MambaBlock<T>
{
    /// <summary>
    /// Creates a fresh per-token decoding state for this block.
    /// </summary>
    /// <param name="batchSize">The batch size used during stepping.</param>
    internal MambaStepState<T> CreateStepState(int batchSize) =>
        new(batchSize, _convKernelSize, _innerDimension);

    /// <summary>
    /// Processes a single token using carried recurrent state, producing the same output the parallel
    /// <see cref="Forward"/> would produce at that position. Mathematically equivalent to the full-sequence
    /// path (Gu &amp; Dao 2023): it reuses the identical causal Conv1D and selective-scan ops, supplying the
    /// conv window and the carried SSM hidden state instead of recomputing the prefix.
    /// </summary>
    /// <param name="tokenInput">The single-token input, shape [batch, 1, modelDim].</param>
    /// <param name="state">The carried recurrent state, advanced in place.</param>
    /// <returns>The block output for this token, shape [batch, 1, modelDim].</returns>
    internal Tensor<T> Step(Tensor<T> tokenInput, MambaStepState<T> state)
    {
        var batchSize = tokenInput.Shape[0];

        // Step 1: Input projection -> x and z branches (identical op to Forward).
        var input2D = Engine.Reshape(tokenInput, new[] { batchSize, _modelDimension });
        var projected = Engine.TensorMatMul(input2D, _inputProjectionWeights);
        var bias2D = Engine.Reshape(_inputProjectionBias, new[] { 1, _innerDimension * 2 });
        var projectedWithBias = Engine.TensorBroadcastAdd(projected, bias2D);
        var projected3D = Engine.Reshape(projectedWithBias, new[] { batchSize, 1, _innerDimension * 2 });

        var xBranch = SliceTensor(projected3D, 2, 0, _innerDimension);                 // [batch, 1, inner]
        var zBranch = SliceTensor(projected3D, 2, _innerDimension, _innerDimension);   // [batch, 1, inner]

        // Step 2: causal Conv1D using the carried window. Shift the window left and append the new x value,
        // then run the identical conv over the window; its last position equals Forward's conv at this step.
        var xCurrent = Engine.Reshape(xBranch, new[] { batchSize, _innerDimension });
        state.ConvWindow = AdvanceConvWindow(state.ConvWindow, xCurrent, batchSize);
        var convWindowOut = DepthwiseConv1DForward(state.ConvWindow, batchSize, _convKernelSize);
        var convCurrent = convWindowOut.GetSliceAlongDimension(_convKernelSize - 1, 1).Clone(); // [batch, inner]
        var convOutput = Engine.Reshape(convCurrent, new[] { batchSize, 1, _innerDimension });

        // Step 3: SiLU.
        var siluOutput = Engine.Swish(convOutput);

        // Step 4: project to SSM parameters (delta, B, C).
        var siluFlat = Engine.Reshape(siluOutput, new[] { batchSize, _innerDimension });
        var xProj = Engine.TensorMatMul(siluFlat, _xProjectionWeights);
        var xProj3D = Engine.Reshape(xProj, new[] { batchSize, 1, _dtRank + _stateDimension * 2 });
        var deltaLowRank = SliceTensor(xProj3D, 2, 0, _dtRank);
        var bParam = SliceTensor(xProj3D, 2, _dtRank, _stateDimension);
        var cParam = SliceTensor(xProj3D, 2, _dtRank + _stateDimension, _stateDimension);

        // Step 5: delta projection + bias + softplus.
        var deltaFlat = Engine.Reshape(deltaLowRank, new[] { batchSize, _dtRank });
        var deltaProjFlat = Engine.TensorMatMul(deltaFlat, _dtProjectionWeights);
        var dtBias2D = Engine.Reshape(_dtProjectionBias, new[] { 1, _innerDimension });
        var deltaProjWithBias = Engine.TensorBroadcastAdd(deltaProjFlat, dtBias2D);
        var deltaProj3D = Engine.Reshape(deltaProjWithBias, new[] { batchSize, 1, _innerDimension });
        var delta = Engine.Softplus(deltaProj3D);

        // Step 6: one selective-scan step with the carried hidden state (identical recurrence to Forward).
        var (scanOutput, hiddenStates) = S6Scan<T>.SequentialScanForward(
            siluOutput, delta, _aLog, bParam, cParam, _dParam,
            batchSize, 1, _innerDimension, _stateDimension,
            state.HiddenState);
        state.HiddenState = hiddenStates.GetSliceAlongDimension(1, 1).Clone(); // h after this token

        // Step 7: output gating y = scan * SiLU(z).
        var zGate = Engine.Swish(zBranch);
        var gatedOutput = Engine.TensorMultiply(scanOutput, zGate);

        // Step 8: output projection.
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize, _innerDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias2D = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        var outputWithBias = Engine.TensorBroadcastAdd(outputFlat, outBias2D);
        var output3D = Engine.Reshape(outputWithBias, new[] { batchSize, 1, _modelDimension });

        // Residual + activation (identical to Forward).
        var residualOutput = Engine.TensorAdd(output3D, tokenInput);
        return ApplyActivation(residualOutput);
    }

    // Shifts the [batch, K, inner] window left by one and appends the new x value at the last slot.
    private Tensor<T> AdvanceConvWindow(Tensor<T> window, Tensor<T> newX, int batchSize)
    {
        var advanced = new Tensor<T>(new[] { batchSize, _convKernelSize, _innerDimension });
        for (int k = 1; k < _convKernelSize; k++)
        {
            // Clone the slice to guarantee contiguous memory before writing it back (non-contiguous
            // views otherwise copy stale data — the same hazard S6Scan guards against).
            advanced.SetSlice(1, k - 1, window.GetSliceAlongDimension(k, 1).Clone());
        }

        advanced.SetSlice(1, _convKernelSize - 1, newX);
        return advanced;
    }
}
