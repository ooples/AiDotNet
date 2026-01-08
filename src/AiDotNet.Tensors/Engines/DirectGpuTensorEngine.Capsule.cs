using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Partial class containing GPU-resident capsule network operations.
/// </summary>
public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// GPU-resident Squash activation for capsule networks.
    /// Formula: squash(v) = ||v||^2 / (1 + ||v||^2) * v / ||v||
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="input">The GPU-resident input tensor with shape [..., capsuleDim].</param>
    /// <param name="axis">The axis over which to compute squash (default: -1 for last axis).</param>
    /// <param name="epsilon">Small value for numerical stability (default: 1e-8).</param>
    /// <returns>The GPU-resident output tensor with the same shape as input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para>
    /// The squash function is the standard non-linear activation used in capsule networks.
    /// It ensures that the output vector length is between 0 and 1, representing the probability
    /// that the entity represented by the capsule exists.
    /// </para>
    /// </remarks>
    public IGpuTensor<T> SquashGpu<T>(IGpuTensor<T> input, int axis = -1, float epsilon = 1e-8f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SquashGpu");

        int rank = input.Shape.Length;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis out of range");

        // For squash over the last dimension (most common case)
        if (axis == rank - 1)
        {
            int capsuleDim = input.Shape[rank - 1];
            int numCapsules = input.ElementCount / capsuleDim;

            var outputBuffer = backend.AllocateBuffer(input.ElementCount);
            backend.Squash(input.Buffer, outputBuffer, numCapsules, capsuleDim, epsilon);

            return new GpuTensor<T>(backend, outputBuffer, input.Shape, GpuTensorRole.Activation, ownsBuffer: true);
        }

        // For squash over a non-last axis, we need to permute the tensor
        // to move the target axis to the end, apply squash, then permute back
        var permutation = new int[rank];
        int j = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i != axis) permutation[j++] = i;
        }
        permutation[rank - 1] = axis;

        // Permute input to move axis to end
        var permutedInput = PermuteGpu(input, permutation);

        int permutedCapsuleDim = permutedInput.Shape[rank - 1];
        int permutedNumCapsules = permutedInput.ElementCount / permutedCapsuleDim;

        var permutedOutputBuffer = backend.AllocateBuffer(permutedInput.ElementCount);
        backend.Squash(permutedInput.Buffer, permutedOutputBuffer, permutedNumCapsules, permutedCapsuleDim, epsilon);

        var permutedOutput = new GpuTensor<T>(backend, permutedOutputBuffer, permutedInput.Shape, GpuTensorRole.Intermediate, ownsBuffer: true);

        // Build inverse permutation and permute back
        var inversePermutation = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            inversePermutation[permutation[i]] = i;
        }

        return PermuteGpu(permutedOutput, inversePermutation);
    }

    /// <summary>
    /// GPU-resident Squash backward operation for capsule networks.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="gradOutput">The GPU-resident gradient from the next layer.</param>
    /// <param name="input">The GPU-resident input tensor from the forward pass.</param>
    /// <param name="axis">The axis over which squash was computed (default: -1 for last axis).</param>
    /// <param name="epsilon">Small value for numerical stability (default: 1e-8).</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public IGpuTensor<T> SquashBackwardGpu<T>(
        IGpuTensor<T> gradOutput,
        IGpuTensor<T> input,
        int axis = -1,
        float epsilon = 1e-8f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SquashBackwardGpu");

        int rank = input.Shape.Length;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis out of range");

        // For squash backward over the last dimension
        if (axis == rank - 1)
        {
            int capsuleDim = input.Shape[rank - 1];
            int numCapsules = input.ElementCount / capsuleDim;

            var gradInputBuffer = backend.AllocateBuffer(input.ElementCount);
            backend.SquashBackward(gradOutput.Buffer, input.Buffer, gradInputBuffer, numCapsules, capsuleDim, epsilon);

            return new GpuTensor<T>(backend, gradInputBuffer, input.Shape, GpuTensorRole.Gradient, ownsBuffer: true);
        }

        // For squash backward over a non-last axis, permute to move axis to end
        var permutation = new int[rank];
        int j = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i != axis) permutation[j++] = i;
        }
        permutation[rank - 1] = axis;

        var permutedGradOutput = PermuteGpu(gradOutput, permutation);
        var permutedInput = PermuteGpu(input, permutation);

        int permutedCapsuleDim = permutedInput.Shape[rank - 1];
        int permutedNumCapsules = permutedInput.ElementCount / permutedCapsuleDim;

        var permutedGradInputBuffer = backend.AllocateBuffer(permutedInput.ElementCount);
        backend.SquashBackward(permutedGradOutput.Buffer, permutedInput.Buffer, permutedGradInputBuffer,
            permutedNumCapsules, permutedCapsuleDim, epsilon);

        var permutedGradInput = new GpuTensor<T>(backend, permutedGradInputBuffer, permutedInput.Shape,
            GpuTensorRole.Intermediate, ownsBuffer: true);

        // Build inverse permutation and permute back
        var inversePermutation = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            inversePermutation[permutation[i]] = i;
        }

        return PermuteGpu(permutedGradInput, inversePermutation);
    }

    /// <summary>
    /// GPU-resident dynamic routing between capsules.
    /// This performs the iterative routing-by-agreement algorithm.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="predictions">Prediction vectors from lower capsules [batch, inputCapsules, outputCapsules, capsuleDim].</param>
    /// <param name="numIterations">Number of routing iterations (typically 3).</param>
    /// <param name="epsilon">Small value for numerical stability.</param>
    /// <returns>A tuple of (output capsules, coupling coefficients).</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    public (IGpuTensor<T> Output, IGpuTensor<T> Couplings) DynamicRoutingGpu<T>(
        IGpuTensor<T> predictions,
        int numIterations = 3,
        float epsilon = 1e-8f)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for DynamicRoutingGpu");

        // predictions shape: [batch, inputCapsules, outputCapsules, capsuleDim]
        if (predictions.Shape.Length != 4)
            throw new ArgumentException("Predictions must be 4D [batch, inputCapsules, outputCapsules, capsuleDim]");

        int batchSize = predictions.Shape[0];
        int inputCapsules = predictions.Shape[1];
        int outputCapsules = predictions.Shape[2];
        int capsuleDim = predictions.Shape[3];

        // Initialize coupling logits (b_ij) to zero: [batch, inputCapsules, outputCapsules]
        int couplingsSize = batchSize * inputCapsules * outputCapsules;
        var couplingsBuffer = backend.AllocateBuffer(couplingsSize);
        backend.Fill(couplingsBuffer, 0.0f, couplingsSize);
        IGpuTensor<T> couplings = new GpuTensor<T>(backend, couplingsBuffer,
            [batchSize, inputCapsules, outputCapsules], GpuTensorRole.Intermediate, ownsBuffer: true);

        IGpuTensor<T>? output = null;

        for (int iter = 0; iter < numIterations; iter++)
        {
            // Step 1: softmax over output capsules to get routing weights c_ij
            // couplings: [B, I, C] -> routingWeights: [B, I, C]
            var routingWeights = SoftmaxAxisGpu(couplings, axis: 2);

            // Step 2: weighted sum of predictions
            // routingWeights: [B, I, C] -> expand to [B, I, C, 1]
            // predictions: [B, I, C, D]
            // weightedPredictions = routingWeights * predictions -> [B, I, C, D]
            var routingExpanded = ReshapeGpu(routingWeights, [batchSize, inputCapsules, outputCapsules, 1]);
            var weightedPred = MultiplyGpu(predictions, routingExpanded);

            // Step 3: sum over input capsules to get s_j: [B, C, D]
            var summed = SumAxisGpu(weightedPred, axis: 1);

            // Step 4: squash to get output capsules v_j
            output = SquashGpu(summed, axis: -1, epsilon);

            // Step 5: update couplings (except on last iteration)
            if (iter < numIterations - 1)
            {
                // agreement = predictions dot output
                // predictions: [B, I, C, D], output: [B, C, D] -> expand to [B, 1, C, D]
                var outputExpanded = ReshapeGpu(output, [batchSize, 1, outputCapsules, capsuleDim]);
                var agreement = MultiplyGpu(predictions, outputExpanded);
                var agreementSum = SumAxisGpu(agreement, axis: 3); // [B, I, C]

                // couplings += agreement
                couplings = AddGpu(couplings, agreementSum);
            }

            // Dispose intermediate tensors to free GPU memory
            routingWeights.Dispose();
            routingExpanded.Dispose();
            weightedPred.Dispose();
            summed.Dispose();
        }

        return (output!, couplings);
    }

    /// <summary>
    /// GPU-resident softmax over a specific axis.
    /// </summary>
    private IGpuTensor<T> SoftmaxAxisGpu<T>(IGpuTensor<T> input, int axis)
    {
        if (!TryGetBackend(out var backend))
            throw new InvalidOperationException("No GPU backend available for SoftmaxAxisGpu");

        int rank = input.Shape.Length;
        if (axis < 0) axis = rank + axis;

        // If softmax is over the last axis, use the standard softmax
        if (axis == rank - 1)
        {
            return SoftmaxGpu<T>(input);
        }

        // Otherwise, permute to move axis to end, apply softmax, permute back
        var permutation = new int[rank];
        int j = 0;
        for (int i = 0; i < rank; i++)
        {
            if (i != axis) permutation[j++] = i;
        }
        permutation[rank - 1] = axis;

        var permuted = PermuteGpu(input, permutation);
        var softmaxed = SoftmaxGpu<T>(permuted);

        // Build inverse permutation
        var inversePermutation = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            inversePermutation[permutation[i]] = i;
        }

        var result = PermuteGpu(softmaxed, inversePermutation);
        permuted.Dispose();
        softmaxed.Dispose();

        return result;
    }

}
