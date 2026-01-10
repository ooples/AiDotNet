using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a fully connected layer using octonion-valued weights and inputs.
/// </summary>
/// <remarks>
/// <para>
/// An octonion linear layer performs matrix-vector multiplication in the 8-dimensional
/// octonion algebra. Each weight and input is an octonion (8 real components), enabling
/// the layer to capture more complex relationships than real-valued layers.
/// </para>
/// <para><b>For Beginners:</b> This layer is like a regular dense layer, but it uses
/// 8-dimensional numbers (octonions) instead of regular numbers.
///
/// Benefits of octonion layers:
/// - Can model more complex relationships with fewer parameters
/// - Useful for certain types of image and signal processing
/// - Better at capturing rotational relationships in data
///
/// The tradeoff is that computations are more expensive per parameter.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (float or double).</typeparam>
public class OctonionLinearLayer<T> : LayerBase<T>
{
    private readonly IAdvancedAlgebraEngine _engine;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The octonion weight matrix connecting input to output neurons.
    /// </summary>
    private Octonion<T>[,] _weights;

    /// <summary>
    /// The octonion bias values for each output neuron.
    /// </summary>
    private Octonion<T>[] _biases;

    /// <summary>
    /// Stored input from forward pass for backpropagation.
    /// </summary>
    private Octonion<T>[,]? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stored pre-activation output for gradient computation.
    /// </summary>
    private Octonion<T>[,]? _lastOutput;

    /// <summary>
    /// Gradient for weights, stored during backward pass.
    /// </summary>
    private Octonion<T>[,]? _weightsGradient;

    /// <summary>
    /// Gradient for biases, stored during backward pass.
    /// </summary>
    private Octonion<T>[]? _biasesGradient;

    // GPU caching fields for backward pass
    private IGpuTensor<T>? _gpuInput;
    private int[]? _gpuInputShape;

    #region GPU Weight Storage Fields

    // GPU weight tensors for GPU-resident training (octonions stored as flat arrays)
    private GpuTensor<T>? _gpuWeights;
    private GpuTensor<T>? _gpuBiases;

    // GPU gradient tensors from BackwardGpu
    private GpuTensor<T>? _gpuWeightGradient;
    private GpuTensor<T>? _gpuBiasGradient;

    // Optimizer state tensors for SGD/NAG/LARS (velocity)
    private GpuTensor<T>? _gpuWeightVelocity;
    private GpuTensor<T>? _gpuBiasVelocity;

    // Optimizer state tensors for Adam/AdamW/LAMB (M and V)
    private GpuTensor<T>? _gpuWeightM;
    private GpuTensor<T>? _gpuWeightV;
    private GpuTensor<T>? _gpuBiasM;
    private GpuTensor<T>? _gpuBiasV;

    #endregion

    /// <summary>
    /// Gets the number of input features (octonion-valued).
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features (octonion-valued).
    /// </summary>
    public int OutputFeatures { get; }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// Each octonion has 8 real components, so the parameter count is:
    /// (InputFeatures * OutputFeatures + OutputFeatures) * 8
    /// </remarks>
    public override int ParameterCount =>
        (InputFeatures * OutputFeatures + OutputFeatures) * 8;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// Returns true only if the activation function also supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => CanActivationBeJitted();

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the OctonionLinearLayer.
    /// </summary>
    /// <param name="inputFeatures">Number of input features (octonion-valued).</param>
    /// <param name="outputFeatures">Number of output features (octonion-valued).</param>
    /// <param name="activationFunction">Optional activation function.</param>
    public OctonionLinearLayer(
        int inputFeatures,
        int outputFeatures,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputFeatures * 8], // Input shape: inputFeatures octonions = inputFeatures * 8 reals
            [outputFeatures * 8], // Output shape: outputFeatures octonions = outputFeatures * 8 reals
            activationFunction ?? new IdentityActivation<T>())
    {
        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;

        _engine = CpuAdvancedAlgebraEngine.Instance;
        _numOps = MathHelper.GetNumericOperations<T>();

        _weights = new Octonion<T>[outputFeatures, inputFeatures];
        _biases = new Octonion<T>[outputFeatures];

        InitializeParameters();
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot initialization adapted for octonions.
    /// </summary>
    private void InitializeParameters()
    {
        var scale = Math.Sqrt(2.0 / (InputFeatures + OutputFeatures));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _weights[o, i] = CreateRandomOctonion(random, scale);
            }
            // Initialize biases to zero
            _biases[o] = CreateZeroOctonion();
        }
    }

    private Octonion<T> CreateRandomOctonion(Random random, double scale)
    {
        return new Octonion<T>(
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale),
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale),
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale),
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale),
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale),
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale),
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale),
            _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale));
    }

    private Octonion<T> CreateZeroOctonion()
    {
        return new Octonion<T>(
            _numOps.Zero, _numOps.Zero, _numOps.Zero, _numOps.Zero,
            _numOps.Zero, _numOps.Zero, _numOps.Zero, _numOps.Zero);
    }

    /// <summary>
    /// Performs the forward pass through the layer.
    /// </summary>
    /// <param name="input">Input tensor with shape [inputFeatures * 8] or [batch, inputFeatures * 8].</param>
    /// <returns>Output tensor with shape [outputFeatures * 8] or [batch, outputFeatures * 8].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int batchSize;
        int inputLen;
        Tensor<T> inputTensor;

        if (input.Rank == 1)
        {
            batchSize = 1;
            inputLen = input.Shape[0];
            inputTensor = input.Reshape([1, inputLen]);
        }
        else if (input.Rank == 2)
        {
            batchSize = input.Shape[0];
            inputLen = input.Shape[1];
            inputTensor = input;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < input.Rank - 1; d++)
            {
                flatBatch *= input.Shape[d];
            }
            batchSize = flatBatch;
            inputLen = input.Shape[input.Rank - 1];
            inputTensor = input.Reshape([batchSize, inputLen]);
        }

        // Validate input shape
        if (inputLen != InputFeatures * 8)
        {
            throw new ArgumentException(
                $"Input size {inputLen} does not match expected {InputFeatures * 8} " +
                $"({InputFeatures} octonions * 8 components).");
        }

        // Convert input tensor to octonion array
        var inputOctonions = TensorToOctonions(inputTensor, batchSize, InputFeatures);
        _lastInput = inputOctonions;

        // Perform octonion matrix multiplication
        var output = _engine.OctonionMatMul(inputOctonions, _weights);

        // Add biases
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < OutputFeatures; o++)
            {
                output[b, o] = output[b, o] + _biases[o];
            }
        }

        _lastOutput = output;

        // Convert back to tensor
        var outputTensor = OctonionsToTensor(output, batchSize, OutputFeatures);

        // Apply activation function
        var activated = ApplyActivation(outputTensor);

        if (_originalInputShape == null || _originalInputShape.Length == 2)
        {
            return activated;
        }

        if (_originalInputShape.Length == 1)
        {
            return activated.Reshape([OutputFeatures * 8]);
        }

        var outputShape = new int[_originalInputShape.Length];
        for (int d = 0; d < _originalInputShape.Length - 1; d++)
        {
            outputShape[d] = _originalInputShape[d];
        }
        outputShape[_originalInputShape.Length - 1] = OutputFeatures * 8;
        return activated.Reshape(outputShape);
    }

    /// <summary>
    /// Performs the GPU-accelerated forward pass for octonion linear transformation.
    /// </summary>
    /// <param name="inputs">The GPU tensor inputs. First element is the input activation.</param>
    /// <returns>A GPU tensor containing the octonion transformation output.</returns>
    /// <remarks>
    /// The octonion layer performs matrix-vector multiplication in 8-dimensional octonion algebra.
    /// This method uses a specialized CUDA kernel to perform octonion operations entirely on GPU.
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var input = inputs[0];

        // Validate GPU engine availability
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend is not available.");

        // Determine batch size from input shape
        int batchSize;
        if (input.Shape.Length == 1)
        {
            batchSize = 1;
        }
        else if (input.Shape.Length == 2)
        {
            batchSize = input.Shape[0];
        }
        else
        {
            // Flatten higher-rank tensors to batch
            batchSize = 1;
            for (int d = 0; d < input.Shape.Length - 1; d++)
            {
                batchSize *= input.Shape[d];
            }
        }

        // Cache input for backward pass if in training mode
        if (IsTrainingMode)
        {
            _gpuInput = input;
            _gpuInputShape = input.Shape.ToArray();
        }

        // Flatten weights to GPU buffer: [outputFeatures * inputFeatures * 8]
        var weightsFlat = new float[OutputFeatures * InputFeatures * 8];
        int wIdx = 0;
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                var oct = _weights[o, i];
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.Scalar);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E1);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E2);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E3);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E4);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E5);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E6);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E7);
            }
        }
        var weightsBuffer = backend.AllocateBuffer(weightsFlat);

        // Flatten biases to GPU buffer: [outputFeatures * 8]
        var biasesFlat = new float[OutputFeatures * 8];
        int bIdx = 0;
        for (int o = 0; o < OutputFeatures; o++)
        {
            var oct = _biases[o];
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.Scalar);
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.E1);
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.E2);
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.E3);
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.E4);
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.E5);
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.E6);
            biasesFlat[bIdx++] = _numOps.ToFloat(oct.E7);
        }
        var biasesBuffer = backend.AllocateBuffer(biasesFlat);

        // Allocate output buffer: [batchSize * outputFeatures * 8]
        var outputBuffer = backend.AllocateBuffer(batchSize * OutputFeatures * 8);

        // Call the GPU kernel for octonion linear forward
        backend.OctonionLinearForward(
            input.Buffer, weightsBuffer, biasesBuffer, outputBuffer,
            batchSize, InputFeatures, OutputFeatures);

        // Clean up temporary GPU buffers
        weightsBuffer.Dispose();
        biasesBuffer.Dispose();

        // Determine output shape
        int[] outputShape;
        if (input.Shape.Length == 1)
        {
            outputShape = [OutputFeatures * 8];
        }
        else if (input.Shape.Length == 2)
        {
            outputShape = [batchSize, OutputFeatures * 8];
        }
        else
        {
            // Restore higher-rank shape
            outputShape = new int[input.Shape.Length];
            for (int d = 0; d < input.Shape.Length - 1; d++)
            {
                outputShape[d] = input.Shape[d];
            }
            outputShape[input.Shape.Length - 1] = OutputFeatures * 8;
        }

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the backward pass on GPU for the octonion linear layer.
    /// Computes gradients using proper octonion algebra with Jacobian of octonion multiplication.
    /// </summary>
    /// <param name="outputGradient">The GPU tensor containing the gradient of the loss with respect to the output.</param>
    /// <returns>The GPU tensor containing the gradient of the loss with respect to the input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (_gpuInput == null || _gpuInputShape == null)
        {
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");
        }

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");
        }

        var backend = gpuEngine.GetBackend();
        if (backend == null)
        {
            throw new InvalidOperationException("GPU backend unavailable.");
        }

        // Determine batch size from cached input shape
        int batchSize;
        if (_gpuInputShape.Length == 1)
        {
            batchSize = 1;
        }
        else if (_gpuInputShape.Length == 2)
        {
            batchSize = _gpuInputShape[0];
        }
        else
        {
            batchSize = 1;
            for (int d = 0; d < _gpuInputShape.Length - 1; d++)
            {
                batchSize *= _gpuInputShape[d];
            }
        }

        // Upload weights to GPU (needed for input gradient computation)
        var weightsFlat = new float[OutputFeatures * InputFeatures * 8];
        int wIdx = 0;
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                var oct = _weights[o, i];
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.Scalar);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E1);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E2);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E3);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E4);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E5);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E6);
                weightsFlat[wIdx++] = _numOps.ToFloat(oct.E7);
            }
        }

        IGpuBuffer? weightsBuffer = null;
        IGpuBuffer? gradInputBuffer = null;
        IGpuBuffer? gradWeightsBuffer = null;
        IGpuBuffer? gradBiasesBuffer = null;

        try
        {
            weightsBuffer = backend.AllocateBuffer(weightsFlat);

            // Allocate gradient buffers
            gradInputBuffer = backend.AllocateBuffer(batchSize * InputFeatures * 8);
            gradWeightsBuffer = backend.AllocateBuffer(OutputFeatures * InputFeatures * 8);
            gradBiasesBuffer = backend.AllocateBuffer(OutputFeatures * 8);

            // Compute input gradient
            backend.OctonionLinearBackwardInput(
                outputGradient.Buffer, _gpuInput.Buffer, weightsBuffer, gradInputBuffer,
                batchSize, InputFeatures, OutputFeatures);

            // Compute weight gradient
            backend.OctonionLinearBackwardWeights(
                outputGradient.Buffer, _gpuInput.Buffer, gradWeightsBuffer,
                batchSize, InputFeatures, OutputFeatures);

            // Compute bias gradient
            backend.OctonionLinearBackwardBiases(
                outputGradient.Buffer, gradBiasesBuffer,
                batchSize, OutputFeatures);

            // Download weight gradients from GPU and store for UpdateParameters
            var weightsGradFlat = new float[OutputFeatures * InputFeatures * 8];
            backend.DownloadBuffer(gradWeightsBuffer, weightsGradFlat);

            _weightsGradient = new Octonion<T>[OutputFeatures, InputFeatures];
            int wgIdx = 0;
            for (int o = 0; o < OutputFeatures; o++)
            {
                for (int i = 0; i < InputFeatures; i++)
                {
                    _weightsGradient[o, i] = new Octonion<T>(
                        _numOps.FromFloat(weightsGradFlat[wgIdx]),
                        _numOps.FromFloat(weightsGradFlat[wgIdx + 1]),
                        _numOps.FromFloat(weightsGradFlat[wgIdx + 2]),
                        _numOps.FromFloat(weightsGradFlat[wgIdx + 3]),
                        _numOps.FromFloat(weightsGradFlat[wgIdx + 4]),
                        _numOps.FromFloat(weightsGradFlat[wgIdx + 5]),
                        _numOps.FromFloat(weightsGradFlat[wgIdx + 6]),
                        _numOps.FromFloat(weightsGradFlat[wgIdx + 7]));
                    wgIdx += 8;
                }
            }

            // Download bias gradients from GPU and store for UpdateParameters
            var biasesGradFlat = new float[OutputFeatures * 8];
            backend.DownloadBuffer(gradBiasesBuffer, biasesGradFlat);

            _biasesGradient = new Octonion<T>[OutputFeatures];
            int bgIdx = 0;
            for (int o = 0; o < OutputFeatures; o++)
            {
                _biasesGradient[o] = new Octonion<T>(
                    _numOps.FromFloat(biasesGradFlat[bgIdx]),
                    _numOps.FromFloat(biasesGradFlat[bgIdx + 1]),
                    _numOps.FromFloat(biasesGradFlat[bgIdx + 2]),
                    _numOps.FromFloat(biasesGradFlat[bgIdx + 3]),
                    _numOps.FromFloat(biasesGradFlat[bgIdx + 4]),
                    _numOps.FromFloat(biasesGradFlat[bgIdx + 5]),
                    _numOps.FromFloat(biasesGradFlat[bgIdx + 6]),
                    _numOps.FromFloat(biasesGradFlat[bgIdx + 7]));
                bgIdx += 8;
            }

            // Store gradients as GPU tensors for UpdateParametersGpu
            var weightsGradientTensor = new Tensor<T>([OutputFeatures * InputFeatures * 8],
                new Vector<T>(DirectGpuEngine.FromFloatArray<T>(weightsGradFlat)));
            var biasesGradientTensor = new Tensor<T>([OutputFeatures * 8],
                new Vector<T>(DirectGpuEngine.FromFloatArray<T>(biasesGradFlat)));
            _gpuWeightGradient = new GpuTensor<T>(backend, weightsGradientTensor, GpuTensorRole.Gradient);
            _gpuBiasGradient = new GpuTensor<T>(backend, biasesGradientTensor, GpuTensorRole.Gradient);

            // Create output gradient tensor with proper shape
            var inputGradient = new GpuTensor<T>(backend, gradInputBuffer, _gpuInputShape, GpuTensorRole.Gradient, ownsBuffer: true);

            // Clear GPU cache after backward pass
            _gpuInput = null;

            // Prevent disposal of gradInputBuffer since it's owned by the returned tensor
            gradInputBuffer = null;

            return inputGradient;
        }
        finally
        {
            weightsBuffer?.Dispose();
            gradWeightsBuffer?.Dispose();
            gradBiasesBuffer?.Dispose();
            gradInputBuffer?.Dispose();
        }
    }

    /// <summary>
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        Tensor<T> gradTensor;
        int batchSize;
        int gradLen = OutputFeatures * 8;

        if (_originalInputShape == null)
        {
            if (outputGradient.Rank == 1)
            {
                batchSize = 1;
                gradTensor = outputGradient.Reshape([1, gradLen]);
            }
            else if (outputGradient.Rank == 2)
            {
                batchSize = outputGradient.Shape[0];
                gradTensor = outputGradient;
            }
            else
            {
                int flatBatch = 1;
                for (int d = 0; d < outputGradient.Rank - 1; d++)
                {
                    flatBatch *= outputGradient.Shape[d];
                }
                batchSize = flatBatch;
                gradTensor = outputGradient.Reshape([batchSize, gradLen]);
            }
        }
        else if (_originalInputShape.Length == 1)
        {
            batchSize = 1;
            gradTensor = outputGradient.Reshape([1, gradLen]);
        }
        else if (_originalInputShape.Length == 2)
        {
            batchSize = outputGradient.Shape[0];
            gradTensor = outputGradient;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < _originalInputShape.Length - 1; d++)
            {
                flatBatch *= _originalInputShape[d];
            }
            batchSize = flatBatch;
            gradTensor = outputGradient.Reshape([batchSize, gradLen]);
        }

        // Apply activation derivative
        var activationGrad = ComputeActivationGradient(gradTensor, _lastOutput);

        // Convert gradient to octonions
        var gradOctonions = TensorToOctonions(activationGrad, batchSize, OutputFeatures);

        // Initialize weight and bias gradients
        _weightsGradient = new Octonion<T>[OutputFeatures, InputFeatures];
        _biasesGradient = new Octonion<T>[OutputFeatures];

        // Compute weight gradients: dW[o,i] = sum_b(grad[b,o].Conjugate() * input[b,i])
        for (int o = 0; o < OutputFeatures; o++)
        {
            var biasGrad = CreateZeroOctonion();
            for (int i = 0; i < InputFeatures; i++)
            {
                var weightGrad = CreateZeroOctonion();
                for (int b = 0; b < batchSize; b++)
                {
                    // Weight gradient uses conjugate of gradient times input
                    weightGrad = weightGrad + (gradOctonions[b, o].Conjugate() * _lastInput[b, i]);
                }
                _weightsGradient[o, i] = weightGrad;
            }
            // Bias gradient is sum of gradients over batch
            for (int b = 0; b < batchSize; b++)
            {
                biasGrad = biasGrad + gradOctonions[b, o];
            }
            _biasesGradient[o] = biasGrad;
        }

        // Compute input gradient: grad_input[b,i] = sum_o(grad[b,o] * weights[o,i].Conjugate())
        var inputGradOctonions = new Octonion<T>[batchSize, InputFeatures];
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                var sum = CreateZeroOctonion();
                for (int o = 0; o < OutputFeatures; o++)
                {
                    // Multiply gradient by conjugate of weight (for proper octonion backprop)
                    sum = sum + (gradOctonions[b, o] * _weights[o, i].Conjugate());
                }
                inputGradOctonions[b, i] = sum;
            }
        }

        var result = OctonionsToTensor(inputGradOctonions, batchSize, InputFeatures);

        if (_originalInputShape == null || _originalInputShape.Length == 2)
        {
            return result;
        }

        if (_originalInputShape.Length == 1)
        {
            return result.Reshape([InputFeatures * 8]);
        }

        var inputShape = new int[_originalInputShape.Length];
        for (int d = 0; d < _originalInputShape.Length - 1; d++)
        {
            inputShape[d] = _originalInputShape[d];
        }
        inputShape[_originalInputShape.Length - 1] = InputFeatures * 8;
        return result.Reshape(inputShape);
    }

    /// <summary> using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        // Update weights: w = w - lr * grad
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                var grad = _weightsGradient[o, i];
                var scaledGrad = new Octonion<T>(
                    _numOps.Multiply(learningRate, grad.Scalar),
                    _numOps.Multiply(learningRate, grad.E1),
                    _numOps.Multiply(learningRate, grad.E2),
                    _numOps.Multiply(learningRate, grad.E3),
                    _numOps.Multiply(learningRate, grad.E4),
                    _numOps.Multiply(learningRate, grad.E5),
                    _numOps.Multiply(learningRate, grad.E6),
                    _numOps.Multiply(learningRate, grad.E7));
                _weights[o, i] = _weights[o, i] - scaledGrad;
            }

            // Update biases
            var biasGrad = _biasesGradient[o];
            var scaledBiasGrad = new Octonion<T>(
                _numOps.Multiply(learningRate, biasGrad.Scalar),
                _numOps.Multiply(learningRate, biasGrad.E1),
                _numOps.Multiply(learningRate, biasGrad.E2),
                _numOps.Multiply(learningRate, biasGrad.E3),
                _numOps.Multiply(learningRate, biasGrad.E4),
                _numOps.Multiply(learningRate, biasGrad.E5),
                _numOps.Multiply(learningRate, biasGrad.E6),
                _numOps.Multiply(learningRate, biasGrad.E7));
            _biases[o] = _biases[o] - scaledBiasGrad;
        }
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all weights and biases.</returns>
    public override Vector<T> GetParameters()
    {
        var paramArray = new T[ParameterCount];
        int idx = 0;

        // Flatten weights
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                var oct = _weights[o, i];
                paramArray[idx++] = oct.Scalar;
                paramArray[idx++] = oct.E1;
                paramArray[idx++] = oct.E2;
                paramArray[idx++] = oct.E3;
                paramArray[idx++] = oct.E4;
                paramArray[idx++] = oct.E5;
                paramArray[idx++] = oct.E6;
                paramArray[idx++] = oct.E7;
            }
        }

        // Flatten biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            var oct = _biases[o];
            paramArray[idx++] = oct.Scalar;
            paramArray[idx++] = oct.E1;
            paramArray[idx++] = oct.E2;
            paramArray[idx++] = oct.E3;
            paramArray[idx++] = oct.E4;
            paramArray[idx++] = oct.E5;
            paramArray[idx++] = oct.E6;
            paramArray[idx++] = oct.E7;
        }

        return new Vector<T>(paramArray);
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}");
        }

        int idx = 0;

        // Restore weights
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _weights[o, i] = new Octonion<T>(
                    parameters[idx], parameters[idx + 1], parameters[idx + 2], parameters[idx + 3],
                    parameters[idx + 4], parameters[idx + 5], parameters[idx + 6], parameters[idx + 7]);
                idx += 8;
            }
        }

        // Restore biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            _biases[o] = new Octonion<T>(
                parameters[idx], parameters[idx + 1], parameters[idx + 2], parameters[idx + 3],
                parameters[idx + 4], parameters[idx + 5], parameters[idx + 6], parameters[idx + 7]);
            idx += 8;
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _originalInputShape = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Exports the layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the octonion linear layer using the
    /// TensorOperations.OctonionMatMul operation. The weights and biases are converted
    /// to tensor format for use in the computation graph.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape is null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension
        // Input shape is [inputFeatures * 8]
        var symbolicInput = new Tensor<T>(new int[] { 1, InputFeatures * 8 });
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Convert weights to tensor format [outputFeatures, inputFeatures, 8]
        var weightsTensor = new Tensor<T>(new int[] { OutputFeatures, InputFeatures, 8 });
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                var oct = _weights[o, i];
                weightsTensor[o, i, 0] = oct.Scalar;
                weightsTensor[o, i, 1] = oct.E1;
                weightsTensor[o, i, 2] = oct.E2;
                weightsTensor[o, i, 3] = oct.E3;
                weightsTensor[o, i, 4] = oct.E4;
                weightsTensor[o, i, 5] = oct.E5;
                weightsTensor[o, i, 6] = oct.E6;
                weightsTensor[o, i, 7] = oct.E7;
            }
        }
        var weightsNode = TensorOperations<T>.Constant(weightsTensor, "weights");

        // Convert biases to tensor format [outputFeatures, 8]
        var biasesTensor = new Tensor<T>(new int[] { OutputFeatures, 8 });
        for (int o = 0; o < OutputFeatures; o++)
        {
            var oct = _biases[o];
            biasesTensor[o, 0] = oct.Scalar;
            biasesTensor[o, 1] = oct.E1;
            biasesTensor[o, 2] = oct.E2;
            biasesTensor[o, 3] = oct.E3;
            biasesTensor[o, 4] = oct.E4;
            biasesTensor[o, 5] = oct.E5;
            biasesTensor[o, 6] = oct.E6;
            biasesTensor[o, 7] = oct.E7;
        }
        var biasesNode = TensorOperations<T>.Constant(biasesTensor, "biases");

        // Perform octonion matrix multiplication
        var outputNode = TensorOperations<T>.OctonionMatMul(inputNode, weightsNode, biasesNode);

        // Apply activation function if needed
        // Note: Octonion activation is applied element-wise to all 8 components
        if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            // Use the activation's ApplyToGraph for proper computation graph integration
            // CanActivationBeJitted() ensures this is only called when activation supports JIT
            outputNode = ScalarActivation.ApplyToGraph(outputNode);
        }

        return outputNode;
    }

    /// <summary>
    /// Converts a tensor to an octonion array.
    /// </summary>
    private Octonion<T>[,] TensorToOctonions(Tensor<T> tensor, int batchSize, int features)
    {
        var result = new Octonion<T>[batchSize, features];

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                int baseIdx = f * 8;
                result[b, f] = new Octonion<T>(
                    tensor[b, baseIdx],
                    tensor[b, baseIdx + 1],
                    tensor[b, baseIdx + 2],
                    tensor[b, baseIdx + 3],
                    tensor[b, baseIdx + 4],
                    tensor[b, baseIdx + 5],
                    tensor[b, baseIdx + 6],
                    tensor[b, baseIdx + 7]);
            }
        }

        return result;
    }

    /// <summary>
    /// Converts an octonion array to a tensor.
    /// </summary>
    private Tensor<T> OctonionsToTensor(Octonion<T>[,] octonions, int batchSize, int features)
    {
        var result = new Tensor<T>([batchSize, features * 8]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                int baseIdx = f * 8;
                var oct = octonions[b, f];
                result[b, baseIdx] = oct.Scalar;
                result[b, baseIdx + 1] = oct.E1;
                result[b, baseIdx + 2] = oct.E2;
                result[b, baseIdx + 3] = oct.E3;
                result[b, baseIdx + 4] = oct.E4;
                result[b, baseIdx + 5] = oct.E5;
                result[b, baseIdx + 6] = oct.E6;
                result[b, baseIdx + 7] = oct.E7;
            }
        }

        return result;
    }

    private Tensor<T> ComputeActivationGradient(Tensor<T> outputGradient, Octonion<T>[,] preActivation)
    {
        // For identity/linear activation (or no activation), gradient passes through unchanged
        if (ScalarActivation is null || ScalarActivation is IdentityActivation<T>)
        {
            return outputGradient;
        }

        // Convert pre-activation octonions to tensor for derivative computation
        int batchSize = preActivation.GetLength(0);
        int features = preActivation.GetLength(1);
        var preActivationTensor = OctonionsToTensor(preActivation, batchSize, features);

        // Compute activation derivative at pre-activation values
        var derivative = DerivativeTensor(ScalarActivation, preActivationTensor);

        // Multiply output gradient element-wise by derivative
        var result = new Tensor<T>(outputGradient.Shape);
        int totalElements = 1;
        foreach (var dim in outputGradient.Shape)
            totalElements *= dim;

        for (int i = 0; i < totalElements; i++)
        {
            result.Data[i] = NumOps.Multiply(outputGradient.Data[i], derivative.Data[i]);
        }

        return result;
    }

    #region GPU Parameter Updates

    /// <summary>
    /// Updates parameters using GPU-based optimizer.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuWeightGradient == null || _gpuBiasGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Convert weights and biases to flat tensors for GPU operations (8 components per octonion)
        int weightSize = OutputFeatures * InputFeatures * 8;
        int biasSize = OutputFeatures * 8;
        var weightTensor = new Tensor<T>([weightSize]);
        var biasTensor = new Tensor<T>([biasSize]);

        int wIdx = 0;
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                var oct = _weights[o, i];
                weightTensor.Data[wIdx++] = oct.Scalar;
                weightTensor.Data[wIdx++] = oct.E1;
                weightTensor.Data[wIdx++] = oct.E2;
                weightTensor.Data[wIdx++] = oct.E3;
                weightTensor.Data[wIdx++] = oct.E4;
                weightTensor.Data[wIdx++] = oct.E5;
                weightTensor.Data[wIdx++] = oct.E6;
                weightTensor.Data[wIdx++] = oct.E7;
            }
        }

        int bIdx = 0;
        for (int o = 0; o < OutputFeatures; o++)
        {
            var oct = _biases[o];
            biasTensor.Data[bIdx++] = oct.Scalar;
            biasTensor.Data[bIdx++] = oct.E1;
            biasTensor.Data[bIdx++] = oct.E2;
            biasTensor.Data[bIdx++] = oct.E3;
            biasTensor.Data[bIdx++] = oct.E4;
            biasTensor.Data[bIdx++] = oct.E5;
            biasTensor.Data[bIdx++] = oct.E6;
            biasTensor.Data[bIdx++] = oct.E7;
        }

        // Ensure GPU weight tensors exist
        _gpuWeights ??= new GpuTensor<T>(backend, weightTensor, GpuTensorRole.Weight);
        _gpuBiases ??= new GpuTensor<T>(backend, biasTensor, GpuTensorRole.Bias);

        // Ensure optimizer state buffers exist
        EnsureOctonionOptimizerState(backend, config.OptimizerType);

        // Apply updates using polymorphic optimizer dispatch
        config.ApplyUpdate(backend, _gpuWeights.Buffer, _gpuWeightGradient.Buffer, BuildOctonionOptimizerState("weights"), weightSize);
        config.ApplyUpdate(backend, _gpuBiases.Buffer, _gpuBiasGradient.Buffer, BuildOctonionOptimizerState("biases"), biasSize);

        // Sync back to CPU octonion arrays for compatibility
        var updatedWeights = _gpuWeights.ToTensor();
        var updatedBiases = _gpuBiases.ToTensor();

        wIdx = 0;
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _weights[o, i] = new Octonion<T>(
                    updatedWeights.Data[wIdx], updatedWeights.Data[wIdx + 1],
                    updatedWeights.Data[wIdx + 2], updatedWeights.Data[wIdx + 3],
                    updatedWeights.Data[wIdx + 4], updatedWeights.Data[wIdx + 5],
                    updatedWeights.Data[wIdx + 6], updatedWeights.Data[wIdx + 7]);
                wIdx += 8;
            }
        }

        bIdx = 0;
        for (int o = 0; o < OutputFeatures; o++)
        {
            _biases[o] = new Octonion<T>(
                updatedBiases.Data[bIdx], updatedBiases.Data[bIdx + 1],
                updatedBiases.Data[bIdx + 2], updatedBiases.Data[bIdx + 3],
                updatedBiases.Data[bIdx + 4], updatedBiases.Data[bIdx + 5],
                updatedBiases.Data[bIdx + 6], updatedBiases.Data[bIdx + 7]);
            bIdx += 8;
        }
    }

    /// <summary>
    /// Ensures GPU optimizer state buffers exist for all octonion parameters.
    /// </summary>
    private void EnsureOctonionOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int weightSize = OutputFeatures * InputFeatures * 8;
        int biasSize = OutputFeatures * 8;

        switch (optimizerType)
        {
            case GpuOptimizerType.Sgd:
            case GpuOptimizerType.Nag:
            case GpuOptimizerType.Lars:
                // Velocity buffers
                _gpuWeightVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // M and V buffers for Adam-family
                _gpuWeightM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                // Squared average buffers (reuse velocity fields)
                _gpuWeightVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], _numOps.Zero), GpuTensorRole.OptimizerState);
                break;
        }
    }

    /// <summary>
    /// Builds the optimizer state for a specific octonion parameter.
    /// </summary>
    private GpuOptimizerState BuildOctonionOptimizerState(string paramName)
    {
        return paramName switch
        {
            "weights" => new GpuOptimizerState { Velocity = _gpuWeightVelocity?.Buffer, M = _gpuWeightM?.Buffer, V = _gpuWeightV?.Buffer, SquaredAvg = _gpuWeightVelocity?.Buffer, AccumulatedGrad = _gpuWeightVelocity?.Buffer },
            "biases" => new GpuOptimizerState { Velocity = _gpuBiasVelocity?.Buffer, M = _gpuBiasM?.Buffer, V = _gpuBiasV?.Buffer, SquaredAvg = _gpuBiasVelocity?.Buffer, AccumulatedGrad = _gpuBiasVelocity?.Buffer },
            _ => new GpuOptimizerState()
        };
    }

    #endregion
}
