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
/// Represents a fully connected layer operating in hyperbolic (Poincare ball) space.
/// </summary>
/// <remarks>
/// <para>
/// A hyperbolic linear layer performs linear transformations in hyperbolic space using
/// Mobius operations. This is particularly useful for learning hierarchical representations
/// where tree-like structures need to be embedded.
/// </para>
/// <para><b>For Beginners:</b> This layer works in hyperbolic space instead of flat Euclidean space.
///
/// Benefits of hyperbolic layers:
/// - Naturally represents hierarchical data (trees, graphs, taxonomies)
/// - Can embed large hierarchies with low distortion
/// - Fewer dimensions needed for complex hierarchical structures
///
/// The layer uses the Poincare ball model where all points are inside a unit ball.
/// Points near the center are "higher" in the hierarchy, points near the edge are "lower".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (float or double).</typeparam>
public class HyperbolicLinearLayer<T> : LayerBase<T>
{
    private readonly IHyperbolicManifoldEngine _engine;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Weight matrix stored in tangent space at the origin.
    /// Shape: [OutputFeatures, InputFeatures]
    /// </summary>
    private Matrix<T> _weights;

    /// <summary>
    /// Bias values as points on the Poincare ball.
    /// Shape: [OutputFeatures, InputFeatures] - each row is a bias point.
    /// </summary>
    private Matrix<T> _biases;

    /// <summary>
    /// The curvature of the hyperbolic space (negative for hyperbolic).
    /// </summary>
    private readonly T _curvature;

    /// <summary>
    /// Stored input from forward pass for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stored pre-activation output for gradient computation.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradient for weights, stored during backward pass.
    /// </summary>
    private Matrix<T>? _weightsGradient;

    /// <summary>
    /// Gradient for biases, stored during backward pass.
    /// </summary>
    private Matrix<T>? _biasesGradient;

    // GPU caching fields for backward pass
    private IGpuTensor<T>? _gpuInput;
    private int[]? _gpuInputShape;

    #region GPU Weight Storage Fields

    // GPU weight tensors for GPU-resident training
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
    /// Gets the number of input features.
    /// </summary>
    public int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int OutputFeatures { get; }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        (OutputFeatures * InputFeatures) + (OutputFeatures * InputFeatures);

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// Hyperbolic operations use TensorOperations for JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the HyperbolicLinearLayer.
    /// </summary>
    /// <param name="inputFeatures">Number of input features.</param>
    /// <param name="outputFeatures">Number of output features.</param>
    /// <param name="curvature">Curvature of hyperbolic space (default -1).</param>
    /// <param name="activationFunction">Optional activation function.</param>
    public HyperbolicLinearLayer(
        int inputFeatures,
        int outputFeatures,
        double curvature = -1.0,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputFeatures],
            [outputFeatures],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (curvature >= 0)
        {
            throw new ArgumentException("Curvature must be negative for hyperbolic space.", nameof(curvature));
        }

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;

        _engine = CpuHyperbolicManifoldEngine.Instance;
        _numOps = MathHelper.GetNumericOperations<T>();
        _curvature = _numOps.FromDouble(curvature);

        _weights = new Matrix<T>(outputFeatures, inputFeatures);
        _biases = new Matrix<T>(outputFeatures, inputFeatures);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes weights using Xavier/Glorot initialization adapted for hyperbolic space.
    /// Weights are initialized in tangent space at the origin.
    /// </summary>
    private void InitializeParameters()
    {
        var scale = Math.Sqrt(2.0 / (InputFeatures + OutputFeatures));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                // Initialize weights in tangent space (small values)
                _weights[o, i] = _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * scale * 0.1);
                // Initialize biases close to origin (small values for Poincare ball)
                _biases[o, i] = _numOps.FromDouble((random.NextDouble() - 0.5) * 2 * 0.01);
            }
        }
    }

    /// <summary>
    /// Performs the forward pass through the layer.
    /// </summary>
    /// <param name="input">Input tensor with shape [inputFeatures] or [batch, inputFeatures].</param>
    /// <returns>Output tensor with shape [outputFeatures] or [batch, outputFeatures].</returns>
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
        if (inputLen != InputFeatures)
        {
            throw new ArgumentException(
                $"Input size {inputLen} does not match expected {InputFeatures}.");
        }

        _lastInput = inputTensor;

        // Output tensor: [batchSize, OutputFeatures]
        var output = new Tensor<T>([batchSize, OutputFeatures]);

        // For each sample in batch
        for (int b = 0; b < batchSize; b++)
        {
            // Extract input vector for this sample
            var inputVec = new Vector<T>(InputFeatures);
            for (int i = 0; i < InputFeatures; i++)
            {
                inputVec[i] = inputTensor[b, i];
            }

            // Project input to Poincare ball (ensure valid point)
            var epsilon = _numOps.FromDouble(1e-5);
            var projectedInput = _engine.PoincareProject(inputVec, _curvature, epsilon);

            // For each output feature
            for (int o = 0; o < OutputFeatures; o++)
            {
                // Get weight vector for this output
                var weightVec = new Vector<T>(InputFeatures);
                for (int i = 0; i < InputFeatures; i++)
                {
                    weightVec[i] = _weights[o, i];
                }

                // Get bias vector for this output
                var biasVec = new Vector<T>(InputFeatures);
                for (int i = 0; i < InputFeatures; i++)
                {
                    biasVec[i] = _biases[o, i];
                }

                // Compute hyperbolic linear transformation:
                // 1. Apply exponential map from origin with weight as tangent vector
                var origin = CreateOriginVector(InputFeatures);
                var weightPoint = _engine.PoincareExpMap(origin, weightVec, _curvature);

                // 2. Mobius addition of input with weight point
                var transformed = _engine.MobiusAdd(projectedInput, weightPoint, _curvature);

                // 3. Mobius addition with bias
                var biasProjected = _engine.PoincareProject(biasVec, _curvature, epsilon);
                var withBias = _engine.MobiusAdd(transformed, biasProjected, _curvature);

                // 4. Compute output as distance from origin (scalar output)
                // This gives a scalar representing "how far down the hierarchy"
                var distance = _engine.PoincareDistance(origin, withBias, _curvature);
                output[b, o] = distance;
            }
        }

        _lastOutput = output;

        // Apply activation function
        var activated = ApplyActivation(output);

        if (_originalInputShape == null || _originalInputShape.Length == 2)
        {
            return activated;
        }

        if (_originalInputShape.Length == 1)
        {
            return activated.Reshape([OutputFeatures]);
        }

        var outputShape = new int[_originalInputShape.Length];
        for (int d = 0; d < _originalInputShape.Length - 1; d++)
        {
            outputShape[d] = _originalInputShape[d];
        }
        outputShape[_originalInputShape.Length - 1] = OutputFeatures;
        return activated.Reshape(outputShape);
    }

    /// <summary>
    /// Performs the GPU-accelerated forward pass for hyperbolic linear transformation.
    /// </summary>
    /// <param name="inputs">The GPU tensor inputs. First element is the input activation.</param>
    /// <returns>A GPU tensor containing the hyperbolic transformation output.</returns>
    /// <remarks>
    /// The hyperbolic layer operates in Poincare ball space using Mobius operations.
    /// This uses GPU kernels for the entire forward pass, keeping data GPU-resident.
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
        if (backend is null)
            throw new InvalidOperationException("GPU backend is not available.");

        // Determine batch size and validate input shape
        int batchSize;
        int inputLen;
        if (input.Shape.Length == 1)
        {
            batchSize = 1;
            inputLen = input.Shape[0];
        }
        else if (input.Shape.Length == 2)
        {
            batchSize = input.Shape[0];
            inputLen = input.Shape[1];
        }
        else
        {
            batchSize = 1;
            for (int d = 0; d < input.Shape.Length - 1; d++)
            {
                batchSize *= input.Shape[d];
            }
            inputLen = input.Shape[^1];
        }

        if (inputLen != InputFeatures)
        {
            throw new ArgumentException($"Input size {inputLen} does not match expected {InputFeatures}.");
        }

        // Cache input for backward pass if in training mode
        if (IsTrainingMode)
        {
            _gpuInput = input;
            _gpuInputShape = input.Shape.ToArray();
        }

        // Cache weights to GPU: flatten [OutputFeatures, InputFeatures] for the kernel
        var weightsFlat = new float[OutputFeatures * InputFeatures];
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                weightsFlat[o * InputFeatures + i] = _numOps.ToFloat(_weights[o, i]);
            }
        }
        IGpuBuffer? weightsBuffer = null;
        IGpuBuffer? biasesBuffer = null;
        try
        {
            weightsBuffer = backend.AllocateBuffer(weightsFlat);

            // Cache biases to GPU: flatten [OutputFeatures, InputFeatures] for the kernel
            var biasesFlat = new float[OutputFeatures * InputFeatures];
            for (int o = 0; o < OutputFeatures; o++)
            {
                for (int i = 0; i < InputFeatures; i++)
                {
                    biasesFlat[o * InputFeatures + i] = _numOps.ToFloat(_biases[o, i]);
                }
            }
            biasesBuffer = backend.AllocateBuffer(biasesFlat);

            // Allocate output buffer
            var outputBuffer = backend.AllocateBuffer(batchSize * OutputFeatures);

            // Get curvature and epsilon as floats
            float curvature = _numOps.ToFloat(_curvature);
            float epsilon = 1e-5f;

            // Validate buffers are allocated (they should be at this point)
            if (weightsBuffer is null || biasesBuffer is null)
                throw new InvalidOperationException("GPU buffer allocation failed");

            // Call the GPU kernel for hyperbolic linear forward
            backend.HyperbolicLinearForward(
                input.Buffer, weightsBuffer, biasesBuffer, outputBuffer,
                batchSize, InputFeatures, OutputFeatures, curvature, epsilon);


            // Determine output shape
            int[] outputShape;
            if (input.Shape.Length == 1)
            {
                outputShape = [OutputFeatures];
            }
            else if (input.Shape.Length == 2)
            {
                outputShape = [batchSize, OutputFeatures];
            }
            else
            {
                var newShape = new int[input.Shape.Length];
                Array.Copy(input.Shape, newShape, input.Shape.Length - 1);
                newShape[^1] = OutputFeatures;
                outputShape = newShape;
            }

            // Note: GPU path does not apply activation function - consider CPU fallback for non-identity activations
            return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
        }
        finally
        {
            weightsBuffer?.Dispose();
            biasesBuffer?.Dispose();
        }
    }

    /// <summary>
    /// Performs the backward pass on GPU for the hyperbolic linear layer.
    /// Computes gradients using Riemannian geometry in the Poincaré ball model.
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

        float curvature = _numOps.ToFloat(_curvature);

        // Upload weights to GPU (needed for input gradient computation)
        var weightsFlat = new float[OutputFeatures * InputFeatures];
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                weightsFlat[o * InputFeatures + i] = _numOps.ToFloat(_weights[o, i]);
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
            gradInputBuffer = backend.AllocateBuffer(batchSize * InputFeatures);
            gradWeightsBuffer = backend.AllocateBuffer(OutputFeatures * InputFeatures);
            gradBiasesBuffer = backend.AllocateBuffer(OutputFeatures * InputFeatures);

            // Compute input gradient
            backend.HyperbolicLinearBackwardInput(
                outputGradient.Buffer, _gpuInput.Buffer, weightsBuffer, gradInputBuffer,
                batchSize, InputFeatures, OutputFeatures, curvature);

            // Compute weight gradient
            backend.HyperbolicLinearBackwardWeights(
                outputGradient.Buffer, _gpuInput.Buffer, gradWeightsBuffer,
                batchSize, InputFeatures, OutputFeatures, curvature);

            // Compute bias gradient
            backend.HyperbolicLinearBackwardBiases(
                outputGradient.Buffer, _gpuInput.Buffer, gradBiasesBuffer,
                batchSize, InputFeatures, OutputFeatures, curvature);

            // Download weight gradients from GPU and store for UpdateParameters
            var weightsGradFlat = new float[OutputFeatures * InputFeatures];
            backend.DownloadBuffer(gradWeightsBuffer, weightsGradFlat);

            _weightsGradient = new Matrix<T>(OutputFeatures, InputFeatures);
            for (int o = 0; o < OutputFeatures; o++)
            {
                for (int i = 0; i < InputFeatures; i++)
                {
                    _weightsGradient[o, i] = _numOps.FromFloat(weightsGradFlat[o * InputFeatures + i]);
                }
            }

            // Download bias gradients from GPU and store for UpdateParameters
            var biasesGradFlat = new float[OutputFeatures * InputFeatures];
            backend.DownloadBuffer(gradBiasesBuffer, biasesGradFlat);

            _biasesGradient = new Matrix<T>(OutputFeatures, InputFeatures);
            for (int o = 0; o < OutputFeatures; o++)
            {
                for (int i = 0; i < InputFeatures; i++)
                {
                    _biasesGradient[o, i] = _numOps.FromFloat(biasesGradFlat[o * InputFeatures + i]);
                }
            }

            // Store gradients as GPU tensors for UpdateParametersGpu
            var weightsGradientTensor = new Tensor<T>([OutputFeatures * InputFeatures],
                new Vector<T>(DirectGpuEngine.FromFloatArray<T>(weightsGradFlat)));
            var biasesGradientTensor = new Tensor<T>([OutputFeatures * InputFeatures],
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

        if (_originalInputShape == null)
        {
            if (outputGradient.Rank == 1)
            {
                batchSize = 1;
                gradTensor = outputGradient.Reshape([1, OutputFeatures]);
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
                gradTensor = outputGradient.Reshape([batchSize, OutputFeatures]);
            }
        }
        else if (_originalInputShape.Length == 1)
        {
            batchSize = 1;
            gradTensor = outputGradient.Reshape([1, OutputFeatures]);
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
            gradTensor = outputGradient.Reshape([batchSize, OutputFeatures]);
        }

        // Initialize gradients
        _weightsGradient = new Matrix<T>(OutputFeatures, InputFeatures);
        _biasesGradient = new Matrix<T>(OutputFeatures, InputFeatures);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute gradients using proper Riemannian gradient descent for Poincaré ball geometry.
        // For the Poincaré ball with curvature c, the conformal factor is:
        //   λ(x) = 2 / (1 - c||x||²)  where c = |curvature|
        // The Riemannian gradient is related to the Euclidean gradient by:
        //   grad_R = (1/λ(x)²) * grad_E = ((1 - c||x||²)² / 4) * grad_E
        var epsilon = _numOps.FromDouble(1e-5);
        var absCurvature = _numOps.Abs(_curvature);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract input vector
            var inputVec = new Vector<T>(InputFeatures);
            for (int i = 0; i < InputFeatures; i++)
            {
                inputVec[i] = _lastInput[b, i];
            }
            var projectedInput = _engine.PoincareProject(inputVec, _curvature, epsilon);

            // Compute squared norm of projected input: ||x||²
            T squaredNorm = _numOps.Zero;
            for (int i = 0; i < InputFeatures; i++)
            {
                squaredNorm = _numOps.Add(squaredNorm, _numOps.Multiply(projectedInput[i], projectedInput[i]));
            }

            // Compute conformal factor for Riemannian gradient:
            // conformalFactor = (1 - c||x||²)² / 4 = 1/λ(x)²
            // This converts Euclidean gradients to Riemannian gradients on the Poincaré ball
            var cNormSquared = _numOps.Multiply(absCurvature, squaredNorm);
            var oneMinusCNorm = _numOps.Subtract(_numOps.One, cNormSquared);
            var conformalFactor = _numOps.Divide(
                _numOps.Multiply(oneMinusCNorm, oneMinusCNorm),
                _numOps.FromDouble(4.0));

            for (int o = 0; o < OutputFeatures; o++)
            {
                T gradOutput = gradTensor[b, o];

                // Scale the output gradient by the conformal factor for Riemannian geometry
                var riemannianGrad = _numOps.Multiply(gradOutput, conformalFactor);

                for (int i = 0; i < InputFeatures; i++)
                {
                    // Weight gradient: Riemannian gradient scaled by input direction
                    var existingWGrad = _weightsGradient[o, i];
                    var inputContrib = _numOps.Multiply(riemannianGrad, projectedInput[i]);
                    _weightsGradient[o, i] = _numOps.Add(existingWGrad, inputContrib);

                    // Bias gradient: Riemannian gradient (conformal factor already applied)
                    // Distributed across input features for the bias point
                    var existingBGrad = _biasesGradient[o, i];
                    var biasContrib = _numOps.Divide(riemannianGrad, _numOps.FromDouble(InputFeatures));
                    _biasesGradient[o, i] = _numOps.Add(existingBGrad, biasContrib);

                    // Input gradient: Riemannian gradient scaled by weight direction
                    var existingIGrad = inputGradient[b, i];
                    var weightContrib = _numOps.Multiply(riemannianGrad, _weights[o, i]);
                    inputGradient[b, i] = _numOps.Add(existingIGrad, weightContrib);
                }
            }
        }

        if (_originalInputShape == null || _originalInputShape.Length == 2)
        {
            return inputGradient;
        }

        return inputGradient.Reshape(_originalInputShape);
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <remarks>
    /// Uses Riemannian gradient descent (exponential map of negative gradient).
    /// </remarks>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        var epsilon = _numOps.FromDouble(1e-5);

        for (int o = 0; o < OutputFeatures; o++)
        {
            // Update weights in tangent space (per-element update)
            for (int i = 0; i < InputFeatures; i++)
            {
                var grad = _weightsGradient[o, i];
                var scaledGrad = _numOps.Multiply(learningRate, grad);
                _weights[o, i] = _numOps.Subtract(_weights[o, i], scaledGrad);
            }

            // Update biases using Riemannian gradient descent
            // Build the full tangent vector from all bias gradients at once
            var biasPoint = new Vector<T>(InputFeatures);
            var tangentVec = new Vector<T>(InputFeatures);
            for (int j = 0; j < InputFeatures; j++)
            {
                biasPoint[j] = _biases[o, j];
                tangentVec[j] = _numOps.Negate(_numOps.Multiply(learningRate, _biasesGradient[o, j]));
            }

            // Project bias to valid region and apply exponential map update once per output
            var projectedBias = _engine.PoincareProject(biasPoint, _curvature, epsilon);
            var updatedBias = _engine.PoincareExpMap(projectedBias, tangentVec, _curvature);
            updatedBias = _engine.PoincareProject(updatedBias, _curvature, epsilon);

            for (int j = 0; j < InputFeatures; j++)
            {
                _biases[o, j] = updatedBias[j];
            }
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
                paramArray[idx++] = _weights[o, i];
            }
        }

        // Flatten biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                paramArray[idx++] = _biases[o, i];
            }
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
                _weights[o, i] = parameters[idx++];
            }
        }

        // Restore biases
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _biases[o, i] = parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _weightsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Exports the layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node.</returns>
    /// <remarks>
    /// The exported computation graph uses TensorOperations.HyperbolicLinear which implements:
    /// 1. Project input to Poincare ball
    /// 2. For each output: exp_origin(weight) → Mobius add with input → Mobius add with bias → distance from origin
    /// The curvature is captured at export time.
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape is null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension [batchSize, inputFeatures]
        var symbolicInput = new Tensor<T>(new int[] { 1, InputFeatures });
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create weight tensor from internal matrix [outputFeatures, inputFeatures]
        var weightTensor = new Tensor<T>(new int[] { OutputFeatures, InputFeatures });
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                weightTensor[o, i] = _weights[o, i];
            }
        }
        var weightsNode = TensorOperations<T>.Constant(weightTensor, "weights");

        // Create bias tensor from internal matrix [outputFeatures, inputFeatures]
        var biasTensor = new Tensor<T>(new int[] { OutputFeatures, InputFeatures });
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                biasTensor[o, i] = _biases[o, i];
            }
        }
        var biasesNode = TensorOperations<T>.Constant(biasTensor, "biases");

        // Get curvature as double for TensorOperations
        double curvature = _numOps.ToDouble(_curvature);

        // Apply HyperbolicLinear operation
        var outputNode = TensorOperations<T>.HyperbolicLinear(
            inputNode,
            weightsNode,
            biasesNode,
            curvature);

        // Apply activation function if needed
        if (ScalarActivation is not null && ScalarActivation is not IdentityActivation<T>)
        {
            outputNode = ApplyActivationToComputationNode(outputNode);
        }

        return outputNode;
    }

    /// <summary>
    /// Applies the activation function to a computation node.
    /// </summary>
    private ComputationNode<T> ApplyActivationToComputationNode(ComputationNode<T> node)
    {
        // ScalarActivation is guaranteed non-null here since this method is only called when ScalarActivation is not null
        if (ScalarActivation is null)
            throw new InvalidOperationException("ScalarActivation cannot be null when applying activation to computation node.");

        // Use ApplyToGraph if the activation supports JIT compilation
        if (ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(node);
        }

        // Fallback: apply activation directly to values and wrap as constant
        var activated = ScalarActivation.Activate(node.Value);
        return TensorOperations<T>.Constant(activated, "activated_output");
    }

    /// <summary>
    /// Creates a vector at the origin of hyperbolic space.
    /// </summary>
    private Vector<T> CreateOriginVector(int dimension)
    {
        var origin = new Vector<T>(dimension);
        for (int i = 0; i < dimension; i++)
        {
            origin[i] = _numOps.Zero;
        }
        return origin;
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

        // Convert weights and biases to flat tensors for GPU operations
        int weightSize = OutputFeatures * InputFeatures;
        int biasSize = OutputFeatures * InputFeatures;
        var weightTensor = new Tensor<T>([weightSize]);
        var biasTensor = new Tensor<T>([biasSize]);
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                weightTensor.Data[o * InputFeatures + i] = _weights[o, i];
                biasTensor.Data[o * InputFeatures + i] = _biases[o, i];
            }
        }

        // Ensure GPU weight tensors exist
        _gpuWeights ??= new GpuTensor<T>(backend, weightTensor, GpuTensorRole.Weight);
        _gpuBiases ??= new GpuTensor<T>(backend, biasTensor, GpuTensorRole.Bias);

        // Ensure optimizer state buffers exist
        EnsureHyperbolicOptimizerState(backend, config.OptimizerType);

        // Apply updates using polymorphic optimizer dispatch
        config.ApplyUpdate(backend, _gpuWeights.Buffer, _gpuWeightGradient.Buffer, BuildHyperbolicOptimizerState("weights"), weightSize);
        config.ApplyUpdate(backend, _gpuBiases.Buffer, _gpuBiasGradient.Buffer, BuildHyperbolicOptimizerState("biases"), biasSize);

        // Sync back to CPU matrices for compatibility
        var updatedWeights = _gpuWeights.ToTensor();
        var updatedBiases = _gpuBiases.ToTensor();
        for (int o = 0; o < OutputFeatures; o++)
        {
            for (int i = 0; i < InputFeatures; i++)
            {
                _weights[o, i] = updatedWeights.Data[o * InputFeatures + i];
                _biases[o, i] = updatedBiases.Data[o * InputFeatures + i];
            }
        }
    }

    /// <summary>
    /// Ensures GPU optimizer state buffers exist for all hyperbolic parameters.
    /// </summary>
    private void EnsureHyperbolicOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int weightSize = OutputFeatures * InputFeatures;
        int biasSize = OutputFeatures * InputFeatures;

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
    /// Builds the optimizer state for a specific hyperbolic parameter.
    /// </summary>
    private GpuOptimizerState BuildHyperbolicOptimizerState(string paramName)
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
