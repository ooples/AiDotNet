using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that performs quantum measurement operations on complex-valued input tensors.
/// </summary>
/// <remarks>
/// <para>
/// The MeasurementLayer transforms complex-valued quantum state amplitudes into classical probabilities.
/// It calculates the probability distribution from a quantum state vector by taking the squared magnitude
/// of each complex amplitude and normalizing the results to ensure they sum to 1.0.
/// </para>
/// <para><b>For Beginners:</b> This layer converts quantum information into regular probabilities.
/// 
/// Think of it like a bridge between the quantum and classical worlds:
/// - In quantum computing, information exists in "superposition" (multiple states at once)
/// - This layer converts that quantum information into classical probabilities
/// - It's similar to how quantum physics says we can only observe probabilities in the real world
/// 
/// For example, if you have a quantum state representing a coin that's in both heads and tails
/// at the same time, the measurement layer would convert this to classical probabilities like
/// "60% chance of heads, 40% chance of tails."
/// 
/// This is a fundamental concept in quantum computing and quantum mechanics.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MeasurementLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the output tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    private int[]? _originalInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> because the MeasurementLayer has no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that MeasurementLayer cannot be trained through backpropagation. Since the
    /// measurement operation is a fixed mathematical procedure with no learnable parameters, this layer always
    /// returns false for SupportsTraining.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer doesn't learn from data.
    /// 
    /// A value of false means:
    /// - The layer has no internal values that change during training
    /// - It always performs the same mathematical operation (converting quantum amplitudes to probabilities)
    /// - It's a fixed transformation rather than a learned one
    /// 
    /// This layer applies the rules of quantum measurement, which are fixed by the laws of physics
    /// rather than something that can be learned or optimized during training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="MeasurementLayer{T}"/> class with the specified size.
    /// </summary>
    /// <param name="size">The size of the quantum state vector (number of basis states).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a MeasurementLayer that operates on quantum state vectors of the specified size.
    /// The input and output shape are both one-dimensional vectors of the specified size.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary information.
    /// 
    /// When creating a MeasurementLayer, you need to specify:
    /// - size: The number of possible states in your quantum system
    /// 
    /// For example:
    /// - For a single qubit (quantum bit), size = 2 (states |0? and |1?)
    /// - For two qubits, size = 4 (states |00?, |01?, |10?, and |11?)
    /// - For n qubits, size = 2^n (all possible combinations)
    /// 
    /// Both the input (quantum amplitudes) and output (classical probabilities) will have this same size.
    /// </para>
    /// </remarks>
    public MeasurementLayer(int size) : base([size], [size])
    {
    }

    /// <summary>
    /// Performs the forward pass of the measurement layer.
    /// </summary>
    /// <param name="input">The input tensor containing complex quantum amplitudes.</param>
    /// <returns>The output tensor containing classical probabilities.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the measurement layer. It calculates the probability
    /// distribution from a quantum state vector by taking the squared magnitude of each complex amplitude
    /// (|z|² = real² + imag²) and normalizing the results to ensure they sum to 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts quantum amplitudes into classical probabilities.
    /// 
    /// During the forward pass:
    /// - The layer receives complex-valued quantum amplitudes
    /// - For each amplitude, it calculates |z|² = real² + imag² (the squared magnitude)
    /// - It normalizes these values so they sum to 1.0 (making them valid probabilities)
    /// - It returns these probabilities as a real-valued tensor
    /// 
    /// This process follows the Born rule from quantum mechanics, which states that
    /// the probability of measuring a particular state is the squared magnitude of
    /// its amplitude in the state vector.
    /// 
    /// For example, if a qubit has amplitudes [0.6+0.3i, 0.7+0.2i], the probabilities
    /// would be approximately [0.45, 0.55] after normalization.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        int stateSize = input.Shape[^1];
        if (stateSize != InputShape[0])
        {
            throw new ArgumentException(
                $"Input size {stateSize} does not match expected {InputShape[0]}.");
        }

        Tensor<T> input2D;
        if (input.Rank == 1)
        {
            input2D = input.Reshape(1, stateSize);
        }
        else if (input.Rank == 2)
        {
            input2D = input;
        }
        else
        {
            int batchDim = 1;
            for (int i = 0; i < input.Rank - 1; i++)
            {
                batchDim *= input.Shape[i];
            }
            input2D = input.Reshape(batchDim, stateSize);
        }

        _lastInput = input2D;

        int batchSize = input2D.Shape[0];
        var probabilities = new Tensor<T>(new[] { batchSize, stateSize });

        for (int b = 0; b < batchSize; b++)
        {
            int baseIndex = b * stateSize;
            T sum = NumOps.Zero;

            for (int i = 0; i < stateSize; i++)
            {
                var complexValue = Tensor<T>.GetComplex(input2D, baseIndex + i);
                var realSquared = NumOps.Multiply(complexValue.Real, complexValue.Real);
                var imagSquared = NumOps.Multiply(complexValue.Imaginary, complexValue.Imaginary);
                var magnitude = NumOps.Add(realSquared, imagSquared);
                probabilities[b, i] = magnitude;
                sum = NumOps.Add(sum, magnitude);
            }

            T invSum = NumOps.Equals(sum, NumOps.Zero)
                ? NumOps.Zero
                : NumOps.Divide(NumOps.One, sum);

            for (int i = 0; i < stateSize; i++)
            {
                probabilities[b, i] = NumOps.Multiply(probabilities[b, i], invSum);
            }
        }

        _lastOutput = probabilities;

        if (input.Rank == 1)
        {
            return probabilities.Reshape([stateSize]);
        }

        if (input.Rank > 2)
        {
            return probabilities.Reshape(_originalInputShape);
        }

        return probabilities;
    }

    /// <summary>
    /// Performs the GPU-accelerated forward pass for quantum measurement.
    /// </summary>
    /// <param name="inputs">The GPU tensor inputs. First element is the complex-valued quantum state.</param>
    /// <returns>A GPU tensor containing the classical probability distribution.</returns>
    /// <remarks>
    /// The measurement layer converts quantum amplitudes to probabilities via the Born rule.
    /// This method downloads complex input, processes measurement on CPU, and uploads probabilities to GPU.
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

        // Download input from GPU for processing
        var inputData = backend.DownloadBuffer(input.Buffer);

        // Convert to Tensor<T>
        var inputTensor = new Tensor<T>(input.Shape);
        for (int i = 0; i < inputData.Length; i++)
        {
            inputTensor[i] = NumOps.FromDouble(inputData[i]);
        }

        // Process using existing Forward logic (handles complex-valued quantum measurement)
        var output = Forward(inputTensor);

        // Upload output to GPU
        var outputData = new float[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            outputData[i] = NumOps.ToFloat(output[i]);
        }

        var outputBuffer = backend.AllocateBuffer(outputData);
        var outputShape = output.Shape;

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the backward pass of the measurement layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the measurement layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradient of the measurement
    /// operation with respect to the complex input amplitudes.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in the quantum amplitudes
    /// affect the final probabilities.
    /// 
    /// During the backward pass:
    /// - The layer receives gradients indicating how the output probabilities should change
    /// - It calculates how the input quantum amplitudes should change to achieve those probability changes
    /// - This involves partial derivatives of the Born rule formula
    /// 
    /// While quantum measurement in the real world is irreversible, in quantum machine learning
    /// we can calculate these gradients for training purposes, even though they don't have a direct
    /// physical interpretation.
    /// 
    /// This allows quantum neural networks to learn from data just like classical neural networks.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _originalInputShape == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        bool shapeMatches = outputGradient.Shape.Length == _lastOutput.Shape.Length;
        if (shapeMatches)
        {
            for (int i = 0; i < _lastOutput.Shape.Length; i++)
            {
                if (outputGradient.Shape[i] != _lastOutput.Shape[i])
                {
                    shapeMatches = false;
                    break;
                }
            }
        }

        Tensor<T> normalizedOutputGradient = outputGradient;
        if (!shapeMatches)
        {
            if (outputGradient.Length == _lastOutput.Length)
            {
                normalizedOutputGradient = outputGradient.Reshape(_lastOutput.Shape);
            }
            else
            {
                throw new ArgumentException("Output gradient shape does not match measurement output.");
            }
        }

        int batchSize = _lastInput.Shape[0];
        int stateSize = _lastInput.Shape[1];
        var inputGradient = new Tensor<T>(new int[] { batchSize, stateSize });
        var two = NumOps.FromDouble(2.0);

        for (int b = 0; b < batchSize; b++)
        {
            int baseIndex = b * stateSize;
            T sum = NumOps.Zero;
            var q = new T[stateSize];

            for (int i = 0; i < stateSize; i++)
            {
                var complexValue = Tensor<T>.GetComplex(_lastInput, baseIndex + i);
                var realSquared = NumOps.Multiply(complexValue.Real, complexValue.Real);
                var imagSquared = NumOps.Multiply(complexValue.Imaginary, complexValue.Imaginary);
                var magnitude = NumOps.Add(realSquared, imagSquared);
                q[i] = magnitude;
                sum = NumOps.Add(sum, magnitude);
            }

            if (NumOps.Equals(sum, NumOps.Zero))
            {
                continue;
            }

            T gDotQ = NumOps.Zero;
            for (int i = 0; i < stateSize; i++)
            {
                gDotQ = NumOps.Add(gDotQ, NumOps.Multiply(normalizedOutputGradient[b, i], q[i]));
            }

            var denom = NumOps.Multiply(sum, sum);

            for (int i = 0; i < stateSize; i++)
            {
                var numerator = NumOps.Subtract(NumOps.Multiply(normalizedOutputGradient[b, i], sum), gDotQ);
                var dLdq = NumOps.Divide(numerator, denom);

                var complexValue = Tensor<T>.GetComplex(_lastInput, baseIndex + i);
                var dReal = NumOps.Multiply(NumOps.Multiply(two, complexValue.Real), dLdq);
                var dImag = NumOps.Multiply(NumOps.Multiply(two, complexValue.Imaginary), dLdq);

                inputGradient[b, i] = NumOps.Add(dReal, dImag);
            }
        }

        return inputGradient.Reshape(_originalInputShape);
    }
    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It constructs a computation graph
    /// that mirrors the forward pass operations (magnitude squared -> Sum -> Normalize) to calculate exact gradients.
    /// The Forward pass uses GetComplex which treats each element as a complex value. For real-valued tensors (T=float/double),
    /// the imaginary part is 0, so |z|² = real² + 0² = real². The output shape matches the input shape.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _originalInputShape == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        bool shapeMatches = outputGradient.Shape.Length == _lastOutput.Shape.Length;
        if (shapeMatches)
        {
            for (int i = 0; i < _lastOutput.Shape.Length; i++)
            {
                if (outputGradient.Shape[i] != _lastOutput.Shape[i])
                {
                    shapeMatches = false;
                    break;
                }
            }
        }

        Tensor<T> normalizedOutputGradient = outputGradient;
        if (!shapeMatches)
        {
            if (outputGradient.Length == _lastOutput.Length)
            {
                normalizedOutputGradient = outputGradient.Reshape(_lastOutput.Shape);
            }
            else
            {
                throw new ArgumentException("Output gradient shape does not match measurement output.");
            }
        }

        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var magnitudeSquared = Autodiff.TensorOperations<T>.Square(inputNode);
        var sumSquared = Autodiff.TensorOperations<T>.Sum(magnitudeSquared, new int[] { 1 }, keepDims: true);

        var epsilonTensor = new Tensor<T>(new int[] { 1 });
        epsilonTensor[0] = NumOps.FromDouble(1e-10);
        var epsilonNode = Autodiff.TensorOperations<T>.Constant(epsilonTensor, "epsilon");
        var safeSum = Autodiff.TensorOperations<T>.Add(sumSquared, epsilonNode);

        var output = Autodiff.TensorOperations<T>.Divide(magnitudeSquared, safeSum);

        output.Gradient = normalizedOutputGradient;
        output.Backward();

        if (inputNode.Gradient == null)
        {
            throw new InvalidOperationException("Gradient computation failed.");
        }

        return inputNode.Gradient.Reshape(_originalInputShape);
    }
    /// <summary>
    /// Updates the parameters of the measurement layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is part of the training process, but since MeasurementLayer has no trainable parameters,
    /// this method does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally update a layer's internal values during training.
    /// 
    /// However, since MeasurementLayer just performs a fixed mathematical operation (quantum measurement)
    /// and doesn't have any internal values that can be learned or adjusted, this method is empty.
    /// 
    /// The measurement process follows the fundamental rules of quantum mechanics, which are 
    /// constant rather than learnable parameters.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // MeasurementLayer doesn't have trainable parameters
    }

    /// <summary>
    /// Gets all trainable parameters from the measurement layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since MeasurementLayer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. Since MeasurementLayer
    /// has no trainable parameters, it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the learnable values in the layer.
    /// 
    /// Since MeasurementLayer:
    /// - Only performs fixed mathematical operations based on quantum mechanics
    /// - Has no weights, biases, or other learnable parameters
    /// - The method returns an empty list
    /// 
    /// This is different from layers like Dense layers, which would return their weights and biases.
    /// The measurement process is governed by the laws of quantum mechanics rather than by
    /// parameters that can be optimized during training.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // MeasurementLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the measurement layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the measurement layer, including the cached inputs and outputs.
    /// This is useful when starting to process a new batch of data or when implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous processing are cleared
    /// - The layer forgets any information from previous data batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Ensuring clean state before a new training epoch
    /// - Preventing information from one batch affecting another
    /// 
    /// While the MeasurementLayer doesn't maintain long-term state across samples,
    /// clearing these cached values helps with memory management and ensuring a clean processing pipeline.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _originalInputShape = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var input = inputNodes[0];
        int size = InputShape[0];

        // MeasurementLayer computes quantum measurement: probabilities = |amplitude|^2 / sum(|amplitude|^2)
        // Input is complex-valued stored as [real_0, imag_0, real_1, imag_1, ...] or [real; imag] halves
        // Assuming interleaved format: extract real and imaginary parts

        // For interleaved format [r0, i0, r1, i1, ...]:
        // Extract even indices (real) and odd indices (imaginary)
        var realPart = TensorOperations<T>.Slice(input, 0, size, step: 2, axis: 0);
        var imagPart = TensorOperations<T>.Slice(input, 1, size, step: 2, axis: 0);

        // Compute |amplitude|^2 = real^2 + imag^2
        var realSquared = TensorOperations<T>.Square(realPart);
        var imagSquared = TensorOperations<T>.Square(imagPart);
        var magnitudeSquared = TensorOperations<T>.Add(realSquared, imagSquared);

        // Compute sum for normalization
        var totalSum = TensorOperations<T>.Sum(magnitudeSquared);

        // Normalize to get probabilities (add epsilon to avoid division by zero)
        var epsilonTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(1e-10) }));
        var epsilon = TensorOperations<T>.Constant(epsilonTensor, "Epsilon");
        var safeDenom = TensorOperations<T>.Add(totalSum, epsilon);
        var probabilities = TensorOperations<T>.Divide(magnitudeSquared, safeDenom);

        return probabilities;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> because MeasurementLayer computes quantum measurement using only
    /// standard arithmetic operations: |amplitude|^2 = real^2 + imag^2, normalized by sum.
    /// </value>
    public override bool SupportsJitCompilation => true;

}
