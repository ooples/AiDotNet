using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a neural network layer that uses quantum computing principles for processing inputs.
/// </summary>
/// <remarks>
/// <para>
/// The QuantumLayer implements a simulated quantum circuit that processes input data using quantum
/// rotations and measurements. It transforms classical inputs into quantum states, applies quantum
/// operations, and converts the results back to classical outputs. This approach can potentially
/// capture complex patterns that traditional neural network layers might miss.
/// </para>
/// <para><b>For Beginners:</b> This layer uses concepts from quantum computing to process data in a unique way.
/// 
/// Think of it like a special filter that:
/// - Transforms regular data into a quantum-like format (similar to how light can be both a wave and a particle)
/// - Performs calculations that explore multiple possibilities simultaneously
/// - Converts the results back into standard values that other layers can work with
/// 
/// While traditional neural networks work with definite values, quantum layers work with probabilities
/// and superpositions (being in multiple states at once). This can help the network find patterns
/// that might be missed with traditional approaches.
/// 
/// You don't need to understand quantum physics to use this layer - just know that it offers a
/// different way of processing information that can be powerful for certain problems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class QuantumLayer<T> : LayerBase<T>
{
    private readonly int _numQubits;
    private Tensor<Complex<T>> _quantumCircuit;
    private Tensor<T> _circuitReal;
    private Tensor<T> _circuitImag;
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T> _rotationAngles;
    private Tensor<T> _angleGradients;
    private readonly INumericOperations<Complex<T>> _complexOps;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> indicating that this layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the QuantumLayer has parameters (rotation angles) that
    /// can be optimized during the training process using backpropagation. The gradients of
    /// these parameters are calculated during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values (rotation angles) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// When you train a neural network containing this layer, the rotation angles will 
    /// automatically adjust to better process your specific data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="QuantumLayer{T}"/> class with specified dimensions.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer.</param>
    /// <param name="numQubits">The number of qubits to use in the quantum circuit.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new QuantumLayer with the specified dimensions. The number of qubits
    /// determines the complexity of the quantum circuit. The quantum circuit is initialized with random
    /// rotation angles, which are the trainable parameters of the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new quantum layer for your neural network.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many numbers come into the layer
    /// - outputSize: How many numbers come out of the layer
    /// - numQubits: How complex the quantum calculations should be
    /// 
    /// More qubits (quantum bits) mean more complex calculations but also require more
    /// computational resources. The layer starts with random settings that will be
    /// refined during training.
    /// 
    /// For example, a layer with 3 qubits can process 8 (2³) different states simultaneously,
    /// which is what gives quantum computing its potential power.
    /// </para>
    /// </remarks>
    public QuantumLayer(int inputSize, int outputSize, int numQubits) : base([inputSize], [outputSize])
    {
        _numQubits = numQubits;
        _complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        // Initialize parameters as Tensor<T>
        _rotationAngles = new Tensor<T>([_numQubits]);
        _angleGradients = new Tensor<T>([_numQubits]);

        // Create quantum circuit as a tensor
        int dimension = 1 << _numQubits;
        _quantumCircuit = new Tensor<Complex<T>>([dimension, dimension]);
        _circuitReal = new Tensor<T>([dimension, dimension]);
        _circuitImag = new Tensor<T>([dimension, dimension]);

        InitializeQuantumCircuit();
    }

    /// <summary>
    /// Performs the forward pass of the quantum layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after quantum processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the quantum layer. It converts the input tensor to
    /// a quantum state, applies the quantum circuit, and then measures the state to produce the output.
    /// The quantum state is normalized to ensure valid probabilities, and the output represents the
    /// probability distribution of the quantum state after measurement.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the quantum circuit.
    /// 
    /// During the forward pass:
    /// 1. Your regular data is converted to a quantum state
    /// 2. The quantum circuit (with its rotation angles) processes this state
    /// 3. The resulting quantum state is measured to get probabilities
    /// 4. These probabilities form the output of the layer
    /// 
    /// This is like running an experiment where quantum particles can exist in multiple
    /// states, and then checking which states they actually end up in when measured.
    /// The layer saves the input for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse to 2D for processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            // 1D: add batch dim
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // Standard 2D
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 1]]);
        }

        _lastInput = processInput;
        int dimension = 1 << _numQubits;
        int inputDim = processInput.Shape[1]; // Use processInput which is always 2D

        // Ensure input matches circuit dimension (pad or slice using engine ops)
        Tensor<T> realState;
        if (inputDim == dimension)
        {
            realState = processInput;
        }
        else if (inputDim < dimension)
        {
            realState = new Tensor<T>([batchSize, dimension]);
            Engine.TensorSetSlice(realState, processInput, [0, 0]);
        }
        else
        {
            realState = Engine.TensorSlice(processInput, [0, 0], [batchSize, dimension]);
        }

        var imagState = new Tensor<T>(realState.Shape);

        // Normalize each batch item: divide by sqrt(sum(|state|^2) + eps)
        var magnitudeSquared = Engine.ComplexMagnitudeSquared(realState, imagState);
        var normPerBatch = Engine.ReduceSum(magnitudeSquared, [1], keepDims: true);
        var epsilonTensor = new Tensor<T>(normPerBatch.Shape);
        epsilonTensor.Fill(NumOps.FromDouble(1e-10));
        var safeDenom = Engine.TensorAdd(normPerBatch, epsilonTensor);
        var denomExpanded = Engine.TensorRepeatElements(safeDenom, dimension, axis: 1);
        var normalizedReal = Engine.TensorDivide(realState, denomExpanded);
        var normalizedImag = Engine.TensorDivide(imagState, denomExpanded);

        // Reshape to [dimension, batch] for complex matmul
        var normalizedRealT = Engine.TensorTranspose(normalizedReal);
        var normalizedImagT = Engine.TensorTranspose(normalizedImag);

        // Apply quantum circuit using complex matrix multiplication
        var (resultRealT, resultImagT) = Engine.ComplexMatMul(_circuitReal, _circuitImag, normalizedRealT, normalizedImagT);

        // Convert amplitudes to probabilities and transpose back to [batch, dimension]
        var probabilitiesT = Engine.ComplexMagnitudeSquared(resultRealT, resultImagT);
        return Engine.TensorTranspose(probabilitiesT);
    }

    /// <summary>
    /// Performs the GPU-accelerated forward pass for the quantum layer.
    /// </summary>
    /// <param name="inputs">The GPU tensor inputs. First element is the input activation.</param>
    /// <returns>A GPU tensor containing the quantum probability distribution output.</returns>
    /// <remarks>
    /// The quantum layer processes data through a simulated quantum circuit with complex-valued operations.
    /// This method uses specialized CUDA kernels to perform quantum operations entirely on GPU.
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
        int inputDim;
        if (input.Shape.Length == 1)
        {
            batchSize = 1;
            inputDim = input.Shape[0];
        }
        else if (input.Shape.Length == 2)
        {
            batchSize = input.Shape[0];
            inputDim = input.Shape[1];
        }
        else
        {
            // Flatten higher-rank tensors to batch
            batchSize = 1;
            for (int d = 0; d < input.Shape.Length - 1; d++)
            {
                batchSize *= input.Shape[d];
            }
            inputDim = input.Shape[input.Shape.Length - 1];
        }

        int dimension = 1 << _numQubits; // State dimension = 2^numQubits

        // Upload quantum circuit to GPU (real and imaginary parts separately)
        var circuitRealFlat = new float[dimension * dimension];
        var circuitImagFlat = new float[dimension * dimension];
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                int idx = i * dimension + j;
                circuitRealFlat[idx] = NumOps.ToFloat(_quantumCircuit[i, j].Real);
                circuitImagFlat[idx] = NumOps.ToFloat(_quantumCircuit[i, j].Imaginary);
            }
        }
        var circuitRealBuffer = backend.AllocateBuffer(circuitRealFlat);
        var circuitImagBuffer = backend.AllocateBuffer(circuitImagFlat);

        // Initialize state: Copy and pad/normalize input on CPU, then allocate with data
        // PERFORMANCE NOTE: This CPU round-trip (Download -> process -> Upload) is a performance
        // bottleneck. A full GPU implementation would use dedicated kernels for state initialization
        // (padding, normalization). Consider adding GPU kernels for these operations in performance-critical
        // scenarios, especially for large batch sizes or when this layer is used frequently.
        var inputData = backend.DownloadBuffer(input.Buffer);
        var stateReal = new float[batchSize * dimension];
        var stateImag = new float[batchSize * dimension]; // Zeros for imaginary part

        for (int b = 0; b < batchSize; b++)
        {
            // Copy and pad input to dimension
            float sumSq = 0;
            for (int i = 0; i < Math.Min(inputDim, dimension); i++)
            {
                int srcIdx = b * inputDim + i;
                float val = inputData[srcIdx];
                stateReal[b * dimension + i] = val;
                sumSq += val * val;
            }
            // Normalize
            float norm = (float)Math.Sqrt(sumSq + 1e-10);
            for (int i = 0; i < dimension; i++)
            {
                stateReal[b * dimension + i] /= norm;
            }
        }

        // Allocate GPU buffers with initialized data
        var stateRealBuffer = backend.AllocateBuffer(stateReal);
        var stateImagBuffer = backend.AllocateBuffer(stateImag);

        // Allocate output state buffers
        var resultRealBuffer = backend.AllocateBuffer(batchSize * dimension);
        var resultImagBuffer = backend.AllocateBuffer(batchSize * dimension);

        // Apply quantum circuit: complex matrix multiplication
        backend.ComplexMatVec(
            circuitRealBuffer, circuitImagBuffer,
            stateRealBuffer, stateImagBuffer,
            resultRealBuffer, resultImagBuffer,
            batchSize, dimension);

        // Compute probabilities: |amplitude|^2
        var probabilitiesBuffer = backend.AllocateBuffer(batchSize * dimension);
        backend.QuantumMeasurement(resultRealBuffer, resultImagBuffer, probabilitiesBuffer, batchSize, dimension);

        // Clean up intermediate buffers
        circuitRealBuffer.Dispose();
        circuitImagBuffer.Dispose();
        stateRealBuffer.Dispose();
        stateImagBuffer.Dispose();
        resultRealBuffer.Dispose();
        resultImagBuffer.Dispose();

        // Determine output shape
        int[] outputShape;
        if (input.Shape.Length == 1)
        {
            outputShape = [dimension];
        }
        else if (input.Shape.Length == 2)
        {
            outputShape = [batchSize, dimension];
        }
        else
        {
            // Restore higher-rank shape
            outputShape = new int[input.Shape.Length];
            for (int d = 0; d < input.Shape.Length - 1; d++)
            {
                outputShape[d] = input.Shape[d];
            }
            outputShape[input.Shape.Length - 1] = dimension;
        }

        return new GpuTensor<T>(backend, probabilitiesBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the backward pass of the quantum layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the quantum layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradient of the loss
    /// with respect to the input and updates the gradients of the rotation angles. The quantum circuit
    /// adjoint (conjugate transpose) is used for backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The error gradient from the next layer is received
    /// 2. This gradient is passed backward through the quantum circuit
    /// 3. Gradients for the rotation angles are calculated and stored
    /// 4. The gradient for the input is calculated and returned
    /// 
    /// This process allows the neural network to learn by adjusting both the input to this layer
    /// and the rotation angles within the quantum circuit. It's part of the "backpropagation"
    /// algorithm that helps neural networks improve over time.
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
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        int batchSize = outputGradient.Shape[0];
        int dimension = 1 << _numQubits;
        int inputDimension = _lastInput.Shape[1];

        // Create input gradient tensor
        var inputGradient = new Tensor<T>([batchSize, inputDimension]);

        for (int b = 0; b < batchSize; b++)
        {
            // Convert output gradient to complex form
            var gradientState = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < dimension; i++)
            {
                gradientState[i] = new Complex<T>(outputGradient[b, i], NumOps.Zero);
            }

            // Backpropagate through quantum circuit
            var backpropGradient = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < dimension; i++)
            {
                backpropGradient[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
                for (int j = 0; j < dimension; j++)
                {
                    // Use the Conjugate method from Complex<T> directly
                    var conjugate = _quantumCircuit[j, i].Conjugate();
                    backpropGradient[i] = _complexOps.Add(backpropGradient[i],
                        _complexOps.Multiply(conjugate, gradientState[j]));
                }
            }

            // Update parameter gradients
            UpdateAngleGradients(gradientState, b);

            // Copy gradients to output tensor
            for (int i = 0; i < Math.Min(inputDimension, dimension); i++)
            {
                // Calculate magnitude manually from the complex number
                var complex = backpropGradient[i];
                var magnitudeSquared = NumOps.Add(
                    NumOps.Multiply(complex.Real, complex.Real),
                    NumOps.Multiply(complex.Imaginary, complex.Imaginary)
                );
                inputGradient[b, i] = NumOps.Sqrt(magnitudeSquared);
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It builds a computation graph
    /// for the quantum measurement process (State -> Circuit -> Measurement) to compute exact gradients
    /// for both the input and the rotation angles.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int dimension = 1 << _numQubits;
        int inputSize = _lastInput.Shape[1];
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Reset angle gradients
        _angleGradients.Fill(NumOps.Zero);

        // Prepare Quantum Circuit Node (Constant for the batch pass)
        // Convert _quantumCircuit to real/imag split format [2*dim, dim]
        var circuitRealImag = new T[dimension * dimension * 2];
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                var complex = _quantumCircuit[i, j];
                circuitRealImag[i * dimension + j] = complex.Real;
                circuitRealImag[(dimension + i) * dimension + j] = complex.Imaginary;
            }
        }
        var circuitTensor = new Tensor<T>(new[] { 2 * dimension, dimension }, new Vector<T>(circuitRealImag));
        // We mark circuit as Variable to allow gradient flow through it, but we won't use circuitNode.Gradient directly
        // for updates (we use UpdateAngleGradients). However, setting it to Variable ensures backprop flows to it.
        // Wait, UpdateAngleGradients takes dL/dPsi (result gradient).
        // We don't need dL/dCircuit from graph?
        // Yes, UpdateAngleGradients computes dL/dAngles using dL/dPsi and Circuit.
        // So we don't strictly need circuit to be a Variable, but we need result (psi) gradient.
        // Result depends on Circuit. If Circuit is Constant, result gradient is still computed w.r.t Result.
        // So Constant is fine.
        var quantumCircuitNode = Autodiff.TensorOperations<T>.Constant(circuitTensor, "QuantumCircuit");

        for (int b = 0; b < batchSize; b++)
        {
            // 1. Build graph for single item
            // Slice input [1, inputSize]
            var inputSlice = new Tensor<T>([1, inputSize]);
            for (int i = 0; i < inputSize; i++) inputSlice[0, i] = _lastInput[b, i];
            var inputNode = Autodiff.TensorOperations<T>.Variable(inputSlice, "input", requiresGradient: true);

            // Pad and Normalize (mirrors ExportComputationGraph)
            // ... simplified logic matching Export ...
            // Input is [1, inputSize]. Flatten to [inputSize].
            var flatInput = Autodiff.TensorOperations<T>.Reshape(inputNode, inputSize);

            // Padding
            int padAmount = dimension - inputSize;
            Autodiff.ComputationNode<T> paddedInput = flatInput;
            if (padAmount > 0)
            {
                var padTensor = new Tensor<T>([padAmount]); // zeros
                padTensor.Fill(NumOps.Zero);
                var padNode = Autodiff.TensorOperations<T>.Constant(padTensor, "pad");
                paddedInput = Autodiff.TensorOperations<T>.Concat(new List<Autodiff.ComputationNode<T>> { flatInput, padNode }, axis: 0);
            }

            // Normalize
            var inputSquared = Autodiff.TensorOperations<T>.Square(paddedInput);
            var sumSquared = Autodiff.TensorOperations<T>.Sum(inputSquared);
            var normFactor = Autodiff.TensorOperations<T>.Sqrt(sumSquared);
            var epsilonTensor = new Tensor<T>(new[] { 1 });
            epsilonTensor[0] = NumOps.FromDouble(1e-10);
            var epsilon = Autodiff.TensorOperations<T>.Constant(epsilonTensor, "Epsilon");
            var safeDenom = Autodiff.TensorOperations<T>.Add(normFactor, epsilon);
            var normalizedInput = Autodiff.TensorOperations<T>.Divide(paddedInput, safeDenom);

            // Create complex state [normalized; zeros]
            var zerosTensor = new Tensor<T>([dimension]);
            zerosTensor.Fill(NumOps.Zero);
            var zerosNode = Autodiff.TensorOperations<T>.Constant(zerosTensor, "zeros");
            var complexState = Autodiff.TensorOperations<T>.Concat(new List<Autodiff.ComputationNode<T>> { normalizedInput, zerosNode }, axis: 0);

            // Apply circuit
            var result = Autodiff.TensorOperations<T>.ComplexMatMul(quantumCircuitNode, complexState, "split");

            // Probabilities
            var resultReal = Autodiff.TensorOperations<T>.Slice(result, 0, dimension, step: 1, axis: 0);
            var resultImag = Autodiff.TensorOperations<T>.Slice(result, dimension, dimension, step: 1, axis: 0);
            var realSquared = Autodiff.TensorOperations<T>.Square(resultReal);
            var imagSquared = Autodiff.TensorOperations<T>.Square(resultImag);
            var probabilities = Autodiff.TensorOperations<T>.Add(realSquared, imagSquared);

            // 2. Set Gradient
            // Output gradient slice [dimension]
            var gradSlice = new Tensor<T>([dimension]);
            for (int i = 0; i < dimension; i++) gradSlice[i] = outputGradient[b, i];
            probabilities.Gradient = gradSlice;

            // 3. Backward
            probabilities.Backward();

            // 4. Store Input Gradient
            var inGrad = inputNode.Gradient;
            if (inGrad != null)
            {
                for (int i = 0; i < inputSize; i++) inputGradient[b, i] = inGrad[0, i];
            }

            // 5. Update Angle Gradients
            // Get dL/dResult from result node
            var resGrad = result.Gradient;
            if (resGrad != null)
            {
                // Reconstruct complex gradient tensor [dimension]
                var complexGrad = new Tensor<Complex<T>>([dimension]);
                for (int i = 0; i < dimension; i++)
                {
                    var r = resGrad[i];
                    var im = resGrad[dimension + i];
                    complexGrad[i] = new Complex<T>(r, im);
                }

                UpdateAngleGradients(complexGrad, b);
            }
        }

        return inputGradient;
    }


    /// <summary>
    /// Updates the parameters of the quantum layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the rotation angles of the quantum circuit based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. After updating the angles, the quantum circuit is reconstructed with the new angles.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. The rotation angles are adjusted based on their gradients
    /// 2. The learning rate controls how big each update step is
    /// 3. Angles are kept within a valid range (0 to 2p)
    /// 4. The quantum circuit is updated with the new angles
    /// 
    /// This is how the quantum layer "learns" from data over time. Smaller learning rates
    /// mean slower but more stable learning, while larger learning rates mean faster but
    /// potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Use Engine operations for gradient update
        var scaledGradients = Engine.TensorMultiplyScalar(_angleGradients, learningRate);
        _rotationAngles = Engine.TensorSubtract(_rotationAngles, scaledGradients);

        // Ensure angles stay within [0, 2π] and apply rotations
        for (int i = 0; i < _numQubits; i++)
        {
            _rotationAngles[i] = MathHelper.Modulo(
                NumOps.Add(_rotationAngles[i], NumOps.FromDouble(2 * Math.PI)),
                NumOps.FromDouble(2 * Math.PI));

            // Apply updated rotation
            ApplyRotation(i, _rotationAngles[i]);
        }

        // Reset angle gradients for the next iteration
        _angleGradients = new Tensor<T>([_numQubits]);
        _angleGradients.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Gets all trainable parameters of the quantum layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (rotation angles).</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (rotation angles) of the quantum layer as a
    /// single vector. This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the quantum layer.
    /// 
    /// The parameters:
    /// - Are the rotation angles that the quantum layer learns during training
    /// - Control how the quantum circuit processes information
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Return rotation angles as a Vector<T>
        return new Vector<T>(_rotationAngles.ToArray());
    }

    /// <summary>
    /// Sets the trainable parameters of the quantum layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (rotation angles) to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters (rotation angles) of the quantum layer from a single vector.
    /// The quantum circuit is reset and reconstructed with the new rotation angles. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the rotation angles in the quantum layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length (equal to the number of qubits)
    /// - The quantum circuit is reset to its starting state
    /// - New rotation angles are applied to rebuild the circuit
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _numQubits)
        {
            throw new ArgumentException($"Expected {_numQubits} parameters, but got {parameters.Length}");
        }

        // Reset the quantum circuit to identity
        ResetQuantumCircuit();

        // Set new rotation angles using tensor ctor (no conversion hot path)
        _rotationAngles = new Tensor<T>([parameters.Length], parameters);

        for (int i = 0; i < _numQubits; i++)
        {
            ApplyRotation(i, _rotationAngles[i]);
        }
    }

    /// <summary>
    /// Resets the internal state of the quantum layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the quantum layer, including the cached input from the
    /// forward pass and the angle gradients. This is useful when starting to process a new sequence or
    /// when implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs from previous calculations are cleared
    /// - Angle gradients are reset to zero
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// The quantum circuit itself (with its learned rotation angles) is not reset,
    /// only the temporary state information.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;

        // Reset angle gradients
        _angleGradients = new Tensor<T>([_numQubits]);
        _angleGradients.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes the quantum circuit with an identity matrix and random rotation angles.
    /// </summary>
    /// <remarks>
    /// This private method sets up the initial quantum circuit by first creating an identity matrix
    /// and then applying random rotation angles to each qubit. The rotation angles are the trainable
    /// parameters of the layer.
    /// </remarks>
    private void InitializeQuantumCircuit()
    {
        int dimension = 1 << _numQubits;

        // Initialize quantum circuit as identity matrix
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                _quantumCircuit[i, j] = i == j ?
                    new Complex<T>(NumOps.One, NumOps.Zero) :
                    new Complex<T>(NumOps.Zero, NumOps.Zero);
                _circuitReal[i, j] = _quantumCircuit[i, j].Real;
                _circuitImag[i, j] = _quantumCircuit[i, j].Imaginary;
            }
        }

        // Initialize rotation angles randomly
        for (int i = 0; i < _numQubits; i++)
        {
            _rotationAngles[i] = NumOps.FromDouble(Random.NextDouble() * 2 * Math.PI);
            ApplyRotation(i, _rotationAngles[i]);
        }
    }

    /// <summary>
    /// Applies a rotation operation to a specific qubit in the quantum circuit.
    /// </summary>
    /// <param name="qubit">The index of the qubit to rotate.</param>
    /// <param name="angle">The rotation angle in radians.</param>
    /// <remarks>
    /// This private method applies a rotation operation to a specific qubit in the quantum circuit.
    /// It calculates the rotation matrix elements based on the angle and applies the transformation
    /// to the quantum circuit.
    /// </remarks>
    private void ApplyRotation(int qubit, T angle)
    {
        int dimension = 1 << _numQubits;

        // Calculate rotation parameters
        var halfAngle = NumOps.Divide(angle, NumOps.FromDouble(2.0));
        var cos = MathHelper.Cos(halfAngle);
        var sin = MathHelper.Sin(halfAngle);

        // Create complex values for the rotation
        var cosComplex = new Complex<T>(cos, NumOps.Zero);
        var sinComplex = new Complex<T>(sin, NumOps.Zero);
        var imaginary = new Complex<T>(NumOps.Zero, NumOps.One);
        var negativeImaginary = new Complex<T>(NumOps.Zero, NumOps.Negate(NumOps.One));

        // Create a temporary copy of the circuit for the transformation
        var tempCircuit = _quantumCircuit.Clone();

        for (int i = 0; i < dimension; i++)
        {
            if ((i & (1 << qubit)) == 0)
            {
                int j = i | (1 << qubit);
                for (int k = 0; k < dimension; k++)
                {
                    var temp = tempCircuit[k, i];

                    // Apply rotation matrix
                    _quantumCircuit[k, i] = _complexOps.Add(
                        _complexOps.Multiply(cosComplex, temp),
                        _complexOps.Multiply(negativeImaginary, _complexOps.Multiply(sinComplex, tempCircuit[k, j]))
                    );

                    _quantumCircuit[k, j] = _complexOps.Add(
                        _complexOps.Multiply(imaginary, _complexOps.Multiply(sinComplex, temp)),
                        _complexOps.Multiply(cosComplex, tempCircuit[k, j])
                    );
                }
            }
        }
    }

    /// <summary>
    /// Updates the gradients of the rotation angles based on the output gradient.
    /// </summary>
    /// <param name="gradientState">The gradient of the loss with respect to the layer's output in complex form.</param>
    /// <param name="batchIndex">The index of the current batch item.</param>
    /// <remarks>
    /// This private method calculates the gradients of the rotation angles based on the output gradient.
    /// These gradients are accumulated across all items in a batch and used to update the parameters
    /// during the UpdateParameters method.
    /// </remarks>
    private void UpdateAngleGradients(Tensor<Complex<T>> gradientState, int batchIndex)
    {
        int dimension = 1 << _numQubits;

        for (int qubit = 0; qubit < _numQubits; qubit++)
        {
            T qubitGradient = NumOps.Zero;

            for (int i = 0; i < dimension; i++)
            {
                if ((i & (1 << qubit)) == 0)
                {
                    int j = i | (1 << qubit);

                    // Calculate gradient contribution for this qubit
                    var gradDiff = gradientState[j] * _quantumCircuit[j, i].Conjugate() -
                                   gradientState[i] * _quantumCircuit[i, j].Conjugate();

                    // Extract the real part of the complex number
                    qubitGradient = NumOps.Add(qubitGradient, gradDiff.Real);
                }
            }

            // Accumulate gradients across batches
            _angleGradients[qubit] = NumOps.Add(_angleGradients[qubit], qubitGradient);
        }
    }

    /// <summary>
    /// Resets the quantum circuit to an identity matrix.
    /// </summary>
    /// <remarks>
    /// This private method resets the quantum circuit to an identity matrix, which is the starting
    /// point before applying any rotations. This is used when setting new parameters to ensure
    /// a clean state before applying the new rotations.
    /// </remarks>
    private void ResetQuantumCircuit()
    {
        int dimension = 1 << _numQubits;

        // Reset quantum circuit to identity matrix
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                _quantumCircuit[i, j] = i == j ?
                    new Complex<T>(NumOps.One, NumOps.Zero) :
                    new Complex<T>(NumOps.Zero, NumOps.Zero);
            }
        }
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
        int dimension = 1 << _numQubits;

        // Convert quantum circuit (Complex<T> tensor) to real/imaginary split format for JIT
        // Format: first dimension rows are real, next dimension rows are imaginary [2*dimension, dimension]
        var circuitRealImag = new T[dimension * dimension * 2];
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                var complex = _quantumCircuit[i, j];
                circuitRealImag[i * dimension + j] = complex.Real;                         // Real part
                circuitRealImag[(dimension + i) * dimension + j] = complex.Imaginary;      // Imaginary part
            }
        }
        var circuitTensor = new Tensor<T>(new[] { 2 * dimension, dimension }, new Vector<T>(circuitRealImag));
        var quantumCircuitNode = TensorOperations<T>.Constant(circuitTensor, "QuantumCircuit");

        // Input is real-valued, padded with zeros to dimension and create complex format
        // Padding: add zeros after the input to reach dimension size
        int inputSize = InputShape[0];
        int padAmount = dimension - inputSize;
        int[,] padWidth = new int[1, 2] { { 0, padAmount > 0 ? padAmount : 0 } };
        var paddedInput = padAmount > 0 ? TensorOperations<T>.Pad(input, padWidth) : input;

        // Compute squared norm for normalization: sum(input^2)
        var inputSquared = TensorOperations<T>.Square(paddedInput);
        var sumSquared = TensorOperations<T>.Sum(inputSquared);
        var normFactor = TensorOperations<T>.Sqrt(sumSquared);

        // Normalize input (avoid division by zero by adding small epsilon)
        var epsilonTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(1e-10) }));
        var epsilon = TensorOperations<T>.Constant(epsilonTensor, "Epsilon");
        var safeDenom = TensorOperations<T>.Add(normFactor, epsilon);
        var normalizedInput = TensorOperations<T>.Divide(paddedInput, safeDenom);

        // Create complex state with zero imaginary part: [normalized_input; zeros]
        var zerosData = new T[dimension];
        var zerosTensor = new Tensor<T>(new[] { dimension }, new Vector<T>(zerosData));
        var zeros = TensorOperations<T>.Constant(zerosTensor, "ZerosImag");
        var complexState = TensorOperations<T>.Concat(new List<ComputationNode<T>> { normalizedInput, zeros }, axis: 0);

        // Apply quantum circuit using complex matrix multiplication
        // result_complex = quantumCircuit @ state_complex
        var result = TensorOperations<T>.ComplexMatMul(quantumCircuitNode, complexState, "split");

        // Extract probabilities: |amplitude|^2 = real^2 + imag^2
        // Result is [2*dimension, 1] with first half real, second half imaginary
        var resultReal = TensorOperations<T>.Slice(result, 0, dimension, step: 1, axis: 0);
        var resultImag = TensorOperations<T>.Slice(result, dimension, dimension, step: 1, axis: 0);
        var realSquared = TensorOperations<T>.Square(resultReal);
        var imagSquared = TensorOperations<T>.Square(resultImag);
        var probabilities = TensorOperations<T>.Add(realSquared, imagSquared);

        return probabilities;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> because QuantumLayer uses complex matrix multiplication which is supported
    /// in TensorOperations via ComplexMatMul. The quantum circuit can be compiled to a static
    /// computation graph.
    /// </value>
    public override bool SupportsJitCompilation => true;

}
