using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

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
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, SupportsBackpropagation = false, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 4, 2")]
public partial class QuantumLayer<T> : LayerBase<T>
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

    /// <summary>
    /// Cached result amplitudes from Forward for use in Backward.
    /// Shape: [batch, dimension] for real and imaginary parts.
    /// </summary>
    private Tensor<T>? _lastResultReal;
    private Tensor<T>? _lastResultImag;
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
    public override int ParameterCount => _rotationAngles.Length;
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

        RegisterTrainableParameter(_circuitReal, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_circuitImag, PersistentTensorRole.Weights);

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
        _originalInputShape = input.Shape.ToArray();
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
            realState = TensorAllocator.Rent<T>([batchSize, dimension]);
            Engine.TensorSetSlice(realState, processInput, [0, 0]);
        }
        else
        {
            realState = Engine.TensorSlice(processInput, [0, 0], [batchSize, dimension]);
        }

        var imagState = new Tensor<T>(realState.Shape.ToArray());

        // Normalize each batch item: divide by sqrt(sum(|state|^2) + eps)
        var magnitudeSquared = Engine.ComplexMagnitudeSquared(realState, imagState);
        var normPerBatch = Engine.ReduceSum(magnitudeSquared, [1], keepDims: true);
        var epsilonTensor = new Tensor<T>(normPerBatch.Shape.ToArray());
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

        // Cache amplitudes for Backward (transpose to [batch, dimension])
        _lastResultReal = Engine.TensorTranspose(resultRealT);
        _lastResultImag = Engine.TensorTranspose(resultImagT);

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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
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

        // Track all allocated buffers for exception safety
        IGpuBuffer? circuitRealBuffer = null;
        IGpuBuffer? circuitImagBuffer = null;
        IGpuBuffer? stateRealBuffer = null;
        IGpuBuffer? stateImagBuffer = null;
        IGpuBuffer? squaredBuffer = null;
        IGpuBuffer? normSqBuffer = null;
        IGpuBuffer? normSqClampedBuffer = null;
        IGpuBuffer? normBuffer = null;
        IGpuBuffer? invNormBuffer = null;
        IGpuBuffer? resultRealBuffer = null;
        IGpuBuffer? resultImagBuffer = null;
        IGpuBuffer? probabilitiesBuffer = null;

        try
        {
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
            var circuitReal = backend.AllocateBuffer(circuitRealFlat);
            circuitRealBuffer = circuitReal;
            var circuitImag = backend.AllocateBuffer(circuitImagFlat);
            circuitImagBuffer = circuitImag;

            // GPU-only state initialization: pad input and L2 normalize per batch
            // Allocate state buffers on GPU
            var stateReal = backend.AllocateBuffer(batchSize * dimension);
            stateRealBuffer = stateReal;
            var stateImag = backend.AllocateBuffer(batchSize * dimension);
            stateImagBuffer = stateImag;

            // Zero the buffers (padding with zeros)
            backend.Fill(stateReal, 0.0f, batchSize * dimension);
            backend.Fill(stateImag, 0.0f, batchSize * dimension);

            // Copy input data with padding using strided copy
            int copyWidth = Math.Min(inputDim, dimension);
            backend.Copy2DStrided(input.Buffer, stateReal, batchSize, copyWidth, dimension, 0);

            // L2 normalize each batch element on GPU
            // Step 1: Square the values
            var squared = backend.AllocateBuffer(batchSize * dimension);
            squaredBuffer = squared;
            backend.Multiply(stateReal, stateReal, squared, batchSize * dimension);

            // Step 2: Sum per batch to get sum of squares
            var normSq = backend.AllocateBuffer(batchSize);
            normSqBuffer = normSq;
            backend.SumAxis(squared, normSq, batchSize, dimension);

            // Step 3: Clamp to avoid division by zero (add epsilon)
            var normSqClamped = backend.AllocateBuffer(batchSize);
            normSqClampedBuffer = normSqClamped;
            backend.Clamp(normSq, normSqClamped, 1e-10f, float.MaxValue, batchSize);

            // Step 4: Sqrt to get L2 norm
            var norm = backend.AllocateBuffer(batchSize);
            normBuffer = norm;
            backend.Sqrt(normSqClamped, norm, batchSize);

            // Step 5: Reciprocal to get 1/norm
            var invNorm = backend.AllocateBuffer(batchSize);
            invNormBuffer = invNorm;
            backend.Reciprocal(norm, invNorm, batchSize);

            // Step 6: Broadcast multiply to normalize each row
            backend.BroadcastMultiplyFirstAxis(stateReal, invNorm, stateReal, batchSize, dimension);

            // Allocate output state buffers
            var resultReal = backend.AllocateBuffer(batchSize * dimension);
            resultRealBuffer = resultReal;
            var resultImag = backend.AllocateBuffer(batchSize * dimension);
            resultImagBuffer = resultImag;

            // Apply quantum circuit: complex matrix multiplication
            backend.ComplexMatVec(
                circuitReal, circuitImag,
                stateReal, stateImag,
                resultReal, resultImag,
                batchSize, dimension);

            // Compute probabilities: |amplitude|^2
            var probabilities = backend.AllocateBuffer(batchSize * dimension);
            probabilitiesBuffer = probabilities;
            backend.QuantumMeasurement(resultReal, resultImag, probabilities, batchSize, dimension);

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

            // Create result (ownership of probabilities buffer transfers)
            var result = GpuTensorHelper.UploadToGpu<T>(backend, probabilities, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
            probabilitiesBuffer = null; // Prevent disposal in finally block since ownership transferred

            return result;
        }
        finally
        {
            // Dispose all intermediate buffers (not the output which was transferred)
            circuitRealBuffer?.Dispose();
            circuitImagBuffer?.Dispose();
            stateRealBuffer?.Dispose();
            stateImagBuffer?.Dispose();
            squaredBuffer?.Dispose();
            normSqBuffer?.Dispose();
            normSqClampedBuffer?.Dispose();
            normBuffer?.Dispose();
            invNormBuffer?.Dispose();
            resultRealBuffer?.Dispose();
            resultImagBuffer?.Dispose();
            probabilitiesBuffer?.Dispose(); // Only disposed on exception (null on success)
        }
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
    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["NumQubits"] = _numQubits.ToString();
        return metadata;
    }

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

    public override Vector<T> GetParameterGradients()
    {
        return new Vector<T>(_angleGradients.ToArray());
    }

    public override void ClearGradients()
    {
        _angleGradients = new Tensor<T>([_numQubits]);
        _angleGradients.Fill(NumOps.Zero);
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

        // Rebuild circuit tensors from updated _quantumCircuit for Forward to use
        int dim = 1 << _numQubits;
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                _circuitReal[i, j] = _quantumCircuit[i, j].Real;
                _circuitImag[i, j] = _quantumCircuit[i, j].Imaginary;
            }
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
        _lastResultReal = null;
        _lastResultImag = null;

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

        // Update circuit tensors to match the rotated _quantumCircuit
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                _circuitReal[i, j] = _quantumCircuit[i, j].Real;
                _circuitImag[i, j] = _quantumCircuit[i, j].Imaginary;
            }
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

}
