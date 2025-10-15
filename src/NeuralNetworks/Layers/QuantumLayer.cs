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
    private Tensor<Complex<T>> _quantumCircuit = default!;
    private Tensor<T>? _lastInput;
    private Vector<T> _rotationAngles = default!;
    private Vector<T> _angleGradients = default!;
    private readonly INumericOperations<Complex<T>> _complexOps = default!;

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
            
        // Initialize parameters
        _rotationAngles = new Vector<T>(_numQubits);
        _angleGradients = new Vector<T>(_numQubits);
            
        // Create quantum circuit as a tensor
        int dimension = 1 << _numQubits;
        _quantumCircuit = new Tensor<Complex<T>>([dimension, dimension]);

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
        _lastInput = input;
        int batchSize = input.Shape[0];
        int dimension = 1 << _numQubits;
        
        // Create output tensor
        var output = new Tensor<T>([batchSize, dimension]);
        
        for (int b = 0; b < batchSize; b++)
        {
            // Convert input to quantum state
            var quantumState = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < Math.Min(input.Shape[1], dimension); i++)
            {
                quantumState[i] = new Complex<T>(input[b, i], NumOps.Zero);
            }
            
            // Normalize the quantum state
            var normFactor = NumOps.Zero;
            for (int i = 0; i < dimension; i++)
            {
                // Calculate magnitude squared manually
                var complex = quantumState[i];
                var magnitudeSquared = NumOps.Add(
                    NumOps.Multiply(complex.Real, complex.Real),
                    NumOps.Multiply(complex.Imaginary, complex.Imaginary)
                );
                normFactor = NumOps.Add(normFactor, magnitudeSquared);
            }
            
            normFactor = NumOps.Sqrt(normFactor);
            if (!NumOps.Equals(normFactor, NumOps.Zero))
            {
                for (int i = 0; i < dimension; i++)
                {
                    quantumState[i] = _complexOps.Divide(quantumState[i], 
                        new Complex<T>(normFactor, NumOps.Zero));
                }
            }

            // Apply quantum circuit
            var result = new Tensor<Complex<T>>([dimension]);
            for (int i = 0; i < dimension; i++)
            {
                result[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
                for (int j = 0; j < dimension; j++)
                {
                    result[i] = _complexOps.Add(result[i], 
                        _complexOps.Multiply(_quantumCircuit[i, j], quantumState[j]));
                }
            }

            // Convert complex amplitudes to probabilities
            for (int i = 0; i < dimension; i++)
            {
                // Calculate magnitude squared manually
                var complex = result[i];
                var magnitudeSquared = NumOps.Add(
                    NumOps.Multiply(complex.Real, complex.Real),
                    NumOps.Multiply(complex.Imaginary, complex.Imaginary)
                );
                output[b, i] = magnitudeSquared;
            }
        }

        return output;
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
    /// 3. Angles are kept within a valid range (0 to 2π)
    /// 4. The quantum circuit is updated with the new angles
    /// 
    /// This is how the quantum layer "learns" from data over time. Smaller learning rates
    /// mean slower but more stable learning, while larger learning rates mean faster but
    /// potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _numQubits; i++)
        {
            // Update rotation angles using gradient descent
            _rotationAngles[i] = NumOps.Subtract(_rotationAngles[i], NumOps.Multiply(learningRate, _angleGradients[i]));

            // Ensure angles stay within [0, 2π]
            _rotationAngles[i] = MathHelper.Modulo(
                NumOps.Add(_rotationAngles[i], NumOps.FromDouble(2 * Math.PI)), 
                NumOps.FromDouble(2 * Math.PI));

            // Apply updated rotation
            ApplyRotation(i, _rotationAngles[i]);
        }

        // Reset angle gradients for the next iteration
        _angleGradients = new Vector<T>(_numQubits);
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
        // Return a copy of the rotation angles
        return (Vector<T>)_rotationAngles.Clone();
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
    
        // Set new rotation angles and apply them
        for (int i = 0; i < _numQubits; i++)
        {
            _rotationAngles[i] = parameters[i];
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
        _angleGradients = new Vector<T>(_numQubits);
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
}