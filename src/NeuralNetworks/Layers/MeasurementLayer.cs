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
    /// (|z|� = real� + imag�) and normalizing the results to ensure they sum to 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts quantum amplitudes into classical probabilities.
    /// 
    /// During the forward pass:
    /// - The layer receives complex-valued quantum amplitudes
    /// - For each amplitude, it calculates |z|� = real� + imag� (the squared magnitude)
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
        _lastInput = input;
        // Assume input is a complex-valued tensor representing quantum states
        var probabilities = new T[input.Shape[0]];
    
        // Get numeric operations for complex numbers
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        for (int i = 0; i < input.Shape[0]; i++)
        {
            // Get the complex value from the input tensor
            var complexValue = Tensor<T>.GetComplex(input, i);
        
            // Calculate |z|� = real� + imag�
            var realSquared = NumOps.Multiply(complexValue.Real, complexValue.Real);
            var imagSquared = NumOps.Multiply(complexValue.Imaginary, complexValue.Imaginary);
            probabilities[i] = NumOps.Add(realSquared, imagSquared);
        }
        // Normalize probabilities
        var sum = NumOps.Zero;
        for (int i = 0; i < probabilities.Length; i++)
        {
            sum = NumOps.Add(sum, probabilities[i]);
        }
        for (int i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] = NumOps.Divide(probabilities[i], sum);
        }
        // Create a new tensor with the calculated probabilities
        _lastOutput = new Tensor<T>([input.Shape[0]], new Vector<T>(probabilities));
        return _lastOutput;
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
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }
        // The gradient of measurement with respect to input amplitudes
        var inputGradientData = new Vector<T>(_lastInput.Length);
        for (int i = 0; i < _lastInput.Shape[0]; i++)
        {
            // Get the complex value from the input tensor
            var complexValue = Tensor<T>.GetComplex(_lastInput, i);
            var prob = _lastOutput[i];
            // Gradient of probability with respect to real and imaginary parts
            // dP/dReal = 2 * real / prob
            // dP/dImag = 2 * imag / prob
            var two = NumOps.FromDouble(2.0);
            var dProbdReal = NumOps.Divide(NumOps.Multiply(two, complexValue.Real), prob);
            var dProbdImag = NumOps.Divide(NumOps.Multiply(two, complexValue.Imaginary), prob);
            // Combine gradients
            var gradientValue = NumOps.Multiply(outputGradient[i], 
                NumOps.Add(dProbdReal, dProbdImag));
        
            inputGradientData[i] = gradientValue;
        }
        // Create a new tensor with the calculated gradients
        return new Tensor<T>(_lastInput.Shape, inputGradientData);
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
    }
}