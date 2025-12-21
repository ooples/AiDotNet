namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Quantum Neural Network, which combines quantum computing principles with neural network architecture.
/// </summary>
/// <remarks>
/// <para>
/// A Quantum Neural Network (QNN) is a neural network architecture that leverages quantum computing principles
/// to potentially solve certain problems more efficiently than classical neural networks. It uses quantum bits (qubits)
/// instead of classical bits, allowing it to process information in ways not possible with traditional neural networks.
/// </para>
/// <para><b>For Beginners:</b> A Quantum Neural Network combines ideas from quantum computing with neural networks.
/// 
/// Think of it like upgrading from a regular calculator to a special calculator with new abilities:
/// - Regular neural networks use normal bits (0 or 1)
/// - Quantum neural networks use quantum bits or "qubits" that can be 0, 1, or both at the same time
/// - This "both at the same time" property (called superposition) gives quantum networks special abilities
/// - These networks might solve certain problems much faster than regular neural networks
/// 
/// For example, a quantum neural network might find patterns in complex data or optimize solutions
/// in ways that would be extremely difficult for traditional neural networks.
/// 
/// While the math behind quantum computing is complex, you can think of a quantum neural network
/// as having the potential to explore many possible solutions simultaneously rather than one at a time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class QuantumNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the number of qubits used in the quantum neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of qubits determines the dimensionality of the quantum state space that the network
    /// can operate in. Each additional qubit doubles the size of this state space, allowing the network
    /// to represent more complex quantum states but also increasing computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many quantum bits (qubits) the network uses.
    /// 
    /// The number of qubits is important because:
    /// - Each qubit exponentially increases the computing power
    /// - With n qubits, the network can represent 2^n states simultaneously
    /// - More qubits allow the network to solve more complex problems
    /// - But more qubits also make the network harder to simulate on classical computers
    /// 
    /// For example, with just 10 qubits, a quantum neural network can represent 1,024 states at once.
    /// With 20 qubits, it can represent over 1 million states simultaneously!
    /// </para>
    /// </remarks>
    private int _numQubits;

    private readonly INormalizer<T, Tensor<T>, Tensor<T>> _normalizer;

    /// <summary>
    /// Initializes a new instance of the <see cref="QuantumNeuralNetwork{T}"/> class with the specified architecture and number of qubits.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the QNN.</param>
    /// <param name="numQubits">The number of qubits to use in the quantum neural network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Quantum Neural Network with the specified architecture and number of qubits.
    /// It initializes the network layers based on the architecture, or creates default quantum network layers if
    /// no specific layers are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Quantum Neural Network with its basic components.
    /// 
    /// When creating a new QNN:
    /// - architecture: Defines the overall structure of the neural network
    /// - numQubits: Sets how many quantum bits the network will use
    /// 
    /// The constructor prepares the network by either:
    /// - Using the specific layers provided in the architecture, or
    /// - Creating default layers designed for quantum processing if none are specified
    /// 
    /// This is like setting up a specialized calculator before you start using it for calculations.
    /// </para>
    /// </remarks>

    public QuantumNeuralNetwork(NeuralNetworkArchitecture<T> architecture, int numQubits,
        INormalizer<T, Tensor<T>, Tensor<T>>? normalizer = null, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _numQubits = numQubits;
        _normalizer = normalizer ?? new NoNormalizer<T, Tensor<T>, Tensor<T>>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Quantum Neural Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for quantum
    /// processing is created based on the number of qubits specified during initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for quantum neural networks are created automatically
    /// - The system checks that any custom layers will work properly with quantum computations
    /// 
    /// Layers are like the different processing stages in the neural network.
    /// For a quantum neural network, these layers are designed to work with quantum principles,
    /// allowing the network to take advantage of quantum effects like superposition and entanglement.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultQuantumNetworkLayers(Architecture, _numQubits));
        }
    }

    /// <summary>
    /// Updates the parameters of the quantum neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the quantum neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the quantum neural network makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - It takes a vector of parameter updates and distributes them to the correct layers
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For a quantum neural network, these parameters might control operations like quantum rotations,
    /// entanglement settings, or other quantum-inspired transformations.
    /// 
    /// This process allows the quantum neural network to improve its performance over time
    /// by adjusting how it processes information.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the quantum neural network for the given input.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the quantum neural network, applying quantum operations
    /// simulated on classical hardware.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where the quantum neural network processes input data and makes a prediction.
    /// It simulates quantum operations on a classical computer, giving an approximation of how a true
    /// quantum computer might behave.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Ensure input is correctly shaped
        if (input.Shape != Architecture.GetInputShape())
        {
            throw new ArgumentException("Input shape does not match the expected input shape.");
        }

        // Simulate quantum state preparation
        var quantumState = PrepareQuantumState(input);

        // Apply quantum layers
        foreach (var layer in Layers)
        {
            var realInput = ExtractRealPart(quantumState);
            var realOutput = layer.Forward(realInput);
            quantumState = ConvertToComplexTensor(realOutput);
        }

        // Measure the quantum state to get classical output
        return MeasureQuantumState(quantumState);
    }

    /// <summary>
    /// Trains the quantum neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration, including forward pass, loss calculation,
    /// backward pass, and parameter update using a quantum-inspired optimization technique.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the quantum neural network learns. It processes an input,
    /// compares its prediction to the expected output, and adjusts its internal settings to improve
    /// future predictions. The adjustments are made using techniques inspired by quantum computing.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var prediction = Predict(input);

        // Calculate and set the loss
        LastLoss = CalculateLoss(prediction, expectedOutput);

        // Backward pass (quantum-inspired gradient calculation)
        var gradients = CalculateQuantumGradients(input, expectedOutput);

        // Update parameters
        UpdateQuantumParameters(gradients);
    }

    /// <summary>
    /// Retrieves metadata about the quantum neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the quantum neural network's
    /// structure and configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This provides a summary of the quantum neural network's setup, including
    /// its structure, the number of qubits it uses, and other important details. It's like getting
    /// a blueprint of the network's current state.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.QuantumNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Layers[Layers.Count - 1].GetOutputShape() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "NumberOfQubits", _numQubits }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes quantum neural network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific parameters and state of the quantum neural network to a binary stream.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This saves the current state of the quantum neural network to a file.
    /// It records all the important information about the network so you can reload it later exactly as it is now.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numQubits);
    }

    /// <summary>
    /// Deserializes quantum neural network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific parameters and state of the quantum neural network from a binary stream.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This loads a saved quantum neural network state from a file. It rebuilds the
    /// network exactly as it was when you saved it, including all its learned information and quantum-specific settings.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numQubits = reader.ReadInt32();
    }

    /// <summary>
    /// Calculates the gradients for the quantum neural network layers.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <returns>A list of gradient tensors for each layer.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network, then calculates gradients
    /// during a backward pass, handling the transition between complex and real tensors.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where the quantum neural network figures out how to
    /// improve its predictions. It processes the input, compares the result to the expected
    /// output, and calculates how to adjust each part of the network to get closer to the
    /// correct answer next time.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> CalculateQuantumGradients(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var gradients = new List<Tensor<T>>();
        var quantumState = PrepareQuantumState(input);

        // Forward pass
        var layerOutputs = new List<Tensor<Complex<T>>> { quantumState };
        foreach (var layer in Layers)
        {
            var realInput = ExtractRealPart(quantumState);
            var realOutput = layer.Forward(realInput);
            quantumState = ConvertToComplexTensor(realOutput);
            layerOutputs.Add(quantumState);
        }

        // Backward pass
        var outputGradient = LossFunction.CalculateDerivative(MeasureQuantumState(quantumState).ToVector(), expectedOutput.ToVector());
        var complexOutputGradient = ConvertToComplexTensor(outputGradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            var realOutputGradient = ExtractRealPart(complexOutputGradient);
            var realLayerGradient = Layers[i].Backward(realOutputGradient);
            var complexLayerGradient = ConvertToComplexTensor(realLayerGradient);

            gradients.Insert(0, ClipGradient(realLayerGradient));
            complexOutputGradient = complexLayerGradient;
        }

        return gradients;
    }

    /// <summary>
    /// Prepares a quantum state from a classical input tensor.
    /// </summary>
    /// <param name="input">The classical input tensor.</param>
    /// <returns>A complex tensor representing the quantum state.</returns>
    /// <remarks>
    /// <para>
    /// This method normalizes the input and converts it into a quantum state representation.
    /// Each classical value is transformed into a complex amplitude, where the magnitude is
    /// the square root of the normalized input value, and the phase is set to zero.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like translating regular numbers into a special "quantum language".
    /// It takes your input and turns it into a form that a quantum system can understand and process.
    /// </para>
    /// </remarks>
    private Tensor<Complex<T>> PrepareQuantumState(Tensor<T> input)
    {
        var (normalizedInput, _) = _normalizer.NormalizeInput(input);
        var quantumState = new Tensor<Complex<T>>([normalizedInput.Length]);

        for (int i = 0; i < normalizedInput.Length; i++)
        {
            var amplitude = NumOps.Sqrt(normalizedInput[i]);
            quantumState[i] = new Complex<T>(amplitude, NumOps.Zero);
        }

        return quantumState;
    }

    /// <summary>
    /// Measures the quantum state to produce a classical output.
    /// </summary>
    /// <param name="quantumState">The quantum state to measure.</param>
    /// <returns>A classical tensor representing the measurement outcome.</returns>
    /// <remarks>
    /// <para>
    /// This method simulates the measurement of a quantum state, collapsing the superposition
    /// into classical probabilities. It calculates the magnitude squared of each complex amplitude.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like "asking" the quantum system for an answer we can understand.
    /// It converts the special quantum information back into regular numbers we can use.
    /// </para>
    /// </remarks>
    private Tensor<T> MeasureQuantumState(Tensor<Complex<T>> quantumState)
    {
        var measuredState = new Tensor<T>([quantumState.Length]);

        for (int i = 0; i < quantumState.Length; i++)
        {
            var magnitudeSquared = NumOps.Add(
                NumOps.Multiply(quantumState[i].Real, quantumState[i].Real),
                NumOps.Multiply(quantumState[i].Imaginary, quantumState[i].Imaginary)
            );
            measuredState[i] = magnitudeSquared;
        }

        return measuredState;
    }

    /// <summary>
    /// Updates the parameters of the quantum neural network layers based on calculated gradients.
    /// </summary>
    /// <param name="gradients">A list of gradient tensors for each layer.</param>
    /// <remarks>
    /// <para>
    /// This method applies the calculated gradients to update the parameters of each layer in the network.
    /// It ensures that the number of gradient tensors matches the number of layers before updating.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how the network "learns". It takes the lessons from the training
    /// (in the form of gradients) and uses them to adjust how each part of the network behaves.
    /// </para>
    /// </remarks>
    private void UpdateQuantumParameters(List<Tensor<T>> gradients)
    {
        if (gradients.Count != Layers.Count)
            throw new ArgumentException("Number of gradient tensors must match number of layers.");

        for (int i = 0; i < Layers.Count; i++)
        {
            Layers[i].UpdateParameters(gradients[i].ToVector());
        }
    }

    /// <summary>
    /// Converts a real-valued tensor to a complex-valued tensor.
    /// </summary>
    /// <param name="realTensor">The real-valued tensor to convert.</param>
    /// <returns>A complex-valued tensor with zero imaginary parts.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new complex tensor where each element's real part is set to the
    /// corresponding value from the input tensor, and the imaginary part is set to zero.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like adding an extra dimension to our numbers. It turns
    /// regular numbers into complex numbers, which are needed for quantum calculations.
    /// </para>
    /// </remarks>
    private Tensor<Complex<T>> ConvertToComplexTensor(Tensor<T> realTensor)
    {
        var complexTensor = new Tensor<Complex<T>>(realTensor.Shape);
        for (int i = 0; i < realTensor.Length; i++)
        {
            complexTensor[i] = new Complex<T>(realTensor[i], NumOps.Zero);
        }

        return complexTensor;
    }

    /// <summary>
    /// Calculates the loss between the predicted output and the expected output.
    /// </summary>
    /// <param name="prediction">The predicted output tensor.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <returns>The calculated loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the specified loss function to calculate how far off the network's
    /// prediction is from the expected output.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This measures how "wrong" the network's guess is. A lower number
    /// means the network is doing better at predicting the correct output.
    /// </para>
    /// </remarks>
    private T CalculateLoss(Tensor<T> prediction, Tensor<T> expectedOutput)
    {
        return LossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
    }

    /// <summary>
    /// Extracts the real part from a complex-valued tensor.
    /// </summary>
    /// <param name="complexTensor">The complex-valued tensor.</param>
    /// <returns>A real-valued tensor containing the real parts of the complex tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new real-valued tensor by extracting the real part of each complex number
    /// in the input tensor.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like focusing on just one part of our special quantum numbers.
    /// We're taking the "regular" part and ignoring the "imaginary" part for certain calculations.
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractRealPart(Tensor<Complex<T>> complexTensor)
    {
        var realTensor = new Tensor<T>(complexTensor.Shape);
        for (int i = 0; i < complexTensor.Length; i++)
        {
            realTensor[i] = complexTensor[i].Real;
        }

        return realTensor;
    }

    /// <summary>
    /// Converts a real-valued vector to a complex-valued tensor.
    /// </summary>
    /// <param name="realVector">The real-valued vector to convert.</param>
    /// <returns>A complex-valued tensor with zero imaginary parts.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new complex tensor where each element's real part is set to the
    /// corresponding value from the input vector, and the imaginary part is set to zero.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is similar to the other conversion method, but it works with
    /// a different type of input (a vector instead of a tensor). It's another way of preparing
    /// our numbers for quantum-style calculations.
    /// </para>
    /// </remarks>
    private Tensor<Complex<T>> ConvertToComplexTensor(Vector<T> realVector)
    {
        var complexTensor = new Tensor<Complex<T>>([realVector.Length]);
        for (int i = 0; i < realVector.Length; i++)
        {
            complexTensor[i] = new Complex<T>(realVector[i], NumOps.Zero);
        }

        return complexTensor;
    }

    /// <summary>
    /// Creates a new instance of the quantum neural network with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="QuantumNeuralNetwork{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new quantum neural network that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// The new instance will have the same architecture, number of qubits, normalizer, and loss function
    /// as the original, but will not share parameter values unless they are explicitly copied after creation.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your quantum neural network that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar quantum neural networks
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its quantum-specific settings
    /// 
    /// Note that while the settings are copied, the learned parameters are not automatically
    /// transferred, so the new instance will need training or parameter copying to match
    /// the performance of the original.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a new instance with the cloned architecture and same configuration
        return new QuantumNeuralNetwork<T>(
            Architecture,
            _numQubits,
            _normalizer,
            LossFunction
        );
    }
}
