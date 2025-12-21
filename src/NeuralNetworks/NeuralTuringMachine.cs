namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Neural Turing Machine, which is a neural network architecture that combines a neural network with external memory.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// A Neural Turing Machine (NTM) extends traditional neural networks by adding an external memory component that
/// the network can read from and write to. This allows the network to store and retrieve information over long
/// sequences, making it particularly effective for tasks requiring complex memory operations.
/// </para>
/// <para><b>For Beginners:</b> A Neural Turing Machine is like a neural network with a "notebook" that it can write to and read from.
/// 
/// Think of it like a student solving a math problem:
/// - The student (neural network) can process information directly
/// - But for complex problems, the student needs to write down intermediate steps in a notebook (external memory)
/// - The student can later refer back to these notes when needed
/// 
/// This memory capability helps the network:
/// - Remember information over long periods
/// - Store and retrieve specific pieces of data
/// - Learn more complex patterns that require step-by-step reasoning
/// 
/// For example, a standard neural network might struggle to add two long numbers, but an NTM can learn to write down 
/// partial results and carry digits, similar to how humans solve addition problems.
/// </para>
/// </remarks>
public class NeuralTuringMachine<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (memory usage regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Memory usage regularization prevents memory addressing from becoming too diffuse or collapsing.
    /// This encourages the NTM to learn focused, interpretable memory access patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the NTM use its memory notebook effectively.
    ///
    /// Memory usage regularization ensures:
    /// - Read/write operations focus on relevant memory locations
    /// - Memory access doesn't spread too thin
    /// - Memory operations are interpretable and efficient
    ///
    /// This is like encouraging a student to:
    /// - Write clearly in specific sections of the notebook
    /// - Not scribble all over every page
    /// - Use the notebook in an organized, focused way
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the memory usage auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much memory usage regularization contributes to the total loss.
    /// Typical values range from 0.001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much we encourage focused memory access.
    ///
    /// Common values:
    /// - 0.005 (default): Balanced memory regularization
    /// - 0.001-0.003: Light regularization
    /// - 0.008-0.01: Strong regularization
    ///
    /// Higher values encourage sharper, more focused memory usage.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastMemoryUsageLoss;

    /// <summary>
    /// The size of the external memory matrix (number of memory locations).
    /// </summary>
    private int _memorySize;

    /// <summary>
    /// The size of each memory vector (the amount of information stored at each memory location).
    /// </summary>
    private int _memoryVectorSize;

    /// <summary>
    /// The size of the controller network that manages memory operations.
    /// </summary>
    private int _controllerSize;

    /// <summary>
    /// The external memory matrices used by the Neural Turing Machine, one per batch element.
    /// </summary>
    private List<Matrix<T>> _memories;

    /// <summary>
    /// The current reading weights for each batch element.
    /// </summary>
    private List<Vector<T>> _readWeights;

    /// <summary>
    /// The current writing weights for each batch element.
    /// </summary>
    private List<Vector<T>> _writeWeights;

    /// <summary>
    /// Indicates whether the network is in training mode.
    /// </summary>
    private bool _isTraining;

    /// <summary>
    /// The activation function to apply to content-based addressing similarity scores.
    /// </summary>
    public IActivationFunction<T>? ContentAddressingActivation { get; }

    /// <summary>
    /// The activation function to apply to interpolation gates.
    /// </summary>
    public IActivationFunction<T>? GateActivation { get; }

    /// <summary>
    /// The activation function to apply to the final output.
    /// </summary>
    public IActivationFunction<T>? OutputActivation { get; }

    /// <summary>
    /// The activation function to apply to content-based addressing similarity scores.
    /// </summary>
    public IVectorActivationFunction<T>? ContentAddressingVectorActivation { get; }

    /// <summary>
    /// The activation function to apply to interpolation gates.
    /// </summary>
    public IVectorActivationFunction<T>? GateVectorActivation { get; }

    /// <summary>
    /// The activation function to apply to the final output.
    /// </summary>
    public IVectorActivationFunction<T>? OutputVectorActivation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralTuringMachine{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the NTM.</param>
    /// <param name="memorySize">The number of memory locations (rows in the memory matrix).</param>
    /// <param name="memoryVectorSize">The size of each memory vector (columns in the memory matrix).</param>
    /// <param name="controllerSize">The size of the controller network that manages memory operations.</param>
    /// <param name="lossFunction">The loss function to use for training.</param>
    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralTuringMachine{T}"/> class with customizable activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the NTM.</param>
    /// <param name="memorySize">The number of memory locations (rows in the memory matrix).</param>
    /// <param name="memoryVectorSize">The size of each memory vector (columns in the memory matrix).</param>
    /// <param name="controllerSize">The size of the controller network that manages memory operations.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, a default will be used based on the task type.</param>
    /// <param name="contentAddressingActivation">The activation function to apply to content-based addressing. If null, softmax will be used.</param>
    /// <param name="gateActivation">The activation function to apply to interpolation gates. If null, sigmoid will be used.</param>
    /// <param name="outputActivation">The activation function to apply to the final output. If null, a default based on task type will be used.</param>
    public NeuralTuringMachine(
        NeuralNetworkArchitecture<T> architecture,
        int memorySize,
        int memoryVectorSize,
        int controllerSize,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? contentAddressingActivation = null,
        IActivationFunction<T>? gateActivation = null,
        IActivationFunction<T>? outputActivation = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        if (memorySize <= 0) throw new ArgumentOutOfRangeException(nameof(memorySize), "Memory size must be positive");
        if (memoryVectorSize <= 0) throw new ArgumentOutOfRangeException(nameof(memoryVectorSize), "Memory vector size must be positive");
        if (controllerSize <= 0) throw new ArgumentOutOfRangeException(nameof(controllerSize), "Controller size must be positive");

        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastMemoryUsageLoss = NumOps.Zero;

        _memorySize = memorySize;
        _memoryVectorSize = memoryVectorSize;
        _controllerSize = controllerSize;

        // Set activation functions (or defaults)
        ContentAddressingActivation = contentAddressingActivation ?? new SoftmaxActivation<T>();
        GateActivation = gateActivation ?? new SigmoidActivation<T>();
        OutputActivation = outputActivation ?? NeuralNetworkHelper<T>.GetDefaultActivationFunction(architecture.TaskType);

        _memories = new List<Matrix<T>>();
        _readWeights = new List<Vector<T>>();
        _writeWeights = new List<Vector<T>>();

        // Initialize with default memory and weights
        InitializeDefaultMemoryAndWeights();
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralTuringMachine{T}"/> class.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the NTM.</param>
    /// <param name="memorySize">The number of memory locations (rows in the memory matrix).</param>
    /// <param name="memoryVectorSize">The size of each memory vector (columns in the memory matrix).</param>
    /// <param name="controllerSize">The size of the controller network that manages memory operations.</param>
    /// <param name="lossFunction">The loss function to use for training.</param>
    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralTuringMachine{T}"/> class with customizable activation functions.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the NTM.</param>
    /// <param name="memorySize">The number of memory locations (rows in the memory matrix).</param>
    /// <param name="memoryVectorSize">The size of each memory vector (columns in the memory matrix).</param>
    /// <param name="controllerSize">The size of the controller network that manages memory operations.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, a default will be used based on the task type.</param>
    /// <param name="contentAddressingActivation">The activation function to apply to content-based addressing. If null, softmax will be used.</param>
    /// <param name="gateActivation">The activation function to apply to interpolation gates. If null, sigmoid will be used.</param>
    /// <param name="outputActivation">The activation function to apply to the final output. If null, a default based on task type will be used.</param>
    public NeuralTuringMachine(
        NeuralNetworkArchitecture<T> architecture,
        int memorySize,
        int memoryVectorSize,
        int controllerSize,
        ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? contentAddressingActivation = null,
        IVectorActivationFunction<T>? gateActivation = null,
        IVectorActivationFunction<T>? outputActivation = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        if (memorySize <= 0) throw new ArgumentOutOfRangeException(nameof(memorySize), "Memory size must be positive");
        if (memoryVectorSize <= 0) throw new ArgumentOutOfRangeException(nameof(memoryVectorSize), "Memory vector size must be positive");
        if (controllerSize <= 0) throw new ArgumentOutOfRangeException(nameof(controllerSize), "Controller size must be positive");

        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastMemoryUsageLoss = NumOps.Zero;

        _memorySize = memorySize;
        _memoryVectorSize = memoryVectorSize;
        _controllerSize = controllerSize;

        // Set activation functions (or defaults)
        ContentAddressingVectorActivation = contentAddressingActivation ?? new SoftmaxActivation<T>();
        GateVectorActivation = gateActivation ?? new SigmoidActivation<T>();
        OutputVectorActivation = outputActivation ?? NeuralNetworkHelper<T>.GetDefaultVectorActivationFunction(architecture.TaskType);

        _memories = new List<Matrix<T>>();
        _readWeights = new List<Vector<T>>();
        _writeWeights = new List<Vector<T>>();

        // Initialize with default memory and weights
        InitializeDefaultMemoryAndWeights();
        InitializeLayers();
    }

    /// <summary>
    /// Initializes default memory and attention weights.
    /// </summary>
    private void InitializeDefaultMemoryAndWeights()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T uniformWeight = numOps.Divide(numOps.One, numOps.FromDouble(_memorySize));

        // Create a default memory
        var memory = new Matrix<T>(_memorySize, _memoryVectorSize);
        _memories.Add(memory);

        // Create initial read/write weights with uniform distribution
        var readWeight = new Vector<T>(_memorySize);
        var writeWeight = new Vector<T>(_memorySize);

        for (int i = 0; i < _memorySize; i++)
        {
            readWeight[i] = uniformWeight;
            writeWeight[i] = uniformWeight;
        }

        _readWeights.Add(readWeight);
        _writeWeights.Add(writeWeight);

        InitializeMemory();
    }

    /// <summary>
    /// Initializes the memory matrices with small random values.
    /// </summary>
    private void InitializeMemory()
    {
        for (int m = 0; m < _memories.Count; m++)
        {
            for (int i = 0; i < _memorySize; i++)
            {
                for (int j = 0; j < _memoryVectorSize; j++)
                {
                    // Initialize with values from normal distribution for better training stability
                    _memories[m][i, j] = MathHelper.GetNormalRandom(NumOps.Zero, NumOps.FromDouble(0.1));
                }
            }
        }
    }

    /// <summary>
    /// Computes the auxiliary loss for memory usage regularization.
    /// </summary>
    /// <returns>The computed memory usage auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes entropy-based regularization for memory read/write addressing.
    /// It encourages focused, sharp memory access patterns while preventing diffuse addressing.
    /// Formula: L = -Σ H(addressing_weights) where H is entropy
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how focused the NTM's memory usage is.
    ///
    /// Memory usage regularization works by:
    /// 1. Measuring entropy of read/write addressing weights
    /// 2. Lower entropy means more focused, organized memory usage
    /// 3. Higher entropy means scattered, disorganized access
    /// 4. We minimize negative entropy to encourage focused access
    ///
    /// This helps because:
    /// - Focused memory access is more interpretable
    /// - Sharp addressing improves efficiency
    /// - Prevents wasting computation on irrelevant locations
    /// - Encourages the NTM to use memory like an organized notebook
    ///
    /// The auxiliary loss is added to the main task loss during training.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            _lastMemoryUsageLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute negative entropy over read and write addressing weights
        // to encourage focused, sharp memory access patterns
        T totalNegativeEntropy = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);  // For numerical stability
        T oneMinusEpsilon = NumOps.Subtract(NumOps.One, epsilon);

        // Compute negative entropy for read weights
        foreach (var readWeight in _readWeights)
        {
            T entropy = NumOps.Zero;
            for (int i = 0; i < readWeight.Length; i++)
            {
                T p = readWeight[i];
                // Entropy: H = -Σ(p * log(p))
                // Clamp p to [epsilon, 1-epsilon] to avoid log(0) and log(>1)
                T pClamped = MathHelper.Clamp(p, epsilon, oneMinusEpsilon);
                T logP = NumOps.Log(pClamped);
                T pLogP = NumOps.Multiply(pClamped, logP);
                entropy = NumOps.Add(entropy, pLogP);
            }
            // Negative entropy (we want to minimize this, encouraging sharp peaks)
            totalNegativeEntropy = NumOps.Subtract(totalNegativeEntropy, entropy);
        }

        // Compute negative entropy for write weights
        foreach (var writeWeight in _writeWeights)
        {
            T entropy = NumOps.Zero;
            for (int i = 0; i < writeWeight.Length; i++)
            {
                T p = writeWeight[i];
                // Clamp p to [epsilon, 1-epsilon] to avoid log(0) and log(>1)
                T pClamped = MathHelper.Clamp(p, epsilon, oneMinusEpsilon);
                T logP = NumOps.Log(pClamped);
                T pLogP = NumOps.Multiply(pClamped, logP);
                entropy = NumOps.Add(entropy, pLogP);
            }
            totalNegativeEntropy = NumOps.Subtract(totalNegativeEntropy, entropy);
        }

        _lastMemoryUsageLoss = totalNegativeEntropy;
        return _lastMemoryUsageLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the memory usage auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about memory usage regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about memory usage regularization, including
    /// addressing entropy, memory configuration, and regularization parameters.
    /// This information is useful for monitoring memory access patterns and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how the NTM uses its memory.
    ///
    /// The diagnostics include:
    /// - Total memory usage loss (how focused memory access is)
    /// - Weight applied to the regularization
    /// - Memory size (number of memory locations)
    /// - Memory vector size (size of each location)
    /// - Whether memory usage regularization is enabled
    ///
    /// This helps you:
    /// - Monitor if memory addressing is focused or scattered
    /// - Debug issues with memory access patterns
    /// - Understand the impact of regularization on memory efficiency
    ///
    /// You can use this information to adjust regularization weights for better memory utilization.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalMemoryUsageLoss", _lastMemoryUsageLoss?.ToString() ?? "0" },
            { "MemoryUsageWeight", AuxiliaryLossWeight?.ToString() ?? "0.005" },
            { "UseMemoryUsageRegularization", UseAuxiliaryLoss.ToString() },
            { "MemorySize", _memorySize.ToString() },
            { "MemoryVectorSize", _memoryVectorSize.ToString() },
            { "BatchMemoryCount", _memories.Count.ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture.
    /// </summary>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNTMLayers(Architecture, _memorySize, _memoryVectorSize, _controllerSize));
        }
    }

    /// <summary>
    /// Sets up memories and attention weights for the given batch size.
    /// </summary>
    /// <param name="batchSize">The batch size to set up for.</param>
    private void SetupBatchMemories(int batchSize)
    {
        T uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_memorySize));

        // Ensure we have the right number of memory matrices
        if (_memories.Count < batchSize)
        {
            // Add new memories for additional batch elements
            int additionalMemories = batchSize - _memories.Count;
            for (int i = 0; i < additionalMemories; i++)
            {
                // Create new memory matrix
                var newMemory = new Matrix<T>(_memorySize, _memoryVectorSize);

                // Initialize with the same pattern as the first memory
                for (int r = 0; r < _memorySize; r++)
                {
                    for (int c = 0; c < _memoryVectorSize; c++)
                    {
                        newMemory[r, c] = _memories[0][r, c];
                    }
                }

                _memories.Add(newMemory);

                // Add new read/write weights
                var readWeight = new Vector<T>(_memorySize);
                var writeWeight = new Vector<T>(_memorySize);

                for (int j = 0; j < _memorySize; j++)
                {
                    readWeight[j] = uniformWeight;
                    writeWeight[j] = uniformWeight;
                }

                _readWeights.Add(readWeight);
                _writeWeights.Add(writeWeight);
            }
        }
        else if (_memories.Count > batchSize)
        {
            // Keep only the needed memories
            _memories = _memories.GetRange(0, batchSize);
            _readWeights = _readWeights.GetRange(0, batchSize);
            _writeWeights = _writeWeights.GetRange(0, batchSize);
        }
    }

    /// <summary>
    /// Performs a forward pass through the Neural Turing Machine.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Get batch size and sequence length from input shape
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape.Length > 1 ? input.Shape[1] : 1;

        // Setup memories for this batch
        SetupBatchMemories(batchSize);

        var outputs = new List<Tensor<T>>();

        // Process each time step
        for (int t = 0; t < sequenceLength; t++)
        {
            // Extract current input
            Tensor<T> currentInput = ExtractTimeStepInput(input, t, sequenceLength);

            // Process through controller
            var controllerOutput = ProcessController(currentInput);

            // Generate parameters for memory operations
            var readParams = GenerateReadParameters(controllerOutput);
            var writeParams = GenerateWriteParameters(controllerOutput);

            // Update attention mechanisms
            UpdateAttentionWeights(readParams, writeParams);

            // Perform memory operations
            var readResults = ReadFromMemories();
            WriteToMemories(writeParams);

            // Generate output
            var output = GenerateOutput(controllerOutput, readResults);
            outputs.Add(output);
        }

        // Combine outputs based on sequence length
        return sequenceLength > 1
            ? CombineSequenceOutputs(outputs)
            : outputs[0];
    }

    /// <summary>
    /// Extracts input for a specific time step from the input tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="timeStep">The time step to extract.</param>
    /// <param name="sequenceLength">The total sequence length.</param>
    /// <returns>The input tensor for the specified time step.</returns>
    private Tensor<T> ExtractTimeStepInput(Tensor<T> input, int timeStep, int sequenceLength)
    {
        if (sequenceLength <= 1)
        {
            return input;
        }

        int batchSize = input.Shape[0];
        int featureSize = input.Shape[2];

        var result = new Tensor<T>(new int[] { batchSize, featureSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < featureSize; f++)
            {
                result[b, f] = input[b, timeStep, f];
            }
        }

        return result;
    }

    /// <summary>
    /// Combines sequence outputs into a single tensor.
    /// </summary>
    /// <param name="outputs">The list of output tensors.</param>
    /// <returns>A combined tensor of all outputs.</returns>
    private Tensor<T> CombineSequenceOutputs(List<Tensor<T>> outputs)
    {
        int batchSize = outputs[0].Shape[0];
        int sequenceLength = outputs.Count;
        int outputSize = outputs[0].Shape[1];

        var combined = new Tensor<T>(new int[] { batchSize, sequenceLength, outputSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < sequenceLength; t++)
            {
                for (int o = 0; o < outputSize; o++)
                {
                    combined[b, t, o] = outputs[t][b, o];
                }
            }
        }

        return combined;
    }

    /// <summary>
    /// Processes input through the controller network.
    /// </summary>
    /// <param name="input">The current input tensor.</param>
    /// <returns>The controller output.</returns>
    private Tensor<T> ProcessController(Tensor<T> input)
    {
        int batchSize = input.Shape[0];

        // Read from memories based on previous weights
        var readResults = ReadFromMemories();

        // Combine input with read results
        var combined = new Tensor<T>(new int[] { batchSize, input.Shape[1] + readResults.Shape[1] });
        for (int b = 0; b < batchSize; b++)
        {
            // Copy input values
            for (int i = 0; i < input.Shape[1]; i++)
            {
                combined[b, i] = input[b, i];
            }

            // Copy read results
            for (int i = 0; i < readResults.Shape[1]; i++)
            {
                combined[b, input.Shape[1] + i] = readResults[b, i];
            }
        }

        // Process through controller layers (first half of layers)
        var current = combined;
        for (int i = 0; i < Layers.Count / 2; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Generates parameters for memory reading from controller output.
    /// </summary>
    /// <param name="controllerOutput">The controller output state.</param>
    /// <returns>A tensor containing read operation parameters.</returns>
    private Tensor<T> GenerateReadParameters(Tensor<T> controllerOutput)
    {
        int batchSize = controllerOutput.Shape[0];
        int controllerOutputSize = controllerOutput.Shape[1];

        // Use first quarter of controller output for read parameters
        int readParamSize = controllerOutputSize / 4;
        var readParams = new Tensor<T>(new int[] { batchSize, readParamSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < readParamSize; i++)
            {
                readParams[b, i] = controllerOutput[b, i];
            }
        }

        return readParams;
    }

    /// <summary>
    /// Generates parameters for memory writing from controller output.
    /// </summary>
    /// <param name="controllerOutput">The controller output state.</param>
    /// <returns>A tensor containing write operation parameters.</returns>
    private Tensor<T> GenerateWriteParameters(Tensor<T> controllerOutput)
    {
        int batchSize = controllerOutput.Shape[0];
        int controllerOutputSize = controllerOutput.Shape[1];

        // Use second quarter of controller output for write parameters
        int writeParamStart = controllerOutputSize / 4;
        int writeParamSize = controllerOutputSize / 4;

        var writeParams = new Tensor<T>(new int[] { batchSize, writeParamSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < writeParamSize; i++)
            {
                writeParams[b, i] = controllerOutput[b, writeParamStart + i];
            }
        }

        return writeParams;
    }

    /// <summary>
    /// Updates attention weights for both reading and writing operations.
    /// </summary>
    /// <param name="readParams">The parameters for read operations.</param>
    /// <param name="writeParams">The parameters for write operations.</param>
    private void UpdateAttentionWeights(Tensor<T> readParams, Tensor<T> writeParams)
    {
        int batchSize = readParams.Shape[0];

        for (int b = 0; b < batchSize; b++)
        {
            // Extract parameters for this batch
            var readVector = ExtractVector(readParams, b);
            var writeVector = ExtractVector(writeParams, b);

            // Update read weights using content-based and location-based addressing
            _readWeights[b] = ComputeAttentionWeights(_readWeights[b], readVector, _memories[b]);

            // Update write weights using content-based and location-based addressing
            _writeWeights[b] = ComputeAttentionWeights(_writeWeights[b], writeVector, _memories[b]);
        }
    }

    /// <summary>
    /// Extracts a vector from a tensor for a specific batch element.
    /// </summary>
    /// <param name="tensor">The tensor to extract from.</param>
    /// <param name="batchIndex">The batch index to extract.</param>
    /// <returns>A vector containing the data for the specified batch element.</returns>
    private Vector<T> ExtractVector(Tensor<T> tensor, int batchIndex)
    {
        int vectorSize = tensor.Shape[1];
        var vector = new Vector<T>(vectorSize);

        for (int i = 0; i < vectorSize; i++)
        {
            vector[i] = tensor[batchIndex, i];
        }

        return vector;
    }

    /// <summary>
    /// Computes attention weights using content-based and location-based addressing.
    /// </summary>
    /// <param name="previousWeights">The previous attention weights.</param>
    /// <param name="parameters">The parameters for attention computation.</param>
    /// <param name="memory">The memory to address.</param>
    /// <returns>The updated attention weights.</returns>
    private Vector<T> ComputeAttentionWeights(Vector<T> previousWeights, Vector<T> parameters, Matrix<T> memory)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Determine how many parameters we need for each part of the addressing mechanism
        int parameterCount = parameters.Length;
        int keyVectorSize = Math.Min(_memoryVectorSize, parameterCount / 4);

        // Extract key vector (for content addressing)
        var keyVector = parameters.Subvector(0, keyVectorSize);

        // Extract key strength (focus sharpness parameter) - typically just one value after key vector
        // Apply softplus using activation functions instead of direct call
        T keyStrengthValue = parameters[keyVectorSize];
        T keyStrength;
        if (ContentAddressingVectorActivation != null)
        {
            var tempVector = new Vector<T>(1) { [0] = keyStrengthValue };
            keyStrength = ContentAddressingVectorActivation.Activate(tempVector)[0];
        }
        else if (ContentAddressingActivation != null)
        {
            keyStrength = ContentAddressingActivation.Activate(keyStrengthValue);
        }
        else
        {
            // Fallback softplus implementation
            keyStrength = numOps.Log(numOps.Add(numOps.One, numOps.Exp(keyStrengthValue)));
        }

        // Extract gate value (interpolation parameter) - typically one value after key strength
        // Apply sigmoid using our gate activation
        T gateValue = parameters[keyVectorSize + 1];
        T gate;
        if (GateVectorActivation != null)
        {
            var tempVector = new Vector<T>(1) { [0] = gateValue };
            gate = GateVectorActivation.Activate(tempVector)[0];
        }
        else if (GateActivation != null)
        {
            gate = GateActivation.Activate(gateValue);
        }
        else
        {
            // Fallback sigmoid implementation
            gate = MathHelper.Sigmoid(gateValue);
        }

        // Extract shift weights (for location addressing) - We'll use 3 values for -1, 0, +1 shifts
        var shifts = new Vector<T>(3);
        for (int i = 0; i < 3; i++)
        {
            shifts[i] = parameters[keyVectorSize + 2 + i];
        }

        // Apply softmax to shifts using our content addressing activation (since it's typically softmax)
        shifts = ApplyActivation(shifts, ActivationType.ContentAddressing);

        // Extract sharpening factor - one value after shifts
        T sharpeningFactorValue = parameters[keyVectorSize + 5];
        T sharpeningFactor;
        if (ContentAddressingVectorActivation != null)
        {
            var tempVector = new Vector<T>(1) { [0] = sharpeningFactorValue };
            sharpeningFactor = numOps.Add(numOps.One, ContentAddressingVectorActivation.Activate(tempVector)[0]);
        }
        else if (ContentAddressingActivation != null)
        {
            sharpeningFactor = numOps.Add(numOps.One, ContentAddressingActivation.Activate(sharpeningFactorValue));
        }
        else
        {
            // Fallback softplus implementation
            sharpeningFactor = numOps.Add(numOps.One, numOps.Log(numOps.Add(numOps.One, numOps.Exp(sharpeningFactorValue))));
        }

        // 1. Content addressing - find similarity between key and each memory row
        var contentWeights = ContentAddressing(memory, keyVector, keyStrength);

        // 2. Interpolation - blend between previous weights and content weights
        var interpolatedWeights = new Vector<T>(_memorySize);
        for (int m = 0; m < _memorySize; m++)
        {
            interpolatedWeights[m] = numOps.Add(
                numOps.Multiply(numOps.Subtract(numOps.One, gate), previousWeights[m]),
                numOps.Multiply(gate, contentWeights[m])
            );
        }

        // 3. Convolutional shift - apply circular shift to weights
        var shiftedWeights = ConvolutionalShift(interpolatedWeights, shifts);

        // 4. Sharpening - focus attention by raising to power and renormalizing
        var sharpenedWeights = Sharpen(shiftedWeights, sharpeningFactor);

        return sharpenedWeights;
    }

    /// <summary>
    /// Applies a scalar activation function element-wise to a vector.
    /// </summary>
    /// <param name="vector">The input vector.</param>
    /// <param name="activation">The activation function to apply.</param>
    /// <returns>The activated vector.</returns>
    private Vector<T> ApplyScalarActivation(Vector<T> vector, IActivationFunction<T>? activation)
    {
        if (activation == null)
            return vector;

        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = activation.Activate(vector[i]);
        }

        return result;
    }

    /// <summary>
    /// Applies the appropriate activation function to a vector.
    /// </summary>
    /// <param name="vector">The input vector.</param>
    /// <param name="activationType">The type of activation to apply.</param>
    /// <returns>The activated vector.</returns>
    private Vector<T> ApplyActivation(Vector<T> vector, ActivationType activationType)
    {
        switch (activationType)
        {
            case ActivationType.ContentAddressing:
                if (ContentAddressingVectorActivation != null)
                    return ContentAddressingVectorActivation.Activate(vector);
                else
                    return ApplyScalarActivation(vector, ContentAddressingActivation);

            case ActivationType.Gate:
                if (GateVectorActivation != null)
                    return GateVectorActivation.Activate(vector);
                else
                    return ApplyScalarActivation(vector, GateActivation);

            case ActivationType.Output:
                if (OutputVectorActivation != null)
                    return OutputVectorActivation.Activate(vector);
                else
                    return ApplyScalarActivation(vector, OutputActivation);

            default:
                throw new ArgumentException("Unknown activation type", nameof(activationType));
        }
    }

    /// <summary>
    /// The types of activation functions used in the NTM.
    /// </summary>
    private enum ActivationType
    {
        ContentAddressing,
        Gate,
        Output
    }

    /// <summary>
    /// Applies content-based addressing to find similar memory locations.
    /// </summary>
    /// <param name="memory">The memory matrix.</param>
    /// <param name="key">The key vector to match against memory.</param>
    /// <param name="keyStrength">The key strength parameter that amplifies similarity.</param>
    /// <returns>A vector of attention weights based on content similarity.</returns>
    private Vector<T> ContentAddressing(Matrix<T> memory, Vector<T> key, T keyStrength)
    {
        var similarities = new Vector<T>(_memorySize);

        // Calculate cosine similarity between key and each memory row
        for (int m = 0; m < _memorySize; m++)
        {
            var memoryRow = new Vector<T>(_memoryVectorSize);
            for (int i = 0; i < _memoryVectorSize; i++)
            {
                memoryRow[i] = memory[m, i];
            }

            similarities[m] = StatisticsHelper<T>.CosineSimilarity(key, memoryRow);
        }

        // Apply key strength (focus factor)
        for (int m = 0; m < _memorySize; m++)
        {
            similarities[m] = NumOps.Multiply(keyStrength, similarities[m]);
        }

        // Apply softmax to get normalized attention weights
        return ApplyActivation(similarities, ActivationType.ContentAddressing);
    }

    /// <summary>
    /// Applies a circular convolution to shift attention weights.
    /// </summary>
    /// <param name="weights">The weights to shift.</param>
    /// <param name="shifts">The distribution of shifts to apply.</param>
    /// <returns>The shifted weights.</returns>
    private Vector<T> ConvolutionalShift(Vector<T> weights, Vector<T> shifts)
    {
        var result = new Vector<T>(_memorySize);

        // Initialize with zeros
        for (int i = 0; i < _memorySize; i++)
        {
            result[i] = NumOps.Zero;
        }

        // Apply each shift with its corresponding weight
        for (int i = 0; i < _memorySize; i++)
        {
            // Apply shifting with circular boundary conditions
            for (int j = 0; j < shifts.Length; j++)
            {
                // Convert shift index (0,1,2) to shift offset (-1,0,1)
                int shift = j - 1;

                // Calculate source index with circular wrapping
                int sourceIndex = (i - shift) % _memorySize;
                if (sourceIndex < 0) sourceIndex += _memorySize;

                // Add weighted contribution
                result[i] = NumOps.Add(result[i],
                    NumOps.Multiply(weights[sourceIndex], shifts[j]));
            }
        }

        return result;
    }

    /// <summary>
    /// Sharpens a weight vector by raising to a power and renormalizing.
    /// </summary>
    /// <param name="weights">The weights to sharpen.</param>
    /// <param name="gamma">The sharpening factor.</param>
    /// <returns>The sharpened weights.</returns>
    private Vector<T> Sharpen(Vector<T> weights, T gamma)
    {
        var result = new Vector<T>(_memorySize);
        T sum = NumOps.Zero;

        // Raise each weight to the power of gamma
        for (int i = 0; i < _memorySize; i++)
        {
            result[i] = NumOps.Power(weights[i], gamma);
            sum = NumOps.Add(sum, result[i]);
        }

        // Avoid division by zero
        if (NumOps.Equals(sum, NumOps.Zero))
        {
            sum = NumOps.FromDouble(1e-6);
        }

        // Normalize to ensure sum is 1
        for (int i = 0; i < _memorySize; i++)
        {
            result[i] = NumOps.Divide(result[i], sum);
        }

        return result;
    }

    /// <summary>
    /// Reads from all batch memories using their respective attention weights.
    /// </summary>
    /// <returns>A tensor containing read results for all batch elements.</returns>
    private Tensor<T> ReadFromMemories()
    {
        int batchSize = _memories.Count;
        var result = new Tensor<T>([batchSize, _memoryVectorSize]);

        for (int b = 0; b < batchSize; b++)
        {
            var readResult = ReadFromMemory(_memories[b], _readWeights[b]);
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                result[b, v] = readResult[v];
            }
        }

        return result;
    }

    /// <summary>
    /// Reads from memory using attention weights.
    /// </summary>
    /// <param name="memory">The memory matrix to read from.</param>
    /// <param name="readWeights">The attention weights for reading.</param>
    /// <returns>The read result vector.</returns>
    private Vector<T> ReadFromMemory(Matrix<T> memory, Vector<T> readWeights)
    {
        var result = new Vector<T>(_memoryVectorSize);

        // Initialize with zeros
        for (int v = 0; v < _memoryVectorSize; v++)
        {
            result[v] = NumOps.Zero;
        }

        // Perform weighted read
        for (int m = 0; m < _memorySize; m++)
        {
            T weight = readWeights[m];

            // Skip if weight is effectively zero (optimization)
            if (NumOps.LessThan(weight, NumOps.FromDouble(1e-10)))
            {
                continue;
            }

            for (int v = 0; v < _memoryVectorSize; v++)
            {
                T weightedValue = NumOps.Multiply(weight, memory[m, v]);
                result[v] = NumOps.Add(result[v], weightedValue);
            }
        }

        return result;
    }

    /// <summary>
    /// Writes to all batch memories using their respective attention weights.
    /// </summary>
    /// <param name="writeParams">The parameters for write operations.</param>
    private void WriteToMemories(Tensor<T> writeParams)
    {
        int batchSize = _memories.Count;
        int paramSize = writeParams.Shape[1];

        for (int b = 0; b < batchSize; b++)
        {
            // Extract write parameters for this batch
            var writeVector = ExtractVector(writeParams, b);

            // Calculate erase and add vectors
            var eraseVector = new Vector<T>(_memoryVectorSize);
            var addVector = new Vector<T>(_memoryVectorSize);

            // Extract erase vector (first half of parameters, apply gate activation to get [0,1] range)
            int eraseSize = Math.Min(_memoryVectorSize, paramSize / 2);

            // Create a temporary vector for the erase parameters
            var eraseParams = new Vector<T>(eraseSize);
            for (int i = 0; i < eraseSize; i++)
            {
                eraseParams[i] = writeVector[i];
            }

            // Apply activation function to the erase parameters
            Vector<T> activatedEraseParams;
            if (GateVectorActivation != null)
            {
                // Use vector activation if available
                activatedEraseParams = GateVectorActivation.Activate(eraseParams);
            }
            else if (GateActivation != null)
            {
                // Use scalar activation if available
                activatedEraseParams = ApplyScalarActivation(eraseParams, GateActivation);
            }
            else
            {
                // Fallback to default sigmoid implementation
                activatedEraseParams = new Vector<T>(eraseParams.Length);
                for (int i = 0; i < eraseParams.Length; i++)
                {
                    activatedEraseParams[i] = MathHelper.Sigmoid(eraseParams[i]);
                }
            }

            // Map the activated values to the erase vector
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                int eraseIndex = v % eraseSize;
                eraseVector[v] = activatedEraseParams[eraseIndex];
            }

            // Extract add vector (second half of parameters)
            int addStart = paramSize / 2;
            int addSize = Math.Min(_memoryVectorSize, paramSize - addStart);
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                int addIndex = addStart + (v % addSize);
                addVector[v] = writeVector[addIndex];
            }

            // Perform erase and add operations
            WriteToMemory(_memories[b], _writeWeights[b], eraseVector, addVector);
        }
    }

    /// <summary>
    /// Writes to memory using attention weights and erase/add vectors.
    /// </summary>
    /// <param name="memory">The memory matrix to write to.</param>
    /// <param name="writeWeights">The attention weights for writing.</param>
    /// <param name="eraseVector">The vector specifying what to erase at each location.</param>
    /// <param name="addVector">The vector specifying what to add at each location.</param>
    private void WriteToMemory(Matrix<T> memory, Vector<T> writeWeights, Vector<T> eraseVector, Vector<T> addVector)
    {
        // Perform write operation for each memory location
        for (int m = 0; m < _memorySize; m++)
        {
            T weight = writeWeights[m];

            // Skip if weight is effectively zero (optimization)
            if (NumOps.LessThan(weight, NumOps.FromDouble(1e-10)))
            {
                continue;
            }

            // Erase phase - memory[i] = memory[i] * (1 - weight * erase[i])
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                T eraseAmount = NumOps.Multiply(weight, eraseVector[v]);
                T retainAmount = NumOps.Subtract(NumOps.One, eraseAmount);
                memory[m, v] = NumOps.Multiply(memory[m, v], retainAmount);
            }

            // Add phase - memory[i] = memory[i] + weight * add[i]
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                T addAmount = NumOps.Multiply(weight, addVector[v]);
                memory[m, v] = NumOps.Add(memory[m, v], addAmount);
            }
        }
    }

    /// <summary>
    /// Generates the final output from controller state and read result.
    /// </summary>
    /// <param name="controllerState">The controller output state.</param>
    /// <param name="readResult">The result from reading memory.</param>
    /// <returns>The final output tensor.</returns>
    private Tensor<T> GenerateOutput(Tensor<T> controllerState, Tensor<T> readResult)
    {
        int batchSize = controllerState.Shape[0];

        // Combine controller state with read result
        var combined = new Tensor<T>(new int[] { batchSize, controllerState.Shape[1] + readResult.Shape[1] });
        for (int b = 0; b < batchSize; b++)
        {
            // Copy controller state
            for (int i = 0; i < controllerState.Shape[1]; i++)
            {
                combined[b, i] = controllerState[b, i];
            }

            // Copy read result
            for (int i = 0; i < readResult.Shape[1]; i++)
            {
                combined[b, controllerState.Shape[1] + i] = readResult[b, i];
            }
        }

        // Process through output layers (second half of layers)
        var current = combined;
        for (int i = Layers.Count / 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Trains the Neural Turing Machine on a single batch of input-output pairs.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input.Shape[0] != expectedOutput.Shape[0])
        {
            throw new ArgumentException("Input and expected output must have the same batch size");
        }

        // Set to training mode
        SetTrainingMode(true);

        // Forward pass
        var predictions = Predict(input);

        // Calculate loss
        var predVector = predictions.ToVector();
        var expectedVector = expectedOutput.ToVector();
        T loss = LossFunction.CalculateLoss(predVector, expectedVector);
        LastLoss = loss;

        // Calculate output gradients
        var gradVector = LossFunction.CalculateDerivative(predVector, expectedVector);
        var outputGradients = new Tensor<T>(predictions.Shape);

        // Copy gradient values to tensor
        int index = 0;
        for (int b = 0; b < predictions.Shape[0]; b++)
        {
            for (int i = 0; i < predictions.Shape[1]; i++)
            {
                outputGradients[b, i] = gradVector[index++];
            }
        }

        // Backpropagation
        BackpropagateNTM(outputGradients);

        // Update parameters using the learning rate
        T learningRate = MathHelper.GetNumericOperations<T>().FromDouble(0.01); // Default learning rate
        UpdateParameters(learningRate);

        // Reset to inference mode
        SetTrainingMode(false);
    }

    /// <summary>
    /// Performs backpropagation through the Neural Turing Machine.
    /// </summary>
    /// <param name="outputGradients">The gradients from the output layer.</param>
    private void BackpropagateNTM(Tensor<T> outputGradients)
    {
        // Clear existing gradients
        foreach (var layer in Layers)
        {
            layer.ClearGradients();
        }

        // Backpropagate through output layers (second half)
        var gradients = outputGradients;
        for (int i = Layers.Count - 1; i >= Layers.Count / 2; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }

        // At this point, we would implement complex backpropagation through memory operations
        // but for simplicity in this improved version, we focus on the neural network path

        // Backpropagate through input layers (first half)
        for (int i = Layers.Count / 2 - 1; i >= 0; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }
    }

    /// <summary>
    /// Updates the parameters of the neural network layers.
    /// </summary>
    /// <param name="learningRate">The learning rate for the update.</param>
    private void UpdateParameters(T learningRate)
    {
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Updates the parameters of the neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.Subvector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Sets the layer to training or evaluation mode.
    /// </summary>
    /// <param name="isTraining">True to set the layer to training mode, false for evaluation mode.</param>
    public override void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(isTraining);
        }
    }


    /// <summary>
    /// Gets metadata about the Neural Turing Machine model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the NTM.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralTuringMachine,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MemorySize", _memorySize },
                { "MemoryVectorSize", _memoryVectorSize },
                { "ControllerSize", _controllerSize },
                { "TotalParameters", ParameterCount },
                { "LayerCount", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes NTM-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write memory configuration
        writer.Write(_memorySize);
        writer.Write(_memoryVectorSize);
        writer.Write(_controllerSize);
        writer.Write(_memories.Count);

        // Write memory contents
        foreach (var memory in _memories)
        {
            for (int i = 0; i < _memorySize; i++)
            {
                for (int j = 0; j < _memoryVectorSize; j++)
                {
                    writer.Write(Convert.ToDouble(memory[i, j]));
                }
            }
        }

        // Write read weights
        writer.Write(_readWeights.Count);
        foreach (var weights in _readWeights)
        {
            for (int i = 0; i < _memorySize; i++)
            {
                writer.Write(Convert.ToDouble(weights[i]));
            }
        }

        // Write write weights
        writer.Write(_writeWeights.Count);
        foreach (var weights in _writeWeights)
        {
            for (int i = 0; i < _memorySize; i++)
            {
                writer.Write(Convert.ToDouble(weights[i]));
            }
        }
    }

    /// <summary>
    /// Deserializes NTM-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read memory configuration
        _memorySize = reader.ReadInt32();
        _memoryVectorSize = reader.ReadInt32();
        _controllerSize = reader.ReadInt32();
        int memoryCount = reader.ReadInt32();

        // Read memory contents
        _memories.Clear();
        for (int b = 0; b < memoryCount; b++)
        {
            var memory = new Matrix<T>(_memorySize, _memoryVectorSize);
            for (int i = 0; i < _memorySize; i++)
            {
                for (int j = 0; j < _memoryVectorSize; j++)
                {
                    memory[i, j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            _memories.Add(memory);
        }

        // Read read weights
        _readWeights.Clear();
        int readWeightsCount = reader.ReadInt32();
        for (int b = 0; b < readWeightsCount; b++)
        {
            var weights = new Vector<T>(_memorySize);
            for (int i = 0; i < _memorySize; i++)
            {
                weights[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _readWeights.Add(weights);
        }

        // Read write weights
        _writeWeights.Clear();
        int writeWeightsCount = reader.ReadInt32();
        for (int b = 0; b < writeWeightsCount; b++)
        {
            var weights = new Vector<T>(_memorySize);
            for (int i = 0; i < _memorySize; i++)
            {
                weights[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _writeWeights.Add(weights);
        }
    }

    /// <summary>
    /// Creates a new instance of the neural turing machine model.
    /// </summary>
    /// <returns>A new instance of the neural turing machine model with the same configuration.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Determine which constructor to use based on whether we're using scalar or vector activations
        if (ContentAddressingVectorActivation != null || GateVectorActivation != null || OutputVectorActivation != null)
        {
            // Use the vector activation constructor
            return new NeuralTuringMachine<T>(
                Architecture,
                _memorySize,
                _memoryVectorSize,
                _controllerSize,
                LossFunction,
                ContentAddressingVectorActivation,
                GateVectorActivation,
                OutputVectorActivation);
        }
        else
        {
            // Use the scalar activation constructor
            return new NeuralTuringMachine<T>(
                Architecture,
                _memorySize,
                _memoryVectorSize,
                _controllerSize,
                LossFunction,
                ContentAddressingActivation,
                GateActivation,
                OutputActivation);
        }
    }

    /// <summary>
    /// Resets the internal state of the neural network.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears the memory and attention weights, essentially
    /// making the network "forget" everything it has learned during sequence processing.
    /// It's useful when starting to process a new sequence that should not be influenced
    /// by previous sequences.</para>
    /// </remarks>
    public override void ResetState()
    {
        T uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_memorySize));

        // Reset memory matrices to small random values
        InitializeMemory();

        // Reset attention weights to uniform distribution
        for (int b = 0; b < _readWeights.Count; b++)
        {
            for (int i = 0; i < _memorySize; i++)
            {
                _readWeights[b][i] = uniformWeight;
                _writeWeights[b][i] = uniformWeight;
            }
        }

        // Reset layer states
        foreach (var layer in Layers)
        {
            layer.ResetState();
        }
    }
}
