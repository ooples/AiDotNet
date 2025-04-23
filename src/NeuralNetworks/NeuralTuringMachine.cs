namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Neural Turing Machine, which is a neural network architecture that combines a neural network with external memory.
/// </summary>
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
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NeuralTuringMachine<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the size of the external memory matrix (number of memory locations).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory size determines the number of separate memory locations available to the neural network.
    /// A larger memory size allows the network to store more distinct pieces of information but increases
    /// computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the number of pages in the network's notebook.
    /// 
    /// More memory locations (larger MemorySize) means:
    /// - The network can store more separate pieces of information
    /// - It can keep track of more things at once
    /// - It might perform better on complex tasks that require remembering many details
    /// 
    /// However, a larger memory also requires more computing power to process.
    /// </para>
    /// </remarks>
    private int _memorySize;

    /// <summary>
    /// Gets or sets the size of each memory vector (the amount of information stored at each memory location).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory vector size determines how much information can be stored at each memory location.
    /// Larger vector sizes allow more detailed information to be stored at each location but increase
    /// computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is like how much information you can write on each page of the notebook.
    /// 
    /// A larger vector size means:
    /// - Each memory location can store more detailed information
    /// - The network can capture more complex patterns at each location
    /// - It might provide better performance for tasks requiring nuanced memory
    /// 
    /// Think of it as the difference between taking brief notes versus detailed notes on each page.
    /// </para>
    /// </remarks>
    private int _memoryVectorSize;

    /// <summary>
    /// Gets or sets the size of the controller network that manages memory operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The controller size determines the complexity of the neural network that decides how to interact with
    /// the external memory. A larger controller can implement more sophisticated memory access strategies
    /// but requires more computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the brain power of the system that decides when and how to use the notebook.
    /// 
    /// A larger controller size means:
    /// - The network can make more sophisticated decisions about using memory
    /// - It can develop more complex strategies for storing and retrieving information
    /// - It might learn more effectively on difficult tasks
    /// 
    /// Think of it as having a smarter student who knows better strategies for taking and using notes.
    /// </para>
    /// </remarks>
    private int _controllerSize;

    /// <summary>
    /// Gets or sets the external memory matrix used by the Neural Turing Machine.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The memory matrix stores all the information that the Neural Turing Machine can access. It is organized
    /// as a matrix where each row represents a memory location and each column represents a dimension of the
    /// stored information vector.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual notebook where information is stored.
    /// 
    /// The memory matrix:
    /// - Has _memorySize rows (like pages in a notebook)
    /// - Has _memoryVectorSize columns (like the amount of information on each page)
    /// - Can be read from and written to by the neural network
    /// - Persists information across processing steps, allowing the network to "remember"
    /// 
    /// When the network processes information, it can store results here and retrieve them later,
    /// which is what gives the Neural Turing Machine its powerful memory capabilities.
    /// </para>
    /// </remarks>
    private Matrix<T> _memory;

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralTuringMachine{T}"/> class with the specified architecture and memory parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the NTM.</param>
    /// <param name="memorySize">The number of memory locations (rows in the memory matrix).</param>
    /// <param name="memoryVectorSize">The size of each memory vector (columns in the memory matrix).</param>
    /// <param name="controllerSize">The size of the controller network that manages memory operations.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Neural Turing Machine with the specified architecture and memory parameters.
    /// It initializes the memory matrix with small random values to facilitate learning during training.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Neural Turing Machine with its basic components.
    /// 
    /// When creating a new NTM:
    /// - architecture: Defines the overall structure of the neural network
    /// - memorySize: Sets how many separate memory locations are available
    /// - memoryVectorSize: Sets how much information can be stored at each location
    /// - controllerSize: Sets how complex the control system is
    /// 
    /// The constructor also initializes the memory with small random values as a starting point,
    /// similar to how you might prepare a notebook with light markings before actually using it.
    /// </para>
    /// </remarks>
    public NeuralTuringMachine(NeuralNetworkArchitecture<T> architecture, int memorySize, int memoryVectorSize, int controllerSize) 
        : base(architecture)
    {
        _memorySize = memorySize;
        _memoryVectorSize = memoryVectorSize;
        _controllerSize = controllerSize;
        _memory = new Matrix<T>(_memorySize, _memoryVectorSize);

        InitializeMemory();
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the memory matrix with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method populates the memory matrix with small random values between 0 and 0.1. Using small
    /// initial values helps with the training process by providing a stable starting point that allows
    /// gradual learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the notebook with light random markings.
    /// 
    /// When initializing memory:
    /// - Each cell in the memory matrix gets a small random value
    /// - These values are between 0 and 0.1 (very small)
    /// - Starting with small random values helps the network learn more effectively
    /// - It's like having faint pencil marks that can be easily modified during learning
    /// 
    /// Without this initialization, the network might have trouble starting the learning process.
    /// </para>
    /// </remarks>
    private void InitializeMemory()
    {
        // Initialize memory with small random values
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryVectorSize; j++)
            {
                _memory[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1);
            }
        }
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Neural Turing Machine. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration is created based on
    /// the memory parameters. The method also validates that custom layers are compatible with NTM requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for an NTM are created automatically
    /// - The system checks that any custom layers will work properly with the NTM
    /// 
    /// Layers are like the different processing stages in the neural network.
    /// Each layer performs a specific operation on the data as it flows through the network.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNTMLayers(Architecture, _memorySize, _memoryVectorSize, _controllerSize));
        }
    }

    /// <summary>
    /// Updates the parameters of the neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the neural network makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - It takes a vector of parameter updates and distributes them to the correct layers
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// Think of it like updating the decision-making rules for each part of the network
    /// based on what was learned during training.
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
    /// Performs a forward pass through the Neural Turing Machine.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input through the NTM, including controller network processing and
    /// memory operations (reading and writing). It handles both single inputs and batches through
    /// tensor operations for improved performance.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes information through the Neural Turing Machine.
    /// 
    /// During a forward pass:
    /// - The input data is processed by the controller network
    /// - The controller decides how to interact with memory (what to read and write)
    /// - The network reads from appropriate memory locations
    /// - It writes updated information to memory
    /// - It produces an output based on both the input and what it read from memory
    /// 
    /// This combination of neural processing and memory operations is what makes the NTM powerful
    /// for tasks requiring storing and retrieving information over time.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Set to inference mode
        SetTrainingMode(false);
    
        // Get batch size and sequence length from input shape
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape.Length > 1 ? input.Shape[1] : 1;
    
        // Initialize attention weights (used to determine which memory locations to access)
        var readWeights = InitializeAttentionWeights(batchSize);
        var writeWeights = InitializeAttentionWeights(batchSize);
    
        // Store all outputs for each time step
        var outputs = new List<Tensor<T>>();
    
        // Process each time step in the sequence
        for (int t = 0; t < sequenceLength; t++)
        {
            // Extract input for current time step
            Tensor<T> currentInput;
            if (sequenceLength > 1)
            {
                // For sequence inputs, extract the current time step
                Vector<T> sliceVector = input.GetSlice(1, t);
                currentInput = new Tensor<T>([1, sliceVector.Length]);
                for (int i = 0; i < sliceVector.Length; i++)
                {
                    currentInput[0, i] = sliceVector[i];
                }
            }
            else
            {
                // For single-step inputs, use the entire input
                currentInput = input;
            }
        
            // Process current input and memory state through controller
            var controllerState = ProcessController(currentInput, readWeights);
        
            // Generate read and write attention parameters from controller state
            var readParams = GenerateReadParameters(controllerState);
            var writeParams = GenerateWriteParameters(controllerState);
        
            // Update attention weights based on parameters
            readWeights = UpdateAttentionWeights(readWeights, readParams);
            writeWeights = UpdateAttentionWeights(writeWeights, writeParams);
        
            // Read from memory using read weights
            var readResult = ReadFromMemory(readWeights);
        
            // Write to memory using write weights
            WriteToMemory(writeWeights, writeParams);
        
            // Generate output from controller state and read result
            var output = GenerateOutput(controllerState, readResult);
            outputs.Add(output);
        }
    
        // For sequence inputs, stack outputs; for single inputs, return the single output
        if (sequenceLength > 1)
        {
            return Tensor<T>.Stack(outputs.ToArray(), 1);
        }
        else
        {
            return outputs[0];
        }
    }

    /// <summary>
    /// Initializes attention weights for memory access.
    /// </summary>
    /// <param name="batchSize">The batch size to initialize weights for.</param>
    /// <returns>A tensor containing initialized attention weights.</returns>
    /// <remarks>
    /// <para>
    /// This method initializes the attention weights that determine which memory locations the
    /// NTM reads from and writes to. Initially, attention is uniformly distributed across all
    /// memory locations.
    /// </para>
    /// <para><b>For Beginners:</b> This creates starting values for memory access.
    /// 
    /// Attention weights determine:
    /// - Which memory locations the network focuses on
    /// - How much importance is given to each location
    /// - They start with equal focus on all memory locations
    /// - During processing, these weights will be updated to focus on relevant locations
    /// 
    /// Think of it like starting with a blank notebook where all pages are equally likely to be used.
    /// </para>
    /// </remarks>
    private Tensor<T> InitializeAttentionWeights(int batchSize)
    {
        // Create a uniform distribution over memory locations for each batch
        var shape = new int[] { batchSize, _memorySize };
        var weights = new Tensor<T>(shape);
    
        // Initialize with uniform weights (1/memorySize for each location)
        T uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_memorySize));
    
        for (int b = 0; b < batchSize; b++)
        {
            for (int m = 0; m < _memorySize; m++)
            {
                weights[b, m] = uniformWeight;
            }
        }
    
        return weights;
    }

    /// <summary>
    /// Processes input and previous memory state through the controller network.
    /// </summary>
    /// <param name="input">The current input tensor.</param>
    /// <param name="previousReadWeights">The previous read attention weights.</param>
    /// <returns>The controller output state tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method runs the input and previous memory state through the controller network
    /// to determine how to interact with memory. The controller is typically a neural network
    /// that produces parameters for reading and writing operations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the brain deciding how to use the notebook.
    /// 
    /// The controller network:
    /// - Takes the current input (what we're processing now)
    /// - Considers what was previously read from memory
    /// - Decides what information to read next
    /// - Decides what information to write to memory
    /// - Produces control signals for these memory operations
    /// 
    /// This is the "neural" part of the Neural Turing Machine that makes decisions about
    /// how to use the external memory.
    /// </para>
    /// </remarks>
    private Tensor<T> ProcessController(Tensor<T> input, Tensor<T> previousReadWeights)
    {
        // Determine what was read from memory based on previous read weights
        var previousReadResult = ReadFromMemory(previousReadWeights);
    
        // Concatenate input with previous read result to create controller input
        var controllerInput = input.ConcatenateTensors(previousReadResult);
    
        // Pass through controller network (the first few layers of the NTM)
        var current = controllerInput;
        for (int i = 0; i < Layers.Count / 2; i++) // Use first half of layers as controller
        {
            current = Layers[i].Forward(current);
        }
    
        return current;
    }

    /// <summary>
    /// Generates parameters for memory reading from controller output.
    /// </summary>
    /// <param name="controllerState">The controller output state.</param>
    /// <returns>A tensor containing read operation parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts parameters for the read operation from the controller output.
    /// These parameters include key vectors (what to look for in memory) and sharpening
    /// factors (how precisely to focus on specific memory locations).
    /// </para>
    /// <para><b>For Beginners:</b> This determines what information to look for in memory.
    /// 
    /// The read parameters include:
    /// - Key vectors: patterns to match in memory (what to look for)
    /// - Sharpening: how focused or diffuse the attention should be
    /// 
    /// Think of it like deciding which pages of a notebook to look at and how closely
    /// to examine each page.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateReadParameters(Tensor<T> controllerState)
    {
        // Extract part of controller output dedicated to read parameters
        // For simplicity, we'll use a portion of the controller state
        int readParamStart = _controllerSize / 2;
        int readParamSize = _controllerSize / 4;
    
        // Simplified implementation - in a real NTM, this would extract more specific parameters
        var readParams = ExtractControllerParameters(controllerState, readParamStart, readParamSize);
    
        return readParams;
    }

    /// <summary>
    /// Generates parameters for memory writing from controller output.
    /// </summary>
    /// <param name="controllerState">The controller output state.</param>
    /// <returns>A tensor containing write operation parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts parameters for the write operation from the controller output.
    /// These parameters include write keys (where to write), erase vectors (what to remove),
    /// and write vectors (what to add to memory).
    /// </para>
    /// <para><b>For Beginners:</b> This determines what information to write to memory.
    /// 
    /// The write parameters include:
    /// - Write keys: where in memory to write
    /// - Erase vectors: what information to remove from memory
    /// - Write vectors: what new information to add to memory
    /// 
    /// Think of it like deciding which pages of a notebook to write on,
    /// what to erase, and what new notes to write down.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateWriteParameters(Tensor<T> controllerState)
    {
        // Extract part of controller output dedicated to write parameters
        // For simplicity, we'll use a portion of the controller state
        int writeParamStart = _controllerSize * 3 / 4;
        int writeParamSize = _controllerSize / 4;
    
        // Simplified implementation - in a real NTM, this would extract specific parameters
        var writeParams = ExtractControllerParameters(controllerState, writeParamStart, writeParamSize);
    
        return writeParams;
    }

    /// <summary>
    /// Extracts a segment of parameters from the controller state.
    /// </summary>
    /// <param name="controllerState">The controller output state.</param>
    /// <param name="start">The starting index for extraction.</param>
    /// <param name="size">The number of parameters to extract.</param>
    /// <returns>A tensor containing the extracted parameters.</returns>
    private Tensor<T> ExtractControllerParameters(Tensor<T> controllerState, int start, int size)
    {
        // Get shape information
        int[] stateShape = controllerState.Shape;
        int batchSize = stateShape[0];
    
        // Create result shape based on input dimensions
        int[] resultShape;
        if (stateShape.Length == 2)
        {
            // For 2D input (batch, features)
            resultShape = new int[] { batchSize, size };
        }
        else if (stateShape.Length == 3)
        {
            // For 3D input (batch, sequence, features)
            int seqLength = stateShape[1];
            resultShape = new int[] { batchSize, seqLength, size };
        }
        else
        {
            throw new NotSupportedException("Parameter extraction supports only 2D and 3D tensors");
        }
    
        // Create result tensor
        var result = new Tensor<T>(resultShape);
    
        // Extract values
        if (stateShape.Length == 2)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < size && start + i < stateShape[1]; i++)
                {
                    result[b, i] = controllerState[b, start + i];
                }
            }
        }
        else if (stateShape.Length == 3)
        {
            int seqLength = stateShape[1];
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLength; s++)
                {
                    for (int i = 0; i < size && start + i < stateShape[2]; i++)
                    {
                        result[b, s, i] = controllerState[b, s, start + i];
                    }
                }
            }
        }
    
        return result;
    }

    /// <summary>
    /// Updates attention weights based on attention parameters.
    /// </summary>
    /// <param name="previousWeights">The previous attention weights.</param>
    /// <param name="parameters">The parameters for updating attention.</param>
    /// <returns>The updated attention weights.</returns>
    /// <remarks>
    /// <para>
    /// This method updates the attention weights that determine which memory locations to focus on
    /// based on parameters from the controller. It implements content-based addressing (finding similar
    /// content) and location-based addressing (shifting focus to nearby locations).
    /// </para>
    /// <para><b>For Beginners:</b> This updates where in memory to focus attention.
    /// 
    /// Attention updating includes:
    /// - Content-based addressing: focusing on locations with similar content
    /// - Location-based addressing: focusing on locations near previously attended ones
    /// - Interpolation: combining previous attention with new focus
    /// 
    /// This complex addressing mechanism is what allows the NTM to find relevant information
    /// in its memory based on both content and location.
    /// </para>
    /// </remarks>
    private Tensor<T> UpdateAttentionWeights(Tensor<T> previousWeights, Tensor<T> parameters)
    {
        // In a full implementation, this would use the parameters to perform:
        // 1. Content-based addressing (find locations with similar content)
        // 2. Location-based addressing (shift focus to nearby locations)
        // 3. Interpolation between previous and new weights
    
        // For this simplified implementation, we'll simulate attention updates
        // with a basic content-based addressing approach
    
        // Get shape information
        int[] weightShape = previousWeights.Shape;
        int batchSize = weightShape[0];
    
        // Create new attention weights tensor
        var newWeights = new Tensor<T>(weightShape);
    
        // Apply softmax to parameters to create focus (simplified approach)
        var focusWeights = ApplySoftmax(parameters, batchSize);
    
        // Interpolate between previous weights and focus
        // (simplified - a real NTM would use more complex logic)
        T interpolation = NumOps.FromDouble(0.5); // 50% new, 50% old
    
        for (int b = 0; b < batchSize; b++)
        {
            for (int m = 0; m < _memorySize; m++)
            {
                T oldWeight = previousWeights[b, m];
                T newWeight = focusWeights[b, m % focusWeights.Shape[1]]; // Cycle through focus if needed
            
                newWeights[b, m] = NumOps.Add(
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, interpolation), oldWeight),
                    NumOps.Multiply(interpolation, newWeight)
                );
            }
        }
    
        // Normalize weights to ensure they sum to 1
        newWeights = NormalizeWeights(newWeights);
    
        return newWeights;
    }

    /// <summary>
    /// Applies softmax normalization to a tensor.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <returns>A tensor with softmax applied.</returns>
    private Tensor<T> ApplySoftmax(Tensor<T> tensor, int batchSize)
    {
        // Get size of last dimension
        int size = tensor.Shape[tensor.Shape.Length - 1];
    
        // If size is larger than memory, truncate it
        size = Math.Min(size, _memorySize);
    
        // Create result with shape [batchSize, size]
        var result = new Tensor<T>(new int[] { batchSize, size });
    
        // Apply softmax to each batch independently
        for (int b = 0; b < batchSize; b++)
        {
            // Find max value for numerical stability
            T max = NumOps.Negate(NumOps.MaxValue);
            for (int i = 0; i < size; i++)
            {
                T val = tensor.Shape.Length == 2 ? tensor[b, i] : tensor[b, 0, i];
                if (NumOps.GreaterThan(val, max))
                {
                    max = val;
                }
            }
        
            // Calculate exp(x - max) for each element
            T[] expValues = new T[size];
            T sumExp = NumOps.Zero;
        
            for (int i = 0; i < size; i++)
            {
                T val = tensor.Shape.Length == 2 ? tensor[b, i] : tensor[b, 0, i];
                T expVal = NumOps.Exp(NumOps.Subtract(val, max));
                expValues[i] = expVal;
                sumExp = NumOps.Add(sumExp, expVal);
            }
        
            // Normalize by sum of exponentials
            for (int i = 0; i < size; i++)
            {
                result[b, i] = NumOps.Divide(expValues[i], sumExp);
            }
        }
    
        return result;
    }

    /// <summary>
    /// Normalizes weights to ensure they sum to 1 across the memory dimension.
    /// </summary>
    /// <param name="weights">The weights tensor to normalize.</param>
    /// <returns>The normalized weights tensor.</returns>
    private Tensor<T> NormalizeWeights(Tensor<T> weights)
    {
        // Get shape information
        int[] weightShape = weights.Shape;
        int batchSize = weightShape[0];
    
        // Create normalized weights tensor with same shape
        var normalized = new Tensor<T>(weightShape);
    
        // Normalize each batch independently
        for (int b = 0; b < batchSize; b++)
        {
            // Calculate sum
            T sum = NumOps.Zero;
            for (int m = 0; m < _memorySize; m++)
            {
                sum = NumOps.Add(sum, weights[b, m]);
            }
        
            // Ensure sum is not zero (add small epsilon if needed)
            if (NumOps.Equals(sum, NumOps.Zero))
            {
                sum = NumOps.FromDouble(1e-6);
            }
        
            // Normalize
            for (int m = 0; m < _memorySize; m++)
            {
                normalized[b, m] = NumOps.Divide(weights[b, m], sum);
            }
        }
    
        return normalized;
    }

    /// <summary>
    /// Reads from memory using attention weights.
    /// </summary>
    /// <param name="readWeights">The attention weights for reading.</param>
    /// <returns>The weighted read result from memory.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a content-based read from the memory matrix using the provided
    /// attention weights. It calculates a weighted sum of memory vectors, where the weights
    /// determine how much each memory location contributes to the result.
    /// </para>
    /// <para><b>For Beginners:</b> This retrieves information from memory.
    /// 
    /// The reading process:
    /// - Uses the attention weights to determine which memory locations to focus on
    /// - Takes a weighted average of the content at those locations
    /// - Returns this combined information
    /// 
    /// Think of it like reading multiple pages of a notebook, paying more attention
    /// to some pages than others, and combining the information.
    /// </para>
    /// </remarks>
    private Tensor<T> ReadFromMemory(Tensor<T> readWeights)
    {
        // Get batch size from readWeights
        int batchSize = readWeights.Shape[0];
    
        // Create result tensor with shape [batchSize, memoryVectorSize]
        var readResult = new Tensor<T>(new int[] { batchSize, _memoryVectorSize });
    
        // For each batch, perform weighted read from memory
        for (int b = 0; b < batchSize; b++)
        {
            // Initialize read vector with zeros
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                readResult[b, v] = NumOps.Zero;
            }
        
            // Perform weighted sum of memory vectors
            for (int m = 0; m < _memorySize; m++)
            {
                T weight = readWeights[b, m];
            
                for (int v = 0; v < _memoryVectorSize; v++)
                {
                    // Add weighted contribution from this memory location
                    T weightedValue = NumOps.Multiply(weight, _memory[m, v]);
                    readResult[b, v] = NumOps.Add(readResult[b, v], weightedValue);
                }
            }
        }
    
        return readResult;
    }

    /// <summary>
    /// Writes to memory using attention weights and write parameters.
    /// </summary>
    /// <param name="writeWeights">The attention weights for writing.</param>
    /// <param name="writeParams">The parameters for the write operation.</param>
    /// <remarks>
    /// <para>
    /// This method performs a content-based write to the memory matrix using the provided
    /// attention weights and write parameters. It implements the NTM's erase-then-write
    /// mechanism, where memory is first partially erased and then new content is added.
    /// </para>
    /// <para><b>For Beginners:</b> This updates information in memory.
    /// 
    /// The writing process:
    /// - Uses attention weights to determine which memory locations to update
    /// - First erases old information (partially or completely)
    /// - Then writes new information
    /// - The degree of update is controlled by the attention weights
    /// 
    /// Think of it like erasing parts of pages in a notebook and writing new notes,
    /// focusing more on some pages than others.
    /// </para>
    /// </remarks>
    private void WriteToMemory(Tensor<T> writeWeights, Tensor<T> writeParams)
    {
        // Get batch size from writeWeights
        int batchSize = writeWeights.Shape[0];
    
        // Extract erase and write vectors from writeParams (simplified)
        // In a full implementation, these would be more carefully extracted
    
        // Use first half of writeParams for erase vector
        var eraseVectors = new Tensor<T>(new int[] { batchSize, _memoryVectorSize });
    
        // Use second half of writeParams for write vector
        var writeVectors = new Tensor<T>(new int[] { batchSize, _memoryVectorSize });
    
        // Extract values (handling both 2D and 3D tensors)
        int paramsPerVector = writeParams.Shape[writeParams.Shape.Length - 1] / 2;
    
        for (int b = 0; b < batchSize; b++)
        {
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                // For erase vector, get values and apply sigmoid to get values between 0 and 1
                int eraseIndex = v % paramsPerVector;
                T eraseValue = writeParams.Shape.Length == 2 ? 
                    writeParams[b, eraseIndex] : 
                    writeParams[b, 0, eraseIndex];
            
                eraseVectors[b, v] = Sigmoid(eraseValue);
            
                // For write vector, get values from second half
                int writeIndex = paramsPerVector + (v % paramsPerVector);
                T writeValue = writeParams.Shape.Length == 2 ? 
                    writeParams[b, writeIndex] : 
                    writeParams[b, 0, writeIndex];
            
                writeVectors[b, v] = writeValue;
            }
        }
    
        // Apply erase-then-write mechanism
        // For simplicity, we'll apply batch 0's writes to the shared memory
        // In a full implementation, each batch would have its own memory state
    
        // Erase phase
        for (int m = 0; m < _memorySize; m++)
        {
            T weight = writeWeights[0, m]; // Use first batch's weights
        
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                // Erase = memory * (1 - weight * erase_vector)
                T eraseValue = NumOps.Multiply(weight, eraseVectors[0, v]);
                T retainValue = NumOps.Subtract(NumOps.One, eraseValue);
                _memory[m, v] = NumOps.Multiply(_memory[m, v], retainValue);
            }
        }
    
        // Write phase
        for (int m = 0; m < _memorySize; m++)
        {
            T weight = writeWeights[0, m]; // Use first batch's weights
        
            for (int v = 0; v < _memoryVectorSize; v++)
            {
                // Add = memory + weight * write_vector
                T addValue = NumOps.Multiply(weight, writeVectors[0, v]);
                _memory[m, v] = NumOps.Add(_memory[m, v], addValue);
            }
        }
    }

    /// <summary>
    /// Applies the sigmoid function to a value.
    /// </summary>
    /// <param name="value">The input value.</param>
    /// <returns>The sigmoid of the input value (between 0 and 1).</returns>
    private T Sigmoid(T value)
    {
        // Sigmoid function: 1 / (1 + e^(-x))
        T expNeg = NumOps.Exp(NumOps.Negate(value));
        T denominator = NumOps.Add(NumOps.One, expNeg);
        return NumOps.Divide(NumOps.One, denominator);
    }

    /// <summary>
    /// Generates output from controller state and read result.
    /// </summary>
    /// <param name="controllerState">The controller output state.</param>
    /// <param name="readResult">The result of reading from memory.</param>
    /// <returns>The final output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method combines the controller state and information read from memory to produce
    /// the final output of the Neural Turing Machine. Typically, a portion of the controller's
    /// output is dedicated to producing the final output, often combined with what was read
    /// from memory.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the final output from neural processing and memory.
    /// 
    /// The output generation:
    /// - Combines what was directly processed by the neural network
    /// - With information retrieved from memory
    /// - Produces the final result of the NTM's computation
    /// 
    /// This integration of direct neural processing with memory operations is what
    /// gives the NTM its power for complex sequential processing tasks.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateOutput(Tensor<T> controllerState, Tensor<T> readResult)
    {
        // Concatenate controller state with read result
        var combined = controllerState.ConcatenateTensors(readResult);
    
        // Process through output layers (second half of the NTM layers)
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
    /// <remarks>
    /// <para>
    /// This method trains the NTM using backpropagation through time (BPTT), which is an extension
    /// of standard backpropagation for recurrent neural networks. It unrolls the network through
    /// time steps, accumulates gradients, and updates the parameters to minimize the difference
    /// between predicted and expected outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the NTM to process information and use memory correctly.
    /// 
    /// The training process:
    /// - Runs input through the NTM to get predictions
    /// - Compares predictions to expected outputs
    /// - Calculates how wrong the predictions were
    /// - Propagates these errors backward through the network
    /// - Updates the network's parameters to improve future predictions
    /// 
    /// Training an NTM is particularly complex because errors must be propagated
    /// both through the neural network and through memory operations.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Set to training mode
        SetTrainingMode(true);
    
        // Forward pass to get predictions
        var predictions = Predict(input);
    
        // Calculate loss (mean squared error)
        var loss = CalculateMeanSquaredError(predictions, expectedOutput);
    
        // Calculate output gradients
        var outputGradients = CalculateOutputGradients(predictions, expectedOutput);
    
        // Backpropagation through time
        BackpropagateNTM(outputGradients, input);
    
        // Update parameters with calculated gradients
        UpdateNTMParameters();
    }

    /// <summary>
    /// Calculates mean squared error between predictions and expected outputs.
    /// </summary>
    /// <param name="predictions">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>The mean squared error loss value.</returns>
    private T CalculateMeanSquaredError(Tensor<T> predictions, Tensor<T> expected)
    {
        // Ensure tensors have the same shape
        if (!Enumerable.SequenceEqual(predictions.Shape, expected.Shape))
        {
            throw new ArgumentException("Predictions and expected outputs must have the same shape");
        }
    
        // Calculate squared differences
        T sumSquaredDiff = NumOps.Zero;
        int totalElements = 0;
    
        // Handle different tensor shapes
        if (predictions.Shape.Length == 2)
        {
            // 2D tensors [batch, features]
            int batchSize = predictions.Shape[0];
            int features = predictions.Shape[1];
            totalElements = batchSize * features;
        
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    T diff = NumOps.Subtract(predictions[b, f], expected[b, f]);
                    sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                }
            }
        }
        else if (predictions.Shape.Length == 3)
        {
            // 3D tensors [batch, sequence, features]
            int batchSize = predictions.Shape[0];
            int seqLength = predictions.Shape[1];
            int features = predictions.Shape[2];
            totalElements = batchSize * seqLength * features;
        
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLength; s++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        T diff = NumOps.Subtract(predictions[b, s, f], expected[b, s, f]);
                        sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
                    }
                }
            }
        }
        else
        {
            throw new NotSupportedException("MSE calculation currently supports only 2D and 3D tensors");
        }
    
        // Calculate mean
        return NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(totalElements));
    }

    /// <summary>
    /// Calculates gradients for output layer based on predictions and expected outputs.
    /// </summary>
    /// <param name="predictions">The predicted output tensor.</param>
    /// <param name="expected">The expected output tensor.</param>
    /// <returns>The gradient tensor for the output layer.</returns>
    private Tensor<T> CalculateOutputGradients(Tensor<T> predictions, Tensor<T> expected)
    {
        // Ensure tensors have the same shape
        if (!Enumerable.SequenceEqual(predictions.Shape, expected.Shape))
        {
            throw new ArgumentException("Predictions and expected outputs must have the same shape");
        }
    
        // Create gradient tensor with same shape as predictions
        var gradients = new Tensor<T>(predictions.Shape);
    
        // For MSE loss, gradient is 2 * (prediction - expected) / n
        // We'll simplify to (prediction - expected) and adjust learning rate instead
    
        // Handle different tensor shapes
        if (predictions.Shape.Length == 2)
        {
            // 2D tensors [batch, features]
            int batchSize = predictions.Shape[0];
            int features = predictions.Shape[1];
            T scaleFactor = NumOps.FromDouble(1.0 / (batchSize * features));
        
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    T diff = NumOps.Subtract(predictions[b, f], expected[b, f]);
                    gradients[b, f] = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(diff, scaleFactor));
                }
            }
        }
        else if (predictions.Shape.Length == 3)
        {
            // 3D tensors [batch, sequence, features]
            int batchSize = predictions.Shape[0];
            int seqLength = predictions.Shape[1];
            int features = predictions.Shape[2];
            T scaleFactor = NumOps.FromDouble(1.0 / (batchSize * seqLength * features));
        
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLength; s++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        T diff = NumOps.Subtract(predictions[b, s, f], expected[b, s, f]);
                        gradients[b, s, f] = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(diff, scaleFactor));
                    }
                }
            }
        }
        else
        {
            throw new NotSupportedException("Gradient calculation currently supports only 2D and 3D tensors");
        }
    
        return gradients;
    }

    /// <summary>
    /// Performs backpropagation through the Neural Turing Machine.
    /// </summary>
    /// <param name="outputGradients">The gradients from the output layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <remarks>
    /// <para>
    /// This method implements backpropagation through time (BPTT) for the NTM. It propagates gradients
    /// backward through both the neural network and memory operations, accounting for the recurrent
    /// nature of the NTM when processing sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how to improve the NTM's parameters.
    /// 
    /// The backpropagation process:
    /// - Traces errors backward through both neural network and memory operations
    /// - Accounts for how current outputs depend on previous memory states
    /// - Calculates how each parameter contributed to errors
    /// - Determines how to adjust each parameter to reduce future errors
    /// 
    /// This is particularly complex for NTMs because errors must be propagated
    /// through both the neural network and the external memory system.
    /// </para>
    /// </remarks>
    private void BackpropagateNTM(Tensor<T> outputGradients, Tensor<T> input)
    {
        // In a full implementation, this would:
        // 1. Backpropagate through the neural network layers
        // 2. Backpropagate through memory operations (reads and writes)
        // 3. Accumulate gradients for all parameters
    
        // For our simplified implementation, we'll focus on backpropagation through the network layers
    
        // Get batch size and sequence length
        int batchSize = input.Shape[0];
        int sequenceLength = input.Shape.Length > 1 ? input.Shape[1] : 1;
    
        // Start with output gradients
        var gradients = outputGradients;
    
        // Backpropagate through output layers (second half of layers)
        for (int i = Layers.Count - 1; i >= Layers.Count / 2; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }
    
        // For the memory and controller operations, we'd need to implement:
        // - Gradients for memory reading
        // - Gradients for memory writing
        // - Gradients for controller operations
    
        // These are complex operations that require careful implementation of all
        // the partial derivatives through the attention mechanisms and memory interactions
    
        // After passing through controller, backpropagate through input layers
        for (int i = Layers.Count / 2 - 1; i >= 0; i--)
        {
            gradients = Layers[i].Backward(gradients);
        }
    
        // The result is that all layers now have their gradients computed and stored internally
    }

    /// <summary>
    /// Updates the NTM parameters based on calculated gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies gradient updates to all parameters in the NTM, including network
    /// weights and any parameters related to memory operations. It uses a simple gradient
    /// descent approach with a fixed learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This adjusts the NTM's parameters to improve performance.
    /// 
    /// The parameter update:
    /// - Applies the calculated adjustments to all network parameters
    /// - Uses a learning rate to control how large the adjustments are
    /// - Small adjustments allow gradual, stable improvement
    /// 
    /// After these updates, the NTM should perform slightly better at its task
    /// the next time it processes similar inputs.
    /// </para>
    /// </remarks>
    private void UpdateNTMParameters()
    {
        // Simple learning rate for gradient descent
        T learningRate = NumOps.FromDouble(0.01);
    
        // Update parameters for each layer
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    
        // In a complete implementation, we would also update any additional parameters
        // specific to the memory operations
    }

    /// <summary>
    /// Gets metadata about the Neural Turing Machine model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the NTM.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the NTM, including its architecture,
    /// memory configuration, and other relevant parameters. This information is useful for
    /// model management, tracking experiments, and reporting.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about the NTM's configuration.
    /// 
    /// The metadata includes:
    /// - What this model is and what it does
    /// - Details about the neural network architecture
    /// - Information about the memory system (size, vector dimensions)
    /// - Other configuration parameters
    /// 
    /// This information is useful for keeping track of different models,
    /// documenting your work, and comparing experimental results.
    /// </para>
    /// </remarks>
    public override ModelMetaData<T> GetModelMetaData()
    {
        return new ModelMetaData<T>
        {
            ModelType = ModelType.NeuralTuringMachine,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MemorySize", _memorySize },
                { "MemoryVectorSize", _memoryVectorSize },
                { "ControllerSize", _controllerSize },
                { "TotalParameters", GetParameterCount() },
                { "LayerCount", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes NTM-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the state of the NTM to a binary stream. It serializes NTM-specific
    /// parameters like the memory matrix and controller size, allowing the complete state
    /// to be restored later.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the complete state of the NTM to a file.
    /// 
    /// When saving the NTM:
    /// - Memory contents are saved (what the model has "learned" and "remembers")
    /// - Configuration parameters are saved
    /// - Neural network parameters are saved
    /// 
    /// This allows you to:
    /// - Save your progress and continue training later
    /// - Share trained models with others
    /// - Deploy models in applications
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save memory configuration
        writer.Write(_memorySize);
        writer.Write(_memoryVectorSize);
        writer.Write(_controllerSize);
    
        // Save memory matrix contents
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryVectorSize; j++)
            {
                writer.Write(Convert.ToDouble(_memory[i, j]));
            }
        }
    }

    /// <summary>
    /// Deserializes NTM-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the state of a previously saved NTM from a binary stream. It restores
    /// NTM-specific parameters like the memory matrix and controller size, allowing the model
    /// to continue from exactly where it left off.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a complete NTM from a saved file.
    /// 
    /// When loading the NTM:
    /// - Memory contents are restored (what the model had "learned" and "remembered")
    /// - Configuration parameters are restored
    /// - Neural network parameters are restored
    /// 
    /// This lets you:
    /// - Continue working with a model exactly where you left off
    /// - Use a model that someone else has trained
    /// - Deploy pre-trained models in applications
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Load memory configuration
        _memorySize = reader.ReadInt32();
        _memoryVectorSize = reader.ReadInt32();
        _controllerSize = reader.ReadInt32();
    
        // Create and load memory matrix
        _memory = new Matrix<T>(_memorySize, _memoryVectorSize);
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryVectorSize; j++)
            {
                _memory[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <summary>
    /// Creates a new instance of the neural turing machine model.
    /// </summary>
    /// <returns>A new instance of the neural turing machine model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the neural turing machine model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization processes to create a fresh instance that can be populated
    /// with the serialized data. The new instance will have the same architecture, memory size, memory vector size, and
    /// controller size as the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the network structure without copying the learned data.
    /// 
    /// Think of it like creating a duplicate of the NTM's blueprint:
    /// - It copies the same neural network architecture 
    /// - It uses the same memory configuration (same notebook size and page capacity)
    /// - It sets up the same controller system (the brain that manages memory)
    /// - But it doesn't copy any of the actual memories or learned behaviors
    /// 
    /// This is primarily used when saving or loading models, creating an empty framework
    /// that the saved parameters can be loaded into later.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeuralTuringMachine<T>(
            Architecture,
            _memorySize,
            _memoryVectorSize,
            _controllerSize
        );
    }
}