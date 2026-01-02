namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Differentiable Neural Computer (DNC), a neural network architecture that combines neural networks with external memory resources.
/// </summary>
/// <remarks>
/// <para>
/// A Differentiable Neural Computer (DNC) is an advanced neural network architecture that augments neural networks with
/// an external memory matrix and mechanisms to read from and write to this memory. DNCs can learn to use their memory
/// to store and retrieve information, enabling them to solve complex, structured problems that require reasoning and
/// algorithm-like behavior. The key components include a controller neural network, a memory matrix, and read/write heads
/// that interact with the memory through differentiable attention mechanisms.
/// </para>
/// <para><b>For Beginners:</b> A Differentiable Neural Computer is like a neural network with a notepad.
/// 
/// Imagine a traditional neural network as a person who can make decisions based on what they see,
/// but can only keep information in their head. A DNC is like giving that person a notepad to:
/// - Write down important information
/// - Organize notes in a systematic way
/// - Look back at previously written notes when making decisions
/// - Learn which information is worth writing down and when to refer back to it
/// 
/// This combination of neural processing with external memory allows the DNC to solve problems that
/// require remembering and reasoning about complex relationships or sequences of information, like
/// navigating a subway map or following a multi-step recipe.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DifferentiableNeuralComputer<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (memory addressing regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Memory addressing regularization prevents soft addressing from becoming too diffuse or collapsing.
    /// This encourages the DNC to learn focused, interpretable memory access patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the DNC use memory effectively.
    ///
    /// Memory addressing regularization ensures:
    /// - Read/write heads focus on relevant memory locations
    /// - Addressing doesn't spread too thin across all locations
    /// - Memory operations are interpretable and efficient
    ///
    /// This is important because:
    /// - Focused addressing improves memory utilization
    /// - Sharp addressing patterns are more interpretable
    /// - Prevents wasting computation on irrelevant memory locations
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the memory addressing auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much memory addressing regularization contributes to the total loss.
    /// Typical values range from 0.001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much we encourage focused memory access.
    ///
    /// Common values:
    /// - 0.005 (default): Balanced addressing regularization
    /// - 0.001-0.003: Light regularization
    /// - 0.008-0.01: Strong regularization
    ///
    /// Higher values encourage sharper memory addressing patterns.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastMemoryAddressingLoss;

    /// <summary>
    /// Gets or sets the number of memory locations in the memory matrix.
    /// </summary>
    /// <value>An integer representing the number of rows in the memory matrix.</value>
    /// <remarks>
    /// <para>
    /// The memory size determines how many separate memory locations are available in the external memory matrix.
    /// Each memory location can store a vector of information of length MemoryWordSize. A larger memory size allows
    /// the DNC to store more information but increases computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the number of pages in the notepad.
    /// 
    /// Think of the memory as a notepad:
    /// - MemorySize is how many separate pages are in the notepad
    /// - More pages mean the system can store more separate pieces of information
    /// - Each page can hold a specific amount of data (determined by MemoryWordSize)
    /// 
    /// For example, if MemorySize is 128, the DNC has 128 different "pages" to write on and read from.
    /// </para>
    /// </remarks>
    private int _memorySize;

    /// <summary>
    /// Gets or sets the size of each memory word or location in the memory matrix.
    /// </summary>
    /// <value>An integer representing the number of columns in the memory matrix.</value>
    /// <remarks>
    /// <para>
    /// The memory word size determines how much information can be stored in each memory location.
    /// Each memory location stores a vector of this length. The memory word size should typically be compatible
    /// with the dimensions of the data being processed.
    /// </para>
    /// <para><b>For Beginners:</b> This is like how much information fits on each page of the notepad.
    /// 
    /// If we continue the notepad analogy:
    /// - MemoryWordSize is how many words or details can fit on each page
    /// - Larger word size means each page can store more detailed information
    /// - This affects how rich and detailed each stored memory can be
    /// 
    /// For example, if MemoryWordSize is 64, each "page" in memory can store 64 numbers,
    /// which could represent complex features or concepts.
    /// </para>
    /// </remarks>
    private int _memoryWordSize;

    /// <summary>
    /// Gets or sets the size of the controller network's output.
    /// </summary>
    /// <value>An integer representing the output dimension of the controller network.</value>
    /// <remarks>
    /// <para>
    /// The controller size determines the dimensionality of the controller network's output, which affects how
    /// much information the controller can process and how it interacts with the memory system. The controller
    /// is responsible for generating commands for the memory access mechanisms.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the brain power of the system that decides what to write and read.
    /// 
    /// Think of the controller as the "brain" of the DNC:
    /// - ControllerSize is how many neurons or processing units this brain has
    /// - A larger controller can learn more complex patterns and make more nuanced decisions
    /// - It determines how sophisticated the DNC's reasoning can be
    /// 
    /// The controller is what decides what information to write to memory, where to write it,
    /// and when and where to read from memory.
    /// </para>
    /// </remarks>
    private int _controllerSize;

    /// <summary>
    /// Gets or sets the number of read heads that can access the memory simultaneously.
    /// </summary>
    /// <value>An integer representing the number of read heads.</value>
    /// <remarks>
    /// <para>
    /// Read heads are mechanisms that allow the DNC to read from different locations in memory simultaneously.
    /// Multiple read heads enable the DNC to integrate information from different memory locations, which is
    /// useful for tasks that require relating multiple pieces of stored information.
    /// </para>
    /// <para><b>For Beginners:</b> This is like how many places in the notepad you can look at simultaneously.
    /// 
    /// Think of read heads as fingers holding different pages in the notepad:
    /// - ReadHeads is how many different pages you can look at at the same time
    /// - More read heads allow the system to combine information from different parts of memory
    /// - This helps when solving problems that require connecting multiple facts or steps
    /// 
    /// For example, with 4 read heads, the DNC could simultaneously recall 4 different facts
    /// from its memory and use them together to make a decision or prediction.
    /// </para>
    /// </remarks>
    private int _readHeads;

    /// <summary>
    /// Gets or sets the memory matrix that stores information.
    /// </summary>
    /// <value>A matrix of size MemorySize x MemoryWordSize.</value>
    /// <remarks>
    /// <para>
    /// The memory matrix is the external memory resource that the DNC uses to store and retrieve information.
    /// It consists of MemorySize locations, each capable of storing a vector of length MemoryWordSize.
    /// The DNC learns to write relevant information to memory and read it when needed.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual notepad where information is stored.
    /// 
    /// The memory matrix is like the notepad itself:
    /// - It's organized as a grid with MemorySize rows (pages)
    /// - Each row can store MemoryWordSize values (words on a page)
    /// - The DNC writes important information here and reads from it later
    /// - The contents of this matrix change as the DNC processes information
    /// 
    /// This external memory is what allows the DNC to remember and reason about
    /// complex information over long periods, unlike regular neural networks.
    /// </para>
    /// </remarks>
    private Matrix<T> _memory;

    /// <summary>
    /// Gets or sets the vector tracking which memory locations are free to be written to.
    /// </summary>
    /// <value>A vector of length MemorySize with values between 0 and 1.</value>
    /// <remarks>
    /// <para>
    /// The usage free vector tracks which memory locations are currently being used. Values close to 1 indicate
    /// that a location is free to be written to, while values close to 0 indicate that the location contains
    /// important information that should not be overwritten. This helps the DNC manage memory allocation efficiently.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks which pages in the notepad are available for writing.
    /// 
    /// Think of UsageFree as a record of which pages in the notepad are:
    /// - Empty or unimportant (values close to 1) and can be used for new information
    /// - Already containing important information (values close to 0) that shouldn't be erased
    /// 
    /// This helps the DNC decide where to write new information without erasing
    /// valuable information it might need later.
    /// </para>
    /// </remarks>
    private Vector<T> _usageFree;

    /// <summary>
    /// Gets or sets the vector that determines where to write in memory.
    /// </summary>
    /// <value>A vector of length MemorySize representing write attention weights.</value>
    /// <remarks>
    /// <para>
    /// The write weighting vector represents the attention over memory locations for writing. Higher values indicate
    /// that more information will be written to the corresponding memory location. This vector is typically sparse,
    /// focusing attention on a few relevant locations.
    /// </para>
    /// <para><b>For Beginners:</b> This determines where the DNC writes information in the notepad.
    /// 
    /// The WriteWeighting is like deciding which page in the notepad to write on:
    /// - It's a list of numbers between 0 and 1 for each page
    /// - Higher numbers mean more information gets written to that page
    /// - Usually, only one or a few pages have high values (the system focuses its attention)
    /// - The DNC learns which pages are best to write to for different types of information
    /// 
    /// This focused writing allows the system to organize information in a way it can find later.
    /// </para>
    /// </remarks>
    private Vector<T> _writeWeighting;

    /// <summary>
    /// Gets or sets the list of vectors that determine where to read from memory.
    /// </summary>
    /// <value>A list of vectors, each of length MemorySize, representing read attention weights.</value>
    /// <remarks>
    /// <para>
    /// The read weightings list contains one vector per read head, where each vector represents the attention over
    /// memory locations for that read head. Higher values indicate that more information will be read from the
    /// corresponding memory location. These vectors are typically sparse, focusing attention on a few relevant locations.
    /// </para>
    /// <para><b>For Beginners:</b> This determines which pages the DNC reads from in the notepad.
    /// 
    /// ReadWeightings is like deciding which pages to look at when retrieving information:
    /// - It's a list of attention patterns, one for each read head
    /// - For each read head, it specifies how much attention to give to each memory location
    /// - Higher values mean that location has more influence on what's being read
    /// - The DNC learns which pages to read from based on what information it needs
    /// 
    /// This allows the system to retrieve relevant information it previously stored.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _readWeightings;

    /// <summary>
    /// Gets or sets the vector that tracks the order in which memory locations were written to.
    /// </summary>
    /// <value>A vector of length MemorySize representing temporal write importance.</value>
    /// <remarks>
    /// <para>
    /// The precedence weighting vector tracks the recency of writes to each memory location. It is used to establish
    /// temporal links between memory locations, allowing the DNC to traverse memory in the order information was written.
    /// This enables the DNC to follow sequences or chains of reasoning.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks the order in which pages were written to in the notepad.
    /// 
    /// The PrecedenceWeighting is like keeping track of which pages were written to most recently:
    /// - It helps maintain a sense of time or sequence in the memory
    /// - Higher values indicate pages that were written to more recently
    /// - This allows the DNC to follow information in chronological order
    /// - It's how the system can remember and follow sequences of steps or events
    /// 
    /// With this information, the DNC can navigate through memory forwards or backwards in time,
    /// following the sequence in which information was stored.
    /// </para>
    /// </remarks>
    private Vector<T> _precedenceWeighting;

    /// <summary>
    /// Gets or sets the matrix representing temporal links between memory locations.
    /// </summary>
    /// <value>A matrix of size MemorySize x MemorySize representing temporal relationships.</value>
    /// <remarks>
    /// <para>
    /// The temporal link matrix represents the temporal relationships between memory locations. If location i was
    /// written to just before location j, then the temporal link matrix will have a high value at position [i, j].
    /// This allows the DNC to follow chains of information in the order they were written.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how pages in the notepad relate to each other in time sequence.
    /// 
    /// The TemporalLinkMatrix is like drawing arrows between pages in the notepad:
    /// - It shows which page was written before or after each other page
    /// - This creates a network of "next page" and "previous page" relationships
    /// - It allows the DNC to follow chains of related information
    /// - For example, it can follow a sequence of steps in a recipe or directions on a map
    /// 
    /// This matrix is what gives the DNC its ability to follow sequences and perform
    /// algorithm-like reasoning.
    /// </para>
    /// </remarks>
    private Matrix<T> _temporalLinkMatrix;

    /// <summary>
    /// Gets or sets the list of vectors read from memory.
    /// </summary>
    /// <value>A list of vectors, each of length MemoryWordSize, representing content read from memory.</value>
    /// <remarks>
    /// <para>
    /// The read vectors list contains one vector per read head, where each vector represents the information read
    /// from memory by that read head. These vectors are computed as weighted sums of memory rows, where the weights
    /// are determined by the corresponding read weighting vector.
    /// </para>
    /// <para><b>For Beginners:</b> This is the information the DNC has actually read from the notepad.
    /// 
    /// ReadVectors represents the information currently being looked at:
    /// - It's a list of content vectors, one for each read head
    /// - Each vector contains the information read from memory by that read head
    /// - This is the information the DNC uses to make its current decisions
    /// - It's like the notes the DNC has taken from reading its notepad
    /// 
    /// These read vectors, combined with the controller's output, are what the DNC uses
    /// to produce its final output for a given input.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _readVectors;

    /// <summary>
    /// Gets or sets the scalar activation function for the network.
    /// </summary>
    private IActivationFunction<T>? _activationFunction;

    /// <summary>
    /// Gets or sets the vector activation function for the network.
    /// </summary>
    private IVectorActivationFunction<T>? _vectorActivationFunction;

    /// <summary>
    /// The output weight matrix for combining controller output with read vectors.
    /// </summary>
    private Matrix<T> _outputWeights;

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="DifferentiableNeuralComputer{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="memorySize">The number of memory locations in the memory matrix.</param>
    /// <param name="memoryWordSize">The size of each memory word or location.</param>
    /// <param name="controllerSize">The size of the controller network's output.</param>
    /// <param name="readHeads">The number of read heads that can access the memory simultaneously.</param>
    /// <param name="activationFunction">The scalar activation function to use. If null, defaults based on task type.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Differentiable Neural Computer with the specified architecture and memory parameters.
    /// It sets up the memory matrix, usage tracking vectors, read/write weightings, and temporal link matrix. The memory
    /// is initialized with small random values, and the usage vector is initialized to indicate that all memory locations
    /// are free.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new DNC with a specific notepad size and reading capacity.
    /// 
    /// When creating a new DNC:
    /// - The architecture defines the neural network's structure
    /// - memorySize determines how many pages are in the notepad
    /// - memoryWordSize determines how much information fits on each page
    /// - controllerSize determines how powerful the "brain" of the system is
    /// - readHeads determines how many pages can be read simultaneously
    /// 
    /// Think of it like configuring a new assistant with specific mental capabilities and
    /// a notepad of specific size to help them remember and reason about information.
    /// </para>
    /// </remarks>
    public DifferentiableNeuralComputer(
        NeuralNetworkArchitecture<T> architecture,
        int memorySize,
        int memoryWordSize,
        int controllerSize,
        int readHeads,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? activationFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastMemoryAddressingLoss = NumOps.Zero;

        _memorySize = memorySize;
        _memoryWordSize = memoryWordSize;
        _controllerSize = controllerSize;
        _readHeads = readHeads;
        _memory = new Matrix<T>(_memorySize, _memoryWordSize);
        _usageFree = new Vector<T>(_memorySize);
        _writeWeighting = new Vector<T>(_memorySize);
        _readWeightings = new List<Vector<T>>();
        _readVectors = new List<Vector<T>>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // Calculate combinedSize (controller output + read vectors)
        int outputSize = architecture.OutputSize;
        int combinedSize = controllerSize + (_readHeads * _memoryWordSize);

        // Determine appropriate activation function based on task type
        _activationFunction = activationFunction ?? NeuralNetworkHelper<T>.GetDefaultActivationFunction(architecture.TaskType);

        for (int i = 0; i < _readHeads; i++)
        {
            _readWeightings.Add(new Vector<T>(_memorySize));
        }

        _precedenceWeighting = new Vector<T>(_memorySize);
        _temporalLinkMatrix = new Matrix<T>(_memorySize, _memorySize);
        _outputWeights = new Matrix<T>(combinedSize, outputSize);

        InitializeOutputWeights();
        InitializeMemory();
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DifferentiableNeuralComputer{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="memorySize">The number of memory locations in the memory matrix.</param>
    /// <param name="memoryWordSize">The size of each memory word or location.</param>
    /// <param name="controllerSize">The size of the controller network's output.</param>
    /// <param name="readHeads">The number of read heads that can access the memory simultaneously.</param>
    /// <param name="activationFunction">The scalar activation function to use. If null, defaults based on task type.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Differentiable Neural Computer with the specified architecture and memory parameters.
    /// It sets up the memory matrix, usage tracking vectors, read/write weightings, and temporal link matrix. The memory
    /// is initialized with small random values, and the usage vector is initialized to indicate that all memory locations
    /// are free.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new DNC with a specific notepad size and reading capacity.
    /// 
    /// When creating a new DNC:
    /// - The architecture defines the neural network's structure
    /// - memorySize determines how many pages are in the notepad
    /// - memoryWordSize determines how much information fits on each page
    /// - controllerSize determines how powerful the "brain" of the system is
    /// - readHeads determines how many pages can be read simultaneously
    /// 
    /// Think of it like configuring a new assistant with specific mental capabilities and
    /// a notepad of specific size to help them remember and reason about information.
    /// </para>
    /// </remarks>
    public DifferentiableNeuralComputer(
        NeuralNetworkArchitecture<T> architecture,
        int memorySize,
        int memoryWordSize,
        int controllerSize,
        int readHeads,
        ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastMemoryAddressingLoss = NumOps.Zero;

        _memorySize = memorySize;
        _memoryWordSize = memoryWordSize;
        _controllerSize = controllerSize;
        _readHeads = readHeads;
        _memory = new Matrix<T>(_memorySize, _memoryWordSize);
        _usageFree = new Vector<T>(_memorySize);
        _writeWeighting = new Vector<T>(_memorySize);
        _readWeightings = new List<Vector<T>>();
        _readVectors = new List<Vector<T>>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        // Calculate combinedSize (controller output + read vectors)
        int outputSize = architecture.OutputSize;
        int combinedSize = controllerSize + (_readHeads * _memoryWordSize);

        // Determine appropriate vector activation function based on task type
        _vectorActivationFunction = vectorActivationFunction ?? NeuralNetworkHelper<T>.GetDefaultVectorActivationFunction(architecture.TaskType);

        for (int i = 0; i < _readHeads; i++)
        {
            _readWeightings.Add(new Vector<T>(_memorySize));
        }

        _precedenceWeighting = new Vector<T>(_memorySize);
        _temporalLinkMatrix = new Matrix<T>(_memorySize, _memorySize);
        _outputWeights = new Matrix<T>(combinedSize, outputSize);

        InitializeOutputWeights();
        InitializeMemory();
        InitializeLayers();
    }

    private void InitializeOutputWeights()
    {
        // Initialize with small random values
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                _outputWeights[i, j] = NumOps.FromDouble((Random.NextDouble() * 0.2) - 0.1);
            }
        }
    }

    /// <summary>
    /// Initializes the memory matrix and usage tracking vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the memory matrix with small random values and sets the usage free vector to indicate
    /// that all memory locations are initially free. The small random values in memory provide a starting point that
    /// breaks symmetry and helps the learning process.
    /// </para>
    /// <para><b>For Beginners:</b> This prepares the notepad with a clean slate.
    /// 
    /// When initializing the memory:
    /// - The memory pages are filled with very small random values
    ///   (not exactly zero to help the learning process start more effectively)
    /// - The system marks all pages as free and available for writing
    /// - This is like preparing a fresh notepad before the DNC starts using it
    /// 
    /// Starting with small random values instead of zeros helps the DNC learn more
    /// effectively by breaking the symmetry that exact zeros would create.
    /// </para>
    /// </remarks>
    private void InitializeMemory()
    {
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                _memory[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1);
            }

            _usageFree[i] = NumOps.FromDouble(1.0);
        }
    }

    /// <summary>
    /// Computes the auxiliary loss for memory addressing regularization.
    /// </summary>
    /// <returns>The computed memory addressing auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes entropy-based regularization for memory read/write addressing.
    /// It encourages focused, sharp addressing patterns while preventing diffuse addressing.
    /// Formula: L = -Σ_heads H(addressing) where H is entropy of addressing weights
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how focused the DNC's memory access is.
    ///
    /// Memory addressing regularization works by:
    /// 1. Measuring entropy of read/write addressing weights
    /// 2. Lower entropy means more focused, sharp addressing
    /// 3. Higher entropy means diffuse, spread-out addressing
    /// 4. We minimize negative entropy to encourage focused access
    ///
    /// This helps because:
    /// - Focused addressing is more interpretable
    /// - Sharp addressing improves memory efficiency
    /// - Prevents wasting computation on many irrelevant locations
    /// - Encourages the DNC to learn clear memory access patterns
    ///
    /// The auxiliary loss is added to the main task loss during training.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            _lastMemoryAddressingLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute negative entropy over read and write addressing weights
        // to encourage focused, sharp memory access patterns
        T totalNegativeEntropy = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);  // For numerical stability
        T oneMinusEpsilon = NumOps.Subtract(NumOps.One, epsilon);

        // Compute negative entropy for read addressing weights (vectorized)
        foreach (var readWeighting in _readWeightings)
        {
            // Vectorized entropy: H = -Σ(p * log(p))
            var pTensor = new Tensor<T>(readWeighting.ToArray(), [readWeighting.Length]);
            // Clamp p to [epsilon, 1-epsilon] to avoid log(0) and log(>1)
            var pClamped = Engine.TensorClamp(pTensor, epsilon, oneMinusEpsilon);
            var logP = Engine.TensorLog(pClamped);
            var pLogP = Engine.TensorMultiply(pClamped, logP);
            T entropy = Engine.TensorSum(pLogP);
            // Negative entropy (we want to minimize this, encouraging sharp peaks)
            totalNegativeEntropy = NumOps.Subtract(totalNegativeEntropy, entropy);
        }

        // Compute negative entropy for write addressing weights (vectorized)
        if (_writeWeighting != null)
        {
            // Vectorized entropy: H = -Σ(p * log(p))
            var pTensor = new Tensor<T>(_writeWeighting.ToArray(), [_writeWeighting.Length]);
            // Clamp p to [epsilon, 1-epsilon] to avoid log(0) and log(>1)
            var pClamped = Engine.TensorClamp(pTensor, epsilon, oneMinusEpsilon);
            var logP = Engine.TensorLog(pClamped);
            var pLogP = Engine.TensorMultiply(pClamped, logP);
            T entropy = Engine.TensorSum(pLogP);
            totalNegativeEntropy = NumOps.Subtract(totalNegativeEntropy, entropy);
        }

        // Store unweighted loss for diagnostics
        _lastMemoryAddressingLoss = totalNegativeEntropy;

        // Apply auxiliary loss weight and return weighted loss
        return NumOps.Multiply(totalNegativeEntropy, AuxiliaryLossWeight);
    }

    /// <summary>
    /// Gets diagnostic information about the memory addressing auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about memory addressing regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about memory addressing regularization, including
    /// addressing entropy, number of read/write heads, and configuration parameters.
    /// This information is useful for monitoring memory access patterns and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how the DNC accesses memory.
    ///
    /// The diagnostics include:
    /// - Total addressing entropy loss (how focused memory access is)
    /// - Weight applied to the regularization
    /// - Number of read and write heads
    /// - Whether addressing regularization is enabled
    ///
    /// This helps you:
    /// - Monitor if memory addressing is focused or diffuse
    /// - Debug issues with memory access patterns
    /// - Understand the impact of regularization on memory usage
    ///
    /// You can use this information to adjust regularization weights for better memory utilization.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalMemoryAddressingLoss", _lastMemoryAddressingLoss?.ToString() ?? "0" },
            { "AddressingWeight", AuxiliaryLossWeight?.ToString() ?? "0.005" },
            { "UseMemoryAddressingRegularization", UseAuxiliaryLoss.ToString() },
            { "NumberOfReadHeads", _readHeads.ToString() },
            { "MemorySize", _memorySize.ToString() },
            { "MemoryWordSize", _memoryWordSize.ToString() }
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
    /// Initializes the layers of the Differentiable Neural Computer based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers of the DNC. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default layers are created based on the architecture's specifications and
    /// the DNC's memory parameters. The layers typically include a controller network and interface layers for
    /// interacting with the memory.
    /// </para>
    /// <para><b>For Beginners:</b> This builds the neural network "brain" of the DNC.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers suitable for a DNC
    /// - These layers include a controller (the main processing network) and interfaces
    ///   to interact with the memory system
    /// - The network calculates how large the interface needs to be based on the memory size
    /// 
    /// This is like assembling the thinking and decision-making parts of the system
    /// that will work together with the memory to solve problems.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDNCLayers(Architecture, _controllerSize, _memoryWordSize, _readHeads, CalculateDNCInterfaceSize(_memoryWordSize, _readHeads)));
        }
    }

    /// <summary>
    /// Calculates the size of the interface vector required for memory operations.
    /// </summary>
    /// <param name="memoryWordSize">The size of each memory word or location.</param>
    /// <param name="readHeads">The number of read heads.</param>
    /// <returns>The total size of the interface vector.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the size of the interface vector that the controller needs to produce to interact with
    /// the memory system. The interface vector includes components for write operations, erase operations, read operations,
    /// and various control gates and modes.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how much information the brain needs to produce to control the memory.
    /// 
    /// The interface size calculation:
    /// - Determines how many control signals the neural network needs to generate
    /// - Includes signals for writing to memory, erasing from memory, reading from memory
    /// - Also includes various control gates and mode switches
    /// - The larger the memory word size and the more read heads, the larger this interface needs to be
    /// 
    /// Think of it like figuring out how many different controls and buttons are needed on
    /// a control panel to properly operate the memory system.
    /// </para>
    /// </remarks>
    private static int CalculateDNCInterfaceSize(int memoryWordSize, int readHeads)
    {
        return memoryWordSize + // Write vector
               memoryWordSize + // Erase vector
               memoryWordSize + // Write key
               3 + // Write strength, allocation gate, write gate
               readHeads * memoryWordSize + // Read keys
               readHeads + // Read strengths
               3 * readHeads; // Read modes (backward, content, forward)
    }

    /// <summary>
    /// Updates the parameters of all layers in the Differentiable Neural Computer.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter vector among all the layers in the network.
    /// Each layer receives a portion of the parameter vector corresponding to its number of parameters.
    /// The method keeps track of the starting index for each layer's parameters in the input vector.
    /// </para>
    /// <para><b>For Beginners:</b> This updates all the internal values of the neural network at once.
    /// 
    /// When updating parameters:
    /// - The input is a long list of numbers representing all values in the entire network
    /// - The method divides this list into smaller chunks
    /// - Each layer gets its own chunk of values
    /// - The layers use these values to adjust their internal settings
    /// 
    /// This method is typically used during training or when loading a pre-trained model,
    /// allowing all network parameters to be updated at once.
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
    /// Makes a prediction using the Differentiable Neural Computer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input data through the DNC to make a prediction. It processes the input through
    /// the controller network, calculates memory interactions, and produces an output. For sequential inputs,
    /// this method should be called repeatedly, with the DNC maintaining its memory state between calls.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the DNC processes new information and makes predictions.
    /// 
    /// The prediction process works like this:
    /// 1. Input data is processed by the controller neural network
    /// 2. The controller produces signals that determine how to interact with memory
    /// 3. Based on these signals, information is written to and read from the memory
    /// 4. The final output combines the controller's processing with information read from memory
    /// 
    /// Unlike traditional neural networks, the DNC maintains its memory state between predictions,
    /// allowing it to build up knowledge over a sequence of inputs.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return ProcessInput(input, false);
    }

    /// <summary>
    /// Trains the Differentiable Neural Computer on a single batch of data.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method trains the DNC on a single batch of data using backpropagation through time (BPTT).
    /// It processes the input through the network, computes the error with respect to the expected output,
    /// and updates the network parameters to reduce this error. For sequential data, this method should
    /// be called with sequences of inputs and expected outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the DNC learns from examples.
    /// 
    /// The training process works like this:
    /// 1. The input is processed through the network (like in prediction)
    /// 2. The output is compared to the expected output to calculate the error
    /// 3. This error is propagated backward through the network
    /// 4. The network's parameters are updated to reduce this error
    /// 
    /// Unlike traditional neural networks, DNCs must be careful to propagate errors through
    /// their memory operations as well as through the neural network components.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Process the input through the network
        Tensor<T> output = ProcessInput(input, true);

        // Calculate error/loss
        var flattenedPredictions = output.ToVector();
        var flattenedExpected = expectedOutput.ToVector();

        // Calculate and store the loss value
        LastLoss = _lossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        // Calculate gradients from the loss
        Vector<T> outputGradients = _lossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);

        // Backpropagate the error through the network
        Tensor<T> inputGradientsTensor = Backpropagate(Tensor<T>.FromVector(outputGradients));
        Vector<T> inputGradients = inputGradientsTensor.ToVector();

        // Get parameter gradients
        Vector<T> parameterGradients = GetParameterGradients();

        // Apply gradient clipping to prevent exploding gradients
        parameterGradients = ClipGradient(parameterGradients);

        // Create optimizer (here we use a simple gradient descent optimizer)
        var optimizer = new GradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Get current parameters
        Vector<T> currentParameters = GetParameters();

        // Update parameters using the optimizer
        Vector<T> updatedParameters = optimizer.UpdateParameters(currentParameters, parameterGradients);

        // Apply updated parameters
        UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Processes an input through the DNC, updating memory state and producing an output.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <param name="isTraining">Whether the DNC is in training mode.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <remarks>
    /// <para>
    /// This method handles the core processing of the DNC. It passes the input through the controller,
    /// calculates memory interactions based on the controller's output, and produces a final output
    /// combining the controller's processing with the information read from memory.
    /// </para>
    /// <para><b>For Beginners:</b> This is the core function that controls how the DNC works with its memory.
    /// 
    /// For each input:
    /// 1. The controller processes the input and produces control signals
    /// 2. These signals determine what to write to memory and where
    /// 3. They also determine what to read from memory and from where
    /// 4. The final output combines what the controller processed with what was read from memory
    /// 
    /// This is like a person thinking about new information (controller processing)
    /// while also consulting their notes (memory reading) to produce a response.
    /// </para>
    /// </remarks>
    private Tensor<T> ProcessInput(Tensor<T> input, bool isTraining)
    {
        // Set training mode for all layers
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(isTraining);
        }

        // Previous read vectors are concatenated with the input to provide context
        Tensor<T> controllerInput = PrepareControllerInput(input);

        // Process the input through the controller network
        Tensor<T> controllerOutput = ProcessThroughController(controllerInput);

        // Parse the controller output to get memory interface signals
        var interfaceOutput = ParseControllerOutput(controllerOutput);

        // Update memory based on interface signals
        UpdateMemory(interfaceOutput);

        // Read from memory based on interface signals
        List<Vector<T>> newReadVectors = ReadFromMemory(interfaceOutput);

        // Update read vectors for the next time step
        _readVectors = newReadVectors;

        // Combine controller output with read vectors to produce final output
        Tensor<T> finalOutput = CombineControllerOutputWithReadVectors(controllerOutput, newReadVectors);

        return finalOutput;
    }

    /// <summary>
    /// Prepares the input for the controller by concatenating it with previous read vectors.
    /// </summary>
    /// <param name="input">The raw input tensor.</param>
    /// <returns>A tensor combining the input with previous read vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method prepares the input for the controller by concatenating it with the previous read vectors.
    /// This provides the controller with context from what was previously read from memory. If there are no
    /// previous read vectors (e.g., at the first time step), zero vectors are used instead.
    /// </para>
    /// <para><b>For Beginners:</b> This combines new information with what was previously read from memory.
    /// 
    /// Think of it like:
    /// - Before studying new material, you review your previous notes
    /// - This helps connect new information with what you already know
    /// - The combined information (new input + previous notes) is what the brain processes
    /// 
    /// This step is crucial for the DNC to maintain continuity across a sequence of inputs.
    /// </para>
    /// </remarks>
    private Tensor<T> PrepareControllerInput(Tensor<T> input)
    {
        // Check if we have previous read vectors
        if (_readVectors.Count == 0)
        {
            // Initialize read vectors with zeros if this is the first step
            for (int i = 0; i < _readHeads; i++)
            {
                _readVectors.Add(new Vector<T>(_memoryWordSize));
                for (int j = 0; j < _memoryWordSize; j++)
                {
                    _readVectors[i][j] = NumOps.Zero;
                }
            }
        }

        // Flatten input tensor
        Vector<T> flattenedInput = input.ToVector();

        // Calculate the total size of the controller input
        int inputSize = flattenedInput.Length;
        int totalSize = inputSize + (_readHeads * _memoryWordSize);

        // Create a new vector for the combined input
        Vector<T> combinedInput = new Vector<T>(totalSize);

        // Copy the input
        for (int i = 0; i < inputSize; i++)
        {
            combinedInput[i] = flattenedInput[i];
        }

        // Copy the read vectors
        int offset = inputSize;
        for (int i = 0; i < _readHeads; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                combinedInput[offset++] = _readVectors[i][j];
            }
        }

        // Convert to tensor with appropriate shape for the controller
        return Tensor<T>.FromVector(combinedInput).Reshape(1, totalSize);
    }

    /// <summary>
    /// Processes the prepared input through the controller network.
    /// </summary>
    /// <param name="controllerInput">The prepared input tensor.</param>
    /// <returns>The output tensor from the controller network.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the prepared input through the controller network, which is responsible for
    /// processing the input and generating the interface signals for memory interactions. The controller
    /// is typically a recurrent neural network (RNN), long short-term memory network (LSTM), or another
    /// type of neural network capable of processing sequential information.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the "brain" of the DNC processes the input information.
    /// 
    /// The controller network:
    /// - Takes the combined input (new data + previously read memory)
    /// - Processes this information using neural network layers
    /// - Produces outputs that will control memory operations and contribute to the final output
    /// - Is like the "thinking" part of the system that decides what to remember and what to recall
    /// 
    /// This step is analogous to a person's brain processing new information and deciding
    /// what to write down and what to look up in their notes.
    /// </para>
    /// </remarks>
    private Tensor<T> ProcessThroughController(Tensor<T> controllerInput)
    {
        // Process through all layers except the last one, which is typically an output layer
        Tensor<T> currentOutput = controllerInput;

        for (int i = 0; i < Layers.Count; i++)
        {
            currentOutput = Layers[i].Forward(currentOutput);

            // Store input/output for each layer if in training mode
            if (IsTrainingMode)
            {
                _layerInputs[i] = currentOutput;
            }
        }

        return currentOutput;
    }

    /// <summary>
    /// Parses the controller output to extract memory interface signals.
    /// </summary>
    /// <param name="controllerOutput">The output tensor from the controller.</param>
    /// <returns>A structure containing parsed interface signals for memory operations.</returns>
    /// <remarks>
    /// <para>
    /// This method parses the controller's output to extract the various signals needed for memory operations.
    /// These include write and erase vectors, read and write weightings, and various control gates and modes.
    /// The method assumes that the controller output follows a specific format where different sections
    /// correspond to different interface components.
    /// </para>
    /// <para><b>For Beginners:</b> This interprets the controller's output as instructions for memory operations.
    /// 
    /// Think of it like translating the brain's signals into specific instructions:
    /// - What information to write to memory
    /// - Where to write it
    /// - What information to erase from memory
    /// - Where to read from
    /// - How to navigate between related pieces of information
    /// 
    /// This step converts the neural network's output into specific memory operations,
    /// allowing the DNC to use its memory in a structured way.
    /// </para>
    /// </remarks>
    private MemoryInterfaceSignals ParseControllerOutput(Tensor<T> controllerOutput)
    {
        // For simplicity, we'll assume controllerOutput is a 2D tensor with shape [1, interfaceSize]
        Vector<T> interfaceVector = controllerOutput.ToVector();

        // Parse the interface vector into its components
        MemoryInterfaceSignals signals = new MemoryInterfaceSignals
        {
            WriteVector = ExtractVector(interfaceVector, 0, _memoryWordSize),
            EraseVector = ExtractVector(interfaceVector, _memoryWordSize, _memoryWordSize),
            ReadKeys = [],
            ReadStrengths = [],
            WriteKey = ExtractVector(interfaceVector, 2 * _memoryWordSize, _memoryWordSize),
            WriteStrength = interfaceVector[3 * _memoryWordSize],
            AllocationGate = SigmoidActivation(interfaceVector[3 * _memoryWordSize + 1]),
            WriteGate = SigmoidActivation(interfaceVector[3 * _memoryWordSize + 2]),
            ReadModes = []
        };

        // Extract read keys and strengths
        int offset = 3 * _memoryWordSize + 3;
        for (int i = 0; i < _readHeads; i++)
        {
            signals.ReadKeys.Add(ExtractVector(interfaceVector, offset, _memoryWordSize));
            offset += _memoryWordSize;
            signals.ReadStrengths.Add(interfaceVector[offset++]);
        }

        // Extract read modes (backward, content, forward)
        for (int i = 0; i < _readHeads; i++)
        {
            Vector<T> readMode = new Vector<T>(3);
            for (int j = 0; j < 3; j++)
            {
                readMode[j] = SigmoidActivation(interfaceVector[offset++]);
            }
            // Normalize read modes to sum to 1
            T sum = NumOps.Add(NumOps.Add(readMode[0], readMode[1]), readMode[2]);
            for (int j = 0; j < 3; j++)
            {
                readMode[j] = NumOps.Divide(readMode[j], sum);
            }
            signals.ReadModes.Add(readMode);
        }

        return signals;
    }

    /// <summary>
    /// Extracts a vector from a larger vector at a specified offset.
    /// </summary>
    /// <param name="source">The source vector to extract from.</param>
    /// <param name="offset">The starting index for extraction.</param>
    /// <param name="length">The number of elements to extract.</param>
    /// <returns>A new vector containing the extracted elements.</returns>
    private Vector<T> ExtractVector(Vector<T> source, int offset, int length)
    {
        Vector<T> result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = source[offset + i];
        }
        return result;
    }

    /// <summary>
    /// Applies the sigmoid activation function to a scalar value.
    /// </summary>
    /// <param name="value">The input value.</param>
    /// <returns>The sigmoid of the input value.</returns>
    private T SigmoidActivation(T value)
    {
        // Convert to double for calculation
        double doubleValue = Convert.ToDouble(value);
        double sigmoid = 1.0 / (1.0 + Math.Exp(-doubleValue));
        return NumOps.FromDouble(sigmoid);
    }

    /// <summary>
    /// Updates the memory matrix based on the interface signals.
    /// </summary>
    /// <param name="signals">The interface signals for memory operations.</param>
    /// <remarks>
    /// <para>
    /// This method updates the DNC's memory matrix based on the interface signals generated by the controller.
    /// It calculates write weightings based on content-based addressing and allocation, writes new information
    /// to memory, and updates the usage free vector and temporal link matrix to track memory usage and relationships.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the DNC writes new information to its memory.
    /// 
    /// The memory update process:
    /// 1. Determines where to write based on content similarity and free space
    /// 2. Writes new information to the selected locations
    /// 3. Updates tracking information about what memory is being used
    /// 4. Updates the temporal links that connect related information
    /// 
    /// This is like a person deciding where to write in their notepad, writing down the information,
    /// and creating references between related notes.
    /// </para>
    /// </remarks>
    private void UpdateMemory(MemoryInterfaceSignals signals)
    {
        // Calculate write weighting based on content addressing and allocation
        Vector<T> contentWeighting = ContentAddressing(_memory, signals.WriteKey, signals.WriteStrength);
        Vector<T> allocationWeighting = Allocate(_usageFree, _precedenceWeighting);

        // Combine content and allocation weightings
        _writeWeighting = CombineWeightings(contentWeighting, allocationWeighting, signals.AllocationGate);

        // Apply the write gate
        for (int i = 0; i < _memorySize; i++)
        {
            _writeWeighting[i] = NumOps.Multiply(_writeWeighting[i], signals.WriteGate);
        }

        // Write to memory
        WriteToMemory(signals.WriteVector, signals.EraseVector);

        // Update usage vector
        UpdateUsage();

        // Update precedence weighting
        UpdatePrecedence();

        // Update temporal link matrix
        UpdateTemporalLinks();
    }

    /// <summary>
    /// Reads from memory based on the interface signals.
    /// </summary>
    /// <param name="signals">The interface signals for memory operations.</param>
    /// <returns>A list of vectors read from memory.</returns>
    /// <remarks>
    /// <para>
    /// This method reads from the DNC's memory matrix based on the interface signals generated by the controller.
    /// It calculates read weightings based on content-based addressing and temporal links, then uses these weightings
    /// to read information from memory.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the DNC retrieves information from its memory.
    /// 
    /// The reading process:
    /// 1. Determines where to read from based on content similarity and temporal links
    /// 2. Calculates weightings for each memory location
    /// 3. Reads information from memory locations based on these weightings
    /// 4. Returns the read information to be used in the network's output
    /// 
    /// This is like a person deciding which pages in their notepad to reference,
    /// reading from those pages, and using that information in their thinking.
    /// </para>
    /// </remarks>
    private List<Vector<T>> ReadFromMemory(MemoryInterfaceSignals signals)
    {
        List<Vector<T>> readVectors = new List<Vector<T>>();

        // Calculate read weightings for each read head
        for (int i = 0; i < _readHeads; i++)
        {
            // Calculate content-based addressing
            Vector<T> contentWeighting = ContentAddressing(_memory, signals.ReadKeys[i], signals.ReadStrengths[i]);

            // Calculate backward weighting (temporal links)
            Vector<T> backwardWeighting = _temporalLinkMatrix.Multiply(_readWeightings[i]);

            // Calculate forward weighting (temporal links)
            Vector<T> forwardWeighting = _temporalLinkMatrix.Transpose().Multiply(_readWeightings[i]);

            // Combine weightings based on read modes
            _readWeightings[i] = new Vector<T>(_memorySize);
            for (int j = 0; j < _memorySize; j++)
            {
                _readWeightings[i][j] = NumOps.Add(
                    NumOps.Add(
                        NumOps.Multiply(backwardWeighting[j], signals.ReadModes[i][0]),
                        NumOps.Multiply(contentWeighting[j], signals.ReadModes[i][1])
                    ),
                    NumOps.Multiply(forwardWeighting[j], signals.ReadModes[i][2])
                );
            }

            // Read from memory using the calculated weighting
            Vector<T> readVector = new Vector<T>(_memoryWordSize);
            for (int j = 0; j < _memoryWordSize; j++)
            {
                readVector[j] = NumOps.Zero;
                for (int k = 0; k < _memorySize; k++)
                {
                    readVector[j] = NumOps.Add(
                        readVector[j],
                        NumOps.Multiply(_readWeightings[i][k], _memory[k, j])
                    );
                }
            }

            readVectors.Add(readVector);
        }

        return readVectors;
    }

    /// <summary>
    /// Combines the controller output with read vectors to produce the final output.
    /// </summary>
    /// <param name="controllerOutput">The output tensor from the controller.</param>
    /// <param name="readVectors">The list of vectors read from memory.</param>
    /// <returns>The final output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method implements a more sophisticated combination of controller output with read vectors
    /// using a learned linear transformation. It uses a dedicated output matrix to properly weight
    /// the contributions from the controller and each read vector, allowing the network to learn
    /// the optimal way to combine these sources of information.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineControllerOutputWithReadVectors(Tensor<T> controllerOutput, List<Vector<T>> readVectors)
    {
        // Determine dimensions
        int outputSize = Architecture.OutputSize;

        // Extract the controller's direct output (excluding memory interface signals)
        // Determine the interface size to know how much of the controller output to use
        int interfaceSize = CalculateDNCInterfaceSize(_memoryWordSize, _readHeads);
        int controllerDirectOutputSize = controllerOutput.Shape[1] - interfaceSize;

        // Create empty output tensor with the right shape
        Tensor<T> finalOutput = new Tensor<T>(new[] { 1, outputSize });

        // Get the direct controller contribution from the first part of the controller output
        Tensor<T> controllerDirectOutput = controllerOutput.Slice(0, 0, 1, controllerDirectOutputSize);

        // Create a combined vector of controller output and read vectors for matrix multiplication
        int combinedSize = controllerDirectOutputSize + (_readHeads * _memoryWordSize);
        Vector<T> combinedVector = new Vector<T>(combinedSize);

        // Copy controller direct output to combined vector
        for (int i = 0; i < controllerDirectOutputSize; i++)
        {
            combinedVector[i] = controllerDirectOutput[0, i];
        }

        // Copy read vectors to combined vector
        int offset = controllerDirectOutputSize;
        for (int i = 0; i < _readHeads; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                combinedVector[offset++] = readVectors[i][j];
            }
        }

        // Apply learnable output matrix to combined vector
        for (int i = 0; i < outputSize; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < combinedSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(combinedVector[j], _outputWeights[j, i]));
            }
            finalOutput[0, i] = sum;
        }

        // Apply appropriate activation function based on task type
        NeuralNetworkHelper<T>.ApplyOutputActivation(finalOutput, Architecture);

        return finalOutput;
    }

    /// <summary>
    /// Calculates content-based addressing weights based on similarity between memory and a key.
    /// </summary>
    /// <param name="memory">The memory matrix.</param>
    /// <param name="key">The key vector to compare with memory.</param>
    /// <param name="strength">The strength of the addressing (sharpness of the focus).</param>
    /// <returns>A vector of weights representing content-based addressing.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates weights for addressing memory based on the similarity between memory rows and a key vector.
    /// The similarity is measured using cosine similarity, and the strength parameter controls how focused the attention is
    /// (higher strength means more focus on the most similar rows).
    /// </para>
    /// <para><b>For Beginners:</b> This finds memory locations that contain similar content to what you're looking for.
    /// 
    /// It works like:
    /// 1. The key vector is what you're looking for (like a search query)
    /// 2. Each memory location is compared to this key to see how similar they are
    /// 3. The strength controls how picky you are (higher means only very similar locations get attention)
    /// 4. The result is a set of weights indicating which memory locations are most relevant
    /// 
    /// This is like looking through a notebook for pages containing information similar to what you need.
    /// </para>
    /// </remarks>
    private Vector<T> ContentAddressing(Matrix<T> memory, Vector<T> key, T strength)
    {
        Vector<T> similarityScores = new Vector<T>(_memorySize);

        // Calculate cosine similarity between key and each memory row
        for (int i = 0; i < _memorySize; i++)
        {
            Vector<T> memoryRow = memory.GetRow(i);
            similarityScores[i] = CosineSimilarity(key, memoryRow);
        }

        // Apply strength (temperature) to similarities
        for (int i = 0; i < _memorySize; i++)
        {
            similarityScores[i] = NumOps.Multiply(similarityScores[i], strength);
        }

        // Apply softmax to get weights
        return Softmax(similarityScores);
    }

    /// <summary>
    /// Calculates the cosine similarity between two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The cosine similarity between the vectors.</returns>
    private T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        // Vectorized cosine similarity using IEngine tensor operations
        var tensorA = new Tensor<T>(a.ToArray(), [a.Length]);
        var tensorB = new Tensor<T>(b.ToArray(), [b.Length]);

        // Calculate dot product: sum(a * b)
        var product = Engine.TensorMultiply(tensorA, tensorB);
        T dotProduct = Engine.TensorSum(product);

        // Calculate magnitudes using vectorized operations
        // magnitudeA = sqrt(sum(a * a))
        var aSquared = Engine.TensorMultiply(tensorA, tensorA);
        T sumASquared = Engine.TensorSum(aSquared);
        T magnitudeA = NumOps.Sqrt(sumASquared);

        // magnitudeB = sqrt(sum(b * b))
        var bSquared = Engine.TensorMultiply(tensorB, tensorB);
        T sumBSquared = Engine.TensorSum(bSquared);
        T magnitudeB = NumOps.Sqrt(sumBSquared);

        // Avoid division by zero
        if (MathHelper.AlmostEqual(magnitudeA, NumOps.Zero) || MathHelper.AlmostEqual(magnitudeB, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        // Calculate cosine similarity
        return NumOps.Divide(dotProduct, NumOps.Multiply(magnitudeA, magnitudeB));
    }

    /// <summary>
    /// Applies the softmax function to a vector.
    /// </summary>
    /// <param name="vector">The input vector.</param>
    /// <returns>The softmax of the input vector.</returns>
    private Vector<T> Softmax(Vector<T> vector)
    {
        // Vectorized softmax using IEngine tensor operations
        // Convert vector to tensor for vectorized operations
        var inputTensor = new Tensor<T>(vector.ToArray(), [vector.Length]);

        // Find maximum for numerical stability using ReduceMax on axis 0 (the only axis)
        var maxTensor = Engine.ReduceMax(inputTensor, [0], keepDims: true, out _);
        T maxVal = maxTensor[0];

        // Create max tensor for broadcasting
        var maxBroadcast = new Tensor<T>([vector.Length]);
        Engine.TensorFill(maxBroadcast, maxVal);

        // Compute exp(x - max) using vectorized operations
        var shifted = Engine.TensorSubtract(inputTensor, maxBroadcast);
        var expTensor = Engine.TensorExp(shifted);

        // Compute sum of exponentials
        T sum = Engine.TensorSum(expTensor);

        // Normalize by dividing by sum
        var result = Engine.TensorDivideScalar(expTensor, sum);

        // Convert back to Vector
        var resultArray = result.ToArray();
        var expVector = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            expVector[i] = resultArray[i];
        }

        return expVector;
    }

    /// <summary>
    /// Calculates allocation weights based on the usage free vector and precedence vector.
    /// </summary>
    /// <param name="usageFree">The vector tracking which memory locations are free.</param>
    /// <param name="precedenceWeighting">The vector tracking the order of memory writes.</param>
    /// <returns>A vector of weights for memory allocation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates weights for allocating memory based on which locations are free (available for writing)
    /// and the order in which memory has been written to previously. It sorts memory locations by their usage and
    /// allocates more weight to freer locations.
    /// </para>
    /// <para><b>For Beginners:</b> This decides where to write new information in memory.
    /// 
    /// It works by:
    /// 1. Considering which memory locations are free or less used
    /// 2. Prioritizing locations that haven't been written to recently
    /// 3. Calculating weights that favor writing to unused or old locations
    /// 4. This prevents important recent information from being overwritten
    /// 
    /// This is like finding empty pages in a notebook, or pages with old notes you don't need anymore,
    /// to write new information.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Calculates allocation weights based on the usage free vector and precedence vector.
    /// </summary>
    /// <param name="usageFree">The vector tracking which memory locations are free.</param>
    /// <param name="precedenceWeighting">The vector tracking the order of memory writes.</param>
    /// <returns>A vector of weights for memory allocation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates weights for allocating memory based on which locations are free (available for writing)
    /// and the order in which memory has been written to previously. It sorts memory locations by their usage and
    /// allocates more weight to freer locations.
    /// </para>
    /// <para><b>For Beginners:</b> This decides where to write new information in memory.
    /// 
    /// It works by:
    /// 1. Considering which memory locations are free or less used
    /// 2. Prioritizing locations that haven't been written to recently
    /// 3. Calculating weights that favor writing to unused or old locations
    /// 4. This prevents important recent information from being overwritten
    /// 
    /// This is like finding empty pages in a notebook, or pages with old notes you don't need anymore,
    /// to write new information.
    /// </para>
    /// </remarks>
    private Vector<T> Allocate(Vector<T> usageFree, Vector<T> precedenceWeighting)
    {
        // Create a list of indices sorted by usage free values (ascending)
        var sortedIndices = Enumerable.Range(0, _memorySize)
            .OrderBy(i => Convert.ToDouble(usageFree[i]))
            .ToList();

        // Create allocation weights vector
        Vector<T> allocationWeighting = new Vector<T>(_memorySize);

        // Initialize allocation weights to zero
        for (int i = 0; i < _memorySize; i++)
        {
            allocationWeighting[i] = NumOps.Zero;
        }

        // Calculate cumulatively multiplied usage free vector
        Vector<T> phi = new Vector<T>(_memorySize)
        {
            // Initial value for phi
            [sortedIndices[0]] = NumOps.One
        };

        // Compute phi values for remaining sorted indices
        for (int i = 1; i < _memorySize; i++)
        {
            int index = sortedIndices[i];
            int prevIndex = sortedIndices[i - 1];

            phi[index] = NumOps.Multiply(phi[prevIndex], NumOps.Subtract(NumOps.One, usageFree[prevIndex]));
        }

        // Calculate allocation weights based on phi and usage free
        for (int i = 0; i < _memorySize; i++)
        {
            int index = sortedIndices[i];
            allocationWeighting[index] = NumOps.Multiply(usageFree[index], phi[index]);
        }

        // Normalize the allocation weights to ensure they sum to 1 (vectorized)
        var allocationTensor = new Tensor<T>(allocationWeighting.ToArray(), [_memorySize]);
        T sum = Engine.TensorSum(allocationTensor);

        // Avoid division by zero
        if (!MathHelper.AlmostEqual(sum, NumOps.Zero))
        {
            // Vectorized division by sum
            var normalizedTensor = Engine.TensorDivideScalar(allocationTensor, sum);
            var normalizedArray = normalizedTensor.ToArray();
            for (int i = 0; i < _memorySize; i++)
            {
                allocationWeighting[i] = normalizedArray[i];
            }
        }
        else
        {
            // If all weights are zero, use a uniform distribution (vectorized)
            T uniformWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_memorySize));
            Engine.TensorFill(allocationTensor, uniformWeight);
            var uniformArray = allocationTensor.ToArray();
            for (int i = 0; i < _memorySize; i++)
            {
                allocationWeighting[i] = uniformArray[i];
            }
        }

        return allocationWeighting;
    }

    /// <summary>
    /// Combines content and allocation weightings based on an allocation gate.
    /// </summary>
    /// <param name="contentWeighting">The content-based weighting.</param>
    /// <param name="allocationWeighting">The allocation-based weighting.</param>
    /// <param name="allocationGate">The allocation gate value between 0 and 1.</param>
    /// <returns>A combined weighting vector.</returns>
    /// <remarks>
    /// <para>
    /// This method combines content-based addressing weights (which focus on memory locations with similar content)
    /// with allocation-based weights (which focus on free memory locations) based on an allocation gate value.
    /// When the allocation gate is close to 1, allocation weights are favored. When it's close to 0, content weights are favored.
    /// </para>
    /// <para><b>For Beginners:</b> This balances between writing to similar locations versus writing to free locations.
    /// 
    /// It works like:
    /// 1. The content weighting prioritizes writing to locations with similar content
    /// 2. The allocation weighting prioritizes writing to unused locations
    /// 3. The allocation gate determines which approach to favor
    /// 4. This balance lets the network decide whether to update existing information or store new information
    /// 
    /// This is like deciding whether to add notes to an existing page on a topic, or start a fresh page.
    /// </para>
    /// </remarks>
    private Vector<T> CombineWeightings(Vector<T> contentWeighting, Vector<T> allocationWeighting, T allocationGate)
    {
        Vector<T> combinedWeighting = new Vector<T>(_memorySize);

        for (int i = 0; i < _memorySize; i++)
        {
            T contentComponent = NumOps.Multiply(contentWeighting[i], NumOps.Subtract(NumOps.One, allocationGate));
            T allocationComponent = NumOps.Multiply(allocationWeighting[i], allocationGate);
            combinedWeighting[i] = NumOps.Add(contentComponent, allocationComponent);
        }

        return combinedWeighting;
    }

    /// <summary>
    /// Writes information to memory based on the write weighting, write vector, and erase vector.
    /// </summary>
    /// <param name="writeVector">The vector specifying what to write.</param>
    /// <param name="eraseVector">The vector specifying what to erase.</param>
    /// <remarks>
    /// <para>
    /// This method updates the memory matrix by first erasing old information based on the erase vector and write weighting,
    /// then writing new information based on the write vector and write weighting. Each memory location is updated proportionally
    /// to its corresponding write weight.
    /// </para>
    /// <para><b>For Beginners:</b> This actually writes the new information to memory.
    /// 
    /// The writing process:
    /// 1. First, it erases existing information (to varying degrees) at the target locations
    /// 2. Then, it writes the new information to those same locations
    /// 3. The amount of erasing and writing at each location depends on the write weighting
    /// 4. This allows for partial updates to existing information
    /// 
    /// This two-step process (erase then write) is like erasing part of a page before writing new notes,
    /// allowing the DNC to update information without completely overwriting everything.
    /// </para>
    /// </remarks>
    private void WriteToMemory(Vector<T> writeVector, Vector<T> eraseVector)
    {
        // First, erase from memory based on erase vector and write weighting
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                // Calculate erase amount
                T eraseAmount = NumOps.Multiply(_writeWeighting[i], eraseVector[j]);

                // Erase from memory
                _memory[i, j] = NumOps.Multiply(_memory[i, j], NumOps.Subtract(NumOps.One, eraseAmount));
            }
        }

        // Then, write to memory based on write vector and write weighting
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                // Calculate write amount
                T writeAmount = NumOps.Multiply(_writeWeighting[i], writeVector[j]);

                // Add to memory
                _memory[i, j] = NumOps.Add(_memory[i, j], writeAmount);
            }
        }
    }

    /// <summary>
    /// Updates the usage free vector based on the write weighting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the usage free vector to reflect which memory locations have been written to. Locations
    /// with higher write weights are marked as less free (closer to 0), indicating that they now contain important
    /// information that should not be quickly overwritten.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the record of which memory locations are in use.
    /// 
    /// After writing to memory:
    /// 1. The usage free vector is updated to reflect which locations are now being used
    /// 2. Locations that were written to more heavily are marked as less free
    /// 3. This helps the DNC avoid overwriting important information in future writes
    /// 4. Over time, memory usage changes as information becomes less relevant
    /// 
    /// This is like keeping track of which pages in your notebook contain important information
    /// and which ones are available for new notes.
    /// </para>
    /// </remarks>
    private void UpdateUsage()
    {
        for (int i = 0; i < _memorySize; i++)
        {
            // Reduce free usage based on write weighting
            _usageFree[i] = NumOps.Multiply(_usageFree[i], NumOps.Subtract(NumOps.One, _writeWeighting[i]));

            // Optionally include a small decay factor to gradually free up memory over time
            T decayFactor = NumOps.FromDouble(0.99); // Slight decay
            _usageFree[i] = NumOps.Add(_usageFree[i], NumOps.Multiply(NumOps.Subtract(NumOps.One, _usageFree[i]),
                NumOps.Subtract(NumOps.One, decayFactor)));
        }
    }

    /// <summary>
    /// Updates the precedence weighting based on the write weighting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the precedence weighting vector to reflect the order in which memory locations were written to.
    /// Locations that are currently being written to are given the highest precedence (values closer to 1), while
    /// precedence values for previously written locations decay slightly.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the record of which memory locations were written to most recently.
    /// 
    /// After writing to memory:
    /// 1. The precedence weighting is updated to track the recency of writes
    /// 2. Locations just written to are marked as most recent
    /// 3. Previously recent locations have their recency value slightly reduced
    /// 4. This creates a time-ordered record of memory usage
    /// 
    /// This is like keeping track of the order in which you wrote notes in your notebook,
    /// which helps you find related information that was written around the same time.
    /// </para>
    /// </remarks>
    private void UpdatePrecedence()
    {
        // Decay old precedence values
        T decay = NumOps.FromDouble(0.9); // Decay factor
        for (int i = 0; i < _memorySize; i++)
        {
            _precedenceWeighting[i] = NumOps.Multiply(_precedenceWeighting[i], decay);
        }

        // Increase precedence for locations currently being written to
        for (int i = 0; i < _memorySize; i++)
        {
            _precedenceWeighting[i] = NumOps.Add(_precedenceWeighting[i],
                NumOps.Multiply(_writeWeighting[i], NumOps.Subtract(NumOps.One, decay)));
        }
    }

    /// <summary>
    /// Updates the temporal link matrix based on the write weighting and precedence weighting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the temporal link matrix to represent the sequential relationships between memory locations.
    /// If location i was written to just before location j, the temporal link matrix will have a high value at position [i, j].
    /// This allows the DNC to follow chains of information in the order they were written.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the connections between memory locations in time sequence.
    /// 
    /// After writing to memory:
    /// 1. The temporal link matrix is updated to track which locations were written to before/after others
    /// 2. This creates a network of "next location" and "previous location" relationships
    /// 3. The strength of these connections depends on how heavily the locations were written to
    /// 4. This allows the DNC to follow chains of related information in sequence
    /// 
    /// This is like drawing arrows between pages in your notebook to show which page comes next
    /// in a sequence, allowing you to follow a train of thought or a multi-step process.
    /// </para>
    /// </remarks>
    private void UpdateTemporalLinks()
    {
        // Update temporal links based on write weightings and precedence
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memorySize; j++)
            {
                if (i != j) // No self-links
                {
                    T link = NumOps.Add(
                        NumOps.Multiply(_temporalLinkMatrix[i, j],
                            NumOps.Subtract(NumOps.One, _writeWeighting[i])),
                        NumOps.Multiply(_precedenceWeighting[j], _writeWeighting[i]));

                    _temporalLinkMatrix[i, j] = link;
                }
                else
                {
                    _temporalLinkMatrix[i, j] = NumOps.Zero; // No self-links
                }
            }
        }
    }

    /// <summary>
    /// Gets metadata about the Differentiable Neural Computer model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the DNC, including its model type, memory configuration,
    /// and additional configuration information. This metadata is useful for model management
    /// and for generating reports about the model's structure and configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your DNC's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (Differentiable Neural Computer)
    /// - Details about memory size and word size
    /// - Number of read heads and controller size
    /// - Information about the network architecture
    /// - Serialized data that can be used to save and reload the model
    /// 
    /// This information is useful for tracking different model configurations
    /// and for saving/loading models for later use.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DifferentiableNeuralComputer,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MemorySize", _memorySize },
                { "MemoryWordSize", _memoryWordSize },
                { "ControllerSize", _controllerSize },
                { "ReadHeads", _readHeads },
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize },
                { "LayerCount", Layers.Count },
                { "ParameterCount", ParameterCount }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Differentiable Neural Computer-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes DNC-specific configuration and state data to a binary stream. It includes
    /// properties such as memory size, memory word size, controller size, read heads count, and the
    /// current state of the memory matrix, usage vector, and other memory tracking structures.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the special configuration and current state of your DNC.
    /// 
    /// It's like taking a snapshot of the DNC that includes:
    /// - Its structural configuration (memory size, read heads, etc.)
    /// - The current contents of memory
    /// - The current state of all memory tracking systems
    /// - The current state of all memory connections
    /// 
    /// This allows you to save both the network's learned parameters and its current memory state,
    /// so you can resume from exactly the same state later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write DNC-specific properties
        writer.Write(_memorySize);
        writer.Write(_memoryWordSize);
        writer.Write(_controllerSize);
        writer.Write(_readHeads);
        writer.Write(IsTrainingMode);

        // Write memory matrix
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                writer.Write(Convert.ToDouble(_memory[i, j]));
            }
        }

        // Write usage free vector
        for (int i = 0; i < _memorySize; i++)
        {
            writer.Write(Convert.ToDouble(_usageFree[i]));
        }

        // Write write weighting
        for (int i = 0; i < _memorySize; i++)
        {
            writer.Write(Convert.ToDouble(_writeWeighting[i]));
        }

        // Write read weightings
        writer.Write(_readWeightings.Count);
        foreach (var readWeighting in _readWeightings)
        {
            for (int i = 0; i < _memorySize; i++)
            {
                writer.Write(Convert.ToDouble(readWeighting[i]));
            }
        }

        // Write precedence weighting
        for (int i = 0; i < _memorySize; i++)
        {
            writer.Write(Convert.ToDouble(_precedenceWeighting[i]));
        }

        // Write temporal link matrix
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memorySize; j++)
            {
                writer.Write(Convert.ToDouble(_temporalLinkMatrix[i, j]));
            }
        }

        // Write read vectors
        writer.Write(_readVectors.Count);
        foreach (var readVector in _readVectors)
        {
            for (int i = 0; i < _memoryWordSize; i++)
            {
                writer.Write(Convert.ToDouble(readVector[i]));
            }
        }
    }

    /// <summary>
    /// Deserializes Differentiable Neural Computer-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads DNC-specific configuration and state data from a binary stream. It retrieves
    /// properties such as memory size, memory word size, controller size, read heads count, and the
    /// saved state of the memory matrix, usage vector, and other memory tracking structures.
    /// </para>
    /// <para><b>For Beginners:</b> This restores the special configuration and state of your DNC from saved data.
    /// 
    /// It's like restoring a snapshot of the DNC that includes:
    /// - Its structural configuration (memory size, read heads, etc.)
    /// - The saved contents of memory
    /// - The saved state of all memory tracking systems
    /// - The saved state of all memory connections
    /// 
    /// This allows you to resume from exactly the same state that was saved,
    /// with both the network's learned parameters and its memory contents intact.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read DNC-specific properties
        int memorySize = reader.ReadInt32();
        int memoryWordSize = reader.ReadInt32();
        int controllerSize = reader.ReadInt32();
        int readHeads = reader.ReadInt32();

        // Check if configuration matches
        if (memorySize != _memorySize || memoryWordSize != _memoryWordSize ||
            controllerSize != _controllerSize || readHeads != _readHeads)
        {
            Console.WriteLine("Warning: Loaded DNC has different configuration than the current instance.");
        }

        // Read training mode
        IsTrainingMode = reader.ReadBoolean();

        // Read memory matrix
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                _memory[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Read usage free vector
        for (int i = 0; i < _memorySize; i++)
        {
            _usageFree[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read write weighting
        for (int i = 0; i < _memorySize; i++)
        {
            _writeWeighting[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read read weightings
        int readWeightingsCount = reader.ReadInt32();
        _readWeightings.Clear();
        for (int k = 0; k < readWeightingsCount; k++)
        {
            Vector<T> readWeighting = new Vector<T>(_memorySize);
            for (int i = 0; i < _memorySize; i++)
            {
                readWeighting[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _readWeightings.Add(readWeighting);
        }

        // Read precedence weighting
        for (int i = 0; i < _memorySize; i++)
        {
            _precedenceWeighting[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read temporal link matrix
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memorySize; j++)
            {
                _temporalLinkMatrix[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Read read vectors
        int readVectorsCount = reader.ReadInt32();
        _readVectors.Clear();
        for (int k = 0; k < readVectorsCount; k++)
        {
            Vector<T> readVector = new Vector<T>(_memoryWordSize);
            for (int i = 0; i < _memoryWordSize; i++)
            {
                readVector[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _readVectors.Add(readVector);
        }
    }

    /// <summary>
    /// Helper class for storing memory interface signals parsed from controller output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class organizes the various interface signals generated by the controller network for interacting with memory.
    /// It includes vectors and values for writing to memory, reading from memory, and controlling memory access modes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a control panel for operating the memory system.
    /// 
    /// It organizes all the different control signals:
    /// - What information to write to memory (write vector)
    /// - What information to erase from memory (erase vector)
    /// - Where to look for information in memory (read keys)
    /// - How focused to be when searching memory (read strengths)
    /// - How to navigate between related pieces of information (read modes)
    /// 
    /// These signals together allow the controller to precisely control how information
    /// is stored in and retrieved from memory.
    /// </para>
    /// </remarks>
    private class MemoryInterfaceSignals
    {
        /// <summary>
        /// Gets the numeric operations provider for type T.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This property provides access to numeric operations (like addition, multiplication, etc.) that work
        /// with the generic type T. This allows the layer to perform mathematical operations regardless of
        /// whether T is float, double, or another numeric type.
        /// </para>
        /// <para><b>For Beginners:</b> This is a toolkit for math operations that works with different number types.
        /// 
        /// It provides:
        /// - Basic math operations (add, subtract, multiply, etc.)
        /// - Ways to convert between different number formats
        /// - Special math functions needed by neural networks
        /// 
        /// This allows the layer to work with different types of numbers (float, double, etc.)
        /// without needing different code for each type.
        /// </para>
        /// </remarks>
        private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// The vector to write to memory.
        /// </summary>
        public Vector<T> WriteVector { get; set; } = new Vector<T>(0);

        /// <summary>
        /// The vector indicating what to erase from memory.
        /// </summary>
        public Vector<T> EraseVector { get; set; } = new Vector<T>(0);

        /// <summary>
        /// The list of keys for content-based reading from memory.
        /// </summary>
        public List<Vector<T>> ReadKeys { get; set; } = new List<Vector<T>>();

        /// <summary>
        /// The list of strengths for content-based reading.
        /// </summary>
        public List<T> ReadStrengths { get; set; } = new List<T>();

        /// <summary>
        /// The key for content-based writing to memory.
        /// </summary>
        public Vector<T> WriteKey { get; set; } = new Vector<T>(0);

        /// <summary>
        /// The strength for content-based writing.
        /// </summary>
        public T WriteStrength { get; set; }

        /// <summary>
        /// The allocation gate controlling whether to use content-based or allocation-based writing.
        /// </summary>
        public T AllocationGate { get; set; }

        /// <summary>
        /// The write gate controlling the overall intensity of writing.
        /// </summary>
        public T WriteGate { get; set; }

        /// <summary>
        /// The list of mode vectors for each read head, controlling whether to read based on content,
        /// based on backward temporal links, or based on forward temporal links.
        /// </summary>
        public List<Vector<T>> ReadModes { get; set; } = new List<Vector<T>>();

        /// <summary>
        /// Initializes a new instance of the MemoryInterfaceSignals class with default values.
        /// </summary>
        public MemoryInterfaceSignals()
        {
            // Initialize numeric properties after NumOps is available
            WriteStrength = NumOps.Zero;
            AllocationGate = NumOps.Zero;
            WriteGate = NumOps.Zero;
        }

        /// <summary>
        /// Initializes a new instance of the MemoryInterfaceSignals class with specified memory word size.
        /// </summary>
        /// <param name="memoryWordSize">The size of memory words.</param>
        public MemoryInterfaceSignals(int memoryWordSize)
        {
            WriteVector = new Vector<T>(memoryWordSize);
            EraseVector = new Vector<T>(memoryWordSize);
            WriteKey = new Vector<T>(memoryWordSize);

            // Initialize scalar properties
            WriteStrength = NumOps.Zero;
            AllocationGate = NumOps.Zero;
            WriteGate = NumOps.Zero;

            // Initialize the vectors with zeros
            for (int i = 0; i < memoryWordSize; i++)
            {
                WriteVector[i] = NumOps.Zero;
                EraseVector[i] = NumOps.Zero;
                WriteKey[i] = NumOps.Zero;
            }
        }
    }

    /// <summary>
    /// Resets the state of the Differentiable Neural Computer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the DNC's memory state, including the memory matrix, usage tracking vectors,
    /// read/write weightings, temporal link matrix, and read vectors. This is useful when starting to process
    /// a new, unrelated sequence of inputs, or when initializing the network for a new task.
    /// </para>
    /// <para><b>For Beginners:</b> This is like erasing the notepad and starting fresh.
    /// 
    /// The reset process:
    /// 1. Clears the memory matrix
    /// 2. Resets all memory tracking systems
    /// 3. Clears all memory connections
    /// 4. Resets all reading and writing mechanisms
    /// 
    /// This is useful when:
    /// - Starting a completely new task
    /// - Ensuring that information from a previous task doesn't influence the current one
    /// - Testing the DNC on different problems independently
    /// 
    /// Note that this doesn't reset the network's learned parameters, just its current memory state.
    /// </para>
    /// </remarks>
    public void ResetMemoryState()
    {
        // Reset memory matrix
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                _memory[i, j] = NumOps.FromDouble(Random.NextDouble() * 0.1);
            }
        }

        // Reset usage free vector
        for (int i = 0; i < _memorySize; i++)
        {
            _usageFree[i] = NumOps.FromDouble(1.0);
        }

        // Reset write weighting
        for (int i = 0; i < _memorySize; i++)
        {
            _writeWeighting[i] = NumOps.Zero;
        }

        // Reset read weightings
        _readWeightings.Clear();
        for (int i = 0; i < _readHeads; i++)
        {
            var readWeighting = new Vector<T>(_memorySize);
            for (int j = 0; j < _memorySize; j++)
            {
                readWeighting[j] = NumOps.Zero;
            }
            _readWeightings.Add(readWeighting);
        }

        // Reset precedence weighting
        for (int i = 0; i < _memorySize; i++)
        {
            _precedenceWeighting[i] = NumOps.Zero;
        }

        // Reset temporal link matrix
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memorySize; j++)
            {
                _temporalLinkMatrix[i, j] = NumOps.Zero;
            }
        }

        // Reset read vectors
        _readVectors.Clear();
    }

    /// <summary>
    /// Processes a sequence of inputs through the DNC.
    /// </summary>
    /// <param name="inputs">A list of input tensors representing a sequence.</param>
    /// <returns>A list of output tensors corresponding to each input.</returns>
    /// <remarks>
    /// <para>
    /// This method processes a sequence of inputs through the DNC, maintaining the memory state between inputs.
    /// This is particularly useful for tasks that require processing sequences of data, like language modeling,
    /// sequence prediction, or graph traversal.
    /// </para>
    /// <para><b>For Beginners:</b> This processes a series of inputs while maintaining memory between them.
    /// 
    /// It works like:
    /// 1. Starting with a fresh memory state (or continuing from the current state)
    /// 2. Processing each input one by one through the network
    /// 3. Using information stored in memory from previous inputs to help process current ones
    /// 4. Building up knowledge across the sequence in the external memory
    /// 
    /// This is ideal for tasks where each input is related to previous ones, like:
    /// - Processing a paragraph of text word by word
    /// - Following a sequence of instructions step by step
    /// - Analyzing a time series of data points
    /// </para>
    /// </remarks>
    public List<Tensor<T>> ProcessSequence(List<Tensor<T>> inputs, bool resetMemory = true)
    {
        if (resetMemory)
        {
            ResetMemoryState();
        }

        List<Tensor<T>> outputs = new List<Tensor<T>>();

        foreach (var input in inputs)
        {
            var output = Predict(input);
            outputs.Add(output);
        }

        return outputs;
    }

    /// <summary>
    /// Creates a new instance of the differentiable neural computer model.
    /// </summary>
    /// <returns>A new instance of the differentiable neural computer model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the differentiable neural computer model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization processes to create a fresh instance that can be populated
    /// with the serialized data. The new instance will have the same architecture, memory size, memory word size,
    /// controller size, read heads count, and activation function type as the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the network structure without copying the learned data.
    /// 
    /// Think of it like creating a blueprint copy of the DNC:
    /// - It copies the same neural network architecture
    /// - It sets up the same memory size (same notepad dimensions)
    /// - It configures the same number of read heads (how many pages to look at at once)
    /// - It uses the same controller size (brain power)
    /// - It keeps the same activation function (how neurons respond to input)
    /// - But it doesn't copy any of the actual memories or learned behaviors
    /// 
    /// This is primarily used when saving or loading models, creating an empty framework
    /// that the saved parameters and memory state can be loaded into later.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Determine which constructor to use based on which activation function is set
        if (_activationFunction != null)
        {
            return new DifferentiableNeuralComputer<T>(
                Architecture,
                _memorySize,
                _memoryWordSize,
                _controllerSize,
                _readHeads,
                _lossFunction,
                _activationFunction
            );
        }
        else
        {
            return new DifferentiableNeuralComputer<T>(
                Architecture,
                _memorySize,
                _memoryWordSize,
                _controllerSize,
                _readHeads,
                _lossFunction,
                _vectorActivationFunction
            );
        }
    }
}
