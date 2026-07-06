using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

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
/// <example>
/// <code>
/// var options = new DifferentiableNeuralComputerOptions { InputSize = 64, MemorySize = 128, MemoryWordSize = 32 };
/// var model = new DifferentiableNeuralComputer&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 10, 64 });
/// var output = model.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Hybrid Computing Using a Neural Network with Dynamic External Memory", "https://www.nature.com/articles/nature20101", Year = 2016, Authors = "Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwinska, Sergio Gomez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, Demis Hassabis")]
public class DifferentiableNeuralComputer<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    private readonly DifferentiableNeuralComputerOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// When true, <see cref="ProcessInput"/> does NOT reset the memory state at the start of the forward
    /// pass, so the carried memory/usage/link state persists across calls. Set only by
    /// <see cref="ProcessSequence"/> (after an initial reset) so a sequence evolves real DNC state across
    /// its timesteps. It stays false for single Predict / training forwards, which reset per call for
    /// deterministic re-evaluation replay (see the reset comment in <see cref="ProcessInput"/>).
    /// </summary>
    private bool _suppressMemoryReset;

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
    // _outputWeights is a back-compat field for legacy serialized DNC checkpoints.
    // The active output projection is the final DenseLayer in the Layers chain
    // (see CreateDefaultDNCLayers + CombineControllerOutputWithReadVectors); this
    // matrix is read/written by the legacy Serialize/Deserialize paths only.
    // _lastCombinedVector was a backward-pass cache for the manual matmul that
    // the new Layers-chain projection no longer needs.
    private Matrix<T> _outputWeights;
#pragma warning disable CS0169
    private Vector<T>? _lastCombinedVector;
#pragma warning restore CS0169

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
    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public DifferentiableNeuralComputer()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 1),
            memorySize: 64, memoryWordSize: 32, controllerSize: 128, readHeads: 4,
            activationFunction: (IActivationFunction<T>?)null)
    {
    }

    public DifferentiableNeuralComputer(
        NeuralNetworkArchitecture<T> architecture,
        int memorySize,
        int memoryWordSize,
        int controllerSize,
        int readHeads,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? activationFunction = null,
        DifferentiableNeuralComputerOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _options = options ?? new DifferentiableNeuralComputerOptions();
        Options = _options;
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
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? vectorActivationFunction = null,
        DifferentiableNeuralComputerOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _options = options ?? new DifferentiableNeuralComputerOptions();
        Options = _options;
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
                _memory[i, j] = NumOps.Zero;
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
               1 + // Write strength
               readHeads + // Free gates (one per read head — controls memory deallocation, Graves §2.1.1)
               2 + // Allocation gate, write gate
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
            int layerParameterCount = checked((int)layer.ParameterCount);
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Reset memory and layer state for deterministic inference
        ResetMemoryState();
        foreach (var layer in Layers)
            layer.ResetState();

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
    /// <summary>
    /// Persistent Adam optimizer for stable convergence across Train() calls.
    /// </summary>
    #pragma warning disable CS0169
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainOptimizer;
#pragma warning restore CS0169

    /// <summary>
    /// The DNC's forward is dynamic and stateful — an external memory matrix
    /// read/written through content addressing, allocation weighting, and
    /// temporal link tracking across an internal recurrence (<see cref="ProcessInput"/>).
    /// That is not a static op graph, so it opts out of the compile-once/replay-many
    /// fused compiled training path and trains on the eager autograd tape (which
    /// re-runs the true dynamic forward every step). See the same override on
    /// <see cref="NeuralTuringMachine{T}"/> for the full rationale (#1643).
    /// </summary>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <inheritdoc/>
    /// <remarks>
    /// DNC overrides ForwardForTraining because its forward pass includes memory
    /// read/write operations that aren't captured in the Layers list. The base
    /// ForwardForTraining only iterates Layers, producing raw controller output
    /// instead of the final DNC output after memory combination.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        return ProcessInput(input, true);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        foreach (var layer in Layers)
            layer.SetTrainingMode(true);

        // Reset memory for clean training pass
        ResetMemoryState();

        TrainWithTape(input, expectedOutput, _optimizer);

        SetTrainingMode(false);
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
    /// <summary>
    /// Materialise every element of <paramref name="t"/>. DNC's memory-
    /// op pipeline produces tensor chains whose values are materialised
    /// only on first read — under <see cref="NeuralNetworkBase{T}.TrainWithTape"/>'s
    /// re-evaluation loop the lazy nodes re-read shared state (_memory,
    /// _readVectors) mutated between the original forward and the replay
    /// forward, producing inconsistent gradients. Pinning the values
    /// here snapshots them so downstream code sees a stable view. Issue
    /// #1332 cluster 2 — required for ForwardPass_ShouldBeFinite_AfterTraining,
    /// GradientFlow_*, OptimizerStep_ParamL2_*, MoreData_ShouldNotDegrade,
    /// Clone_*, DifferentInputs_AfterTraining_*, LossStrictlyDecreasesOnMemorizationTask.
    /// </summary>
    private void PinElements(Tensor<T> t)
    {
        for (int i = 0; i < t.Length; i++)
            if (NumOps.IsNaN(t.GetFlat(i)))
                throw new InvalidOperationException(
                    $"Lazy tensor produced NaN at index {i} during pin (#1332 cluster 2).");
    }

    private void PinElements(Matrix<T> m)
    {
        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                if (NumOps.IsNaN(m[i, j]))
                    throw new InvalidOperationException(
                        $"Lazy matrix produced NaN at [{i},{j}] during pin (#1332 cluster 2).");
    }

    private Tensor<T> ProcessInput(Tensor<T> input, bool isTraining)
    {
        // Set training mode for all layers
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(isTraining);
        }

        // Reset the per-call memory state at the start of EVERY forward pass.
        // Previously this was reset in Predict and Train, but TrainWithTape
        // calls ForwardForTraining multiple times within a single Train —
        // the optimizer's re-evaluation step (see NeuralNetworkBase.TapeStepContext
        // ComputeForward) replays the forward pass to recompute the loss
        // after a tentative parameter update. Without a per-call reset,
        // _memory keeps mutating across those replays, the read vectors
        // pick up the corrupted state, and after a handful of training
        // iterations the controller input slot for read vectors is NaN —
        // exactly the failure surface of #1332 cluster 2
        // (ForwardPass_ShouldBeFinite_AfterTraining, GradientFlow_*,
        // OptimizerStep_ParamL2_DoesNotExplode all share this cascade).
        // ProcessSequence suppresses this so a sequence carries real memory state across its timesteps;
        // single Predict / training forwards keep the per-call reset for replay determinism.
        if (!_suppressMemoryReset)
        {
            ResetMemoryState();
        }

        // Previous read vectors are concatenated with the input to provide context
        Tensor<T> controllerInput = PrepareControllerInput(input);
        PinElements(controllerInput);

        // Process the input through the controller network
        Tensor<T> controllerOutput = ProcessThroughController(controllerInput);
        PinElements(controllerOutput);

        // #1678: fully-differentiable, tape-tracked memory interaction. The read vectors are produced by
        // Engine tensor ops (content addressing / write / read) instead of the legacy Matrix/Vector +
        // NumOps scalar math, so the gradient flows from the loss back through the memory operations to the
        // controller's interface signals (the old path ran off the autodiff tape and severed that gradient).
        Tensor<T> readVectorsTensor = ComputeReadVectorsDifferentiable(controllerOutput);

        // Combine controller output with read vectors to produce final output.
        Tensor<T> finalOutput = CombineControllerOutputWithReadVectors(controllerOutput, readVectorsTensor);
        PinElements(finalOutput);

        return finalOutput;
    }

    /// <summary>
    /// Full Differentiable Neural Computer memory interaction (Graves et al. 2016, "Hybrid computing using a
    /// neural network with dynamic external memory") implemented with tape-tracked <see cref="Tensor{T}"/>
    /// <see cref="Engine"/> ops, so the gradient of the loss flows back through the memory operations to the
    /// controller's interface signals — the legacy <c>Matrix&lt;T&gt;</c>/<c>Vector&lt;T&gt;</c>/<c>NumOps</c>
    /// path ran off the autodiff tape and severed that gradient (#1678).
    /// </summary>
    /// <param name="controllerOutput">The controller output [1, controllerSize + interfaceSize].</param>
    /// <returns>The concatenated read vectors [1, readHeads * memoryWordSize], tape-connected.</returns>
    /// <remarks>
    /// <para>
    /// This performs every stage of the DNC write/read cycle: content-based write addressing, dynamic memory
    /// <b>allocation</b> (usage → free-list → allocation weighting), gated write (allocation + content), the
    /// erase/add memory update, precedence + <b>temporal-link</b> bookkeeping, and content + forward/backward
    /// temporal read addressing across all read heads.
    /// </para>
    /// <para>
    /// <b>Tape topology.</b> The differentiable learning signal flows through content addressing (write/read
    /// keys and strengths), the allocation/write gates, the erase/write vectors, and the read-mode mixture —
    /// every continuous interface signal. The discrete/historical bookkeeping (usage, the allocation sort,
    /// precedence, temporal links) is maintained as <i>carried detached state</i> per Graves §2.1: the
    /// allocation ordering is a non-differentiable sort, and under the per-forward memory reset used for
    /// training-replay determinism (see <see cref="ProcessInput"/>) the usage/temporal terms are degenerate
    /// (no prior writes/reads to retain), so detaching them changes no training gradient. All state (memory,
    /// usage, write/read weightings, precedence, links) is mirrored back each step so sequence processing
    /// (<see cref="ProcessSequence"/>), serialization, and auxiliary-loss diagnostics observe the real
    /// evolving DNC state rather than a zero-memory shortcut.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeReadVectorsDifferentiable(Tensor<T> controllerOutput)
    {
        int w = _memoryWordSize;
        int n = _memorySize;
        int r = _readHeads;
        int interfaceSize = CalculateDNCInterfaceSize(w, r);
        int total = controllerOutput.Shape[1];

        // #1681 (review): fail fast with an explicit expected-vs-actual width before slicing. Custom
        // controller layers (the generic ValidateCustomLayers only checks layer compatibility) could emit a
        // different width, which would otherwise make TensorNarrow throw cryptically or mis-split the direct
        // and interface regions.
        int expected = _controllerSize + interfaceSize;
        if (total != expected)
        {
            throw new ArgumentException(
                $"DNC controller output width {total} does not match the expected {expected} " +
                $"(controllerSize {_controllerSize} + interfaceSize {interfaceSize}). The controller's penultimate " +
                $"layer must emit controllerSize + CalculateDNCInterfaceSize(memoryWordSize, readHeads) columns.");
        }
        int directSize = total - interfaceSize; // == _controllerSize

        // Interface signals occupy the trailing interfaceSize columns; the leading directSize columns are the
        // controller's direct output consumed by CombineControllerOutputWithReadVectors. TensorNarrow keeps
        // the slice on the tape so gradients reach controllerOutput.
        Tensor<T> iface = Engine.TensorNarrow(controllerOutput, dim: 1, start: directSize, length: interfaceSize);
        Tensor<T> Field(int off, int len) => Engine.TensorNarrow(iface, dim: 1, start: off, length: len);
        Tensor<T> OnePlus(Tensor<T> x) => Engine.TensorAddScalar(Engine.Softplus(x), NumOps.One); // β ≥ 1 (DNC §2.1.4)
        Tensor<T> OneMinus(Tensor<T> x) => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(-1.0)), NumOps.One);

        // ---- parse the interface vector (layout mirrors CalculateDNCInterfaceSize) ----
        int o = 0;
        Tensor<T> writeVec = Field(o, w); o += w;                                     // v_t   [1,W]
        Tensor<T> erase = Engine.Sigmoid(Field(o, w)); o += w;                        // e_t   [1,W] in (0,1)
        Tensor<T> writeKey = Field(o, w); o += w;                                     // k^w_t [1,W]
        Tensor<T> writeStrength = OnePlus(Field(o, 1)); o += 1;                       // β^w_t [1,1]
        var freeGateVals = new T[r];                                                  // f^i_t (R), detached for usage bookkeeping
        for (int i = 0; i < r; i++) { freeGateVals[i] = Engine.Sigmoid(Field(o, 1)).ToArray()[0]; o += 1; }
        Tensor<T> allocGate = Engine.Sigmoid(Field(o, 1)); o += 1;                    // g^a_t [1,1]
        Tensor<T> writeGate = Engine.Sigmoid(Field(o, 1)); o += 1;                    // g^w_t [1,1]
        int readOff = o;
        int modesBase = readOff + r * (w + 1);

        // Carried memory state M_{t-1} as a tape leaf (detached values from the previous step / reset).
        Tensor<T> memory = MemoryToLeaf();                                            // [N,W]

        // ---- write addressing: usage → allocation (detached bookkeeping, Graves §2.1.1–2.1.2) ----
        // Retention ψ_t = ∏_i (1 − f^i_t w^{r,i}_{t-1});  u_t = (u_{t-1} + w^w_{t-1} − u_{t-1}∘w^w_{t-1}) ∘ ψ_t.
        var usage = new T[n];
        for (int k = 0; k < n; k++)
        {
            T retention = NumOps.One;
            for (int i = 0; i < r; i++)
            {
                T wr = i < _readWeightings.Count ? _readWeightings[i][k] : NumOps.Zero;
                retention = NumOps.Multiply(retention, NumOps.Subtract(NumOps.One, NumOps.Multiply(freeGateVals[i], wr)));
            }
            T uPrev = NumOps.Subtract(NumOps.One, _usageFree[k]);                     // u_{t-1} = 1 − free
            T ww = _writeWeighting[k];                                                // w^w_{t-1}
            T u = NumOps.Add(uPrev, NumOps.Subtract(ww, NumOps.Multiply(uPrev, ww)));
            usage[k] = NumOps.Multiply(u, retention);
        }
        Tensor<T> allocation = ColumnToLeaf(ComputeAllocation(usage));                // a_t [N,1] detached

        // Content write weighting c^w_t (tape), then w^w_t = g^w_t (g^a_t a_t + (1−g^a_t) c^w_t)  [N,1] tape.
        Tensor<T> contentWrite = ContentAddressingTensor(memory, writeKey, writeStrength);
        Tensor<T> writeWeight = Engine.TensorBroadcastMultiply(
            Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(allocation, allocGate),
                Engine.TensorBroadcastMultiply(contentWrite, OneMinus(allocGate))),
            writeGate);

        // Memory update M_t = M_{t-1} ∘ (1 − w^w_t ⊗ e_t) + w^w_t ⊗ v_t  [N,W] tape.
        Tensor<T> wErase = Engine.TensorMatMul(writeWeight, erase);                   // [N,1]·[1,W] = [N,W]
        Tensor<T> keep = OneMinus(wErase);
        Tensor<T> written = Engine.TensorMatMul(writeWeight, writeVec);               // [N,W]
        Tensor<T> memoryAfter = Engine.TensorAdd(Engine.TensorMultiply(memory, keep), written);

        // ---- precedence + temporal links (detached bookkeeping, Graves §2.2) ----
        var wwNew = writeWeight.ToArray();                                            // w^w_t values [N]
        var precPrev = _precedenceWeighting;
        T wwSum = NumOps.Zero;
        for (int k = 0; k < n; k++) wwSum = NumOps.Add(wwSum, wwNew[k]);
        T precScale = NumOps.Subtract(NumOps.One, wwSum);
        var precNew = new Vector<T>(n);
        for (int k = 0; k < n; k++) precNew[k] = NumOps.Add(NumOps.Multiply(precScale, precPrev[k]), wwNew[k]);
        var linkNew = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) { linkNew[i, j] = NumOps.Zero; continue; }
                T scale = NumOps.Subtract(NumOps.One, NumOps.Add(wwNew[i], wwNew[j]));
                linkNew[i, j] = NumOps.Add(
                    NumOps.Multiply(scale, _temporalLinkMatrix[i, j]),
                    NumOps.Multiply(wwNew[i], precPrev[j]));
            }
        }

        // ---- read addressing: content + forward/backward temporal modes (Graves §2.2.1) ----
        var readVecs = new List<Tensor<T>>(r);
        var readWeightVals = new List<Vector<T>>(r);
        for (int i = 0; i < r; i++)
        {
            Tensor<T> readKey = Field(readOff + i * (w + 1), w);                      // k^{r,i}_t [1,W]
            Tensor<T> readStrength = OnePlus(Field(readOff + i * (w + 1) + w, 1));    // β^{r,i}_t [1,1]
            Tensor<T> modes = Engine.Softmax(Field(modesBase + i * 3, 3), axis: 1);   // π^i_t [1,3] (backward, content, forward)
            Tensor<T> piBackward = Engine.TensorNarrow(modes, dim: 1, start: 0, length: 1);
            Tensor<T> piContent = Engine.TensorNarrow(modes, dim: 1, start: 1, length: 1);
            Tensor<T> piForward = Engine.TensorNarrow(modes, dim: 1, start: 2, length: 1);

            Tensor<T> contentRead = ContentAddressingTensor(memoryAfter, readKey, readStrength); // c^{r,i}_t [N,1] tape
            // Forward f = L_t w^{r,i}_{t-1}, backward b = L_tᵀ w^{r,i}_{t-1} (detached; zero under reset). [N,1].
            Vector<T> wrPrev = i < _readWeightings.Count ? _readWeightings[i] : new Vector<T>(n);
            Tensor<T> forward = ColumnToLeaf(LinkMatVec(linkNew, wrPrev, transpose: false));
            Tensor<T> backward = ColumnToLeaf(LinkMatVec(linkNew, wrPrev, transpose: true));

            // w^{r,i}_t = π_b b + π_c c + π_f f  [N,1] tape (gradient flows via read modes π and content c).
            Tensor<T> readWeight = Engine.TensorAdd(
                Engine.TensorAdd(
                    Engine.TensorBroadcastMultiply(backward, piBackward),
                    Engine.TensorBroadcastMultiply(contentRead, piContent)),
                Engine.TensorBroadcastMultiply(forward, piForward));
            // read vector r^i_t = (w^{r,i}_t)ᵀ M_t  → [1,N]·[N,W] = [1,W].
            Tensor<T> readVec = Engine.TensorMatMul(Engine.Reshape(readWeight, [1, n]), memoryAfter);
            readVecs.Add(readVec);

            var wrVals = readWeight.ToArray();
            var wrVec = new Vector<T>(n);
            for (int k = 0; k < n; k++) wrVec[k] = wrVals[k];
            readWeightVals.Add(wrVec);
        }

        // ---- mirror the evolving state back into the carried detached fields ----
        MirrorMemoryState(memoryAfter, usage, wwNew, precNew, linkNew, readWeightVals);
        UpdateReadVectorState(readVecs);

        return r == 1 ? readVecs[0] : Engine.TensorConcatenate(readVecs.ToArray(), axis: 1); // [1, R*W]
    }

    /// <summary>
    /// Computes the dynamic memory allocation weighting from the usage vector (Graves et al. 2016 §2.1.2):
    /// sort locations by ascending usage (most-free first), then a[φ_j] = (1 − u[φ_j]) ∏_{k&lt;j} u[φ_k].
    /// Returns an [N] array. The sort is non-differentiable, so this is carried detached state.
    /// </summary>
    private T[] ComputeAllocation(T[] usage)
    {
        int n = usage.Length;
        var order = Enumerable.Range(0, n).OrderBy(i => Convert.ToDouble(usage[i])).ToArray();
        var allocation = new T[n];
        T product = NumOps.One;
        for (int j = 0; j < n; j++)
        {
            int idx = order[j];
            allocation[idx] = NumOps.Multiply(NumOps.Subtract(NumOps.One, usage[idx]), product);
            product = NumOps.Multiply(product, usage[idx]);
        }
        return allocation;
    }

    /// <summary>Builds a detached [N, W] tape leaf from the carried memory matrix <c>_memory</c>.</summary>
    private Tensor<T> MemoryToLeaf()
    {
        int n = _memorySize, w = _memoryWordSize;
        var data = new T[n * w];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < w; j++)
                data[i * w + j] = _memory[i, j];
        return new Tensor<T>(data, [n, w]);
    }

    /// <summary>Builds a detached [N, 1] column tape leaf from an [N] array.</summary>
    private Tensor<T> ColumnToLeaf(T[] column) => new Tensor<T>(column, [column.Length, 1]);

    /// <summary>
    /// Computes L·v (forward) or Lᵀ·v (backward) for the temporal link matrix, returning an [N] array.
    /// </summary>
    private T[] LinkMatVec(Matrix<T> link, Vector<T> v, bool transpose)
    {
        int n = link.Rows;
        var result = new T[n];
        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                T lij = transpose ? link[j, i] : link[i, j];
                sum = NumOps.Add(sum, NumOps.Multiply(lij, v[j]));
            }
            result[i] = sum;
        }
        return result;
    }

    /// <summary>
    /// Mirrors the step's evolving DNC state (memory, usage, write/read weightings, precedence, temporal
    /// links) back into the carried detached fields, so sequence processing, serialization, and diagnostics
    /// observe the real state. These are bookkeeping values and play no role on the autodiff tape.
    /// </summary>
    private void MirrorMemoryState(Tensor<T> memoryAfter, T[] usage, T[] writeWeighting, Vector<T> precedence, Matrix<T> link, List<Vector<T>> readWeightings)
    {
        int n = _memorySize, w = _memoryWordSize;
        var md = memoryAfter.ToArray();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < w; j++)
                _memory[i, j] = md[i * w + j];
        for (int k = 0; k < n; k++)
        {
            _usageFree[k] = NumOps.Subtract(NumOps.One, usage[k]); // store free = 1 − usage
            _writeWeighting[k] = writeWeighting[k];
        }
        _precedenceWeighting = precedence;
        _temporalLinkMatrix = link;
        _readWeightings = readWeightings;
    }

    /// <summary>
    /// Tape-tracked cosine-similarity content addressing: softmax over locations of
    /// strength · cos(memoryRow, key). Returns a [N, 1] weighting. All ops run on the Engine so the
    /// gradient flows to <paramref name="key"/> and <paramref name="strength"/>.
    /// </summary>
    private Tensor<T> ContentAddressingTensor(Tensor<T> memory, Tensor<T> key, Tensor<T> strength)
    {
        int n = memory.Shape[0];
        int w = memory.Shape[1];
        T eps = NumOps.FromDouble(1e-8);

        Tensor<T> dot = Engine.TensorMatMul(memory, Engine.Reshape(key, [w, 1]));                       // [N,1]
        Tensor<T> mNorm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(memory, memory), new[] { 1 }, keepDims: true)); // [N,1]
        Tensor<T> kNorm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(key, key), new[] { 1 }, keepDims: true));       // [1,1]
        Tensor<T> denom = Engine.TensorAddScalar(Engine.TensorBroadcastMultiply(mNorm, kNorm), eps);    // [N,1]
        Tensor<T> cos = Engine.TensorMultiply(dot, Engine.TensorReciprocal(denom));                     // [N,1]
        Tensor<T> scaled = Engine.TensorBroadcastMultiply(cos, strength);                               // [N,1] (β broadcast)
        return Engine.Softmax(scaled, axis: 0);                                                         // [N,1]
    }

    /// <summary>
    /// Mirrors the tape-tracked read vectors back into the detached <c>_readVectors</c> / <c>_memory</c>
    /// bookkeeping used by the next step's <see cref="PrepareControllerInput"/> (no gradient role).
    /// </summary>
    private void UpdateReadVectorState(List<Tensor<T>> readVecs)
    {
        _readVectors = new List<Vector<T>>(readVecs.Count);
        foreach (var rv in readVecs)
        {
            var span = rv.ToArray();
            var v = new Vector<T>(_memoryWordSize);
            for (int j = 0; j < _memoryWordSize && j < span.Length; j++) v[j] = span[j];
            _readVectors.Add(v);
        }
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

        // Normalize the raw input to rank-2 [batch=1, features] and keep the
        // tape connection. The previous implementation flattened to a fresh
        // Vector<T> via ToVector() and rebuilt a tensor, which detaches the
        // input from the gradient tape — backward then can't propagate
        // dL/d(controller_input) to the leaf input, breaking the gradient
        // pathway from loss → memory ops → controller_input → leaf.
        int featuresFromInput = input.Length;
        Tensor<T> inputFlat = Engine.Reshape(input, [1, featuresFromInput]);

        // Concatenate the read vectors as a single zero-filled (or
        // previous-step) tensor. Read vectors are produced by the memory
        // ops which currently run off the tape, so this addition is a
        // constant for backward purposes — but the concat itself stays on
        // the tape so dL/d(inputFlat) still flows back through the input
        // axis. Issue #1332 cluster 2 / #2 in the work plan.
        int readVecLen = _readHeads * _memoryWordSize;
        Tensor<T> readVecTensor = new Tensor<T>([1, readVecLen]);
        int offset = 0;
        for (int i = 0; i < _readHeads; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                readVecTensor[0, offset++] = _readVectors[i][j];
            }
        }

        return Engine.TensorConcatenate(new[] { inputFlat, readVecTensor }, axis: 1);
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
        // Walk every layer EXCEPT the final output-projection layer. The output
        // projection (Graves et al. 2016 §2 eq. 8: W_y[v_t; r_t^1; ...; r_t^R]) is
        // applied later in CombineControllerOutputWithReadVectors, after the read
        // vectors have been computed from memory. CreateDefaultDNCLayers always
        // appends that projection as Layers[Last].
        Tensor<T> currentOutput = controllerInput;
        int controllerLayerCount = Math.Max(1, Layers.Count - 1);

        for (int i = 0; i < controllerLayerCount; i++)
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
    private Tensor<T> CombineControllerOutputWithReadVectors(Tensor<T> controllerOutput, Tensor<T> readVectorsTensor)
    {
        // Graves et al. 2016 §2 eq. 8: y_t = W_y [v_t; r_t^1; ...; r_t^R]
        // where v_t is the controller's direct output (controllerOutput sliced to
        // exclude the interface signals) and r_t^i are the read vectors. The
        // projection W_y is implemented as Layers[Last] (a DenseLayer<T> appended
        // by CreateDefaultDNCLayers) so its parameters live in the standard Layers
        // chain — that's what makes the gradient tape capture this op and flow
        // gradients back to the controller layers.
        int interfaceSize = CalculateDNCInterfaceSize(_memoryWordSize, _readHeads);
        int controllerDirectOutputSize = controllerOutput.Shape[1] - interfaceSize;

        // Tape-aware slice of the controller's direct contribution. Engine.TensorNarrow
        // records the slice op, so the gradient of the loss w.r.t. controllerDirect
        // flows back to the full controllerOutput, then back through the controller
        // layers via the standard Layers-chain backward pass.
        Tensor<T> controllerDirect = Engine.TensorNarrow(controllerOutput, dim: 1, start: 0, length: controllerDirectOutputSize);

        // readVectorsTensor is tape-connected (produced by ComputeReadVectorsDifferentiable via Engine
        // tensor ops), so the output projection's gradient now flows through the memory operations back to
        // the controller's interface signals — closing the previously-documented off-tape gap (#1678).

        // Concatenate controller-direct with read-vectors along the feature axis so the output projection
        // sees [v_t; r_t^1; ...; r_t^R].
        Tensor<T> combined = Engine.TensorConcatenate(new[] { controllerDirect, readVectorsTensor }, axis: 1);

        // Apply the output projection (Graves 2016 §2 eq. 8 W_y). This is the final
        // layer in the Layers chain, registered for parameter updates and
        // gradient capture by NeuralNetworkBase.
        Tensor<T> finalOutput = Layers[Layers.Count - 1].Forward(combined);

        // Apply appropriate activation function based on task type
        NeuralNetworkHelper<T>.ApplyOutputActivation(finalOutput, Architecture);

        return finalOutput;
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
            ModelData = SerializeForMetadata()
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

        // Write output weights
        writer.Write(_outputWeights.Rows);
        writer.Write(_outputWeights.Columns);
        for (int i = 0; i < _outputWeights.Rows; i++)
            for (int j = 0; j < _outputWeights.Columns; j++)
                writer.Write(Convert.ToDouble(_outputWeights[i, j]));

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

        // Read output weights
        int owRows = reader.ReadInt32();
        int owCols = reader.ReadInt32();
        _outputWeights = new Matrix<T>(owRows, owCols);
        for (int i = 0; i < owRows; i++)
            for (int j = 0; j < owCols; j++)
                _outputWeights[i, j] = NumOps.FromDouble(reader.ReadDouble());

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
                _memory[i, j] = NumOps.Zero;
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

        // Carry the evolving memory/usage/link state across the sequence's timesteps: suppress the per-call
        // reset inside ProcessInput so each step reads the state mirrored by the previous step. This is what
        // makes allocation (usage) and the temporal links non-degenerate over a sequence. Restored afterward.
        bool previousSuppress = _suppressMemoryReset;
        _suppressMemoryReset = true;
        try
        {
            foreach (var input in inputs)
            {
                var output = Predict(input);
                outputs.Add(output);
            }
        }
        finally
        {
            _suppressMemoryReset = previousSuppress;
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
                lossFunction: _lossFunction,
                activationFunction: _activationFunction
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
                lossFunction: _lossFunction,
                vectorActivationFunction: _vectorActivationFunction
            );
        }
    }
}
