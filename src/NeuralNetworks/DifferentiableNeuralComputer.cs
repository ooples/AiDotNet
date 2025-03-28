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
public class DifferentiableNeuralComputer<T> : NeuralNetworkBase<T>
{
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
    /// Initializes a new instance of the <see cref="DifferentiableNeuralComputer{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="memorySize">The number of memory locations in the memory matrix.</param>
    /// <param name="memoryWordSize">The size of each memory word or location.</param>
    /// <param name="controllerSize">The size of the controller network's output.</param>
    /// <param name="readHeads">The number of read heads that can access the memory simultaneously.</param>
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
    public DifferentiableNeuralComputer(NeuralNetworkArchitecture<T> architecture, int memorySize, int memoryWordSize, int controllerSize, int readHeads) 
        : base(architecture)
    {
        _memorySize = memorySize;
        _memoryWordSize = memoryWordSize;
        _controllerSize = controllerSize;
        _readHeads = readHeads;
        _memory = new Matrix<T>(_memorySize, _memoryWordSize);
        _usageFree = new Vector<T>(_memorySize);
        _writeWeighting = new Vector<T>(_memorySize);
        _readWeightings = [];
        _readVectors = [];

        for (int i = 0; i < _readHeads; i++)
        {
            _readWeightings.Add(new Vector<T>(_memorySize));
        }

        _precedenceWeighting = new Vector<T>(_memorySize);
        _temporalLinkMatrix = new Matrix<T>(_memorySize, _memorySize);

        InitializeMemory();
        InitializeLayers();
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
               readHeads * memoryWordSize + // Read vectors
               3 + // Write gate, allocation gate, write mode
               3 * readHeads; // Read modes
    }

    /// <summary>
    /// Makes a prediction using the current state of the Differentiable Neural Computer.
    /// </summary>
    /// <param name="input">The input vector to make a prediction for.</param>
    /// <returns>The predicted output vector after processing through the DNC.</returns>
    /// <remarks>
    /// <para>
    /// This method processes an input through the DNC to produce a prediction. The input first passes through the
    /// controller network, which generates an interface vector. This interface vector is used to interact with the
    /// memory system, writing and reading information. The controller's output is then combined with the information
    /// read from memory to produce the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes input data through both the neural network and the memory system.
    /// 
    /// The prediction process works like this:
    /// - The input data enters the controller (neural network "brain")
    /// - The controller processes the input and decides how to interact with memory
    /// - The system writes relevant information to memory and reads from memory as needed
    /// - The controller's output is combined with what was read from memory
    /// - This combined information passes through output layers to produce the final prediction
    /// 
    /// This integration of neural processing with memory operations is what gives the DNC
    /// its ability to handle complex, structured problems that require remembering and
    /// reasoning about information.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var controllerState = input;
        var readVectors = new List<Vector<T>>();

        for (int i = 0; i < _readHeads; i++)
        {
            readVectors.Add(new Vector<T>(_memoryWordSize));
        }

        // Controller
        for (int i = 0; i < 2; i++)
        {
            controllerState = Layers[i].Forward(Tensor<T>.FromVector(controllerState)).ToVector();
        }

        // Memory interface
        var interfaceVector = Layers[2].Forward(Tensor<T>.FromVector(controllerState)).ToVector();
        ProcessMemoryInterface(interfaceVector, readVectors);

        // Concatenate controller output with read vectors
        var concatenated = Vector<T>.Concatenate(controllerState, Vector<T>.Concatenate(readVectors));

        // Output layers
        var output = Layers[3].Forward(Tensor<T>.FromVector(concatenated)).ToVector();
        output = Layers[4].Forward(Tensor<T>.FromVector(output)).ToVector();

        return output;
    }

    /// <summary>
    /// Processes the interface vector to interact with the memory system.
    /// </summary>
    /// <param name="interfaceVector">The interface vector produced by the controller.</param>
    /// <param name="readVectors">The list of vectors to store read results.</param>
    /// <remarks>
    /// <para>
    /// This method extracts control signals from the interface vector and uses them to perform memory operations.
    /// It updates the usage vector, calculates write weightings, writes to memory, updates temporal linkage information,
    /// and reads from memory. The results of the read operations are stored in the provided read vectors list.
    /// </para>
    /// <para><b>For Beginners:</b> This interprets the brain's commands to operate the memory system.
    /// 
    /// The memory interface process:
    /// - Extracts different parts of the interface vector to determine what to do with memory
    /// - Decides where to write in memory and what to write
    /// - Updates tracking information about which memory locations are used
    /// - Maintains the temporal links between memory locations (what was written before/after what)
    /// - Determines where to read from and what information to extract
    /// 
    /// This is like translating the brain's intentions into specific actions of writing in
    /// certain places in the notepad and reading from other places.
    /// </para>
    /// </remarks>
    private void ProcessMemoryInterface(Vector<T> interfaceVector, List<Vector<T>> readVectors)
    {
        int offset = 0;

        // Read vectors
        for (int i = 0; i < _readHeads; i++)
        {
            readVectors[i] = interfaceVector.SubVector(offset, _memoryWordSize);
            offset += _memoryWordSize;
        }

        // Write vector
        var writeVector = interfaceVector.SubVector(offset, _memoryWordSize);
        offset += _memoryWordSize;

        // Erase vector
        var eraseVector = interfaceVector.SubVector(offset, _memoryWordSize);
        offset += _memoryWordSize;

        // Write gate, allocation gate, write mode
        var writeGate = interfaceVector[offset++];
        var allocateGate = interfaceVector[offset++];
        var writeMode = interfaceVector[offset++];

        // Read modes
        var readModes = new List<Vector<T>>();
        for (int i = 0; i < _readHeads; i++)
        {
            readModes.Add(interfaceVector.SubVector(offset, 3));
            offset += 3;
        }

        // Update usage vector
        UpdateUsageVector();

        // Allocation weighting
        var allocationWeighting = CalculateAllocationWeighting();

        // Write weighting
        _writeWeighting = CalculateWriteWeighting(writeMode, allocationWeighting);

        // Write to memory
        WriteToMemory(writeVector, eraseVector, writeGate);

        // Update temporal linkage
        UpdateTemporalLinkage();

        // Read from memory
        ReadFromMemory(readModes);
    }

    /// <summary>
    /// Updates the usage tracking vector based on current write operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the usage tracking vector to reflect which memory locations have been written to.
    /// The usage free value for each location is reduced proportionally to the write weighting for that location.
    /// This helps the DNC track which memory locations are available for future write operations.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the record of which notepad pages are available.
    /// 
    /// The usage update process:
    /// - For each memory location (page in the notepad), update how "free" or available it is
    /// - If a location has just been written to (high value in WriteWeighting), mark it as less free
    /// - This ensures that important information isn't immediately overwritten
    /// - Locations that haven't been written to remain available for future use
    /// 
    /// This helps the system manage its memory efficiently, writing new information to
    /// unused or less important locations.
    /// </para>
    /// </remarks>
    private void UpdateUsageVector()
    {
        for (int i = 0; i < _memorySize; i++)
        {
            _usageFree[i] = NumOps.Multiply(
                _usageFree[i],
                NumOps.Subtract(NumOps.One, _writeWeighting[i])
            );
        }
    }

    /// <summary>
    /// Calculates the allocation weighting based on memory usage.
    /// </summary>
    /// <returns>A vector of allocation weights for memory locations.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the allocation weighting vector, which determines where to write new information
    /// based on memory usage. Locations that are free (high usage free value) and have been free for a long time
    /// receive higher allocation weights. This implements a dynamic memory allocation strategy that efficiently
    /// reuses memory locations.
    /// </para>
    /// <para><b>For Beginners:</b> This determines which free pages are best to write on.
    /// 
    /// The allocation weighting:
    /// - Decides which unused or less important memory locations to write to
    /// - Prioritizes locations that have been unused for a long time
    /// - Creates a graded list of preferences, with the most available locations getting highest priority
    /// - This implements a smart strategy for reusing memory efficiently
    /// 
    /// Think of it like choosing which blank pages in a notebook to use first when
    /// taking new notes.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateAllocationWeighting()
    {
        var sortedUsageFree = _usageFree.OrderBy(x => x).ToList();
        var allocationWeighting = Vector<T>.CreateDefault(_memorySize, NumOps.Zero);

        for (int i = 0; i < _memorySize; i++)
        {
            var product = NumOps.One;
            for (int j = 0; j < i; j++)
            {
                product = NumOps.Multiply(product, NumOps.Subtract(NumOps.One, _usageFree[j]));
            }
            allocationWeighting[i] = NumOps.Multiply(_usageFree[i], product);
        }

        return allocationWeighting;
    }

    /// <summary>
    /// Calculates the write weighting based on write mode and allocation weighting.
    /// </summary>
    /// <param name="writeMode">The write mode parameter from the interface vector.</param>
    /// <param name="allocationWeighting">The allocation weighting vector.</param>
    /// <returns>A vector of write weights for memory locations.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the write weighting vector, which determines where in memory to write information.
    /// It combines allocation-based writing (writing to free locations) with content-based writing (writing to locations
    /// based on their content similarity). The balance between these approaches is controlled by the write mode parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This decides exactly where to write in the notepad.
    /// 
    /// The write weighting calculation:
    /// - Balances two strategies for choosing where to write:
    ///   1. Writing to empty pages (allocation-based)
    ///   2. Writing to pages with similar content (content-based)
    /// - The writeMode parameter controls which strategy is more important
    /// - This creates a set of weights that determine how much to write to each location
    /// - Usually only a few locations receive significant weight
    /// 
    /// Think of it like deciding whether to write new information on a fresh page
    /// or to add it to a page that already contains related information.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateWriteWeighting(T writeMode, Vector<T> allocationWeighting)
    {
        var contentWeighting = ContentBasedAddressing(_writeWeighting, _memory);
        return allocationWeighting.Multiply(NumOps.Subtract(NumOps.One, writeMode))
            .Add(contentWeighting.Multiply(writeMode));
    }

    /// <summary>
    /// Writes information to memory based on the write weighting.
    /// </summary>
    /// <param name="writeVector">The vector of information to write.</param>
    /// <param name="eraseVector">The vector specifying what information to erase.</param>
    /// <param name="writeGate">The gate parameter controlling the strength of the write operation.</param>
    /// <remarks>
    /// <para>
    /// This method updates the memory matrix by writing new information and erasing old information. For each memory
    /// location, it first erases some content based on the erase vector and write weighting, then writes new content
    /// based on the write vector and write weighting. The write gate parameter controls the overall strength of the
    /// write operation.
    /// </para>
    /// <para><b>For Beginners:</b> This actually writes the information to the selected pages in the notepad.
    /// 
    /// The writing process:
    /// - For each memory location, based on its write weight:
    ///   1. First erase some of the existing information (like erasing parts of the page)
    ///   2. Then write new information to the location
    /// - The writeGate controls how strongly to perform this operation
    /// - If writeGate is low, only minor changes are made
    /// - If writeGate is high, significant information is written
    /// 
    /// This two-step process (erase then write) gives the DNC fine control over
    /// how information is stored and updated in memory.
    /// </para>
    /// </remarks>
    private void WriteToMemory(Vector<T> writeVector, Vector<T> eraseVector, T writeGate)
    {
        for (int i = 0; i < _memorySize; i++)
        {
            var eraseAmount = eraseVector.Multiply(_writeWeighting[i]);
            var writeAmount = writeVector.Multiply(_writeWeighting[i]);
            for (int j = 0; j < _memoryWordSize; j++)
            {
                _memory[i, j] = NumOps.Add(
                    NumOps.Multiply(
                        _memory[i, j],
                        NumOps.Subtract(NumOps.One, NumOps.Multiply(eraseAmount[j], writeGate))
                    ),
                    NumOps.Multiply(writeAmount[j], writeGate)
                );
            }
        }
    }

    /// <summary>
    /// Updates the temporal linkage information based on current write operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the temporal link matrix and precedence weighting vector based on current write operations.
    /// The temporal link matrix records which memory locations were written to before or after others, allowing the
    /// DNC to traverse memory in the order information was written. The precedence weighting vector tracks the recency
    /// of writes to each location.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the record of which pages were written to in what order.
    /// 
    /// The temporal linkage update:
    /// - Updates the "before/after" relationships between memory locations
    /// - Records which page was written to just before the current page
    /// - Updates which pages were most recently written to
    /// - This creates a chain of links that can be followed forward or backward in time
    /// 
    /// Think of it like keeping track of the order in which you wrote in different
    /// pages of the notepad, so you can later follow the sequence of information.
    /// </para>
    /// </remarks>
    private void UpdateTemporalLinkage()
    {
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memorySize; j++)
            {
                if (i != j)
                {
                    _temporalLinkMatrix[i, j] = NumOps.Add(
                        NumOps.Multiply(
                            NumOps.Subtract(NumOps.One, _writeWeighting[i]),
                            NumOps.Subtract(NumOps.One, _writeWeighting[j])
                        ),
                        NumOps.Multiply(_precedenceWeighting[i], _writeWeighting[j])
                    );
                }
            }
        }

        _precedenceWeighting = _precedenceWeighting.Multiply(NumOps.Subtract(NumOps.One, _writeWeighting.Sum()))
            .Add(_writeWeighting);
    }

    /// <summary>
    /// Reads information from memory based on read modes and weightings.
    /// </summary>
    /// <param name="readModes">The list of mode vectors for each read head.</param>
    /// <remarks>
    /// <para>
    /// This method reads information from memory for each read head, based on the read modes and weightings.
    /// For each read head, it computes new read weightings based on content similarity, backward traversal,
    /// and forward traversal, then reads the weighted average of memory rows according to these weightings.
    /// The resulting read vectors are stored in the read vectors list.
    /// </para>
    /// <para><b>For Beginners:</b> This reads information from selected pages in the notepad.
    /// 
    /// The reading process:
    /// - For each read head (finger in the notepad):
    ///   1. Decide where to read from using three strategies:
    ///      - Content-based: look at pages with similar content
    ///      - Backward: look at pages written before the current focus
    ///      - Forward: look at pages written after the current focus
    ///   2. The readModes control which strategy is more important for each head
    ///   3. Read information from the selected locations and combine it
    /// 
    /// This flexible reading system allows the DNC to retrieve information in different ways:
    /// by content similarity, by following chains of information backward, or by following them forward.
    /// </para>
    /// </remarks>
    private void ReadFromMemory(List<Vector<T>> readModes)
    {
        var newReadVectors = new List<Vector<T>>();
        for (int i = 0; i < _readHeads; i++)
        {
            var backwardWeighting = _temporalLinkMatrix.Multiply(_readWeightings[i]);
            var forwardWeighting = _temporalLinkMatrix.Transpose().Multiply(_readWeightings[i]);
            var contentWeighting = ContentBasedAddressing(_readWeightings[i], _memory);

            _readWeightings[i] = backwardWeighting.Multiply(readModes[i][0])
                .Add(contentWeighting.Multiply(readModes[i][1]))
                .Add(forwardWeighting.Multiply(readModes[i][2]));

            var readVector = Vector<T>.CreateDefault(_memoryWordSize, NumOps.Zero);
            for (int j = 0; j < _memorySize; j++)
            {
                readVector = readVector.Add(_memory.GetRow(j).Multiply(_readWeightings[i][j]));
            }
            newReadVectors.Add(readVector);
        }

        _readVectors = newReadVectors;
    }

    /// <summary>
    /// Performs content-based addressing of memory.
    /// </summary>
    /// <param name="key">The key vector to compare with memory contents.</param>
    /// <param name="memory">The memory matrix to search in.</param>
    /// <returns>A vector of attention weights based on content similarity.</returns>
    /// <remarks>
    /// <para>
    /// This method computes attention weights for memory locations based on the cosine similarity between the key vector
    /// and the content of each memory location. Locations with content similar to the key receive higher weights.
    /// The similarities are normalized using a softmax function to produce a probability distribution over locations.
    /// </para>
    /// <para><b>For Beginners:</b> This finds pages in the notepad that contain similar information to what we're looking for.
    /// 
    /// The content-based addressing:
    /// - Takes a "key" (what we're looking for) and compares it to every memory location
    /// - Calculates how similar each location's content is to the key
    /// - Gives higher weight to locations that contain similar information
    /// - Normalizes these weights so they form a probability distribution
    /// 
    /// This is like scanning through a notepad to find pages that contain information
    /// similar to what you're interested in right now.
    /// </para>
    /// </remarks>
    private Vector<T> ContentBasedAddressing(Vector<T> key, Matrix<T> memory)
    {
        var similarities = Vector<T>.CreateDefault(_memorySize, NumOps.Zero);
        for (int i = 0; i < _memorySize; i++)
        {
            similarities[i] = CosineSimilarity(key, memory.GetRow(i));
        }
        return Softmax(similarities);
    }

    /// <summary>
    /// Calculates the cosine similarity between two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The cosine similarity, a value between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the cosine similarity between two vectors, which is the cosine of the angle between them.
    /// It is calculated as the dot product of the vectors divided by the product of their norms. The result is a value
    /// between -1 and 1, where 1 means the vectors are identical in direction, 0 means they are orthogonal, and -1 means
    /// they are opposite in direction.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how similar two pieces of information are.
    /// 
    /// The cosine similarity:
    /// - Measures how similar two vectors (pieces of information) are in direction
    /// - Ranges from -1 (completely opposite) to 1 (exactly the same)
    /// - Ignores the magnitude (size) of the vectors, focusing only on their direction
    /// - Is a standard way to compare the similarity of content in machine learning
    /// 
    /// This is like comparing the topics of two pages in a notepad, regardless of
    /// how much is written on each page.
    /// </para>
    /// </remarks>
    private T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        var dotProduct = a.DotProduct(b);
        var normA = a.Norm();
        var normB = b.Norm();

        return NumOps.Divide(dotProduct, NumOps.Multiply(normA, normB));
    }

    /// <summary>
    /// Applies the softmax function to a vector to produce a probability distribution.
    /// </summary>
    /// <param name="vector">The input vector.</param>
    /// <returns>A vector representing a probability distribution.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the softmax function to a vector of values, converting them into a probability distribution.
    /// Each element of the input vector is exponentiated, and then the results are normalized by dividing by their sum.
    /// The output vector will have all positive elements that sum to 1.
    /// </para>
    /// <para><b>For Beginners:</b> This converts a list of scores into probabilities that add up to 100%.
    /// 
    /// The softmax function:
    /// - Takes a vector of any values and converts them to a probability distribution
    /// - Makes all values positive and ensures they sum to 1 (100%)
    /// - Preserves the ordering of values (higher inputs become higher probabilities)
    /// - Exaggerates differences between values
    /// 
    /// This is used to convert similarity scores or other measurements into attention weights
    /// that can be used as probabilities for selecting where to read from or write to.
    /// </para>
    /// </remarks>
    private Vector<T> Softmax(Vector<T> vector)
    {
        var expVector = new Vector<T>(vector.Select(x => NumOps.Exp(x)).ToArray());
        var sum = expVector.Sum();

        return expVector.Divide(sum);
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
    /// Serializes the Differentiable Neural Computer to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Differentiable Neural Computer to a binary stream. It writes the number of layers,
    /// followed by the type name and serialized state of each layer. It also serializes the memory matrix and other DNC-specific
    /// components. This allows the DNC to be saved to disk and later restored with its learned parameters and memory state intact.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the complete state of the DNC to a file.
    /// 
    /// When saving the DNC:
    /// - First, it saves the neural network layers and their parameters
    /// - Then, it saves the current state of the memory matrix
    /// - This preserves both what the DNC has learned and what it currently "remembers"
    /// 
    /// This is like taking a snapshot of both the brain and the notepad of the system,
    /// allowing you to restore exactly the same state later.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }

        // Serialize memory and other DNC-specific components
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                // Fix null reference warning
                string valueStr = _memory[i, j]?.ToString() ?? string.Empty;
                writer.Write(valueStr);
            }
        }
    }

    /// <summary>
    /// Deserializes the Differentiable Neural Computer from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer type information is invalid or instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Differentiable Neural Computer from a binary stream. It reads the number of layers,
    /// followed by the type name and serialized state of each layer. It also deserializes the memory matrix and other DNC-specific
    /// components. This allows a previously saved DNC to be restored from disk with all its learned parameters and memory state.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved DNC from a file.
    /// 
    /// When loading the DNC:
    /// - First, it loads the neural network layers and their parameters
    /// - Then, it loads the saved state of the memory matrix
    /// - This restores both what the DNC had learned and what it was "remembering"
    /// 
    /// This is like restoring a complete snapshot of both the brain and the notepad of the system,
    /// bringing it back to exactly the same state it was in when saved.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);
        }

        // Deserialize memory and other DNC-specific components
        for (int i = 0; i < _memorySize; i++)
        {
            for (int j = 0; j < _memoryWordSize; j++)
            {
                _memory[i, j] = (T)Convert.ChangeType(reader.ReadString(), typeof(T));
            }
        }
    }
}