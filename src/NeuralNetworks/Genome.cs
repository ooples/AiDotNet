namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a genome in a neuroevolutionary algorithm, containing a collection of connections between nodes.
/// </summary>
/// <remarks>
/// <para>
/// A Genome is a fundamental data structure in neuroevolutionary algorithms like NEAT (NeuroEvolution of Augmenting Topologies).
/// It encodes the structure and weights of a neural network as a set of connections between nodes. Each connection has a weight,
/// an enabled/disabled state, and an innovation number that tracks its evolutionary history. Genomes can be mutated, crossed over,
/// and evaluated for fitness, allowing neural networks to evolve over generations rather than being trained through traditional
/// gradient-based methods.
/// </para>
/// <para><b>For Beginners:</b> A Genome is like a blueprint for a neural network.
/// 
/// Think of a Genome as:
/// - A DNA-like structure that defines how a neural network is built
/// - A collection of connections (wires) between nodes (neurons)
/// - Each connection has a weight (strength) and can be enabled or disabled
/// - Instead of training this network with examples, it evolves through generations
/// 
/// Just as biological organisms evolve through natural selection, these neural network blueprints
/// can evolve to solve problems through a process of selection, mutation, and reproduction.
/// The best-performing blueprints are selected to create the next generation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class Genome<T>
{
    /// <summary>
    /// Gets the list of connections that make up this genome.
    /// </summary>
    /// <value>A list of connections between nodes.</value>
    /// <remarks>
    /// <para>
    /// The Connections property contains all the connection genes that define the structure and weights of the
    /// neural network encoded by this genome. Each connection specifies a link between two nodes, with a weight
    /// determining the strength of the connection, an enabled/disabled state, and an innovation number for tracking
    /// its evolutionary history.
    /// </para>
    /// <para><b>For Beginners:</b> This is the collection of all wires connecting the neurons.
    /// 
    /// Think of Connections as:
    /// - All the wires connecting different neurons in the network
    /// - Each connection defines which neurons are connected
    /// - Each connection has a strength (weight) and can be turned on or off
    /// - Each connection also has an ID number (innovation) that tracks when it first appeared
    /// 
    /// This is the core of what makes each neural network unique - which neurons are connected,
    /// how strongly, and which connections are active.
    /// </para>
    /// </remarks>
    public List<Connection<T>> Connections { get; private set; }

    /// <summary>
    /// Gets or sets the fitness score of this genome.
    /// </summary>
    /// <value>A value representing how well this genome performs on its task.</value>
    /// <remarks>
    /// <para>
    /// The Fitness property represents how well this genome (the neural network it encodes) performs on its intended task.
    /// Higher fitness values typically indicate better performance. This fitness score is used during the evolutionary process
    /// to determine which genomes are selected for reproduction and which are eliminated.
    /// </para>
    /// <para><b>For Beginners:</b> This is the score that shows how well this network performs.
    /// 
    /// Think of Fitness as:
    /// - A scorecard that rates how good this neural network is at its job
    /// - Higher scores mean better performance
    /// - This score determines which networks get to "reproduce" and pass on their features
    /// - Low-scoring networks are eliminated, just like natural selection
    /// 
    /// For example, if the network is controlling a game character, the fitness might be
    /// the score achieved in the game. The highest-scoring networks would be selected
    /// to create the next generation.
    /// </para>
    /// </remarks>
    public T Fitness { get; set; }

    /// <summary>
    /// Gets the number of input nodes in the neural network.
    /// </summary>
    /// <value>An integer representing the number of input nodes.</value>
    /// <remarks>
    /// <para>
    /// The InputSize property defines how many input nodes the neural network has. These are the nodes that receive
    /// external input data. In the context of neuroevolution, the input size is typically fixed throughout evolution,
    /// as it is determined by the dimensionality of the problem's input data.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many data points the network takes as input.
    /// 
    /// Think of InputSize as:
    /// - The number of sensory neurons that receive information from outside
    /// - Each input neuron handles one piece of data
    /// - For a game AI, inputs might include position, speed, obstacle distances, etc.
    /// - For image processing, inputs might be pixel values
    /// 
    /// This value is usually fixed and determined by what kind of data the network needs to process.
    /// </para>
    /// </remarks>
    public int InputSize { get; private set; }

    /// <summary>
    /// Gets the number of output nodes in the neural network.
    /// </summary>
    /// <value>An integer representing the number of output nodes.</value>
    /// <remarks>
    /// <para>
    /// The OutputSize property defines how many output nodes the neural network has. These are the nodes that produce
    /// the network's output or decisions. In the context of neuroevolution, the output size is typically fixed throughout
    /// evolution, as it is determined by the dimensionality of the problem's output data.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many results the network produces as output.
    /// 
    /// Think of OutputSize as:
    /// - The number of action neurons that produce decisions or predictions
    /// - Each output neuron represents one aspect of the network's response
    /// - For a game AI, outputs might be movement directions or action choices
    /// - For classification, outputs might represent different categories
    /// 
    /// Like InputSize, this value is usually fixed and determined by what kind of decisions
    /// or predictions the network needs to make.
    /// </para>
    /// </remarks>
    public int OutputSize { get; private set; }

    /// <summary>
    /// Gets the numeric operations helper for the specified type T.
    /// </summary>
    /// <value>An interface providing numeric operations for type T.</value>
    /// <remarks>
    /// <para>
    /// This property provides access to numeric operations appropriate for the generic type T. It allows the
    /// Genome class to perform mathematical operations on values of type T, regardless of whether T is a
    /// float, double, or other numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that lets the code do math with different number types.
    /// 
    /// Think of NumOps as:
    /// - A calculator that knows how to work with whatever number type we're using
    /// - It provides operations like addition, multiplication, etc. for type T
    /// - This allows the same code to work with different number types (float, double, etc.)
    /// - You don't need to worry about this detail for understanding how genomes work
    /// 
    /// This is a technical implementation detail that helps the code be more flexible.
    /// </para>
    /// </remarks>
    private INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the <see cref="Genome{T}"/> class with the specified network dimensions.
    /// </summary>
    /// <param name="inputSize">The number of input nodes in the neural network.</param>
    /// <param name="outputSize">The number of output nodes in the neural network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Genome with the specified number of input and output nodes. It initializes
    /// an empty list of connections and sets the initial fitness to zero. This creates a bare neural network
    /// structure that can be populated with connections through mutation or other means.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new, empty neural network blueprint.
    /// 
    /// When creating a new Genome:
    /// - You specify how many inputs and outputs the network will have
    /// - It starts with no connections between neurons (an empty blueprint)
    /// - Its fitness score starts at zero since it hasn't been tested yet
    /// 
    /// This is like starting with a blank canvas before drawing the neural network.
    /// Connections will be added later through the evolution process.
    /// </para>
    /// </remarks>
    public Genome(int inputSize, int outputSize)
    {
        Connections = new List<Connection<T>>();
        InputSize = inputSize;
        OutputSize = outputSize;
        Fitness = NumOps.Zero;
    }

    /// <summary>
    /// Adds a new connection to this genome.
    /// </summary>
    /// <param name="fromNode">The identifier of the source node.</param>
    /// <param name="toNode">The identifier of the target node.</param>
    /// <param name="weight">The weight of the connection.</param>
    /// <param name="isEnabled">A value indicating whether the connection is enabled.</param>
    /// <param name="innovation">The innovation number of the connection.</param>
    /// <remarks>
    /// <para>
    /// This method adds a new connection to the genome, specifying the source and target nodes, the connection weight,
    /// whether the connection is enabled, and its innovation number. In neuroevolutionary algorithms, connections can
    /// be added through mutation operations, allowing the neural network's structure to evolve over time.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new wire connecting two neurons in the network.
    /// 
    /// When adding a connection:
    /// - fromNode and toNode specify which neurons to connect
    /// - weight determines the strength of the connection
    /// - isEnabled determines if the connection is active or inactive
    /// - innovation is a unique ID number that helps track the connection's history
    /// 
    /// This is how the neural network's structure grows during evolution.
    /// New connections allow more complex behavior to emerge.
    /// </para>
    /// </remarks>
    public void AddConnection(int fromNode, int toNode, T weight, bool isEnabled, int innovation)
    {
        Connections.Add(new Connection<T>(fromNode, toNode, weight, isEnabled, innovation));
    }

    /// <summary>
    /// Disables a connection with the specified innovation number.
    /// </summary>
    /// <param name="innovation">The innovation number of the connection to disable.</param>
    /// <remarks>
    /// <para>
    /// This method disables a connection in the genome by setting its IsEnabled property to false. This allows
    /// the network's structure to be modified without removing connections entirely. In neuroevolutionary algorithms,
    /// disabling connections is a common mutation operation that allows the network to explore different structural
    /// configurations while preserving the genetic information for potential future use.
    /// </para>
    /// <para><b>For Beginners:</b> This turns off a specific connection in the network.
    /// 
    /// When disabling a connection:
    /// - The connection is identified by its innovation number (unique ID)
    /// - The connection isn't removed, just deactivated
    /// - This is like turning off a specific wire in the network
    /// - The connection can be re-enabled later if needed
    /// 
    /// This is important in evolution because sometimes temporarily disabling
    /// a connection leads to better performance, and the connection's information
    /// is preserved for possible future use.
    /// </para>
    /// </remarks>
    public void DisableConnection(int innovation)
    {
        var conn = Connections.Find(c => c.Innovation == innovation);
        if (conn != null)
        {
            conn.IsEnabled = false;
        }
    }

    /// <summary>
    /// Activates the neural network encoded by this genome with the given input vector.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector produced by the neural network.</returns>
    /// <exception cref="ArgumentException">Thrown when the input vector's length doesn't match the expected input size.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a cycle is detected in the network during topological sorting.</exception>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the neural network encoded by this genome. It first validates
    /// that the input vector has the correct size, then initializes node values for input and output nodes.
    /// It sorts the connections topologically to ensure they are processed in the correct order (from inputs
    /// toward outputs), then processes each connection by multiplying the source node's value by the connection
    /// weight and adding the result to the target node's value. Finally, it applies activation functions and
    /// collects the output values.
    /// </para>
    /// <para><b>For Beginners:</b> This runs data through the neural network to get an output.
    /// 
    /// The activation process works like this:
    /// - First, it checks that you've provided the right number of inputs
    /// - It sets up all the neurons with their initial values
    /// - It sorts the connections to ensure signals flow in the right direction
    /// - For each connection, it multiplies the signal by the connection's weight
    /// - It applies activation functions to introduce non-linearity
    /// - Finally, it collects the values from the output neurons
    /// 
    /// This is how the network actually processes information and makes decisions,
    /// similar to how signals flow through neurons in a brain.
    /// </para>
    /// </remarks>
    public Vector<T> Activate(Vector<T> input)
    {
        if (input.Length != InputSize)
            throw new ArgumentException($"Input size mismatch. Expected {InputSize}, got {input.Length}.");

        var nodeValues = new Dictionary<int, T>();
        var nodeActivations = new Dictionary<int, IActivationFunction<T>>();
        var processedNodes = new HashSet<int>();

        // Initialize input nodes
        for (int i = 0; i < InputSize; i++)
        {
            nodeValues[i] = input[i];
            processedNodes.Add(i);
        }

        // Initialize output nodes
        for (int i = 0; i < OutputSize; i++)
        {
            nodeValues[InputSize + i] = NumOps.Zero;
        }

        // Topological sort of connections
        var sortedConnections = TopologicalSort(Connections);

        // Process connections in topological order
        foreach (var conn in sortedConnections)
        {
            if (!conn.IsEnabled) continue;

            if (!nodeValues.ContainsKey(conn.FromNode))
                nodeValues[conn.FromNode] = NumOps.Zero;

            if (!nodeValues.ContainsKey(conn.ToNode))
                nodeValues[conn.ToNode] = NumOps.Zero;

            var value = NumOps.Multiply(nodeValues[conn.FromNode], conn.Weight);
            nodeValues[conn.ToNode] = NumOps.Add(nodeValues[conn.ToNode], value);

            processedNodes.Add(conn.ToNode);
        }

        // Apply activation functions to all processed nodes
        foreach (var node in processedNodes)
        {
            if (nodeActivations.TryGetValue(node, out var activation))
            {
                nodeValues[node] = activation.Activate(nodeValues[node]);
            }
        }

        // Collect output values
        var output = new T[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            output[i] = nodeValues.TryGetValue(InputSize + i, out var value) ? value : NumOps.Zero;
        }

        return new Vector<T>(output);
    }

    /// <summary>
    /// Sorts the connections in topological order from inputs to outputs.
    /// </summary>
    /// <param name="connections">The list of connections to sort.</param>
    /// <returns>A list of connections sorted in topological order.</returns>
    /// <exception cref="InvalidOperationException">Thrown when a cycle is detected in the network.</exception>
    /// <remarks>
    /// <para>
    /// This method performs a topological sort on the connections to ensure they are processed in the correct order
    /// during activation. A topological sort arranges the connections so that for each connection, all connections
    /// leading to its source node come before it in the sorted list. This is necessary because the value of a node
    /// depends on the values of all nodes that have connections to it. The method also detects and reports any cycles
    /// in the network, which would make a strict topological order impossible.
    /// </para>
    /// <para><b>For Beginners:</b> This sorts the connections to ensure signals flow in the right direction.
    /// 
    /// Topological sorting:
    /// - Makes sure connections are processed in the correct order
    /// - Ensures that a neuron's inputs are all calculated before its output
    /// - Detects any loops (cycles) in the network that would cause problems
    /// - Is like sorting the steps in a recipe to make sure earlier steps come first
    /// 
    /// This is important because neural networks need to process information in 
    /// a specific order, flowing from inputs through hidden nodes to outputs.
    /// </para>
    /// </remarks>
    private List<Connection<T>> TopologicalSort(List<Connection<T>> connections)
    {
        var sorted = new List<Connection<T>>();
        var visited = new HashSet<int>();
        var tempMark = new HashSet<int>();

        void Visit(int node)
        {
            if (tempMark.Contains(node))
                throw new InvalidOperationException("Cycle detected in the network.");
            if (!visited.Contains(node))
            {
                tempMark.Add(node);
                foreach (var conn in connections.Where(c => c.FromNode == node))
                {
                    Visit(conn.ToNode);
                }
                visited.Add(node);
                tempMark.Remove(node);
                sorted.InsertRange(0, connections.Where(c => c.FromNode == node));
            }
        }

        foreach (var conn in connections)
        {
            if (!visited.Contains(conn.FromNode))
                Visit(conn.FromNode);
        }

        return sorted;
    }

    /// <summary>
    /// Creates a deep copy of this genome.
    /// </summary>
    /// <returns>A new Genome instance that is a copy of this instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the genome, including all its connections. The new genome has the same
    /// input size, output size, and connections as the original, but it is a separate instance that can be modified
    /// independently. Cloning is often used in neuroevolutionary algorithms when creating offspring genomes that
    /// will then be mutated or otherwise modified.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an exact copy of the neural network blueprint.
    /// 
    /// When cloning a genome:
    /// - A new, independent copy of the genome is created
    /// - The copy has the same input size, output size, and connections
    /// - The copy can be modified without affecting the original
    /// - This is useful during reproduction when creating offspring
    /// 
    /// Think of it like photocopying a blueprint - you get an identical copy
    /// that you can then modify independently.
    /// </para>
    /// </remarks>
    public Genome<T> Clone()
    {
        var clone = new Genome<T>(InputSize, OutputSize);
        foreach (var conn in Connections)
        {
            clone.AddConnection(conn.FromNode, conn.ToNode, conn.Weight, conn.IsEnabled, conn.Innovation);
        }

        return clone;
    }

    /// <summary>
    /// Serializes this genome to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the state of the genome to a binary stream. It writes the input size, output size,
    /// connection count, and then the details of each connection. This allows the genome to be saved to disk
    /// and later restored with all its structural information intact.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the neural network blueprint to a file.
    /// 
    /// When serializing a genome:
    /// - It saves the number of inputs and outputs
    /// - It saves how many connections the network has
    /// - For each connection, it saves all its details (source, target, weight, etc.)
    /// 
    /// This is like saving a blueprint to disk so you can reload it later
    /// or share it with others.
    /// </para>
    /// </remarks>
    public void Serialize(BinaryWriter writer)
    {
        writer.Write(InputSize);
        writer.Write(OutputSize);
        writer.Write(Connections.Count);
        foreach (var conn in Connections)
        {
            writer.Write(conn.FromNode);
            writer.Write(conn.ToNode);
            writer.Write(Convert.ToDouble(conn.Weight));
            writer.Write(conn.IsEnabled);
            writer.Write(conn.Innovation);
        }
    }

    /// <summary>
    /// Deserializes this genome from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method restores the state of the genome from a binary stream. It reads the input size, output size,
    /// connection count, and then the details of each connection. This allows a previously saved genome to be
    /// restored from disk with all its structural information intact.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a neural network blueprint from a file.
    /// 
    /// When deserializing a genome:
    /// - It reads the number of inputs and outputs
    /// - It reads how many connections the network has
    /// - For each connection, it reads all its details (source, target, weight, etc.)
    /// 
    /// This is like loading a blueprint from disk so you can use it
    /// or continue modifying it.
    /// </para>
    /// </remarks>
    public void Deserialize(BinaryReader reader)
    {
        InputSize = reader.ReadInt32();
        OutputSize = reader.ReadInt32();
        int connectionCount = reader.ReadInt32();
        Connections.Clear();
        for (int i = 0; i < connectionCount; i++)
        {
            int fromNode = reader.ReadInt32();
            int toNode = reader.ReadInt32();
            T weight = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T));
            bool isEnabled = reader.ReadBoolean();
            int innovation = reader.ReadInt32();
            AddConnection(fromNode, toNode, weight, isEnabled, innovation);
        }
    }
}
