namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a connection between two nodes in a neural network, particularly used in evolving neural networks.
/// </summary>
/// <remarks>
/// <para>
/// A Connection represents a weighted link between two nodes in a neural network. Connections are fundamental
/// elements in neural networks, allowing signals to flow from one node to another with a specific weight.
/// This class is particularly designed for use in evolutionary algorithms like NEAT (NeuroEvolution of Augmenting 
/// Topologies), which evolve both the weights and structure of neural networks. The Innovation number serves as 
/// a historical marker to track the evolutionary lineage of connections.
/// </para>
/// <para><b>For Beginners:</b> A Connection is like a wire connecting two parts of a neural network.
/// 
/// Think of a neural network as a system of connected nodes (like neurons in a brain):
/// - Each Connection is like a wire that passes signals from one node to another
/// - The Weight determines how strong the signal is (like a volume knob)
/// - IsEnabled acts like an on/off switch for the connection
/// - The Innovation number is like a birth certificate that shows when this connection first appeared
/// 
/// For example, if node 3 connects to node 5 with a weight of 0.7, signals from node 3 will reach node 5,
/// but their strength will be multiplied by 0.7 (either amplified or reduced depending on the original value).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for the weight, typically float or double.</typeparam>
public class Connection<T>
{
    /// <summary>
    /// Gets the identifier of the node from which the connection originates.
    /// </summary>
    /// <value>The identifier of the source node.</value>
    /// <remarks>
    /// <para>
    /// The FromNode property identifies the node where the connection begins. In the context of signal flow,
    /// this is the source node that sends a signal through the connection to the target node.
    /// </para>
    /// <para><b>For Beginners:</b> This is the ID number of the starting point of the connection.
    /// 
    /// Think of it like the address of the house where a mail delivery begins:
    /// - Each node in the network has a unique ID number (like a house address)
    /// - The FromNode tells you where the signal starts its journey
    /// - In a diagram, this would be the node on the left side of an arrow
    /// 
    /// For example, if FromNode is 3, the connection starts at node #3 in the network.
    /// </para>
    /// </remarks>
    public int FromNode { get; }

    /// <summary>
    /// Gets the identifier of the node to which the connection leads.
    /// </summary>
    /// <value>The identifier of the target node.</value>
    /// <remarks>
    /// <para>
    /// The ToNode property identifies the node where the connection ends. In the context of signal flow,
    /// this is the target node that receives a signal from the source node through the connection.
    /// </para>
    /// <para><b>For Beginners:</b> This is the ID number of the end point of the connection.
    /// 
    /// Think of it like the destination address for a mail delivery:
    /// - The ToNode tells you where the signal ends its journey
    /// - In a diagram, this would be the node on the right side of an arrow
    /// 
    /// For example, if ToNode is 5, the connection ends at node #5 in the network.
    /// </para>
    /// </remarks>
    public int ToNode { get; }

    /// <summary>
    /// Gets or sets the weight of the connection, which determines the strength of the signal transmission.
    /// </summary>
    /// <value>The weight of the connection.</value>
    /// <remarks>
    /// <para>
    /// The Weight property determines how strongly the signal from the source node influences the target node.
    /// Positive weights enhance the signal, negative weights inhibit it, and the magnitude indicates the strength
    /// of the effect. During learning or evolution, weights are adjusted to improve the network's performance.
    /// </para>
    /// <para><b>For Beginners:</b> The Weight is like a volume knob for the signal passing through the connection.
    /// 
    /// Think of the weight as a multiplier:
    /// - A positive weight (like 0.7) passes the signal through, but might reduce its strength
    /// - A negative weight (like -0.3) reverses the signal (if it was "turn right", it becomes "turn left")
    /// - A weight with larger magnitude (like 2.5) amplifies the signal, making it more influential
    /// - A weight near zero (like 0.01) makes the signal very weak, almost ignoring it
    /// 
    /// When a neural network learns, it's mostly adjusting these weight values to get better results.
    /// </para>
    /// </remarks>
    public T Weight { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the connection is enabled and actively transmitting signals.
    /// </summary>
    /// <value>True if the connection is enabled; otherwise, false.</value>
    /// <remarks>
    /// <para>
    /// The IsEnabled property determines whether the connection actively participates in signal transmission.
    /// When disabled, the connection is still part of the network structure but does not pass any signal.
    /// This is particularly useful in evolutionary algorithms, where disabling connections (rather than removing them)
    /// allows for potential reactivation in future generations.
    /// </para>
    /// <para><b>For Beginners:</b> IsEnabled works like an on/off switch for the connection.
    /// 
    /// Think of it like a light switch:
    /// - When IsEnabled is true, the connection is active and passes signals
    /// - When IsEnabled is false, the connection is inactive and blocks signals
    /// 
    /// This feature is especially useful in evolving neural networks because:
    /// - It allows the network to "turn off" connections that aren't helpful
    /// - The connection can be turned back on later if needed
    /// - This is often better than completely removing the connection
    /// 
    /// For example, during evolution, a connection might be disabled if it leads to poor performance,
    /// but could be re-enabled in a later generation if the network's structure changes.
    /// </para>
    /// </remarks>
    public bool IsEnabled { get; set; }

    /// <summary>
    /// Gets the innovation number, a historical marker that uniquely identifies the connection in the context of evolution.
    /// </summary>
    /// <value>The innovation number of the connection.</value>
    /// <remarks>
    /// <para>
    /// The Innovation property is a unique identifier assigned to each new connection during the evolution process.
    /// It serves as a historical marker that helps track the origin of connections across different generations
    /// and individuals. This is crucial in evolutionary algorithms like NEAT, where matching genes (connections)
    /// across individuals is necessary for effective crossover operations.
    /// </para>
    /// <para><b>For Beginners:</b> The Innovation number is like a birth certificate or ID for the connection.
    /// 
    /// Think of it as a timestamp or serial number:
    /// - Each time a new type of connection appears during evolution, it gets a unique Innovation number
    /// - Two connections with the same Innovation number in different networks represent the same "genetic innovation"
    /// - This helps when combining two networks (like parents having a child):
    ///   - If both parent networks have a connection with Innovation #42, the child will inherit one of them
    ///   - If only one parent has a connection with Innovation #86, the child might inherit that unique connection
    /// 
    /// This tracking system helps maintain diversity while allowing networks to share successful innovations.
    /// </para>
    /// </remarks>
    public int Innovation { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Connection{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="fromNode">The identifier of the source node.</param>
    /// <param name="toNode">The identifier of the target node.</param>
    /// <param name="weight">The weight of the connection.</param>
    /// <param name="isEnabled">A value indicating whether the connection is enabled.</param>
    /// <param name="innovation">The innovation number of the connection.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Connection with the specified parameters. It sets up all properties
    /// of the connection, including which nodes it connects, its weight, whether it's enabled, and its
    /// innovation number for evolutionary tracking.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "recipe" for creating a new connection between two nodes.
    /// 
    /// When creating a new connection, you need to specify:
    /// - fromNode: The ID of the starting node
    /// - toNode: The ID of the ending node
    /// - weight: How strong the connection is
    /// - isEnabled: Whether the connection is turned on
    /// - innovation: The unique ID number for this connection type
    /// 
    /// For example, new Connection(3, 5, 0.7, true, 42) creates a connection:
    /// - From node 3 to node 5
    /// - With a weight of 0.7
    /// - That is currently enabled (active)
    /// - With innovation number 42 (tracking its evolutionary origin)
    /// </para>
    /// </remarks>
    public Connection(int fromNode, int toNode, T weight, bool isEnabled, int innovation)
    {
        FromNode = fromNode;
        ToNode = toNode;
        Weight = weight;
        IsEnabled = isEnabled;
        Innovation = innovation;
    }
}
