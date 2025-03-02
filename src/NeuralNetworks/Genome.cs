namespace AiDotNet.NeuralNetworks;

public class Genome<T>
{
    public List<Connection<T>> Connections { get; private set; }
    public T Fitness { get; set; }
    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    private INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    public Genome(int inputSize, int outputSize)
    {
        Connections = [];
        InputSize = inputSize;
        OutputSize = outputSize;
        Fitness = NumOps.Zero;
    }

    public void AddConnection(int fromNode, int toNode, T weight, bool isEnabled, int innovation)
    {
        Connections.Add(new Connection<T>(fromNode, toNode, weight, isEnabled, innovation));
    }

    public void DisableConnection(int innovation)
    {
        var conn = Connections.Find(c => c.Innovation == innovation);
        if (conn != null)
        {
            conn.IsEnabled = false;
        }
    }

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

    public Genome<T> Clone()
    {
        var clone = new Genome<T>(InputSize, OutputSize);
        foreach (var conn in Connections)
        {
            clone.AddConnection(conn.FromNode, conn.ToNode, conn.Weight, conn.IsEnabled, conn.Innovation);
        }

        return clone;
    }

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
