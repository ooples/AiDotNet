using System.Collections.Generic;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents a neural network architecture discovered through NAS
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class Architecture<T>
    {
        /// <summary>
        /// Operations in the architecture: (to_node, from_node, operation)
        /// </summary>
        public List<(int ToNode, int FromNode, string Operation)> Operations { get; set; } = new List<(int, int, string)>();

        /// <summary>
        /// Optional per-node channel counts (typically output channels) used for cost estimation.
        /// </summary>
        /// <remarks>
        /// <para>
        /// When provided, hardware cost models can use these values to scale operation costs more accurately
        /// (e.g., accounting for channel expansion/reduction across layers). If not provided, cost models may
        /// fall back to assuming a uniform channel count.
        /// </para>
        /// <para><b>For Beginners:</b> This is a simple mapping like: node 0 has 16 channels, node 1 has 32 channels, etc.
        /// Some operations change how many features (channels) flow through the network, which affects compute cost.
        /// </para>
        /// </remarks>
        public Dictionary<int, int> NodeChannels { get; set; } = new Dictionary<int, int>();

        /// <summary>
        /// Number of nodes in the architecture
        /// </summary>
        public int NodeCount { get; set; }

        /// <summary>
        /// Adds an operation to the architecture
        /// </summary>
        public void AddOperation(int toNode, int fromNode, string operation)
        {
            Operations.Add((toNode, fromNode, operation));
            NodeCount = System.Math.Max(NodeCount, System.Math.Max(toNode, fromNode) + 1);
        }

        /// <summary>
        /// Gets a description of the architecture
        /// </summary>
        public string GetDescription()
        {
            var lines = new List<string>();
            lines.Add($"Architecture with {NodeCount} nodes:");

            foreach (var (toNode, fromNode, operation) in Operations)
            {
                lines.Add($"  Node {toNode} <- {operation} <- Node {fromNode}");
            }

            return string.Join("\n", lines);
        }
    }
}
