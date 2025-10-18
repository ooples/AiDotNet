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
