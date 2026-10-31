namespace AiDotNet.Finance.Graph;

/// <summary>
/// Checks shared by the graph models that take a fixed adjacency matrix.
/// </summary>
internal static class GraphAdjacency
{
    /// <summary>
    /// Requires a supplied adjacency matrix to be <paramref name="numNodes"/> x <paramref name="numNodes"/>. A matrix of
    /// another size used to reach the Laplacian and the graph convolution, which index it by node and failed with an
    /// IndexOutOfRangeException far from the argument that caused it.
    /// </summary>
    /// <param name="adjacency">The supplied matrix, or null when the model builds its own.</param>
    /// <param name="numNodes">The node count the model is configured for.</param>
    /// <param name="paramName">The constructor parameter that supplied the matrix.</param>
    internal static void RequireNodeCount(double[,]? adjacency, int numNodes, string paramName)
    {
        if (adjacency is null) return;
        if (adjacency.GetLength(0) != numNodes || adjacency.GetLength(1) != numNodes)
        {
            throw new ArgumentException(
                $"The adjacency matrix is {adjacency.GetLength(0)} x {adjacency.GetLength(1)}, but the model has {numNodes} " +
                $"nodes (NumNodes). Pass a {numNodes} x {numNodes} matrix, or set NumNodes to the node count of the graph.",
                paramName);
        }
    }
}
