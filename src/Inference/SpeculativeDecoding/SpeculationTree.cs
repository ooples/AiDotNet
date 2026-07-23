using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Internal tree structure for speculation.
/// </summary>
/// <typeparam name="T">The numeric type for probabilities.</typeparam>
internal class SpeculationTree<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Root node of the tree.
    /// </summary>
    public TreeNode<T> Root { get; }

    /// <summary>
    /// Total number of nodes in the tree.
    /// </summary>
    public int TotalNodes { get; set; }

    private readonly int _branchFactor;
    private readonly int _maxDepth;

    /// <summary>
    /// Creates a new speculation tree.
    /// </summary>
    /// <param name="branchFactor">Number of branches per node.</param>
    /// <param name="maxDepth">Maximum tree depth.</param>
    public SpeculationTree(int branchFactor, int maxDepth)
    {
        _branchFactor = branchFactor;
        _maxDepth = maxDepth;
        Root = new TreeNode<T> { Depth = 0 };
        TotalNodes = 1;
    }

    /// <summary>
    /// Gets all paths through the tree.
    /// </summary>
    /// <returns>List of paths, each path as a vector of token IDs.</returns>
    public List<Vector<int>> GetAllPaths()
    {
        var paths = new List<Vector<int>>();
        CollectPaths(Root, new List<int>(), paths);
        return paths;
    }

    /// <summary>
    /// Gets probabilities for a specific path.
    /// </summary>
    /// <param name="pathIndex">Index of the path.</param>
    /// <returns>Vector of probabilities for each token in the path.</returns>
    public Vector<T> GetPathProbabilities(int pathIndex)
    {
        var allPaths = GetAllPaths();
        if (pathIndex >= allPaths.Count)
            return new Vector<T>(0);

        var path = allPaths[pathIndex];
        var probs = new Vector<T>(path.Length);

        // Traverse tree to collect probabilities
        var node = Root;
        for (int i = 0; i < path.Length; i++)
        {
            TreeNode<T>? child = null;
            foreach (var c in node.Children)
            {
                if (c.Token == path[i])
                {
                    child = c;
                    break;
                }
            }

            if (child != null)
            {
                probs[i] = child.Probability;
                node = child;
            }
            else
            {
                probs[i] = NumOps.FromDouble(0.01); // Default probability
            }
        }

        return probs;
    }

    private void CollectPaths(TreeNode<T> node, List<int> current, List<Vector<int>> paths)
    {
        if (node.Children.Count == 0)
        {
            if (current.Count > 0)
                paths.Add(new Vector<int>(current.ToArray()));
            return;
        }

        foreach (var child in node.Children)
        {
            current.Add(child.Token);
            CollectPaths(child, current, paths);
            current.RemoveAt(current.Count - 1);
        }
    }
}
