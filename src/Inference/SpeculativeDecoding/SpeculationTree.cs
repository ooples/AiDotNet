namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Internal tree structure for speculation.
/// </summary>
internal class SpeculationTree
{
    public TreeNode Root { get; }
    public int TotalNodes { get; set; }
    private readonly int _branchFactor;
    private readonly int _maxDepth;

    public SpeculationTree(int branchFactor, int maxDepth)
    {
        _branchFactor = branchFactor;
        _maxDepth = maxDepth;
        Root = new TreeNode { Depth = 0 };
        TotalNodes = 1;
    }

    public List<int[]> GetAllPaths()
    {
        var paths = new List<int[]>();
        CollectPaths(Root, [], paths);
        return paths;
    }

    public float[] GetPathProbabilities(int pathIndex)
    {
        var allPaths = GetAllPaths();
        if (pathIndex >= allPaths.Count)
            return [];

        var path = allPaths[pathIndex];
        var probs = new float[path.Length];

        // Traverse tree to collect probabilities
        var node = Root;
        for (int i = 0; i < path.Length; i++)
        {
            var child = node.Children.FirstOrDefault(c => c.Token == path[i]);
            if (child != null)
            {
                probs[i] = child.Probability;
                node = child;
            }
            else
            {
                probs[i] = 0.01f; // Default
            }
        }

        return probs;
    }

    private void CollectPaths(TreeNode node, List<int> current, List<int[]> paths)
    {
        if (node.Children.Count == 0)
        {
            if (current.Count > 0)
                paths.Add([.. current]);
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
