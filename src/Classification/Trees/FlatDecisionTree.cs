namespace AiDotNet.Classification.Trees;

/// <summary>
/// A trained decision tree as flat arrays, for evaluating many rows: no recursion, no per-row allocation, and the
/// thresholds as doubles compared against a row already converted to doubles.
/// </summary>
/// <remarks>
/// <para>
/// Nodes are laid out in preorder, so a node's left child is the next node, and each node is one small struct: a step
/// down the tree reads one cache line.
/// </para>
/// <para>
/// A row goes left when its value is not above the threshold, as the tree's own traversal does
/// (<c>Compare(value, threshold) &lt;= 0</c>); a NaN value goes left in both. Converting to double keeps every
/// comparison only for <c>double</c> and <c>float</c> features, so callers use this for those types alone.
/// </para>
/// </remarks>
internal sealed class FlatDecisionTree<T>
{
    /// <summary>A split (<see cref="Feature"/> &gt;= 0) or a leaf (<see cref="Feature"/> = -1).</summary>
    private struct Node
    {
        public double Threshold;
        public int Feature;

        /// <summary>A split's right child; a leaf's index into the leaf probabilities.</summary>
        public int Next;
    }

    private readonly Node[] _nodes;
    private readonly List<Vector<T>> _leaves = new();
    private int _laid;

    private FlatDecisionTree(int nodes)
    {
        _nodes = new Node[nodes];
    }

    /// <summary>Flattens the tree under <paramref name="root"/>.</summary>
    internal static FlatDecisionTree<T> From(DecisionNode<T> root)
    {
        int count = 0;
        var walk = new Stack<DecisionNode<T>>();
        walk.Push(root);
        while (walk.Count > 0)
        {
            var node = walk.Pop();
            count++;
            if (!node.IsLeaf)
            {
                walk.Push(LeftOf(node));
                walk.Push(RightOf(node));
            }
        }

        var tree = new FlatDecisionTree<T>(count);
        tree.Lay(root, MathHelper.GetNumericOperations<T>());
        return tree;
    }

    /// <summary>Lays out <paramref name="node"/> and its subtree in preorder; returns the node's index.</summary>
    private int Lay(DecisionNode<T> node, INumericOperations<T> numOps)
    {
        int index = _laid++;
        if (node.IsLeaf)
        {
            _nodes[index].Feature = -1;
            _nodes[index].Next = _leaves.Count;
            _leaves.Add(node.ClassProbabilities
                ?? throw new InvalidOperationException("ClassProbabilities has not been initialized."));
            return index;
        }

        _nodes[index].Feature = node.FeatureIndex;
        _nodes[index].Threshold = numOps.ToDouble(node.Threshold);
        Lay(LeftOf(node), numOps);
        _nodes[index].Next = Lay(RightOf(node), numOps);
        return index;
    }

    private static DecisionNode<T> LeftOf(DecisionNode<T> node) =>
        node.Left ?? throw new InvalidOperationException("node.Left is null where it is required.");

    private static DecisionNode<T> RightOf(DecisionNode<T> node) =>
        node.Right ?? throw new InvalidOperationException("node.Right is null where it is required.");

    /// <summary>The class probabilities of the leaf reached by the row starting at <paramref name="offset"/>.</summary>
    public Vector<T> Leaf(double[] rows, int offset)
    {
        int index = 0;
        while (true)
        {
            ref Node node = ref _nodes[index];
            if (node.Feature < 0)
                return _leaves[node.Next];
            index = rows[offset + node.Feature] > node.Threshold ? node.Next : index + 1;
        }
    }
}
