namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual in genetic programming with a tree structure.
/// </summary>
public class TreeIndividual : IEvolvable<NodeGene, double>
{
    private NodeGene _rootNode;
    private double _fitness;
    private readonly Random _random;
    private readonly List<string> _availableFunctions = ["+", "-", "*", "/"];
    private readonly int _maxInitialDepth = 6;
    private readonly int _maxDepth = 12;

    /// <summary>
    /// Creates a new tree individual with a random tree.
    /// </summary>
    /// <param name="random">Random number generator for initialization.</param>
    /// <param name="terminals">The terminal symbols available.</param>
    /// <param name="fullMethod">Whether to use the full or grow method.</param>
    public TreeIndividual(Random random, List<string> terminals, bool fullMethod = false)
    {
        _random = random;
        _rootNode = GenerateRandomTree(0, terminals, fullMethod);
    }

    /// <summary>
    /// Creates a tree individual with the specified root node.
    /// </summary>
    /// <param name="rootNode">The root node of the tree.</param>
    public TreeIndividual(NodeGene rootNode)
    {
        _rootNode = rootNode;
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Generates a random subtree.
    /// </summary>
    /// <summary>
    /// Generates a random subtree.
    /// </summary>
    private NodeGene GenerateRandomTree(int depth, List<string> terminals, bool fullMethod, int? maxDepth = null)
    {
        int effectiveMaxDepth = maxDepth ?? _maxInitialDepth;

        // Terminal node is forced at max depth or randomly in grow method
        bool isTerminal = depth >= effectiveMaxDepth || (!fullMethod && depth > 1 && _random.Next(2) == 0);

        if (isTerminal)
        {
            // Choose random terminal
            string value = terminals[_random.Next(terminals.Count)];
            return new NodeGene(GeneticNodeType.Terminal, value);
        }
        else
        {
            // Choose random function
            string function = _availableFunctions[_random.Next(_availableFunctions.Count)];
            NodeGene node = new NodeGene(GeneticNodeType.Function, function);

            // Add children based on function arity
            int arity = GetFunctionArity(function);
            for (int i = 0; i < arity; i++)
            {
                node.Children.Add(GenerateRandomTree(depth + 1, terminals, fullMethod, maxDepth));
            }

            return node;
        }
    }

    /// <summary>
    /// Gets the arity (number of arguments) for a function.
    /// </summary>
    private int GetFunctionArity(string function)
    {
        return function switch
        {
            "+" or "-" or "*" or "/" => 2,
            "sin" or "cos" or "exp" or "log" => 1,
            _ => throw new ArgumentException($"Unknown function: {function}")
        };
    }

    /// <summary>
    /// Evaluates the tree for a given input and returns the result.
    /// </summary>
    public double Evaluate(Dictionary<string, double> variables)
    {
        return EvaluateNode(_rootNode, variables);
    }

    /// <summary>
    /// Recursively evaluates a node in the tree.
    /// </summary>
    private double EvaluateNode(NodeGene node, Dictionary<string, double> variables)
    {
        if (node.Type == GeneticNodeType.Terminal)
        {
            // If it's a variable, get its value
            if (variables.TryGetValue(node.Value, out double value))
            {
                return value;
            }

            // Otherwise, it's a constant
            return double.Parse(node.Value);
        }
        else
        {
            // Evaluate children
            var childValues = node.Children.Select(c => EvaluateNode(c, variables)).ToArray();

            // Apply function
            return node.Value switch
            {
                "+" => childValues[0] + childValues[1],
                "-" => childValues[0] - childValues[1],
                "*" => childValues[0] * childValues[1],
                "/" => childValues[1] == 0 ? 1.0 : childValues[0] / childValues[1], // Protected division
                "sin" => Math.Sin(childValues[0]),
                "cos" => Math.Cos(childValues[0]),
                "exp" => Math.Exp(childValues[0]),
                "log" => childValues[0] <= 0 ? 0.0 : Math.Log(childValues[0]), // Protected log
                _ => throw new ArgumentException($"Unknown function: {node.Value}")
            };
        }
    }

    /// <summary>
    /// Selects a random node from the tree.
    /// </summary>
    public NodeGene SelectRandomNode()
    {
        List<NodeGene> allNodes = new List<NodeGene>();
        CollectNodes(_rootNode, allNodes);
        return allNodes[_random.Next(allNodes.Count)];
    }

    /// <summary>
    /// Collects all nodes in the tree.
    /// </summary>
    private void CollectNodes(NodeGene node, List<NodeGene> nodes)
    {
        nodes.Add(node);
        foreach (var child in node.Children)
        {
            CollectNodes(child, nodes);
        }
    }

    /// <summary>
    /// Replaces a subtree with another subtree.
    /// </summary>
    public void ReplaceSubtree(NodeGene target, NodeGene replacement)
    {
        if (target == _rootNode)
        {
            _rootNode = replacement;
            return;
        }

        ReplaceInNode(_rootNode, target, replacement);
    }

    /// <summary>
    /// Ensures the tree doesn't exceed the maximum allowed depth.
    /// </summary>
    private void EnsureValidDepth()
    {
        int currentDepth = GetDepth();
        if (currentDepth > _maxDepth)
        {
            // If tree is too deep, replace it with a simpler one
            var terminals = new List<string> { "x", "1.0", "2.0", "0.5" }; // Example terminals
            _rootNode = GenerateRandomTree(0, terminals, false, _maxDepth);
        }
    }

    /// <summary>
    /// Recursively searches and replaces a node in the tree.
    /// </summary>
    private bool ReplaceInNode(NodeGene current, NodeGene target, NodeGene replacement)
    {
        for (int i = 0; i < current.Children.Count; i++)
        {
            if (current.Children[i] == target)
            {
                current.Children[i] = replacement;
                return true;
            }

            if (ReplaceInNode(current.Children[i], target, replacement))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Performs point mutation on the tree by changing a random terminal or function.
    /// </summary>
    public void PointMutation()
    {
        var node = SelectRandomNode();

        if (node.Type == GeneticNodeType.Terminal)
        {
            // For simplicity, this assumes all terminals are variables or constants
            // In a real implementation, you'd need to handle different terminal types
            node.Value = _random.NextDouble().ToString("F2");
        }
        else
        {
            // Get functions with same arity to maintain tree validity
            int currentArity = node.Children.Count;
            var compatibleFunctions = _availableFunctions
                .Where(f => GetFunctionArity(f) == currentArity)
                .ToList();

            if (compatibleFunctions.Count > 0)
            {
                node.Value = compatibleFunctions[_random.Next(compatibleFunctions.Count)];
            }
        }
    }

    /// <summary>
    /// Performs subtree mutation by replacing a random subtree with a new one.
    /// </summary>
    public void SubtreeMutation()
    {
        var node = SelectRandomNode();
        var terminals = new List<string> { "x", "1.0", "2.0", "0.5" }; // Example terminals

        // Calculate current depth of the node to ensure we don't exceed max depth
        int currentDepth = GetNodeDepthInTree(node);
        int maxAllowedSubtreeDepth = _maxDepth - currentDepth;

        // Generate a new subtree with limited depth
        var newSubtree = GenerateRandomTree(0, terminals, false, maxAllowedSubtreeDepth);

        if (node == _rootNode)
        {
            _rootNode = newSubtree;
        }
        else
        {
            ReplaceSubtree(node, newSubtree);
        }
    }

    /// <summary>
    /// Gets the depth of a specific node within the tree.
    /// </summary>
    private int GetNodeDepthInTree(NodeGene target)
    {
        return FindNodeDepth(_rootNode, target, 0);
    }

    /// <summary>
    /// Recursively finds the depth of a target node in the tree.
    /// </summary>
    private int FindNodeDepth(NodeGene current, NodeGene target, int depth)
    {
        if (current == target)
            return depth;

        foreach (var child in current.Children)
        {
            int result = FindNodeDepth(child, target, depth + 1);
            if (result >= 0)
                return result;
        }

        return -1; // Node not found in this branch
    }

    /// <summary>
    /// Performs permutation mutation by randomizing the order of arguments.
    /// </summary>
    public void PermutationMutation()
    {
        var node = SelectRandomNode();

        if (node.Type == GeneticNodeType.Function && node.Children.Count > 1)
        {
            // Only for commutative operations like + and * 
            if (node.Value == "+" || node.Value == "*")
            {
                // Shuffle children
                var shuffled = node.Children.OrderBy(x => _random.Next()).ToList();
                node.Children = shuffled;
            }
        }
    }

    /// <summary>
    /// Gets the depth of the tree.
    /// </summary>
    public int GetDepth()
    {
        return GetNodeDepth(_rootNode);
    }

    /// <summary>
    /// Gets the depth of a node.
    /// </summary>
    private int GetNodeDepth(NodeGene node)
    {
        if (node.Children.Count == 0)
            return 0;

        return 1 + node.Children.Max(GetNodeDepth);
    }

    /// <summary>
    /// Gets a string representation of the tree.
    /// </summary>
    public string GetExpression()
    {
        return NodeToString(_rootNode);
    }

    /// <summary>
    /// Converts a node to string representation.
    /// </summary>
    private string NodeToString(NodeGene node)
    {
        if (node.Type == GeneticNodeType.Terminal)
            return node.Value;

        if (node.Children.Count == 1)
            return $"{node.Value}({NodeToString(node.Children[0])})";

        return $"({NodeToString(node.Children[0])} {node.Value} {NodeToString(node.Children[1])})";
    }

    public ICollection<NodeGene> GetGenes()
    {
        // Since tree structure doesn't fit neatly into a flat list,
        // we return a collection containing just the root node
        return [_rootNode];
    }

    public void SetGenes(ICollection<NodeGene> genes)
    {
        // Assume the first gene is the root node
        if (genes != null && genes.Count > 0)
        {
            _rootNode = genes.First();
        }
    }

    public double GetFitness()
    {
        return _fitness;
    }

    public void SetFitness(double fitness)
    {
        _fitness = fitness;
    }

    public IEvolvable<NodeGene, double> Clone()
    {
        var clone = new TreeIndividual(_rootNode.Clone());
        clone._fitness = _fitness;
        return clone;
    }
}
