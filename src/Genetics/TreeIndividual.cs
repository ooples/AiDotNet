namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual in genetic programming with a tree structure.
/// </summary>
/// <remarks>
/// <para>
/// The TreeIndividual class implements a tree-based representation for genetic programming.
/// Each individual represents a complete expression tree that can be evaluated, evolved, and 
/// modified through genetic operations. This representation is particularly useful for problems
/// like symbolic regression, where the goal is to discover mathematical expressions that fit data.
/// </para>
/// <para><b>For Beginners:</b> Think of a TreeIndividual like an actual tree representing a mathematical formula.
/// 
/// Imagine a physical tree:
/// - The trunk splits into large branches (operations like +, -, *, /)
/// - These branches further split into smaller branches (sub-operations)
/// - Eventually, you reach the leaves (variables like "x" or constants like "5")
/// - The entire tree structure represents a complete mathematical formula
/// - For example, (x + 3) * (2 - y) would be a tree with multiplication at the root,
///   addition and subtraction as branches, and x, 3, 2, and y as leaves
/// 
/// During evolution, these trees get pruned, grafted, and modified to find the formula
/// that best solves the problem at hand, such as fitting a curve to a set of data points.
/// </para>
/// </remarks>
public class TreeIndividual : IEvolvable<NodeGene, double>
{
    /// <summary>
    /// The root node of the expression tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the root node of the expression tree, which is the starting point for 
    /// evaluating the expression and traversing the entire tree. All genetic operations like 
    /// mutation and crossover involve manipulating this tree structure.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the main trunk of the tree.
    /// 
    /// Just as a tree grows from its trunk:
    /// - The root node is where the expression tree starts
    /// - All branches and operations stem from this point
    /// - It's the first operation that gets performed when calculating the formula
    /// - For example, in (3 + 4) * 2, the multiplication is the root node
    /// 
    /// When you evaluate or modify the tree, you always start from this root node.
    /// </para>
    /// </remarks>
    private NodeGene _rootNode = default!;

    /// <summary>
    /// The fitness score that indicates how well this individual solves the problem.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the individual's fitness score, which quantifies how well this expression
    /// performs on the given problem. For symbolic regression, this might represent how closely
    /// the expression matches a set of data points.
    /// </para>
    /// <para><b>For Beginners:</b> This is like measuring how well the tree produces the desired fruit.
    /// 
    /// For example:
    /// - If your tree represents a formula trying to match data points
    /// - The fitness measures how close the formula's output is to the expected results
    /// - A lower error (higher fitness) means the formula fits the data better
    /// - Trees with better fitness are more likely to be selected for reproduction
    /// 
    /// The fitness score drives the entire evolutionary process toward finding
    /// better mathematical expressions.
    /// </para>
    /// </remarks>
    private double _fitness;

    /// <summary>
    /// Random number generator for stochastic operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field provides a source of randomness for various operations like tree generation,
    /// mutation, and node selection. Consistent use of the same random generator ensures
    /// reproducible results when using the same seed.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the element of chance in how trees grow and change.
    /// 
    /// In nature:
    /// - Trees grow with some unpredictability in how branches form
    /// - This randomness ensures diversity in the forest
    /// - Similarly, this random generator helps create diverse formulas
    /// - It determines things like "where should a new branch grow?" or "which branch should be pruned?"
    /// 
    /// This controlled randomness is essential for effective exploration of the solution space.
    /// </para>
    /// </remarks>
    private readonly Random _random = default!;

    /// <summary>
    /// The list of available function symbols that can be used in the expression tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the set of function symbols (operators) that can be used in the expression trees.
    /// The default set includes basic arithmetic operations (+, -, *, /), but this can be extended
    /// to include other functions like sine, cosine, logarithm, etc.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the different types of branches your tree can have.
    /// 
    /// Imagine:
    /// - Different types of branches represent different mathematical operations
    /// - Some branches split into exactly two sub-branches (like addition or multiplication)
    /// - Others might transform a single sub-branch (like sine or square root)
    /// - The tree can only use these predefined branch types when growing
    /// 
    /// These functions determine what kinds of mathematical expressions can be formed.
    /// </para>
    /// </remarks>
    private readonly List<string> _availableFunctions = ["+", "-", "*", "/"];

    /// <summary>
    /// The maximum depth allowed for initially generated trees.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field limits how deep the expression trees can be when first created.
    /// Limiting the initial depth helps prevent overly complex expressions at the start
    /// of evolution, allowing the algorithm to gradually discover useful complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This is like limiting how tall a young tree can grow.
    /// 
    /// When planting a new tree:
    /// - You don't want it to immediately grow too tall and complex
    /// - This limit ensures new trees start at a manageable size
    /// - Trees with simpler structures are easier to understand and evaluate
    /// - As evolution progresses, trees may grow more complex if needed
    /// 
    /// This constraint helps the algorithm start with simpler expressions
    /// that can be gradually refined.
    /// </para>
    /// </remarks>
    private readonly int _maxInitialDepth = 6;

    /// <summary>
    /// The maximum depth allowed for any tree during evolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field imposes an absolute limit on how deep the expression trees can grow
    /// throughout the evolutionary process. This constraint prevents bloat (excessive growth 
    /// of expressions without improving fitness) and keeps the expressions manageable.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting an ultimate height limit for your trees.
    /// 
    /// In genetic programming:
    /// - Trees tend to grow larger over time, even when it doesn't improve their performance
    /// - This phenomenon is called "bloat" - like a tree growing unnecessarily tall and wide
    /// - The maximum depth limit prevents trees from becoming too large and unwieldy
    /// - If a tree exceeds this limit, it might be pruned back to a reasonable size
    /// 
    /// This constraint helps maintain efficient, interpretable expressions
    /// rather than allowing them to become needlessly complex.
    /// </para>
    /// </remarks>
    private readonly int _maxDepth = 12;

    /// <summary>
    /// Creates a new tree individual with a random tree.
    /// </summary>
    /// <param name="random">Random number generator for initialization.</param>
    /// <param name="terminals">The terminal symbols available.</param>
    /// <param name="fullMethod">Whether to use the full or grow method.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new TreeIndividual with a randomly generated expression tree.
    /// The tree is built using either the "full" method (where all leaves are at the same depth)
    /// or the "grow" method (which allows leaves at varying depths). The terminals parameter
    /// provides the set of variables and constants that can appear as leaves in the tree.
    /// </para>
    /// <para><b>For Beginners:</b> This is like planting a new tree from seed.
    /// 
    /// When creating a new random formula tree:
    /// - You provide what "leaves" are available (variables or constants like x, y, 1.0, etc.)
    /// - You decide whether to grow a "full" tree (where all branches extend to the same depth)
    ///   or a more natural "grow" tree (where some branches are shorter than others)
    /// - The constructor then builds a complete random tree structure representing a formula
    /// 
    /// This creates a starting point for the evolutionary algorithm to begin exploring
    /// possible mathematical expressions.
    /// </para>
    /// </remarks>
    public TreeIndividual(Random random, List<string> terminals, bool fullMethod = false)
    {
        _random = random;
        _rootNode = GenerateRandomTree(0, terminals, fullMethod);
    }

    /// <summary>
    /// Creates a tree individual with the specified root node.
    /// </summary>
    /// <param name="rootNode">The root node of the tree.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new TreeIndividual with a predefined expression tree
    /// starting at the provided root node. This is used when creating new individuals
    /// from existing trees, such as during crossover or when cloning an individual.
    /// </para>
    /// <para><b>For Beginners:</b> This is like transplanting a fully grown tree.
    /// 
    /// Imagine:
    /// - Instead of growing a tree from seed, you're starting with an existing tree
    /// - You specify exactly what the tree structure looks like
    /// - This is used when creating offspring by combining parts of parent trees
    /// - Or when making an exact copy of a tree that performed well
    /// 
    /// This constructor allows the algorithm to work with predefined expression trees
    /// rather than only randomly generated ones.
    /// </para>
    /// </remarks>
    public TreeIndividual(NodeGene rootNode)
    {
        _rootNode = rootNode;
        _random = new Random();
    }

    /// <summary>
    /// Generates a random subtree.
    /// </summary>
    /// <param name="depth">The current depth in the tree.</param>
    /// <param name="terminals">The list of available terminals.</param>
    /// <param name="fullMethod">Whether to use the full or grow method.</param>
    /// <param name="maxDepth">The maximum depth allowed for this subtree.</param>
    /// <returns>A randomly generated node representing the root of a subtree.</returns>
    /// <remarks>
    /// <para>
    /// This method recursively builds a random expression (sub)tree. It decides whether to create
    /// a terminal node (leaf) or a function node based on the current depth and method chosen.
    /// The "full" method creates trees where all leaves are at the same depth, while the "grow"
    /// method allows leaves at varying depths, creating more diverse trees.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the natural growth process for branches on a tree.
    /// 
    /// As the tree grows:
    /// - At each point, it decides whether to create a leaf or another branching point
    /// - If we're at the maximum allowed height, it always creates a leaf
    /// - In "full" mode, it only creates leaves at the maximum height
    /// - In "grow" mode, it might randomly create leaves at any height
    /// - When creating a branching point, it chooses a random operation (like + or *)
    /// - It then continues the process for each new branch
    /// 
    /// This recursive process builds the entire expression tree from top to bottom.
    /// </para>
    /// </remarks>
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
    /// <param name="function">The function symbol.</param>
    /// <returns>The number of arguments the function takes.</returns>
    /// <remarks>
    /// <para>
    /// This method determines how many arguments (child nodes) a function node should have.
    /// Different operators require different numbers of operands - for example, binary operators
    /// like addition need two operands, while unary operators like sine need only one.
    /// </para>
    /// <para><b>For Beginners:</b> This is like knowing how many branches should grow from each type of branching point.
    /// 
    /// For example:
    /// - Addition (+) needs exactly two branches (the two numbers being added)
    /// - Multiplication (*) also needs exactly two branches
    /// - A function like sine (sin) needs only one branch (the angle)
    /// - This method tells the tree how many sub-branches to create for each operation
    /// 
    /// This ensures that the tree structure properly matches the mathematical operations it represents.
    /// </para>
    /// </remarks>
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
    /// <param name="variables">A dictionary mapping variable names to their values.</param>
    /// <returns>The result of evaluating the expression.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates the expression tree for a specific set of variable values.
    /// It starts at the root node and recursively evaluates the entire tree, substituting
    /// variable values from the provided dictionary and applying operations as defined in the tree.
    /// </para>
    /// <para><b>For Beginners:</b> This is like harvesting the fruit that the tree produces.
    /// 
    /// Imagine:
    /// - Your tree represents a mathematical formula like (x + 2) * y
    /// - You provide specific values for the variables (e.g., x = 3, y = 4)
    /// - This method calculates the result by following the tree from trunk to branches to leaves
    /// - It substitutes the values (x = 3, y = 4) and computes (3 + 2) * 4 = 20
    /// 
    /// This calculation is essential for determining how well the formula performs
    /// for specific inputs, which is needed to assess its fitness.
    /// </para>
    /// </remarks>
    public double Evaluate(Dictionary<string, double> variables)
    {
        return EvaluateNode(_rootNode, variables);
    }

    /// <summary>
    /// Recursively evaluates a node in the tree.
    /// </summary>
    /// <param name="node">The node to evaluate.</param>
    /// <param name="variables">A dictionary mapping variable names to their values.</param>
    /// <returns>The result of evaluating the node.</returns>
    /// <remarks>
    /// <para>
    /// This method recursively evaluates a node and its subtree. For terminal nodes, it either
    /// looks up variable values or parses constants. For function nodes, it first evaluates
    /// all child nodes and then applies the function to the results.
    /// </para>
    /// <para><b>For Beginners:</b> This is like calculating the output of each part of the tree.
    /// 
    /// The process works like this:
    /// - If the node is a leaf (terminal), it either:
    ///   * Returns the value of a variable (like x = 3)
    ///   * Returns a constant value (like 5.0)
    /// - If the node is a branch (function), it:
    ///   * First calculates the value of each sub-branch
    ///   * Then applies the operation (like addition or multiplication)
    ///   * Returns the result of that operation
    /// 
    /// This recursive process ensures that operations are performed in the correct order,
    /// following the structure of the expression tree.
    /// </para>
    /// </remarks>
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
    /// <returns>A randomly selected node from anywhere in the tree.</returns>
    /// <remarks>
    /// <para>
    /// This method selects a random node from anywhere in the expression tree.
    /// It first collects all nodes in the tree into a flat list, then randomly selects one.
    /// This is used for operations like mutation and crossover that need to target specific nodes.
    /// </para>
    /// <para><b>For Beginners:</b> This is like randomly choosing a branch or leaf from the tree.
    /// 
    /// Imagine:
    /// - You want to modify the tree by changing or replacing a part of it
    /// - First, you need to select which part to modify
    /// - This method randomly picks any branch or leaf in the entire tree
    /// - Each branch or leaf has an equal chance of being selected
    /// 
    /// This random selection is important for genetic operations like mutation,
    /// where different parts of the expression might be modified.
    /// </para>
    /// </remarks>
    public NodeGene SelectRandomNode()
    {
        List<NodeGene> allNodes = new List<NodeGene>();
        CollectNodes(_rootNode, allNodes);
        return allNodes[_random.Next(allNodes.Count)];
    }

    /// <summary>
    /// Collects all nodes in the tree.
    /// </summary>
    /// <param name="node">The current node being processed.</param>
    /// <param name="nodes">The list to collect nodes into.</param>
    /// <remarks>
    /// <para>
    /// This method recursively traverses the tree and collects all nodes into a flat list.
    /// It performs a pre-order traversal, first adding the current node and then recursively
    /// processing all of its children.
    /// </para>
    /// <para><b>For Beginners:</b> This is like making an inventory of every branch and leaf on the tree.
    /// 
    /// The process works like this:
    /// - Start at a given branch (or the trunk if starting from the beginning)
    /// - Add that branch to your inventory list
    /// - For each sub-branch connected to this branch, repeat the process
    /// - Continue until you've listed every branch and leaf on the tree
    /// 
    /// This complete inventory allows operations like "select a random node" to work properly
    /// by considering all parts of the tree.
    /// </para>
    /// </remarks>
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
    /// <param name="target">The node to be replaced.</param>
    /// <param name="replacement">The node to replace it with.</param>
    /// <remarks>
    /// <para>
    /// This method replaces a specific node (and its entire subtree) with a different node.
    /// If the target is the root node, the entire tree is replaced. Otherwise, it searches
    /// for the target within the tree and replaces it when found.
    /// </para>
    /// <para><b>For Beginners:</b> This is like grafting a new branch onto a tree.
    /// 
    /// Imagine:
    /// - You identify a specific branch you want to replace
    /// - You have a new branch ready to put in its place
    /// - This method cuts off the old branch and attaches the new one at the same spot
    /// - If you're replacing the trunk, the entire tree is replaced
    /// 
    /// This operation is fundamental for genetic programming crossover,
    /// where subtrees from different individuals are exchanged.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method checks if the tree exceeds the maximum allowed depth and, if so,
    /// replaces it with a simpler tree. This helps prevent bloat and keeps expressions
    /// from becoming overly complex during evolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like pruning a tree that has grown too tall.
    /// 
    /// Imagine:
    /// - You have a height limit for trees in your garden
    /// - You measure a tree and find it has grown beyond this limit
    /// - Rather than just cutting the top off, you replace it with a younger, smaller tree
    /// - This ensures the tree stays manageable and doesn't waste resources on excessive growth
    /// 
    /// This constraint helps control the complexity of evolved expressions,
    /// making them more interpretable and computationally efficient.
    /// </para>
    /// </remarks>
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
    /// <param name="current">The current node being examined.</param>
    /// <param name="target">The node to find and replace.</param>
    /// <param name="replacement">The node to replace it with.</param>
    /// <returns>True if the replacement was made, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method recursively searches through the tree to find a specific target node.
    /// When found, it replaces the node with the provided replacement. The method returns
    /// true if the replacement was successful, or false if the target node was not found
    /// in the current subtree.
    /// </para>
    /// <para><b>For Beginners:</b> This is like searching through the tree to find a specific branch to replace.
    /// 
    /// The process works like this:
    /// - Start at a particular branch of the tree
    /// - Check if any of its immediate sub-branches match the one you're looking for
    /// - If a match is found, replace it with the new branch and return success
    /// - If not, check each of the sub-branches using the same process
    /// - If the branch isn't found anywhere, return failure
    /// 
    /// This search-and-replace operation is necessary when modifying specific
    /// parts of the tree structure.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method implements point mutation, which changes the value of a randomly selected node
    /// without altering the tree structure. For terminal nodes, it changes the value to a new constant.
    /// For function nodes, it changes the operation to another compatible operation with the same arity.
    /// </para>
    /// <para><b>For Beginners:</b> This is like changing the type of a branch or the value of a leaf without altering the structure.
    /// 
    /// Imagine:
    /// - You randomly select a branch or leaf on the tree
    /// - If it's a leaf with a number (like 2.0), you change it to a different number (like 3.5)
    /// - If it's a branch with an operation (like +), you change it to a similar operation (like -)
    /// - The overall structure of the tree remains the same
    /// 
    /// This creates a small variation that might improve the formula's performance
    /// without drastically changing its structure.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method implements subtree mutation, which replaces a randomly selected subtree
    /// with a new, randomly generated subtree. This can introduce significant changes to the
    /// expression while maintaining valid structure. The depth of the new subtree is limited
    /// to ensure the overall tree doesn't exceed the maximum allowed depth.
    /// </para>
    /// <para><b>For Beginners:</b> This is like cutting off a branch and growing a completely new one in its place.
    /// 
    /// Imagine:
    /// - You randomly select a branch on your tree
    /// - You cut off that entire branch, including all its sub-branches
    /// - You grow a completely new, random branch in its place
    /// - The new branch might represent a very different sub-expression
    /// - You make sure the new branch doesn't make the tree too tall
    /// 
    /// This type of mutation can introduce major changes to the formula,
    /// potentially discovering new approaches to solving the problem.
    /// </para>
    /// </remarks>
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
    /// <param name="target">The node to find the depth of.</param>
    /// <returns>The depth of the target node in the tree.</returns>
    /// <remarks>
    /// <para>
    /// This method determines how deep a specific node is within the tree, measured as the
    /// number of edges from the root to the node. It uses the FindNodeDepth method to recursively
    /// search for the node and track its depth.
    /// </para>
    /// <para><b>For Beginners:</b> This is like measuring how far a branch is from the trunk.
    /// 
    /// Imagine:
    /// - You want to know how many branching points there are between the trunk and a specific branch
    /// - This method counts the number of connections you have to follow to reach that branch
    /// - For example, the trunk is at depth 0, its immediate branches are at depth 1, and so on
    /// - This information helps ensure that a tree doesn't grow too tall when modifying it
    /// 
    /// Knowing a node's depth is crucial for operations like subtree mutation,
    /// where you need to ensure the overall tree doesn't exceed the maximum depth.
    /// </para>
    /// </remarks>
    private int GetNodeDepthInTree(NodeGene target)
    {
        return FindNodeDepth(_rootNode, target, 0);
    }

    /// <summary>
    /// Recursively finds the depth of a target node in the tree.
    /// </summary>
    /// <param name="current">The current node being examined.</param>
    /// <param name="target">The node to find the depth of.</param>
    /// <param name="depth">The current depth in the traversal.</param>
    /// <returns>The depth of the target node, or -1 if not found in this branch.</returns>
    /// <remarks>
    /// <para>
    /// This method recursively searches for a target node in the tree, tracking its depth
    /// as it goes. If the target is found, its depth is returned. If the target is not found
    /// in the current branch, -1 is returned to indicate failure.
    /// </para>
    /// <para><b>For Beginners:</b> This is like following the branches to find a specific one and counting the steps.
    /// 
    /// The process works like this:
    /// - Start at a particular branch (the current node)
    /// - Check if it's the branch you're looking for
    /// - If it is, return how many steps you've taken to get here
    /// - If not, check each sub-branch using the same process
    /// - If you can't find the branch anywhere in this part of the tree, return -1 (not found)
    /// 
    /// This recursive search helps determine exactly how deep a specific branch is in the tree.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method implements permutation mutation, which randomly reorders the child nodes
    /// of a selected function node. This is only applicable to commutative operations like
    /// addition and multiplication, where the order of operands doesn't affect the result.
    /// </para>
    /// <para><b>For Beginners:</b> This is like rearranging the order of branches without changing their content.
    /// 
    /// Imagine:
    /// - You randomly select a branch in the tree
    /// - If it's a branching point with an operation like addition or multiplication
    /// - You shuffle the order of its sub-branches
    /// - For operations like addition, this doesn't change the actual result (a + b = b + a)
    /// - But it might change how the expression evolves in future generations
    /// 
    /// This type of mutation creates subtle variations without changing the actual behavior of the formula.
    /// </para>
    /// </remarks>
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
    /// <returns>The maximum depth of the tree, which is the length of the longest path from root to leaf.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the maximum depth of the tree, which is the length of the longest
    /// path from the root node to any leaf node. It uses the GetNodeDepth method to recursively
    /// determine the depth of the tree.
    /// </para>
    /// <para><b>For Beginners:</b> This is like measuring how tall the tree is.
    /// 
    /// Imagine:
    /// - You want to know the maximum number of branches you need to traverse to reach any leaf
    /// - Starting from the trunk, this method finds the longest possible path to any leaf
    /// - For example, a tree with just a trunk and leaves would have depth 1
    /// - A complex tree with many nested branches might have depth 10 or more
    /// 
    /// Knowing the tree's depth is important for controlling complexity and ensuring
    /// the tree doesn't grow beyond manageable limits.
    /// </para>
    /// </remarks>
    public int GetDepth()
    {
        return GetNodeDepth(_rootNode);
    }

    /// <summary>
    /// Gets the depth of a node.
    /// </summary>
    /// <param name="node">The node to measure the depth of.</param>
    /// <returns>The maximum depth of the subtree rooted at this node.</returns>
    /// <remarks>
    /// <para>
    /// This method recursively calculates the maximum depth of the subtree rooted at a given node.
    /// For leaf nodes, the depth is 0. For function nodes, the depth is 1 plus the maximum depth
    /// of any of its children.
    /// </para>
    /// <para><b>For Beginners:</b> This is like measuring how many levels of branching exist below a specific branch.
    /// 
    /// The process works like this:
    /// - If the branch has no sub-branches (it's a leaf), its depth is 0
    /// - Otherwise, find the depth of each sub-branch
    /// - Add 1 to the maximum of these depths
    /// - This gives how many levels of branching exist below this point
    /// 
    /// This recursive measurement helps track the complexity of different parts of the tree.
    /// </para>
    /// </remarks>
    private int GetNodeDepth(NodeGene node)
    {
        if (node.Children.Count == 0)
            return 0;

        return 1 + node.Children.Max(GetNodeDepth);
    }

    /// <summary>
    /// Gets a string representation of the tree.
    /// </summary>
    /// <returns>A string representing the expression tree in a human-readable format.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the expression tree to a string representation that can be read
    /// and understood by humans. It uses conventional mathematical notation with parentheses
    /// to indicate precedence.
    /// </para>
    /// <para><b>For Beginners:</b> This is like writing down the mathematical formula that the tree represents.
    /// 
    /// For example:
    /// - A tree representing addition of x and 2 becomes "x + 2"
    /// - A tree with nested operations becomes "(x + 2) * 3"
    /// - Function applications like sine become "sin(x)"
    /// 
    /// This textual representation makes it easier to understand what formula
    /// the tree actually represents, which is useful for analysis and debugging.
    /// </para>
    /// </remarks>
    public string GetExpression()
    {
        return NodeToString(_rootNode);
    }

    /// <summary>
    /// Converts a node to string representation.
    /// </summary>
    /// <param name="node">The node to convert to a string.</param>
    /// <returns>A string representation of the node and its subtree.</returns>
    /// <remarks>
    /// <para>
    /// This method recursively converts a node and its subtree to a string representation.
    /// Terminal nodes are converted directly to their value. For function nodes, the method
    /// recursively converts their children and combines them with the function symbol according
    /// to standard mathematical notation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like translating a branch and all its sub-branches into mathematical notation.
    /// 
    /// The process works like this:
    /// - If it's a leaf (terminal), just write its value (like "x" or "5.0")
    /// - If it's a unary function (like sine), write the function name followed by its argument in parentheses, like "sin(x)"
    /// - If it's a binary operation (like addition), write the two arguments with the operation between them, like "(3 + x)"
    /// - Use parentheses to show the correct order of operations
    /// 
    /// This recursive translation ensures that the entire expression tree is correctly
    /// represented as a conventional mathematical formula.
    /// </para>
    /// </remarks>
    private string NodeToString(NodeGene node)
    {
        if (node.Type == GeneticNodeType.Terminal)
            return node.Value;

        if (node.Children.Count == 1)
            return $"{node.Value}({NodeToString(node.Children[0])})";

        return $"({NodeToString(node.Children[0])} {node.Value} {NodeToString(node.Children[1])})";
    }

    /// <summary>
    /// Gets the genes of this individual.
    /// </summary>
    /// <returns>A collection containing the root node of the tree.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a collection containing just the root node of the tree.
    /// Since a tree structure doesn't fit neatly into a flat list of genes like other
    /// genetic representations, this implementation uses the root node as a proxy for
    /// the entire tree. The tree structure is preserved through the node's children.
    /// </para>
    /// <para><b>For Beginners:</b> This is like identifying the trunk of the tree as a representative of the whole tree.
    /// 
    /// Since a tree is a connected structure:
    /// - The root node (trunk) connects to all other nodes
    /// - By providing the root, you're implicitly providing the whole tree
    /// - This simplifies working with tree structures in the genetic algorithm framework
    /// - It allows tree individuals to be compatible with the IEvolvable interface
    /// 
    /// This approach treats the entire tree as a single, complex "gene"
    /// rather than trying to flatten it into separate genes.
    /// </para>
    /// </remarks>
    public ICollection<NodeGene> GetGenes()
    {
        // Since tree structure doesn't fit neatly into a flat list,
        // we return a collection containing just the root node
        return [_rootNode];
    }

    /// <summary>
    /// Sets the genes of this individual.
    /// </summary>
    /// <param name="genes">A collection containing the new root node.</param>
    /// <remarks>
    /// <para>
    /// This method sets the root node of the tree from the provided collection of genes.
    /// It assumes that the first gene in the collection is the root node, which contains
    /// references to all other nodes in the tree through its children.
    /// </para>
    /// <para><b>For Beginners:</b> This is like replacing the entire tree by attaching a new trunk.
    /// 
    /// When setting genes:
    /// - The method expects a collection that contains a root node
    /// - It takes the first element from this collection as the new root
    /// - Since the root connects to all other nodes, this effectively replaces the entire tree
    /// - This allows the tree to be modified through the standard genetic interface
    /// 
    /// This approach aligns with the GetGenes method, treating the entire tree
    /// as a single complex "gene" for compatibility with the genetic framework.
    /// </para>
    /// </remarks>
    public void SetGenes(ICollection<NodeGene> genes)
    {
        // Assume the first gene is the root node
        if (genes != null && genes.Count > 0)
        {
            _rootNode = genes.First();
        }
    }

    /// <summary>
    /// Gets the fitness of this individual.
    /// </summary>
    /// <returns>The fitness score as a double value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the individual's fitness score as required by the IEvolvable interface.
    /// For genetic programming, this typically represents how well the expression fits the data
    /// or solves the problem at hand.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking how good the formula is at solving the problem.
    /// 
    /// The fitness score:
    /// - Measures how well this formula performs for its intended purpose
    /// - For fitting data points, it might measure how close the predictions are to actual values
    /// - Better-performing formulas have higher fitness scores
    /// - This score determines which formulas are more likely to be selected for reproduction
    /// 
    /// This is one of the core methods required by the genetic algorithm to evaluate
    /// and compare different solutions.
    /// </para>
    /// </remarks>
    public double GetFitness()
    {
        return _fitness;
    }

    /// <summary>
    /// Sets the fitness of this individual.
    /// </summary>
    /// <param name="fitness">The fitness score to set.</param>
    /// <remarks>
    /// <para>
    /// This method sets the individual's fitness score as required by the IEvolvable interface.
    /// It is typically called after evaluating the expression on the problem data to assess
    /// its performance.
    /// </para>
    /// <para><b>For Beginners:</b> This is like recording a score for how well the formula performs.
    /// 
    /// After testing the formula:
    /// - You calculate a score based on how well it solves the problem
    /// - This method stores that score with the formula
    /// - Better formulas get higher scores
    /// - These scores are used to select which formulas contribute to the next generation
    /// 
    /// This is one of the core methods required by the genetic algorithm to track
    /// the quality of different solutions during evolution.
    /// </para>
    /// </remarks>
    public void SetFitness(double fitness)
    {
        _fitness = fitness;
    }

    /// <summary>
    /// Creates a deep copy of this individual.
    /// </summary>
    /// <returns>A new TreeIndividual that is a copy of this one.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a completely independent copy of the individual, including its entire
    /// expression tree. It uses the NodeGene.Clone method to recursively clone the tree structure.
    /// This ensures that modifications to the clone don't affect the original.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating an exact duplicate of the entire tree.
    /// 
    /// When cloning:
    /// - A completely new tree is created
    /// - Every branch and leaf is copied exactly to the new tree
    /// - The new tree produces exactly the same formula as the original
    /// - Changes to one tree won't affect the other
    /// 
    /// This is essential for genetic operations where you need to preserve
    /// existing individuals while creating modified versions.
    /// </para>
    /// </remarks>
    public IEvolvable<NodeGene, double> Clone()
    {
        return new TreeIndividual(_rootNode.Clone());
    }
}