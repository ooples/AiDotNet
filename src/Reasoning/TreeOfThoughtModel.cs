using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AiDotNet.Reasoning;

/// <summary>
/// Implements a Tree-of-Thought (ToT) reasoning model that systematically explores
/// reasoning paths in a tree structure.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations</typeparam>
/// <remarks>
/// <para>
/// Tree-of-Thought extends chain-of-thought by maintaining a tree of possible reasoning
/// paths instead of a single chain. At each step, the model considers multiple possible
/// next steps, evaluates them, and may pursue several promising directions in parallel.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine solving a puzzle where at each step you can try different
/// moves. Instead of just picking one and hoping for the best, Tree-of-Thought explores
/// multiple possibilities, keeps track of what works, and can backtrack if needed. It's
/// like having a map of all possible solution paths.
/// </para>
/// </remarks>
public class TreeOfThoughtModel<T> : ReasoningModelBase<T>
{
    private readonly NeuralNetwork<T> _thoughtGenerator = default!;
    private readonly NeuralNetwork<T> _stateEvaluator = default!;
    private readonly NeuralNetwork<T> _thoughtSelector = default!;
    private readonly TreeOfThoughtOptions<T> _totOptions = default!;
    private readonly Dictionary<string, TreeNode<T>> _nodeCache = default!;

    /// <summary>
    /// Represents a node in the reasoning tree.
    /// </summary>
    private class TreeNode<TNode>
    {
        public Tensor<TNode> State { get; set; }
        public TreeNode<TNode>? Parent { get; set; }
        public List<TreeNode<TNode>> Children { get; set; }
        public TNode Value { get; set; } = default!;  // Evaluation score
        public int Depth { get; set; }
        public bool IsExpanded { get; set; }
        public bool IsTerminal { get; set; }

        public TreeNode(Tensor<TNode> state, TreeNode<TNode>? parent = null)
        {
            State = state;
            Parent = parent;
            Children = new List<TreeNode<TNode>>();
            Depth = parent?.Depth + 1 ?? 0;
            IsExpanded = false;
            IsTerminal = false;
        }
    }

    /// <summary>
    /// Gets the maximum reasoning depth this model can handle effectively.
    /// </summary>
    public override int MaxReasoningDepth => _totOptions.MaxTreeDepth;

    /// <summary>
    /// Gets whether this model supports iterative refinement.
    /// </summary>
    public override bool SupportsIterativeRefinement => true;

    /// <summary>
    /// Initializes a new instance of the TreeOfThoughtModel class.
    /// </summary>
    /// <param name="options">Configuration options for the model</param>
    public TreeOfThoughtModel(TreeOfThoughtOptions<T> options)
        : base(options)
    {
        _totOptions = options;
        _nodeCache = new Dictionary<string, TreeNode<T>>();

        // Build the thought generator network
        // Using a simple architecture due to API constraints
        var generatorArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);
        _thoughtGenerator = new NeuralNetwork<T>(generatorArchitecture);

        // Build the state evaluator network
        var evaluatorArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Simple);
        _stateEvaluator = new NeuralNetwork<T>(evaluatorArchitecture);

        // Build the thought selector network
        var selectorArchitecture = new NeuralNetworkArchitecture<T>(NetworkComplexity.Simple);
        _thoughtSelector = new NeuralNetwork<T>(selectorArchitecture);
    }

    /// <summary>
    /// Trains the Tree-of-Thought model.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Build a training tree
        var root = new TreeNode<T>(input);
        var trainingTree = BuildTrainingTree(root, expectedOutput);

        // Train thought generator on successful paths
        var successPaths = ExtractSuccessfulPaths(trainingTree, expectedOutput);
        foreach (var path in successPaths)
        {
            for (int i = 0; i < path.Count - 1; i++)
            {
                var currentState = path[i].State;
                var childStates = CombineChildStates(path[i + 1]);
                _thoughtGenerator.Train(currentState, childStates);
            }
        }

        // Train state evaluator
        TrainStateEvaluator(trainingTree, expectedOutput);

        // Train thought selector
        TrainThoughtSelector(trainingTree);
    }

    /// <summary>
    /// Makes a prediction using tree-of-thought reasoning.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var startTime = DateTime.UtcNow;

        // Build reasoning tree
        var root = new TreeNode<T>(input);
        var searchResult = TreeSearch(root);

        // Extract the best path
        var bestPath = ExtractBestPath(searchResult);
        LastReasoningSteps = bestPath.Select(n => n.State).ToList();

        // Store diagnostics
        LastDiagnostics["TreeSize"] = CountTreeNodes(root);
        LastDiagnostics["SearchTime"] = (DateTime.UtcNow - startTime).TotalMilliseconds;
        LastDiagnostics["BestPathDepth"] = bestPath.Count;
        LastDiagnostics["SearchStrategy"] = _totOptions.SearchStrategy.ToString();

        return bestPath.Last().State;
    }

    /// <summary>
    /// Performs multi-step reasoning using tree exploration.
    /// </summary>
    public override List<Tensor<T>> ReasonStepByStep(Tensor<T> input, int maxSteps = 10)
    {
        var root = new TreeNode<T>(input);
        var steps = new List<Tensor<T>> { input };

        // Perform limited tree search
        var current = root;
        for (int i = 0; i < maxSteps && !current.IsTerminal; i++)
        {
            // Generate and evaluate children
            var children = GenerateChildren(current);
            
            if (children.Count == 0)
                break;

            // Select best child based on strategy
            var bestChild = SelectBestChild(children);
            steps.Add(bestChild.State);
            current = bestChild;
        }

        return steps;
    }

    /// <summary>
    /// Generates an explanation by showing the explored tree structure.
    /// </summary>
    public override Tensor<T> GenerateExplanation(Tensor<T> input, Tensor<T> prediction)
    {
        // Build a partial tree to show exploration
        var root = new TreeNode<T>(input);
        var explorationTree = BuildExplorationTree(root, prediction, _totOptions.ExplanationDepth);

        // Convert tree structure to explanation tensor
        return ConvertTreeToExplanation(explorationTree);
    }

    /// <summary>
    /// Validates the logical consistency of a reasoning chain.
    /// </summary>
    public override bool ValidateReasoningChain(List<Tensor<T>> reasoningSteps)
    {
        if (reasoningSteps.Count < 2)
            return true;

        // Check if each transition is valid in the tree structure
        for (int i = 0; i < reasoningSteps.Count - 1; i++)
        {
            var isValidTransition = EvaluateTransition(reasoningSteps[i], reasoningSteps[i + 1]);
            if (!isValidTransition)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Calculates confidence scores based on tree exploration.
    /// </summary>
    protected override Vector<T> CalculateConfidenceScores(List<Tensor<T>> reasoningSteps)
    {
        var scores = new T[reasoningSteps.Count];

        for (int i = 0; i < reasoningSteps.Count; i++)
        {
            // Evaluate state quality
            var evaluation = _stateEvaluator.Predict(reasoningSteps[i]);
            scores[i] = evaluation[0];
        }

        return new Vector<T>(scores);
    }

    /// <summary>
    /// Performs refinement by exploring alternative branches.
    /// </summary>
    protected override Tensor<T> PerformRefinementStep(Tensor<T> input, Tensor<T> currentReasoning, int iteration)
    {
        // Build a tree from current reasoning
        var root = new TreeNode<T>(currentReasoning);
        
        // Explore alternative branches
        var alternatives = ExploreAlternatives(root, _totOptions.RefinementBranches);

        // Select best alternative
        return SelectBestAlternative(alternatives);
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    protected override ModelType GetModelType()
    {
        return ModelType.TreeOfThoughtModel;
    }

    /// <summary>
    /// Gets a description of the model.
    /// </summary>
    protected override string GetModelDescription()
    {
        return $"Tree-of-Thought reasoning model with branching factor {_totOptions.BranchingFactor} and max depth {_totOptions.MaxTreeDepth}";
    }

    /// <summary>
    /// Estimates the complexity of the model.
    /// </summary>
    protected override double EstimateComplexity()
    {
        var genParams = _thoughtGenerator.GetParameters().Length;
        var evalParams = _stateEvaluator.GetParameters().Length;
        var selParams = _thoughtSelector.GetParameters().Length;
        
        // Account for tree exploration complexity
        var treeComplexity = Math.Pow(_totOptions.BranchingFactor, _totOptions.MaxTreeDepth);
        
        return (genParams + evalParams + selParams) * Math.Log(treeComplexity);
    }

    #region Private Helper Methods

    private TreeNode<T> TreeSearch(TreeNode<T> root)
    {
        switch (_totOptions.SearchStrategy)
        {
            case TreeSearchStrategy.BeamSearch:
                return BeamSearch(root);
            case TreeSearchStrategy.BreadthFirst:
                return BreadthFirstSearch(root);
            case TreeSearchStrategy.DepthFirst:
                return DepthFirstSearch(root);
            case TreeSearchStrategy.MonteCarlo:
                return MonteCarloTreeSearch(root);
            case TreeSearchStrategy.AStar:
                return AStarSearch(root);
            default:
                return BeamSearch(root);
        }
    }

    private TreeNode<T> BeamSearch(TreeNode<T> root)
    {
        var beam = new List<TreeNode<T>> { root };
        var bestNode = root;
        var bestValue = EvaluateNode(root);

        for (int depth = 0; depth < _totOptions.MaxTreeDepth && beam.Count > 0; depth++)
        {
            var nextBeam = new List<TreeNode<T>>();

            foreach (var node in beam)
            {
                if (node.IsTerminal)
                    continue;

                var children = GenerateChildren(node);
                
                foreach (var child in children)
                {
                    var value = EvaluateNode(child);
                    
                    if (NumOps.GreaterThan(value, bestValue))
                    {
                        bestValue = value;
                        bestNode = child;
                    }

                    nextBeam.Add(child);
                }
            }

            // Keep only top k nodes
            beam = nextBeam
                .OrderByDescending(n => n.Value)
                .Take(_totOptions.BeamWidth)
                .ToList();

            // Check for early termination
            if (beam.Any(n => n.IsTerminal))
            {
                var terminal = beam.First(n => n.IsTerminal);
                if (NumOps.GreaterThan(terminal.Value, NumOps.FromDouble(_totOptions.TerminalValueThreshold)))
                {
                    return terminal;
                }
            }
        }

        return bestNode;
    }

    private TreeNode<T> BreadthFirstSearch(TreeNode<T> root)
    {
        var queue = new Queue<TreeNode<T>>();
        queue.Enqueue(root);
        
        var bestNode = root;
        var bestValue = EvaluateNode(root);

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            
            if (current.Depth >= _totOptions.MaxTreeDepth || current.IsTerminal)
                continue;

            var children = GenerateChildren(current);
            
            foreach (var child in children)
            {
                var value = EvaluateNode(child);
                
                if (NumOps.GreaterThan(value, bestValue))
                {
                    bestValue = value;
                    bestNode = child;
                }

                queue.Enqueue(child);
            }
        }

        return bestNode;
    }

    private TreeNode<T> DepthFirstSearch(TreeNode<T> root)
    {
        var bestNode = root;
        var bestValue = EvaluateNode(root);

        void DFS(TreeNode<T> node)
        {
            if (node.Depth >= _totOptions.MaxTreeDepth || node.IsTerminal)
                return;

            var children = GenerateChildren(node);
            
            foreach (var child in children)
            {
                var value = EvaluateNode(child);
                
                if (NumOps.GreaterThan(value, bestValue))
                {
                    bestValue = value;
                    bestNode = child;
                }

                DFS(child);
            }
        }

        DFS(root);
        return bestNode;
    }

    private TreeNode<T> MonteCarloTreeSearch(TreeNode<T> root)
    {
        var iterations = _totOptions.MonteCarloIterations;
        
        for (int i = 0; i < iterations; i++)
        {
            // Selection
            var leaf = SelectLeaf(root);
            
            // Expansion
            if (!leaf.IsTerminal && leaf.Depth < _totOptions.MaxTreeDepth)
            {
                var children = GenerateChildren(leaf);
                if (children.Count > 0)
                {
                    leaf = children[Random.Next(children.Count)];
                }
            }

            // Simulation
            var value = Simulate(leaf);

            // Backpropagation
            Backpropagate(leaf, value);
        }

        // Return best child of root
        return root.Children.Count > 0 
            ? root.Children.OrderByDescending(c => c.Value).First()
            : root;
    }

    private TreeNode<T> AStarSearch(TreeNode<T> root)
    {
        var openSet = new SortedSet<TreeNode<T>>(new NodeComparer(NumOps));
        var closedSet = new HashSet<TreeNode<T>>();
        
        openSet.Add(root);

        while (openSet.Count > 0)
        {
            var current = openSet.Min!;
            openSet.Remove(current);

            if (current.IsTerminal || current.Depth >= _totOptions.MaxTreeDepth)
                return current;

            closedSet.Add(current);

            var children = GenerateChildren(current);
            
            foreach (var child in children)
            {
                if (closedSet.Contains(child))
                    continue;

                var g = current.Depth + 1;  // Cost from start
                var h = EstimateRemainingCost(child);  // Heuristic
                child.Value = NumOps.Add(NumOps.FromDouble(g), h);

                openSet.Add(child);
            }
        }

        return root;
    }

    private class NodeComparer : IComparer<TreeNode<T>>
    {
        private readonly INumericOperations<T> _numOps = default!;
        
        public NodeComparer(INumericOperations<T> numOps)
        {
            _numOps = numOps;
        }
        
        public int Compare(TreeNode<T>? x, TreeNode<T>? y)
        {
            if (x == null || y == null) return 0;
            // Compare values using numeric operations
            var diff = _numOps.Subtract(x.Value, y.Value);
            if (_numOps.GreaterThan(diff, _numOps.Zero)) return 1;
            if (_numOps.LessThan(diff, _numOps.Zero)) return -1;
            return 0;
        }
    }

    private List<TreeNode<T>> GenerateChildren(TreeNode<T> parent)
    {
        if (parent.IsExpanded)
            return parent.Children;

        var children = new List<TreeNode<T>>();
        
        // Generate candidate thoughts
        var candidates = _thoughtGenerator.Predict(parent.State);
        var candidateStates = SplitCandidates(candidates, _totOptions.BranchingFactor);

        foreach (var state in candidateStates)
        {
            // Check if this thought should be selected
            var combined = CombineStates(parent.State, state);
            var selection = _thoughtSelector.Predict(combined);
            
            if (NumOps.GreaterThan(selection[0], NumOps.FromDouble(_totOptions.SelectionThreshold)))
            {
                var child = new TreeNode<T>(state, parent);
                child.IsTerminal = IsTerminalState(state);
                children.Add(child);
            }
        }

        parent.Children = children;
        parent.IsExpanded = true;

        return children;
    }

    private List<Tensor<T>> SplitCandidates(Tensor<T> candidates, int count)
    {
        var stateSize = _totOptions.StateShape[0];
        var results = new List<Tensor<T>>();

        for (int i = 0; i < count; i++)
        {
            var state = new Tensor<T>(_totOptions.StateShape);
            
            for (int j = 0; j < stateSize; j++)
            {
                var index = i * stateSize + j;
                if (index < candidates.Length)
                {
                    state[j] = candidates[index];
                }
            }

            results.Add(state);
        }

        return results;
    }

    private Tensor<T> CombineStates(Tensor<T> parent, Tensor<T> child)
    {
        var combined = new Tensor<T>(new[] { parent.Length + child.Length });
        
        for (int i = 0; i < parent.Length; i++)
        {
            combined[i] = parent[i];
        }
        
        for (int i = 0; i < child.Length; i++)
        {
            combined[parent.Length + i] = child[i];
        }

        return combined;
    }

    private T EvaluateNode(TreeNode<T> node)
    {
        var evaluation = _stateEvaluator.Predict(node.State);
        node.Value = evaluation[0];
        return node.Value;
    }

    private TreeNode<T> SelectBestChild(List<TreeNode<T>> children)
    {
        return children.OrderByDescending(c => EvaluateNode(c)).First();
    }

    private bool IsTerminalState(Tensor<T> state)
    {
        var evaluation = _stateEvaluator.Predict(state);
        var score = evaluation[0];
        return NumOps.GreaterThan(score, NumOps.FromDouble(_totOptions.TerminalValueThreshold));
    }

    private TreeNode<T> SelectLeaf(TreeNode<T> root)
    {
        var current = root;
        
        while (current.Children.Count > 0)
        {
            // UCB1 selection
            current = current.Children.OrderByDescending(c => 
            {
                var exploitation = c.Value;
                var exploration = NumOps.Sqrt(
                    NumOps.Divide(
                        NumOps.FromDouble(2.0 * Math.Log(current.Children.Count)),
                        NumOps.FromDouble(c.Children.Count + 1)
                    )
                );
                return NumOps.Add(exploitation, exploration);
            }).First();
        }

        return current;
    }

    private T Simulate(TreeNode<T> node)
    {
        var stateData = new T[node.State.Length];
        for (int i = 0; i < node.State.Length; i++)
        {
            stateData[i] = node.State[i];
        }
        var current = new Tensor<T>(node.State.Shape);
        for (int i = 0; i < node.State.Length; i++)
        {
            current[i] = stateData[i];
        }
        var depth = node.Depth;

        while (depth < _totOptions.MaxTreeDepth && !IsTerminalState(current))
        {
            // Random rollout
            var candidates = _thoughtGenerator.Predict(current);
            var states = SplitCandidates(candidates, _totOptions.BranchingFactor);
            current = states[Random.Next(states.Count)];
            depth++;
        }

        var finalEval = _stateEvaluator.Predict(current);
        return finalEval[0];
    }

    private void Backpropagate(TreeNode<T> node, T value)
    {
        var current = node;
        
        while (current != null)
        {
            // Update node value with running average
            var alpha = NumOps.FromDouble(0.1);
            var oneMinusAlpha = NumOps.Subtract(NumOps.One, alpha);
            
            current.Value = NumOps.Add(
                NumOps.Multiply(current.Value, oneMinusAlpha),
                NumOps.Multiply(value, alpha)
            );

            current = current.Parent;
        }
    }

    private T EstimateRemainingCost(TreeNode<T> node)
    {
        // Heuristic: inverse of current value + depth penalty
        var depthPenalty = NumOps.FromDouble(node.Depth * 0.1);
        var valueComponent = NumOps.Subtract(NumOps.One, node.Value);
        return NumOps.Add(valueComponent, depthPenalty);
    }

    private List<TreeNode<T>> ExtractBestPath(TreeNode<T> endNode)
    {
        var path = new List<TreeNode<T>>();
        var current = endNode;

        while (current != null)
        {
            path.Insert(0, current);
            current = current.Parent;
        }

        return path;
    }

    private int CountTreeNodes(TreeNode<T> root)
    {
        var count = 1;
        foreach (var child in root.Children)
        {
            count += CountTreeNodes(child);
        }
        return count;
    }

    private TreeNode<T> BuildTrainingTree(TreeNode<T> root, Tensor<T> target)
    {
        // Build a tree that leads to the target
        var queue = new Queue<TreeNode<T>>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            
            if (current.Depth >= _totOptions.MaxTreeDepth)
                continue;

            // Generate children biased toward target
            var children = GenerateTrainingChildren(current, target);
            
            foreach (var child in children)
            {
                if (IsCloseToTarget(child.State, target))
                {
                    child.IsTerminal = true;
                }
                queue.Enqueue(child);
            }
        }

        return root;
    }

    private List<TreeNode<T>> GenerateTrainingChildren(TreeNode<T> parent, Tensor<T> target)
    {
        var children = new List<TreeNode<T>>();
        
        for (int i = 0; i < _totOptions.BranchingFactor; i++)
        {
            var progress = (double)(parent.Depth + 1) / _totOptions.MaxTreeDepth;
            var interpolated = InterpolateTensors(parent.State, target, progress);
            
            // Add noise for diversity
            var noise = GenerateNoise(NumOps.FromDouble(0.1));
            var childState = interpolated.Add(noise);
            
            var child = new TreeNode<T>(childState, parent);
            children.Add(child);
        }

        parent.Children = children;
        parent.IsExpanded = true;

        return children;
    }

    private Tensor<T> InterpolateTensors(Tensor<T> start, Tensor<T> end, double progress)
    {
        var result = new Tensor<T>(start.Shape);
        var progressT = NumOps.FromDouble(progress);
        var oneMinusProgress = NumOps.Subtract(NumOps.One, progressT);

        for (int i = 0; i < Math.Min(start.Length, end.Length); i++)
        {
            var interpolated = NumOps.Add(
                NumOps.Multiply(start[i], oneMinusProgress),
                NumOps.Multiply(end[i], progressT)
            );
            result[i] = interpolated;
        }

        return result;
    }

    private bool IsCloseToTarget(Tensor<T> state, Tensor<T> target)
    {
        var distance = TensorDistance(state, target);
        return NumOps.LessThan(distance, NumOps.FromDouble(_totOptions.TargetProximityThreshold));
    }

    private T TensorDistance(Tensor<T> a, Tensor<T> b)
    {
        var sum = NumOps.Zero;
        
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    private List<List<TreeNode<T>>> ExtractSuccessfulPaths(TreeNode<T> root, Tensor<T> target)
    {
        var paths = new List<List<TreeNode<T>>>();
        
        void ExtractPaths(TreeNode<T> node, List<TreeNode<T>> currentPath)
        {
            currentPath.Add(node);
            
            if (node.IsTerminal && IsCloseToTarget(node.State, target))
            {
                paths.Add(new List<TreeNode<T>>(currentPath));
            }
            else
            {
                foreach (var child in node.Children)
                {
                    ExtractPaths(child, currentPath);
                }
            }

            currentPath.RemoveAt(currentPath.Count - 1);
        }

        ExtractPaths(root, new List<TreeNode<T>>());
        return paths;
    }

    private Tensor<T> CombineChildStates(TreeNode<T> parent)
    {
        var combined = new Tensor<T>(new[] { _totOptions.StateShape[0] * _totOptions.BranchingFactor });
        var stateSize = _totOptions.StateShape[0];

        for (int i = 0; i < parent.Children.Count && i < _totOptions.BranchingFactor; i++)
        {
            var childState = parent.Children[i].State;
            
            for (int j = 0; j < stateSize; j++)
            {
                combined[i * stateSize + j] = childState[j];
            }
        }

        return combined;
    }

    private void TrainStateEvaluator(TreeNode<T> root, Tensor<T> target)
    {
        var queue = new Queue<TreeNode<T>>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();
            
            // Calculate value based on distance to target
            var distance = TensorDistance(node.State, target);
            var value = NumOps.Exp(NumOps.Negate(distance));
            
            var valueTarget = new Tensor<T>(new[] { 1 });
            valueTarget[0] = value;
            
            _stateEvaluator.Train(node.State, valueTarget);

            foreach (var child in node.Children)
            {
                queue.Enqueue(child);
            }
        }
    }

    private void TrainThoughtSelector(TreeNode<T> root)
    {
        var queue = new Queue<TreeNode<T>>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            var parent = queue.Dequeue();
            
            if (parent.Children.Count == 0)
                continue;

            // Find best child
            var bestChild = parent.Children.OrderByDescending(c => c.Value).First();

            foreach (var child in parent.Children)
            {
                var combined = CombineStates(parent.State, child.State);
                var shouldSelect = child == bestChild ? NumOps.One : NumOps.Zero;
                
                var target = new Tensor<T>(new[] { 1 });
                target[0] = shouldSelect;
                
                _thoughtSelector.Train(combined, target);
                
                queue.Enqueue(child);
            }
        }
    }

    private bool EvaluateTransition(Tensor<T> from, Tensor<T> to)
    {
        var combined = CombineStates(from, to);
        var selection = _thoughtSelector.Predict(combined);
        return NumOps.GreaterThan(selection[0], NumOps.FromDouble(0.5));
    }

    private TreeNode<T> BuildExplorationTree(TreeNode<T> root, Tensor<T> target, int maxDepth)
    {
        var queue = new Queue<TreeNode<T>>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            
            if (current.Depth >= maxDepth)
                continue;

            var children = GenerateChildren(current);
            
            foreach (var child in children.Take(2))  // Limit for explanation
            {
                queue.Enqueue(child);
            }
        }

        return root;
    }

    private Tensor<T> ConvertTreeToExplanation(TreeNode<T> root)
    {
        // Flatten tree structure into explanation tensor
        var explanation = new Tensor<T>(_totOptions.StateShape);
        var nodeCount = CountTreeNodes(root);
        
        var queue = new Queue<TreeNode<T>>();
        queue.Enqueue(root);
        var index = 0;

        while (queue.Count > 0 && index < explanation.Length)
        {
            var node = queue.Dequeue();
            
            // Add node contribution to explanation
            var weight = NumOps.Divide(node.Value, NumOps.FromDouble(nodeCount));
            
            for (int i = 0; i < node.State.Length && index + i < explanation.Length; i++)
            {
                var value = NumOps.Multiply(node.State[i], weight);
                explanation[index + i] = value;
            }

            index += node.State.Length;

            foreach (var child in node.Children)
            {
                queue.Enqueue(child);
            }
        }

        return explanation;
    }

    private List<Tensor<T>> ExploreAlternatives(TreeNode<T> root, int branchCount)
    {
        var alternatives = new List<Tensor<T>>();
        
        for (int i = 0; i < branchCount; i++)
        {
            var children = GenerateChildren(root);
            
            if (children.Count > 0)
            {
                var selected = children[Random.Next(children.Count)];
                alternatives.Add(selected.State);
            }
        }

        return alternatives;
    }

    private Tensor<T> SelectBestAlternative(List<Tensor<T>> alternatives)
    {
        var bestValue = NumOps.Zero;
        var bestAlternative = alternatives[0];

        foreach (var alt in alternatives)
        {
            var evaluation = _stateEvaluator.Predict(alt);
            var value = evaluation[0];
            
            if (NumOps.GreaterThan(value, bestValue))
            {
                bestValue = value;
                bestAlternative = alt;
            }
        }

        return bestAlternative;
    }

    private Tensor<T> GenerateNoise(T scale)
    {
        var noise = new Tensor<T>(_totOptions.StateShape);
        
        for (int i = 0; i < noise.Length; i++)
        {
            var value = NumOps.Multiply(
                NumOps.FromDouble((Random.NextDouble() - 0.5) * 2.0),
                scale
            );
            noise[i] = value;
        }

        return noise;
    }

    #endregion

    #region IParameterizable Implementation

    public override Vector<T> GetParameters()
    {
        // Get parameters from all networks
        var genParams = _thoughtGenerator.GetParameters();
        var evalParams = _stateEvaluator.GetParameters();
        var selParams = _thoughtSelector.GetParameters();
        
        // Combine all parameters
        var totalSize = genParams.Length + evalParams.Length + selParams.Length;
        var allParams = new T[totalSize];
        
        int offset = 0;
        var genParamsArray = new T[genParams.Length];
        for (int i = 0; i < genParams.Length; i++)
        {
            genParamsArray[i] = genParams[i];
        }
        genParamsArray.CopyTo(allParams, offset);
        offset += genParams.Length;
        
        var evalParamsArray = new T[evalParams.Length];
        for (int i = 0; i < evalParams.Length; i++)
        {
            evalParamsArray[i] = evalParams[i];
        }
        evalParamsArray.CopyTo(allParams, offset);
        offset += evalParams.Length;
        
        var selParamsArray = new T[selParams.Length];
        for (int i = 0; i < selParams.Length; i++)
        {
            selParamsArray[i] = selParams[i];
        }
        selParamsArray.CopyTo(allParams, offset);
        
        return new Vector<T>(allParams);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Get current parameter sizes
        var genParams = _thoughtGenerator.GetParameters();
        var evalParams = _stateEvaluator.GetParameters();
        var selParams = _thoughtSelector.GetParameters();
        
        int offset = 0;
        
        // Set parameters for thought generator
        var genData = new T[genParams.Length];
        for (int i = 0; i < genParams.Length; i++)
        {
            genData[i] = parameters[offset + i];
        }
        _thoughtGenerator.SetParameters(new Vector<T>(genData));
        offset += genParams.Length;
        
        // Set parameters for state evaluator
        var evalData = new T[evalParams.Length];
        for (int i = 0; i < evalParams.Length; i++)
        {
            evalData[i] = parameters[offset + i];
        }
        _stateEvaluator.SetParameters(new Vector<T>(evalData));
        offset += evalParams.Length;
        
        // Set parameters for thought selector
        var selData = new T[selParams.Length];
        for (int i = 0; i < selParams.Length; i++)
        {
            selData[i] = parameters[offset + i];
        }
        _thoughtSelector.SetParameters(new Vector<T>(selData));
    }

    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new TreeOfThoughtModel<T>(_totOptions);
        newModel.SetParameters(parameters);
        return newModel;
    }

    #endregion

    #region IModelSerializer Implementation

    public override byte[] Serialize()
    {
        var data = new List<byte>();
        
        // Serialize options
        using (var ms = new MemoryStream())
        using (var writer = new BinaryWriter(ms))
        {
            // Write key options
            writer.Write(_totOptions.BranchingFactor);
            writer.Write(_totOptions.MaxTreeDepth);
            writer.Write(_totOptions.HiddenSize);
            writer.Write(_totOptions.AttentionHeads);
            writer.Write((int)_totOptions.SearchStrategy);
            writer.Write(_totOptions.BeamWidth);
            writer.Write(_totOptions.SelectionThreshold);
            writer.Write(_totOptions.TerminalValueThreshold);
            writer.Write(_totOptions.MonteCarloIterations);
            writer.Write(_totOptions.TargetProximityThreshold);
            writer.Write(_totOptions.ExplanationDepth);
            writer.Write(_totOptions.RefinementBranches);
            writer.Write(_totOptions.EnableNodeCache);
            writer.Write(_totOptions.UseParallelExploration);
            
            // Write state shape
            writer.Write(_totOptions.StateShape.Length);
            foreach (var dim in _totOptions.StateShape)
            {
                writer.Write(dim);
            }
            
            data.AddRange(ms.ToArray());
        }
        
        // Serialize neural networks
        data.AddRange(_thoughtGenerator.Serialize());
        data.AddRange(_stateEvaluator.Serialize());
        data.AddRange(_thoughtSelector.Serialize());

        return data.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        int offset = 0;
        
        // Deserialize options
        using (var ms = new MemoryStream(data))
        using (var reader = new BinaryReader(ms))
        {
            _totOptions.BranchingFactor = reader.ReadInt32();
            _totOptions.MaxTreeDepth = reader.ReadInt32();
            _totOptions.HiddenSize = reader.ReadInt32();
            _totOptions.AttentionHeads = reader.ReadInt32();
            _totOptions.SearchStrategy = (TreeSearchStrategy)reader.ReadInt32();
            _totOptions.BeamWidth = reader.ReadInt32();
            _totOptions.SelectionThreshold = reader.ReadDouble();
            _totOptions.TerminalValueThreshold = reader.ReadDouble();
            _totOptions.MonteCarloIterations = reader.ReadInt32();
            _totOptions.TargetProximityThreshold = reader.ReadDouble();
            _totOptions.ExplanationDepth = reader.ReadInt32();
            _totOptions.RefinementBranches = reader.ReadInt32();
            _totOptions.EnableNodeCache = reader.ReadBoolean();
            _totOptions.UseParallelExploration = reader.ReadBoolean();
            
            // Read state shape
            var shapeLength = reader.ReadInt32();
            var stateShape = new int[shapeLength];
            for (int i = 0; i < shapeLength; i++)
            {
                stateShape[i] = reader.ReadInt32();
            }
            _totOptions.StateShape = stateShape;
            
            offset = (int)ms.Position;
        }
        
        // Extract and deserialize neural networks
        // First, find the size of each network's serialized data
        // This is a simplified approach - in production, you'd want to store sizes
        var remainingBytes = data.Length - offset;
        var bytesPerNetwork = remainingBytes / 3; // Rough estimate
        
        var generatorBytes = new byte[bytesPerNetwork];
        Array.Copy(data, offset, generatorBytes, 0, bytesPerNetwork);
        _thoughtGenerator.Deserialize(generatorBytes);
        offset += bytesPerNetwork;
        
        var evaluatorBytes = new byte[bytesPerNetwork];
        Array.Copy(data, offset, evaluatorBytes, 0, bytesPerNetwork);
        _stateEvaluator.Deserialize(evaluatorBytes);
        offset += bytesPerNetwork;
        
        var selectorBytes = new byte[remainingBytes - 2 * bytesPerNetwork];
        Array.Copy(data, offset, selectorBytes, 0, selectorBytes.Length);
        _thoughtSelector.Deserialize(selectorBytes);
    }

    #endregion

    #region ICloneable Implementation

    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = new TreeOfThoughtModel<T>(_totOptions);
        copy.SetParameters(GetParameters());
        return copy;
    }


    #endregion
}