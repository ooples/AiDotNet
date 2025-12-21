using AiDotNet.Autodiff;
namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a symbolic expression tree for mathematical operations that can be used for symbolic regression.
/// </summary>
/// <typeparam name="T">The numeric type used in the expression tree (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An ExpressionTree is like a mathematical formula represented as a tree structure.
/// Each node in the tree is either a number, a variable, or an operation (like addition or multiplication).
/// This allows the AI to create and evolve mathematical formulas that can model your data.
/// </remarks>
public class ExpressionTree<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the type of this node (constant, variable, or operation).
    /// </summary>
    public ExpressionNodeType Type { get; private set; }

    /// <summary>
    /// Gets the value stored in this node. For constants, this is the actual value.
    /// For variables, this is the index of the variable in the input vector.
    /// </summary>
    public T Value { get; private set; }

    /// <summary>
    /// Gets the left child node of this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> In operations like addition (a + b), the left child represents 'a'.
    /// </remarks>
    public ExpressionTree<T, TInput, TOutput>? Left { get; private set; }

    /// <summary>
    /// Gets the right child node of this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> In operations like addition (a + b), the right child represents 'b'.
    /// </remarks>
    public ExpressionTree<T, TInput, TOutput>? Right { get; private set; }

    /// <summary>
    /// Gets the parent node of this node.
    /// </summary>
    public ExpressionTree<T, TInput, TOutput>? Parent { get; private set; }

    /// <summary>
    /// Gets the complexity of this expression tree, measured as the total number of nodes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Complexity tells you how complicated the formula is.
    /// A higher number means a more complex formula with more terms and operations.
    /// </remarks>
    public int Complexity => 1 + (Left?.Complexity ?? 0) + (Right?.Complexity ?? 0);

    /// <summary>
    /// Sets the type of this node.
    /// </summary>
    /// <param name="type">The node type to set.</param>
    public void SetType(ExpressionNodeType type)
    {
        Type = type;
    }

    /// <summary>
    /// Sets the value of this node.
    /// </summary>
    /// <param name="value">The value to set.</param>
    public void SetValue(T value)
    {
        Value = value;
    }

    /// <summary>
    /// Sets the left child of this node and updates the parent reference of the child.
    /// </summary>
    /// <param name="left">The node to set as the left child.</param>
    public void SetLeft(ExpressionTree<T, TInput, TOutput>? left)
    {
        Left = left;
        if (left != null)
        {
            left.Parent = this;
        }
    }

    /// <summary>
    /// Sets the right child of this node and updates the parent reference of the child.
    /// </summary>
    /// <param name="right">The node to set as the right child.</param>
    public void SetRight(ExpressionTree<T, TInput, TOutput>? right)
    {
        Right = right;
        if (right != null)
        {
            right.Parent = this;
        }
    }

    /// <summary>
    /// Returns a string representation of this expression tree.
    /// </summary>
    /// <returns>A string representing the mathematical expression.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This converts the tree into a readable mathematical formula.
    /// For example, an addition node with children might return "(2 + x[0])".
    /// </remarks>
    public override string ToString()
    {
        return Type switch
        {
            ExpressionNodeType.Constant => Value?.ToString(),
            ExpressionNodeType.Variable => $"x[{Value}]",
            ExpressionNodeType.Add => $"({Left} + {Right})",
            ExpressionNodeType.Subtract => $"({Left} - {Right})",
            ExpressionNodeType.Multiply => $"({Left} * {Right})",
            ExpressionNodeType.Divide => $"({Left} / {Right})",
            _ => throw new ArgumentException("Invalid node type"),
        } ?? string.Empty;
    }

    /// <summary>
    /// Helper object for performing numeric operations on type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Shared random number generator for all mutation and crossover operations.
    /// </summary>
    /// <remarks>
    /// Using ThreadLocal ensures thread safety while maintaining good randomness quality.
    /// Each thread gets its own Random instance, avoiding issues with multiple threads
    /// accessing a shared Random instance or multiple instances created with the same seed.
    /// </remarks>
    private static readonly ThreadLocal<Random> _random = new ThreadLocal<Random>(() => RandomHelper.CreateSecureRandom());

    /// <summary>
    /// Creates a new expression tree node with the specified properties.
    /// </summary>
    /// <param name="type">The type of node to create.</param>
    /// <param name="value">The value for this node (for constants and variables).</param>
    /// <param name="left">The left child node.</param>
    /// <param name="right">The right child node.</param>
    /// <param name="lossFunction">Optional loss function to use for training. If null, uses Mean Squared Error (MSE) for symbolic regression.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new part of your mathematical formula.
    /// You can create simple nodes (like numbers or variables) or operation nodes
    /// (like addition or multiplication) that connect to other nodes.
    /// </remarks>
    public ExpressionTree(ExpressionNodeType type, T? value = default, ExpressionTree<T, TInput, TOutput>? left = null, ExpressionTree<T, TInput, TOutput>? right = null, ILossFunction<T>? lossFunction = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        Type = type;
        Value = value ?? _numOps.Zero;
        Left = left;
        Right = right;
        _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Cached count of features used in this expression tree.
    /// </summary>
    private int _featureCount;

    /// <summary>
    /// The default loss function used by this model for gradient computation.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets the default loss function used by this model for gradient computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For ExpressionTree (symbolic regression), the default loss function is Mean Squared Error (MSE),
    /// which is the standard loss function for regression problems.
    /// </para>
    /// </remarks>
    public ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <summary>
    /// Gets the number of features (variables) used in this expression tree.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many different input variables
    /// your formula uses. For example, if your formula uses x[0], x[1], and x[2],
    /// the feature count would be 3.
    /// </remarks>
    public int FeatureCount
    {
        get
        {
            if (_featureCount == 0)
            {
                _featureCount = CalculateFeatureCount();
            }

            return _featureCount;
        }
    }

    /// <summary>
    /// Checks if a specific feature (variable) is used in this expression tree.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used, false otherwise.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This checks if your formula uses a specific input variable.
    /// For example, if featureIndex is 2, it checks if x[2] appears anywhere in your formula.
    /// </remarks>
    public bool IsFeatureUsed(int featureIndex)
    {
        return IsFeatureUsedRecursive(this, featureIndex);
    }

    /// <summary>
    /// Calculates the number of unique features used in this expression tree.
    /// </summary>
    /// <returns>The count of unique features actually used in the tree.</returns>
    /// <remarks>
    /// This method counts the unique feature indices used in the tree. For example,
    /// if the tree uses features x[0] and x[5], this returns 2 (the count of unique features),
    /// not 6. This accurately represents how many different input variables the formula uses.
    /// </remarks>
    private int CalculateFeatureCount()
    {
        HashSet<int> uniqueFeatures = new HashSet<int>();
        CollectUniqueFeatures(this, uniqueFeatures);
        return uniqueFeatures.Count;
    }

    /// <summary>
    /// Recursively collects unique feature indices used in a node and its children.
    /// </summary>
    /// <param name="node">The node to check.</param>
    /// <param name="uniqueFeatures">The set to collect unique feature indices.</param>
    private void CollectUniqueFeatures(ExpressionTree<T, TInput, TOutput> node, HashSet<int> uniqueFeatures)
    {
        if (node == null) return;

        if (node.Type == ExpressionNodeType.Variable)
        {
            uniqueFeatures.Add(_numOps.ToInt32(node.Value));
        }

        if (node.Left != null)
        {
            CollectUniqueFeatures(node.Left, uniqueFeatures);
        }

        if (node.Right != null)
        {
            CollectUniqueFeatures(node.Right, uniqueFeatures);
        }
    }

    /// <summary>
    /// Recursively checks if a specific feature is used in a node or its children.
    /// </summary>
    /// <param name="node">The node to check.</param>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used, false otherwise.</returns>
    private bool IsFeatureUsedRecursive(ExpressionTree<T, TInput, TOutput> node, int featureIndex)
    {
        if (node.Type == ExpressionNodeType.Variable && _numOps.ToInt32(node.Value) == featureIndex)
        {
            return true;
        }

        bool leftUsed = node.Left != null && IsFeatureUsedRecursive(node.Left, featureIndex);
        bool rightUsed = node.Right != null && IsFeatureUsedRecursive(node.Right, featureIndex);

        return leftUsed || rightUsed;
    }

    /// <summary>
    /// Evaluates this expression tree for a given input vector.
    /// </summary>
    /// <param name="input">The input vector containing values for variables.</param>
    /// <returns>The result of evaluating the expression.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This calculates the result of your formula for a specific set of input values.
    /// For example, if your formula is "2*x[0] + x[1]" and your input is [3, 4], the result would be 2*3 + 4 = 10.
    /// </remarks>
    public T Evaluate(Vector<T> input)
    {
        return Type switch
        {
            ExpressionNodeType.Constant => Value,
            ExpressionNodeType.Variable => input[_numOps.ToInt32(Value)],
            ExpressionNodeType.Add => _numOps.Add(Left!.Evaluate(input), Right!.Evaluate(input)),
            ExpressionNodeType.Subtract => _numOps.Subtract(Left!.Evaluate(input), Right!.Evaluate(input)),
            ExpressionNodeType.Multiply => _numOps.Multiply(Left!.Evaluate(input), Right!.Evaluate(input)),
            ExpressionNodeType.Divide => _numOps.Divide(Left!.Evaluate(input), Right!.Evaluate(input)),
            _ => throw new ArgumentException("Invalid node type"),
        };
    }

    /// <summary>
    /// Writes this expression tree to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This saves your formula to a file or stream so you can load it later.
    /// </remarks>
    public void Serialize(BinaryWriter writer)
    {
        writer.Write((int)Type);
        writer.Write(Convert.ToDouble(Value));
        writer.Write(Left != null);
        Left?.Serialize(writer);
        writer.Write(Right != null);
        Right?.Serialize(writer);
    }

    /// <summary>
    /// Deserializes an expression tree from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader containing the serialized tree data.</param>
    /// <returns>A new ExpressionTree instance created from the serialized data.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method reads a saved expression tree from binary data and reconstructs it.
    /// Think of it like opening a saved file that contains your mathematical formula.
    /// </remarks>
    public ExpressionTree<T, TInput, TOutput> Deserialize(BinaryReader reader)
    {
        ExpressionNodeType type = (ExpressionNodeType)reader.ReadInt32();
        T value = _numOps.FromDouble(reader.ReadDouble());
        bool hasLeft = reader.ReadBoolean();
        ExpressionTree<T, TInput, TOutput>? left = hasLeft ? Deserialize(reader) : null;
        bool hasRight = reader.ReadBoolean();
        ExpressionTree<T, TInput, TOutput>? right = hasRight ? Deserialize(reader) : null;

        return new ExpressionTree<T, TInput, TOutput>(type, value, left, right);
    }

    /// <summary>
    /// Creates a modified version of this expression tree by applying random mutations.
    /// </summary>
    /// <param name="mutationRate">The probability (0.0 to 1.0) that a mutation will occur at each node.</param>
    /// <returns>A new expression tree with mutations applied.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Mutation is like making small random changes to a formula to see if it improves.
    /// For example, changing a "+" to a "*" or changing a constant from 2.5 to 3.1.
    /// This is inspired by how genetic mutations work in nature and helps the AI explore different solutions.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> Mutate(double mutationRate)
    {
        ExpressionTree<T, TInput, TOutput> mutatedTree = (ExpressionTree<T, TInput, TOutput>)Copy();

        if (_random.Value!.NextDouble() < mutationRate)
        {
            switch (_random.Value!.Next(3))
            {
                case 0: // Change node type
                    mutatedTree.Type = (ExpressionNodeType)_random.Value!.Next(Enum.GetValues(typeof(ExpressionNodeType)).Length);
                    break;
                case 1: // Change value (for Constant or Variable nodes)
                    if (mutatedTree.Type == ExpressionNodeType.Constant)
                    {
                        mutatedTree.Value = _numOps.FromDouble(_random.Value!.NextDouble() * 10 - 5); // Random value between -5 and 5
                    }
                    else if (mutatedTree.Type == ExpressionNodeType.Variable)
                    {
                        mutatedTree.Value = _numOps.FromDouble(_random.Value!.Next(10)); // Assume max 10 variables
                    }
                    break;
                case 2: // Regenerate subtree
                    int maxDepth = 3;
                    mutatedTree = GenerateRandomTree(maxDepth);
                    break;
            }
        }

        // Recursively mutate children
        if (mutatedTree.Left != null)
        {
            mutatedTree.Left = (ExpressionTree<T, TInput, TOutput>)mutatedTree.Left.Mutate(mutationRate);
        }
        if (mutatedTree.Right != null)
        {
            mutatedTree.Right = (ExpressionTree<T, TInput, TOutput>)mutatedTree.Right.Mutate(mutationRate);
        }

        return mutatedTree;
    }

    /// <summary>
    /// Combines this expression tree with another to create a new "offspring" expression tree.
    /// </summary>
    /// <param name="other">The other expression tree to combine with.</param>
    /// <param name="crossoverRate">The probability (0.0 to 1.0) that crossover will occur.</param>
    /// <returns>A new expression tree that combines parts from both parent trees.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Crossover is like taking parts from two different formulas and combining them
    /// to create a new formula. For example, if one formula is (x + 2) and another is (y * 3),
    /// crossover might create (x * 3) by taking parts from each. This mimics how genetic traits
    /// are passed from parents to children in nature.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> Crossover(IFullModel<T, TInput, TOutput> other, double crossoverRate)
    {
        if (!(other is ExpressionTree<T, TInput, TOutput> otherTree))
        {
            throw new ArgumentException("Crossover can only be performed with another ExpressionTree.");
        }

        ExpressionTree<T, TInput, TOutput> offspring = (ExpressionTree<T, TInput, TOutput>)Copy();

        if (_random.Value!.NextDouble() < crossoverRate)
        {
            // Select a random subtree from the other parent
            ExpressionTree<T, TInput, TOutput> selectedSubtree = SelectRandomSubtree(otherTree);

            // Replace a random subtree in the offspring with the selected subtree
            ReplaceRandomSubtree(offspring, selectedSubtree);
        }

        return offspring;
    }

    /// <summary>
    /// Creates a deep copy of this expression tree.
    /// </summary>
    /// <returns>A new expression tree with the same structure and values as this one.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This creates an exact duplicate of the formula, like making a photocopy.
    /// This is important because we often need to make changes to a formula without modifying the original.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> Copy()
    {
        return new ExpressionTree<T, TInput, TOutput>(
            Type,
            Value,
            Left?.Clone() as ExpressionTree<T, TInput, TOutput>,
            Right?.Clone() as ExpressionTree<T, TInput, TOutput>
        );
    }

    /// <summary>
    /// Creates a random expression tree with a specified maximum depth.
    /// </summary>
    /// <param name="maxDepth">The maximum depth of the tree to generate.</param>
    /// <returns>A randomly generated expression tree.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a random mathematical formula with a limit on how complex it can be.
    /// The maxDepth parameter controls this complexity - higher values allow for more complex formulas.
    /// </remarks>
    private ExpressionTree<T, TInput, TOutput> GenerateRandomTree(int maxDepth)
    {
        if (maxDepth == 0 || _random.Value!.NextDouble() < 0.3) // 30% chance of leaf node
        {
            if (_random.Value!.NextDouble() < 0.5)
            {
                return new ExpressionTree<T, TInput, TOutput>(ExpressionNodeType.Constant, _numOps.FromDouble(_random.Value!.NextDouble() * 10 - 5));
            }
            else
            {
                return new ExpressionTree<T, TInput, TOutput>(ExpressionNodeType.Variable, _numOps.FromDouble(_random.Value!.Next(10)));
            }
        }
        else
        {
            ExpressionNodeType operationType = (ExpressionNodeType)_random.Value!.Next(2, 6); // Add, Subtract, Multiply, or Divide
            return new ExpressionTree<T, TInput, TOutput>(
                operationType,
                default,
                GenerateRandomTree(maxDepth - 1),
                GenerateRandomTree(maxDepth - 1)
            );
        }
    }

    /// <summary>
    /// Selects a random subtree from the given expression tree.
    /// </summary>
    /// <param name="tree">The expression tree to select from.</param>
    /// <returns>A randomly selected subtree.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This picks a random part of a formula. For example, in the formula (x + (y * 2)),
    /// it might select the whole formula, just (y * 2), or even just y or 2.
    /// </remarks>
    private ExpressionTree<T, TInput, TOutput> SelectRandomSubtree(ExpressionTree<T, TInput, TOutput> tree)
    {
        if (tree.Left == null && tree.Right == null)
        {
            return tree;
        }
        else if (_random.Value!.NextDouble() < 0.3) // 30% chance of selecting current node
        {
            return tree;
        }
        else
        {
            if (tree.Left != null && (tree.Right == null || _random.Value!.NextDouble() < 0.5))
            {
                return SelectRandomSubtree(tree.Left);
            }
            else
            {
                return SelectRandomSubtree(tree.Right!);
            }
        }
    }

    /// <summary>
    /// Replaces a random subtree in the given tree with the provided replacement subtree.
    /// </summary>
    /// <param name="tree">The tree to modify.</param>
    /// <param name="replacement">The replacement subtree.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This replaces a random part of a formula with a different part.
    /// For example, in (x + y), it might replace y with (z * 2) to create (x + (z * 2)).
    /// </remarks>
    private void ReplaceRandomSubtree(ExpressionTree<T, TInput, TOutput> tree, ExpressionTree<T, TInput, TOutput> replacement)
    {
        if (_random.Value!.NextDouble() < 0.3) // 30% chance of replacing current node
        {
            tree.Type = replacement.Type;
            tree.Value = replacement.Value;
            tree.Left = replacement.Left?.Clone() as ExpressionTree<T, TInput, TOutput>;
            tree.Right = replacement.Right?.Clone() as ExpressionTree<T, TInput, TOutput>;
        }
        else
        {
            if (tree.Left != null && (tree.Right == null || _random.Value!.NextDouble() < 0.5))
            {
                ReplaceRandomSubtree(tree.Left, replacement);
            }
            else if (tree.Right != null)
            {
                ReplaceRandomSubtree(tree.Right, replacement);
            }
        }
    }

    /// <summary>
    /// Fits the expression tree to the provided training data.
    /// </summary>
    /// <param name="X">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <b>For Beginners:</b> For expression trees, "fitting" just checks if the formula can work with your data.
    /// Unlike other AI models, the formula itself doesn't change during fitting - it's predefined by the tree structure.
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        // For ExpressionTree, Fit is the same as Train
        Train(X, y);
    }

    /// <summary>
    /// Trains the expression tree on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <b>For Beginners:</b> For expression trees, "training" just validates that the formula can process your data.
    /// The formula itself doesn't learn or change during training - it's predefined by the tree structure.
    /// </remarks>
    public void Train(Matrix<T> x, Vector<T> y)
    {
        // For ExpressionTree, we don't actually train the model
        // The structure is defined by the tree, and we don't adjust it based on data
        // However, we can use this method to validate that our tree can process the input
        if (x.Columns < FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {x.Columns} columns, but the model expects at least {FeatureCount} features.");
        }
    }

    /// <summary>
    /// Makes predictions using this expression tree for multiple input samples.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <returns>A vector containing the predicted values for each input sample.</returns>
    /// <exception cref="ArgumentException">Thrown when the input matrix has incorrect dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method takes your data (like height, weight, age values) and
    /// runs each row through the mathematical formula represented by this tree to get predictions.
    /// For example, if your tree represents "2x + y", and your input has values [3,4], the prediction would be 2*3 + 4 = 10.
    ///
    /// <b>Note:</b> If the input has more features than the model requires, the extra features are allowed but ignored.
    /// Only the first FeatureCount features are used in predictions. This flexibility supports transfer learning scenarios
    /// where input data may contain additional features not used by this particular model.
    /// </remarks>
    public Vector<T> Predict(Matrix<T> input)
    {
        if (input.Columns < FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {input.Columns} columns, but the model expects at least {FeatureCount} features.");
        }

        Vector<T> predictions = new(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = Evaluate(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Gets metadata about this expression tree model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about this model.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This provides useful information about your formula, like how complex it is
    /// and how many input variables it needs. Think of it as a summary sheet about your mathematical model.
    /// </remarks>
    public ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ExpressionTree,
            FeatureCount = FeatureCount,
            Complexity = Complexity,
            Description = ToString(),
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NodeType", Type },
                { "HasLeftChild", Left != null },
                { "HasRightChild", Right != null }
            }
        };
    }

    /// <summary>
    /// Converts this expression tree to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized expression tree.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This converts your mathematical formula into a compact format that can be
    /// saved to a file or sent over the internet. It's like zipping up your formula for storage.
    /// </remarks>
    public byte[] Serialize()
    {
        using MemoryStream ms = new();
        using BinaryWriter writer = new(ms);
        Serialize(writer);

        return ms.ToArray();
    }

    /// <summary>
    /// Loads an expression tree from a byte array, replacing the current tree's structure.
    /// </summary>
    /// <param name="data">The byte array containing the serialized expression tree.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This loads a previously saved formula from a compact format and
    /// replaces the current formula with it. It's like opening a saved file and loading its contents.
    /// </remarks>
    public void Deserialize(byte[] data)
    {
        using MemoryStream ms = new(data);
        using BinaryReader reader = new(ms);
        ExpressionTree<T, TInput, TOutput> deserializedTree = Deserialize(reader);
        this.Type = deserializedTree.Type;
        this.Value = deserializedTree.Value;
        this.Left = deserializedTree.Left;
        this.Right = deserializedTree.Right;
    }

    /// <summary>
    /// Gets a list of all nodes in this expression tree.
    /// </summary>
    /// <returns>A list containing all nodes in the tree.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This collects all the parts of your formula into a list.
    /// For example, if your formula is (x + 2) * y, this would give you a list containing:
    /// the multiplication operation, the addition operation, the x variable, the constant 2, and the y variable.
    /// </remarks>
    public List<ExpressionTree<T, TInput, TOutput>> GetAllNodes()
    {
        var nodes = new List<ExpressionTree<T, TInput, TOutput>>();
        CollectNodes(this, nodes);

        return nodes;
    }

    /// <summary>
    /// Helper method that recursively collects all nodes in the tree.
    /// </summary>
    /// <param name="node">The current node being processed.</param>
    /// <param name="nodes">The list to add nodes to.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper method that walks through every part of your formula
    /// and adds each piece to a list. It uses recursion (calling itself) to visit every branch of the tree.
    /// </remarks>
    private void CollectNodes(ExpressionTree<T, TInput, TOutput>? node, List<ExpressionTree<T, TInput, TOutput>> nodes)
    {
        if (node == null) return;
        nodes.Add(node);
        CollectNodes(node.Left, nodes);
        CollectNodes(node.Right, nodes);
    }

    /// <summary>
    /// Finds a node in the tree by its unique identifier.
    /// </summary>
    /// <param name="id">The unique identifier of the node to find.</param>
    /// <returns>The node with the specified ID, or null if no such node exists.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Every part of your formula has a unique ID number.
    /// This method helps you find a specific part by its ID, like finding a person by their social security number.
    /// </remarks>
    public ExpressionTree<T, TInput, TOutput>? FindNodeById(int id)
    {
        return GetAllNodes().FirstOrDefault(n => n.Id == id);
    }

    /// <summary>
    /// Gets the unique identifier for this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a unique number assigned to each part of your formula,
    /// making it easy to identify and reference specific parts of the expression tree.
    /// </remarks>
    public int Id { get; } = Interlocked.Increment(ref _nextId);

    /// <summary>
    /// Static counter used to generate unique IDs for expression tree nodes.
    /// </summary>
    private static int _nextId;

    /// <summary>
    /// Creates a new expression tree with updated coefficient values.
    /// </summary>
    /// <param name="newCoefficients">The new coefficient values to use.</param>
    /// <returns>A new expression tree with the updated coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of new coefficients doesn't match the current number.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This changes the constant numbers in your formula without changing its structure.
    /// For example, if your formula is "2x + 3", this might change it to "4x + 1" by updating the coefficients 2 and 3.
    /// This is useful when fine-tuning a model to make better predictions.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> UpdateCoefficients(Vector<T> newCoefficients)
    {
        if (newCoefficients.Length != this.Coefficients.Length)
        {
            throw new ArgumentException($"The number of new coefficients ({newCoefficients.Length}) must match the current number of coefficients ({this.Coefficients.Length}).");
        }

        ExpressionTree<T, TInput, TOutput> updatedTree = (ExpressionTree<T, TInput, TOutput>)this.Clone();
        int coefficientIndex = 0;

        void UpdateConstantNodes(ExpressionTree<T, TInput, TOutput> node)
        {
            if (node.Type == ExpressionNodeType.Constant)
            {
                node.Value = newCoefficients[coefficientIndex++];
            }
            if (node.Left != null)
            {
                UpdateConstantNodes(node.Left);
            }
            if (node.Right != null)
            {
                UpdateConstantNodes(node.Right);
            }
        }

        UpdateConstantNodes(updatedTree);

        return updatedTree;
    }

    /// <summary>
    /// Creates a deep copy of this expression tree.
    /// </summary>
    /// <returns>A new, identical expression tree.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This creates an exact duplicate of the entire formula tree.
    /// Unlike the Copy method which returns a general IFullModel, this method returns 
    /// a specific ExpressionTree. This is useful when you need to make changes to a
    /// copy without affecting the original formula.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        // Reuse existing Copy method which already creates a deep copy
        return Copy();
    }

    /// <summary>
    /// Creates a clone of this expression tree.
    /// </summary>
    /// <returns>A new expression tree with the same structure and values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This creates an exact duplicate of your formula.
    /// It's essentially the same as DeepCopy and Copy - it makes a complete
    /// duplicate that you can modify without changing the original.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> Clone()
    {
        // Reuse existing Copy method
        return Copy();
    }

    /// <summary>
    /// Gets the parameters of this expression tree.
    /// </summary>
    /// <returns>A vector containing all coefficient values in this expression tree.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This returns all the constant numbers from your formula.
    /// For example, if your formula is "2x + 3y + 5", this would give you [2, 3, 5].
    /// These numbers are the adjustable parameters that can be tuned to improve predictions.
    /// </remarks>
    public Vector<T> GetParameters()
    {
        // Return the coefficients which are the model's parameters
        return Coefficients;
    }

    /// <summary>
    /// Creates a new expression tree with updated parameters.
    /// </summary>
    /// <param name="parameters">The new parameter values to use.</param>
    /// <returns>A new expression tree with the updated parameters.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This replaces all the constant numbers in your formula 
    /// with new values. For example, changing "2x + 3" to "4x + 1" by providing [4, 1]
    /// as the new parameters. The structure of the formula stays the same.
    /// </remarks>
    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        // This is equivalent to UpdateCoefficients
        return UpdateCoefficients(parameters);
    }

    /// <summary>
    /// Gets the indices of all features (variables) used in this expression tree.
    /// </summary>
    /// <returns>A collection of feature indices.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you which input variables are actually used in your formula.
    /// For example, if your formula only uses x[0] and x[2], this returns [0, 2], showing that
    /// the formula uses the first and third variables but not the second one.
    /// </remarks>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        HashSet<int> activeIndices = new();

        void CollectFeatureIndices(ExpressionTree<T, TInput, TOutput> node)
        {
            if (node.Type == ExpressionNodeType.Variable)
            {
                activeIndices.Add(_numOps.ToInt32(node.Value));
            }

            if (node.Left != null)
            {
                CollectFeatureIndices(node.Left);
            }

            if (node.Right != null)
            {
                CollectFeatureIndices(node.Right);
            }
        }

        CollectFeatureIndices(this);
        return activeIndices;
    }

    /// <summary>
    /// Gets the feature importance scores for this expression tree.
    /// </summary>
    /// <returns>A dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Feature importance tells you which input variables matter most in your formula.
    /// For expression trees, importance is calculated by counting how many times each variable appears in the formula.
    /// Variables that appear more frequently are considered more important.
    /// </remarks>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        // Count occurrences of each feature in the tree
        Dictionary<int, int> featureCounts = new();

        void CountFeatureOccurrences(ExpressionTree<T, TInput, TOutput> node)
        {
            if (node == null) return;

            if (node.Type == ExpressionNodeType.Variable)
            {
                int featureIndex = _numOps.ToInt32(node.Value);
                if (featureCounts.ContainsKey(featureIndex))
                {
                    featureCounts[featureIndex]++;
                }
                else
                {
                    featureCounts[featureIndex] = 1;
                }
            }

            if (node.Left != null)
            {
                CountFeatureOccurrences(node.Left);
            }

            if (node.Right != null)
            {
                CountFeatureOccurrences(node.Right);
            }
        }

        CountFeatureOccurrences(this);

        // Convert counts to importance scores (normalized by total occurrences)
        int totalCount = 0;
        foreach (var count in featureCounts.Values)
        {
            totalCount += count;
        }

        Dictionary<string, T> importance = new();
        if (totalCount > 0)
        {
            foreach (var kvp in featureCounts)
            {
                string featureName = $"x[{kvp.Key}]";
                double normalizedImportance = (double)kvp.Value / totalCount;
                importance[featureName] = _numOps.FromDouble(normalizedImportance);
            }
        }

        return importance;
    }

    /// <summary>
    /// Sets the active feature indices for this expression tree.
    /// </summary>
    /// <param name="featureIndices">The feature indices to use.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This restricts the formula to only use specific input variables.
    /// Any variables in the tree that are not in the active set will be replaced with constant zero values.
    /// This is useful for feature selection and understanding which variables are most important.
    /// </remarks>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices));
        }

        HashSet<int> activeSet = new(featureIndices);

        void DeactivateInactiveFeatures(ExpressionTree<T, TInput, TOutput> node)
        {
            if (node == null) return;

            // If this is a variable node and it's not in the active set, replace it with zero
            if (node.Type == ExpressionNodeType.Variable)
            {
                int featureIndex = _numOps.ToInt32(node.Value);
                if (!activeSet.Contains(featureIndex))
                {
                    node.SetType(ExpressionNodeType.Constant);
                    node.SetValue(_numOps.Zero);
                }
            }

            // Recursively process children
            if (node.Left != null)
            {
                DeactivateInactiveFeatures(node.Left);
            }

            if (node.Right != null)
            {
                DeactivateInactiveFeatures(node.Right);
            }
        }

        DeactivateInactiveFeatures(this);

        // Clear the cached feature count since we've modified the tree
        _featureCount = 0;
    }

    /// <summary>
    /// Trains the expression tree on a single input-output pair.
    /// </summary>
    /// <param name="input">The input data (Vector, Matrix, or Tensor).</param>
    /// <param name="expectedOutput">The expected output value.</param>
    /// <remarks>
    /// <b>For Beginners:</b> For expression trees, training doesn't actually change the formula.
    /// This method validates that the formula can process your input data correctly.
    /// </remarks>
    public void Train(TInput input, TOutput expectedOutput)
    {
        // For expression trees, we primarily validate input compatibility
        if (input is Matrix<T> matrix)
        {
            ValidateMatrixFeatures(matrix);
        }
        else if (input is Vector<T> vector)
        {
            ValidateVectorFeatures(vector);
        }
        else if (input is Tensor<T> tensor)
        {
            ValidateTensorFeatures(tensor);
        }
        else
        {
            throw new ArgumentException($"Unsupported input type: {input?.GetType().Name ?? "null"}. Expected Matrix<T>, Vector<T>, or Tensor<T>.");
        }
    }

    /// <summary>
    /// Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="target">The target/expected output.</param>
    /// <param name="lossFunction">The loss function to use. If null, uses the model's default loss function.</param>
    /// <returns>A vector containing gradients with respect to all model parameters (constants in the expression tree).</returns>
    /// <exception cref="ArgumentNullException">If input or target is null.</exception>
    /// <remarks>
    /// <para>
    /// This method computes gradients using numerical differentiation (finite differences).
    /// For each constant in the expression tree, it slightly perturbs the value and
    /// measures how the loss changes, approximating the gradient.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This calculates how to adjust each constant in your mathematical formula to reduce error.
    /// Since expression trees are symbolic, we use a numerical approximation:
    /// we slightly change each constant and see how much the error changes.
    /// </para>
    /// </remarks>
    public Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (target == null)
            throw new ArgumentNullException(nameof(target));

        var loss = lossFunction ?? DefaultLossFunction;
        var parameters = Coefficients;
        var gradients = new Vector<T>(parameters.Length);

        // Small epsilon for finite differences
        T epsilon = _numOps.FromDouble(1e-7);

        // Compute loss at current parameters
        var currentPrediction = Predict(input);
        Vector<T> currentPredVec = ConvertOutputToVector(currentPrediction);
        Vector<T> targetVec = ConvertOutputToVector(target);
        T currentLoss = loss.CalculateLoss(currentPredVec, targetVec);

        // Compute gradient for each parameter using finite differences
        for (int i = 0; i < parameters.Length; i++)
        {
            // Save original value
            T originalValue = parameters[i];

            // Perturb parameter forward
            parameters[i] = _numOps.Add(originalValue, epsilon);
            SetParameters(parameters);  // ✅ CRITICAL: Use SetParameters, NOT UpdateCoefficients
            var forwardPrediction = Predict(input);
            Vector<T> forwardPredVec = ConvertOutputToVector(forwardPrediction);
            T forwardLoss = loss.CalculateLoss(forwardPredVec, targetVec);

            // Compute gradient: (f(x+ε) - f(x)) / ε
            T gradient = _numOps.Divide(_numOps.Subtract(forwardLoss, currentLoss), epsilon);
            gradients[i] = gradient;

            // Restore original value
            parameters[i] = originalValue;
            SetParameters(parameters);  // ✅ Restore original
        }

        return gradients;
    }

    /// <summary>
    /// Applies pre-computed gradients to update the model parameters (constants in the expression tree).
    /// </summary>
    /// <param name="gradients">The gradient vector to apply.</param>
    /// <param name="learningRate">The learning rate for the update.</param>
    /// <exception cref="ArgumentNullException">If gradients is null.</exception>
    /// <exception cref="ArgumentException">If gradient vector length doesn't match parameter count.</exception>
    /// <remarks>
    /// <para>
    /// Updates constants using: constant = constant - learningRate * gradient
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After computing gradients (seeing which direction to adjust each constant),
    /// this method actually adjusts them. The learning rate controls how big of an adjustment to make.
    /// </para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        var parameters = Coefficients;

        if (gradients.Length != parameters.Length)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match parameter count ({parameters.Length})",
                nameof(gradients));
        }

        // Apply gradient descent: params = params - learningRate * gradients
        for (int i = 0; i < parameters.Length; i++)
        {
            T update = _numOps.Multiply(learningRate, gradients[i]);
            parameters[i] = _numOps.Subtract(parameters[i], update);
        }

        SetParameters(parameters);
    }

    /// <summary>
    /// Helper method to convert output to Vector<T> for loss computation.
    /// </summary>
    private Vector<T> ConvertOutputToVector(TOutput output)
    {
        if (output is Vector<T> vec)
            return vec;
        if (output is T scalar)
            return new Vector<T>(new[] { scalar });
        if (output is Matrix<T> mat)
            return mat.ToVector();

        // Try to convert using reflection if needed
        throw new InvalidOperationException($"Cannot convert output of type {typeof(TOutput).Name} to Vector<T> for gradient computation.");
    }

    /// <summary>
    /// Makes a prediction for an input example.
    /// </summary>
    /// <param name="input">The input data (Vector, Matrix, or Tensor).</param>
    /// <returns>The predicted output.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method applies your mathematical formula to the input data
    /// to calculate a prediction. It handles different types of inputs (vectors, matrices, or tensors).
    /// </remarks>
    public TOutput Predict(TInput input)
    {
        if (input is Matrix<T> matrix)
        {
            ValidateMatrixFeatures(matrix);
            Vector<T> predictions = PredictMatrix(matrix);

            // Try to convert the result to TOutput
            if (predictions is TOutput typedResult)
            {
                return typedResult;
            }
            else if (typeof(TOutput) == typeof(object))
            {
                return (TOutput)(object)predictions;
            }

            throw new InvalidOperationException($"Cannot convert prediction vector to {typeof(TOutput).Name}.");
        }
        else if (input is Tensor<T> tensor)
        {
            ValidateTensorFeatures(tensor);
            Vector<T> predictions = PredictTensor(tensor);

            // Try to convert the result to TOutput
            if (predictions is TOutput typedResult)
            {
                return typedResult;
            }
            else if (typeof(TOutput) == typeof(object))
            {
                return (TOutput)(object)predictions;
            }

            throw new InvalidOperationException($"Cannot convert prediction vector to {typeof(TOutput).Name}.");
        }

        throw new ArgumentException($"Unsupported input type: {input?.GetType().Name ?? "null"}. Expected Matrix<T>, Vector<T>, or Tensor<T>.");
    }

    /// <summary>
    /// Validates that a matrix has compatible features for this expression tree.
    /// </summary>
    /// <param name="matrix">The matrix to validate.</param>
    private void ValidateMatrixFeatures(Matrix<T> matrix)
    {
        if (matrix.Columns < FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {matrix.Columns} columns, but the model requires at least {FeatureCount} features.");
        }
    }

    /// <summary>
    /// Validates that a vector has compatible features for this expression tree.
    /// </summary>
    /// <param name="vector">The vector to validate.</param>
    private void ValidateVectorFeatures(Vector<T> vector)
    {
        if (vector.Length < FeatureCount)
        {
            throw new ArgumentException($"Input vector has {vector.Length} elements, but the model requires at least {FeatureCount} features.");
        }
    }

    /// <summary>
    /// Validates that a tensor has compatible features for this expression tree.
    /// </summary>
    /// <param name="tensor">The tensor to validate.</param>
    private void ValidateTensorFeatures(Tensor<T> tensor)
    {
        if (tensor.Shape.Length < 1 || tensor.Shape[tensor.Shape.Length - 1] < FeatureCount)
        {
            throw new ArgumentException($"Input tensor's last dimension is {(tensor.Shape.Length > 0 ? tensor.Shape[tensor.Shape.Length - 1] : 0)}, " +
                $"but the model requires at least {FeatureCount} features.");
        }
    }

    /// <summary>
    /// Makes predictions for all rows in a matrix.
    /// </summary>
    /// <param name="matrix">The input matrix, where each row is a sample.</param>
    /// <returns>A vector containing predictions for each row.</returns>
    private Vector<T> PredictMatrix(Matrix<T> matrix)
    {
        Vector<T> predictions = new Vector<T>(matrix.Rows);
        for (int i = 0; i < matrix.Rows; i++)
        {
            predictions[i] = Evaluate(matrix.GetRow(i));
        }
        return predictions;
    }

    /// <summary>
    /// Makes predictions for all samples in a tensor.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A vector containing predictions for each sample.</returns>
    private Vector<T> PredictTensor(Tensor<T> tensor)
    {
        // Calculate the batch size (product of all dimensions except the last one)
        int batchSize = 1;
        for (int i = 0; i < tensor.Shape.Length - 1; i++)
        {
            batchSize *= tensor.Shape[i];
        }

        Vector<T> predictions = new Vector<T>(batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            // Extract vector for this batch item using the Flatten and Slice methods
            // First, flatten the tensor then extract the appropriate slice
            Vector<T> flatTensor = tensor.ToVector();
            int featureSize = tensor.Shape[tensor.Shape.Length - 1];
            int startIndex = i * featureSize;

            // Create a vector from the slice
            Vector<T> inputVector = new Vector<T>(featureSize);
            for (int j = 0; j < featureSize; j++)
            {
                inputVector[j] = flatTensor[startIndex + j];
            }

            predictions[i] = Evaluate(inputVector);
        }

        return predictions;
    }

    /// <summary>
    /// Gets a vector containing all coefficient values in this expression tree.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This collects all the constant numbers from your formula into a list.
    /// For example, if your formula is "2x + 3y + 5", this would give you [2, 3, 5].
    /// These numbers are called "coefficients" and are important when optimizing your model.
    /// </remarks>
    public Vector<T> Coefficients
    {
        get
        {
            List<T> coefficients = new List<T>();

            void CollectCoefficients(ExpressionTree<T, TInput, TOutput> node)
            {
                if (node.Type == ExpressionNodeType.Constant)
                {
                    coefficients.Add(node.Value);
                }
                if (node.Left != null)
                {
                    CollectCoefficients(node.Left);
                }
                if (node.Right != null)
                {
                    CollectCoefficients(node.Right);
                }
            }

            CollectCoefficients(this);
            return new Vector<T>(coefficients.ToArray());
        }
    }

    /// <summary>
    /// Sets the parameters (constant values) of this expression tree, modifying it in place.
    /// </summary>
    /// <param name="parameters">The new parameter values to assign to constant nodes.</param>
    /// <exception cref="ArgumentException">Thrown when the parameter count doesn't match the number of constant nodes.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This method replaces all the constant numbers in your formula with new values,
    /// modifying the current tree directly. Unlike UpdateCoefficients and WithParameters which create new
    /// trees with the updated values, this method mutates the tree in place. Use this when you want to
    /// modify the tree directly, such as during optimization iterations.
    /// <para>
    /// <b>Note:</b> This implementation uses two tree traversals (counting and assignment)
    /// to validate parameter count BEFORE modifying the tree. This ensures atomicity:
    /// if the parameter count is wrong, the tree remains unchanged.
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        // Count the number of constant nodes in the tree
        int constantNodeCount = 0;

        // Local function to count constant nodes in the tree via recursive traversal
        void CountConstants(ExpressionTree<T, TInput, TOutput>? node)
        {
            if (node == null)
                return;
            if (node.Type == ExpressionNodeType.Constant)
            {
                constantNodeCount++;
            }
            if (node.Left != null) CountConstants(node.Left);
            if (node.Right != null) CountConstants(node.Right);
        }

        CountConstants(this);

        if (parameters.Length != constantNodeCount)
        {
            throw new ArgumentException(
                $"Parameter count mismatch: expected {constantNodeCount} parameters (one for each constant node), but got {parameters.Length}.",
                nameof(parameters));
        }

        // Assign parameter values to constant nodes in a deterministic traversal order
        // Local function returns next index to use - includes null check for safety
        int AssignAndReturnNextIndex(ExpressionTree<T, TInput, TOutput>? node, int currentIndex)
        {
            if (node == null)
                return currentIndex;

            int nextIndex = currentIndex;
            if (node.Type == ExpressionNodeType.Constant)
            {
                node.SetValue(parameters[nextIndex]);
                nextIndex++;
            }

            if (node.Left != null)
                nextIndex = AssignAndReturnNextIndex(node.Left, nextIndex);
            if (node.Right != null)
                nextIndex = AssignAndReturnNextIndex(node.Right, nextIndex);

            return nextIndex;
        }

        int finalIndex = AssignAndReturnNextIndex(this, 0);

        // Validate that all parameters were consumed during assignment
        // This catches any discrepancy between counting and assignment traversals
        if (finalIndex != parameters.Length)
        {
            throw new InvalidOperationException(
                $"Internal error: expected to consume {parameters.Length} parameters, but only consumed {finalIndex}. " +
                "This indicates a mismatch between counting and assignment traversals.");
        }
    }

    /// <summary>
    /// Gets the number of parameters (constant nodes) in this expression tree.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many constant values are in your formula.
    /// For example, if your formula is "2x + 3y + 5", there are 3 parameters: 2, 3, and 5.
    /// This value is obtained from the Coefficients property, which returns a vector of all constant values.
    /// </remarks>
    public virtual int ParameterCount
    {
        get
        {
            int CountConstants(ExpressionTree<T, TInput, TOutput>? node)
            {
                if (node == null) return 0;
                int count = node.Type == ExpressionNodeType.Constant ? 1 : 0;
                count += CountConstants(node.Left);
                count += CountConstants(node.Right);
                return count;
            }
            return CountConstants(this);
        }
    }

    /// <summary>
    /// Saves the expression tree model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This saves your mathematical formula to a file so you can load it later
    /// without having to recreate it. The file contains the tree structure, all node types, and values.
    /// </remarks>
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Loads an expression tree model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This loads a previously saved formula from a file, allowing you to
    /// reuse it without recreating it. The loaded formula can immediately be used for predictions.
    /// </remarks>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);
        Deserialize(serializedData);
    }

    /// <summary>
    /// Saves the expression tree's current state (structure and values) to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the expression tree state to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the complete expression tree structure, including all node types,
    /// values, and connections. It uses the existing Serialize method and writes the data
    /// to the provided stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a snapshot of your mathematical formula.
    ///
    /// When you call SaveState:
    /// - The entire tree structure is written to the stream
    /// - All node types (constants, variables, operations) are preserved
    /// - All values and connections are saved
    ///
    /// This is particularly useful for:
    /// - Checkpointing during evolutionary algorithm training
    /// - Knowledge distillation with symbolic models
    /// - Saving the best formula found during optimization
    /// - Creating formula ensembles
    ///
    /// You can later use LoadState to restore the formula to this exact state.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        try
        {
            var data = this.Serialize();
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to save expression tree state to stream: {ex.Message}", ex);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unexpected error while saving expression tree state: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the expression tree's state (structure and values) from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the expression tree state from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes expression tree state that was previously saved with SaveState,
    /// restoring the complete tree structure, node types, values, and connections.
    /// It uses the existing Deserialize method after reading data from the stream.
    /// </para>
    /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your mathematical formula.
    ///
    /// When you call LoadState:
    /// - The tree structure is read from the stream
    /// - All node types and values are restored
    /// - The formula becomes identical to when SaveState was called
    ///
    /// After loading, the formula can:
    /// - Make predictions using the restored structure
    /// - Continue evolving during optimization
    /// - Be used for symbolic regression or genetic programming
    ///
    /// This is essential for:
    /// - Resuming interrupted evolutionary training
    /// - Loading the best formula after optimization
    /// - Deploying symbolic models to production
    /// - Knowledge distillation with interpretable models
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
    /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data.</exception>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        try
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            if (data.Length == 0)
                throw new InvalidOperationException("Stream contains no data.");

            this.Deserialize(data);
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to read expression tree state from stream: {ex.Message}", ex);
        }
        catch (InvalidOperationException)
        {
            // Re-throw InvalidOperationException from Deserialize
            throw;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to deserialize expression tree state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
        }
    }

    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether this expression tree supports JIT compilation.
    /// </summary>
    /// <value>True - expression trees are inherently computation graphs and support JIT compilation.</value>
    /// <remarks>
    /// <para>
    /// Expression trees are already symbolic computation graphs, making them ideal for JIT compilation.
    /// The tree structure directly represents the mathematical operations to be performed,
    /// which can be compiled into optimized native code.
    /// </para>
    /// <para><b>For Beginners:</b> Expression trees are like ready-made recipes for JIT compilation.
    ///
    /// Since an expression tree already describes your formula as a series of operations
    /// (add, multiply, etc.), the JIT compiler can:
    /// - Convert it directly to fast machine code
    /// - Optimize common patterns (e.g., constant folding)
    /// - Inline operations for better performance
    ///
    /// This provides 2-5x speedup for complex symbolic expressions.
    /// </para>
    /// </remarks>
    public bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the expression tree as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes (variables and constants).</param>
    /// <returns>The root computation node representing the complete expression.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the expression tree into a computation graph by:
    /// 1. Creating variable nodes for each unique variable in the tree
    /// 2. Recursively building the computation graph from the tree structure
    /// 3. Adding all input nodes (variables) to the inputNodes list
    /// </para>
    /// <para><b>For Beginners:</b> This converts your symbolic formula into a computation graph.
    ///
    /// For example, the expression tree representing "(x[0] * 2) + x[1]" becomes:
    /// - Variable node for x[0]
    /// - Constant node for 2
    /// - Multiply node connecting them
    /// - Variable node for x[1]
    /// - Add node combining the multiply result with x[1]
    ///
    /// The JIT compiler then optimizes this graph and generates fast code.
    ///
    /// <b>Note:</b> Only variables are added to inputNodes. Constants are embedded in the graph.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create a mapping from variable indices to their computation nodes
        var variableNodes = new Dictionary<int, ComputationNode<T>>();

        // Recursively build the computation graph
        var outputNode = BuildComputationGraph(this, variableNodes);

        // Add all variable nodes to inputNodes in sorted order for consistency
        foreach (var kvp in variableNodes.OrderBy(x => x.Key))
        {
            inputNodes.Add(kvp.Value);
        }

        return outputNode;
    }

    /// <summary>
    /// Recursively builds a computation graph from an expression tree node.
    /// </summary>
    /// <param name="node">The expression tree node to convert.</param>
    /// <param name="variableNodes">Dictionary mapping variable indices to their computation nodes.</param>
    /// <returns>The computation node representing this expression tree node.</returns>
    private ComputationNode<T> BuildComputationGraph(
        ExpressionTree<T, TInput, TOutput> node,
        Dictionary<int, ComputationNode<T>> variableNodes)
    {
        switch (node.Type)
        {
            case ExpressionNodeType.Constant:
                // Create a constant tensor (scalar)
                var constantTensor = new Tensor<T>(new[] { 1 });
                constantTensor[0] = node.Value;
                return new ComputationNode<T>(constantTensor);

            case ExpressionNodeType.Variable:
                // Get or create variable node
                int varIndex = _numOps.ToInt32(node.Value);
                if (!variableNodes.ContainsKey(varIndex))
                {
                    // Create placeholder for this variable
                    var varTensor = new Tensor<T>(new[] { 1 });
                    varTensor[0] = _numOps.Zero;  // Placeholder value
                    variableNodes[varIndex] = new ComputationNode<T>(varTensor);
                }
                return variableNodes[varIndex];

            case ExpressionNodeType.Add:
                if (node.Left == null || node.Right == null)
                    throw new InvalidOperationException("Add operation requires both left and right operands.");
                return TensorOperations<T>.Add(
                    BuildComputationGraph(node.Left, variableNodes),
                    BuildComputationGraph(node.Right, variableNodes));

            case ExpressionNodeType.Subtract:
                if (node.Left == null || node.Right == null)
                    throw new InvalidOperationException("Subtract operation requires both left and right operands.");
                return TensorOperations<T>.Subtract(
                    BuildComputationGraph(node.Left, variableNodes),
                    BuildComputationGraph(node.Right, variableNodes));

            case ExpressionNodeType.Multiply:
                if (node.Left == null || node.Right == null)
                    throw new InvalidOperationException("Multiply operation requires both left and right operands.");
                return TensorOperations<T>.ElementwiseMultiply(
                    BuildComputationGraph(node.Left, variableNodes),
                    BuildComputationGraph(node.Right, variableNodes));

            case ExpressionNodeType.Divide:
                if (node.Left == null || node.Right == null)
                    throw new InvalidOperationException("Divide operation requires both left and right operands.");
                return TensorOperations<T>.Divide(
                    BuildComputationGraph(node.Left, variableNodes),
                    BuildComputationGraph(node.Right, variableNodes));

            default:
                throw new InvalidOperationException($"Unknown expression node type: {node.Type}");
        }
    }

    #endregion
}
