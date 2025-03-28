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
public class ExpressionTree<T> : ISymbolicModel<T>
{
    /// <summary>
    /// Gets the type of this node (constant, variable, or operation).
    /// </summary>
    public NodeType Type { get; private set; }
    
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
    public ExpressionTree<T>? Left { get; private set; }
    
    /// <summary>
    /// Gets the right child node of this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> In operations like addition (a + b), the right child represents 'b'.
    /// </remarks>
    public ExpressionTree<T>? Right { get; private set; }
    
    /// <summary>
    /// Gets the parent node of this node.
    /// </summary>
    public ExpressionTree<T>? Parent { get; private set; }

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
    public void SetType(NodeType type)
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
    public void SetLeft(ExpressionTree<T>? left)
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
    public void SetRight(ExpressionTree<T>? right)
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
            NodeType.Constant => Value?.ToString(),
            NodeType.Variable => $"x[{Value}]",
            NodeType.Add => $"({Left} + {Right})",
            NodeType.Subtract => $"({Left} - {Right})",
            NodeType.Multiply => $"({Left} * {Right})",
            NodeType.Divide => $"({Left} / {Right})",
            _ => throw new ArgumentException("Invalid node type"),
        } ?? string.Empty;
    }

    /// <summary>
    /// Helper object for performing numeric operations on type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new expression tree node with the specified properties.
    /// </summary>
    /// <param name="type">The type of node to create.</param>
    /// <param name="value">The value for this node (for constants and variables).</param>
    /// <param name="left">The left child node.</param>
    /// <param name="right">The right child node.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new part of your mathematical formula.
    /// You can create simple nodes (like numbers or variables) or operation nodes
    /// (like addition or multiplication) that connect to other nodes.
    /// </remarks>
    public ExpressionTree(NodeType type, T? value = default, ExpressionTree<T>? left = null, ExpressionTree<T>? right = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        Type = type;
        Value = value ?? _numOps.Zero;
        Left = left;
        Right = right;
    }

    /// <summary>
    /// Cached count of features used in this expression tree.
    /// </summary>
    private int _featureCount;

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
    /// Calculates the number of features used in this expression tree.
    /// </summary>
    /// <returns>The number of features used.</returns>
    private int CalculateFeatureCount()
    {
        return CalculateFeatureCountRecursive(this);
    }

    /// <summary>
    /// Recursively calculates the number of features used in a node and its children.
    /// </summary>
    /// <param name="node">The node to check.</param>
    /// <returns>The number of features used.</returns>
    private int CalculateFeatureCountRecursive(ExpressionTree<T> node)
    {
        if (node.Type == NodeType.Variable)
        {
            return _numOps.ToInt32(node.Value) + 1; // Add 1 because feature indices are 0-based
        }

        int leftCount = node.Left != null ? CalculateFeatureCountRecursive(node.Left) : 0;
        int rightCount = node.Right != null ? CalculateFeatureCountRecursive(node.Right) : 0;

        return Math.Max(leftCount, rightCount);
    }

    /// <summary>
    /// Recursively checks if a specific feature is used in a node or its children.
    /// </summary>
    /// <param name="node">The node to check.</param>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used, false otherwise.</returns>
    private bool IsFeatureUsedRecursive(ExpressionTree<T> node, int featureIndex)
    {
        if (node.Type == NodeType.Variable && _numOps.ToInt32(node.Value) == featureIndex)
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
            NodeType.Constant => Value,
            NodeType.Variable => input[_numOps.ToInt32(Value)],
            NodeType.Add => _numOps.Add(Left!.Evaluate(input), Right!.Evaluate(input)),
            NodeType.Subtract => _numOps.Subtract(Left!.Evaluate(input), Right!.Evaluate(input)),
            NodeType.Multiply => _numOps.Multiply(Left!.Evaluate(input), Right!.Evaluate(input)),
            NodeType.Divide => _numOps.Divide(Left!.Evaluate(input), Right!.Evaluate(input)),
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
    public ExpressionTree<T> Deserialize(BinaryReader reader)
    {
        NodeType type = (NodeType)reader.ReadInt32();
        T value = _numOps.FromDouble(reader.ReadDouble());
        bool hasLeft = reader.ReadBoolean();
        ExpressionTree<T>? left = hasLeft ? Deserialize(reader) : null;
        bool hasRight = reader.ReadBoolean();
        ExpressionTree<T>? right = hasRight ? Deserialize(reader) : null;

        return new ExpressionTree<T>(type, value, left, right);
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
    public ISymbolicModel<T> Mutate(double mutationRate)
    {
        ExpressionTree<T> mutatedTree = (ExpressionTree<T>)Copy();
        Random random = new Random();

        if (random.NextDouble() < mutationRate)
        {
            switch (random.Next(3))
            {
                case 0: // Change node type
                    mutatedTree.Type = (NodeType)random.Next(Enum.GetValues(typeof(NodeType)).Length);
                    break;
                case 1: // Change value (for Constant or Variable nodes)
                    if (mutatedTree.Type == NodeType.Constant)
                    {
                        mutatedTree.Value = _numOps.FromDouble(random.NextDouble() * 10 - 5); // Random value between -5 and 5
                    }
                    else if (mutatedTree.Type == NodeType.Variable)
                    {
                        mutatedTree.Value = _numOps.FromDouble(random.Next(10)); // Assume max 10 variables
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
            mutatedTree.Left = (ExpressionTree<T>)mutatedTree.Left.Mutate(mutationRate);
        }
        if (mutatedTree.Right != null)
        {
            mutatedTree.Right = (ExpressionTree<T>)mutatedTree.Right.Mutate(mutationRate);
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
    public ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate)
    {
        if (!(other is ExpressionTree<T> otherTree))
        {
            throw new ArgumentException("Crossover can only be performed with another ExpressionTree.");
        }

        ExpressionTree<T> offspring = (ExpressionTree<T>)Copy();
        Random random = new Random();

        if (random.NextDouble() < crossoverRate)
        {
            // Select a random subtree from the other parent
            ExpressionTree<T> selectedSubtree = SelectRandomSubtree(otherTree);

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
    public ISymbolicModel<T> Copy()
    {
        return new ExpressionTree<T>(
            Type,
            Value,
            Left?.Copy() as ExpressionTree<T>,
            Right?.Copy() as ExpressionTree<T>
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
    private ExpressionTree<T> GenerateRandomTree(int maxDepth)
    {
        Random random = new Random();
        if (maxDepth == 0 || random.NextDouble() < 0.3) // 30% chance of leaf node
        {
            if (random.NextDouble() < 0.5)
            {
                return new ExpressionTree<T>(NodeType.Constant, _numOps.FromDouble(random.NextDouble() * 10 - 5));
            }
            else
            {
                return new ExpressionTree<T>(NodeType.Variable, _numOps.FromDouble(random.Next(10)));
            }
        }
        else
        {
            NodeType operationType = (NodeType)random.Next(2, 6); // Add, Subtract, Multiply, or Divide
            return new ExpressionTree<T>(
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
    private ExpressionTree<T> SelectRandomSubtree(ExpressionTree<T> tree)
    {
        Random random = new Random();
        if (tree.Left == null && tree.Right == null)
        {
            return tree;
        }
        else if (random.NextDouble() < 0.3) // 30% chance of selecting current node
        {
            return tree;
        }
        else
        {
            if (tree.Left != null && (tree.Right == null || random.NextDouble() < 0.5))
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
    private void ReplaceRandomSubtree(ExpressionTree<T> tree, ExpressionTree<T> replacement)
    {
        Random random = new Random();
        if (random.NextDouble() < 0.3) // 30% chance of replacing current node
        {
            tree.Type = replacement.Type;
            tree.Value = replacement.Value;
            tree.Left = replacement.Left?.Copy() as ExpressionTree<T>;
            tree.Right = replacement.Right?.Copy() as ExpressionTree<T>;
        }
        else
        {
            if (tree.Left != null && (tree.Right == null || random.NextDouble() < 0.5))
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
        if (x.Columns != FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {x.Columns} columns, but the model expects {FeatureCount} features.");
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
    /// </remarks>
    public Vector<T> Predict(Matrix<T> input)
    {
        if (input.Columns != FeatureCount)
        {
            throw new ArgumentException($"Input matrix has {input.Columns} columns, but the model expects {FeatureCount} features.");
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
        ExpressionTree<T> deserializedTree = Deserialize(reader);
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
    public List<ExpressionTree<T>> GetAllNodes()
    {
        var nodes = new List<ExpressionTree<T>>();
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
    private void CollectNodes(ExpressionTree<T>? node, List<ExpressionTree<T>> nodes)
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
    public ExpressionTree<T>? FindNodeById(int id)
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
    public ISymbolicModel<T> UpdateCoefficients(Vector<T> newCoefficients)
    {
        if (newCoefficients.Length != this.Coefficients.Length)
        {
            throw new ArgumentException($"The number of new coefficients ({newCoefficients.Length}) must match the current number of coefficients ({this.Coefficients.Length}).");
        }

        ExpressionTree<T> updatedTree = (ExpressionTree<T>)this.Copy();
        int coefficientIndex = 0;

        void UpdateConstantNodes(ExpressionTree<T> node)
        {
            if (node.Type == NodeType.Constant)
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

            void CollectCoefficients(ExpressionTree<T> node)
            {
                if (node.Type == NodeType.Constant)
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
}