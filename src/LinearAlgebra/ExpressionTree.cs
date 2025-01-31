namespace AiDotNet.LinearAlgebra;

public class ExpressionTree<T> : ISymbolicModel<T>
{
    public NodeType Type { get; private set; }
    public T Value { get; private set; }
    public ExpressionTree<T>? Left { get; private set; }
    public ExpressionTree<T>? Right { get; private set; }
    public ExpressionTree<T>? Parent { get; private set; }

    public int Complexity => 1 + (Left?.Complexity ?? 0) + (Right?.Complexity ?? 0);

    public void SetType(NodeType type)
    {
        Type = type;
    }

    public void SetValue(T value)
    {
        Value = value;
    }

    public void SetLeft(ExpressionTree<T>? left)
    {
        Left = left;
        if (left != null)
        {
            left.Parent = this;
        }
    }

    public void SetRight(ExpressionTree<T>? right)
    {
        Right = right;
        if (right != null)
        {
            right.Parent = this;
        }
    }

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

    private readonly INumericOperations<T> NumOps;

    public ExpressionTree(NodeType type, T? value = default, ExpressionTree<T>? left = null, ExpressionTree<T>? right = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Type = type;
        Value = value ?? NumOps.Zero;
        Left = left;
        Right = right;
    }

    private int _featureCount;

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

    public bool IsFeatureUsed(int featureIndex)
    {
        return IsFeatureUsedRecursive(this, featureIndex);
    }

    private int CalculateFeatureCount()
    {
        return CalculateFeatureCountRecursive(this);
    }

    private int CalculateFeatureCountRecursive(ExpressionTree<T> node)
    {
        if (node.Type == NodeType.Variable)
        {
            return NumOps.ToInt32(node.Value) + 1; // Add 1 because feature indices are 0-based
        }

        int leftCount = node.Left != null ? CalculateFeatureCountRecursive(node.Left) : 0;
        int rightCount = node.Right != null ? CalculateFeatureCountRecursive(node.Right) : 0;

        return Math.Max(leftCount, rightCount);
    }

    private bool IsFeatureUsedRecursive(ExpressionTree<T> node, int featureIndex)
    {
        if (node.Type == NodeType.Variable && NumOps.ToInt32(node.Value) == featureIndex)
        {
            return true;
        }

        bool leftUsed = node.Left != null && IsFeatureUsedRecursive(node.Left, featureIndex);
        bool rightUsed = node.Right != null && IsFeatureUsedRecursive(node.Right, featureIndex);

        return leftUsed || rightUsed;
    }

    public T Evaluate(Vector<T> input)
    {
        return Type switch
        {
            NodeType.Constant => Value,
            NodeType.Variable => input[NumOps.ToInt32(Value)],
            NodeType.Add => NumOps.Add(Left!.Evaluate(input), Right!.Evaluate(input)),
            NodeType.Subtract => NumOps.Subtract(Left!.Evaluate(input), Right!.Evaluate(input)),
            NodeType.Multiply => NumOps.Multiply(Left!.Evaluate(input), Right!.Evaluate(input)),
            NodeType.Divide => NumOps.Divide(Left!.Evaluate(input), Right!.Evaluate(input)),
            _ => throw new ArgumentException("Invalid node type"),
        };
    }

    public void Serialize(BinaryWriter writer)
    {
        writer.Write((int)Type);
        writer.Write(Convert.ToDouble(Value));
        writer.Write(Left != null);
        Left?.Serialize(writer);
        writer.Write(Right != null);
        Right?.Serialize(writer);
    }

    public ExpressionTree<T> Deserialize(BinaryReader reader)
    {
        NodeType type = (NodeType)reader.ReadInt32();
        T value = NumOps.FromDouble(reader.ReadDouble());
        bool hasLeft = reader.ReadBoolean();
        ExpressionTree<T>? left = hasLeft ? Deserialize(reader) : null;
        bool hasRight = reader.ReadBoolean();
        ExpressionTree<T>? right = hasRight ? Deserialize(reader) : null;

        return new ExpressionTree<T>(type, value, left, right);
    }

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
                        mutatedTree.Value = NumOps.FromDouble(random.NextDouble() * 10 - 5); // Random value between -5 and 5
                    }
                    else if (mutatedTree.Type == NodeType.Variable)
                    {
                        mutatedTree.Value = NumOps.FromDouble(random.Next(10)); // Assume max 10 variables
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

    public ISymbolicModel<T> Copy()
    {
        return new ExpressionTree<T>(
            Type,
            Value,
            Left?.Copy() as ExpressionTree<T>,
            Right?.Copy() as ExpressionTree<T>
        );
    }

    private ExpressionTree<T> GenerateRandomTree(int maxDepth)
    {
        Random random = new Random();
        if (maxDepth == 0 || random.NextDouble() < 0.3) // 30% chance of leaf node
        {
            if (random.NextDouble() < 0.5)
            {
                return new ExpressionTree<T>(NodeType.Constant, NumOps.FromDouble(random.NextDouble() * 10 - 5));
            }
            else
            {
                return new ExpressionTree<T>(NodeType.Variable, NumOps.FromDouble(random.Next(10)));
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

    public void Fit(Matrix<T> X, Vector<T> y)
    {
        // For ExpressionTree, Fit is the same as Train
        Train(X, y);
    }

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

    public byte[] Serialize()
    {
        using MemoryStream ms = new();
        using BinaryWriter writer = new(ms);
        Serialize(writer);

        return ms.ToArray();
    }

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

    public List<ExpressionTree<T>> GetAllNodes()
    {
        var nodes = new List<ExpressionTree<T>>();
        CollectNodes(this, nodes);
        return nodes;
    }

    private void CollectNodes(ExpressionTree<T>? node, List<ExpressionTree<T>> nodes)
    {
        if (node == null) return;
        nodes.Add(node);
        CollectNodes(node.Left, nodes);
        CollectNodes(node.Right, nodes);
    }

    public ExpressionTree<T>? FindNodeById(int id)
    {
        return GetAllNodes().FirstOrDefault(n => n.Id == id);
    }

    public int Id { get; } = Interlocked.Increment(ref _nextId);

    private static int _nextId;

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