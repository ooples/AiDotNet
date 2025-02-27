namespace AiDotNet.Regression;

public abstract class AsyncDecisionTreeRegressionBase<T> : IAsyncTreeBasedModel<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected DecisionTreeNode<T>? Root;
    protected DecisionTreeOptions Options { get; private set; }
    protected IRegularization<T> Regularization { get; private set; }

    public virtual int MaxDepth => Options.MaxDepth;
    public Vector<T> FeatureImportances { get; protected set; }
    public virtual int NumberOfTrees => 1;

    protected AsyncDecisionTreeRegressionBase(DecisionTreeOptions? options, IRegularization<T>? regularization)
    {
        Options = options ?? new();
        NumOps = MathHelper.GetNumericOperations<T>();
        FeatureImportances = new Vector<T>(0);
        Regularization = regularization ?? new NoRegularization<T>();
    }

    public abstract Task TrainAsync(Matrix<T> x, Vector<T> y);

    public abstract Task<Vector<T>> PredictAsync(Matrix<T> input);

    public abstract ModelMetadata<T> GetModelMetadata();

    protected abstract Task CalculateFeatureImportancesAsync(int featureCount);

    public void Train(Matrix<T> x, Vector<T> y)
    {
        TrainAsync(x, y).GetAwaiter().GetResult();
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        return PredictAsync(input).GetAwaiter().GetResult();
    }

    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        writer.Write(Options.MaxDepth);
        writer.Write(Options.MinSamplesSplit);
        writer.Write(double.IsNaN(Options.MaxFeatures) ? -1 : Options.MaxFeatures);
        writer.Write(Options.Seed ?? -1);
        writer.Write((int)Options.SplitCriterion);

        // Serialize feature importances
        writer.Write(FeatureImportances.Length);
        foreach (var importance in FeatureImportances)
        {
            writer.Write(Convert.ToDouble(importance));
        }

        // Serialize tree structure
        SerializeNode(writer, Root);

        return ms.ToArray();
    }

    public virtual void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        Options.MaxDepth = reader.ReadInt32();
        Options.MinSamplesSplit = reader.ReadInt32();
        int maxFeatures = reader.ReadInt32();
        Options.MaxFeatures = maxFeatures == -1 ? double.NaN : maxFeatures;
        int seed = reader.ReadInt32();
        Options.Seed = seed == -1 ? null : seed;
        Options.SplitCriterion = (SplitCriterion)reader.ReadInt32();

        // Deserialize feature importances
        int featureCount = reader.ReadInt32();
        var importances = new T[featureCount];
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        FeatureImportances = new Vector<T>(importances);

        // Deserialize tree structure
        Root = DeserializeNode(reader);
    }

    private void SerializeNode(BinaryWriter writer, DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);
        writer.Write(node.FeatureIndex);
        writer.Write(Convert.ToDouble(node.SplitValue));
        writer.Write(Convert.ToDouble(node.Prediction));
        writer.Write(node.IsLeaf);

        SerializeNode(writer, node.Left);
        SerializeNode(writer, node.Right);
    }

    private DecisionTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        bool hasNode = reader.ReadBoolean();
        if (!hasNode) return null;

        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = reader.ReadInt32(),
            SplitValue = NumOps.FromDouble(reader.ReadDouble()),
            Prediction = NumOps.FromDouble(reader.ReadDouble()),
            IsLeaf = reader.ReadBoolean()
        };

        node.Left = DeserializeNode(reader);
        node.Right = DeserializeNode(reader);

        return node;
    }
}