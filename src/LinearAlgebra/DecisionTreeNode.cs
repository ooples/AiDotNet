public class DecisionTreeNode<T>
{
    public int FeatureIndex { get; set; }
    public T SplitValue { get; set; }
    public T Prediction { get; set; }
    public DecisionTreeNode<T>? Left { get; set; }
    public DecisionTreeNode<T>? Right { get; set; }
    public bool IsLeaf { get; set; }
    public List<Sample<T>> Samples { get; set; } = new List<Sample<T>>();

    private INumericOperations<T> NumOps { get; set; }

    public DecisionTreeNode()
    {
        Left = null;
        Right = null;
        IsLeaf = true;

        NumOps = MathHelper.GetNumericOperations<T>();
        SplitValue = NumOps.Zero;
        Prediction = NumOps.Zero;
    }

    public DecisionTreeNode(int featureIndex, T splitValue)
    {
        FeatureIndex = featureIndex;
        SplitValue = splitValue;
        IsLeaf = false;

        NumOps = MathHelper.GetNumericOperations<T>();
        SplitValue = NumOps.Zero;
        Prediction = NumOps.Zero;
    }

    public DecisionTreeNode(T prediction)
    {
        Prediction = prediction;
        IsLeaf = true;

        NumOps = MathHelper.GetNumericOperations<T>();
        SplitValue = NumOps.Zero;
        Prediction = NumOps.Zero;
    }
}