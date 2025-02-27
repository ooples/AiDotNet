namespace AiDotNet.LinearAlgebra;

public class ConditionalInferenceTreeNode<T> : DecisionTreeNode<T>
{
    public T PValue { get; set; }

    public ConditionalInferenceTreeNode()
    {
        PValue = MathHelper.GetNumericOperations<T>().Zero;
    }
}