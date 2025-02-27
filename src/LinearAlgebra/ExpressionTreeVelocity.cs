namespace AiDotNet.LinearAlgebra;

public class ExpressionTreeVelocity<T>
{
    public Dictionary<int, T> NodeValueChanges { get; set; }
    public List<NodeModification> StructureChanges { get; set; }

    public ExpressionTreeVelocity()
    {
        NodeValueChanges = [];
        StructureChanges = [];
    }
}