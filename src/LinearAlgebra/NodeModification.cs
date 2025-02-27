namespace AiDotNet.LinearAlgebra;

public class NodeModification
{
    public int NodeId { get; set; }
    public ModificationType Type { get; set; }
    public NodeType? NewNodeType { get; set; }
}