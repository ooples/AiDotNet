namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a modification to be applied to a node in a computational graph.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In AI and machine learning, a computational graph is like a recipe that shows how 
/// calculations flow from inputs to outputs. Each step in this recipe is called a "node".
/// 
/// Sometimes, we need to change these nodes - maybe we want to remove a step, add a new one, or change how a step works.
/// This class helps us keep track of what changes we want to make to which nodes.
/// 
/// Think of it like editing instructions in a recipe: you might want to replace one ingredient with another,
/// remove a step, or add a new technique. This class helps keep track of those edits before you apply them.
/// </para>
/// </remarks>
public class NodeModification
{
    /// <summary>
    /// Gets or sets the unique identifier of the node to be modified.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like an ID number that uniquely identifies which specific node (calculation step)
    /// in the computational graph we want to modify. Each node has its own unique ID so we can tell them apart.</para>
    /// </remarks>
    public int NodeId { get; set; }

    /// <summary>
    /// Gets or sets the type of modification to apply to the node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This property specifies what kind of change we want to make to the node.
    /// For example, we might want to:
    /// - Add a new node
    /// - Remove an existing node
    /// - Change how a node works
    /// - Connect or disconnect nodes from each other
    /// 
    /// The ModificationType enum contains all the possible types of changes we can make.
    /// </para>
    /// </remarks>
    public ModificationType Type { get; set; }

    /// <summary>
    /// Gets or sets the new type for the node when changing its functionality.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If we're changing how a node works (its type), this property specifies what we're changing it to.
    /// 
    /// For example, we might change a node that performs addition to one that performs multiplication instead.
    /// 
    /// This property is nullable (has the ? symbol) because it's only relevant when we're changing a node's type.
    /// For other modifications like adding or removing nodes, this property would be null (not used).
    /// </para>
    /// </remarks>
    public ExpressionNodeType? NewNodeType { get; set; }
}
