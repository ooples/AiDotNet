namespace AiDotNet.Enums;

/// <summary>
/// Represents the types of modifications that can be applied to a model structure.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This enum represents different ways to change the structure of an AI model.
/// Think of a model as a network of connected nodes (similar to a flowchart). These values
/// describe the basic operations you can perform on this network: adding new nodes,
/// removing existing ones, or changing what a node does.
/// </para>
/// </remarks>
public enum ModificationType
{
    /// <summary>
    /// Adds a new node to the model structure.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This operation adds a new processing unit (node) to your AI model.
    /// It's like adding a new step in a recipe or a new decision point in a flowchart.
    /// Adding nodes can help your model learn more complex patterns.
    /// </para>
    /// </remarks>
    AddNode,

    /// <summary>
    /// Removes an existing node from the model structure.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This operation removes a processing unit from your AI model.
    /// Removing nodes can help simplify your model, which might make it faster
    /// or less prone to overfitting (when a model learns noise in the data rather
    /// than true patterns).
    /// </para>
    /// </remarks>
    RemoveNode,

    /// <summary>
    /// Changes the type or function of an existing node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This operation changes what a node does without adding or removing it.
    /// It's like changing a step in a recipe while keeping the same number of steps.
    /// For example, you might change a node from using a simple calculation to a more
    /// complex one, or change how it processes information.
    /// </para>
    /// </remarks>
    ChangeNodeType
}
