namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents the velocity (rate and direction of change) for an expression tree during optimization.
/// </summary>
/// <typeparam name="T">The numeric type used in the expression tree (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Think of this class as tracking how a mathematical formula should change
/// during optimization. Just like velocity in physics describes how fast and in what direction
/// an object is moving, this class describes how the formula is "moving" or changing during the
/// optimization process. It keeps track of which numbers should change and how the structure
/// of the formula might be modified.
/// </remarks>
public class ExpressionTreeVelocity<T>
{
    /// <summary>
    /// A dictionary mapping node IDs to their value changes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tracks how the constant numbers in your formula should change.
    /// For example, if your formula has "2x + 3" and the optimization suggests changing the 2 to 2.5
    /// and the 3 to 3.2, this dictionary would store those suggested changes.
    /// The keys are the unique IDs of nodes in the expression tree, and the values are the
    /// amounts by which those nodes' values should change.
    /// </remarks>
    public Dictionary<int, T> NodeValueChanges { get; set; }

    /// <summary>
    /// A list of structural modifications to apply to the expression tree.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tracks changes to the structure of your formula, not just the numbers.
    /// For example, it might suggest replacing "x + y" with "x * y" in part of your formula.
    /// These are more dramatic changes than just adjusting coefficient values.
    /// </remarks>
    public List<NodeModification> StructureChanges { get; set; }

    /// <summary>
    /// Initializes a new instance of the ExpressionTreeVelocity class with empty collections.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new, empty set of suggested changes for your formula.
    /// It's like starting with a blank slate before the optimization process fills in what changes should be made.
    /// </remarks>
    public ExpressionTreeVelocity()
    {
        NodeValueChanges = [];
        StructureChanges = [];
    }
}
