namespace AiDotNet.Enums;

/// <summary>
/// Defines the different types of nodes that can exist in a computational graph.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A computational graph is a way to represent mathematical operations as a network
/// of connected nodes. Think of it like a recipe with steps: some nodes are ingredients (constants and variables),
/// while others are actions (like add, subtract). This is how AI models internally organize calculations.
/// Each node in the graph performs a specific operation on its inputs and passes the result to the next node.
/// </para>
/// </remarks>
public enum ExpressionNodeType
{
    /// <summary>
    /// A node that represents a fixed numerical value that doesn't change during computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Constant is simply a fixed number that doesn't change, like the number 5 or 3.14.
    /// In AI models, constants are often used for things like weights, biases, or other fixed parameters.
    /// </para>
    /// </remarks>
    Constant,

    /// <summary>
    /// A node that represents a value that can change during computation, such as an input or parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Variable is a value that can change, like a placeholder that gets filled in with
    /// different values. In AI models, variables often represent the input data (like images or text) or
    /// values that get updated during training (like weights that the model is learning).
    /// </para>
    /// </remarks>
    Variable,

    /// <summary>
    /// A node that performs addition on its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An Add node takes two or more values and adds them together. For example,
    /// if one input is 3 and another is 4, the Add node outputs 7. In neural networks, addition is often
    /// used to combine different signals or to add a bias term to a weighted sum.
    /// </para>
    /// </remarks>
    Add,

    /// <summary>
    /// A node that performs subtraction on its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Subtract node takes two values and subtracts the second from the first. For example,
    /// if the inputs are 7 and 3, the Subtract node outputs 4. In AI models, subtraction might be used in
    /// calculating differences or errors between predicted and actual values.
    /// </para>
    /// </remarks>
    Subtract,

    /// <summary>
    /// A node that performs multiplication on its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Multiply node takes two or more values and multiplies them together. For example,
    /// if one input is 3 and another is 4, the Multiply node outputs 12. In neural networks, multiplication
    /// is commonly used to apply weights to input values, controlling how much influence each input has.
    /// </para>
    /// </remarks>
    Multiply,

    /// <summary>
    /// A node that performs division on its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Divide node takes two values and divides the first by the second. For example,
    /// if the inputs are 10 and 2, the Divide node outputs 5. In AI models, division might be used for
    /// normalization (adjusting values to a standard scale) or calculating ratios between values.
    /// </para>
    /// </remarks>
    Divide
}
