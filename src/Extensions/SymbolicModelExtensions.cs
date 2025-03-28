namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for working with symbolic models in AI applications.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Symbolic models are a way to represent mathematical expressions or formulas
/// that your AI can understand and manipulate. Think of them as the "math equations" behind your AI models.
/// </para>
/// <para>
/// This class provides methods to convert between different representations of these models:
/// <list type="bullet">
///   <item><description>Vectors (lists of numbers)</description></item>
///   <item><description>Expression trees (mathematical formulas organized in a tree structure)</description></item>
/// </list>
/// </para>
/// <para>
/// For example, a simple linear equation like "2x + 3" could be represented as either:
/// <list type="bullet">
///   <item><description>A vector [2, 3] (where 2 is the coefficient of x and 3 is the constant)</description></item>
///   <item><description>An expression tree with addition at the root, multiplication of 2 and x as the left child, and 3 as the right child</description></item>
/// </list>
/// </para>
/// </remarks>
public static class SymbolicModelExtensions
{
    /// <summary>
    /// Converts a vector to a symbolic model representation.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the vector.</typeparam>
    /// <param name="vector">The vector to convert.</param>
    /// <returns>A symbolic model representation of the vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a list of numbers (vector) and converts it into a form
    /// that can be used as a symbolic model. This is useful when you have coefficients (like weights in a linear model)
    /// and want to use them in more advanced symbolic operations.
    /// </para>
    /// <para>
    /// For example, if you have a vector [2, 3] representing the equation "2x + 3", this method
    /// creates a symbolic model that understands this is a mathematical expression, not just a list of numbers.
    /// </para>
    /// </remarks>
    public static ISymbolicModel<T> ToSymbolicModel<T>(this Vector<T> vector)
    {
        return new VectorModel<T>(vector);
    }

    /// <summary>
    /// Converts an expression tree to a symbolic model representation.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the expression tree.</typeparam>
    /// <param name="expressionTree">The expression tree to convert.</param>
    /// <returns>A symbolic model representation of the expression tree.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a mathematical formula (represented as an expression tree)
    /// and ensures it can be used as a symbolic model. Expression trees represent formulas like "2x + 3" as a tree structure,
    /// which is useful for complex mathematical operations.
    /// </para>
    /// <para>
    /// Since expression trees already implement the symbolic model interface, this method simply returns the input,
    /// but it allows you to treat expression trees consistently with other symbolic model types.
    /// </para>
    /// </remarks>
    public static ISymbolicModel<T> ToSymbolicModel<T>(this ExpressionTree<T> expressionTree)
    {
        return expressionTree;
    }

    /// <summary>
    /// Converts a symbolic model to a vector representation.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the model.</typeparam>
    /// <param name="model">The symbolic model to convert.</param>
    /// <returns>A vector representation of the symbolic model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model type cannot be converted to a vector.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a symbolic model (which could be in different forms)
    /// and converts it to a simple list of numbers (vector). This is useful when you need to perform
    /// basic mathematical operations or when you need to save the model in a simple format.
    /// </para>
    /// <para>
    /// For example, an expression tree representing "2x + 3" would be converted to the vector [2, 3].
    /// </para>
    /// </remarks>
    public static Vector<T> ToVector<T>(this ISymbolicModel<T> model)
    {
        return model switch
        {
            VectorModel<T> vectorModel => vectorModel.Coefficients,
            ExpressionTree<T> expressionTree => expressionTree.ToVector(),
            _ => throw new InvalidOperationException($"Cannot convert {model.GetType()} to Vector<T>")
        };
    }

    /// <summary>
    /// Converts a symbolic model to an expression tree representation.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the model.</typeparam>
    /// <param name="model">The symbolic model to convert.</param>
    /// <returns>An expression tree representation of the symbolic model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model type cannot be converted to an expression tree.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes a symbolic model (which could be in different forms)
    /// and converts it to an expression tree (a tree structure representing a mathematical formula).
    /// This is useful when you need to perform complex operations like differentiation or simplification.
    /// </para>
    /// <para>
    /// For example, a vector model [2, 3] representing "2x + 3" would be converted to an expression tree
    /// with addition at the root, multiplication of 2 and x as the left child, and 3 as the right child.
    /// </para>
    /// </remarks>
    public static ExpressionTree<T> ToExpressionTree<T>(this ISymbolicModel<T> model)
    {
        return model switch
        {
            ExpressionTree<T> expressionTree => expressionTree,
            VectorModel<T> vectorModel => vectorModel.ToExpressionTree(),
            _ => throw new InvalidOperationException($"Cannot convert {model.GetType()} to ExpressionTree<T>")
        };
    }

    /// <summary>
    /// Updates a symbolic model using a velocity vector, typically used in optimization algorithms.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the model.</typeparam>
    /// <param name="model">The symbolic model to update.</param>
    /// <param name="velocity">The velocity vector containing update values.</param>
    /// <exception cref="NotSupportedException">Thrown when the model type does not support updating from velocity.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method updates the values in a symbolic model by adding the corresponding values
    /// from a "velocity" vector. In AI optimization, velocity represents how much and in which direction
    /// the model's parameters should change to improve its performance.
    /// </para>
    /// <para>
    /// Think of it like this: if your model is a point on a map, the velocity tells you which direction to move
    /// and how far to go to get closer to your destination (the optimal solution).
    /// </para>
    /// <para>
    /// This is commonly used in optimization algorithms like:
    /// <list type="bullet">
    ///   <item><description>Gradient Descent: where velocity represents the negative gradient</description></item>
    ///   <item><description>Particle Swarm Optimization: where velocity is influenced by both personal and global best positions</description></item>
    ///   <item><description>Momentum-based methods: where velocity accumulates over time to help escape local minima</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static void UpdateFromVelocity<T>(this ISymbolicModel<T> model, Vector<T> velocity)
    {
        if (model is VectorModel<T> vectorModel)
        {
            UpdateVectorModel(vectorModel, velocity);
        }
        else if (model is ExpressionTree<T> expressionTree)
        {
            UpdateExpressionTree(expressionTree, velocity);
        }
        else
        {
            throw new NotSupportedException($"Updating from velocity is not supported for type {model.GetType()}");
        }
    }

    /// <summary>
    /// Updates a vector model using a velocity vector.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the model.</typeparam>
    /// <param name="model">The vector model to update.</param>
    /// <param name="velocity">The velocity vector containing update values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method updates each coefficient in a vector model
    /// by adding the corresponding value from the velocity vector.
    /// </para>
    /// <para>
    /// For example, if your model has coefficients [2, 3] and the velocity is [0.1, -0.2],
    /// the updated coefficients would be [2.1, 2.8].
    /// </para>
    /// </remarks>
    private static void UpdateVectorModel<T>(VectorModel<T> model, Vector<T> velocity)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var updatedCoefficients = new Vector<T>(model.Coefficients.Length);
        for (int i = 0; i < model.Coefficients.Length; i++)
        {
            updatedCoefficients[i] = numOps.Add(model.Coefficients[i], velocity[i]);
        }

        model.UpdateCoefficients(updatedCoefficients);
    }

    /// <summary>
    /// Updates an expression tree using a velocity vector.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the tree.</typeparam>
    /// <param name="tree">The expression tree to update.</param>
    /// <param name="velocity">The velocity vector containing update values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method updates each constant value in an expression tree
    /// by adding the corresponding value from the velocity vector.
    /// </para>
    /// <para>
    /// For example, if your expression tree represents "2x + 3" and the velocity is [0.1, -0.2],
    /// the updated expression would be "2.1x + 2.8".
    /// </para>
    /// <para>
    /// The method traverses the tree structure recursively, updating only the constant nodes
    /// (nodes that represent fixed numbers, not variables or operations).
    /// </para>
    /// </remarks>
    private static void UpdateExpressionTree<T>(ExpressionTree<T> tree, Vector<T> velocity)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int index = 0;

        void UpdateNode(ExpressionTree<T> node)
        {
            if (node.Type == NodeType.Constant)
            {
                node.SetValue(numOps.Add(node.Value, velocity[index++]));
            }

            if (node.Left != null) UpdateNode(node.Left);
            if (node.Right != null) UpdateNode(node.Right);
        }

        UpdateNode(tree);
    }
}