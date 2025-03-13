namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates and manipulates symbolic models for genetic programming.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Symbolic models are mathematical expressions represented as data structures 
/// that can be manipulated by algorithms. In genetic programming, these models evolve over time to 
/// solve problems, similar to how natural selection works in nature.
/// </para>
/// <para>
/// This factory helps you create, mutate, and combine symbolic models without needing to know their 
/// internal implementation details. Think of it like a workshop where you can build and modify 
/// mathematical expressions that represent solutions to your problems.
/// </para>
/// </remarks>
public static class SymbolicModelFactory<T>
{
    /// <summary>
    /// Random number generator used for creating and modifying models.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is used to introduce randomness when creating or changing models, 
    /// similar to how random mutations occur in nature.
    /// </remarks>
    private static readonly Random _random = new();
    
    /// <summary>
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a random symbolic model.
    /// </summary>
    /// <param name="useExpressionTrees">If true, creates an expression tree model; otherwise, creates a vector model.</param>
    /// <param name="dimensions">The number of dimensions for a vector model, or ignored for expression trees.</param>
    /// <param name="maxDepth">The maximum depth of the expression tree (only used if useExpressionTrees is true).</param>
    /// <returns>A randomly generated symbolic model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a random mathematical model that can be used as a starting 
    /// point in genetic programming. There are two types of models it can create:
    /// </para>
    /// <para>
    /// 1. Expression Trees: These are like mathematical formulas represented as trees, where each node is an 
    /// operation (like addition or multiplication) and the leaves are numbers or variables.
    /// </para>
    /// <para>
    /// 2. Vector Models: These are simpler models represented as lists of numbers.
    /// </para>
    /// <para>
    /// The maxDepth parameter controls how complex the expression tree can be - higher values allow for more 
    /// complex mathematical expressions.
    /// </para>
    /// </remarks>
    public static ISymbolicModel<T> CreateRandomModel(bool useExpressionTrees, int dimensions, int maxDepth = 5)
    {
        if (useExpressionTrees)
        {
            return CreateRandomExpressionTree(maxDepth);
        }
        else
        {
            return new VectorModel<T>(CreateRandomVector(dimensions));
        }
    }

    /// <summary>
    /// Creates an empty symbolic model.
    /// </summary>
    /// <param name="useExpressionTrees">If true, creates an expression tree model; otherwise, creates a vector model.</param>
    /// <param name="dimensions">The number of dimensions for a vector model, or ignored for expression trees.</param>
    /// <returns>An empty symbolic model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a blank model that you can use as a starting point. 
    /// For expression trees, it creates a simple tree with just a constant value of zero. For vector models, 
    /// it creates a vector filled with zeros.
    /// </para>
    /// <para>
    /// Empty models are useful when you want to build a model from scratch rather than starting with a 
    /// random one.
    /// </para>
    /// </remarks>
    public static ISymbolicModel<T> CreateEmptyModel(bool useExpressionTrees, int dimensions)
    {
        if (useExpressionTrees)
        {
            return new ExpressionTree<T>(NodeType.Constant, NumOps.Zero);
        }
        else
        {
            return new VectorModel<T>(new Vector<T>(dimensions));
        }
    }

    /// <summary>
    /// Mutates a symbolic model.
    /// </summary>
    /// <param name="model">The model to mutate.</param>
    /// <param name="mutationRate">The probability of mutation (between 0 and 1).</param>
    /// <returns>A new model that is a mutated version of the input model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mutation introduces small random changes to a model. This is similar to how 
    /// genetic mutations work in nature - most changes are small, but they can sometimes lead to improvements.
    /// </para>
    /// <para>
    /// The mutationRate parameter controls how likely changes are to occur. A higher value means more changes.
    /// For example, a mutation rate of 0.1 means each part of the model has a 10% chance of being changed.
    /// </para>
    /// <para>
    /// Mutation is important in genetic programming because it helps explore new possible solutions that 
    /// might be better than the current ones.
    /// </para>
    /// </remarks>
    public static ISymbolicModel<T> Mutate(ISymbolicModel<T> model, double mutationRate)
    {
        return model.Mutate(mutationRate);
    }

    /// <summary>
    /// Performs crossover between two symbolic models to create two new models.
    /// </summary>
    /// <param name="parent1">The first parent model.</param>
    /// <param name="parent2">The second parent model.</param>
    /// <param name="crossoverRate">The probability of crossover occurring (between 0 and 1).</param>
    /// <returns>A tuple containing two new models created by combining parts of the parent models.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Crossover combines parts of two existing models to create new ones, similar to 
    /// how genetic traits from two parents combine in their children.
    /// </para>
    /// <para>
    /// The crossoverRate parameter controls how likely it is for parts to be exchanged between the models. 
    /// A higher value means more parts will be exchanged.
    /// </para>
    /// <para>
    /// Crossover is a key operation in genetic programming because it allows good features from different 
    /// solutions to be combined, potentially creating even better solutions.
    /// </para>
    /// </remarks>
    public static (ISymbolicModel<T>, ISymbolicModel<T>) Crossover(ISymbolicModel<T> parent1, ISymbolicModel<T> parent2, double crossoverRate)
    {
        var child1 = parent1.Crossover(parent2, crossoverRate);
        var child2 = parent2.Crossover(parent1, crossoverRate);

        return (child1, child2);
    }

    /// <summary>
    /// Creates a random expression tree.
    /// </summary>
    /// <param name="maxDepth">The maximum depth of the tree.</param>
    /// <returns>A randomly generated expression tree.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This private method is used internally to create random expression trees. 
    /// It builds a tree of mathematical operations (like addition, subtraction, etc.) with random values. 
    /// The maxDepth parameter controls how complex the tree can be.
    /// </remarks>
    private static ExpressionTree<T> CreateRandomExpressionTree(int maxDepth)
    {
        if (maxDepth == 0 || _random.NextDouble() < 0.3) // 30% chance of generating a leaf node
        {
            return new ExpressionTree<T>(NodeType.Constant, NumOps.FromDouble(_random.NextDouble()));
        }

        NodeType nodeType = (NodeType)_random.Next(0, 4); // Randomly choose between Add, Subtract, Multiply, Divide
        var left = CreateRandomExpressionTree(maxDepth - 1);
        var right = CreateRandomExpressionTree(maxDepth - 1);

        return new ExpressionTree<T>(nodeType, default, left, right);
    }

    /// <summary>
    /// Creates a random vector.
    /// </summary>
    /// <param name="dimensions">The number of dimensions for the vector.</param>
    /// <returns>A randomly generated vector.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This private method is used internally to create random vectors filled with 
    /// random values between 0 and 1. The dimensions parameter determines how many numbers will be in the vector.
    /// </remarks>
    private static Vector<T> CreateRandomVector(int dimensions)
    {
        var vector = new Vector<T>(dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            vector[i] = NumOps.FromDouble(_random.NextDouble());
        }

        return vector;
    }
}