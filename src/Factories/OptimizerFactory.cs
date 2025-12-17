namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates optimizer instances for training machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An optimizer is an algorithm that adjusts the parameters of a machine learning model 
/// to minimize errors and improve performance. Think of it like a navigator that helps your model find the 
/// best path to the correct answers.
/// </para>
/// <para>
/// This factory helps you create different types of optimizers without needing to know their internal 
/// implementation details. Think of it like ordering a specific tool from a catalog - you just specify 
/// what you need, and the factory provides it.
/// </para>
/// </remarks>
public static class OptimizerFactory<T, TInput, TOutput>
{
    /// <summary>
    /// A dictionary that maps optimizer types to their corresponding implementation classes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This dictionary stores information about which class to use for each 
    /// optimizer type, making it easy to look up the right implementation when needed.
    /// </remarks>
    private static readonly Dictionary<OptimizerType, Type> _optimizerTypes = [];

    /// <summary>
    /// Static constructor that initializes the optimizer type dictionary.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This code runs once when the OptimizerFactory is first used, registering 
    /// all the available optimizer types so they can be created later.
    /// </remarks>
    static OptimizerFactory()
    {
        // Register all optimizer types
        RegisterOptimizerType(OptimizerType.Adam, typeof(AdamOptimizer<T, TInput, TOutput>));
        RegisterOptimizerType(OptimizerType.GradientDescent, typeof(GradientDescentOptimizer<T, TInput, TOutput>));
        RegisterOptimizerType(OptimizerType.StochasticGradientDescent, typeof(StochasticGradientDescentOptimizer<T, TInput, TOutput>));
        RegisterOptimizerType(OptimizerType.AntColony, typeof(AntColonyOptimizer<T, TInput, TOutput>));
        RegisterOptimizerType(OptimizerType.GeneticAlgorithm, typeof(GeneticAlgorithmOptimizer<T, TInput, TOutput>));
        RegisterOptimizerType(OptimizerType.SimulatedAnnealing, typeof(SimulatedAnnealingOptimizer<T, TInput, TOutput>));
        RegisterOptimizerType(OptimizerType.ParticleSwarm, typeof(ParticleSwarmOptimizer<T, TInput, TOutput>));
        RegisterOptimizerType(OptimizerType.Normal, typeof(NormalOptimizer<T, TInput, TOutput>));
    }

    /// <summary>
    /// Registers an optimizer type with its corresponding implementation class.
    /// </summary>
    /// <param name="optimizerType">The type of optimizer to register.</param>
    /// <param name="type">The implementation class for the optimizer type.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method adds a new entry to our catalog of optimizers, connecting 
    /// the optimizer type (like "Adam" or "GradientDescent") with the actual class that implements it.
    /// </remarks>
    private static void RegisterOptimizerType(OptimizerType optimizerType, Type type)
    {
        _optimizerTypes[optimizerType] = type;
    }

    /// <summary>
    /// Determines the optimizer type from an existing optimizer instance.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="optimizer">The optimizer instance to identify.</param>
    /// <returns>The type of the provided optimizer.</returns>
    /// <exception cref="ArgumentException">Thrown when the optimizer type cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines an optimizer object and tells you what type it is. 
    /// It's like looking at a tool and identifying whether it's a hammer, screwdriver, or wrench.
    /// </para>
    /// <para>
    /// This is useful when you have an optimizer object but don't know its specific type, such as 
    /// when saving or loading models.
    /// </para>
    /// </remarks>
    public static OptimizerType GetOptimizerType(IOptimizer<T, TInput, TOutput> optimizer)
    {
        foreach (var kvp in _optimizerTypes)
        {
            if (kvp.Value.IsInstanceOfType(optimizer))
            {
                return kvp.Key;
            }
        }

        throw new ArgumentException($"Unknown optimizer type: {optimizer.GetType().Name}");
    }

    /// <summary>
    /// Creates an optimizer of the specified type with the given options.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="optimizerTypeEnum">The type of optimizer to create.</param>
    /// <param name="options">Configuration options for the optimizer.</param>
    /// <returns>An implementation of IOptimizer<T, TInput, TOutput> for the specified optimizer type.</returns>
    /// <exception cref="ArgumentException">Thrown when an unknown optimizer type is specified.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer type is registered but null, or when instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a specific type of optimizer based on what you request. 
    /// Different optimizers use different strategies to improve your model.
    /// </para>
    /// <para>
    /// Available optimizer types include:
    /// <list type="bullet">
    /// <item><description>Adam: An adaptive optimizer that combines the benefits of several other methods, often works well for deep learning.</description></item>
    /// <item><description>GradientDescent: The simplest optimizer that moves in the direction that reduces error the most.</description></item>
    /// <item><description>StochasticGradientDescent: A faster version of gradient descent that uses random subsets of data.</description></item>
    /// <item><description>AntColony: Inspired by how ants find food, good for path-finding and routing problems.</description></item>
    /// <item><description>GeneticAlgorithm: Mimics natural selection to evolve better solutions over generations.</description></item>
    /// <item><description>SimulatedAnnealing: Inspired by metallurgy, good at avoiding getting stuck in suboptimal solutions.</description></item>
    /// <item><description>ParticleSwarm: Inspired by bird flocking behavior, good for continuous optimization problems.</description></item>
    /// <item><description>Normal: A basic optimizer with standard behavior.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IOptimizer<T, TInput, TOutput> CreateOptimizer(OptimizerType optimizerTypeEnum, OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (!_optimizerTypes.TryGetValue(optimizerTypeEnum, out Type? optimizerGenericType))
        {
            throw new ArgumentException($"Unknown optimizer type: {optimizerTypeEnum}");
        }

        if (optimizerGenericType == null)
        {
            throw new InvalidOperationException($"Optimizer type {optimizerTypeEnum} is registered but null.");
        }

        Type concreteOptimizerType = optimizerGenericType.MakeGenericType(typeof(T));
        object? instance = Activator.CreateInstance(concreteOptimizerType, options);

        return instance == null
            ? throw new InvalidOperationException($"Failed to create instance of {concreteOptimizerType.Name}")
            : (IOptimizer<T, TInput, TOutput>)instance;
    }
}
