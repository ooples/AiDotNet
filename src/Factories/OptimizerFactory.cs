namespace AiDotNet.Factories;

public static class OptimizerFactory
{
    private static readonly Dictionary<OptimizerType, Type> _optimizerTypes = [];

    static OptimizerFactory()
    {
        // Register all optimizer types
        RegisterOptimizerType(OptimizerType.Adam, typeof(AdamOptimizer<>));
        RegisterOptimizerType(OptimizerType.GradientDescent, typeof(GradientDescentOptimizer<>));
        RegisterOptimizerType(OptimizerType.StochasticGradientDescent, typeof(StochasticGradientDescentOptimizer<>));
        RegisterOptimizerType(OptimizerType.AntColony, typeof(AntColonyOptimizer<>));
        RegisterOptimizerType(OptimizerType.GeneticAlgorithm, typeof(GeneticAlgorithmOptimizer<>));
        RegisterOptimizerType(OptimizerType.SimulatedAnnealing, typeof(SimulatedAnnealingOptimizer<>));
        RegisterOptimizerType(OptimizerType.ParticleSwarm, typeof(ParticleSwarmOptimizer<>));
        RegisterOptimizerType(OptimizerType.Normal, typeof(NormalOptimizer<>));
    }

    private static void RegisterOptimizerType(OptimizerType optimizerType, Type type)
    {
        _optimizerTypes[optimizerType] = type;
    }

    public static OptimizerType GetOptimizerType<T>(IOptimizer<T> optimizer)
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

    public static IOptimizer<T> CreateOptimizer<T>(OptimizerType optimizerTypeEnum, OptimizationAlgorithmOptions options)
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
            : (IOptimizer<T>)instance;
    }
}