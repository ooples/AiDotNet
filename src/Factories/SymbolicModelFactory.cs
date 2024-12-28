namespace AiDotNet.Factories;

public static class SymbolicModelFactory<T>
{
    private static readonly Random _random = new();

    public static ISymbolicModel<T> CreateRandomModel(bool useExpressionTrees, int dimensions, INumericOperations<T> numOps)
    {
        if (useExpressionTrees)
        {
            return GenerateRandomExpressionTree(5, numOps); // Assuming max depth of 5 for random trees
        }
        else
        {
            return new VectorModel<T>(CreateRandomVector(dimensions, numOps), numOps);
        }
    }

    public static ISymbolicModel<T> CreateEmptyModel(bool useExpressionTrees, int dimensions, INumericOperations<T> numOps)
    {
        if (useExpressionTrees)
        {
            return new ExpressionTree<T>(NodeType.Constant, numOps.Zero);
        }
        else
        {
            return new VectorModel<T>(new Vector<T>(dimensions, numOps), numOps);
        }
    }

    public static ISymbolicModel<T> Mutate(ISymbolicModel<T> model, double mutationRate, INumericOperations<T> numOps)
    {
        return model.Mutate(mutationRate, numOps);
    }

    public static (ISymbolicModel<T>, ISymbolicModel<T>) Crossover(ISymbolicModel<T> parent1, ISymbolicModel<T> parent2, double crossoverRate, INumericOperations<T> numOps)
    {
        var child1 = parent1.Crossover(parent2, crossoverRate, numOps);
        var child2 = parent2.Crossover(parent1, crossoverRate, numOps);

        return (child1, child2);
    }

    private static ExpressionTree<T> GenerateRandomExpressionTree(int maxDepth, INumericOperations<T> numOps)
    {
        if (maxDepth == 0 || _random.NextDouble() < 0.3) // 30% chance of generating a leaf node
        {
            return new ExpressionTree<T>(NodeType.Constant, numOps.FromDouble(_random.NextDouble()));
        }

        NodeType nodeType = (NodeType)_random.Next(0, 4); // Randomly choose between Add, Subtract, Multiply, Divide
        var left = GenerateRandomExpressionTree(maxDepth - 1, numOps);
        var right = GenerateRandomExpressionTree(maxDepth - 1, numOps);

        return new ExpressionTree<T>(nodeType, default, left, right);
    }

    private static Vector<T> CreateRandomVector(int dimensions, INumericOperations<T> numOps)
    {
        var vector = new Vector<T>(dimensions, numOps);
        for (int i = 0; i < dimensions; i++)
        {
            vector[i] = numOps.FromDouble(_random.NextDouble());
        }

        return vector;
    }
}