namespace AiDotNet.Factories;

public static class SymbolicModelFactory<T>
{
    private static readonly Random _random = new();
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static ISymbolicModel<T> CreateRandomModel(bool useExpressionTrees, int dimensions)
    {
        if (useExpressionTrees)
        {
            return CreateRandomExpressionTree(5); // Assuming max depth of 5 for random trees
        }
        else
        {
            return new VectorModel<T>(CreateRandomVector(dimensions));
        }
    }

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

    public static ISymbolicModel<T> Mutate(ISymbolicModel<T> model, double mutationRate)
    {
        return model.Mutate(mutationRate);
    }

    public static (ISymbolicModel<T>, ISymbolicModel<T>) Crossover(ISymbolicModel<T> parent1, ISymbolicModel<T> parent2, double crossoverRate)
    {
        var child1 = parent1.Crossover(parent2, crossoverRate);
        var child2 = parent2.Crossover(parent1, crossoverRate);

        return (child1, child2);
    }

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