namespace AiDotNet.Extensions;

public static class SymbolicModelExtensions
{
    public static ISymbolicModel<T> ToSymbolicModel<T>(this Vector<T> vector)
    {
        return new VectorModel<T>(vector);
    }

    public static ISymbolicModel<T> ToSymbolicModel<T>(this ExpressionTree<T> expressionTree)
    {
        return expressionTree;
    }

    public static Vector<T> ToVector<T>(this ISymbolicModel<T> model)
    {
        return model switch
        {
            VectorModel<T> vectorModel => vectorModel.Coefficients,
            ExpressionTree<T> expressionTree => expressionTree.ToVector(),
            _ => throw new InvalidOperationException($"Cannot convert {model.GetType()} to Vector<T>")
        };
    }

    public static ExpressionTree<T> ToExpressionTree<T>(this ISymbolicModel<T> model)
    {
        return model switch
        {
            ExpressionTree<T> expressionTree => expressionTree,
            VectorModel<T> vectorModel => vectorModel.ToExpressionTree(),
            _ => throw new InvalidOperationException($"Cannot convert {model.GetType()} to ExpressionTree<T>")
        };
    }

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