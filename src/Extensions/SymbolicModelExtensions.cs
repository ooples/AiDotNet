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
}