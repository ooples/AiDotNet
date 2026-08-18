using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Resolves the spatial input contract explicitly declared by a generated vision-model constructor.
/// </summary>
/// <remarks>
/// Generated smoke fixtures may intentionally instantiate a bounded version of a paper-scale model.
/// Their input tensor must follow that constructor's architecture. Feeding an independently selected
/// spatial size changes the operation graph and can inflate attention work quadratically, so it is
/// neither a valid bounded test nor a valid paper-scale benchmark.
/// </remarks>
internal static class GeneratedVisionFixtureContract
{
    /// <summary>
    /// Reads positive integer <c>inputHeight</c> and <c>inputWidth</c> named arguments from the
    /// <c>NeuralNetworkArchitecture</c> object creation embedded in a constructor expression.
    /// </summary>
    internal static bool TryGetArchitectureSpatialSize(
        string constructorExpression,
        out int height,
        out int width)
    {
        height = 0;
        width = 0;

        if (string.IsNullOrWhiteSpace(constructorExpression))
            return false;

        ExpressionSyntax expression = SyntaxFactory.ParseExpression(constructorExpression);
        foreach (ObjectCreationExpressionSyntax creation in
                 expression.DescendantNodesAndSelf().OfType<ObjectCreationExpressionSyntax>())
        {
            if (creation.Type.ToString().IndexOf(
                    "NeuralNetworkArchitecture",
                    System.StringComparison.Ordinal) < 0)
                continue;

            ArgumentListSyntax? arguments = creation.ArgumentList;
            if (arguments is null)
                continue;

            int? candidateHeight = FindPositiveIntegerArgument(arguments, "inputHeight");
            int? candidateWidth = FindPositiveIntegerArgument(arguments, "inputWidth");
            if (candidateHeight.HasValue && candidateWidth.HasValue)
            {
                height = candidateHeight.Value;
                width = candidateWidth.Value;
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Applies the model-declared per-sample shape to a generated fixture while preserving any
    /// leading fixture-only axes, such as an explicit batch axis or an RGB channel axis omitted by
    /// a two-dimensional architecture declaration.
    /// </summary>
    internal static int[] ConformToDeclaredShape(int[] fallback, int[] declared)
    {
        if (fallback is null || fallback.Length == 0)
            return System.Array.Empty<int>();

        int[] result = (int[])fallback.Clone();
        if (declared is null || declared.Length == 0 || declared.Length > fallback.Length
            || declared.Any(axis => axis <= 0))
        {
            return result;
        }

        int leadingAxes = fallback.Length - declared.Length;
        if (leadingAxes > 2)
            return result;

        long fallbackElements = 1;
        long declaredElements = 1;
        for (int i = 0; i < declared.Length; i++)
        {
            int fallbackAxis = fallback[leadingAxes + i];
            if (fallbackAxis <= 0
                || fallbackElements > long.MaxValue / fallbackAxis
                || declaredElements > long.MaxValue / declared[i])
            {
                return result;
            }

            fallbackElements *= fallbackAxis;
            declaredElements *= declared[i];
        }

        // The runtime declaration may correct aspect ratio/rank semantics, but it must not turn a
        // bounded smoke fixture back into a production-resolution benchmark. Exact architecture
        // literals remain authoritative in the generator's compile-time path.
        if (declaredElements > fallbackElements)
            return result;

        for (int i = 0; i < declared.Length; i++)
            result[leadingAxes + i] = declared[i];

        return result;
    }

    private static int? FindPositiveIntegerArgument(ArgumentListSyntax arguments, string name)
    {
        ArgumentSyntax? argument = arguments.Arguments.FirstOrDefault(candidate =>
            candidate.NameColon?.Name.Identifier.ValueText == name);
        if (argument?.Expression is not LiteralExpressionSyntax literal
            || !literal.IsKind(SyntaxKind.NumericLiteralExpression)
            || literal.Token.Value is not int value
            || value <= 0)
        {
            return null;
        }

        return value;
    }
}
