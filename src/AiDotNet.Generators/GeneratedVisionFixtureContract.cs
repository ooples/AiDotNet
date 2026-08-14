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
