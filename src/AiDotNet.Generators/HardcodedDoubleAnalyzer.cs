using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental generator that flags hardcoded <c>double</c> fields in generic
/// <c>&lt;T&gt;</c> classes. These fields should typically use the generic type parameter
/// <c>T</c>, <c>Vector&lt;T&gt;</c>, or <c>Matrix&lt;T&gt;</c> to preserve precision
/// and enable SIMD vectorization.
/// </summary>
[Generator]
public class HardcodedDoubleAnalyzer : IIncrementalGenerator
{
    private static readonly DiagnosticDescriptor DoubleFieldInGenericClass = new(
        id: "AIDN060",
        title: "Hardcoded double field in generic <T> class",
        messageFormat: "Field '{0}' is declared as 'double' in generic class '{1}'. Consider using 'T' instead to preserve precision across numeric types.",
        category: "AiDotNet.TypeSafety",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DoubleArrayFieldInGenericClass = new(
        id: "AIDN061",
        title: "Hardcoded double[] field in generic <T> class",
        messageFormat: "Field '{0}' is declared as 'double[]' in generic class '{1}'. Consider using 'Vector<T>' instead for SIMD acceleration.",
        category: "AiDotNet.TypeSafety",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DoubleMatrixFieldInGenericClass = new(
        id: "AIDN062",
        title: "Hardcoded double[,]/double[][] field in generic <T> class",
        messageFormat: "Field '{0}' is declared as '{2}' in generic class '{1}'. Consider using 'Matrix<T>' instead for SIMD acceleration.",
        category: "AiDotNet.TypeSafety",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var fieldDeclarations = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsFieldInGenericClass(node),
            transform: static (ctx, _) => AnalyzeField(ctx))
            .Where(static result => result is not null);

        var collected = fieldDeclarations.Collect();

        context.RegisterSourceOutput(collected, static (spc, results) =>
        {
            foreach (var result in results)
            {
                if (result is not null)
                {
                    spc.ReportDiagnostic(result);
                }
            }
        });
    }

    private static bool IsFieldInGenericClass(SyntaxNode node)
    {
        if (node is not FieldDeclarationSyntax field)
            return false;

        // Must be inside a class with type parameters
        var parent = field.Parent;
        if (parent is ClassDeclarationSyntax classDecl)
            return classDecl.TypeParameterList is not null && classDecl.TypeParameterList.Parameters.Count > 0;

        return false;
    }

    private static Diagnostic? AnalyzeField(GeneratorSyntaxContext ctx)
    {
        var fieldSyntax = (FieldDeclarationSyntax)ctx.Node;

        // Skip const fields (mathematical constants are fine as double)
        foreach (var modifier in fieldSyntax.Modifiers)
        {
            if (modifier.IsKind(SyntaxKind.ConstKeyword))
                return null;
        }

        // Skip static fields (typically shared constants or configuration)
        foreach (var modifier in fieldSyntax.Modifiers)
        {
            if (modifier.IsKind(SyntaxKind.StaticKeyword))
                return null;
        }

        var variableDeclaration = fieldSyntax.Declaration;
        if (variableDeclaration.Variables.Count == 0)
            return null;

        var firstVariable = variableDeclaration.Variables[0];
        var fieldSymbol = ctx.SemanticModel.GetDeclaredSymbol(firstVariable) as IFieldSymbol;
        if (fieldSymbol is null)
            return null;

        // Verify the containing class has a type parameter named T
        var containingType = fieldSymbol.ContainingType;
        if (containingType is null || !containingType.IsGenericType)
            return null;

        bool hasTypeParamT = false;
        foreach (var tp in containingType.TypeParameters)
        {
            if (tp.Name == "T")
            {
                hasTypeParamT = true;
                break;
            }
        }
        if (!hasTypeParamT)
            return null;

        // Skip fields named with known exclusion patterns
        string fieldName = fieldSymbol.Name;
        if (IsExcludedFieldName(fieldName))
            return null;

        var fieldType = fieldSymbol.Type;
        var location = firstVariable.Identifier.GetLocation();
        string className = containingType.Name;

        // Check for double scalar
        if (fieldType.SpecialType == SpecialType.System_Double)
        {
            return Diagnostic.Create(DoubleFieldInGenericClass, location, fieldName, className);
        }

        // Check for double[] (single-dimensional array)
        if (fieldType is IArrayTypeSymbol arrayType)
        {
            if (arrayType.ElementType.SpecialType == SpecialType.System_Double)
            {
                if (arrayType.Rank == 1)
                {
                    return Diagnostic.Create(DoubleArrayFieldInGenericClass, location, fieldName, className);
                }

                if (arrayType.Rank == 2)
                {
                    return Diagnostic.Create(DoubleMatrixFieldInGenericClass, location, fieldName, className, "double[,]");
                }
            }

            // Check for double[][] (jagged array)
            if (arrayType.Rank == 1 &&
                arrayType.ElementType is IArrayTypeSymbol innerArray &&
                innerArray.ElementType.SpecialType == SpecialType.System_Double &&
                innerArray.Rank == 1)
            {
                return Diagnostic.Create(DoubleMatrixFieldInGenericClass, location, fieldName, className, "double[][]");
            }
        }

        return null;
    }

    /// <summary>
    /// Fields with these name patterns are excluded from analysis because they
    /// represent Random boundaries, configuration values, or mathematical helpers
    /// that legitimately need to be double.
    /// </summary>
    private static bool IsExcludedFieldName(string name)
    {
        // Random-related fields
        if (name.Contains("random") || name.Contains("Random"))
            return true;

        // Threshold/epsilon configuration often set from constructor double params
        if (name == "_epsilon" || name == "_tolerance")
            return true;

        return false;
    }
}
