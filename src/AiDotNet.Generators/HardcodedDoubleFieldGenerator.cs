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
public class HardcodedDoubleFieldGenerator : IIncrementalGenerator
{
    // NOTE: Severity is Info (not Warning) because TreatWarningsAsErrors=True
    // in the main project. Existing codebase has ~2000 known-debt double fields.
    // This diagnostic surfaces in IDE to prevent NEW regressions without blocking CI.
    // Promote to Warning once the remaining internal conversions are done.
    private static readonly DiagnosticDescriptor DoubleFieldInGenericClass = new(
        id: "AIDN060",
        title: "Hardcoded double field in generic <T> class",
        messageFormat: "Field '{0}' is declared as 'double' in generic class '{1}'. Consider using 'T' instead to preserve precision across numeric types.",
        category: "AiDotNet.TypeSafety",
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DoubleArrayFieldInGenericClass = new(
        id: "AIDN061",
        title: "Hardcoded double[] field in generic <T> class",
        messageFormat: "Field '{0}' is declared as 'double[]' in generic class '{1}'. Consider using 'Vector<T>' instead for SIMD acceleration.",
        category: "AiDotNet.TypeSafety",
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DoubleMatrixFieldInGenericClass = new(
        id: "AIDN062",
        title: "Hardcoded double[,]/double[][] field in generic <T> class",
        messageFormat: "Field '{0}' is declared as '{2}' in generic class '{1}'. Consider using 'Matrix<T>' instead for SIMD acceleration.",
        category: "AiDotNet.TypeSafety",
        defaultSeverity: DiagnosticSeverity.Info,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var fieldDeclarations = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsFieldInGenericClass(node),
            transform: static (ctx, _) => AnalyzeField(ctx))
            .Where(static result => !result.IsDefault && result.Length > 0);

        var collected = fieldDeclarations.Collect();

        context.RegisterSourceOutput(collected, static (spc, results) =>
        {
            foreach (var batch in results)
            {
                foreach (var diag in batch)
                {
                    spc.ReportDiagnostic(diag);
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

    private static ImmutableArray<Diagnostic> AnalyzeField(GeneratorSyntaxContext ctx)
    {
        var fieldSyntax = (FieldDeclarationSyntax)ctx.Node;

        // Skip const, static, readonly fields
        foreach (var modifier in fieldSyntax.Modifiers)
        {
            if (modifier.IsKind(SyntaxKind.ConstKeyword) ||
                modifier.IsKind(SyntaxKind.StaticKeyword) ||
                modifier.IsKind(SyntaxKind.ReadOnlyKeyword))
                return ImmutableArray<Diagnostic>.Empty;
        }

        var variableDeclaration = fieldSyntax.Declaration;
        if (variableDeclaration.Variables.Count == 0)
            return ImmutableArray<Diagnostic>.Empty;

        // Use first variable to check containing type (same for all variables in declaration)
        var firstSymbol = ctx.SemanticModel.GetDeclaredSymbol(variableDeclaration.Variables[0]) as IFieldSymbol;
        if (firstSymbol is null)
            return ImmutableArray<Diagnostic>.Empty;

        var containingType = firstSymbol.ContainingType;
        if (containingType is null || !containingType.IsGenericType)
            return ImmutableArray<Diagnostic>.Empty;

        // Check if the class has a numeric type parameter (T, TValue, TElement, etc.)
        // In this codebase, generic numeric classes always have a type parameter
        // that could represent float/double — any generic class qualifies
        if (containingType.TypeParameters.Length == 0)
            return ImmutableArray<Diagnostic>.Empty;

        string className = containingType.Name;
        var builder = ImmutableArray.CreateBuilder<Diagnostic>();

        // Check ALL variables in the declaration (handles: double a, b, c;)
        foreach (var variable in variableDeclaration.Variables)
        {
            var fieldSymbol = ctx.SemanticModel.GetDeclaredSymbol(variable) as IFieldSymbol;
            if (fieldSymbol is null)
                continue;

            string fieldName = fieldSymbol.Name;
            if (IsExcludedFieldName(fieldName))
                continue;

            var fieldType = fieldSymbol.Type;
            var location = variable.Identifier.GetLocation();

            if (fieldType.SpecialType == SpecialType.System_Double)
            {
                builder.Add(Diagnostic.Create(DoubleFieldInGenericClass, location, fieldName, className));
            }
            else if (fieldType is IArrayTypeSymbol arrayType)
            {
                if (arrayType.ElementType.SpecialType == SpecialType.System_Double)
                {
                    if (arrayType.Rank == 1)
                        builder.Add(Diagnostic.Create(DoubleArrayFieldInGenericClass, location, fieldName, className));
                    else if (arrayType.Rank == 2)
                        builder.Add(Diagnostic.Create(DoubleMatrixFieldInGenericClass, location, fieldName, className, "double[,]"));
                }
                else if (arrayType.Rank == 1 &&
                    arrayType.ElementType is IArrayTypeSymbol innerArray &&
                    innerArray.ElementType.SpecialType == SpecialType.System_Double &&
                    innerArray.Rank == 1)
                {
                    builder.Add(Diagnostic.Create(DoubleMatrixFieldInGenericClass, location, fieldName, className, "double[][]"));
                }
            }
        }

        return builder.ToImmutable();
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
