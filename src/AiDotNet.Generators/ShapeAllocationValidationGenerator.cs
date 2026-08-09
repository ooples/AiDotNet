using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Build-time rule against allocating a throwaway <c>int[]</c> for a shape that is only read.
/// </summary>
/// <remarks>
/// <para>
/// <c>Tensor&lt;T&gt;.Shape</c> is an immutable <c>TensorShape</c> WRAPPER, so <c>.ToArray()</c>
/// materialises a fresh <c>int[]</c> every single call. The backing field
/// <c>internal readonly int[] _shape</c> is directly reachable — AiDotNet.Tensors lists both
/// <c>AiDotNet</c> and <c>AiDotNetTests</c> in its <c>InternalsVisibleTo</c> — so a read-only use pays
/// an allocation for nothing. There were 144 such call sites in <c>src</c> when this rule was written.
/// </para>
/// <para>
/// WHY THIS IS DELIBERATELY CONSERVATIVE. <c>.ToArray()</c> is not merely wasteful; it is a DEFENSIVE
/// COPY, and <c>_shape</c> hands out the tensor's live array. Anywhere the result is returned, stored,
/// or mutated, that copy is load-bearing and removing it introduces aliasing — the same defect class as
/// a shallow <c>MemberwiseClone</c> deep-copy, or the VAE decode that returned a shared buffer and so
/// aliased every result. A rule that cannot tell those apart would be advising a bug.
/// </para>
/// <para>
/// So this reports ONLY the provably-safe shape: assigned to a local whose every subsequent use is an
/// element read or a length query. Anything else — returned, assigned to a field, passed as an argument
/// (the callee may store it), captured, reassigned — is left alone, even though many such sites are
/// probably fine too. Under-reporting is the correct bias here: ADNTEST002 in this same project sits
/// <c>Disabled</c> today because, enabled, it fired on 46 models and most were not defects. A rule that
/// cries wolf gets switched off, and then it protects nothing.
/// </para>
/// </remarks>
[Generator]
public class ShapeAllocationValidationGenerator : IIncrementalGenerator
{
    private const string TensorShapeTypeName = "AiDotNet.Tensors.LinearAlgebra.TensorShape";

    private static readonly DiagnosticDescriptor RedundantShapeCopyDescriptor = new(
        id: "ADNPERF001",
        title: "Shape.ToArray() allocates a throwaway int[] for a read-only use",
        messageFormat: "'{0}' is assigned from Shape.ToArray() but is only ever read ({1}), so the copy "
                       + "is pure allocation. Tensor.Shape is an immutable wrapper - ToArray() builds a "
                       + "new int[] on every call - while the backing field _shape is internal and "
                       + "visible here via InternalsVisibleTo. Use '_shape' directly. This is reported "
                       + "ONLY where the value provably does not escape; where the defensive copy is "
                       + "load-bearing (returned, stored, or mutated) it is correct and is not flagged.",
        category: "AiDotNet.Performance",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) =>
                    node is InvocationExpressionSyntax
                    {
                        Expression: MemberAccessExpressionSyntax
                        {
                            Name.Identifier.ValueText: "ToArray",
                            Expression: MemberAccessExpressionSyntax { Name.Identifier.ValueText: "Shape" }
                        },
                        ArgumentList.Arguments.Count: 0
                    },
                transform: static (ctx, _) => Analyze(ctx))
            .Where(static r => r is not null)
            .Collect();

        context.RegisterSourceOutput(candidates, static (spc, findings) =>
        {
            foreach (var f in findings)
            {
                if (f is null) continue;
                spc.ReportDiagnostic(Diagnostic.Create(
                    RedundantShapeCopyDescriptor, f.Value.Location, f.Value.Name, f.Value.Detail));
            }
        });
    }

    // A plain tuple rather than a record struct: this project targets netstandard2.0, which has no
    // IsExternalInit, so positional record properties do not compile here.
    private static (Location Location, string Name, string Detail)? Analyze(GeneratorSyntaxContext ctx)
    {
        var invocation = (InvocationExpressionSyntax)ctx.Node;

        // Confirm the receiver really is a TensorShape. Plenty of unrelated types expose a Shape
        // property, and flagging those would be exactly the false-positive problem this rule is
        // written to avoid.
        var shapeAccess = ((MemberAccessExpressionSyntax)invocation.Expression).Expression;
        var shapeType = ctx.SemanticModel.GetTypeInfo(shapeAccess).Type;
        if (shapeType is null || shapeType.ToDisplayString() != TensorShapeTypeName)
            return null;

        // Only the `var x = t.Shape.ToArray();` shape is analysable. An inline argument escapes into
        // a callee we cannot see, and a field assignment escapes by definition.
        if (invocation.Parent is not EqualsValueClauseSyntax equals ||
            equals.Parent is not VariableDeclaratorSyntax declarator ||
            declarator.Parent is not VariableDeclarationSyntax ||
            declarator.Parent.Parent is not LocalDeclarationStatementSyntax)
        {
            return null;
        }

        if (ctx.SemanticModel.GetDeclaredSymbol(declarator) is not ILocalSymbol local)
            return null;

        // The enclosing body bounds the search. Without one there is nothing to prove.
        SyntaxNode? body = invocation.FirstAncestorOrSelf<BaseMethodDeclarationSyntax>();
        body ??= invocation.FirstAncestorOrSelf<AccessorDeclarationSyntax>();
        body ??= invocation.FirstAncestorOrSelf<LocalFunctionStatementSyntax>();
        if (body is null) return null;

        int reads = 0;

        foreach (var identifier in body.DescendantNodes().OfType<IdentifierNameSyntax>())
        {
            if (identifier.Identifier.ValueText != local.Name) continue;
            if (identifier.SpanStart <= declarator.SpanStart) continue;

            var symbol = ctx.SemanticModel.GetSymbolInfo(identifier).Symbol;
            if (symbol is null || !SymbolEqualityComparer.Default.Equals(symbol, local)) continue;

            if (!IsReadOnlyUse(identifier))
                return null;   // escapes, or is mutated - the copy may well be load-bearing

            reads++;
        }

        // Zero reads means the local is dead. That is a different defect and not this rule's business;
        // reporting it here would attribute a dead-store to allocation.
        if (reads == 0) return null;

        return (invocation.GetLocation(), local.Name, reads == 1 ? "1 read" : $"{reads} reads");
    }

    /// <summary>
    /// True only for uses that cannot let the array escape or change: indexing it, or asking its length.
    /// </summary>
    private static bool IsReadOnlyUse(IdentifierNameSyntax identifier)
    {
        switch (identifier.Parent)
        {
            // s[i] — reading an element. Writing one (s[i] = v) is caught below.
            case ElementAccessExpressionSyntax element when element.Expression == identifier:
                return element.Parent is not AssignmentExpressionSyntax assign || assign.Left != element;

            // s.Length / s.Rank — a scalar query, the array itself does not travel.
            case MemberAccessExpressionSyntax member when member.Expression == identifier:
                return member.Name.Identifier.ValueText is "Length" or "Rank";

            default:
                return false;
        }
    }
}
