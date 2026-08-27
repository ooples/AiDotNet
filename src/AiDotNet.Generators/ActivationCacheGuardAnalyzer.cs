using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps a forward pass from pinning its activations: reports a backward-cache field written inside
/// a Forward body without <c>ShouldCacheForBackward</c> guarding it.
/// </summary>
/// <remarks>
/// <para>
/// An ungated cache write keeps that activation reachable for the whole pass, so peak memory becomes
/// O(sum of all activations) instead of O(max live set) -- in inference exactly as much as in
/// training, because nothing tells the layer no backward is coming. Measured on one VAE decoder:
/// 524.4MB retained versus 257.8MB once gated, and 49GB allocated before OutOfMemoryException at
/// the paper-default 512x512 fp64 size.
/// </para>
/// <para>
/// <c>LayerBase{T}.ShouldCacheForBackward</c> already exists and already documents itself as "the
/// canonical guard every such cache write should be gated on". The problem was never a missing
/// mechanism, it was adoption: 186 files write these caches, 77 used the guard, and 111 did not.
/// Prose does not hold that line -- the next layer added to the library gets written by copying its
/// neighbour. This analyzer is the ratchet, so the count can only go down.
/// </para>
/// <para>
/// Deliberately narrow, to stay at zero false positives: it fires only on a field the codebase has
/// already marked as scratch with <c>[Scratch]</c>, or one whose name follows the established
/// backward-cache convention (<c>_lastX</c> / <c>_xOutput</c>). Persistent state a layer must keep
/// across calls -- RNN hidden state, BatchNorm running statistics -- is neither, so it is untouched.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class ActivationCacheGuardAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A backward-cache write in a forward body with no ShouldCacheForBackward guard.</summary>
    private static readonly DiagnosticDescriptor UnguardedCacheWrite = new(
        "ADN0065",
        "Backward-activation cache written without ShouldCacheForBackward",
        "'{0}' is assigned directly in '{1}', so this activation stays reachable for the whole "
            + "forward even when no backward will read it. Route the write through the base instead "
            + "-- 'SaveForBackward(ref {0}, value)' -- which keeps the decision in one place, the way "
            + "PyTorch's ctx.save_for_backward owns saved-tensor lifetime rather than the module",
        "AiDotNet.Performance",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "Ungated activation caches make peak inference memory O(sum of activations) "
            + "instead of O(max live set).");

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(UnguardedCacheWrite);

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(AnalyzeMethod, SyntaxKind.MethodDeclaration);
    }

    private static void AnalyzeMethod(SyntaxNodeAnalysisContext context)
    {
        var method = (MethodDeclarationSyntax)context.Node;
        if (!method.Identifier.Text.StartsWith("Forward")) return;
        if (method.Body is null) return;

        var model = context.SemanticModel;

        foreach (var assignment in method.Body.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            if (!assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)) continue;

            var target = assignment.Left;
            // Field, or an element of a field array (_resBlockOutputs[i] = x).
            if (target is ElementAccessExpressionSyntax element) target = element.Expression;

            if (model.GetSymbolInfo(target).Symbol is not IFieldSymbol field) continue;
            if (field.IsStatic || field.IsConst) continue;
            if (!IsBackwardCacheField(field)) continue;

            // Null clears (ResetState-style housekeeping inside a forward) release memory rather
            // than retain it, so they are never the problem this rule is about.
            if (assignment.Right.IsKind(SyntaxKind.NullLiteralExpression)) continue;

            if (IsGuarded(assignment, model)) continue;

            context.ReportDiagnostic(Diagnostic.Create(
                UnguardedCacheWrite,
                assignment.GetLocation(),
                field.Name,
                method.Identifier.Text));
        }
    }

    /// <summary>
    /// True for a field the codebase already treats as a backward-activation cache: marked
    /// <c>[Scratch]</c>, or following the <c>_lastX</c> / <c>_xOutput</c> naming the layers use.
    /// </summary>
    private static bool IsBackwardCacheField(IFieldSymbol field)
    {
        if (field.GetAttributes().Any(a => a.AttributeClass?.Name is "ScratchAttribute" or "Scratch"))
            return true;

        string name = field.Name;
        if (!name.StartsWith("_")) return false;
        return name.StartsWith("_last") || name.EndsWith("Output") || name.EndsWith("Outputs");
    }

    /// <summary>
    /// Both guards count. <c>ShouldCacheForBackward</c> is the canonical one, but a good deal of the
    /// library predates it and gates on <c>IsTrainingMode &amp;&amp;
    /// ShouldCacheActivationsForManualBackward</c>, which is the same decision minus the capture
    /// term. Treating only the newer name as a guard reported layers that were already correct.
    /// </summary>
    private static bool IsGuardName(string name)
        => name is "ShouldCacheForBackward" or "ShouldCacheActivationsForManualBackward";

    /// <summary>
    /// True when the assignment sits under an <c>if</c> whose condition reads
    /// <c>ShouldCacheForBackward</c>, directly or through a local that was assigned from it.
    /// </summary>
    private static bool IsGuarded(SyntaxNode assignment, SemanticModel model)
    {
        for (SyntaxNode? node = assignment; node is not null; node = node.Parent)
        {
            if (node is not IfStatementSyntax ifStatement) continue;

            if (ConditionReachesGuard(ifStatement.Condition, model, ifStatement)) return true;
        }

        return false;
    }

    private static bool ConditionReachesGuard(
        ExpressionSyntax condition, SemanticModel model, SyntaxNode scope)
    {
        foreach (var identifier in condition.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>())
        {
            if (IsGuardName(identifier.Identifier.Text)) return true;

            // A local standing in for the guard: bool cacheBwd = ShouldCacheForBackward;
            if (model.GetSymbolInfo(identifier).Symbol is not ILocalSymbol local) continue;

            var method = scope.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();
            if (method?.Body is null) continue;

            foreach (var declarator in method.Body.DescendantNodes().OfType<VariableDeclaratorSyntax>())
            {
                if (declarator.Identifier.Text != local.Name) continue;
                var initializer = declarator.Initializer?.Value;
                if (initializer is null) continue;
                if (initializer.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>()
                        .Any(i => IsGuardName(i.Identifier.Text)))
                    return true;
            }
        }

        return false;
    }
}
