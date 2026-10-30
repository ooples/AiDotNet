using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps a forward pass from pinning its activations: reports a backward-cache field written inside
/// a Forward body without a training-pass guard.
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
/// Deliberately narrow, to stay at zero false positives: it fires only on a reference-typed field the
/// codebase has already marked as scratch with <c>[Scratch]</c>, or one whose name follows the
/// established backward-cache convention (<c>_lastX</c> / <c>_xOutput</c> / <c>_xOut</c>). Persistent
/// state a layer must keep across calls -- RNN hidden state, BatchNorm running statistics -- is neither,
/// so it is untouched. A field that follows the convention but keeps its value on purpose says so with
/// <c>[KeepsActivation("reason")]</c>.
/// </para>
/// <para>
/// ADN0065 is an error in the AiDotNet assembly. The violations that existed when it became one are
/// listed in <see cref="ModelDefectBaselines.UnguardedActivationCaches"/>, keyed by field; a field
/// outside that list fails the build, and a listed field that no longer violates is ADNDEF004, so the
/// list can only shrink.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class ActivationCacheGuardAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A backward-cache write in a forward body with no training-pass guard.</summary>
    private static readonly DiagnosticDescriptor UnguardedCacheWrite = new(
        "ADN0065",
        "Backward-activation cache written without ShouldCacheForBackward",
        "'{0}' is written directly in '{1}', so this activation stays reachable for the whole "
            + "forward even when no backward will read it. Route the write through the base instead "
            + "-- 'SaveForBackward(ref {0}, value)' -- which keeps the decision in one place, the way "
            + "PyTorch's ctx.save_for_backward owns saved-tensor lifetime rather than the module. If the "
            + "field must outlive the pass on purpose, declare it with [KeepsActivation(\"why\")]",
        "AiDotNet.Performance",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Ungated activation caches make peak inference memory O(sum of activations) "
            + "instead of O(max live set).");

    /// <summary>A <c>[KeepsActivation]</c> declaration with no reason.</summary>
    private static readonly DiagnosticDescriptor BlankKeepReason = new(
        "ADN0068",
        "[KeepsActivation] needs a reason",
        "'{0}' is declared [KeepsActivation] without a reason; state why it must keep its activation "
            + "across calls, or remove the attribute and gate the write",
        "AiDotNet.Performance",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A blank reason claims nothing, so it does not exempt the field from ADN0065.");

    private const string KeepsActivationName = "KeepsActivationAttribute";

    private readonly ImmutableHashSet<string> _baseline;
    private readonly string _baselineName;

    /// <summary>Uses the checked-in <see cref="ModelDefectBaselines.UnguardedActivationCaches"/>.</summary>
    public ActivationCacheGuardAnalyzer()
        : this(ModelDefectBaselines.UnguardedActivationCaches, nameof(ModelDefectBaselines.UnguardedActivationCaches))
    {
    }

    /// <summary>Uses the given baseline, so the ratchet itself can be tested.</summary>
    /// <param name="baseline">Field keys (<c>Containing.Type.field</c>) whose violations predate the rule.</param>
    /// <param name="baselineName">The name ADNDEF004 reports a stale entry under.</param>
    public ActivationCacheGuardAnalyzer(IEnumerable<string> baseline, string baselineName)
    {
        if (baseline is null) throw new ArgumentNullException(nameof(baseline));
        if (baselineName is null) throw new ArgumentNullException(nameof(baselineName));
        _baseline = baseline.ToImmutableHashSet(StringComparer.Ordinal);
        _baselineName = baselineName;
    }

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(UnguardedCacheWrite, BlankKeepReason, ModelDefectClassAnalyzer.StaleBaseline);

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            // The rule and its baseline describe the library; test and tool assemblies define their
            // own throwaway layers and are not held to it.
            if (start.Compilation.AssemblyName != "AiDotNet") return;

            var violating = new ConcurrentDictionary<string, byte>(StringComparer.Ordinal);

            start.RegisterSymbolAction(symbolContext =>
            {
                var field = (IFieldSymbol)symbolContext.Symbol;
                var keep = KeepsActivation(field);
                if (keep is null || HasReason(keep)) return;
                var location = keep.ApplicationSyntaxReference?.GetSyntax(symbolContext.CancellationToken).GetLocation()
                    ?? field.Locations.FirstOrDefault() ?? Location.None;
                symbolContext.ReportDiagnostic(Diagnostic.Create(BlankKeepReason, location, field.Name));
            }, SymbolKind.Field);

            start.RegisterSyntaxNodeAction(
                nodeContext => AnalyzeMethod(nodeContext, violating),
                SyntaxKind.MethodDeclaration);

            // ADNDEF004: the baseline only shrinks.
            start.RegisterCompilationEndAction(end =>
            {
                foreach (var entry in _baseline.Where(e => !violating.ContainsKey(e)).OrderBy(e => e, StringComparer.Ordinal))
                    end.ReportDiagnostic(Diagnostic.Create(
                        ModelDefectClassAnalyzer.StaleBaseline, Location.None, entry, _baselineName));
            });
        });
    }

    private void AnalyzeMethod(SyntaxNodeAnalysisContext context, ConcurrentDictionary<string, byte> violating)
    {
        var method = (MethodDeclarationSyntax)context.Node;
        if (!method.Identifier.Text.StartsWith("Forward", StringComparison.Ordinal)) return;
        if (method.Body is null) return;

        var model = context.SemanticModel;

        foreach (var node in method.Body.DescendantNodes())
        {
            ExpressionSyntax target;
            ExpressionSyntax? value;
            if (node is AssignmentExpressionSyntax assignment)
            {
                if (!assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)) continue;
                target = assignment.Left;
                // Field, or an element of a field array/list/dictionary (_resBlockOutputs[i] = x).
                if (target is ElementAccessExpressionSyntax element) target = element.Expression;
                value = assignment.Right;
            }
            else if (node is InvocationExpressionSyntax invocation
                && invocation.Expression is MemberAccessExpressionSyntax { Name.Identifier.Text: "Add" } add)
            {
                // Appending to a collection field retains the activation exactly like assigning it:
                // _cache.Add(x). The value is the last argument (a dictionary's Add takes the key first).
                target = add.Expression;
                value = invocation.ArgumentList.Arguments.LastOrDefault()?.Expression;
            }
            else
            {
                continue;
            }

            if (model.GetSymbolInfo(target, context.CancellationToken).Symbol is not IFieldSymbol field) continue;
            if (field.IsStatic || field.IsConst) continue;
            // Tensor<T>.Add(other) is arithmetic that returns a new tensor; only a collection's Add stores.
            if (node is InvocationExpressionSyntax && !IsCollection(field.Type)) continue;
            if (!IsBackwardCacheField(field)) continue;
            if (KeepsActivation(field) is { } keep && HasReason(keep)) continue;

            // Null clears (ResetState-style housekeeping inside a forward) release memory rather
            // than retain it, so they are never the problem this rule is about.
            if (value is not null && IsNullLike(value)) continue;

            if (value is not null && IsGuardedValue(value, model)) continue;
            if (IsGuarded(node, model)) continue;

            string key = FieldKey(field);
            violating.TryAdd(key, 0);
            if (_baseline.Contains(key)) continue;

            context.ReportDiagnostic(Diagnostic.Create(
                UnguardedCacheWrite,
                node.GetLocation(),
                field.Name,
                method.Identifier.Text));
        }
    }

    /// <summary>The baseline key of a field: its containing type's display name, a dot, the field name.</summary>
    internal static string FieldKey(IFieldSymbol field) => field.ContainingType.ToDisplayString() + "." + field.Name;

    private static bool IsCollection(ITypeSymbol type)
        => IsGenericCollectionInterface(type) || type.AllInterfaces.Any(IsGenericCollectionInterface);

    private static bool IsGenericCollectionInterface(ITypeSymbol type)
        => type.OriginalDefinition.MetadataName == "ICollection`1"
            && type.ContainingNamespace?.ToDisplayString() == "System.Collections.Generic";

    private static AttributeData? KeepsActivation(IFieldSymbol field)
        => field.GetAttributes().FirstOrDefault(a => a.AttributeClass?.Name == KeepsActivationName);

    private static bool HasReason(AttributeData keep)
        => keep.ConstructorArguments.Length >= 1
            && keep.ConstructorArguments[0].Value is string reason
            && !string.IsNullOrWhiteSpace(reason);

    /// <summary>
    /// True for a field the codebase already treats as a backward-activation cache: marked
    /// <c>[Scratch]</c>, or following the <c>_lastX</c> / <c>_xOutput</c> / <c>_xOut</c> naming the layers
    /// use. A value-typed or type-parameter-typed field (a batch size, a scalar loss of type T) cannot
    /// hold a tensor, so it is never one.
    /// </summary>
    private static bool IsBackwardCacheField(IFieldSymbol field)
    {
        if (field.Type.IsValueType || field.Type.TypeKind == TypeKind.TypeParameter) return false;

        if (field.GetAttributes().Any(a => a.AttributeClass?.Name is "ScratchAttribute" or "Scratch"))
            return true;

        string name = field.Name;
        if (!name.StartsWith("_", StringComparison.Ordinal)) return false;
        return name.StartsWith("_last", StringComparison.Ordinal)
            || name.EndsWith("Output", StringComparison.Ordinal)
            || name.EndsWith("Outputs", StringComparison.Ordinal)
            || name.EndsWith("Out", StringComparison.Ordinal);
    }

    private static bool IsNullLike(ExpressionSyntax value)
    {
        var stripped = StripParentheses(value);
        return stripped.IsKind(SyntaxKind.NullLiteralExpression) || stripped.IsKind(SyntaxKind.DefaultLiteralExpression);
    }

    /// <summary>
    /// True when the written value is itself gated: <c>_x = guard ? value : null</c>, which stores
    /// nothing whenever the guard is false.
    /// </summary>
    private static bool IsGuardedValue(ExpressionSyntax value, SemanticModel model)
    {
        if (StripParentheses(value) is not ConditionalExpressionSyntax conditional) return false;
        if (!IsNullLike(conditional.WhenFalse)) return false;
        return IsGuardCondition(conditional.Condition, model, conditional, depth: 0);
    }

    /// <summary>
    /// The <c>ShouldCache*</c> names. <c>ShouldCacheForBackward</c> is the canonical one, but a good deal
    /// of the library predates it and gates on <c>IsTrainingMode &amp;&amp;
    /// ShouldCacheActivationsForManualBackward</c>, which is the same decision minus the capture term.
    /// Treating only the newer name as a guard reported layers that were already correct.
    /// </summary>
    private static bool IsGuardName(string name)
        => name is "ShouldCacheForBackward" or "ShouldCacheActivationsForManualBackward";

    /// <summary>
    /// True when the write sits under an <c>if</c> whose condition is a guard (see
    /// <see cref="IsGuardCondition"/>).
    /// </summary>
    private static bool IsGuarded(SyntaxNode write, SemanticModel model)
    {
        for (SyntaxNode? node = write; node is not null; node = node.Parent)
        {
            if (node is not IfStatementSyntax ifStatement) continue;

            if (IsGuardCondition(ifStatement.Condition, model, ifStatement, depth: 0)) return true;
        }

        return false;
    }

    /// <summary>
    /// A condition is a guard when it reads a <c>ShouldCache*</c> name, when its <c>&amp;&amp;</c> chain
    /// requires a training pass outside an inference scope (<c>IsTrainingMode &amp;&amp;
    /// !InferenceMode.IsActive</c>, which is how caches read only by an auxiliary loss or a learning rule
    /// are gated: <c>ShouldCacheForBackward</c> is false under a recording tape, exactly when those
    /// consumers run), or when it reads a local that was initialised from such a guard.
    /// </summary>
    private static bool IsGuardCondition(ExpressionSyntax condition, SemanticModel model, SyntaxNode scope, int depth)
    {
        if (depth > 4) return false;

        if (condition.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>().Any(i => IsGuardName(i.Identifier.Text)))
            return true;

        var conjuncts = new List<ExpressionSyntax>();
        CollectConjuncts(condition, conjuncts);
        bool requiresTraining = conjuncts.Any(IsTrainingModeRead);
        bool excludesInference = conjuncts.Any(IsNotInferenceModeRead);
        if (requiresTraining && excludesInference) return true;

        // A local standing in for the guard: bool cacheBwd = ShouldCacheForBackward; if (cacheBwd) ...
        foreach (var identifier in condition.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>())
        {
            if (model.GetSymbolInfo(identifier).Symbol is not ILocalSymbol local) continue;
            var initializer = LocalInitializer(local, scope);
            if (initializer is not null && IsGuardCondition(initializer, model, scope, depth + 1)) return true;
        }

        return false;
    }

    private static ExpressionSyntax? LocalInitializer(ILocalSymbol local, SyntaxNode scope)
    {
        var method = scope.AncestorsAndSelf().OfType<MethodDeclarationSyntax>().FirstOrDefault();
        if (method?.Body is null) return null;

        foreach (var declarator in method.Body.DescendantNodes().OfType<VariableDeclaratorSyntax>())
        {
            if (declarator.Identifier.Text == local.Name && declarator.Initializer?.Value is { } value)
                return value;
        }

        return null;
    }

    private static void CollectConjuncts(ExpressionSyntax expression, List<ExpressionSyntax> conjuncts)
    {
        var stripped = StripParentheses(expression);
        if (stripped is BinaryExpressionSyntax binary && binary.IsKind(SyntaxKind.LogicalAndExpression))
        {
            CollectConjuncts(binary.Left, conjuncts);
            CollectConjuncts(binary.Right, conjuncts);
            return;
        }

        conjuncts.Add(stripped);
    }

    private static bool IsTrainingModeRead(ExpressionSyntax expression)
        => StripParentheses(expression) switch
        {
            IdentifierNameSyntax identifier => identifier.Identifier.Text == "IsTrainingMode",
            MemberAccessExpressionSyntax member => member.Expression is ThisExpressionSyntax
                && member.Name.Identifier.Text == "IsTrainingMode",
            _ => false,
        };

    private static bool IsNotInferenceModeRead(ExpressionSyntax expression)
        => StripParentheses(expression) is PrefixUnaryExpressionSyntax { RawKind: (int)SyntaxKind.LogicalNotExpression } negation
            && StripParentheses(negation.Operand) is MemberAccessExpressionSyntax member
            && member.Name.Identifier.Text == "IsActive"
            && member.Expression switch
            {
                IdentifierNameSyntax type => type.Identifier.Text == "InferenceMode",
                MemberAccessExpressionSyntax qualified => qualified.Name.Identifier.Text == "InferenceMode",
                _ => false,
            };

    private static ExpressionSyntax StripParentheses(ExpressionSyntax expression)
    {
        while (expression is ParenthesizedExpressionSyntax parenthesized) expression = parenthesized.Expression;
        return expression;
    }
}
