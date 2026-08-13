using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps cloning automated: reports a hand-written clone the base already reproduces, and reports a
/// model the clone plan cannot rebuild.
/// </summary>
/// <remarks>
/// <para>
/// Without this the overrides grow back. <c>CreateNewInstance</c>, <c>DeepCopy</c> and <c>Clone</c>
/// were abstract on eleven base classes, so every concrete model was compelled to write one, and
/// 1465 of them did. Making the bases concrete removes the compulsion but not the habit: the next
/// model added to the library will still be written with a copy of its neighbour's override, and
/// nothing would say otherwise.
/// </para>
/// <para>
/// The two rules are deliberately different in severity, because they describe different situations.
/// A redundant override is a mistake with a mechanical fix -- delete it -- and there are currently
/// none, so it is an error and stays at zero. A model the plan cannot rebuild is a backlog item with
/// a real fix (store the constructor argument in a field so it can be read back), and there are
/// hundreds, so it is informational and names the parameter that blocks each one.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class CloneAutomationAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A hand-written clone the base class already reproduces.</summary>
    private static readonly DiagnosticDescriptor RedundantOverride = new(
        "ADN0058",
        "Clone override duplicates what the base class already does",
        "'{0}.{1}' only reconstructs the type, and the clone plan already records the constructor to "
            + "do that. Delete the override: the base reproduces it, and a hand-written copy is a "
            + "place a future constructor argument can be dropped without anything failing",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <summary>A model whose constructor cannot be replayed from what the instance still holds.</summary>
    private static readonly DiagnosticDescriptor Unreproducible = new(
        "ADN0059",
        "Model cannot be rebuilt from its own state",
        "'{0}' cannot be rebuilt by the clone plan, so it still needs a hand-written clone. {1} "
            + "Store each one in a field named after it ('_name') and the generator will replay the "
            + "constructor instead",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true);

    /// <inheritdoc/>
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(RedundantOverride, Unreproducible);

    /// <inheritdoc/>
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(AnalyzeMethod, SyntaxKind.MethodDeclaration);
        context.RegisterSyntaxNodeAction(AnalyzeType, SyntaxKind.ClassDeclaration);
    }

    /// <summary>
    /// Reports an override whose whole body is a reconstruction the plan already performs.
    /// </summary>
    /// <param name="context">The analysis context.</param>
    /// <remarks>
    /// Only a body that is exactly one <c>return new ...;</c> is reported. An override that also
    /// resolves a shape, branches on a mode or copies a field is doing something the base does not,
    /// and telling someone to delete it would be wrong -- that work has to move deliberately.
    /// </remarks>
    private static void AnalyzeMethod(SyntaxNodeAnalysisContext context)
    {
        var method = (MethodDeclarationSyntax)context.Node;

        if (!method.Modifiers.Any(m => m.ValueText == "override")) return;
        if (method.ParameterList.Parameters.Count != 0) return;

        var name = method.Identifier.ValueText;
        if (name is not ("CreateNewInstance" or "DeepCopy" or "Clone")) return;

        if (!IsSingleReturnOfNewObject(method) && !IsPureForwarder(method, name)) return;

        if (context.ContainingSymbol is not IMethodSymbol symbol) return;

        // An override that satisfies an abstract member is not optional, whatever its body looks
        // like. A test file declares its own MockModelBase with `public abstract Clone()`, and
        // telling three mocks to delete the only implementation of it produced CS0534 instead of a
        // cleaner tree. Redundancy is a property of the base being CONCRETE, not of the body alone.
        if (symbol.OverriddenMethod is null || symbol.OverriddenMethod.IsAbstract) return;

        if (symbol.ContainingType is not INamedTypeSymbol type) return;
        if (ClonePlanGenerator.CollectConstructorParameters(type, IsModel(type)) is null) return;

        context.ReportDiagnostic(Diagnostic.Create(
            RedundantOverride, method.Identifier.GetLocation(), type.Name, name));
    }

    /// <summary>
    /// Reports a model the plan cannot rebuild, naming the constructor parameters that block it.
    /// </summary>
    /// <param name="context">The analysis context.</param>
    private static void AnalyzeType(SyntaxNodeAnalysisContext context)
    {
        var declaration = (ClassDeclarationSyntax)context.Node;

        if (declaration.Modifiers.Any(m => m.ValueText is "abstract" or "static")) return;
        // GetDeclaredSymbol, not ContainingSymbol: for a class declaration the containing symbol is
        // the namespace, so the cast below silently never matched and this rule reported nothing.
        if (context.SemanticModel.GetDeclaredSymbol(declaration) is not INamedTypeSymbol type) return;
        if (!IsModel(type)) return;
        if (ClonePlanGenerator.CollectConstructorParameters(type, isModel: true) is not null) return;

        var constructors = type.InstanceConstructors
            .Where(c => c.DeclaredAccessibility is Accessibility.Public or Accessibility.Internal)
            .Where(c => !c.IsStatic && c.Parameters.Length > 0)
            .ToList();

        if (constructors.Count == 0) return;

        var widest = constructors.Max(c => c.Parameters.Length);
        var candidates = constructors.Where(c => c.Parameters.Length == widest).ToList();

        // An ambiguous overload set is a different situation from a missing field, and saying
        // "add a backing field" would send someone to fix the wrong thing.
        var reason = candidates.Count > 1
            ? $"It declares {candidates.Count} constructors taking {widest} arguments, so nothing "
              + "records which one this instance was built with."
            : "These constructor parameters have no member holding their value: "
              + string.Join(", ", candidates[0].Parameters
                  .Where(p => p.RefKind != RefKind.None || ClonePlanGenerator.FindSource(type, p) is null)
                  .Select(p => $"'{p.Name}'"))
              + ".";

        context.ReportDiagnostic(Diagnostic.Create(
            Unreproducible, declaration.Identifier.GetLocation(), type.Name, reason));
    }

    /// <summary>
    /// Determines whether the body is exactly one object creation returned.
    /// </summary>
    /// <param name="method">The override to inspect.</param>
    /// <returns><see langword="true"/> when the body reconstructs and does nothing else.</returns>
    private static bool IsSingleReturnOfNewObject(MethodDeclarationSyntax method)
    {
        var expression = method.ExpressionBody?.Expression;

        if (expression is null)
        {
            if (method.Body is null || method.Body.Statements.Count != 1) return false;
            if (method.Body.Statements[0] is not ReturnStatementSyntax { Expression: { } returned })
            {
                return false;
            }

            expression = returned;
        }

        // An object initializer sets state the constructor did not, which the plan does not replay.
        return expression is ObjectCreationExpressionSyntax { Initializer: null };
    }

    /// <summary>
    /// True when the override only calls its own sibling and adds nothing.
    /// </summary>
    /// <param name="method">The override being analysed.</param>
    /// <param name="name">The override's name.</param>
    /// <returns><see langword="true"/> for a body that is exactly <c>SomeSibling()</c>.</returns>
    /// <remarks>
    /// <para>
    /// This class is not merely redundant, it is FATAL. The bases define
    /// <c>Clone() =&gt; DeepCopy()</c>, so a type that also defines <c>DeepCopy() =&gt; Clone()</c>
    /// closes a two-frame cycle as soon as its own real <c>Clone</c> is removed. 227 types carried
    /// that forwarder and 85 of them were already cyclic; <c>SuperNet</c> crashed the test host with
    /// a stack overflow after 12015 repetitions.
    /// </para>
    /// <para>
    /// It is also the deletion hazard the rest of this analyzer does not model. Proving the BASE
    /// reproduces an override says nothing about whether a SIBLING in the same type delegates to
    /// what is being removed, so removing <c>Clone</c> is correct in isolation and fatal next to a
    /// forwarder. Reporting the forwarder means the deletion loop removes BOTH, and the pair cannot
    /// regrow into a cycle.
    /// </para>
    /// </remarks>
    private static bool IsPureForwarder(MethodDeclarationSyntax method, string name)
    {
        var expression = method.ExpressionBody?.Expression;

        if (expression is null)
        {
            if (method.Body is null || method.Body.Statements.Count != 1) return false;
            if (method.Body.Statements[0] is not ReturnStatementSyntax { Expression: { } returned })
            {
                return false;
            }

            expression = returned;
        }

        // Only an unqualified or this-qualified call, and never to itself -- `Clone() => Clone()`
        // would be its own infinite recursion rather than a forwarder to a sibling.
        var invoked = expression switch
        {
            InvocationExpressionSyntax { ArgumentList.Arguments.Count: 0 } call => call.Expression switch
            {
                IdentifierNameSyntax id => id.Identifier.ValueText,
                MemberAccessExpressionSyntax { Expression: ThisExpressionSyntax } member
                    => member.Name.Identifier.ValueText,
                _ => null,
            },
            _ => null,
        };

        return invoked is "Clone" or "DeepCopy" or "CreateNewInstance" && invoked != name;
    }

    /// <summary>
    /// Determines whether the library treats this type as a model.
    /// </summary>
    /// <param name="type">The type to classify.</param>
    /// <returns><see langword="true"/> when it declares the full model surface.</returns>
    private static bool IsModel(INamedTypeSymbol type)
        => type.AllInterfaces.Any(i => i.Name == "IFullModel");
}
