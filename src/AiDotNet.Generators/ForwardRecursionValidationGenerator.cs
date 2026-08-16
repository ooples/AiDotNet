using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Build-time rule against <c>base.Forward(...)</c> called from inside a <c>ForwardTraced</c> override,
/// which is unconditional infinite recursion.
/// </summary>
/// <remarks>
/// <para>
/// <c>LayerBase.Forward</c> is a NON-VIRTUAL recording wrapper: it calls the observer, then dispatches
/// to <c>ForwardTraced</c> virtually. So <c>base.Forward(input)</c> written inside a
/// <c>ForwardTraced</c> override does not reach the base class's computation at all - it re-enters the
/// SAME override through virtual dispatch and recurses until the stack overflows. The intended call is
/// <c>base.ForwardTraced(input)</c>, which is non-virtual and lands on the base implementation.
/// </para>
/// <para>
/// WHY THIS IS A RULE AND NOT A FIXED BUG. Five adapters had it at once - LongLoRA, LoRA-FA, LoRA+,
/// PiSSA and VB-LoRA - every one of them a subclass of <c>LoRAAdapterBase</c>, which overrides
/// <c>ForwardTraced</c> and not <c>Forward</c>. The mistake is not a typo, it is what the naming
/// invites: "call the base class's forward pass" reads as <c>base.Forward</c>, and it compiles, and
/// the comment above each call said exactly that. Nothing had ever forwarded these adapters, so all
/// five sat there until a sweep that constructs every layer walked into them and aborted the test run
/// with a StackOverflowException - which no <c>catch</c> can trap, so it takes the whole process down
/// rather than failing one test.
/// </para>
/// <para>
/// Error, not warning, and no ladder: unlike the shape-annotation backlog there is no population of
/// existing violations to work through - the five are fixed - so anything this reports is new, and
/// every occurrence is a total breakage of the layer it appears in.
/// </para>
/// </remarks>
[Generator]
public class ForwardRecursionValidationGenerator : IIncrementalGenerator
{
    /// <summary>The wrapper methods that dispatch back into a traced override.</summary>
    /// <remarks>
    /// Each pairs with the <c>ForwardTraced*</c> override it dispatches to, so calling any of them
    /// from inside the corresponding override recurses. The dictionary/params overloads were added
    /// when multi-input tracing was closed, and they have the same hazard as the single-input one.
    /// </remarks>
    private static readonly string[] TracedOverrideNames =
    {
        "ForwardTraced", "ForwardTracedPorts", "ForwardTracedMany",
    };

    private static readonly DiagnosticDescriptor BaseForwardRecursionDescriptor = new(
        id: "ADNTRACE001",
        title: "base.Forward(...) inside a ForwardTraced override is infinite recursion",
        messageFormat: "'{0}' calls base.Forward(...) from inside its '{1}' override. Forward is a "
                       + "non-virtual wrapper that dispatches back to '{1}', so this re-enters the "
                       + "same method and recurses until the stack overflows - it never reaches the "
                       + "base class. A StackOverflowException cannot be caught, so this aborts the "
                       + "whole test host rather than failing one test. Call 'base.{1}(...)' instead, "
                       + "which is non-virtual and lands on the base implementation.",
        category: "AiDotNet.Correctness",
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
                            Expression: BaseExpressionSyntax,
                            Name.Identifier.ValueText: "Forward",
                        },
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
                    BaseForwardRecursionDescriptor, f.Value.Location, f.Value.TypeName, f.Value.MethodName));
            }
        });
    }

    // A plain tuple rather than a record struct: this project targets netstandard2.0, which has no
    // IsExternalInit, so positional record properties do not compile here.
    private static (Location Location, string TypeName, string MethodName)? Analyze(GeneratorSyntaxContext ctx)
    {
        var invocation = (InvocationExpressionSyntax)ctx.Node;

        // Walk out to the enclosing method. Only a ForwardTraced* override is a recursion; a
        // base.Forward call from a helper, a property, or a plain Forward override is not this defect
        // and flagging it would make the rule the kind that gets switched off.
        var method = invocation.FirstAncestorOrSelf<MethodDeclarationSyntax>();
        if (method is null) return null;

        string methodName = method.Identifier.ValueText;
        if (System.Array.IndexOf(TracedOverrideNames, methodName) < 0) return null;

        // A local function inside the override has its own name; the ancestor walk already stops at
        // the nearest MethodDeclaration, so reaching here means the call really is in the override
        // body. Confirm it is an override rather than a same-named method that shadows nothing.
        if (ctx.SemanticModel.GetDeclaredSymbol(method) is not IMethodSymbol symbol || !symbol.IsOverride)
            return null;

        return (invocation.GetLocation(), symbol.ContainingType?.Name ?? "<unknown>", methodName);
    }
}
