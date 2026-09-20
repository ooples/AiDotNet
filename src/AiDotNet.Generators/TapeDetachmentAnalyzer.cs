using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Reports a second network read through <c>Predict</c> inside the loss lambda passed to
/// <c>TrainWithCustomLoss</c>, where the value it returns is detached from the gradient tape.
/// </summary>
/// <remarks>
/// <para>
/// <c>NeuralNetworkBase.Predict</c> wraps its forward in <c>using var noGrad = new NoGradScope&lt;T&gt;()</c>
/// (and switches to eval mode). That is exactly right for inference and exactly wrong inside a tape: the
/// tensor it hands back carries no gradient history, so any loss term built from it is a CONSTANT. Nothing
/// throws and no shape is wrong -- the affected half of the objective simply stops contributing, and the
/// trained network quietly optimizes whatever else is in the loss.
/// </para>
/// <para>
/// Measured cost of the silence: <c>ReinforcementLearning/Agents/SACAgent</c> read both critics this way, so
/// its actor ascended only its own entropy and never once followed <c>dQ/da</c>; and
/// <c>NeuralNetworks/ConditionalGAN</c> read the discriminator this way, so its generator had no adversarial
/// signal at all. Both shipped green, because the model-level liveness invariant only asks whether ANY
/// parameter moved and the critics / discriminator in those models do train.
/// </para>
/// <para>
/// The fix is always the same and the mechanism already exists: <c>ForwardForTraining</c> is public, virtual
/// and tape-tracked, and <c>TrainWithCustomLoss</c> collects only the tensors of the network it was called
/// on -- so the second network supplies gradient through the loss without being updated by that step.
/// </para>
/// <para>
/// Deliberately narrow, to stay at zero false positives. It fires only on an invocation literally named
/// <c>Predict</c> that sits inside a lambda passed as an argument to <c>TrainWithCustomLoss</c>. A
/// <c>Predict</c> call outside a tape is ordinary inference and is untouched, and a detached read that is
/// genuinely intended (a target network's bootstrap value, say) should be hoisted out of the lambda, which
/// also silences this rule and documents the intent at the same time.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class TapeDetachmentAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A Predict call inside a custom-loss lambda, whose result cannot carry gradient.</summary>
    private static readonly DiagnosticDescriptor DetachedPredictInTape = new(
        "AIDN101",
        "Predict inside a custom-loss lambda detaches the gradient",
        "'{0}' runs inside the loss lambda passed to '{1}', but Predict wraps its forward in a "
            + "NoGradScope, so the tensor it returns is detached from the tape and contributes no "
            + "gradient -- the loss term built from it is a constant. Call ForwardForTraining(...) "
            + "instead, which is tape-tracked; TrainWithCustomLoss collects only the trained network's "
            + "tensors, so the other network supplies gradient without being updated here",
        "AiDotNet.Correctness",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "A detached read inside a tape silences half an objective without failing: the "
            + "update still runs, the loss is still finite, and only the term that mattered stops "
            + "moving. Promote to Error once the existing violations are at zero.");

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(DetachedPredictInTape);

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(AnalyzeInvocation, SyntaxKind.InvocationExpression);
    }

    private static void AnalyzeInvocation(SyntaxNodeAnalysisContext context)
    {
        var invocation = (InvocationExpressionSyntax)context.Node;
        if (InvokedName(invocation) is not "Predict") return;

        // The Predict must sit inside a lambda that is itself an argument to a tape-running call.
        string? tapeMethod = EnclosingTapeMethod(invocation);
        if (tapeMethod is null) return;

        context.ReportDiagnostic(Diagnostic.Create(
            DetachedPredictInTape,
            invocation.GetLocation(),
            invocation.Expression.ToString(),
            tapeMethod));
    }

    /// <summary>The simple name of the method an invocation targets, ignoring any receiver.</summary>
    private static string? InvokedName(InvocationExpressionSyntax invocation) => invocation.Expression switch
    {
        MemberAccessExpressionSyntax member => member.Name.Identifier.Text,
        IdentifierNameSyntax identifier => identifier.Identifier.Text,
        _ => null,
    };

    /// <summary>
    /// Walks outward from <paramref name="node"/> looking for a lambda (or anonymous method) that is passed
    /// as an argument to a tape-running call, and returns that call's name. Returns null when the node is
    /// not inside such a lambda, which is the ordinary-inference case this rule must not touch.
    /// </summary>
    private static string? EnclosingTapeMethod(SyntaxNode node)
    {
        for (SyntaxNode? current = node; current is not null; current = current.Parent)
        {
            bool isLambda = current is SimpleLambdaExpressionSyntax
                or ParenthesizedLambdaExpressionSyntax
                or AnonymousMethodExpressionSyntax;
            if (!isLambda) continue;

            // A lambda only matters here when it is handed to the tape runner as an argument.
            if (current.Parent is not ArgumentSyntax argument) continue;
            if (argument.Parent is not ArgumentListSyntax list) continue;
            if (list.Parent is not InvocationExpressionSyntax outer) continue;

            if (InvokedName(outer) is { } name && IsTapeRunner(name))
                return name;
        }

        return null;
    }

    /// <summary>
    /// The calls that evaluate their loss callback under a live <c>GradientTape</c>. Kept as an explicit
    /// list rather than a heuristic so a method that merely takes a lambda cannot start reporting.
    /// </summary>
    private static bool IsTapeRunner(string name)
        => name is "TrainWithCustomLoss" or "TrainWithCustomLossNoStep";
}
