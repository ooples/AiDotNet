using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps a trainable weight from being rebound to an engine result inside
/// <c>UpdateParameters</c>: reports <c>_field = Engine.Something(...)</c> and points at the
/// in-place form instead.
/// </summary>
/// <remarks>
/// <para>
/// <c>Engine.*</c> results can come from the per-step <c>TensorArena</c>, which recycles its
/// buffers once the step ends. Assigning one to a trainable-parameter field makes that arena
/// scratch the weight, so the next step's allocations overwrite it and training silently
/// produces NaN or garbage rather than failing. Nothing throws; the numbers are simply wrong.
/// </para>
/// <para>
/// The in-place forms exist and are zero-allocation -- <c>TensorAddInPlace(a, b)</c> is
/// <c>a[i] += b[i]</c>, <c>TensorSubtractInPlace(a, b)</c> is <c>a[i] -= b[i]</c>. They write
/// through the field's existing storage, so the weight keeps the buffer it was registered with
/// and stays valid across steps. This is also what the GPU path already did: layers that had a
/// <c>DirectGpuTensorEngine</c> branch called <c>SgdMomentumUpdateGpu(_gamma, ...)</c> in place
/// while the CPU branch beside it rebound the field.
/// </para>
/// <para>
/// This rule exists because the count grew rather than shrank while the issue (#1842) was open:
/// ~31 sites when reported, 40 a month later, 587 across 83 files when finally measured. That is
/// the signature of a pattern being copied from neighbouring layers, which prose cannot stop.
/// All 587 are fixed; the analyzer is the ratchet that keeps it at zero.
/// </para>
/// <para>
/// Deliberately narrow, to stay at zero false positives. It fires only inside
/// <c>UpdateParameters</c>, only on instance fields, and never on a field the codebase has
/// already marked as scratch or buffer storage -- those are meant to be rebound.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class ParameterUpdateInPlaceAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A persistent field rebound to an Engine result inside UpdateParameters.</summary>
    private static readonly DiagnosticDescriptor ParameterReboundToEngineResult = new(
        "AIDN100",
        "Trainable parameter reassigned to an Engine result in UpdateParameters",
        "'{0}' is reassigned to the result of '{1}' inside UpdateParameters, which can hand the "
            + "parameter an arena-scratch buffer that is recycled before the next step reads it -- "
            + "training then produces wrong numbers with no exception. Update it in place instead, "
            + "e.g. 'Engine.TensorSubtractInPlace({0}, delta)' or 'Engine.TensorAddInPlace({0}, delta)', "
            + "so the parameter keeps its own registered storage",
        "AiDotNet.ParameterAutomation",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Engine results may be arena scratch. A trainable parameter bound to one is "
            + "silently corrupted when the arena recycles, so the failure surfaces as bad training "
            + "results rather than an error.");

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(ParameterReboundToEngineResult);

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
        if (method.Identifier.Text != "UpdateParameters") return;
        if (method.Body is null && method.ExpressionBody is null) return;

        var model = context.SemanticModel;
        if (model.GetDeclaredSymbol(method) is not IMethodSymbol methodSymbol) return;
        var registrations =
            ParameterMemberSemanticModel.GetRegistrationClassifications(methodSymbol.ContainingType);

        foreach (var assignment in method.DescendantNodes()
            .OfType<AssignmentExpressionSyntax>()
            .Where(candidate => candidate.IsKind(SyntaxKind.SimpleAssignmentExpression)))
        {
            var field = ResolveCurrentInstanceRootField(assignment.Left, model);
            if (field is null) continue;
            if (field.IsStatic || field.IsConst) continue;
            var classification =
                ParameterMemberSemanticModel.ClassifyWithRegistrations(field, registrations);
            if (classification.Kind != ParameterMemberSemanticModel.Kind.Trainable) continue;

            string? engineCall = GetEngineCallName(assignment.Right);
            if (engineCall is null) continue;

            context.ReportDiagnostic(Diagnostic.Create(
                ParameterReboundToEngineResult,
                assignment.GetLocation(),
                field.Name,
                engineCall));
        }
    }

    /// <summary>
    /// Resolves the current instance's field for both direct assignments and container element
    /// assignments such as <c>_weights[i] = ...</c> or <c>this._weights[key] = ...</c>.
    /// </summary>
    private static IFieldSymbol? ResolveCurrentInstanceRootField(
        ExpressionSyntax left,
        SemanticModel model)
    {
        ExpressionSyntax candidate = left;
        while (true)
        {
            switch (candidate)
            {
                case ParenthesizedExpressionSyntax parenthesized:
                    candidate = parenthesized.Expression;
                    continue;

                case ElementAccessExpressionSyntax elementAccess:
                    candidate = elementAccess.Expression;
                    continue;

                case IdentifierNameSyntax identifier:
                    return model.GetSymbolInfo(identifier).Symbol as IFieldSymbol;

                case MemberAccessExpressionSyntax memberAccess:
                    if (model.GetSymbolInfo(memberAccess).Symbol is IFieldSymbol field)
                    {
                        return memberAccess.Expression is ThisExpressionSyntax or BaseExpressionSyntax
                            ? field
                            : null;
                    }
                    return null;

                default:
                    return null;
            }
        }
    }

    /// <summary>
    /// The invoked member name when the expression is (or wraps) an <c>Engine.X(...)</c> call.
    /// Walks the outermost invocation only -- a nested Engine call used as an ARGUMENT is fine,
    /// since only the value the field ends up bound to matters.
    /// </summary>
    private static string? GetEngineCallName(ExpressionSyntax expression)
    {
        if (expression is not InvocationExpressionSyntax invocation) return null;
        if (invocation.Expression is not MemberAccessExpressionSyntax member) return null;

        // Engine.X(...), this.Engine.X(...), or SomeEngine.X(...) where the receiver is the
        // layer's engine property. Matching on the receiver's NAME keeps this analyzer free of
        // a hard dependency on the Tensors assembly being resolvable during analysis.
        string receiver = member.Expression switch
        {
            IdentifierNameSyntax identifier => identifier.Identifier.Text,
            MemberAccessExpressionSyntax nested => nested.Name.Identifier.Text,
            _ => string.Empty,
        };

        if (!receiver.EndsWith("Engine", System.StringComparison.Ordinal)) return null;
        return $"{receiver}.{member.Name.Identifier.Text}";
    }
}
