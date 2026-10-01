using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Reports a layer factory that declares a <c>NeuralNetworkArchitecture&lt;T&gt;</c> parameter and
/// never reads it, so the stack it builds is sized from option defaults instead.
/// </summary>
/// <remarks>
/// <para>
/// The signature is a promise: a caller passing an architecture has every reason to expect the
/// layers to match it. Where the parameter is unread that promise is silently broken, and the
/// breakage is not cosmetic — <c>ResolveLazyLayerShapes</c> performs an architecture-driven warm-up,
/// so a lazy layer resolves its weights from the DECLARED shape while the forward runs on whatever
/// the options produced.
/// </para>
/// <para>
/// Measured across <c>LayerHelper</c>: 179 factories take an architecture and 104 never read it, and
/// 61 of the models calling those are constructed by a fixture that passes an explicit
/// architecture. The other 90 factories follow the established shape — architecture wins when
/// meaningfully set, options are the fallback:
/// </para>
/// <code>
/// int inputSize = architecture.CalculatedInputSize > 0
///     ? architecture.CalculatedInputSize
///     : numFeatures;
/// </code>
/// <para>
/// This is the ratchet for that convention, so the count can only go down. It is deliberately NOT a
/// rule that every factory must take an architecture: 270 factories take explicit paper-named
/// dimensions instead (<c>numMels</c>, <c>encoderDim</c>, <c>numBands</c>) which the model passes
/// from its own options, and for a vocoder or a Conformer that is the correct source of truth.
/// Taking the parameter and ignoring it is the defect; not taking it is a different, valid design.
/// </para>
/// <para>
/// Severity is <c>Info</c> while the backlog is open, and that is temporary by design. At
/// <c>Warning</c> this rule is a BUILD ERROR — <c>src/AiDotNet.csproj</c> sets
/// <c>TreatWarningsAsErrors</c> and AIDN106 is not in <c>WarningsNotAsErrors</c> — which is exactly
/// the ratchet wanted, but it cannot be switched on until the backlog is empty. Verified rather
/// than assumed: raising it to Warning produced 218 build errors across 104 factories. Flip it back
/// to Warning as the last step of clearing them.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class UnusedArchitectureParameterAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A factory that declares an architecture parameter and never reads it.</summary>
    private static readonly DiagnosticDescriptor UnreadArchitectureParameter = new(
        "AIDN106",
        "Layer factory ignores the architecture it was handed",
        "'{0}' declares a NeuralNetworkArchitecture parameter '{1}' and never reads it, so the "
            + "layer stack is sized from option defaults and a caller's architecture is silently "
            + "discarded. Either honour it — architecture.CalculatedInputSize when set, the option "
            + "otherwise, which is the shape the other 90 factories already use — or drop the "
            + "parameter so the signature stops promising something it does not do",
        "AiDotNet.ModelMetadata",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "An unread architecture parameter breaks the contract its own signature "
            + "declares, and ResolveLazyLayerShapes resolves lazy weights against that declared "
            + "shape while the forward runs on the shape the options produced.");

    private const string ArchitectureMetadataName = "NeuralNetworkArchitecture`1";

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(UnreadArchitectureParameter);

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

        // Scoped to the layer factories. A general "unused parameter" rule would fire on every
        // interface implementation and virtual override in the repository, where an unused
        // parameter is required by the signature rather than a mistake.
        if (!method.Identifier.Text.StartsWith("CreateDefault", System.StringComparison.Ordinal))
        {
            return;
        }

        var parameter = FindArchitectureParameter(method, context.SemanticModel);
        if (parameter is null)
        {
            return;
        }

        if (context.SemanticModel.GetDeclaredSymbol(parameter) is not IParameterSymbol symbol)
        {
            return;
        }

        // An expression-bodied factory that forwards its architecture on is reading it.
        SyntaxNode? body = method.Body is not null ? method.Body : method.ExpressionBody;
        if (body is null)
        {
            return;
        }

        foreach (var identifier in body.DescendantNodes().OfType<IdentifierNameSyntax>())
        {
            if (identifier.Identifier.Text != symbol.Name)
            {
                continue;
            }

            // Compare SYMBOLS, not names. A nested lambda or local function can shadow the
            // parameter, and counting a shadowed identifier as a read would silently excuse the
            // exact defect this rule exists to catch.
            if (SymbolEqualityComparer.Default.Equals(
                    context.SemanticModel.GetSymbolInfo(identifier).Symbol, symbol))
            {
                return;
            }
        }

        context.ReportDiagnostic(Diagnostic.Create(
            UnreadArchitectureParameter,
            parameter.GetLocation(),
            method.Identifier.Text,
            symbol.Name));
    }

    /// <summary>
    /// Returns the factory's <c>NeuralNetworkArchitecture&lt;T&gt;</c> parameter, if it has one.
    /// </summary>
    /// <remarks>
    /// Resolved through the semantic model rather than by matching the text of the type name, so an
    /// alias, a differently-qualified spelling or a nullable annotation all resolve to the same
    /// type. Matching on text is how the first census of this population missed every
    /// <c>internal</c> factory and under-counted it by fourteen.
    /// </remarks>
    private static ParameterSyntax? FindArchitectureParameter(
        MethodDeclarationSyntax method,
        SemanticModel semanticModel)
    {
        foreach (var parameter in method.ParameterList.Parameters)
        {
            if (parameter.Type is null)
            {
                continue;
            }

            var type = semanticModel.GetTypeInfo(parameter.Type).Type;

            // Unwrap the nullable annotation; NeuralNetworkArchitecture<T>? is the same promise.
            if (type is INamedTypeSymbol { IsGenericType: true } named
                && named.OriginalDefinition.MetadataName == ArchitectureMetadataName)
            {
                return parameter;
            }
        }

        return null;
    }
}
