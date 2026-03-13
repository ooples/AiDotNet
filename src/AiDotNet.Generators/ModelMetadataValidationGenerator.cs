using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that validates all concrete model classes
/// have the required metadata attributes and documentation.
/// </summary>
/// <remarks>
/// <para>
/// Automatically discovers all non-abstract classes implementing IFullModel anywhere in their
/// inheritance chain (via Roslyn's AllInterfaces which recursively walks all base types and
/// interfaces — no hardcoded type list required) and checks for:
/// </para>
/// <list type="bullet">
/// <item><description>Required attributes: ModelDomain, ModelCategory, ModelTask, ModelComplexity, ModelInput</description></item>
/// <item><description>XML documentation: summary, beginner-friendly remarks, examples</description></item>
/// <item><description>Paper URL validation</description></item>
/// </list>
/// </remarks>
[Generator]
public class ModelMetadataValidationGenerator : IIncrementalGenerator
{
    // Diagnostic descriptors
    private static readonly DiagnosticDescriptor MissingAttribute = new(
        id: "AIDN001",
        title: "Missing required model metadata attribute",
        messageFormat: "Model class '{0}' is missing required attribute '[{1}]'",
        category: "AiDotNet.ModelMetadata",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Every concrete model class must have all required metadata attributes: ModelDomain, ModelCategory, ModelTask, ModelComplexity, and ModelInput.");

    private static readonly DiagnosticDescriptor MissingSummary = new(
        id: "AIDN010",
        title: "Missing XML doc summary",
        messageFormat: "Model class '{0}' is missing XML doc summary",
        category: "AiDotNet.ModelMetadata",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor MissingBeginnerRemarks = new(
        id: "AIDN011",
        title: "Missing beginner-friendly remarks",
        messageFormat: "Model class '{0}' is missing beginner-friendly remarks (XML remarks with 'For Beginners' content)",
        category: "AiDotNet.ModelMetadata",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor MissingExample = new(
        id: "AIDN012",
        title: "Missing usage example",
        messageFormat: "Model class '{0}' is missing XML doc example block",
        category: "AiDotNet.ModelMetadata",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor InvalidPaperUrl = new(
        id: "AIDN020",
        title: "Invalid ModelPaper URL",
        messageFormat: "ModelPaper URL '{0}' on '{1}' is not well-formed (must start with https://)",
        category: "AiDotNet.ModelMetadata",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    // Fully-qualified attribute names (without "Attribute" suffix for matching)
    private const string ModelDomainAttributeName = "AiDotNet.Attributes.ModelDomainAttribute";
    private const string ModelCategoryAttributeName = "AiDotNet.Attributes.ModelCategoryAttribute";
    private const string ModelTaskAttributeName = "AiDotNet.Attributes.ModelTaskAttribute";
    private const string ModelComplexityAttributeName = "AiDotNet.Attributes.ModelComplexityAttribute";
    private const string ModelInputAttributeName = "AiDotNet.Attributes.ModelInputAttribute";
    private const string ModelPaperAttributeName = "AiDotNet.Attributes.ModelPaperAttribute";
    private const string ModelMetadataExemptAttributeName = "AiDotNet.Attributes.ModelMetadataExemptAttribute";

    // Interface/base type names to detect model classes
    private const string IFullModelName = "AiDotNet.Interfaces.IFullModel";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Syntax-first filter: find all non-abstract class declarations with base types
        var classDeclarations = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => GetModelClassOrNull(ctx))
            .Where(static s => s is not null);

        // Collect and combine with compilation
        var collected = classDeclarations.Collect().Combine(context.CompilationProvider);

        context.RegisterSourceOutput(collected, static (spc, source) =>
        {
            var (candidates, compilation) = source;
            Execute(spc, candidates, compilation);
        });
    }

    /// <summary>
    /// Fast syntax filter: only consider non-abstract class declarations that have base types.
    /// </summary>
    private static bool IsCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax cds)
            return false;

        // Must have a base list (inherits from something)
        if (cds.BaseList is null || cds.BaseList.Types.Count == 0)
            return false;

        // Must not be abstract
        foreach (var modifier in cds.Modifiers)
        {
            if (modifier.Text == "abstract")
                return false;
        }

        return true;
    }

    /// <summary>
    /// Semantic transform: resolve the class symbol and check if it implements IFullModel.
    /// </summary>
    private static INamedTypeSymbol? GetModelClassOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;

        if (ImplementsIFullModel(symbol))
            return symbol;

        return null;
    }

    /// <summary>
    /// Checks whether a type implements IFullModel&lt;,,&gt; anywhere in its interface hierarchy.
    /// </summary>
    private static bool ImplementsIFullModel(INamedTypeSymbol type)
    {
        foreach (var iface in type.AllInterfaces)
        {
            if (iface.IsGenericType &&
                iface.OriginalDefinition.ToDisplayString().StartsWith(IFullModelName, System.StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<INamedTypeSymbol?> candidates,
        Compilation compilation)
    {
        if (candidates.IsDefaultOrEmpty)
            return;

        // Resolve attribute type symbols for comparison
        var domainAttr = compilation.GetTypeByMetadataName(ModelDomainAttributeName);
        var categoryAttr = compilation.GetTypeByMetadataName(ModelCategoryAttributeName);
        var taskAttr = compilation.GetTypeByMetadataName(ModelTaskAttributeName);
        var complexityAttr = compilation.GetTypeByMetadataName(ModelComplexityAttributeName);
        var inputAttr = compilation.GetTypeByMetadataName(ModelInputAttributeName);
        var paperAttr = compilation.GetTypeByMetadataName(ModelPaperAttributeName);

        var exemptAttr = compilation.GetTypeByMetadataName(ModelMetadataExemptAttributeName);

        // If attributes don't exist in the compilation yet, skip validation
        if (domainAttr is null || categoryAttr is null || taskAttr is null ||
            complexityAttr is null || inputAttr is null)
        {
            return;
        }

        var seen = new System.Collections.Generic.HashSet<INamedTypeSymbol>(SymbolEqualityComparer.Default);
        foreach (var modelClass in candidates)
        {
            if (modelClass is null)
                continue;

            if (!seen.Add(modelClass))
                continue;

            // Skip classes marked with [ModelMetadataExempt]
            if (exemptAttr is not null && HasAttribute(modelClass.GetAttributes(), exemptAttr))
                continue;

            ValidateRequiredAttributes(context, modelClass, domainAttr, categoryAttr, taskAttr, complexityAttr, inputAttr);
            ValidatePaperUrls(context, modelClass, paperAttr);
            ValidateXmlDocumentation(context, modelClass);
        }
    }

    private static void ValidateRequiredAttributes(
        SourceProductionContext context,
        INamedTypeSymbol modelClass,
        INamedTypeSymbol domainAttr,
        INamedTypeSymbol categoryAttr,
        INamedTypeSymbol taskAttr,
        INamedTypeSymbol complexityAttr,
        INamedTypeSymbol inputAttr)
    {
        var attributes = modelClass.GetAttributes();
        var className = modelClass.Name;
        var location = modelClass.Locations.FirstOrDefault();

        if (!HasAttribute(attributes, domainAttr))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                MissingAttribute, location, className, "ModelDomain"));
        }

        if (!HasAttribute(attributes, categoryAttr))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                MissingAttribute, location, className, "ModelCategory"));
        }

        if (!HasAttribute(attributes, taskAttr))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                MissingAttribute, location, className, "ModelTask"));
        }

        if (!HasAttribute(attributes, complexityAttr))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                MissingAttribute, location, className, "ModelComplexity"));
        }

        if (!HasAttribute(attributes, inputAttr))
        {
            context.ReportDiagnostic(Diagnostic.Create(
                MissingAttribute, location, className, "ModelInput"));
        }
    }

    private static void ValidatePaperUrls(
        SourceProductionContext context,
        INamedTypeSymbol modelClass,
        INamedTypeSymbol? paperAttr)
    {
        if (paperAttr is null)
            return;

        var attributes = modelClass.GetAttributes();
        var className = modelClass.Name;
        var location = modelClass.Locations.FirstOrDefault();

        foreach (var attr in attributes)
        {
            if (!SymbolEqualityComparer.Default.Equals(attr.AttributeClass, paperAttr))
                continue;

            // ModelPaperAttribute(string title, string url) - url is the second constructor arg
            if (attr.ConstructorArguments.Length >= 2)
            {
                var url = attr.ConstructorArguments[1].Value as string;
                if (url is not null && !url.StartsWith("https://", System.StringComparison.OrdinalIgnoreCase))
                {
                    context.ReportDiagnostic(Diagnostic.Create(
                        InvalidPaperUrl, location, url, className));
                }
            }
        }
    }

    private static void ValidateXmlDocumentation(
        SourceProductionContext context,
        INamedTypeSymbol modelClass)
    {
        var className = modelClass.Name;
        var location = modelClass.Locations.FirstOrDefault();
        var xmlDoc = modelClass.GetDocumentationCommentXml();

        if (string.IsNullOrWhiteSpace(xmlDoc))
        {
            context.ReportDiagnostic(Diagnostic.Create(MissingSummary, location, className));
            context.ReportDiagnostic(Diagnostic.Create(MissingBeginnerRemarks, location, className));
            context.ReportDiagnostic(Diagnostic.Create(MissingExample, location, className));
            return;
        }

        if (!xmlDoc.Contains("<summary>"))
        {
            context.ReportDiagnostic(Diagnostic.Create(MissingSummary, location, className));
        }

        if (!xmlDoc.Contains("For Beginners"))
        {
            context.ReportDiagnostic(Diagnostic.Create(MissingBeginnerRemarks, location, className));
        }

        if (!xmlDoc.Contains("<example>") && !xmlDoc.Contains("<code>"))
        {
            context.ReportDiagnostic(Diagnostic.Create(MissingExample, location, className));
        }
    }

    private static bool HasAttribute(ImmutableArray<AttributeData> attributes, INamedTypeSymbol attributeType)
    {
        foreach (var attr in attributes)
        {
            if (SymbolEqualityComparer.Default.Equals(attr.AttributeClass, attributeType))
                return true;
        }

        return false;
    }
}
