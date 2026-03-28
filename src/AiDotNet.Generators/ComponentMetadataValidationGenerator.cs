using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental source generator that validates all concrete activation function,
/// loss function, and layer classes have the required metadata attributes.
/// Reports errors for any class that extends the known base types but is missing
/// the required [*Property], [*Category], and [*Task] attributes.
/// </summary>
[Generator]
public class ComponentMetadataValidationGenerator : IIncrementalGenerator
{
    // Diagnostic descriptors
    // NOTE: Temporarily set to Warning while annotations are being added across all
    // component classes. Will be restored to Error once all components are annotated.
    private static readonly DiagnosticDescriptor MissingActivationAttributes = new(
        id: "AIDN050",
        title: "Activation function missing required metadata attributes",
        messageFormat: "Activation function '{0}' is missing required attribute '[{1}]'. All IActivationFunction implementations must have [ActivationProperty], [ActivationCategory], and [ActivationTask].",
        category: "AiDotNet.ComponentMetadata",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor MissingLossAttributes = new(
        id: "AIDN051",
        title: "Loss function missing required metadata attributes",
        messageFormat: "Loss function '{0}' is missing required attribute '[{1}]'. All LossFunctionBase subclasses must have [LossProperty], [LossCategory], and [LossTask].",
        category: "AiDotNet.ComponentMetadata",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor MissingLayerAttributes = new(
        id: "AIDN052",
        title: "Layer missing required metadata attributes",
        messageFormat: "Layer '{0}' is missing required attribute '[{1}]'. All LayerBase subclasses must have [LayerProperty], [LayerCategory], and [LayerTask].",
        category: "AiDotNet.ComponentMetadata",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    // Base type / interface prefixes
    private const string IActivationFunctionPrefix = "AiDotNet.Interfaces.IActivationFunction<";
    private const string LossFunctionBasePrefix = "AiDotNet.LossFunctions.LossFunctionBase<";
    private const string ISelfSupervisedLossPrefix = "AiDotNet.Interfaces.ISelfSupervisedLoss<";
    private const string LayerBasePrefix = "AiDotNet.NeuralNetworks.Layers.LayerBase<";

    // Attribute names
    private const string ActivationPropertyAttr = "AiDotNet.Attributes.ActivationPropertyAttribute";
    private const string ActivationCategoryAttr = "AiDotNet.Attributes.ActivationCategoryAttribute";
    private const string ActivationTaskAttr = "AiDotNet.Attributes.ActivationTaskAttribute";

    private const string LossPropertyAttr = "AiDotNet.Attributes.LossPropertyAttribute";
    private const string LossCategoryAttr = "AiDotNet.Attributes.LossCategoryAttribute";
    private const string LossTaskAttr = "AiDotNet.Attributes.LossTaskAttribute";

    private const string LayerPropertyAttr = "AiDotNet.Attributes.LayerPropertyAttribute";
    private const string LayerCategoryAttr = "AiDotNet.Attributes.LayerCategoryAttribute";
    private const string LayerTaskAttr = "AiDotNet.Attributes.LayerTaskAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var classDeclarations = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => IsCandidate(node),
            transform: static (ctx, _) => GetComponentClassOrNull(ctx))
            .Where(static s => s is not null);

        var collected = classDeclarations.Collect().Combine(context.CompilationProvider);

        context.RegisterSourceOutput(collected, static (spc, source) =>
        {
            var (candidates, compilation) = source;
            Execute(spc, candidates, compilation);
        });
    }

    private static bool IsCandidate(SyntaxNode node)
    {
        if (node is not ClassDeclarationSyntax cds)
            return false;
        if (cds.BaseList is null || cds.BaseList.Types.Count == 0)
            return false;
        foreach (var modifier in cds.Modifiers)
        {
            if (modifier.Text == "abstract")
                return false;
        }
        return true;
    }

    private static ComponentCandidate? GetComponentClassOrNull(GeneratorSyntaxContext ctx)
    {
        var symbol = ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol;
        if (symbol is null || symbol.IsAbstract)
            return null;

        var kind = ClassifyComponent(symbol);
        if (kind == ComponentKind.None)
            return null;

        return new ComponentCandidate(symbol, kind);
    }

    private static ComponentKind ClassifyComponent(INamedTypeSymbol symbol)
    {
        // Check interfaces for IActivationFunction<T>
        foreach (var iface in symbol.AllInterfaces)
        {
            if (iface.IsGenericType)
            {
                var display = iface.OriginalDefinition.ToDisplayString();
                if (display.StartsWith(IActivationFunctionPrefix, System.StringComparison.Ordinal))
                    return ComponentKind.Activation;
                if (display.StartsWith(ISelfSupervisedLossPrefix, System.StringComparison.Ordinal))
                    return ComponentKind.Loss;
            }
        }

        // Check base type chain for LossFunctionBase<T> and LayerBase<T>
        var baseType = symbol.BaseType;
        while (baseType is not null)
        {
            if (baseType.IsGenericType)
            {
                var display = baseType.OriginalDefinition.ToDisplayString();
                if (display.StartsWith(LossFunctionBasePrefix, System.StringComparison.Ordinal))
                    return ComponentKind.Loss;
                if (display.StartsWith(LayerBasePrefix, System.StringComparison.Ordinal))
                    return ComponentKind.Layer;
            }
            baseType = baseType.BaseType;
        }

        return ComponentKind.None;
    }

    private static void Execute(
        SourceProductionContext context,
        ImmutableArray<ComponentCandidate?> candidates,
        Compilation compilation)
    {
        if (candidates.IsDefaultOrEmpty)
            return;

        var seen = new System.Collections.Generic.HashSet<INamedTypeSymbol>(SymbolEqualityComparer.Default);

        foreach (var candidate in candidates)
        {
            if (candidate is null)
                continue;

            var symbol = candidate.Value.Symbol;
            if (!seen.Add(symbol))
                continue;

            switch (candidate.Value.Kind)
            {
                case ComponentKind.Activation:
                    ValidateActivation(context, symbol);
                    break;
                case ComponentKind.Loss:
                    ValidateLoss(context, symbol);
                    break;
                case ComponentKind.Layer:
                    ValidateLayer(context, symbol);
                    break;
            }
        }
    }

    private static void ValidateActivation(SourceProductionContext context, INamedTypeSymbol symbol)
    {
        var attrs = symbol.GetAttributes();
        var location = symbol.Locations.FirstOrDefault();
        var name = symbol.Name;

        if (!HasAttributeEndingWith(attrs, "ActivationPropertyAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingActivationAttributes, location, name, "ActivationProperty"));

        if (!HasAttributeEndingWith(attrs, "ActivationCategoryAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingActivationAttributes, location, name, "ActivationCategory"));

        if (!HasAttributeEndingWith(attrs, "ActivationTaskAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingActivationAttributes, location, name, "ActivationTask"));
    }

    private static void ValidateLoss(SourceProductionContext context, INamedTypeSymbol symbol)
    {
        var attrs = symbol.GetAttributes();
        var location = symbol.Locations.FirstOrDefault();
        var name = symbol.Name;

        if (!HasAttributeEndingWith(attrs, "LossPropertyAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingLossAttributes, location, name, "LossProperty"));

        if (!HasAttributeEndingWith(attrs, "LossCategoryAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingLossAttributes, location, name, "LossCategory"));

        if (!HasAttributeEndingWith(attrs, "LossTaskAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingLossAttributes, location, name, "LossTask"));
    }

    private static void ValidateLayer(SourceProductionContext context, INamedTypeSymbol symbol)
    {
        var attrs = symbol.GetAttributes();
        var location = symbol.Locations.FirstOrDefault();
        var name = symbol.Name;

        if (!HasAttributeEndingWith(attrs, "LayerPropertyAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingLayerAttributes, location, name, "LayerProperty"));

        if (!HasAttributeEndingWith(attrs, "LayerCategoryAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingLayerAttributes, location, name, "LayerCategory"));

        if (!HasAttributeEndingWith(attrs, "LayerTaskAttribute"))
            context.ReportDiagnostic(Diagnostic.Create(MissingLayerAttributes, location, name, "LayerTask"));
    }

    private static bool HasAttributeEndingWith(ImmutableArray<AttributeData> attrs, string suffix)
    {
        foreach (var attr in attrs)
        {
            if (attr.AttributeClass is not null &&
                attr.AttributeClass.ToDisplayString().EndsWith(suffix, System.StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }

    private readonly struct ComponentCandidate
    {
        public INamedTypeSymbol Symbol { get; }
        public ComponentKind Kind { get; }

        public ComponentCandidate(INamedTypeSymbol symbol, ComponentKind kind)
        {
            Symbol = symbol;
            Kind = kind;
        }
    }

    private enum ComponentKind
    {
        None,
        Activation,
        Loss,
        Layer
    }
}
