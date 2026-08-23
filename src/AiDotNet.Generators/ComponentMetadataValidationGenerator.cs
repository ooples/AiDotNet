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
    // Diagnostic descriptors for Tier 2 component attribute pairing
    private static readonly DiagnosticDescriptor ComponentTypeMissingPipelineStage = new(
        id: "AIDN060",
        title: "Component has [ComponentType] but is missing [PipelineStage]",
        messageFormat: "Component '{0}' has [ComponentType] but is missing [PipelineStage]. All components should declare which pipeline stage they operate in.",
        category: "AiDotNet.ComponentMetadata",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor PipelineStageMissingComponentType = new(
        id: "AIDN061",
        title: "Component has [PipelineStage] but is missing [ComponentType]",
        messageFormat: "Component '{0}' has [PipelineStage] but is missing [ComponentType]. All pipeline components should declare their component type.",
        category: "AiDotNet.ComponentMetadata",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

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

    private static readonly DiagnosticDescriptor InvalidLayerGradientContract = new(
        id: "AIDN086",
        title: "Layer declares a contradictory gradient contract",
        messageFormat: "Layer '{0}' has an invalid [LayerProperty] gradient contract: {1}",
        category: "AiDotNet.ComponentMetadata",
        defaultSeverity: DiagnosticSeverity.Error,
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

    // Tier 2 component attribute names
    private const string ComponentTypeAttr = "AiDotNet.Attributes.ComponentTypeAttribute";
    private const string PipelineStageAttr = "AiDotNet.Attributes.PipelineStageAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // ComponentCandidate was a struct, but it CARRIED an INamedTypeSymbol -- a value type
        // wrapped around a reference that roots the whole Compilation leaks exactly as badly as the
        // bare symbol would. It now carries the metadata name instead, and the symbol is
        // re-resolved at the point of validation.
        //
        // Same scope limit as ModelMetadataValidationGenerator: the validators report straight to
        // SourceProductionContext, and the generated-output diff cannot see diagnostics, so the
        // validation logic is left untouched and only the retention is fixed. AIDN050/051/052 and
        // AIDN07x counts are compared across the build instead.
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
        if (kind != ComponentKind.None)
            return new ComponentCandidate(GeneratorHelpers.MetadataNameOf(symbol), kind);

        // Also include classes that have [ComponentType] or [PipelineStage] for Tier 2 validation
        if (HasComponentTypeOrPipelineStage(symbol))
            return new ComponentCandidate(GeneratorHelpers.MetadataNameOf(symbol), ComponentKind.General);

        return null;
    }

    private static bool HasComponentTypeOrPipelineStage(INamedTypeSymbol symbol)
    {
        var attrs = symbol.GetAttributes();
        foreach (var attr in attrs)
        {
            if (attr.AttributeClass is not null)
            {
                string fullName = attr.AttributeClass.ToDisplayString();
                if (fullName == ComponentTypeAttr || fullName == PipelineStageAttr)
                    return true;
            }
        }
        return false;
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

        var seen = new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal);

        foreach (var candidate in candidates)
        {
            if (candidate is null)
                continue;

            var metadataName = candidate.Value.MetadataName;
            if (metadataName.Length == 0)
                continue;
            if (!seen.Add(metadataName))
                continue;

            if (GeneratorHelpers.ResolveSourceType(compilation, metadataName) is not INamedTypeSymbol symbol)
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

            // Tier 2: Validate [ComponentType] / [PipelineStage] pairing on all candidates
            ValidateComponentPipelinePairing(context, symbol);
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

        var property = attrs.FirstOrDefault(attr =>
            attr.AttributeClass?.ToDisplayString() == LayerPropertyAttr);
        if (property is null)
            return;

        bool isTrainable = GetNamedBool(property, "IsTrainable", defaultValue: true);
        bool supportsBackpropagation = GetNamedBool(property, "SupportsBackpropagation", defaultValue: true);
        bool usesSurrogateGradient = GetNamedBool(property, "UsesSurrogateGradient", defaultValue: false);
        bool trainsViaCustomLoss = GetNamedBool(property, "TrainsViaCustomLoss", defaultValue: false);

        string? invalidReason = null;
        if (usesSurrogateGradient && trainsViaCustomLoss)
            invalidReason = "UsesSurrogateGradient and TrainsViaCustomLoss are mutually exclusive";
        else if (usesSurrogateGradient && (!isTrainable || !supportsBackpropagation))
            invalidReason = "a surrogate-gradient layer must be trainable and support backpropagation";
        else if (trainsViaCustomLoss && (!isTrainable || supportsBackpropagation))
            invalidReason = "a custom-loss layer must be trainable and set SupportsBackpropagation = false because its Forward output is not the gradient objective";

        if (invalidReason is not null)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                InvalidLayerGradientContract,
                location,
                name,
                invalidReason));
        }
    }

    private static bool GetNamedBool(AttributeData attribute, string name, bool defaultValue)
    {
        foreach (var argument in attribute.NamedArguments)
        {
            if (argument.Key == name && argument.Value.Value is bool value)
                return value;
        }

        return defaultValue;
    }

    private static void ValidateComponentPipelinePairing(SourceProductionContext context, INamedTypeSymbol symbol)
    {
        var attrs = symbol.GetAttributes();
        bool hasComponentType = HasAttributeByFullName(attrs, ComponentTypeAttr);
        bool hasPipelineStage = HasAttributeByFullName(attrs, PipelineStageAttr);

        if (hasComponentType && !hasPipelineStage)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                ComponentTypeMissingPipelineStage,
                symbol.Locations.FirstOrDefault(),
                symbol.Name));
        }

        if (hasPipelineStage && !hasComponentType)
        {
            context.ReportDiagnostic(Diagnostic.Create(
                PipelineStageMissingComponentType,
                symbol.Locations.FirstOrDefault(),
                symbol.Name));
        }
    }

    private static bool HasAttributeByFullName(ImmutableArray<AttributeData> attrs, string fullName)
    {
        foreach (var attr in attrs)
        {
            if (attr.AttributeClass is not null &&
                attr.AttributeClass.ToDisplayString() == fullName)
            {
                return true;
            }
        }
        return false;
    }

    private static bool HasAttributeEndingWith(ImmutableArray<AttributeData> attrs, string suffix)
    {
        // Match only AiDotNet attributes to avoid false positives from third-party assemblies
        const string aiDotNetPrefix = "AiDotNet.";
        foreach (var attr in attrs)
        {
            if (attr.AttributeClass is not null)
            {
                string fullName = attr.AttributeClass.ToDisplayString();
                if (fullName.EndsWith(suffix, System.StringComparison.Ordinal) &&
                    fullName.StartsWith(aiDotNetPrefix, System.StringComparison.Ordinal))
                {
                    return true;
                }
            }
        }
        return false;
    }

    /// <summary>
    /// A candidate as plain values. Holding the INamedTypeSymbol here rooted the entire Compilation
    /// from cached pipeline state, which a readonly struct does nothing to prevent.
    /// </summary>
    private readonly struct ComponentCandidate : System.IEquatable<ComponentCandidate>
    {
        public string MetadataName { get; }
        public ComponentKind Kind { get; }

        public ComponentCandidate(string metadataName, ComponentKind kind)
        {
            MetadataName = metadataName;
            Kind = kind;
        }

        public bool Equals(ComponentCandidate other)
            => Kind == other.Kind
            && string.Equals(MetadataName, other.MetadataName, System.StringComparison.Ordinal);

        public override bool Equals(object? obj) => obj is ComponentCandidate c && Equals(c);

        public override int GetHashCode()
        {
            unchecked
            {
                return ((MetadataName?.GetHashCode() ?? 0) * 31) + (int)Kind;
            }
        }
    }


    private enum ComponentKind
    {
        None,
        Activation,
        Loss,
        Layer,
        General
    }
}
