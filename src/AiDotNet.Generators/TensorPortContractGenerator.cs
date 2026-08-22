using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Generates tensor-port and model-input contracts, and reports statically visible sequential
/// value-domain mismatches before a model can be run.
/// </summary>
[Generator]
public sealed class TensorPortContractGenerator : IIncrementalGenerator
{
    private const string PortAttributeName = "AiDotNet.Attributes.TensorPortAttribute";
    private const string RankRouteAttributeName = "AiDotNet.Attributes.RankRoutedInputDomainAttribute";
    private const string ShapeConstraintAttributeName = "AiDotNet.Attributes.ModelInputShapeConstraintAttribute";
    private const string GenerateMethodAttributeName = "AiDotNet.Attributes.GenerateInputContractAttribute";
    private const string TensorInputAttributeName = "AiDotNet.Attributes.TensorInputAttribute";
    private const string GraphContractTypeName = "AiDotNet.NeuralNetworks.Layers.LayerGraphContract";

    private static readonly DiagnosticDescriptor IntegerRangeRequired = new(
        "ADNPORT001",
        "Integer-index port has no range",
        "Port '{0}' on '{1}' accepts integer indices but declares neither MaxExclusiveMember nor MaxExclusiveResolver. Set one to the vocabulary/range owner so generated callers can create legal IDs.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Index ports must publish a legal half-open range so fixtures and graph validation can produce useful guidance.");

    private static readonly DiagnosticDescriptor MissingContractMember = new(
        "ADNPORT002",
        "Tensor-port contract member does not exist",
        "Port '{0}' on '{1}' refers to member '{2}', but that field, property or method does not exist. Declare the member or correct the attribute name.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DuplicatePort = new(
        "ADNPORT003",
        "Duplicate tensor-port declaration",
        "'{0}' declares more than one {1} port named '{2}'. Remove the duplicate or give each semantic port a unique stable name.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor SequentialDomainMismatch = new(
        "ADNPORT004",
        "Sequential layer value domains are incompatible",
        "'{0}' produces {1}, but the next layer '{2}' requires {3}. Put independent lookups in a generated composite/named branch instead of a flat sequential list.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A lookup layer cannot consume the continuous activations produced by an earlier layer.");

    private static readonly DiagnosticDescriptor InvalidShapeConstraint = new(
        "ADNPORT005",
        "Invalid model input-shape constraint",
        "'{0}' declares MinimumElementCountMember '{1}', but that field, property or method does not exist. Declare the member or correct the attribute name.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor GeneratedContractRequiresPartial = new(
        "ADNPORT006",
        "Generated tensor contract type must be partial",
        "'{0}' uses a generated tensor-port or model-input contract and must be declared partial. Add the partial modifier so the generator can emit the contract without hand-written overrides.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Partial types let generated contracts share the model or layer declaration without hand-written overrides.");

    private static readonly DiagnosticDescriptor InvalidContractValues = new(
        "ADNPORT007",
        "Generated tensor contract contains impossible values",
        "'{0}' has an invalid generated tensor contract: {1}. Correct the attribute values; ranks must be positive when set, layer indexes cannot be negative, and ExactRank cannot be smaller than MinimumRank.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Impossible rank and geometry declarations are rejected during compilation instead of failing a generated fixture or model forward pass.");

    private static readonly DiagnosticDescriptor InvalidContractMemberSignature = new(
        "ADNPORT008",
        "Tensor-contract member has an invalid signature",
        "Port '{0}' on '{1}' uses '{2}', but that member must be {3}. Change the member signature so the generated contract is deterministic and type-safe.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor InvalidGeneratedForward = new(
        "ADNPORT009",
        "Generated input-forward declaration is ambiguous",
        "'{0}' must declare exactly one [GenerateInputContract] method whose parameters are Tensor<T> values. {1}.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DuplicateStablePortId = new(
        "ADNPORT010",
        "Tensor-port stable identity is ambiguous",
        "'{0}' declares stable port id '{1}' more than once in input variant '{2}'. Stable IDs must remain unique across inherited and local declarations.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor InvalidPortRelationship = new(
        "ADNPORT011",
        "Tensor-port relationship is not executable",
        "'{0}' has an invalid relationship in input variant '{1}': {2}.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Derived/defaulted ports and shape relations must be resolvable without runtime guesswork.");

    private static readonly DiagnosticDescriptor AmbiguousVariantSignature = new(
        "ADNPORT012",
        "Input variants cannot be distinguished",
        "'{0}' variants {1} require the same external port set [{2}]. Give each overload a structurally distinct required signature.",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private enum Domain
    {
        Unspecified = 0,
        Continuous = 1,
        IntegerIndices = 2,
        BooleanMask = 3,
        AdditiveMask = 4,
        Deferred = 5,
        Custom = 6,
    }

    private enum Direction
    {
        Input = 0,
        Output = 1,
    }

    private enum Source
    {
        External = 0,
        Derived = 1,
        Defaulted = 2,
        Internal = 3,
    }

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Symbols must not live in cached pipeline state -- an ISymbol roots the whole Compilation.
        // Same fix and same trade-off as the rest of this series: the pipeline carries metadata
        // names, the symbol is re-resolved at the point of use, and CompilationProvider costs
        // nothing in caching terms because a symbol pipeline could never cache to begin with.
        var types = context.SyntaxProvider.CreateSyntaxProvider(
                static (node, _) => node is TypeDeclarationSyntax declaration
                    && (declaration.AttributeLists.Count > 0
                        || declaration.Members.OfType<MethodDeclarationSyntax>()
                            .Any(method => method.AttributeLists.Count > 0)),
                static (ctx, _) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) is INamedTypeSymbol symbol
                    ? MetadataNameOf(symbol)
                    : null)
            .Where(static n => n is not null)
            .Select(static (n, _) => n ?? string.Empty)
            .Collect()
            .Combine(context.CompilationProvider);

        context.RegisterSourceOutput(types, static (spc, source) => EmitContracts(spc, source.Left, source.Right));

        var factoryFindings = context.SyntaxProvider.CreateSyntaxProvider(
                static (node, _) => node is MethodDeclarationSyntax method
                    && method.DescendantNodes().OfType<YieldStatementSyntax>().Any(),
                static (ctx, _) => InspectFactory(ctx))
            .Where(static findings => !findings.IsDefaultOrEmpty)
            .Collect();

        context.RegisterSourceOutput(factoryFindings, static (spc, groups) =>
        {
            foreach (var group in groups)
                foreach (var finding in group)
                    spc.ReportDiagnostic(Diagnostic.Create(
                        SequentialDomainMismatch,
                        finding.Location,
                        finding.Producer,
                        Describe(finding.Produced),
                        finding.Consumer,
                        Describe(finding.Required)));
        });
    }

    private static void EmitContracts(SourceProductionContext spc, ImmutableArray<string> metadataNames, Compilation compilation)
    {
        var resolvedNames = new HashSet<string>(System.StringComparer.Ordinal);
        var resolvedSymbols = new List<INamedTypeSymbol?>();
        foreach (var metadataName in metadataNames)
        {
            if (metadataName.Length == 0 || !resolvedNames.Add(metadataName)) continue;
            if (compilation.GetTypeByMetadataName(metadataName) is INamedTypeSymbol resolved)
                resolvedSymbols.Add(resolved);
        }
        var symbols = resolvedSymbols.ToImmutableArray();

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var type in symbols)
        {
            if (type is null || !seen.Add(type.ToDisplayString())) continue;

            var ownPortAttributes = type.GetAttributes()
                .Where(a => a.AttributeClass?.ToDisplayString() == PortAttributeName)
                .ToList();
            var declaredPorts = (ownPortAttributes.Count == 0
                    ? ownPortAttributes
                    : PortAttributesIncludingBases(type))
                .Select(ParsePort)
                .Where(p => p is not null)
                .Cast<Port>()
                .ToList();

            var generatedMethods = type.GetMembers()
                .OfType<IMethodSymbol>()
                .Where(method => method.GetAttributes().Any(attribute =>
                    attribute.AttributeClass?.ToDisplayString() == GenerateMethodAttributeName))
                .ToList();
            IMethodSymbol? generatedMethod = generatedMethods.Count == 1 ? generatedMethods[0] : null;
            var ports = declaredPorts.Count > 0 || generatedMethod is null
                ? declaredPorts
                : ParseMethodPorts(generatedMethod).ToList();

            var rankRoute = type.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == RankRouteAttributeName);
            var shapeConstraint = type.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == ShapeConstraintAttributeName);

            if (ports.Count == 0 && rankRoute is null && shapeConstraint is null
                && generatedMethods.Count == 0) continue;

            bool invalid = false;
            if (generatedMethods.Count > 1)
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    InvalidGeneratedForward,
                    type.Locations.FirstOrDefault(),
                    type.Name,
                    $"Found {generatedMethods.Count} annotated methods."));
                invalid = true;
            }
            else if (generatedMethod is not null)
            {
                var unsupported = generatedMethod.Parameters
                    .Where(parameter => !IsTensor(parameter.Type))
                    .Select(parameter => parameter.Name)
                    .ToArray();
                var generatedPorts = ParseMethodPorts(generatedMethod).ToArray();
                ITypeSymbol? numericType = type.TypeParameters.FirstOrDefault();
                string? reason = generatedMethod.Parameters.Length == 0
                    ? "The method has no input parameters."
                    : unsupported.Length > 0
                        ? "Non-tensor parameters: " + string.Join(", ", unsupported) + "."
                    : numericType is null || !IsTensorOf(generatedMethod.ReturnType, numericType)
                        ? "The method must return Tensor<T> for the component's numeric type."
                    : generatedMethod.IsStatic
                        ? "The method must be an instance method."
                    : generatedMethod.IsGenericMethod
                        ? "The method cannot introduce its own type parameters."
                    : generatedMethod.IsAbstract
                        ? "The method must contain the component's unique forward implementation."
                    : generatedMethod.Parameters.Any(parameter => parameter.RefKind != RefKind.None)
                        ? "Tensor parameters cannot use ref, in, or out modifiers."
                    : generatedPorts.Select(port => port.Variant).Distinct(StringComparer.Ordinal).Count() > 1
                        ? "One forward method cannot mix alternative input variants; declare one variant per component method."
                    : generatedPorts.Any(port => port.Source is Source.Derived or Source.Internal)
                        ? "Method parameters can only be External or Defaulted; derived/internal tensors belong inside the method body."
                    : generatedMethod.Parameters.Select((parameter, index) => (parameter, index))
                        .Any(item => generatedPorts[item.index].Source == Source.Defaulted
                                     && (!item.parameter.HasExplicitDefaultValue
                                         || item.parameter.ExplicitDefaultValue is not null))
                        ? "A Defaulted tensor parameter must declare '= null' so the generated bridge can omit it safely."
                    : null;
                if (reason is not null)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidGeneratedForward,
                        generatedMethod.Locations.FirstOrDefault(),
                        type.Name,
                        reason));
                    invalid = true;
                }
            }

            if (declaredPorts.Count > 0 && generatedMethod is not null)
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    InvalidGeneratedForward,
                    generatedMethod.Locations.FirstOrDefault(),
                    type.Name,
                    "Use either class-level [TensorPort] declarations or a generated forward method, not both."));
                invalid = true;
            }
            if (!IsPartial(type))
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    GeneratedContractRequiresPartial,
                    type.Locations.FirstOrDefault(),
                    type.Name));
                invalid = true;
            }

            if (rankRoute is not null)
            {
                int maximumIndexRank = rankRoute.ConstructorArguments[0].Value is int maximum ? maximum : 0;
                int layerIndex = rankRoute.ConstructorArguments[1].Value is int index ? index : -1;
                string? reason = maximumIndexRank <= 0
                    ? $"MaximumIndexRank is {maximumIndexRank}; it must be greater than zero"
                    : layerIndex < 0
                        ? $"LayerIndex is {layerIndex}; it must be zero or greater"
                        : null;
                if (reason is not null)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidContractValues,
                        rankRoute.ApplicationSyntaxReference?.GetSyntax().GetLocation()
                            ?? type.Locations.FirstOrDefault(),
                        type.Name,
                        reason));
                    invalid = true;
                }
            }

            foreach (var duplicate in ports.GroupBy(
                         p => (p.Direction, p.Variant, p.Name), PortKeyComparer.Instance)
                         .Where(g => g.Count() > 1))
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    DuplicatePort,
                    type.Locations.FirstOrDefault(),
                    type.Name,
                    duplicate.Key.Direction.ToString().ToLowerInvariant(),
                    duplicate.Key.Name));
                invalid = true;
            }


            foreach (var duplicate in ports
                         .Where(port => port.Direction == Direction.Input)
                         .GroupBy(port => (port.Variant, port.StableId), VariantStableIdComparer.Instance)
                         .Where(group => group.Count() > 1))
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    DuplicateStablePortId,
                    type.Locations.FirstOrDefault(),
                    type.Name,
                    duplicate.Key.StableId,
                    duplicate.Key.Variant));
                invalid = true;
            }

            foreach (var port in ports)
            {
                if (port.Domain == Domain.IntegerIndices
                    && string.IsNullOrWhiteSpace(port.MaxExclusiveMember)
                    && string.IsNullOrWhiteSpace(port.MaxExclusiveResolver)
                    && string.IsNullOrWhiteSpace(port.DomainResolver))
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        IntegerRangeRequired, type.Locations.FirstOrDefault(), port.Name, type.Name));
                    invalid = true;
                }

                foreach (string member in RequiredMembers(port))
                {
                    if (!HasMember(type, member))
                    {
                        spc.ReportDiagnostic(Diagnostic.Create(
                            MissingContractMember,
                            type.Locations.FirstOrDefault(),
                            port.Name,
                            type.Name,
                            member));
                        invalid = true;
                    }
                }

                string? memberSignatureError = ValidateMemberSignatures(type, port);
                if (memberSignatureError is not null)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidContractMemberSignature,
                        type.Locations.FirstOrDefault(),
                        port.Name,
                        type.Name,
                        memberSignatureError.Split('|')[0],
                        memberSignatureError.Split('|')[1]));
                    invalid = true;
                }

                string? portValueError = ValidatePortValues(port);
                if (portValueError is not null)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidContractValues,
                        type.Locations.FirstOrDefault(),
                        type.Name,
                        $"port '{port.Name}' {portValueError}"));
                    invalid = true;
                }
            }

            foreach (var port in ports.Where(port =>
                         !string.IsNullOrWhiteSpace(port.SameShapeAs)))
            {
                bool relationExists = ports.Any(candidate =>
                    candidate.Direction == Direction.Input
                    && candidate.Variant == port.Variant
                    && candidate.Name == port.SameShapeAs);
                if (!relationExists)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidContractValues,
                        type.Locations.FirstOrDefault(),
                        type.Name,
                        $"port '{port.Name}' refers to SameShapeAs '{port.SameShapeAs}', but that "
                        + $"input does not exist in variant '{port.Variant}'"));
                    invalid = true;
                }
            }

            foreach (var port in ports.Where(port => port.Direction == Direction.Input))
            {
                string? relationshipError = port.Source == Source.Derived
                    && string.IsNullOrWhiteSpace(port.SameShapeAs)
                    && string.IsNullOrWhiteSpace(port.ShapeMember)
                        ? $"derived port '{port.Name}' declares neither SameShapeAs nor ShapeMember"
                    : port.Source == Source.Defaulted && port.Required
                        ? $"defaulted port '{port.Name}' is also marked Required"
                        : null;
                if (relationshipError is not null)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidPortRelationship,
                        type.Locations.FirstOrDefault(),
                        type.Name,
                        port.Variant,
                        relationshipError));
                    invalid = true;
                }
            }

            foreach (var variantGroup in ports
                         .Where(port => port.Direction == Direction.Input)
                         .GroupBy(port => port.Variant, StringComparer.Ordinal))
            {
                var byName = variantGroup
                    .GroupBy(port => port.Name, StringComparer.Ordinal)
                    .ToDictionary(group => group.Key, group => group.First(), StringComparer.Ordinal);
                var reportedCyclePorts = new HashSet<string>(StringComparer.Ordinal);
                foreach (var start in variantGroup.Where(port => !string.IsNullOrWhiteSpace(port.SameShapeAs)))
                {
                    if (reportedCyclePorts.Contains(start.Name)) continue;
                    var seenRelations = new HashSet<string>(StringComparer.Ordinal) { start.Name };
                    var current = start;
                    while (!string.IsNullOrWhiteSpace(current.SameShapeAs)
                           && byName.TryGetValue(current.SameShapeAs!, out var related))
                    {
                        if (!seenRelations.Add(related.Name))
                        {
                            foreach (string name in seenRelations)
                                reportedCyclePorts.Add(name);
                            spc.ReportDiagnostic(Diagnostic.Create(
                                InvalidPortRelationship,
                                type.Locations.FirstOrDefault(),
                                type.Name,
                                variantGroup.Key,
                                $"SameShapeAs cycle reaches '{related.Name}'"));
                            invalid = true;
                            break;
                        }
                        current = related;
                    }
                }
            }

            foreach (var ambiguous in ports
                         .Where(port => port.Direction == Direction.Input)
                         .GroupBy(port => port.Variant, StringComparer.Ordinal)
                         .Select(group => new
                         {
                             Variant = group.Key,
                             RequiredKey = string.Join("\u001f", group
                                 .Where(port => port.Source == Source.External && port.Required)
                                 .Select(port => port.Name)
                                 .OrderBy(name => name, StringComparer.Ordinal))
                         })
                         .GroupBy(item => item.RequiredKey, StringComparer.Ordinal)
                         .Where(group => group.Count() > 1))
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    AmbiguousVariantSignature,
                    type.Locations.FirstOrDefault(),
                    type.Name,
                    string.Join(", ", ambiguous.Select(item => "'" + item.Variant + "'")),
                    string.Join(", ", ambiguous.Key.Split(new[] { '\u001f' }, StringSplitOptions.RemoveEmptyEntries))));
                invalid = true;
            }

            int minimumRank = 0;
            int minimumElements = 0;
            int exactRank = 0;
            int maximumRank = 0;
            int[] minimumAxisSizes = Array.Empty<int>();
            int[] axisDivisors = Array.Empty<int>();
            string? minimumElementsMember = null;
            if (shapeConstraint is not null)
            {
                minimumRank = NamedInt(shapeConstraint, "MinimumRank");
                minimumElements = NamedInt(shapeConstraint, "MinimumElementCount");
                exactRank = NamedInt(shapeConstraint, "ExactRank");
                maximumRank = NamedInt(shapeConstraint, "MaximumRank");
                minimumAxisSizes = NamedIntArray(shapeConstraint, "MinimumAxisSizes");
                axisDivisors = NamedIntArray(shapeConstraint, "AxisDivisors");
                minimumElementsMember = NamedString(shapeConstraint, "MinimumElementCountMember");
                string? invalidValueReason = minimumRank < 0
                    ? $"MinimumRank is {minimumRank}; it cannot be negative"
                    : exactRank < 0
                        ? $"ExactRank is {exactRank}; it cannot be negative"
                        : maximumRank < 0
                            ? $"MaximumRank is {maximumRank}; it cannot be negative"
                        : minimumElements < 0
                            ? $"MinimumElementCount is {minimumElements}; it cannot be negative"
                            : exactRank > 0 && minimumRank > exactRank
                                ? $"MinimumRank {minimumRank} exceeds ExactRank {exactRank}"
                                : maximumRank > 0 && minimumRank > maximumRank
                                    ? $"MinimumRank {minimumRank} exceeds MaximumRank {maximumRank}"
                                    : exactRank > 0 && maximumRank > 0 && exactRank > maximumRank
                                        ? $"ExactRank {exactRank} exceeds MaximumRank {maximumRank}"
                                    : exactRank > 0
                                      && Math.Max(minimumAxisSizes.Length, axisDivisors.Length) > exactRank
                                        ? $"per-axis rules exceed ExactRank {exactRank}"
                                    : maximumRank > 0
                                      && Math.Max(minimumAxisSizes.Length, axisDivisors.Length) > maximumRank
                                        ? $"per-axis rules exceed MaximumRank {maximumRank}"
                                        : minimumAxisSizes.Any(value => value < 0)
                                            ? "MinimumAxisSizes contains a negative value"
                                            : axisDivisors.Any(value => value < 0)
                                                ? "AxisDivisors contains a negative value"
                                : null;
                if (invalidValueReason is not null)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidContractValues,
                        shapeConstraint.ApplicationSyntaxReference?.GetSyntax().GetLocation()
                            ?? type.Locations.FirstOrDefault(),
                        type.Name,
                        invalidValueReason));
                    invalid = true;
                }
                if (!string.IsNullOrWhiteSpace(minimumElementsMember)
                    && !HasMember(type, minimumElementsMember!))
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidShapeConstraint,
                        type.Locations.FirstOrDefault(),
                        type.Name,
                        minimumElementsMember));
                    invalid = true;
                }
                else if (!string.IsNullOrWhiteSpace(minimumElementsMember)
                         && FindMember(type, minimumElementsMember!) is { } shapeMember
                         && !IsParameterlessIntValue(shapeMember))
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        InvalidContractMemberSignature,
                        shapeConstraint.ApplicationSyntaxReference?.GetSyntax().GetLocation()
                            ?? type.Locations.FirstOrDefault(),
                        "model input",
                        type.Name,
                        minimumElementsMember,
                        "an int field/property or a parameterless int method"));
                    invalid = true;
                }
            }

            if (invalid) continue;

            string ns = type.ContainingNamespace.IsGlobalNamespace
                ? string.Empty
                : type.ContainingNamespace.ToDisplayString();
            string typeParameters = type.TypeParameters.Length == 0
                ? string.Empty
                : "<" + string.Join(", ", type.TypeParameters.Select(p => p.Name)) + ">";

            var sb = new StringBuilder();
            sb.AppendLine("// <auto-generated/>");
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using System.Collections.Generic;");
            if (ns.Length > 0) sb.Append("namespace ").Append(ns).AppendLine(";").AppendLine();
            sb.Append("partial class ").Append(type.Name).AppendLine(typeParameters);
            sb.AppendLine("{");

            var inputs = ports.Where(p => p.Direction == Direction.Input).ToList();
            var outputs = ports.Where(p => p.Direction == Direction.Output).ToList();
            bool layerContractOwner = InheritsFromLayerBase(type);
            bool modelContractOwner = InheritsFromNeuralNetworkBase(type);

            if (inputs.Count > 0 && layerContractOwner && !Declares(type, "InputPorts"))
            {
                sb.AppendLine("    /// <summary>Generated from [TensorPort] declarations.</summary>");
                sb.AppendLine("    public override global::System.Collections.Generic.IReadOnlyList<global::AiDotNet.NeuralNetworks.Layers.LayerPort> InputPorts =>");
                sb.AppendLine("    [");
                foreach (var port in inputs)
                    sb.Append("        ").Append(PortExpression(type, port, isInput: true)).AppendLine(",");
                sb.AppendLine("    ];").AppendLine();
            }

            if (outputs.Count > 0 && layerContractOwner && !Declares(type, "OutputPorts"))
            {
                sb.AppendLine("    /// <summary>Generated from [TensorPort] declarations.</summary>");
                sb.AppendLine("    public override global::System.Collections.Generic.IReadOnlyList<global::AiDotNet.NeuralNetworks.Layers.LayerPort> OutputPorts =>");
                sb.AppendLine("    [");
                foreach (var port in outputs)
                    sb.Append("        ").Append(PortExpression(type, port, isInput: false)).AppendLine(",");
                sb.AppendLine("    ];").AppendLine();
            }

            if (inputs.Count > 0 && modelContractOwner && !Declares(type, "GetInputContract"))
            {
                sb.AppendLine("    /// <summary>Generated public model-boundary contract.</summary>");
                sb.AppendLine("    public override global::AiDotNet.NeuralNetworks.InputContractManifest GetInputContract(int[]? inputShape = null)");
                sb.AppendLine("    {");
                sb.AppendLine("        int[] shape = inputShape is { Length: > 0 } ? (int[])inputShape.Clone() : GetInputShape();");
                sb.AppendLine("        return new global::AiDotNet.NeuralNetworks.InputContractManifest(");
                sb.AppendLine("            GetType().Name,");
                sb.AppendLine("            new global::AiDotNet.NeuralNetworks.Layers.LayerPort[]");
                sb.AppendLine("            {");
                foreach (var port in inputs)
                    sb.Append("                ").Append(PortExpression(type, port, isInput: true, shapeOverride: "shape")).AppendLine(",");
                sb.AppendLine("            },");
                sb.AppendLine("            shapeConstraint: GetInputShapeConstraint());");
                sb.AppendLine("    }").AppendLine();
            }

            if (inputs.Count > 0 && !Declares(type, "GetInputDomain") && rankRoute is null)
            {
                sb.AppendLine("    /// <summary>Generated from the primary [TensorPort] declaration.</summary>");
                sb.Append("    public override global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain GetInputDomain(int[]? inputShape) => ")
                    .Append(DomainExpression(inputs[0], "inputShape")).AppendLine(";").AppendLine();
            }

            if (inputs.Count > 1 && InheritsFromLayerBase(type))
                EmitTypedInputFacade(sb, type, inputs);

            if (generatedMethod is not null && InheritsFromLayerBase(type))
                EmitGeneratedForwardBridge(sb, generatedMethod, inputs);

            if (inputs.Any(p => p.PropagatesInputDomain) && !Declares(type, "PropagatesInputDomain"))
            {
                sb.AppendLine("    /// <summary>Generated identity-domain propagation contract.</summary>");
                sb.AppendLine("    public override bool PropagatesInputDomain => true;").AppendLine();
            }

            if (rankRoute is not null && !Declares(type, "GetInputDomain"))
            {
                int maxRank = rankRoute.ConstructorArguments[0].Value is int mr ? mr : 0;
                int layerIndex = rankRoute.ConstructorArguments[1].Value is int li ? li : -1;
                sb.AppendLine("    /// <summary>Generated rank-routed external input-domain contract.</summary>");
                sb.AppendLine("    public override global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain GetInputDomain(int[]? inputShape)");
                sb.AppendLine("    {");
                sb.Append("        if (inputShape is not null && inputShape.Length <= ").Append(maxRank)
                    .Append(" && Layers.Count > ").Append(layerIndex)
                    .Append(" && Layers[").Append(layerIndex)
                    .AppendLine("] is global::AiDotNet.NeuralNetworks.Layers.LayerBase<T> front)");
                sb.AppendLine("            return front.GetInputDomain(inputShape);");
                sb.AppendLine("        return base.GetInputDomain(inputShape);");
                sb.AppendLine("    }").AppendLine();
            }

            if (shapeConstraint is not null && !Declares(type, "GetInputShapeConstraint"))
            {
                string elementExpression = !string.IsNullOrWhiteSpace(minimumElementsMember)
                    ? MemberExpression(type, minimumElementsMember!)
                    : minimumElements.ToString(System.Globalization.CultureInfo.InvariantCulture);
                sb.AppendLine("    /// <summary>Generated from [ModelInputShapeConstraint].</summary>");
                sb.Append("    public override global::AiDotNet.NeuralNetworks.ModelInputShapeConstraint GetInputShapeConstraint() => new(")
                    .Append(minimumRank).Append(", ").Append(elementExpression).Append(", ")
                    .Append(exactRank).Append(", ").Append(maximumRank).Append(", ")
                    .Append(IntArrayExpression(minimumAxisSizes)).Append(", ")
                    .Append(IntArrayExpression(axisDivisors)).AppendLine(");");
            }

            sb.AppendLine("}");
            string hint = (ns.Length == 0 ? string.Empty : ns.Replace('.', '_') + "_")
                + type.Name + ".TensorPorts.g.cs";
            spc.AddSource(hint, sb.ToString());
        }
    }

    private static ImmutableArray<FactoryFinding> InspectFactory(GeneratorSyntaxContext context)
    {
        var methodSyntax = (MethodDeclarationSyntax)context.Node;
        var method = context.SemanticModel.GetDeclaredSymbol(methodSyntax);
        if (method is null)
            return ImmutableArray<FactoryFinding>.Empty;

        var creations = methodSyntax.DescendantNodes()
            .OfType<YieldStatementSyntax>()
            .Where(y => y.Expression is not null)
            .Select(y => (
                Yield: y,
                Type: context.SemanticModel.GetTypeInfo(y.Expression!).Type as INamedTypeSymbol,
                IsGraphRoot: IsGraphRoot(y.Expression!, context.SemanticModel)))
            .Where(x => x.Type is not null)
            .OrderBy(x => x.Yield.SpanStart)
            .ToList();

        if (creations.Count < 2) return ImmutableArray<FactoryFinding>.Empty;

        var builder = ImmutableArray.CreateBuilder<FactoryFinding>();
        Domain produced = DeclaredDomain(creations[0].Type!, Direction.Output, Domain.Continuous);
        for (int i = 1; i < creations.Count; i++)
        {
            var producer = creations[i - 1];
            var consumer = creations[i];
            if (consumer.IsGraphRoot)
            {
                produced = DeclaredDomain(
                    consumer.Type!, Direction.Output, Domain.Continuous);
                continue;
            }
            Domain required = DeclaredDomain(consumer.Type!, Direction.Input, Domain.Continuous);
            if (!Accepts(required, produced))
            {
                builder.Add(new FactoryFinding(
                    consumer.Yield.GetLocation(),
                    producer.Type!.Name,
                    consumer.Type!.Name,
                    produced,
                    required));
            }

            Domain declaredOutput = DeclaredDomain(
                consumer.Type!, Direction.Output, Domain.Continuous);
            if (declaredOutput != Domain.Unspecified)
                produced = declaredOutput;
        }

        return builder.ToImmutable();
    }

    private static bool IsGraphRoot(ExpressionSyntax expression, SemanticModel semanticModel)
    {
        if (expression is not InvocationExpressionSyntax invocation) return false;
        var method = semanticModel.GetSymbolInfo(invocation).Symbol as IMethodSymbol;
        return method?.ContainingType.ToDisplayString() == GraphContractTypeName
               && method.Name is "FromExternalInput" or "FromDerivedInput";
    }

    private static bool Accepts(Domain required, Domain produced)
        => required == Domain.Unspecified || produced == Domain.Unspecified
           || required == Domain.Continuous || required == produced;

    private static Domain DeclaredDomain(INamedTypeSymbol type, Direction direction, Domain fallback)
    {
        for (INamedTypeSymbol? current = type; current is not null; current = current.BaseType)
        {
            foreach (var attr in current.GetAttributes())
            {
                if (attr.AttributeClass?.ToDisplayString() != PortAttributeName
                    || attr.ConstructorArguments.Length < 3
                    || attr.ConstructorArguments[1].Value is not int rawDirection
                    || rawDirection != (int)direction
                    || attr.ConstructorArguments[2].Value is not int rawDomain)
                    continue;
                return (Domain)rawDomain;
            }
        }
        return fallback;
    }

    private static IEnumerable<AttributeData> PortAttributesIncludingBases(INamedTypeSymbol type)
    {
        var hierarchy = new Stack<INamedTypeSymbol>();
        for (INamedTypeSymbol? current = type; current is not null; current = current.BaseType)
            hierarchy.Push(current);

        while (hierarchy.Count > 0)
        {
            foreach (var attribute in hierarchy.Pop().GetAttributes())
                if (attribute.AttributeClass?.ToDisplayString() == PortAttributeName)
                    yield return attribute;
        }
    }

    private static IEnumerable<Port> ParseMethodPorts(IMethodSymbol method)
    {
        foreach (var parameter in method.Parameters)
        {
            var attribute = parameter.GetAttributes().FirstOrDefault(item =>
                item.AttributeClass?.ToDisplayString() == TensorInputAttributeName);
            Domain domain = attribute?.ConstructorArguments.Length > 0
                && attribute.ConstructorArguments[0].Value is int rawDomain
                    ? (Domain)rawDomain
                    : Domain.Continuous;
            string name = attribute is null
                ? parameter.Name
                : NamedString(attribute, "Name") ?? parameter.Name;
            Source defaultSource = parameter.HasExplicitDefaultValue ? Source.Defaulted : Source.External;
            Source source = attribute is null
                ? defaultSource
                : (Source)NamedInt(attribute, "Source", (int)defaultSource);
            bool required = source == Source.External && !parameter.HasExplicitDefaultValue;

            yield return new Port(
                name,
                Direction.Input,
                domain,
                attribute is null ? 1 : NamedInt(attribute, "Role", 1),
                required,
                attribute is null ? null : NamedString(attribute, "MaxExclusiveMember"),
                attribute is null ? null : NamedString(attribute, "MaxExclusiveResolver"),
                attribute is null ? null : NamedString(attribute, "CustomProviderKey"),
                attribute is null ? null : NamedString(attribute, "DomainResolver"),
                null,
                false,
                name,
                source,
                attribute is null ? "default" : NamedString(attribute, "Variant") ?? "default",
                attribute is null ? 0 : NamedInt(attribute, "ExactRank"),
                attribute is null ? 0 : NamedInt(attribute, "MinimumRank"),
                attribute is null ? 0 : NamedInt(attribute, "MaximumRank"),
                attribute is null ? 0 : NamedInt(attribute, "MinimumElementCount"),
                attribute is null ? null : NamedString(attribute, "SameShapeAs"),
                attribute is null ? Array.Empty<int>() : NamedIntArray(attribute, "MinimumAxisSizes"),
                attribute is null ? Array.Empty<int>() : NamedIntArray(attribute, "AxisDivisors"));
        }
    }

    private static Port? ParsePort(AttributeData attr)
    {
        if (attr.ConstructorArguments.Length < 3
            || attr.ConstructorArguments[0].Value is not string name
            || attr.ConstructorArguments[1].Value is not int direction
            || attr.ConstructorArguments[2].Value is not int domain)
            return null;

        return new Port(
            name,
            (Direction)direction,
            (Domain)domain,
            NamedInt(attr, "Role"),
            NamedBool(attr, "Required", defaultValue: true),
            NamedString(attr, "MaxExclusiveMember"),
            NamedString(attr, "MaxExclusiveResolver"),
            NamedString(attr, "CustomProviderKey"),
            NamedString(attr, "DomainResolver"),
            NamedString(attr, "ShapeMember"),
            NamedBool(attr, "PropagatesInputDomain", defaultValue: false),
            NamedString(attr, "StableId") ?? name,
            (Source)NamedInt(attr, "Source"),
            NamedString(attr, "Variant") ?? "default",
            NamedInt(attr, "ExactRank"),
            NamedInt(attr, "MinimumRank"),
            NamedInt(attr, "MaximumRank"),
            NamedInt(attr, "MinimumElementCount"),
            NamedString(attr, "SameShapeAs"),
            NamedIntArray(attr, "MinimumAxisSizes"),
            NamedIntArray(attr, "AxisDivisors"));
    }

    private static string PortExpression(
        INamedTypeSymbol type,
        Port port,
        bool isInput,
        string? shapeOverride = null)
    {
        string shape = shapeOverride ?? (string.IsNullOrWhiteSpace(port.ShapeMember)
            ? (isInput ? "GetInputShape()" : "GetOutputShape()")
            : MemberExpression(type, port.ShapeMember!));
        string domain = DomainExpression(
            port,
            shapeOverride ?? (isInput ? "GetInputShape()" : "null"));
        return "new global::AiDotNet.NeuralNetworks.Layers.LayerPort("
            + Literal(port.Name) + ", " + shape + ", "
            + (port.Required ? "true" : "false") + ", " + domain + ", "
            + "(global::AiDotNet.NeuralNetworks.Layers.TensorPortRole)" + port.Role + ", "
            + Literal(port.StableId) + ", "
            + "(global::AiDotNet.NeuralNetworks.Layers.TensorPortSource)" + (int)port.Source + ", "
            + Literal(port.Variant) + ", " + ShapeConstraintExpression(port) + ")";
    }

    private static string ShapeConstraintExpression(Port port)
    {
        if (port.ExactRank == 0 && port.MinimumRank == 0 && port.MaximumRank == 0
            && port.MinimumElementCount == 0 && string.IsNullOrWhiteSpace(port.SameShapeAs)
            && port.MinimumAxisSizes.Length == 0 && port.AxisDivisors.Length == 0)
            return "global::AiDotNet.NeuralNetworks.Layers.PortShapeConstraint.None";

        return "new global::AiDotNet.NeuralNetworks.Layers.PortShapeConstraint { ExactRank = "
            + port.ExactRank + ", MinimumRank = " + port.MinimumRank
            + ", MaximumRank = " + port.MaximumRank
            + ", MinimumElementCount = " + port.MinimumElementCount
            + ", SameShapeAs = " + (port.SameShapeAs is null ? "null" : Literal(port.SameShapeAs))
            + ", MinimumAxisSizes = " + IntArrayExpression(port.MinimumAxisSizes)
            + ", AxisDivisors = " + IntArrayExpression(port.AxisDivisors) + " }";
    }

    private static void EmitTypedInputFacade(
        StringBuilder sb,
        INamedTypeSymbol type,
        IReadOnlyList<Port> inputs)
    {
        if (type.TypeParameters.Length == 0) return;
        string numericType = type.TypeParameters[0].Name;

        foreach (var variantGroup in inputs.GroupBy(port => port.Variant, StringComparer.Ordinal))
        {
            var variantPorts = variantGroup
                .Where(port => port.Source is Source.External or Source.Defaulted)
                .OrderByDescending(port => port.Required)
                .ToList();
            if (variantPorts.Count == 0) continue;
            if (variantPorts.Count == 1)
            {
                if (inputs.Select(port => port.Variant).Distinct(StringComparer.Ordinal).Count() <= 1)
                    continue;
                var port = variantPorts[0];
                sb.Append("    public global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                    .Append(numericType).Append("> Forward").Append(Identifier(variantGroup.Key))
                    .Append("(global::AiDotNet.Tensors.LinearAlgebra.Tensor<").Append(numericType)
                    .Append("> input) => Forward(new global::System.Collections.Generic.Dictionary<string, global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                    .Append(numericType).Append(">> { [").Append(Literal(port.Name))
                    .AppendLine("] = input });").AppendLine();
                continue;
            }

            string structName = string.Equals(variantGroup.Key, "default", StringComparison.Ordinal)
                ? "Inputs"
                : Identifier(variantGroup.Key) + "Inputs";
            sb.AppendLine("    /// <summary>Generated immutable typed input facade.</summary>");
            sb.Append("    public readonly struct ").Append(structName).AppendLine();
            sb.AppendLine("    {");
            foreach (var port in variantPorts)
            {
                string nullable = port.Required ? string.Empty : "?";
                sb.Append("        public global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                    .Append(numericType).Append(">").Append(nullable).Append(' ')
                    .Append(Identifier(port.Name)).AppendLine(" { get; }");
            }

            sb.Append("        public ").Append(structName).Append('(');
            for (int i = 0; i < variantPorts.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                var port = variantPorts[i];
                sb.Append("global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                    .Append(numericType).Append('>');
                if (!port.Required) sb.Append('?');
                sb.Append(' ').Append(CamelIdentifier(port.Name));
                if (!port.Required) sb.Append(" = null");
            }
            sb.AppendLine(")");
            sb.AppendLine("        {");
            foreach (var port in variantPorts)
                sb.Append("            ").Append(Identifier(port.Name)).Append(" = ")
                    .Append(CamelIdentifier(port.Name)).AppendLine(";");
            sb.AppendLine("        }");
            sb.AppendLine("    }").AppendLine();

            sb.Append("    public global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                .Append(numericType).Append("> Forward(").Append(structName).AppendLine(" inputs)");
            sb.AppendLine("    {");
            sb.Append("        var named = new global::System.Collections.Generic.Dictionary<string, global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                .Append(numericType).AppendLine(">>();");
            foreach (var port in variantPorts)
            {
                string property = "inputs." + Identifier(port.Name);
                if (port.Required)
                    sb.Append("        named[").Append(Literal(port.Name)).Append("] = ")
                        .Append(property).AppendLine(";");
                else
                    sb.Append("        if (").Append(property).Append(" is not null) named[")
                        .Append(Literal(port.Name)).Append("] = ").Append(property).AppendLine(";");
            }
            sb.AppendLine("        return Forward(named);");
            sb.AppendLine("    }").AppendLine();
        }
    }

    private static void EmitGeneratedForwardBridge(
        StringBuilder sb,
        IMethodSymbol method,
        IReadOnlyList<Port> inputs)
    {
        string numericType = method.ContainingType.TypeParameters.Length > 0
            ? method.ContainingType.TypeParameters[0].Name
            : "double";
        if (method.Parameters.Length == 1)
        {
            sb.Append("    protected override global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                .Append(numericType).Append("> ForwardTraced(global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
                .Append(numericType).AppendLine("> input)");
            sb.Append("        => ").Append(method.Name).AppendLine("(input);").AppendLine();
            return;
        }

        sb.Append("    protected override global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
            .Append(numericType).AppendLine("> ForwardTracedPorts(");
        sb.Append("        global::System.Collections.Generic.IReadOnlyDictionary<string, global::AiDotNet.Tensors.LinearAlgebra.Tensor<")
            .Append(numericType).AppendLine(">> inputs)");
        sb.Append("        => ").Append(method.Name).Append('(');
        for (int i = 0; i < method.Parameters.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            var parameter = method.Parameters[i];
            var port = inputs[i];
            if (parameter.HasExplicitDefaultValue && parameter.ExplicitDefaultValue is null)
                sb.Append("inputs.TryGetValue(").Append(Literal(port.Name)).Append(", out var ")
                    .Append(CamelIdentifier(port.Name)).Append(") ? ")
                    .Append(CamelIdentifier(port.Name)).Append(" : null");
            else
                sb.Append("inputs[").Append(Literal(port.Name)).Append(']');
        }
        sb.AppendLine(");").AppendLine();
    }

    private static string DomainExpression(Port port, string inputShapeExpression)
    {
        if (!string.IsNullOrWhiteSpace(port.DomainResolver))
            return port.DomainResolver + "(" + inputShapeExpression + ")";

        return port.Domain switch
        {
        Domain.Unspecified => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Unspecified",
        Domain.BooleanMask => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.BooleanMask",
        Domain.AdditiveMask => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.AdditiveMask",
        Domain.Deferred => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Deferred(\"The generated port has not been bound.\")",
        Domain.Custom => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Custom("
            + Literal(port.CustomProviderKey ?? string.Empty) + ")",
        Domain.IntegerIndices when !string.IsNullOrWhiteSpace(port.MaxExclusiveResolver)
            => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Indices("
                + port.MaxExclusiveResolver + "(" + inputShapeExpression + "))",
        Domain.IntegerIndices => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Indices("
            + port.MaxExclusiveMember + ")",
        _ => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Continuous",
        };
    }

    private static string MemberExpression(INamedTypeSymbol type, string member)
    {
        var symbol = FindMember(type, member);
        return symbol is IMethodSymbol ? member + "()" : member;
    }

    private static IEnumerable<string> RequiredMembers(Port port)
    {
        if (!string.IsNullOrWhiteSpace(port.MaxExclusiveMember)) yield return port.MaxExclusiveMember!;
        if (!string.IsNullOrWhiteSpace(port.MaxExclusiveResolver)) yield return port.MaxExclusiveResolver!;
        if (!string.IsNullOrWhiteSpace(port.DomainResolver)) yield return port.DomainResolver!;
        if (!string.IsNullOrWhiteSpace(port.ShapeMember)) yield return port.ShapeMember!;
    }

    private static string? ValidateMemberSignatures(INamedTypeSymbol type, Port port)
    {
        if (!string.IsNullOrWhiteSpace(port.MaxExclusiveMember))
        {
            var member = FindMember(type, port.MaxExclusiveMember!);
            if (member is not null && !IsParameterlessIntValue(member))
                return port.MaxExclusiveMember + "|an int field/property or a parameterless int method";
        }

        if (!string.IsNullOrWhiteSpace(port.MaxExclusiveResolver))
        {
            var member = FindMember(type, port.MaxExclusiveResolver!);
            if (member is not IMethodSymbol method || !IsInt(method.ReturnType)
                || method.Parameters.Length != 1 || !IsIntArray(method.Parameters[0].Type))
                return port.MaxExclusiveResolver + "|a method with signature int Method(int[]?)";
        }

        if (!string.IsNullOrWhiteSpace(port.DomainResolver))
        {
            var member = FindMember(type, port.DomainResolver!);
            if (member is not IMethodSymbol method || !IsLayerInputDomain(method.ReturnType)
                || method.Parameters.Length != 1 || !IsIntArray(method.Parameters[0].Type))
                return port.DomainResolver
                    + "|a method with signature LayerInputDomain Method(int[]?)";
        }

        if (!string.IsNullOrWhiteSpace(port.ShapeMember))
        {
            var member = FindMember(type, port.ShapeMember!);
            bool valid = member switch
            {
                IFieldSymbol field => IsIntArray(field.Type),
                IPropertySymbol property => IsIntArray(property.Type) && !property.IsWriteOnly,
                IMethodSymbol method => method.Parameters.Length == 0 && IsIntArray(method.ReturnType),
                _ => false
            };
            if (member is not null && !valid)
                return port.ShapeMember + "|an int[] field/property or a parameterless int[] method";
        }

        return null;
    }

    private static string? ValidatePortValues(Port port)
    {
        if (!string.IsNullOrWhiteSpace(port.DomainResolver)
            && (!string.IsNullOrWhiteSpace(port.MaxExclusiveMember)
                || !string.IsNullOrWhiteSpace(port.MaxExclusiveResolver)
                || !string.IsNullOrWhiteSpace(port.CustomProviderKey)))
            return "declares DomainResolver together with a fixed range/custom provider; the resolver must be the single domain authority";
        if (!string.IsNullOrWhiteSpace(port.MaxExclusiveMember)
            && !string.IsNullOrWhiteSpace(port.MaxExclusiveResolver))
            return "declares both MaxExclusiveMember and MaxExclusiveResolver; choose one range owner";
        if (port.Domain == Domain.Custom && string.IsNullOrWhiteSpace(port.CustomProviderKey))
            return "uses a Custom domain without a CustomProviderKey";
        if (port.Domain != Domain.Custom && !string.IsNullOrWhiteSpace(port.CustomProviderKey))
            return "declares CustomProviderKey but does not use the Custom domain";
        if (port.ExactRank < 0 || port.MinimumRank < 0 || port.MaximumRank < 0)
            return "contains a negative rank";
        if (port.MinimumElementCount < 0)
            return "has a negative MinimumElementCount";
        if (port.ExactRank > 0 && port.MinimumRank > port.ExactRank)
            return $"has MinimumRank {port.MinimumRank} greater than ExactRank {port.ExactRank}";
        if (port.MaximumRank > 0 && port.MinimumRank > port.MaximumRank)
            return $"has MinimumRank {port.MinimumRank} greater than MaximumRank {port.MaximumRank}";
        if (port.ExactRank > 0 && port.MaximumRank > 0 && port.ExactRank > port.MaximumRank)
            return $"has ExactRank {port.ExactRank} greater than MaximumRank {port.MaximumRank}";
        int axisRuleCount = Math.Max(port.MinimumAxisSizes.Length, port.AxisDivisors.Length);
        if (port.ExactRank > 0 && axisRuleCount > port.ExactRank)
            return $"declares {axisRuleCount} per-axis rules for ExactRank {port.ExactRank}";
        if (port.MaximumRank > 0 && axisRuleCount > port.MaximumRank)
            return $"declares {axisRuleCount} per-axis rules for MaximumRank {port.MaximumRank}";
        if (port.MinimumAxisSizes.Any(value => value < 0))
            return "has a negative per-axis minimum";
        if (port.AxisDivisors.Any(value => value < 0))
            return "has a negative axis divisor";
        if (!string.IsNullOrWhiteSpace(port.SameShapeAs)
            && string.Equals(port.SameShapeAs, port.Name, StringComparison.Ordinal))
            return "cannot declare SameShapeAs itself";
        if (string.IsNullOrWhiteSpace(port.StableId))
            return "has an empty StableId";
        if (string.IsNullOrWhiteSpace(port.Variant))
            return "has an empty Variant";
        return null;
    }

    private static bool IsParameterlessIntValue(ISymbol symbol) => symbol switch
    {
        IFieldSymbol field => IsInt(field.Type),
        IPropertySymbol property => IsInt(property.Type) && !property.IsWriteOnly,
        IMethodSymbol method => method.Parameters.Length == 0 && IsInt(method.ReturnType),
        _ => false
    };

    private static bool IsInt(ITypeSymbol type) => type.SpecialType == SpecialType.System_Int32;

    private static bool IsLayerInputDomain(ITypeSymbol type) => type is INamedTypeSymbol named
        && named.Name == "LayerInputDomain"
        && named.ContainingNamespace.ToDisplayString() == "AiDotNet.NeuralNetworks.Layers";

    private static bool IsIntArray(ITypeSymbol type) => type is IArrayTypeSymbol array
        && array.Rank == 1 && IsInt(array.ElementType);

    private static bool IsTensor(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol named) return false;
        return named.Name == "Tensor"
            && named.Arity == 1
            && named.ContainingNamespace.ToDisplayString() == "AiDotNet.Tensors.LinearAlgebra";
    }

    private static bool IsTensorOf(ITypeSymbol type, ITypeSymbol numericType)
        => type is INamedTypeSymbol named
           && IsTensor(named)
           && SymbolEqualityComparer.Default.Equals(named.TypeArguments[0], numericType);

    private static bool InheritsFromLayerBase(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? current = type.BaseType; current is not null; current = current.BaseType)
            if (current.Name == "LayerBase"
                && current.ContainingNamespace.ToDisplayString() == "AiDotNet.NeuralNetworks.Layers")
                return true;
        return false;
    }

    private static bool InheritsFromNeuralNetworkBase(INamedTypeSymbol type)
    {
        for (INamedTypeSymbol? current = type.BaseType; current is not null; current = current.BaseType)
            if (current.Name == "NeuralNetworkBase"
                && current.ContainingNamespace.ToDisplayString() == "AiDotNet.NeuralNetworks")
                return true;
        return false;
    }

    private static bool HasMember(INamedTypeSymbol type, string member) => FindMember(type, member) is not null;

    private static bool IsPartial(INamedTypeSymbol type)
        => type.DeclaringSyntaxReferences.Length > 0
           && type.DeclaringSyntaxReferences.All(reference =>
               reference.GetSyntax() is TypeDeclarationSyntax declaration
               && declaration.Modifiers.Any(modifier => modifier.ValueText == "partial"));

    private static ISymbol? FindMember(INamedTypeSymbol type, string member)
    {
        for (INamedTypeSymbol? current = type; current is not null; current = current.BaseType)
        {
            var found = current.GetMembers(member).FirstOrDefault();
            if (found is not null) return found;
        }
        return null;
    }

    private static bool Declares(INamedTypeSymbol type, string member)
        => type.GetMembers(member).Any();

    private static int NamedInt(AttributeData attr, string name, int defaultValue = 0)
        => attr.NamedArguments.FirstOrDefault(p => p.Key == name).Value.Value is int value
            ? value
            : defaultValue;

    private static int[] NamedIntArray(AttributeData attr, string name)
    {
        var argument = attr.NamedArguments.FirstOrDefault(pair => pair.Key == name).Value;
        if (argument.Kind != TypedConstantKind.Array || argument.IsNull)
            return Array.Empty<int>();
        return argument.Values
            .Where(value => value.Value is int)
            .Select(value => (int)value.Value!)
            .ToArray();
    }

    private static bool NamedBool(AttributeData attr, string name, bool defaultValue)
        => attr.NamedArguments.FirstOrDefault(p => p.Key == name).Value.Value is bool value ? value : defaultValue;

    private static string? NamedString(AttributeData attr, string name)
        => attr.NamedArguments.FirstOrDefault(p => p.Key == name).Value.Value as string;

    private static string Literal(string value)
        => "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

    private static string IntArrayExpression(IReadOnlyList<int> values)
        => values.Count == 0
            ? "global::System.Array.Empty<int>()"
            : "new int[] { " + string.Join(", ", values) + " }";

    private static string Identifier(string value)
    {
        var builder = new StringBuilder();
        bool capitalize = true;
        foreach (char character in value)
        {
            if (!char.IsLetterOrDigit(character))
            {
                capitalize = true;
                continue;
            }
            builder.Append(capitalize ? char.ToUpperInvariant(character) : character);
            capitalize = false;
        }
        if (builder.Length == 0) return "Value";
        if (char.IsDigit(builder[0])) builder.Insert(0, '_');
        return builder.ToString();
    }

    private static string CamelIdentifier(string value)
    {
        string identifier = Identifier(value);
        return identifier.Length == 0
            ? "value"
            : char.ToLowerInvariant(identifier[0]) + identifier.Substring(1);
    }

    private static string Describe(Domain domain) => domain switch
    {
        Domain.IntegerIndices => "integer indices",
        Domain.BooleanMask => "a Boolean mask",
        Domain.AdditiveMask => "an additive attention mask",
        Domain.Unspecified => "a preserved input domain",
        Domain.Deferred => "a deferred domain",
        Domain.Custom => "a custom domain",
        _ => "continuous values",
    };

    private sealed class PortKeyComparer : IEqualityComparer<(Direction Direction, string Variant, string Name)>
    {
        public static PortKeyComparer Instance { get; } = new();
        public bool Equals(
            (Direction Direction, string Variant, string Name) x,
            (Direction Direction, string Variant, string Name) y)
            => x.Direction == y.Direction
               && string.Equals(x.Variant, y.Variant, StringComparison.Ordinal)
               && string.Equals(x.Name, y.Name, StringComparison.Ordinal);
        public int GetHashCode((Direction Direction, string Variant, string Name) obj)
            => (((int)obj.Direction * 397) ^ StringComparer.Ordinal.GetHashCode(obj.Variant)) * 397
               ^ StringComparer.Ordinal.GetHashCode(obj.Name);
    }

    private sealed class VariantStableIdComparer : IEqualityComparer<(string Variant, string StableId)>
    {
        public static VariantStableIdComparer Instance { get; } = new();
        public bool Equals((string Variant, string StableId) x, (string Variant, string StableId) y)
            => string.Equals(x.Variant, y.Variant, StringComparison.Ordinal)
               && string.Equals(x.StableId, y.StableId, StringComparison.Ordinal);
        public int GetHashCode((string Variant, string StableId) obj)
            => StringComparer.Ordinal.GetHashCode(obj.Variant) * 397
               ^ StringComparer.Ordinal.GetHashCode(obj.StableId);
    }

    private sealed class Port
    {
        public Port(
            string name,
            Direction direction,
            Domain domain,
            int role,
            bool required,
            string? maxExclusiveMember,
            string? maxExclusiveResolver,
            string? customProviderKey,
            string? domainResolver,
            string? shapeMember,
            bool propagatesInputDomain,
            string stableId,
            Source source,
            string variant,
            int exactRank,
            int minimumRank,
            int maximumRank,
            int minimumElementCount,
            string? sameShapeAs,
            int[] minimumAxisSizes,
            int[] axisDivisors)
        {
            Name = name;
            Direction = direction;
            Domain = domain;
            Role = role;
            Required = required;
            MaxExclusiveMember = maxExclusiveMember;
            MaxExclusiveResolver = maxExclusiveResolver;
            CustomProviderKey = customProviderKey;
            DomainResolver = domainResolver;
            ShapeMember = shapeMember;
            PropagatesInputDomain = propagatesInputDomain;
            StableId = stableId;
            Source = source;
            Variant = variant;
            ExactRank = exactRank;
            MinimumRank = minimumRank;
            MaximumRank = maximumRank;
            MinimumElementCount = minimumElementCount;
            SameShapeAs = sameShapeAs;
            MinimumAxisSizes = minimumAxisSizes;
            AxisDivisors = axisDivisors;
        }

        public string Name { get; }
        public Direction Direction { get; }
        public Domain Domain { get; }
        public int Role { get; }
        public bool Required { get; }
        public string? MaxExclusiveMember { get; }
        public string? MaxExclusiveResolver { get; }
        public string? CustomProviderKey { get; }
        public string? DomainResolver { get; }
        public string? ShapeMember { get; }
        public bool PropagatesInputDomain { get; }
        public string StableId { get; }
        public Source Source { get; }
        public string Variant { get; }
        public int ExactRank { get; }
        public int MinimumRank { get; }
        public int MaximumRank { get; }
        public int MinimumElementCount { get; }
        public string? SameShapeAs { get; }
        public int[] MinimumAxisSizes { get; }
        public int[] AxisDivisors { get; }
    }

    private sealed class FactoryFinding
    {
        public FactoryFinding(
            Location location,
            string producer,
            string consumer,
            Domain produced,
            Domain required)
        {
            Location = location;
            Producer = producer;
            Consumer = consumer;
            Produced = produced;
            Required = required;
        }

        public Location Location { get; }
        public string Producer { get; }
        public string Consumer { get; }
        public Domain Produced { get; }
        public Domain Required { get; }
    }

    /// <summary>
    /// Builds the metadata name GetTypeByMetadataName expects, including the arity suffix for
    /// generics and '+' separators for nested types.
    /// </summary>
    private static string MetadataNameOf(INamedTypeSymbol symbol)
    {
        var name = symbol.MetadataName;
        for (var containing = symbol.ContainingType; containing is not null; containing = containing.ContainingType)
        {
            name = containing.MetadataName + "+" + name;
        }

        var ns = symbol.ContainingNamespace;
        if (ns is not null && !ns.IsGlobalNamespace)
        {
            name = ns.ToDisplayString() + "." + name;
        }

        return name;
    }
}
