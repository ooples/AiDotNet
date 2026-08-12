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
    private const string ValidateFactoryAttributeName = "AiDotNet.Attributes.ValidateSequentialLayerDomainsAttribute";

    private static readonly DiagnosticDescriptor IntegerRangeRequired = new(
        "ADNPORT001",
        "Integer-index port has no range",
        "Port '{0}' on '{1}' accepts integer indices but declares neither MaxExclusiveMember nor MaxExclusiveResolver",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "Index ports must publish a legal half-open range so fixtures and graph validation can produce useful guidance.");

    private static readonly DiagnosticDescriptor MissingContractMember = new(
        "ADNPORT002",
        "Tensor-port contract member does not exist",
        "Port '{0}' on '{1}' refers to member '{2}', but that field, property or method does not exist",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DuplicatePort = new(
        "ADNPORT003",
        "Duplicate tensor-port declaration",
        "'{0}' declares more than one {1} port named '{2}'",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor SequentialDomainMismatch = new(
        "ADNPORT004",
        "Sequential layer value domains are incompatible",
        "'{0}' produces {1}, but the next layer '{2}' requires {3}. Put independent lookups in a generated composite/named branch instead of a flat sequential list",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description: "A lookup layer cannot consume the continuous activations produced by an earlier layer.");

    private static readonly DiagnosticDescriptor InvalidShapeConstraint = new(
        "ADNPORT005",
        "Invalid model input-shape constraint",
        "'{0}' declares MinimumElementCountMember '{1}', but that field, property or method does not exist",
        "AiDotNet.TensorPorts",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private enum Domain
    {
        Unspecified = 0,
        Continuous = 1,
        IntegerIndices = 2,
        BooleanMask = 3,
    }

    private enum Direction
    {
        Input = 0,
        Output = 1,
    }

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var types = context.SyntaxProvider.CreateSyntaxProvider(
                static (node, _) => node is TypeDeclarationSyntax { AttributeLists.Count: > 0 },
                static (ctx, _) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol)
            .Where(static symbol => symbol is not null)
            .Collect();

        context.RegisterSourceOutput(types, static (spc, symbols) => EmitContracts(spc, symbols));

        var factoryFindings = context.SyntaxProvider.CreateSyntaxProvider(
                static (node, _) => node is MethodDeclarationSyntax { AttributeLists.Count: > 0 } method
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

    private static void EmitContracts(SourceProductionContext spc, ImmutableArray<INamedTypeSymbol?> symbols)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var type in symbols)
        {
            if (type is null || !seen.Add(type.ToDisplayString())) continue;

            var ports = type.GetAttributes()
                .Where(a => a.AttributeClass?.ToDisplayString() == PortAttributeName)
                .Select(ParsePort)
                .Where(p => p is not null)
                .Cast<Port>()
                .ToList();

            var rankRoute = type.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == RankRouteAttributeName);
            var shapeConstraint = type.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == ShapeConstraintAttributeName);

            if (ports.Count == 0 && rankRoute is null && shapeConstraint is null) continue;

            bool invalid = false;
            foreach (var duplicate in ports.GroupBy(p => (p.Direction, p.Name), PortKeyComparer.Instance)
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

            foreach (var port in ports)
            {
                if (port.Domain == Domain.IntegerIndices
                    && string.IsNullOrWhiteSpace(port.MaxExclusiveMember)
                    && string.IsNullOrWhiteSpace(port.MaxExclusiveResolver))
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
            }

            int minimumRank = 0;
            int minimumElements = 0;
            string? minimumElementsMember = null;
            if (shapeConstraint is not null)
            {
                minimumRank = NamedInt(shapeConstraint, "MinimumRank");
                minimumElements = NamedInt(shapeConstraint, "MinimumElementCount");
                minimumElementsMember = NamedString(shapeConstraint, "MinimumElementCountMember");
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

            if (inputs.Count > 0 && !Declares(type, "InputPorts"))
            {
                sb.AppendLine("    /// <summary>Generated from [TensorPort] declarations.</summary>");
                sb.AppendLine("    public override global::System.Collections.Generic.IReadOnlyList<global::AiDotNet.NeuralNetworks.Layers.LayerPort> InputPorts =>");
                sb.AppendLine("    [");
                foreach (var port in inputs)
                    sb.Append("        ").Append(PortExpression(type, port, isInput: true)).AppendLine(",");
                sb.AppendLine("    ];").AppendLine();
            }

            if (outputs.Count > 0 && !Declares(type, "OutputPorts"))
            {
                sb.AppendLine("    /// <summary>Generated from [TensorPort] declarations.</summary>");
                sb.AppendLine("    public override global::System.Collections.Generic.IReadOnlyList<global::AiDotNet.NeuralNetworks.Layers.LayerPort> OutputPorts =>");
                sb.AppendLine("    [");
                foreach (var port in outputs)
                    sb.Append("        ").Append(PortExpression(type, port, isInput: false)).AppendLine(",");
                sb.AppendLine("    ];").AppendLine();
            }

            if (inputs.Count > 0 && !Declares(type, "GetInputDomain") && rankRoute is null)
            {
                sb.AppendLine("    /// <summary>Generated from the primary [TensorPort] declaration.</summary>");
                sb.Append("    public override global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain GetInputDomain(int[]? inputShape) => ")
                    .Append(DomainExpression(inputs[0], "inputShape")).AppendLine(";").AppendLine();
            }

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
                    .Append(minimumRank).Append(", ").Append(elementExpression).AppendLine(");");
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
        if (method is null || !method.GetAttributes().Any(a =>
                a.AttributeClass?.ToDisplayString() == ValidateFactoryAttributeName))
            return ImmutableArray<FactoryFinding>.Empty;

        var creations = methodSyntax.DescendantNodes()
            .OfType<YieldStatementSyntax>()
            .Where(y => y.Expression is not null)
            .Select(y => (Yield: y, Type: context.SemanticModel.GetTypeInfo(y.Expression!).Type as INamedTypeSymbol))
            .Where(x => x.Type is not null)
            .OrderBy(x => x.Yield.SpanStart)
            .ToList();

        if (creations.Count < 2) return ImmutableArray<FactoryFinding>.Empty;

        var builder = ImmutableArray.CreateBuilder<FactoryFinding>();
        for (int i = 1; i < creations.Count; i++)
        {
            var producer = creations[i - 1];
            var consumer = creations[i];
            Domain produced = DeclaredDomain(producer.Type!, Direction.Output, Domain.Continuous);
            Domain required = DeclaredDomain(consumer.Type!, Direction.Input, Domain.Continuous);
            if (Accepts(required, produced)) continue;

            builder.Add(new FactoryFinding(
                consumer.Yield.GetLocation(),
                producer.Type!.Name,
                consumer.Type!.Name,
                produced,
                required));
        }

        return builder.ToImmutable();
    }

    private static bool Accepts(Domain required, Domain produced)
        => produced == Domain.Unspecified || required == Domain.Continuous || required == produced;

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
            NamedString(attr, "ShapeMember"),
            NamedBool(attr, "PropagatesInputDomain", defaultValue: false));
    }

    private static string PortExpression(INamedTypeSymbol type, Port port, bool isInput)
    {
        string shape = string.IsNullOrWhiteSpace(port.ShapeMember)
            ? (isInput ? "GetInputShape()" : "GetOutputShape()")
            : MemberExpression(type, port.ShapeMember!);
        string domain = DomainExpression(port, isInput ? "GetInputShape()" : "null");
        return "new global::AiDotNet.NeuralNetworks.Layers.LayerPort("
            + Literal(port.Name) + ", " + shape + ", "
            + (port.Required ? "true" : "false") + ", " + domain + ", "
            + "(global::AiDotNet.NeuralNetworks.Layers.TensorPortRole)" + port.Role + ")";
    }

    private static string DomainExpression(Port port, string inputShapeExpression) => port.Domain switch
    {
        Domain.Unspecified => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Unspecified",
        Domain.BooleanMask => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.BooleanMask",
        Domain.IntegerIndices when !string.IsNullOrWhiteSpace(port.MaxExclusiveResolver)
            => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Indices("
                + port.MaxExclusiveResolver + "(" + inputShapeExpression + "))",
        Domain.IntegerIndices => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Indices("
            + port.MaxExclusiveMember + ")",
        _ => "global::AiDotNet.NeuralNetworks.Layers.LayerInputDomain.Continuous",
    };

    private static string MemberExpression(INamedTypeSymbol type, string member)
    {
        var symbol = FindMember(type, member);
        return symbol is IMethodSymbol ? member + "()" : member;
    }

    private static IEnumerable<string> RequiredMembers(Port port)
    {
        if (!string.IsNullOrWhiteSpace(port.MaxExclusiveMember)) yield return port.MaxExclusiveMember!;
        if (!string.IsNullOrWhiteSpace(port.MaxExclusiveResolver)) yield return port.MaxExclusiveResolver!;
        if (!string.IsNullOrWhiteSpace(port.ShapeMember)) yield return port.ShapeMember!;
    }

    private static bool HasMember(INamedTypeSymbol type, string member) => FindMember(type, member) is not null;

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

    private static int NamedInt(AttributeData attr, string name)
        => attr.NamedArguments.FirstOrDefault(p => p.Key == name).Value.Value is int value ? value : 0;

    private static bool NamedBool(AttributeData attr, string name, bool defaultValue)
        => attr.NamedArguments.FirstOrDefault(p => p.Key == name).Value.Value is bool value ? value : defaultValue;

    private static string? NamedString(AttributeData attr, string name)
        => attr.NamedArguments.FirstOrDefault(p => p.Key == name).Value.Value as string;

    private static string Literal(string value)
        => "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

    private static string Describe(Domain domain) => domain switch
    {
        Domain.IntegerIndices => "integer indices",
        Domain.BooleanMask => "a Boolean mask",
        Domain.Unspecified => "an unspecified pass-through domain",
        _ => "continuous values",
    };

    private sealed class PortKeyComparer : IEqualityComparer<(Direction Direction, string Name)>
    {
        public static PortKeyComparer Instance { get; } = new();
        public bool Equals((Direction Direction, string Name) x, (Direction Direction, string Name) y)
            => x.Direction == y.Direction && string.Equals(x.Name, y.Name, StringComparison.Ordinal);
        public int GetHashCode((Direction Direction, string Name) obj)
            => ((int)obj.Direction * 397) ^ StringComparer.Ordinal.GetHashCode(obj.Name);
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
            string? shapeMember,
            bool propagatesInputDomain)
        {
            Name = name;
            Direction = direction;
            Domain = domain;
            Role = role;
            Required = required;
            MaxExclusiveMember = maxExclusiveMember;
            MaxExclusiveResolver = maxExclusiveResolver;
            ShapeMember = shapeMember;
            PropagatesInputDomain = propagatesInputDomain;
        }

        public string Name { get; }
        public Direction Direction { get; }
        public Domain Domain { get; }
        public int Role { get; }
        public bool Required { get; }
        public string? MaxExclusiveMember { get; }
        public string? MaxExclusiveResolver { get; }
        public string? ShapeMember { get; }
        public bool PropagatesInputDomain { get; }
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
}
