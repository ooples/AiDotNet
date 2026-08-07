using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;

namespace AiDotNet.Generators;

/// <summary>
/// Build-time validation of <c>[TensorLayout]</c> shape declarations.
/// </summary>
/// <remarks>
/// <para>
/// The shape system has two halves that only pay off if the declarations are internally coherent:
/// <c>[TensorLayout]</c> names a tensor's axes, and <c>IShapeContract</c> says how big each output axis
/// is. Both are consumed by <c>ShapeInference.NameAxes</c>, which REFUSES ambiguous or unusable
/// declarations rather than guessing — correct behaviour, but it means a bad declaration does not fail
/// loudly, it silently makes inference decline. That is the worst of both worlds: the annotation looks
/// present and does nothing.
/// </para>
/// <para>
/// Every rule here is a defect that was actually hit while building the system, not a hypothetical:
/// </para>
/// <list type="bullet">
/// <item>
/// ADNSHAPE001 — DenseLayer declared <c>[Batch, Features]</c> and <c>[Batch, Time, Features]</c> BOTH
/// batch-optional, so both accepted rank 2, with different axis names. Inference could not tell whether
/// a rank-2 input's leading axis was Batch or Time, so it declined on a completely ordinary case. Cost
/// a debugging session to find by hand.
/// </item>
/// <item>
/// ADNSHAPE002 — a repeated role in one layout cannot be addressed by name, so the entire naming is
/// refused. Hit while writing DenseLayer's rank-4 contract with anonymous placeholders.
/// </item>
/// <item>
/// ADNSHAPE003 — a contract with no input layout can never resolve, because resolving starts by naming
/// the INPUT axes. A declaration that cannot fire is worse than none.
/// </item>
/// </list>
/// <para>
/// WHAT THIS DELIBERATELY DOES NOT ATTEMPT: shape algebra through a composed chain. A model's layers are
/// built at runtime by <c>LayerHelper</c> factories that loop over constructor options, so the sequence
/// does not exist at compile time and an analyzer claiming to verify it would be guessing. That check
/// lives in <c>NeuralNetworkBase.ReportLayerContractMismatches</c>, which runs where the chain is real.
/// </para>
/// </remarks>
[Generator]
public class ShapeDeclarationValidationGenerator : IIncrementalGenerator
{
    private const string LayoutAttributeName = "AiDotNet.Attributes.TensorLayoutAttribute";
    private const string ShapeContractName = "AiDotNet.Interfaces.IShapeContract";

    private static readonly DiagnosticDescriptor AmbiguousRankDescriptor = new(
        id: "ADNSHAPE001",
        title: "Two tensor layouts accept the same rank with different axis names",
        messageFormat: "'{0}' declares two {1} layouts that both accept rank {2} but name its axes "
                       + "differently ([{3}] and [{4}]). Shape inference cannot tell which reading applies, "
                       + "so it refuses to name the axes at all and every shape query on this type silently "
                       + "declines. Drop BatchOptional from the longer layout, or give the axes distinct roles.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DuplicateAxisDescriptor = new(
        id: "ADNSHAPE002",
        title: "A tensor layout repeats an axis role",
        messageFormat: "'{0}' declares a {1} layout [{2}] that uses the role '{3}' more than once. Roles "
                       + "are how a relation refers to its input, so a repeated role cannot be addressed and "
                       + "the whole naming is refused. Give each axis a distinct role.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor OverridesForwardDescriptor = new(
        id: "ADNSHAPE004",
        title: "Layer overrides Forward instead of ForwardTraced and is invisible to graph tracing",
        messageFormat: "'{0}' overrides Forward(Tensor<T>) instead of ForwardTraced. Forward is the "
                       + "single point that records which tensor a layer consumed and produced, which is "
                       + "how a model's dataflow is recovered without the model declaring it. Overriding "
                       + "it bypasses that recording, so this layer becomes a HOLE in every traced graph "
                       + "- silently, because the trace still succeeds and simply omits it. Rename the "
                       + "override to 'protected override Tensor<T> ForwardTraced'.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor ContractWithoutInputLayoutDescriptor = new(
        id: "ADNSHAPE003",
        title: "Type implements IShapeContract but declares no input layout",
        messageFormat: "'{0}' implements IShapeContract but declares no [TensorLayout(Direction = Input)]. "
                       + "Resolving a contract starts by NAMING the input axes, so without an input layout "
                       + "the contract can never fire and the type infers nothing.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) =>
                    node is Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax { AttributeLists.Count: > 0 }
                    || node is Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax { BaseList: not null },
                transform: static (ctx, _) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol)
            .Where(static s => s is not null)
            .Collect();

        context.RegisterSourceOutput(candidates, static (spc, symbols) => Validate(spc, symbols));
    }

    private static void Validate(SourceProductionContext spc, ImmutableArray<INamedTypeSymbol?> symbols)
    {
        // Partial types surface once per declaration; report each type only once.
        var seen = new HashSet<string>();

        foreach (var type in symbols)
        {
            if (type is null) continue;
            if (!seen.Add(type.ToDisplayString())) continue;

            var layouts = type.GetAttributes()
                .Where(a => a.AttributeClass?.ToDisplayString() == LayoutAttributeName)
                .Select(Parse)
                .Where(l => l.Axes.Count > 0)
                .ToList();

            bool hasContract = type.AllInterfaces.Any(i => i.ToDisplayString() == ShapeContractName);

            if (hasContract && !layouts.Any(l => l.IsInput))
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    ContractWithoutInputLayoutDescriptor, type.Locations.FirstOrDefault(), type.Name));
            }

            // A layer that overrides Forward is invisible to tracing. Caught at BUILD time because the
            // failure is otherwise silent: the trace succeeds and simply omits the layer, so the
            // recovered graph is wrong in a way nothing downstream can detect.
            //
            // Scoped to LayerBase descendants, and that scoping is load-bearing rather than tidiness.
            // Plenty of types declare a Forward(Tensor<T>) that overrides some OTHER base - TabNetClassifier
            // and ~30 model classes among them - and they are not layers, are not traced, and reporting
            // them would be 60 false errors telling people to rename methods that are already correct.
            if (DerivesFromLayerBase(type))
            {
            foreach (var member in type.GetMembers("Forward").OfType<IMethodSymbol>())
            {
                if (!member.IsOverride) continue;
                // The dictionary and params overloads are separate methods; only the single-tensor
                // one is the traced entry point.
                if (member.Parameters.Length != 1) continue;
                if (member.Parameters[0].Type is not INamedTypeSymbol { Name: "Tensor" }) continue;

                spc.ReportDiagnostic(Diagnostic.Create(
                    OverridesForwardDescriptor,
                    member.Locations.FirstOrDefault() ?? type.Locations.FirstOrDefault(),
                    type.Name));
            }
            }

            if (layouts.Count == 0) continue;

            foreach (var layout in layouts)
            {
                var dupe = layout.Axes
                    .GroupBy(a => a)
                    .FirstOrDefault(g => g.Count() > 1);
                if (dupe is not null)
                {
                    spc.ReportDiagnostic(Diagnostic.Create(
                        DuplicateAxisDescriptor, type.Locations.FirstOrDefault(),
                        type.Name, layout.DirectionName, string.Join(", ", layout.Axes), dupe.Key));
                }
            }

            // The ambiguity that actually bit: two layouts in the same direction accepting one rank with
            // different names. BatchOptional means a layout also accepts rank-1-less, which is precisely
            // how a 3-axis declaration collides with a 2-axis one.
            foreach (var group in layouts.GroupBy(l => l.IsInput))
            {
                var byRank = new Dictionary<int, List<string>>();
                foreach (var layout in group)
                {
                    foreach (int rank in layout.AcceptedRanks())
                    {
                        string rendered = string.Join(", ", layout.AxesForRank(rank));
                        if (!byRank.TryGetValue(rank, out var list))
                        {
                            byRank[rank] = new List<string> { rendered };
                        }
                        else if (!list.Contains(rendered))
                        {
                            spc.ReportDiagnostic(Diagnostic.Create(
                                AmbiguousRankDescriptor, type.Locations.FirstOrDefault(),
                                type.Name, group.Key ? "input" : "output", rank, list[0], rendered));
                            list.Add(rendered);
                        }
                    }
                }
            }
        }
    }

    /// <summary>True when the type is a layer - the only thing graph tracing observes.</summary>
    private static bool DerivesFromLayerBase(INamedTypeSymbol type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.Name == "LayerBase") return true;
        }

        return false;
    }

    private readonly struct Layout
    {
        public Layout(List<string> axes, bool isInput, bool batchOptional)
        {
            Axes = axes;
            IsInput = isInput;
            BatchOptional = batchOptional;
        }

        public List<string> Axes { get; }
        public bool IsInput { get; }
        public bool BatchOptional { get; }
        public string DirectionName => IsInput ? "input" : "output";

        public IEnumerable<int> AcceptedRanks()
        {
            yield return Axes.Count;

            // THE SAME THREE CONDITIONS AS TensorLayoutAttribute.AcceptsRank, and
            // the leading-axis test is the one that was missing here. This copy
            // dropped `Axes[0] == Batch`, so a batch-optional layout whose first
            // axis is NOT Batch was reported as accepting rank-1-less at build
            // time -- a build error for a shape the runtime accepts, and the exact
            // mirror of the attribute's own omission (it lacked `Axes.Length > 1`
            // and so accepted rank 0). Two copies of one rule drifted apart in
            // opposite directions, each wrong in the direction the other was not.
            //
            // The generator cannot call the attribute: it runs inside the compiler
            // against symbols, not loaded types. So the rule is restated with the
            // conditions spelled out and cross-referenced, and the two are kept in
            // step by the shared-rank test in the attribute's own suite.
            if (BatchOptional
                && Axes.Count > 1
                && string.Equals(Axes[0], "Batch", System.StringComparison.Ordinal))
            {
                yield return Axes.Count - 1;
            }
        }

        public IEnumerable<string> AxesForRank(int rank)
            => rank == Axes.Count ? Axes : Axes.Skip(Axes.Count - rank);
    }

    private static Layout Parse(AttributeData attribute)
    {
        var axes = new List<string>();
        foreach (var arg in attribute.ConstructorArguments)
        {
            if (arg.Kind == TypedConstantKind.Array)
            {
                foreach (var element in arg.Values) axes.Add(RenderAxis(element));
            }
            else
            {
                axes.Add(RenderAxis(arg));
            }
        }

        bool isInput = true;
        bool batchOptional = false;
        foreach (var named in attribute.NamedArguments)
        {
            if (named.Key == "BatchOptional" && named.Value.Value is bool b) batchOptional = b;
            // TensorLayoutDirection.Output == 1; Input (the default) == 0.
            if (named.Key == "Direction" && named.Value.Value is int d) isInput = d == 0;
        }

        return new Layout(axes, isInput, batchOptional);
    }

    private static string RenderAxis(TypedConstant constant)
    {
        if (constant.Type is INamedTypeSymbol { TypeKind: TypeKind.Enum } enumType)
        {
            foreach (var member in enumType.GetMembers().OfType<IFieldSymbol>())
            {
                if (member.HasConstantValue && Equals(member.ConstantValue, constant.Value)) return member.Name;
            }
        }

        return constant.Value?.ToString() ?? "?";
    }
}
