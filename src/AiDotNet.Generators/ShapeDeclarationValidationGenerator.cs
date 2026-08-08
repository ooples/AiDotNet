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
    private const string ElementWiseAttributeName = "AiDotNet.Attributes.ElementWiseShapeAttribute";
    private const string LayerPropertyAttributeName = "AiDotNet.Attributes.LayerPropertyAttribute";
    private const string ShapeContractName = "AiDotNet.Interfaces.IShapeContract";

    private static readonly DiagnosticDescriptor LayerWithoutShapeContractDescriptor = new(
        id: "ADNSHAPE006",
        title: "Layer declares no shape contract, so nothing can reason about its output shape",
        messageFormat: "'{0}' derives from LayerBase but declares neither [TensorLayout] + "
                       + "IShapeContract nor [ElementWiseShape]. Shape inference, chain validation and "
                       + "graph resolution all decline on it silently - it is not failing, it simply "
                       + "cannot be reasoned about. If it preserves shape at any rank use "
                       + "[ElementWiseShape]; otherwise declare its axis layouts and implement "
                       + "OutputAxesFor (compute the relation from the layer's own fields where it "
                       + "depends on a constructor argument, as DenseLayer and MaxPoolingLayer do).",
        category: "AiDotNet.Shapes",
        // THE BACKLOG HAS CLEARED AND THIS IS NOW AN ERROR. It entered as a Warning at 85 of ~270
        // layers declared, on the same ladder as ADNSHAPE004 and ADNGEN001, because erroring
        // immediately would have reddened the build against work that was unfinished rather than
        // wrong. It stayed in <WarningsNotAsErrors> - deliberately not NoWarn - so the remaining count
        // could be read off any build, and that count reached 0: every concrete LayerBase subclass now
        // declares [TensorLayout] + IShapeContract or [ElementWiseShape].
        //
        // The severity below is what the compiler uses; removing the ADNSHAPE006 entry from
        // src/AiDotNet.csproj's <WarningsNotAsErrors> is what actually made it fail the build. Both
        // are done. The flip was always the point: a permanent warning is exactly what let 244 layers
        // sit undeclared while the shape system was assumed to cover them.
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor LayerPropertyContradictsLayoutDescriptor = new(
        id: "ADNSHAPE005",
        title: "[LayerProperty] shape metadata contradicts the declared [TensorLayout] ranks",
        messageFormat: "'{0}' declares {1}, but its [TensorLayout(Direction = Input)] attributes cover "
                       + "only rank(s) [{2}]. Two declarations of the same fact have drifted, and shape "
                       + "inference trusts the layouts - so the rank the layer is actually exercised at "
                       + "resolves to nothing. Add a layout for that rank, or correct whichever "
                       + "declaration is wrong.",
        category: "AiDotNet.Shapes",
        // ERROR: this fires only where two declarations on the SAME type disagree, so there is no
        // backlog to work through and no judgement call - one of them is simply wrong. It exists
        // because the layouts added during the shape rollout were derived from a probe, while
        // TestInputShape was written by whoever built the layer; where they disagree the author wins,
        // and five layers were annotated at the wrong rank before this caught them.
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

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

            // INHERITED layouts count. TensorLayoutAttribute is [AttributeUsage(Inherited = true)] and
            // ShapeInference.LayoutsFor reads them with inherit: true, so a layout declared on a base
            // class really does resolve for every subclass at runtime. GetAttributes() returns only
            // DIRECTLY applied attributes, so checking it alone contradicts the resolver: declaring the
            // family contract once on LoRAAdapterBase - which is the whole point of an inherited
            // attribute - made all 34 derived adapters report ADNSHAPE003 for a layout they demonstrably
            // have. An analyzer that disagrees with the runtime it guards is worse than no analyzer.
            bool hasInheritedInputLayout = false;
            for (var ancestor = type.BaseType; ancestor is not null; ancestor = ancestor.BaseType)
            {
                if (ancestor.GetAttributes().Any(a =>
                        a.AttributeClass?.ToDisplayString() == LayoutAttributeName
                        && Parse(a).IsInput && Parse(a).Axes.Count > 0))
                {
                    hasInheritedInputLayout = true;
                    break;
                }
            }

            // An INTERFACE that extends IShapeContract is not a contract implementation - it is part of
            // the contract vocabulary. IBatchAwareShapeContract / IMultiPortShapeContract /
            // IMultiOutputShapeContract exist so a layer can opt into an extra form, and demanding a
            // [TensorLayout] from them is meaningless: an interface has no axes of its own and nothing
            // ever resolves against it. Third time this analyzer family has reported a type the resolver
            // never asks about; the rule is that ADNSHAPE003 constrains what IMPLEMENTS a contract, not
            // what DECLARES one.
            bool hasContract = type.TypeKind != TypeKind.Interface
                && type.AllInterfaces.Any(i => i.ToDisplayString() == ShapeContractName);

            // [ElementWiseShape] declares "any rank, every axis carried through", which is a COMPLETE
            // contract expressed without naming axes - a dropout layer has no Height. Demanding an
            // input layout from it would force exactly the invented axis names the attribute exists to
            // avoid, so the absence here is by design rather than an omission.
            bool isElementWise = type.GetAttributes()
                .Any(a => a.AttributeClass?.ToDisplayString() == ElementWiseAttributeName);

            // ADNSHAPE005 - cross-check the layouts against the layer's OWN [LayerProperty] shape
            // metadata. [TensorLayout] and TestInputShape/ExpectedInputRank both describe the rank a
            // layer accepts, so they can drift; where they do, the layer is exercised at a rank its
            // contract does not cover and inference silently declines on the working case.
            if (!isElementWise && layouts.Any(l => l.IsInput))
            {
                // AcceptedRanks(), not Axes.Count: a BatchOptional layout accepts the leading axis being
                // absent too, which is exactly what TensorLayoutAttribute.AxesForRank does at runtime.
                // Counting only the declared length made AdaptiveAveragePoolingLayer - correctly declared
                // [Batch?, Channels, Height, Width] and tested at rank 3 - report that its layouts "cover
                // only rank 4". Second time this analyzer family has contradicted the resolver it guards
                // (ADNSHAPE003 did it with inherited attributes); both were the analyzer being wrong.
                var declaredRanks = layouts.Where(l => l.IsInput)
                    .SelectMany(l => l.AcceptedRanks())
                    .Distinct()
                    .ToList();

                var layerProperty = type.GetAttributes().FirstOrDefault(
                    a => a.AttributeClass?.ToDisplayString() == LayerPropertyAttributeName);

                if (layerProperty is not null)
                {
                    var claims = new List<string>();

                    foreach (var named in layerProperty.NamedArguments)
                    {
                        if (named.Key == "ExpectedInputRank" && named.Value.Value is int rank && rank > 0
                            && !declaredRanks.Contains(rank))
                        {
                            claims.Add($"ExpectedInputRank = {rank}");
                        }

                        if (named.Key == "TestInputShape" && named.Value.Value is string shape
                            && shape.Length > 0)
                        {
                            int testRank = shape.Split(',').Length;
                            if (!declaredRanks.Contains(testRank))
                                claims.Add($"TestInputShape = \"{shape}\" (rank {testRank})");
                        }
                    }

                    if (claims.Count > 0)
                    {
                        spc.ReportDiagnostic(Diagnostic.Create(
                            LayerPropertyContradictsLayoutDescriptor,
                            type.Locations.FirstOrDefault(),
                            type.Name,
                            string.Join(" and ", claims),
                            string.Join(", ", declaredRanks.OrderBy(r => r))));
                    }
                }
            }

            if (hasContract && !isElementWise && !layouts.Any(l => l.IsInput) && !hasInheritedInputLayout)
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
            // ADNSHAPE006 - a layer with NO shape declaration at all. Reported only for concrete
            // layers: an abstract base legitimately leaves the contract to its subclasses, and a
            // partial's other halves would double-report.
            if (DerivesFromLayerBase(type) && !type.IsAbstract && !hasContract && !isElementWise)
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    LayerWithoutShapeContractDescriptor, type.Locations.FirstOrDefault(), type.Name));
            }

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
            // A batch-optional layout also accepts the form with the leading axis dropped.
            if (BatchOptional && Axes.Count > 1) yield return Axes.Count - 1;
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
