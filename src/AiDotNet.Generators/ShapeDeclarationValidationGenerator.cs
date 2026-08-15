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
    private const string LayerPropertyAttributeName = "AiDotNet.Attributes.LayerPropertyAttribute";
    private const string PreprocessesInputAttributeName = "AiDotNet.Attributes.PreprocessesInputAttribute";
    private const string StackInputLayoutAttributeName = "AiDotNet.Attributes.StackInputLayoutAttribute";

    /// <summary>Reports a concrete neural-network model that publishes no caller-facing shape manifest.</summary>
    private static readonly DiagnosticDescriptor ModelWithoutShapeContractDescriptor = new(
        id: "ADNSHAPE007",
        title: "Concrete model publishes no shape manifest",
        messageFormat: "'{0}' derives from NeuralNetworkBase but publishes neither an input nor an "
                       + "output [TensorLayout]. Every concrete model must declare the caller-facing "
                       + "ranks and semantic axes it supports, either on a measured family base or on "
                       + "the model itself.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true,
        description:
            "The generated model-family tests can only choose a correct tensor when the model names its "
            + "supported layouts. Symbolic IShapeContract relations remain valuable where output sizes "
            + "are statically expressible, but a layout manifest is mandatory even when dimensions are "
            + "data-dependent.");

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
        // THE BACKLOG CLEARED AND THIS IS AN ERROR. It entered as a Warning at 85 of ~270 layers
        // declared and reached 0; removing its <WarningsNotAsErrors> entry is what makes it fail the
        // build. A permanent warning is exactly what let 244 layers sit undeclared while the shape
        // system was assumed to cover them.
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor ModelWithoutInputLayoutDescriptor = new(
        id: "ADNSHAPE008",
        title: "Concrete model declares no caller-facing input layout",
        messageFormat: "'{0}' publishes a partial shape manifest but no effective "
                       + "[TensorLayout(Direction = Input)]. "
                       + "Declare the tensor ranks and semantic axes accepted by Predict on the model "
                       + "or on a measured family base.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor ModelWithoutOutputLayoutDescriptor = new(
        id: "ADNSHAPE009",
        title: "Concrete model declares no caller-facing output layout",
        messageFormat: "'{0}' publishes a partial shape manifest but no effective "
                       + "[TensorLayout(Direction = Output)]. "
                       + "Declare the semantic axes returned by Predict; OutputAxesFor supplies sizes, "
                       + "but callers also need the supported output layout.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor PreprocessorWithoutStackLayoutDescriptor = new(
        id: "ADNSHAPE010",
        title: "Input preprocessing omits the layer-stack entry layout",
        messageFormat: "'{0}' uses [PreprocessesInput] but declares no [StackInputLayout]. The attribute "
                       + "would otherwise suppress the model-to-first-layer check without declaring the "
                       + "transformed tensor that Layers[0] actually receives.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
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
        // ERROR: fires only where two declarations on the SAME type disagree, so there is no backlog
        // and no judgement call - one of them is simply wrong.
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);
    private const string ShapeContractName = "AiDotNet.Interfaces.IShapeContract";
    private const string ElementWiseAttributeName = "AiDotNet.Attributes.ElementWiseShapeAttribute";

    // THE TENSOR TYPE LIVES IN AiDotNet.Tensors.LinearAlgebra, NOT AiDotNet.LinearAlgebra.
    // ADNSHAPE004 previously compared against the latter, which matches nothing in this
    // repository, so every Forward(Tensor<T>) override fell through the guard and the gate
    // reported success on exactly the layers it exists to catch. Kept as a constant, and
    // matched with generics OMITTED so the open and constructed forms need only one string;
    // TrainableParameterGenerator.TensorTypeName holds the same value for the same reason.
    private const string TensorTypeName = "AiDotNet.Tensors.LinearAlgebra.Tensor";

    private static readonly SymbolDisplayFormat NamespaceQualifiedNoGenerics =
        new SymbolDisplayFormat(
            typeQualificationStyle: SymbolDisplayTypeQualificationStyle.NameAndContainingTypesAndNamespaces,
            genericsOptions: SymbolDisplayGenericsOptions.None);

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
        // A layer that bypasses ForwardTraced makes graph recovery silently incomplete. The
        // migration backlog is zero, so this is a permanent compiler gate rather than advisory
        // debt that can silently accumulate again.
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor ContractWithoutInputLayoutDescriptor = new(
        id: "ADNSHAPE003",
        title: "Type implements IShapeContract but declares no input layout",
        messageFormat: "'{0}' implements IShapeContract but declares no [TensorLayout(Direction = Input)]. "
                       + "Resolving a contract starts by NAMING the input axes, so without an input layout "
                       + "the contract can never fire and the type infers nothing.",
        category: "AiDotNet.Shapes",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) =>
                    node is Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax { AttributeLists.Count: > 0 }
                    || node is Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax { BaseList: not null },
                transform: static (ctx, _) => AnalyzeType(ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol))
            .Where(static f => f is not null)
            .Collect();

        // VALIDATE IN THE TRANSFORM, NOT AFTER Collect(). The pipeline previously carried
        // INamedTypeSymbol all the way through Collect() and into the output step. A symbol
        // roots its Compilation and has no value equality, so Roslyn could neither release the
        // previous compilation nor cache this step -- the generator re-ran in full on every
        // keystroke while pinning the old compilation in memory, which is the cost this
        // incremental API exists to avoid.
        //
        // Symbols are still touched, but only INSIDE the transform, where doing so is
        // expected. What crosses the pipeline boundary afterwards is a list of serializable
        // diagnostics: descriptor, primitives-only span, and message arguments.
        context.RegisterSourceOutput(candidates, static (spc, findings) =>
        {
            var seen = new HashSet<string>();
            foreach (var group in findings)
            {
                if (group is null) continue;
                // Partial types surface once per declaration; report each type only once.
                if (!seen.Add(group.Value.TypeKey)) continue;
                foreach (var f in group.Value.Findings) spc.ReportDiagnostic(f.ToDiagnostic());
            }
        });
    }

    /// <summary>Runs every rule against one type and returns the findings as DATA.</summary>
    /// <remarks>
    /// Called from the transform, which is where touching symbols is expected and where they
    /// do not become pipeline state. Nothing that roots a Compilation leaves this method.
    /// </remarks>
    private static TypeFindings? AnalyzeType(INamedTypeSymbol? type)
    {
        if (type is null) return null;
        var spc = new FindingCollector();
        {

            var layouts = type.GetAttributes()
                .Where(a => a.AttributeClass?.ToDisplayString() == LayoutAttributeName)
                .Select(Parse)
                .Where(l => l.Axes.Count > 0)
                .ToList();

            // The interface guard is load-bearing. IShapeContract's own opt-in interfaces
            // (IBatchAwareShapeContract and friends) satisfy AllInterfaces, and reporting them told
            // three interface DECLARATIONS to carry axis layouts, which is meaningless - an interface
            // has no shape.
            bool hasContract = type.TypeKind != TypeKind.Interface
                && type.AllInterfaces.Any(i => i.ToDisplayString() == ShapeContractName);

            // INHERITED, to match how the contract itself is inherited. AllInterfaces walks the base
            // chain, so a subclass of an IShapeContract base "has a contract"; GetAttributes does NOT,
            // so the same subclass looked as though it had declared no input layout. Every model
            // deriving from AudioNeuralNetworkBase, ForecastingModelBase or SegmentationModelBase --
            // 508 types -- was reported despite its base declaring the layout, and the only way to
            // satisfy the rule was to restate each base's contract on every subclass, where a later
            // change to the base would silently disagree with hundreds of stale copies.
            //
            // Only THIS check looks up the chain. The duplicate-axis and ambiguous-rank rules below
            // keep using `layouts`, the type's own declarations, because a base's malformed layout
            // should be reported once on the base and not again on each of its subclasses.
            // [ElementWiseShape] IS an input declaration, just a rank-agnostic one: it states that
            // whatever goes in comes out with exactly the same dimensions, at any rank.
            // ShapeContractGenerator already treats it that way and emits the contract directly from
            // it. Demanding a [TensorLayout] as well would mean writing a FIXED set of ranks onto a
            // layer documented to accept any -- narrowing a correct contract into a false one, and
            // making a rank-5 input violate a rule the layer does not actually have.
            bool declaresInputLayout = HasEffectiveLayout(type, isInput: true);
            bool declaresOutputLayout = HasEffectiveLayout(type, isInput: false);

            // Interfaces are excluded: IMultiPortShapeContract and friends REFINE IShapeContract
            // rather than implement it for a concrete tensor, so there is no layout for them to
            // declare. Reporting them asked for an input rank from a type that never has an input.
            if (hasContract && !declaresInputLayout && type.TypeKind != TypeKind.Interface)
            {
                spc.ReportDiagnostic(new ShapeFinding(
                    ContractWithoutInputLayoutDescriptor, type.Locations.FirstOrDefault(), type.Name));
            }

            // ADNSHAPE005 and ADNSHAPE006 need this on its OWN, separately from declaresInputLayout
            // above. That flag deliberately folds [ElementWiseShape] in and walks the base chain,
            // which is right for "can this type resolve an input layout" - but these two rules ask a
            // narrower question about the type itself: an element-wise layer has a contract without
            // axis names, so it must be exempt from the rank cross-check and from the
            // no-declaration report rather than merely counted as declared.
            bool isElementWise = type.GetAttributes()
                .Any(a => a.AttributeClass?.ToDisplayString() == ElementWiseAttributeName);

            // ADNSHAPE005 - cross-check the layouts against the layer's OWN [LayerProperty] shape
            // metadata. Both describe the rank a layer accepts, so they can drift; where they do, the
            // layer is exercised at a rank its contract does not cover and inference silently declines
            // on the working case.
            if (!isElementWise && layouts.Any(l => l.IsInput))
            {
                // AcceptedRanks(), not Axes.Count: a BatchOptional layout also accepts the leading axis
                // being absent, which is what TensorLayoutAttribute.AxesForRank does at runtime.
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
                        spc.ReportDiagnostic(new ShapeFinding(
                            LayerPropertyContradictsLayoutDescriptor,
                            type.Locations.FirstOrDefault(),
                            type.Name,
                            string.Join(" and ", claims),
                            string.Join(", ", declaredRanks.OrderBy(r => r))));
                    }
                }
            }

            // ADNSHAPE006 - a layer with NO shape declaration at all. Concrete layers only: an abstract
            // base legitimately leaves the contract to its subclasses.
            if (DerivesFromLayerBase(type) && !type.IsAbstract && !hasContract && !isElementWise)
            {
                spc.ReportDiagnostic(new ShapeFinding(
                    LayerWithoutShapeContractDescriptor, type.Locations.FirstOrDefault(), type.Name));
            }

            // ADNSHAPE007-009 - every concrete model must publish its caller-facing layout even when
            // no universally correct symbolic dimension relation exists. Requiring IShapeContract here
            // would force data-dependent models to lie; requiring input/output TensorLayout declarations
            // gives the test generator the ranks and semantic axes it needs without inventing dimensions.
            bool isConcreteModel = DerivesFromNeuralNetworkBase(type) && !type.IsAbstract;
            if (isConcreteModel && !declaresInputLayout && !declaresOutputLayout)
            {
                spc.ReportDiagnostic(new ShapeFinding(
                    ModelWithoutShapeContractDescriptor, type.Locations.FirstOrDefault(), type.Name));
            }
            else if (isConcreteModel)
            {
                if (!declaresInputLayout)
                {
                    spc.ReportDiagnostic(new ShapeFinding(
                        ModelWithoutInputLayoutDescriptor, type.Locations.FirstOrDefault(), type.Name));
                }

                if (!declaresOutputLayout)
                {
                    spc.ReportDiagnostic(new ShapeFinding(
                        ModelWithoutOutputLayoutDescriptor, type.Locations.FirstOrDefault(), type.Name));
                }
            }

            if (isConcreteModel
                && HasEffectiveAttribute(type, PreprocessesInputAttributeName)
                && !HasEffectiveAttribute(type, StackInputLayoutAttributeName))
            {
                spc.ReportDiagnostic(new ShapeFinding(
                    PreprocessorWithoutStackLayoutDescriptor, type.Locations.FirstOrDefault(), type.Name));
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
                // FULLY QUALIFIED. `Name: "Tensor"` accepted any type called Tensor from any
                // namespace or referenced package. ADNSHAPE004 is an Error that GATES THE
                // BUILD, so a false positive here is a hard break on correct code -- and the
                // comment above already explains that this scoping is load-bearing rather
                // than tidiness.
                if (member.Parameters[0].Type is not INamedTypeSymbol tensorParam) continue;
                if (tensorParam.ConstructedFrom.ToDisplayString(NamespaceQualifiedNoGenerics) != TensorTypeName)
                {
                    continue;
                }

                spc.ReportDiagnostic(new ShapeFinding(
                    OverridesForwardDescriptor,
                    member.Locations.FirstOrDefault() ?? type.Locations.FirstOrDefault(),
                    type.Name));
            }
            }

            if (layouts.Count == 0) return Finish(type, spc);

            foreach (var layout in layouts)
            {
                var dupe = layout.Axes
                    .GroupBy(a => a)
                    .FirstOrDefault(g => g.Count() > 1);
                if (dupe is not null)
                {
                    spc.ReportDiagnostic(new ShapeFinding(
                        DuplicateAxisDescriptor, layout.Location ?? type.Locations.FirstOrDefault(),
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
                            spc.ReportDiagnostic(new ShapeFinding(
                                AmbiguousRankDescriptor, layout.Location ?? type.Locations.FirstOrDefault(),
                                type.Name, group.Key ? "input" : "output", rank, list[0], rendered));
                            list.Add(rendered);
                        }
                    }
                }
            }
        }

        return Finish(type, spc);
    }

    /// <summary>Projects whatever was collected into the equatable result, or null when clean.</summary>
    private static TypeFindings? Finish(INamedTypeSymbol type, FindingCollector spc)
        => spc.Count == 0 ? null : new TypeFindings(type.ToDisplayString(), spc.ToImmutable());

    /// <summary>Collects findings as data during analysis.</summary>
    private sealed class FindingCollector
    {
        private readonly List<ShapeFinding> _items = new();
        public int Count => _items.Count;
        public void ReportDiagnostic(ShapeFinding f) => _items.Add(f);
        public System.Collections.Immutable.ImmutableArray<ShapeFinding> ToImmutable() => _items.ToImmutableArray();
    }

    /// <summary>One type's findings, keyed so partial declarations report once.</summary>
    private readonly struct TypeFindings : System.IEquatable<TypeFindings>
    {
        public TypeFindings(string typeKey, System.Collections.Immutable.ImmutableArray<ShapeFinding> findings)
        { TypeKey = typeKey; Findings = findings; }
        public string TypeKey { get; }
        public System.Collections.Immutable.ImmutableArray<ShapeFinding> Findings { get; }
        public bool Equals(TypeFindings other) => TypeKey == other.TypeKey && Findings.SequenceEqual(other.Findings);
        public override bool Equals(object? o) => o is TypeFindings t && Equals(t);
        public override int GetHashCode() => (TypeKey?.GetHashCode() ?? 0) * 397 ^ Findings.Length;
    }

    /// <summary>Separates diagnostic arguments inside the joined <c>Args</c> string.</summary>
    /// <remarks>
    /// A unit separator, because it cannot occur in a type or member name and so cannot split one
    /// argument into two. Joined rather than kept as an object[] because an array compares by
    /// reference, would never be equal across builds, and would defeat the pipeline caching this
    /// projection exists to restore.
    /// </remarks>
    private const char ArgSeparator = '\u001f';

    /// <summary>The same separator as text, derived so the two cannot drift apart.</summary>
    /// <remarks>
    /// ESCAPED, NOT A RAW CONTROL CHARACTER, and derived rather than repeated. A literal U+001F is
    /// valid C# but invisible in a diff and easy for an editor or a text filter to drop -- and if
    /// the text form silently became "", string.Join would concatenate every argument into one and
    /// Args.Split would then throw at report time. Neither failure would be visible in review.
    /// </remarks>
    private static readonly string ArgSeparatorText = ArgSeparator.ToString();

    /// <summary>A diagnostic reduced to primitives so it neither roots a Compilation nor breaks equality.</summary>
    private readonly struct ShapeFinding : System.IEquatable<ShapeFinding>
    {
        public ShapeFinding(DiagnosticDescriptor d, Location? loc, params object?[] args)
        {
            Descriptor = d;
            var ls = loc?.GetLineSpan();
            FilePath = ls?.Path ?? string.Empty;
            Start = loc?.SourceSpan.Start ?? 0;
            Length = loc?.SourceSpan.Length ?? 0;
            StartLine = ls?.StartLinePosition.Line ?? 0;
            StartChar = ls?.StartLinePosition.Character ?? 0;
            EndLine = ls?.EndLinePosition.Line ?? 0;
            EndChar = ls?.EndLinePosition.Character ?? 0;
            Args = string.Join(ArgSeparatorText, args.Select(a => a?.ToString() ?? string.Empty));
        }
        public DiagnosticDescriptor Descriptor { get; }
        public string FilePath { get; }
        public int Start { get; }
        public int Length { get; }
        public int StartLine { get; }
        public int StartChar { get; }
        public int EndLine { get; }
        public int EndChar { get; }
        public string Args { get; }
        public Diagnostic ToDiagnostic()
        {
            var loc = string.IsNullOrEmpty(FilePath)
                ? Location.None
                : Location.Create(FilePath,
                    new Microsoft.CodeAnalysis.Text.TextSpan(Start, Length),
                    new Microsoft.CodeAnalysis.Text.LinePositionSpan(
                        new Microsoft.CodeAnalysis.Text.LinePosition(StartLine, StartChar),
                        new Microsoft.CodeAnalysis.Text.LinePosition(EndLine, EndChar)));
            return Diagnostic.Create(Descriptor, loc, Args.Length == 0 ? new object[0] : Args.Split(ArgSeparator));
        }
        public bool Equals(ShapeFinding o) => ReferenceEquals(Descriptor, o.Descriptor) && FilePath == o.FilePath && Start == o.Start && Args == o.Args;
        public override bool Equals(object? o) => o is ShapeFinding f && Equals(f);
        public override int GetHashCode() => ((Descriptor?.Id?.GetHashCode() ?? 0) * 397 ^ Start) * 397 ^ (Args?.GetHashCode() ?? 0);
    }

    /// <summary>True when the type is a layer - the only thing graph tracing observes.</summary>
    private static bool DerivesFromLayerBase(INamedTypeSymbol type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            // Namespace-qualified for the same reason: any base type named LayerBase from
            // any assembly previously satisfied this, widening an Error diagnostic onto
            // unrelated hierarchies.
            if (b.ConstructedFrom.ToDisplayString(new SymbolDisplayFormat(
                    typeQualificationStyle: SymbolDisplayTypeQualificationStyle.NameAndContainingTypesAndNamespaces))
                == "AiDotNet.NeuralNetworks.Layers.LayerBase")
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>True for a concrete neural-network model rather than an ordinary model or layer.</summary>
    private static bool DerivesFromNeuralNetworkBase(INamedTypeSymbol type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.ConstructedFrom.ToDisplayString(NamespaceQualifiedNoGenerics)
                == "AiDotNet.NeuralNetworks.NeuralNetworkBase")
            {
                return true;
            }
        }

        return false;
    }

    private static bool HasEffectiveLayout(INamedTypeSymbol type, bool isInput)
    {
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (current.GetAttributes().Any(
                    a => a.AttributeClass?.ToDisplayString() == ElementWiseAttributeName))
            {
                return true;
            }

            if (current.GetAttributes()
                .Where(a => a.AttributeClass?.ToDisplayString() == LayoutAttributeName)
                .Select(Parse)
                .Any(layout => layout.Axes.Count > 0 && layout.IsInput == isInput))
            {
                return true;
            }
        }

        return false;
    }

    private static bool HasEffectiveAttribute(INamedTypeSymbol type, string attributeName)
    {
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (current.GetAttributes().Any(a => a.AttributeClass?.ToDisplayString() == attributeName))
                return true;
        }

        return false;
    }

    private readonly struct Layout
    {
        public Layout(List<string> axes, bool isInput, bool batchOptional, Location? location)
        {
            Axes = axes;
            IsInput = isInput;
            BatchOptional = batchOptional;
            // THE OFFENDING ATTRIBUTE, not the class name. Reporting on the type left the
            // author with a red squiggle on the class and several [TensorLayout] attributes
            // to choose between, for an Error that stops their build.
            Location = location;
        }

        public Location? Location { get; }

        public List<string> Axes { get; }
        public bool IsInput { get; }
        public bool BatchOptional { get; }
        public string DirectionName => IsInput ? "input" : "output";

        /// <summary>The ranks this layout accepts, including the unbatched form.</summary>
        /// <remarks>
        /// THE SECOND COPY OF ONE RULE. TensorLayoutAttribute.AcceptsRank is the first, and the two
        /// had already drifted APART IN OPPOSITE DIRECTIONS: this copy dropped `Axes[0] == Batch`
        /// and so reported a build ERROR for a rank the runtime accepts, while the attribute dropped
        /// `Axes.Length > 1` and so accepted rank 0. Both guards are present in both copies now.
        ///
        /// The generator cannot call the attribute: it runs inside the compiler against symbols, not
        /// loaded types, so the attribute type is never loaded. The rule is therefore restated here
        /// with its conditions spelled out, and TensorLayoutRankTests drives one shared table through
        /// this restatement and through TensorLayoutAttribute.AcceptsRank so a future divergence is a
        /// test failure rather than a silent disagreement.
        /// </remarks>
        public IEnumerable<int> AcceptedRanks()
        {
            yield return Axes.Count;

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
            // Roslyn exposes an enum constant using its declared underlying integral type. Do not
            // assume that boxed value is always Int32: analyzer test compilations and consumer-defined
            // metadata can surface another integral type even though the enum members are 0 and 1.
            if (named.Key == "Direction" && named.Value.Value is not null)
            {
                // TensorLayoutDirection.Output == 1; Input (the default) == 0.
                isInput = System.Convert.ToInt64(
                    named.Value.Value,
                    System.Globalization.CultureInfo.InvariantCulture) == 0;
            }
        }

        return new Layout(axes, isInput, batchOptional, attribute.ApplicationSyntaxReference?.GetSyntax().GetLocation());
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
