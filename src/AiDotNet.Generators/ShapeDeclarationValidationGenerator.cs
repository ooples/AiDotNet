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
        // WARNING UNTIL THE CONVERSION LANDS, THEN ERROR.
        //
        // This rule fires on 873 layers, and the commits that rename those overrides to
        // ForwardTraced are spread across the later slices of the #1789 split. At Error severity
        // it GATES THE BUILD, so slice 01 -- which introduces the rule -- makes every intermediate
        // merge state of the split red by construction: measured on integration/1789, merging
        // slice 01 alone produced 873 ADNSHAPE004 errors, and merging 02 and 03 on top did not
        // clear them because their conversions are not the ones this rule is waiting for.
        //
        // The consequence is not cosmetic. No slice can be built, so no slice can be
        // build-verified before merge, per-slice CI reports a failure that says nothing about the
        // slice, and any tooling that checks a branch compiles is useless across the whole split.
        //
        // Warning keeps every diagnostic visible and every intermediate state buildable. The final
        // slice of the split flips this back to Error, at the point where the violation count is
        // zero and the gate can hold. See the sibling entry in AnalyzerReleases.Unshipped.md.
        //
        // Note the asymmetry with ADNSHAPE001/002, which stay Error: those describe a contract
        // that is self-inconsistent right now, not one the split is mid-way through satisfying.
        defaultSeverity: DiagnosticSeverity.Warning,
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

            bool hasContract = type.AllInterfaces.Any(i => i.ToDisplayString() == ShapeContractName);

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
            bool declaresInputLayout = false;
            for (var t = type; t is not null && !declaresInputLayout; t = t.BaseType)
            {
                declaresInputLayout = t.GetAttributes()
                    .Where(a => a.AttributeClass?.ToDisplayString() == LayoutAttributeName)
                    .Select(Parse)
                    .Any(l => l.Axes.Count > 0 && l.IsInput);
            }

            // Interfaces are excluded: IMultiPortShapeContract and friends REFINE IShapeContract
            // rather than implement it for a concrete tensor, so there is no layout for them to
            // declare. Reporting them asked for an input rank from a type that never has an input.
            if (hasContract && !declaresInputLayout && type.TypeKind != TypeKind.Interface)
            {
                spc.ReportDiagnostic(new ShapeFinding(
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
            // TensorLayoutDirection.Output == 1; Input (the default) == 0.
            if (named.Key == "Direction" && named.Value.Value is int d) isInput = d == 0;
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
