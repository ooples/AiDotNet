using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;

namespace AiDotNet.Generators;

/// <summary>
/// Emits <c>OutputAxesFor</c> into a layer's partial class from the <c>[TensorLayout]</c> attributes it
/// already declares.
/// </summary>
/// <remarks>
/// <para>
/// Annotating the layer inventory by hand meant writing the same twenty-line <c>OutputAxesFor</c> body
/// into file after file - the SSM folder alone has 35 layers whose contracts are byte-identical. That is
/// not merely tedious: a hand-copied method can DRIFT from the attributes above it, and then the
/// declaration and the contract disagree with nothing to catch it. The layouts already say what the
/// axes are; the contract should be derived from them rather than restated beside them.
/// </para>
/// <para>
/// The rule is deliberately narrow. For a declared output layout of rank R, an axis whose role also
/// appears in the input layout of the SAME rank is <c>Same(role)</c> - the layer carries it through.
/// Anything else - an axis with no input counterpart, a rank with no matching input layout - is NOT
/// derivable from the layouts alone, so that rank is skipped and the layer must write the method
/// itself. A generator that guessed at scaling or window relations would be inventing shape algebra
/// out of axis names, which is exactly the guesswork the discovery sweep exists to replace with
/// measurement.
/// </para>
/// <para>
/// Skips any type that already declares <c>OutputAxesFor</c>, so a hand-written contract always wins.
/// That is what lets a layer with a genuinely computed relation opt out without fighting the generator.
/// </para>
/// </remarks>
[Generator]
public class ShapeContractGenerator : IIncrementalGenerator
{
    private const string LayoutAttributeName = "AiDotNet.Attributes.TensorLayoutAttribute";
    private const string ElementWiseAttributeName = "AiDotNet.Attributes.ElementWiseShapeAttribute";
    private const string ShapeContractName = "AiDotNet.Interfaces.IShapeContract";

    /// <summary>
    /// Mirror of <c>AiDotNet.Enums.TensorAxis</c>, with the same underlying values.
    /// </summary>
    /// <remarks>
    /// A generator targets netstandard2.0 and runs AGAINST the analysed assembly rather than
    /// referencing it, so the real enum is not available here. Mirroring it keeps the values type-safe
    /// anyway: Roslyn hands back a boxed <c>int</c> for an enum argument, and casting that to a named
    /// enum makes an out-of-range value obvious instead of silently producing a bad identifier the way
    /// an int-to-string switch would. The values MUST stay in step with the real enum.
    /// </remarks>
    internal enum Axis
    {
        Batch = 0,
        Channels = 1,
        Height = 2,
        Width = 3,
        Depth = 4,
        Time = 5,
        Length = 6,
        Features = 7,
        Frames = 8,
        Heads = 9,
        Classes = 10,
        Other = 99,
    }

    /// <summary>Positional axes used only to express "this axis is carried through".</summary>
    /// <remarks>
    /// An element-wise layer's axes have no intrinsic meaning - that is the whole point of the
    /// attribute - so these are bookkeeping labels letting each output axis reference the input axis in
    /// the same position. They are NOT a claim that axis 2 of a dropout layer is a Height.
    /// </remarks>
    private static readonly Axis[] PositionalAxes =
        { Axis.Batch, Axis.Channels, Axis.Height, Axis.Width, Axis.Depth, Axis.Time };

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Symbols must not live in cached pipeline state: an ISymbol roots the entire Compilation,
        // so every cached entry pins a compilation in memory. That is the leak this whole series
        // targets, and this generator had it despite never touching CompilationProvider -- which is
        // exactly why scoping the original work by "uses CompilationProvider" missed it.
        //
        // The pipeline now carries metadata names and the symbol is re-resolved at the point of use.
        // Introducing CompilationProvider here costs nothing in caching terms: this generator could
        // never cache anyway, because a symbol is not value-equatable. Retention is fixed;
        // re-execution is not. Making it genuinely cacheable means lifting all of the symbol
        // reading below into the transform behind a value model, which is a larger and riskier
        // change than the retention fix warrants on its own.
        var candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) =>
                    node is Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax { AttributeLists.Count: > 0 },
                transform: static (ctx, _) =>
                    ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) is INamedTypeSymbol symbol
                        ? GeneratorHelpers.MetadataNameOf(symbol)
                        : null)
            .Where(static n => n is not null)
            .Select(static (n, _) => n ?? string.Empty)
            .Collect()
            .Combine(context.CompilationProvider);

        context.RegisterSourceOutput(candidates, static (spc, source) => Emit(spc, source.Left, source.Right));
    }

    private static void Emit(SourceProductionContext spc, ImmutableArray<string> metadataNames, Compilation compilation)
    {
        var seen = new HashSet<string>();
        var resolved = new HashSet<string>(System.StringComparer.Ordinal);

        foreach (var metadataName in metadataNames)
        {
            if (metadataName.Length == 0) continue;
            if (!resolved.Add(metadataName)) continue;

            if (GeneratorHelpers.ResolveSourceType(compilation, metadataName) is not INamedTypeSymbol type) continue;
            if (!seen.Add(type.ToDisplayString())) continue;

            var elementWise = type.GetAttributes()
                .FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == ElementWiseAttributeName);
            bool implementsShapeContract = type.AllInterfaces.Any(i => i.ToDisplayString() == ShapeContractName);
            bool canAugmentRuntimeType = type.ContainingType is null
                && type.DeclaringSyntaxReferences.Any(reference =>
                    reference.GetSyntax() is Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax declaration
                    && declaration.Modifiers.Any(modifier => modifier.Text == "partial"));
            if (!implementsShapeContract && (elementWise is null || !canAugmentRuntimeType)) continue;

            // A hand-written contract always wins - see the remarks. Only fill the gap.
            bool declaresOutputAxes = type.GetMembers("OutputAxesFor").Any();
            if (declaresOutputAxes && elementWise is null) continue;

            var arms = new List<string>();

            // [ElementWiseShape]: shape in equals shape out, at every rank. Emitted directly rather
            // than derived from layouts, because such a layer HAS no meaningful layout to declare -
            // naming its axes would invent meanings it does not have.
            if (elementWise is not null)
            {
                int maxRank = 6;
                foreach (var named in elementWise.NamedArguments)
                {
                    if (named.Key == "MaxRank" && named.Value.Value is int m && m > 0) maxRank = m;
                }
                if (maxRank > PositionalAxes.Length) maxRank = PositionalAxes.Length;

                // The layouts are emitted too, not just the contract. ShapeInference.NameAxes - the
                // production path LayerGraph.ResolveShapes goes through - names an input shape's axes
                // from [TensorLayout] and ONLY from [TensorLayout]. Without these, OutputAxesFor below
                // is unreachable: naming returns null, InferOutputShape returns null, and the layer
                // has a contract that nothing can ever resolve. The conformance sweep caught exactly
                // that, on all 13 element-wise layers at once.
                //
                // Does naming these axes "invent meanings they do not have"? The names were already
                // invented - OutputAxesFor has always used PositionalAxes. Emitting them here does not
                // add an invention, it puts the existing one where the resolver can see it. It is safe
                // precisely because every relation is Same(role): role -> role is the identity for ANY
                // consistent choice of names, so the resolved sizes do not depend on the names at all.
                // That reasoning does NOT extend to a layer that actually transforms an axis, which is
                // why this shorthand is limited to element-wise layers.
                var attributes = new List<string>();

                for (int rank = 1; rank <= maxRank; rank++)
                {
                    var entries = Enumerable.Range(0, rank).Select(i =>
                        $"                new OutputAxisContract(TensorAxis.{PositionalAxes[i]}, "
                        + $"AxisRelation.Same(TensorAxis.{PositionalAxes[i]})),");

                    arms.Add($"            {rank} => new[]\n            {{\n"
                             + string.Join("\n", entries) + "\n            },");

                    // One layout per rank and no BatchOptional, so exactly one declaration accepts each
                    // rank. NameAxes refuses ambiguity by design, and two layouts covering one rank with
                    // different axis names would make it refuse every one of these layers.
                    string axisList = string.Join(", ", Enumerable.Range(0, rank)
                        .Select(i => $"global::AiDotNet.Enums.TensorAxis.{PositionalAxes[i]}"));

                    foreach (var direction in new[] { "Input", "Output" })
                    {
                        attributes.Add(
                            $"[global::AiDotNet.Attributes.TensorLayoutAttribute({axisList}, "
                            + $"Direction = global::AiDotNet.Attributes.TensorLayoutDirection.{direction})]");
                    }
                }

                EmitPartial(
                    spc,
                    type,
                    arms,
                    attributes,
                    emitShapePreserving: canAugmentRuntimeType
                        && !type.GetMembers("IsShapePreserving").Any(),
                    emitOutputAxes: !declaresOutputAxes,
                    addShapeContractInterface: canAugmentRuntimeType && !implementsShapeContract);
                continue;
            }

            var layouts = type.GetAttributes()
                .Where(a => a.AttributeClass?.ToDisplayString() == LayoutAttributeName)
                .Select(Parse)
                .Where(l => l.Axes.Count > 0)
                .ToList();

            var inputs = layouts.Where(l => l.IsInput).ToList();
            var outputs = layouts.Where(l => !l.IsInput).ToList();
            if (inputs.Count == 0 || outputs.Count == 0) continue;

            // One arm per rank the output layout ACCEPTS, not one per layout. A BatchOptional layout
            // accepts its declared rank and that rank minus the leading axis, exactly as
            // TensorLayoutAttribute.AxesForRank does. Keying on Axes.Count alone emitted an arm only at
            // the full rank, so a layer declared [Batch?, C, H, W] answered at rank 4 and returned null
            // at rank 3 - the very rank its [LayerProperty(ExpectedInputRank = 3)] exercises. Nothing
            // caught it: ADNSHAPE005 expands BatchOptional, so the declaration looked complete while the
            // generated contract silently was not. Three separate annotation batches hit this and
            // hand-wrote all-Same contracts to work around it.
            var emitted = new HashSet<int>();

            foreach (var output in outputs)
            {
                foreach (int rank in output.AcceptedRanks())
                {
                    if (!emitted.Add(rank)) continue;

                    var outputAxes = output.AxesAtRank(rank);
                    var input = inputs.FirstOrDefault(i => i.AcceptedRanks().Contains(rank));
                    if (input is null) continue;

                    var inputAxes = input.AxesAtRank(rank);

                    // Every output axis must have an input counterpart, or the contract is not derivable.
                    if (!outputAxes.All(a => inputAxes.Contains(a))) continue;

                    var entries = outputAxes.Select(a =>
                        $"                new OutputAxisContract(TensorAxis.{a}, AxisRelation.Same(TensorAxis.{a})),");

                    arms.Add($"            {rank} => new[]\n            {{\n"
                             + string.Join("\n", entries) + "\n            },");
                }
            }

            if (arms.Count == 0) continue;

            EmitPartial(spc, type, arms);
        }
    }

    private static void EmitPartial(
        SourceProductionContext spc,
        INamedTypeSymbol type,
        List<string> arms,
        List<string>? attributes = null,
        bool emitShapePreserving = false,
        bool emitOutputAxes = true,
        bool addShapeContractInterface = false)
    {
        {
            string ns = type.ContainingNamespace.IsGlobalNamespace
                ? string.Empty
                : type.ContainingNamespace.ToDisplayString();

            string typeParams = type.TypeParameters.Length > 0
                ? "<" + string.Join(", ", type.TypeParameters.Select(p => p.Name)) + ">"
                : string.Empty;

            var sb = new StringBuilder();
            sb.AppendLine("// <auto-generated/>");
            sb.AppendLine("#nullable enable");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine("using AiDotNet.Enums;");
            sb.AppendLine("using AiDotNet.Interfaces;");
            sb.AppendLine();
            if (ns.Length > 0) sb.AppendLine($"namespace {ns};").AppendLine();

            // Attributes are merged across a type's partial declarations, so these apply to the layer
            // as if written on its own file. Fully qualified because two Tensors namespaces in the
            // project's global usings also define a TensorLayout - the short name does not bind here.
            if (attributes is not null)
            {
                foreach (var attribute in attributes) sb.AppendLine(attribute);
            }

            string interfaceClause = addShapeContractInterface ? " : IShapeContract" : string.Empty;
            sb.AppendLine($"partial class {type.Name}{typeParams}{interfaceClause}");
            sb.AppendLine("{");
            if (emitOutputAxes)
            {
                sb.AppendLine("    /// <summary>Derived from this type's [TensorLayout] declarations.</summary>");
                sb.AppendLine("    /// <remarks>");
                sb.AppendLine("    /// Generated so the contract cannot drift from the layouts it restates. Ranks whose");
                sb.AppendLine("    /// axes are not all carried through are omitted rather than guessed - declare");
                sb.AppendLine("    /// OutputAxesFor by hand on this type to override the whole method.");
                sb.AppendLine("    /// </remarks>");
                sb.AppendLine("    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank switch");
                sb.AppendLine("        {");
                foreach (var arm in arms) sb.AppendLine(arm);
                sb.AppendLine("            _ => null,");
                sb.AppendLine("        };");
            }
            if (emitShapePreserving)
            {
                sb.AppendLine();
                sb.AppendLine("    /// <summary>Generated from [ElementWiseShape]: the output shape is exactly the concrete input shape.</summary>");
                sb.AppendLine("    protected override bool IsShapePreserving => true;");
            }
            sb.AppendLine("}");

            string hint = (ns.Length > 0 ? ns.Replace('.', '_') + "_" : string.Empty)
                          + type.Name + ".ShapeContract.g.cs";

            spc.AddSource(hint, sb.ToString());
        }
    }

    private sealed class Layout
    {
        public List<string> Axes = new();
        public bool IsInput;
        public bool BatchOptional;

        /// <summary>Every rank this layout accepts, matching TensorLayoutAttribute.AxesForRank.</summary>
        public IEnumerable<int> AcceptedRanks()
        {
            yield return Axes.Count;
            if (BatchOptional && Axes.Count > 1) yield return Axes.Count - 1;
        }

        /// <summary>The axis names at a given accepted rank; the leading axis is dropped when elided.</summary>
        public List<string> AxesAtRank(int rank)
            => rank == Axes.Count ? Axes : Axes.GetRange(Axes.Count - rank, rank);
    }

    private static Layout Parse(AttributeData attr)
    {
        var layout = new Layout();

        foreach (var arg in attr.ConstructorArguments)
        {
            if (arg.Kind == TypedConstantKind.Array)
            {
                foreach (var element in arg.Values)
                {
                    if (element.Value is int v) layout.Axes.Add(AxisName(v));
                }
            }
            else if (arg.Value is int v2)
            {
                layout.Axes.Add(AxisName(v2));
            }
        }

        // Direction defaults to Input when unspecified, matching TensorLayoutAttribute.
        layout.IsInput = true;
        foreach (var named in attr.NamedArguments)
        {
            if (named.Key == "Direction" && named.Value.Value is int d) layout.IsInput = d == 0;
            if (named.Key == "BatchOptional" && named.Value.Value is bool b) layout.BatchOptional = b;
        }

        return layout;
    }

    /// <summary>
    /// Turns the boxed <c>int</c> Roslyn hands back for an enum argument into a named axis.
    /// </summary>
    /// <remarks>
    /// Goes through <see cref="Axis"/> rather than a bare int-to-string switch so an unrecognised value
    /// cannot quietly become a plausible-looking identifier. An undefined value maps to
    /// <see cref="Axis.Other"/>, which the layout rules already treat as unnamed.
    /// </remarks>
    private static string AxisName(int value) =>
        (Enum.IsDefined(typeof(Axis), value) ? (Axis)value : Axis.Other).ToString();

}
