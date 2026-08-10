using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Registers a model's weight-bearing FIELDS with its parameter component registry, so a model
/// author writes no parameter plumbing at all -- the same automation layers already get from
/// <see cref="TrainableParameterGenerator"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this is separate from the layer generator.</b> That generator emits
/// <c>GetTrainableParameters</c>/<c>SetTrainableParameters</c> and calls <c>RegisterBuffer</c> --
/// all <c>LayerBase</c> members. A model has none of them; it has
/// <c>RegisterParameterComponent</c> and a <c>RegisterComponents</c> hook. The two paths share a
/// discovery IDEA and no code, and the layer path is green across five build configurations, so it
/// is left untouched rather than parameterised.
/// </para>
/// <para>
/// <b>What it fixes.</b> A model that holds weights in plain fields rather than layers had no way
/// to be counted. Its <c>ParameterCount</c> and <c>GetParameters</c> both omitted them, so the two
/// AGREED -- and the count-vs-vector contract test passed while the weights were never trained,
/// never saved, and never restored. AnomalyDetectorBase reported <c>1</c> (a threshold scalar) for
/// detectors holding twelve gradient-trained weight matrices; MetaLearnerBase discarded every
/// algorithm-level weight on save. This closes that by construction.
/// </para>
/// <para>
/// <b>Inverted default, as with layers.</b> An unmarked weight-capable field IS a parameter.
/// <c>[Scratch]</c> (rebuilt every forward) and <c>[Buffer]</c> (persistent, never optimized) are
/// the opt-outs. Getting the default the other way round is what allowed a weight to go missing
/// silently; getting an opt-out wrong is at least visible, and <c>ScratchFieldsAreTransientTests</c>
/// checks the <c>[Scratch]</c> claim mechanically.
/// </para>
/// <para>
/// <b>Gated on the hook, not on a base-class name.</b> The type must actually inherit both
/// <c>RegisterParameterComponent</c> and an overridable <c>RegisterComponents</c>. That covers the
/// ModelBase trunk -- MetaLearnerBase, AnomalyDetectorBase, GaussianProcessBase, ClassifierBase,
/// RegressionBase, ClusteringBase -- without naming any of them, and self-limits: a root that has
/// not yet grown a registry is skipped and keeps reporting AIDN084, which is honest rather than
/// silently half-automated.
/// </para>
/// </remarks>
[Generator]
public class ModelParameterGenerator : IIncrementalGenerator
{
    private const string ScratchAttributeName = "AiDotNet.Attributes.ScratchAttribute";
    private const string BufferAttributeName = "AiDotNet.Attributes.BufferAttribute";
    private const string TensorTypeName = "AiDotNet.Tensors.LinearAlgebra.Tensor";
    private const string MatrixTypeName = "AiDotNet.Tensors.LinearAlgebra.Matrix";
    private const string VectorTypeName = "AiDotNet.Tensors.LinearAlgebra.Vector";

    private const string RegisterHook = "RegisterComponents";
    private const string RegisterCall = "RegisterParameterComponent";
    private const string ExtraTensorsHook = "GetExtraTrainableTensors";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var classDeclarations = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax cds &&
                    cds.Modifiers.Any(m => m.Text == "partial"),
                transform: static (ctx, _) => (ClassDeclarationSyntax)ctx.Node)
            .Where(static c => c is not null);

        var compilationAndClasses = context.CompilationProvider.Combine(classDeclarations.Collect());
        context.RegisterSourceOutput(compilationAndClasses,
            static (spc, source) => Execute(source.Left, source.Right, spc));
    }

    private static void Execute(Compilation compilation,
                                ImmutableArray<ClassDeclarationSyntax> classes,
                                SourceProductionContext context)
    {
        if (classes.IsDefaultOrEmpty) return;

        var scratchSymbol = compilation.GetTypeByMetadataName(ScratchAttributeName);
        var bufferSymbol = compilation.GetTypeByMetadataName(BufferAttributeName);

        var processed = new HashSet<string>();

        foreach (var classDecl in classes)
        {
            var model = compilation.GetSemanticModel(classDecl.SyntaxTree);
            if (model.GetDeclaredSymbol(classDecl) is not INamedTypeSymbol classSymbol) continue;
            if (classSymbol.IsAbstract) continue;

            var elem = ElementTypeParam(classSymbol);
            if (elem is null) continue;

            // Two trunks, two hooks.
            //
            // ModelBase and its descendants have the component registry, which takes a source per
            // field and so can carry tensors, matrices and vectors alike.
            //
            // NeuralNetworkBase has no registry; it has GetExtraTrainableTensors(), and that hook is
            // ALREADY consumed in fourteen places -- ParameterCount, GetParameters, SetParameters,
            // serialization, cloning, gradient collection, GPU mirroring. Bolting a second registry
            // onto that base would mean threading it through every one of them, and missing one is
            // precisely how a weight goes quiet. Emitting into the hook that is already wired costs
            // nothing and cannot miss a site. Its element type is Tensor<T>, so only tensor fields
            // are automated there; matrices and vectors on that trunk stay reported.
            bool hasRegistry = InheritsRegistry(classSymbol) && !DeclaresOwn(classSymbol, RegisterHook);
            bool hasExtraTensors = !hasRegistry
                                   && InheritsExtraTensorsHook(classSymbol)
                                   && !DeclaresOwn(classSymbol, ExtraTensorsHook);
            if (!hasRegistry && !hasExtraTensors) continue;

            if (!processed.Add(classSymbol.ToDisplayString())) continue;

            if (hasExtraTensors)
            {
                var tensors = new List<string>();
                foreach (var member in classSymbol.GetMembers())
                {
                    if (member is not IFieldSymbol tf) continue;
                    if (!IsRegisterableField(tf, scratchSymbol, bufferSymbol)) continue;
                    if (SourceFor(tf.Type, elem) != "TensorFieldParameterSource") continue;
                    tensors.Add(tf.Name);
                }

                if (tensors.Count == 0) continue;
                context.AddSource(
                    HintName(classSymbol) + ".ModelExtraTensors.g.cs",
                    GenerateExtraTensorsSource(classSymbol, elem, tensors));
                continue;
            }

            var fields = new List<(string Name, string SourceType)>();
            foreach (var member in classSymbol.GetMembers())
            {
                if (member is not IFieldSymbol f) continue;
                if (!IsRegisterableField(f, scratchSymbol, bufferSymbol)) continue;

                var src = SourceFor(f.Type, elem);
                if (src is null) continue;
                fields.Add((f.Name, src));
            }

            if (fields.Count == 0) continue;

            context.AddSource(HintName(classSymbol) + ".ModelParameters.g.cs",
                              GenerateSource(classSymbol, elem, fields));
        }
    }

    private static string HintName(INamedTypeSymbol t) =>
        t.ToDisplayString().Replace('.', '_').Replace('<', '_').Replace('>', '_');

    /// <summary>Field-level gates shared by both emission paths.</summary>
    private static bool IsRegisterableField(IFieldSymbol f,
                                            INamedTypeSymbol? scratchSymbol,
                                            INamedTypeSymbol? bufferSymbol)
    {
        if (f.IsStatic || f.IsConst) return false;
        // Auto-property backing fields have names that are not valid C# to emit.
        if (f.IsImplicitlyDeclared || f.AssociatedSymbol is not null) return false;
        if (HasAttr(f, scratchSymbol) || HasAttr(f, bufferSymbol)) return false;
        // A gradient accumulator is sized like a weight and is not one. The layer path uses the
        // same suffix convention.
        if (f.Name.EndsWith("Gradient", System.StringComparison.Ordinal) ||
            f.Name.EndsWith("Gradients", System.StringComparison.Ordinal)) return false;
        return true;
    }

    /// <summary>An overridable <c>GetExtraTrainableTensors()</c> is reachable on a base type.</summary>
    private static bool InheritsExtraTensorsHook(INamedTypeSymbol type)
    {
        for (var c = type.BaseType; c is not null; c = c.BaseType)
        {
            foreach (var m in c.GetMembers(ExtraTensorsHook))
            {
                if (m is IMethodSymbol ms && ms.Parameters.Length == 0 &&
                    (ms.IsVirtual || ms.IsOverride || ms.IsAbstract)) return true;
            }
        }
        return false;
    }

    private static string GenerateExtraTensorsSource(INamedTypeSymbol classSymbol, string elem,
                                                     List<string> tensors)
    {
        var sb = OpenPartial(classSymbol, out var closers);
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Auto-generated: surfaces this model's tensor weights that live outside Layers,");
        sb.AppendLine("    /// in declaration order, after whatever the base already yields.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    /// <remarks>");
        sb.AppendLine("    /// Fields marked [Scratch] or [Buffer] are excluded, and a null field is skipped");
        sb.AppendLine("    /// rather than yielded -- an unfitted model has no weights there yet. Declare");
        sb.AppendLine($"    /// {ExtraTensorsHook}() by hand to take ownership and this disappears.");
        sb.AppendLine("    /// </remarks>");
        sb.AppendLine($"    protected override global::System.Collections.Generic.IEnumerable<Tensor<{elem}>> {ExtraTensorsHook}()");
        sb.AppendLine("    {");
        sb.AppendLine($"        foreach (var __t in base.{ExtraTensorsHook}()) yield return __t;");
        foreach (var name in tensors)
        {
            sb.AppendLine($"        if ({name} is not null) yield return {name};");
        }
        sb.AppendLine("    }");
        sb.AppendLine("}");
        for (int i = 0; i < closers; i++) sb.AppendLine("}");
        return sb.ToString();
    }

    private static StringBuilder OpenPartial(INamedTypeSymbol classSymbol, out int closers)
    {
        var ns = classSymbol.ContainingNamespace.ToDisplayString();
        var typeParams = classSymbol.TypeParameters.Length > 0
            ? "<" + string.Join(", ", classSymbol.TypeParameters.Select(tp => tp.Name)) + ">"
            : "";

        var containing = new List<INamedTypeSymbol>();
        for (var outer = classSymbol.ContainingType; outer is not null; outer = outer.ContainingType)
            containing.Insert(0, outer);
        closers = containing.Count;

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated />");
        sb.AppendLine("// Generated by ModelParameterGenerator");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using AiDotNet.Models.Parameters;");
        sb.AppendLine("using AiDotNet.Tensors.LinearAlgebra;");
        sb.AppendLine();
        sb.AppendLine($"namespace {ns};");
        sb.AppendLine();
        foreach (var ct in containing)
        {
            var ctp = ct.TypeParameters.Length > 0
                ? "<" + string.Join(", ", ct.TypeParameters.Select(tp => tp.Name)) + ">"
                : "";
            sb.AppendLine($"partial class {ct.Name}{ctp}");
            sb.AppendLine("{");
        }
        sb.AppendLine($"partial class {classSymbol.Name}{typeParams}");
        sb.AppendLine("{");
        return sb;
    }

    /// <summary>The registry is present when both the call and the overridable hook are reachable.</summary>
    private static bool InheritsRegistry(INamedTypeSymbol type)
    {
        bool call = false, hook = false;
        for (var c = type.BaseType; c is not null; c = c.BaseType)
        {
            foreach (var m in c.GetMembers())
            {
                if (m is not IMethodSymbol ms) continue;
                if (ms.Name == RegisterCall && ms.Parameters.Length == 1) call = true;
                else if (ms.Name == RegisterHook && ms.Parameters.Length == 0 &&
                         (ms.IsVirtual || ms.IsOverride || ms.IsAbstract)) hook = true;
            }
            if (call && hook) return true;
        }
        return false;
    }

    private static bool DeclaresOwn(INamedTypeSymbol type, string name) =>
        type.GetMembers(name).OfType<IMethodSymbol>().Any(m => m.Parameters.Length == 0);

    /// <summary>
    /// The numeric element type. Conventionally the parameter named <c>T</c>: models in this
    /// library are <c>Foo&lt;T&gt;</c> or descend from <c>ModelBase&lt;T, TInput, TOutput&gt;</c>,
    /// where the second and third are the input and output shapes rather than the scalar type.
    /// </summary>
    private static string? ElementTypeParam(INamedTypeSymbol type)
    {
        for (var c = type; c is not null; c = c.ContainingType)
        {
            foreach (var tp in c.TypeParameters)
            {
                if (tp.Name == "T") return tp.Name;
            }
        }
        return type.TypeParameters.Length > 0 ? type.TypeParameters[0].Name : null;
    }

    /// <summary>
    /// The write-through source for a field's type, or null when the field is not a weight this
    /// generator can describe. Deliberately narrow: a collection or an array of weights has no
    /// single well-defined ordering the author has agreed to, and guessing one would bake a wrong
    /// serialization layout into every future checkpoint. Those keep reporting AIDN084.
    /// </summary>
    private static string? SourceFor(ITypeSymbol type, string elem)
    {
        // A nullable annotation is not a disqualifier here as it is for layers: a fitted model
        // allocates in Fit, so 157 of this library's 339 model weight fields are legitimately
        // nullable. The field sources report zero for an absent field.
        if (type is not INamedTypeSymbol named) return null;
        var open = named.OriginalDefinition.ToDisplayString();
        if (named.TypeArguments.Length != 1) return null;
        if (named.TypeArguments[0].ToDisplayString() != elem) return null;

        if (open.StartsWith(TensorTypeName + "<", System.StringComparison.Ordinal))
            return "TensorFieldParameterSource";
        if (open.StartsWith(MatrixTypeName + "<", System.StringComparison.Ordinal))
            return "MatrixFieldParameterSource";
        if (open.StartsWith(VectorTypeName + "<", System.StringComparison.Ordinal))
            return "VectorFieldWriteThroughSource";
        return null;
    }

    private static bool HasAttr(IFieldSymbol field, INamedTypeSymbol? attr) =>
        attr is not null && field.GetAttributes()
            .Any(a => SymbolEqualityComparer.Default.Equals(a.AttributeClass, attr));

    private static string GenerateSource(INamedTypeSymbol classSymbol, string elem,
                                         List<(string Name, string SourceType)> fields)
    {
        var ns = classSymbol.ContainingNamespace.ToDisplayString();
        var typeParams = classSymbol.TypeParameters.Length > 0
            ? "<" + string.Join(", ", classSymbol.TypeParameters.Select(tp => tp.Name)) + ">"
            : "";

        var containing = new List<INamedTypeSymbol>();
        for (var outer = classSymbol.ContainingType; outer is not null; outer = outer.ContainingType)
            containing.Insert(0, outer);

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated />");
        sb.AppendLine("// Generated by ModelParameterGenerator");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("using AiDotNet.Models.Parameters;");
        sb.AppendLine("using AiDotNet.Tensors.LinearAlgebra;");
        sb.AppendLine();
        sb.AppendLine($"namespace {ns};");
        sb.AppendLine();

        foreach (var ct in containing)
        {
            var ctp = ct.TypeParameters.Length > 0
                ? "<" + string.Join(", ", ct.TypeParameters.Select(tp => tp.Name)) + ">"
                : "";
            sb.AppendLine($"partial class {ct.Name}{ctp}");
            sb.AppendLine("{");
        }

        sb.AppendLine($"partial class {classSymbol.Name}{typeParams}");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Auto-generated: registers this model's weight-bearing fields, in declaration");
        sb.AppendLine("    /// order, which is the serialization order.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine("    /// <remarks>");
        sb.AppendLine("    /// Fields marked [Scratch] or [Buffer] are excluded. To take ownership of the");
        sb.AppendLine("    /// order or the contents, declare RegisterComponents() by hand and this");
        sb.AppendLine("    /// generated override disappears.");
        sb.AppendLine("    /// </remarks>");
        sb.AppendLine($"    protected override void {RegisterHook}()");
        sb.AppendLine("    {");
        sb.AppendLine($"        base.{RegisterHook}();");
        foreach (var f in fields)
        {
            sb.AppendLine($"        {RegisterCall}(new {f.SourceType}<{elem}>(() => {f.Name}));");
        }
        sb.AppendLine("    }");
        sb.AppendLine("}");

        for (int i = 0; i < containing.Count; i++) sb.AppendLine("}");

        return sb.ToString();
    }
}
