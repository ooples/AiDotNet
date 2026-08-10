using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Drives the migration to automatic parameter and buffer registration, and keeps it from
/// regressing once complete.
/// </summary>
/// <remarks>
/// <para>
/// A layer's parameter surface should be DERIVED, never written. <c>LayerBase</c> folds
/// <c>ParameterCount</c>, <c>GetParameters</c> and <c>SetParameters</c> out of one ordered
/// enumeration — own tensors, registered buffers, then sub-layers — so a hand-written copy can only
/// restate that fold or silently disagree with it. Disagreement is not cosmetic:
/// <c>SetParameters</c> pairs by length, so a checkpoint restores into the wrong tensors and the
/// model keeps its initial weights with nothing failing.
/// </para>
/// <para>
/// The measured cost of leaving this to authors: RWKVLayer registered 8 weight matrices and omitted
/// 10 more learned tensors — both LayerNorm affine pairs, the time- and channel-mixing coefficients
/// RWKV is named for, and the first-token bonus. The optimizer never updated them, and nothing
/// reported it, because the layer's three hand-written surfaces agreed with each other on the wrong
/// set.
/// </para>
/// <para>
/// Implemented as an <c>IIncrementalGenerator</c> that reports diagnostics rather than a
/// <c>DiagnosticAnalyzer</c>: this project wires generators through
/// <c>OutputItemType="Analyzer"</c> and standalone analyzer types are not loaded, which is why the
/// first version compiled, shipped the right IDs, and silently reported nothing. Every other
/// diagnostic in this repo (AIDN050-052 and friends) uses this shape.
/// </para>
/// <para>
/// Warning severity during migration so the build stays green while layers are converted. These
/// become errors once the count reaches zero, at which point forgetting is impossible rather than
/// merely detectable.
/// </para>
/// </remarks>
[Generator]
public class ParameterAutomationAnalyzer : IIncrementalGenerator
{
    private const string Category = "AiDotNet.ParameterAutomation";

    private static readonly DiagnosticDescriptor MissingAutoParameters = new(
        id: "AIDN080",
        title: "Layer does not use automatic parameter discovery",
        messageFormat: "Layer '{0}' is not marked [AutoParameters]; its tensor fields must be registered by hand, which is how parameters get silently omitted from training",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Add [AutoParameters] so every non-nullable tensor field is registered automatically. " +
                     "Mark exceptions with [Buffer] (persistent, never trained) or [Scratch] (transient).");

    private static readonly DiagnosticDescriptor RedundantParameterSurface = new(
        id: "AIDN081",
        title: "Parameter surface is derived and should not be overridden",
        messageFormat: "'{0}' overrides {1}; LayerBase derives it from the same registry, so this can only restate the fold or drift from it",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Delete the override and let LayerBase derive the value. ParameterCount, GetParameters and " +
                     "SetParameters fold one enumeration in one order, so they cannot disagree.");

    private static readonly DiagnosticDescriptor RedundantModelSurface = new(
        id: "AIDN082",
        title: "Model parameter surface is derived and should not be overridden",
        messageFormat: "'{0}' overrides {1}; the model base derives it from the registered components, so this can only restate the fold or drift from it",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Declare the model's components in RegisterComponents and delete the override. " +
                     "A hand-written model surface is how a count and a vector come to disagree: 44 " +
                     "diffusion models mixed a COUNT from one child with a VECTOR LENGTH from another " +
                     "in one expression, and VideoUNetPredictor's estimate was nine times out.");

    private static readonly DiagnosticDescriptor UndiscoverableWeight = new(
        id: "AIDN083",
        title: "Field holds weights the parameter generator cannot see",
        messageFormat: "'{0}.{1}' is {2}, so automatic discovery skips it; it contributes to no ParameterCount, no checkpoint and no optimizer",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Make it a non-readonly Tensor<T> if it is trained, or mark it [Buffer] (persistent " +
                     "state that is never optimized) or [Scratch] (rebuilt each forward). Say which — " +
                     "silence here is indistinguishable from an oversight, and the parameter-count " +
                     "contract test cannot catch it because the count and the vector omit the SAME field " +
                     "and therefore agree on a wrong answer. InformerModel reported 1,688 parameters " +
                     "against a real 167,640 this way: every Q/K/V/O projection, FFN weight and LayerNorm " +
                     "gain it owned was readonly, so nothing counted, saved or trained them.");
    private static readonly DiagnosticDescriptor UndeclaredModelWeight = new(
        id: "AIDN084",
        title: "Model holds weights it never declares to the parameter walk",
        messageFormat: "'{0}' holds '{1}' ({2}) outside Layers and declares no parameter components; "
                     + "it reaches no ParameterCount, no checkpoint and no optimizer",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Declare it: yield it from GetExtraTrainableTensors (NeuralNetworkBase), or register "
                   + "it in RegisterComponents (the registry-backed roots). [TrainableParameter] will NOT "
                   + "help -- TrainableParameterGenerator only processes LayerBase subclasses, so on a "
                   + "model the attribute is silently inert. If the field is not trainable, say so with "
                   + "[Buffer] or [Scratch]. This is the defect the count-vs-vector contract test cannot "
                   + "see, because an undeclared weight is missing from the count AND the vector, so the "
                   + "two agree on a wrong answer: LLaVA counted 512 weights it never handed out, and "
                   + "Flamingo's Perceiver Resampler latents -- the array the whole architecture is named "
                   + "for -- were in neither surface and were lost on every save.");


    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var classes = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax,
                transform: static (ctx, _) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) as INamedTypeSymbol)
            .Where(static s => s is not null);

        context.RegisterSourceOutput(classes.Collect(), static (spc, symbols) =>
        {
            var seen = new System.Collections.Generic.HashSet<string>();
            foreach (var type in symbols)
            {
                if (type is null || type.IsAbstract || type.TypeKind != TypeKind.Class) continue;

                // EVERY root whose base derives the parameter surface. This used to name only the two
                // diffusion roots, so a hand-written surface on a NeuralNetworkBase, ModelBase,
                // ClassifierBase, RegressionBase, ClusteringBase, RL or TimeSeries model was invisible
                // to the compiler -- which is how ~330 of them survived. Naming all of them turns the
                // remaining work into a build-time list instead of something found by sweeping.
                if (ExtendsAny(type, "AiDotNet.Diffusion.DiffusionModelBase<",
                                     "AiDotNet.Diffusion.VAE.VAEModelBase<",
                                     "AiDotNet.NeuralNetworks.NeuralNetworkBase<",
                                     "AiDotNet.Models.ModelBase<",
                                     "AiDotNet.Models.ModelWrapperBase<",
                                     "AiDotNet.Regression.RegressionBase<",
                                     "AiDotNet.Classification.ClassifierBase<",
                                     "AiDotNet.Clustering.ClusteringBase<",
                                     "AiDotNet.ReinforcementLearning.Agents.ReinforcementLearningAgentBase<",
                                     "AiDotNet.TimeSeries.TimeSeriesModelBase<"))
                {
                    var modelLoc = type.Locations.FirstOrDefault(l => l.IsInSource);
                    if (modelLoc is not null && seen.Add(type.ToDisplayString()))
                    {
                        foreach (var member in type.GetMembers())
                        {
                            if (!member.IsOverride) continue;
                            string? ms = member switch
                            {
                                IPropertySymbol p2 when p2.Name == "ParameterCount" => "ParameterCount",
                                IMethodSymbol m2 when m2.Name == "GetParameters" && m2.Parameters.Length == 0 => "GetParameters",
                                IMethodSymbol m2 when m2.Name == "SetParameters" && m2.Parameters.Length == 1 => "SetParameters",
                                IMethodSymbol m3 when m3.Name == "UpdateParameters" && m3.Parameters.Length == 1
                                                      && m3.Parameters[0].Type.OriginalDefinition.ToDisplayString()
                                                          .StartsWith("AiDotNet.Tensors.LinearAlgebra.Vector<", System.StringComparison.Ordinal)
                                    => "UpdateParameters",
                                _ => null,
                            };
                            if (ms is null) continue;
                            var mloc = member.Locations.FirstOrDefault(l => l.IsInSource);
                            if (mloc is not null)
                                spc.ReportDiagnostic(Diagnostic.Create(RedundantModelSurface, mloc, type.Name, ms));
                        }

                        // AIDN084: weights the model owns but never declares. The count-vs-vector
                        // contract test is blind to these -- an undeclared weight is missing from the
                        // count AND the vector, so the two agree on a wrong answer. Only reported when
                        // the model declares NOTHING, so a model that already uses the hook or the
                        // registry is assumed to know what it owns.
                        bool declares = type.GetMembers().Any(m =>
                            m.Name is "GetExtraTrainableTensors" or "GetExtraTrainableLayers"
                                   or "RegisterComponents" or "GetParameterChunks");
                        if (!declares)
                        {
                            foreach (var f in type.GetMembers().OfType<IFieldSymbol>())
                            {
                                if (f.IsStatic || f.IsImplicitlyDeclared || f.AssociatedSymbol is not null) continue;
                                if (HasAnyAttribute(f, "BufferAttribute", "Buffer", "ScratchAttribute", "Scratch")) continue;
                                if (!IsWeightCapableType(f.Type)) continue;

                                var fl = f.Locations.FirstOrDefault(l => l.IsInSource);
                                if (fl is not null)
                                    spc.ReportDiagnostic(Diagnostic.Create(
                                        UndeclaredModelWeight, fl, type.Name, f.Name, f.Type.ToDisplayString()));
                            }
                        }
                    }
                    continue;
                }

                if (!ExtendsLayerBase(type)) continue;
                if (!seen.Add(type.ToDisplayString())) continue;   // partial declarations

                var location = type.Locations.FirstOrDefault(l => l.IsInSource);
                if (location is null) continue;

                bool auto = type.GetAttributes()
                    .Any(a => a.AttributeClass?.Name is "AutoParametersAttribute" or "AutoParameters");

                // A layer holding no tensors of its own has nothing to discover; nagging it is noise.
                bool ownsTensors = type.GetMembers().OfType<IFieldSymbol>()
                    .Any(f => !f.IsStatic && !f.IsImplicitlyDeclared && f.AssociatedSymbol is null
                              && IsTensorType(f.Type));

                if (!auto && ownsTensors)
                    spc.ReportDiagnostic(Diagnostic.Create(MissingAutoParameters, location, type.Name));

                // AIDN083: fields the generator's discovery predicate will silently skip. Only checked
                // under [AutoParameters], where discovery is the ONLY way a weight reaches the surface;
                // without the attribute the author is registering by hand and knows what they own.
                if (auto)
                {
                    foreach (var f in type.GetMembers().OfType<IFieldSymbol>())
                    {
                        if (f.IsStatic || f.IsImplicitlyDeclared || f.AssociatedSymbol is not null) continue;
                        if (HasAnyAttribute(f, "BufferAttribute", "Buffer", "ScratchAttribute", "Scratch")) continue;

                        // Nullable is the sanctioned way to say "optional / not always present", and the
                        // gradient and cache fields that use it are legion. Flagging them would bury the
                        // real finding, so only the two skips that silently ate real weights are reported.
                        bool nullable = f.NullableAnnotation == NullableAnnotation.Annotated
                                        || f.Type.NullableAnnotation == NullableAnnotation.Annotated;
                        if (nullable) continue;

                        // An ARRAY of matrices is skipped for the same reason a single one is, and
                        // LoHaAdapter holds its Hadamard factors that way -- report the element type
                        // so the message names something the author can actually find in the file.
                        var probe = f.Type is IArrayTypeSymbol arr ? arr.ElementType : f.Type;

                        string? why = null;
                        if (IsTensorType(probe) && f.IsReadOnly)
                            why = "readonly";
                        else if (IsMatrixOrVectorType(probe))
                            why = probe.OriginalDefinition.ToDisplayString().Contains(".Matrix<")
                                ? (f.Type is IArrayTypeSymbol ? "an array of Matrix<T>" : "Matrix<T>") + " rather than Tensor<T>"
                                : (f.Type is IArrayTypeSymbol ? "an array of Vector<T>" : "Vector<T>") + " rather than Tensor<T>";

                        if (why is null) continue;
                        var fl = f.Locations.FirstOrDefault(l => l.IsInSource);
                        if (fl is null) continue;
                        spc.ReportDiagnostic(Diagnostic.Create(UndiscoverableWeight, fl, type.Name, f.Name, why));
                    }
                }

                foreach (var member in type.GetMembers())
                {
                    if (!member.IsOverride) continue;
                    string? surface = member switch
                    {
                        IPropertySymbol p when p.Name == "ParameterCount" => "ParameterCount",
                        IMethodSymbol m when m.Name == "GetParameters" && m.Parameters.Length == 0 => "GetParameters",
                        IMethodSymbol m when m.Name == "SetParameters" && m.Parameters.Length == 1 => "SetParameters",
                        _ => null,
                    };
                    if (surface is null) continue;
                    var ml = member.Locations.FirstOrDefault(l => l.IsInSource);
                    if (ml is null) continue;
                    spc.ReportDiagnostic(Diagnostic.Create(RedundantParameterSurface, ml, type.Name, surface));
                }
            }
        });
    }

    /// <summary>True when <paramref name="type"/> derives from any of the given base metadata prefixes.</summary>
    private static bool ExtendsAny(INamedTypeSymbol type, params string[] prefixes)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            var name = b.OriginalDefinition.ToDisplayString();
            foreach (var prefix in prefixes)
            {
                if (name.StartsWith(prefix, System.StringComparison.Ordinal)) return true;
            }
        }
        return false;
    }

    private static bool ExtendsLayerBase(INamedTypeSymbol type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.OriginalDefinition.ToDisplayString()
                .StartsWith("AiDotNet.NeuralNetworks.Layers.LayerBase<", System.StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    private static bool IsTensorType(ITypeSymbol type)
    {
        for (var c = type; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                .StartsWith("AiDotNet.Tensors.LinearAlgebra.Tensor<", System.StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Matrix&lt;T&gt; and Vector&lt;T&gt;: numeric containers the discovery predicate does not accept,
    /// so weights held in them reach no parameter surface. LoRALayer held its A and B this way and
    /// derived a ParameterCount of zero.
    /// </summary>
    private static bool IsMatrixOrVectorType(ITypeSymbol type)
    {
        var name = type.OriginalDefinition.ToDisplayString();
        return name.StartsWith("AiDotNet.Tensors.LinearAlgebra.Matrix<", System.StringComparison.Ordinal)
            || name.StartsWith("AiDotNet.Tensors.LinearAlgebra.Vector<", System.StringComparison.Ordinal);
    }
    /// <summary>
    /// Numeric containers that can hold weights: <c>Tensor&lt;T&gt;</c>, <c>Matrix&lt;T&gt;</c> and
    /// <c>Vector&lt;T&gt;</c>, including arrays and the common collections of them.
    /// </summary>
    /// <remarks>
    /// A TYPE test, deliberately, with no inspection of the field's name. An earlier version of
    /// AIDN084 guessed from names -- matching "weight", "bias", "gamma" and carving out plural
    /// "betas" because the diffusion noise schedule is spelled that way. That is a taxonomy nobody
    /// maintains: it misses a weight called <c>_theta</c>, claims a hyperparameter called
    /// <c>_weightDecay</c>, and silently changes meaning when a field is renamed. A diagnostic that
    /// is wrong in both directions gets suppressed, and then it protects nothing.
    /// <para>
    /// So the analyzer does not decide what a field IS. It asks the author to, once: declare it as a
    /// parameter, or mark it <c>[Buffer]</c> or <c>[Scratch]</c>. The answer lives in the code,
    /// survives renames, and the compiler enforces it from then on.
    /// </para>
    /// </remarks>
    private static bool IsWeightCapableType(ITypeSymbol type)
    {
        var probe = type;
        if (probe is IArrayTypeSymbol arr) probe = arr.ElementType;

        // One level of List<...> / IReadOnlyList<...> / Dictionary<_, ...>, which is how the
        // per-level and per-branch weight collections are held.
        if (probe is INamedTypeSymbol named && named.IsGenericType && named.TypeArguments.Length > 0)
        {
            var open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal)
                || open.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal)
                || open.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal)
                || open.StartsWith("System.Collections.Generic.Dictionary<", System.StringComparison.Ordinal))
            {
                probe = named.TypeArguments[named.TypeArguments.Length - 1];
                if (probe is IArrayTypeSymbol inner) probe = inner.ElementType;
                if (probe is INamedTypeSymbol n2 && n2.IsGenericType
                    && n2.OriginalDefinition.ToDisplayString()
                        .StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal))
                {
                    probe = n2.TypeArguments[0];
                }
            }
        }

        return IsTensorType(probe) || IsMatrixOrVectorType(probe);
    }


    private static bool HasAnyAttribute(ISymbol symbol, params string[] names)
    {
        foreach (var a in symbol.GetAttributes())
        {
            var n = a.AttributeClass?.Name;
            if (n is null) continue;
            foreach (var candidate in names)
            {
                if (string.Equals(n, candidate, System.StringComparison.Ordinal)) return true;
            }
        }
        return false;
    }
}
