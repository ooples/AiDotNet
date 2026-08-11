using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
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
    private static readonly DiagnosticDescriptor MissingPartialForAutomation = new(
        id: "AIDN085",
        title: "Model must be partial for its weights to be registered automatically",
        messageFormat: "'{0}' owns weights outside Layers but is not declared 'partial', so the "
                     + "parameter generator cannot register them and they reach no ParameterCount, "
                     + "no checkpoint and no optimizer",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Add 'partial' to the class declaration. That is the entire fix -- the generator "
                   + "emits the registration into a second partial declaration, so there is no hook "
                   + "to write, maintain or forget, and 'partial' changes no semantics on its own. "
                   + "This exists because a source generator cannot add the keyword to a declaration "
                   + "it does not own, and it is the one prerequisite the automation cannot supply "
                   + "for itself. Without it the weights are silently absent, which looks exactly "
                   + "like a model that has none.");

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



    private static readonly DiagnosticDescriptor CountUsedAsReadiness = new(
        id: "AIDN087",
        title: "ParameterCount compared against zero as a readiness test",
        messageFormat: "'{0}' tests ParameterCount against zero; a zero count means \"not sized yet\", "
                     + "not \"has no parameters\", so this branch treats a deferred component as an "
                     + "empty one",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Ask about READINESS instead. A lazily sized component reports 0 until its shape "
                   + "arrives, so `count == 0` conflates 'nothing to do' with 'ask me later' and the two "
                   + "need opposite handling: the first is a no-op, the second must park the payload and "
                   + "replay it at materialization. This is not hypothetical -- LayerBase carried a "
                   + "comment saying a zero count 'is not a claim that the layer has no parameters; it is "
                   + "the layer saying it does not know yet' directly above a guard that threw on exactly "
                   + "that, rejecting every restore into a deferred layer and accounting for ~104 CI "
                   + "failures. Prefer ParameterLayoutSnapshot readiness (ParameterFree vs ShapeDeferred "
                   + "vs ShapeResolvedUnmaterialized vs Materialized), or on a layer the local pair "
                   + "IsShapeResolved || ParametersAreConstructionSized. ParameterFree is the only state "
                   + "that genuinely means zero.");

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // AIDN087 is a CONSUMPTION rule, so it works over expressions rather than declarations: the
        // other rules in this file catch "you wrote a surface you should not have", and this one
        // catches "you asked the surface the wrong question". Both failure modes ship silently.
        var zeroCountComparisons = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => IsParameterCountZeroComparison(node),
                transform: static (ctx, _) => ctx.Node.GetLocation())
            .Where(static loc => loc is not null);

        context.RegisterSourceOutput(zeroCountComparisons.Collect(), static (spc, locations) =>
        {
            foreach (var loc in locations)
            {
                if (loc is null) continue;
                var file = loc.SourceTree?.FilePath ?? string.Empty;

                // The manifest and the layer base are where the readiness distinction is DEFINED, so
                // they are the two places allowed to compare against zero while implementing it.
                if (file.EndsWith("ParameterManifest.cs", System.StringComparison.OrdinalIgnoreCase)
                    || file.EndsWith("ParameterComponentRegistry.cs", System.StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                spc.ReportDiagnostic(Diagnostic.Create(
                    CountUsedAsReadiness, loc, System.IO.Path.GetFileNameWithoutExtension(file)));
            }
        });

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

                        // ModelParameterGenerator registers a model's weight fields for it, and this
                        // analyzer cannot see that: generated trees are not in the compilation the
                        // analyzer runs against, so the generated RegisterComponents is invisible and
                        // every automatically-handled field would be reported anyway. Rather than
                        // depend on generator/analyzer ordering, ask the same question the generator
                        // asks. The two must agree or the build contradicts itself -- so the predicate
                        // below is deliberately the same shape as ModelParameterGenerator.SourceFor
                        // and its surrounding gates.
                        //
                        // The point of AIDN084 survives, and sharpens: what remains reported is
                        // exactly the weights automation CANNOT reach -- collections and arrays with
                        // no author-agreed ordering, tensors over some other element type, and types
                        // on a root that has no registry yet.
                        bool automatable = !type.IsAbstract && IsPartial(type) && !declares;

                        // AIDN085: the one prerequisite of the automation that a source generator
                        // cannot supply for itself. C# does not let a generator add `partial` to
                        // someone else's declaration, so the build has to ask -- and asking is enough,
                        // because the keyword is the entire fix and is inert on its own.
                        //
                        // Covers weight FIELDS and LAYER-BEARING members alike. The layer case was
                        // first attempted as a separate diagnostic that tried to prove the layers
                        // were unreachable -- never added to Layers, surfaced by nothing. That does
                        // not work: reachability is a runtime aliasing property and the idioms are
                        // many. Autoformer writes `_encoderLayers.Add(Layers[i])`, making the field a
                        // VIEW INTO Layers rather than a separate stack; another model registers by
                        // a nameof(...) list. Both were accused, both were fine.
                        //
                        // Asking for `partial` needs no such proof. The generated hook seeds Layers
                        // into its seen-set and deduplicates by reference, so discovering a member
                        // whose layers are ALREADY reachable yields nothing at all. The keyword is
                        // therefore correct whether or not the layers were orphaned -- which is why
                        // this can be demanded without ever having to decide which case it is.
                        // NOT gated on `declares`. A class that declares the TENSORS hook can
                        // still hold LAYER members needing automation -- LLaVANeuralNetwork does,
                        // and the wholesale check silenced the demand for its grounding head.
                        // Each member kind is gated on the hook that would actually cover it.
                        bool coversFields = DeclaresAnyOf(type, "RegisterComponents",
                                                          "GetExtraTrainableTensors",
                                                          "GetParameterChunks");
                        bool coversLayers = DeclaresAnyOf(type, "GetExtraTrainableLayers");
                        if (!type.IsAbstract && !IsPartial(type)
                            && (InheritsRegistry(type) || InheritsExtraTensorsHook(type))
                            && type.GetMembers().Any(m =>
                                   !m.IsStatic && !m.IsImplicitlyDeclared
                                   && !HasAnyAttribute(m, "BufferAttribute", "Buffer",
                                                          "ScratchAttribute", "Scratch",
                                                          "ParameterAliasAttribute", "ParameterAlias")
                                   && ((!coversFields && m is IFieldSymbol f
                                        && f.AssociatedSymbol is null
                                        && IsWeightCapableType(f.Type))
                                       || (!coversLayers && LayerBearingType(m) is not null)
                                       || IsComponentBearing(m)))
                            && modelLoc is not null)
                        {
                            spc.ReportDiagnostic(Diagnostic.Create(
                                MissingPartialForAutomation, modelLoc, type.Name));
                        }

                        // Neural networks use their already-wired tensor/layer hooks. Other roots use
                        // the generated stable-ID component registrar. Keeping this precedence in sync
                        // with ModelParameterGenerator prevents a tensor from reaching both folds.
                        bool generatorWillYieldTensors = automatable && InheritsExtraTensorsHook(type);
                        bool generatorWillRegister = automatable && !generatorWillYieldTensors
                                                     && InheritsRegistry(type);

                        if (!declares)
                        {
                            foreach (var f in type.GetMembers().OfType<IFieldSymbol>())
                            {
                                if (f.IsStatic || f.IsImplicitlyDeclared || f.AssociatedSymbol is not null) continue;
                                if (HasAnyAttribute(f, "BufferAttribute", "Buffer", "ScratchAttribute", "Scratch",
                                                       "ParameterAliasAttribute", "ParameterAlias")) continue;
                                if (!IsWeightCapableType(f.Type)) continue;
                                if (generatorWillRegister && GeneratorHandles(f, type)) continue;
                                if (generatorWillYieldTensors && GeneratorHandles(f, type)
                                    && IsTensorType(f.Type)) continue;

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
        var normalized = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        if (normalized is IArrayTypeSymbol arrayType)
            return IsTensorType(arrayType.ElementType);
        if (normalized is not INamedTypeSymbol namedType)
            return false;

        string definition = namedType.OriginalDefinition.ToDisplayString();
        if (definition.StartsWith(
                "AiDotNet.Tensors.LinearAlgebra.Tensor<", System.StringComparison.Ordinal))
            return true;

        bool indexedCollection = namedType.TypeArguments.Length == 1 &&
            (definition.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal) ||
             definition.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal) ||
             definition.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal) ||
             definition.StartsWith("System.Collections.Generic.IEnumerable<", System.StringComparison.Ordinal) ||
             definition.StartsWith("System.Collections.Generic.IReadOnlyCollection<", System.StringComparison.Ordinal));
        if (indexedCollection)
            return IsTensorType(namedType.TypeArguments[0]);

        bool keyedCollection = namedType.TypeArguments.Length == 2 &&
            (definition.StartsWith("System.Collections.Generic.Dictionary<", System.StringComparison.Ordinal) ||
             definition.StartsWith("System.Collections.Generic.IDictionary<", System.StringComparison.Ordinal) ||
             definition.StartsWith("System.Collections.Generic.IReadOnlyDictionary<", System.StringComparison.Ordinal));
        if (keyedCollection)
            return IsTensorType(namedType.TypeArguments[1]);

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

    // ---- Mirror of ModelParameterGenerator's gates -------------------------------------------
    // These answer "will the generator register this?" so AIDN084 stays silent about work already
    // automated. They must track ModelParameterGenerator; if the two drift, the build either nags
    // about fields that are handled or goes quiet about ones that are not.

    /// <summary>
    /// The displayed type of a member carrying trainable layers, or null. Mirrors
    /// ModelParameterGenerator.LayerAccessorFor -- a single layer, a collection of layers, or a
    /// sub-network -- so what the build DEMANDS and what the generator SUPPLIES stay one set.
    /// </summary>
    private static string? LayerBearingType(ISymbol member)
    {
        ITypeSymbol? t = member switch
        {
            IFieldSymbol f when f.AssociatedSymbol is null => f.Type,
            IPropertySymbol p when p.GetMethod is not null => p.Type,
            _ => null,
        };
        if (t is null) return null;
        var bare = t.WithNullableAnnotation(NullableAnnotation.NotAnnotated);

        for (var c = bare as INamedTypeSymbol; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
                return t.ToDisplayString();
        }
        if (IsLayerLike(bare)) return t.ToDisplayString();

        ITypeSymbol? element = null;
        if (bare is IArrayTypeSymbol arr) element = arr.ElementType;
        else if (bare is INamedTypeSymbol named && named.TypeArguments.Length == 1)
        {
            var open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IEnumerable<", System.StringComparison.Ordinal))
                element = named.TypeArguments[0];
        }
        if (element is not null &&
            IsLayerLike(element.WithNullableAnnotation(NullableAnnotation.NotAnnotated)))
            return t.ToDisplayString();

        return null;
    }

    private static bool IsLayerLike(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol named) return false;
        if (named.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.Interfaces.ILayer<", System.StringComparison.Ordinal)) return true;
        for (var c = named; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.Layers.LayerBase<", System.StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>
    /// A member that IS a parameterized component, or a collection of them.
    /// </summary>
    /// <remarks>
    /// The ensemble shape: the parameters live in sub-models rather than in fields or layers.
    /// Mirrors ModelParameterGenerator.ComponentKindFor so the build demands `partial` for exactly
    /// the shapes the generator can then handle -- if the two disagree, a model either gets nagged
    /// with no fix available or is quietly left unautomated. The element type is deliberately NOT
    /// matched here: over-demanding `partial` costs a keyword, while under-demanding it costs a
    /// silently unregistered sub-model.
    /// </remarks>
    private static bool IsComponentBearing(ISymbol member)
    {
        ITypeSymbol? t = member switch
        {
            IFieldSymbol f when f.AssociatedSymbol is null => f.Type,
            IPropertySymbol p when p.GetMethod is not null => p.Type,
            _ => null,
        };
        if (t is null) return false;
        var bare = t.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        if (IsParameterSourceLike(bare)) return true;

        ITypeSymbol? element = null;
        if (bare is IArrayTypeSymbol arr) element = arr.ElementType;
        else if (bare is INamedTypeSymbol named && named.TypeArguments.Length == 1)
        {
            var open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("System.Collections.Generic.", System.StringComparison.Ordinal))
                element = named.TypeArguments[0];
        }
        return element is not null
               && IsParameterSourceLike(element.WithNullableAnnotation(NullableAnnotation.NotAnnotated));
    }

    private static bool IsParameterSourceLike(ITypeSymbol type)
    {
        // A sub-network is surfaced as LAYERS instead; counting it here as well would demand the
        // keyword for a shape that is already handled by the other route.
        for (var c = type as INamedTypeSymbol; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
                return false;
        }
        foreach (var i in type.AllInterfaces)
        {
            if (i.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.Interfaces.IParameterSource<", System.StringComparison.Ordinal))
                return true;
        }
        return type is INamedTypeSymbol n
               && n.OriginalDefinition.ToDisplayString()
                   .StartsWith("AiDotNet.Interfaces.IParameterSource<", System.StringComparison.Ordinal);
    }

    private static bool DeclaresAnyOf(INamedTypeSymbol type, params string[] names)
    {
        foreach (var n in names)
        {
            if (type.GetMembers(n).OfType<IMethodSymbol>().Any()) return true;
        }
        return false;
    }

    /// <summary>The generator only emits into a partial declaration.</summary>
    private static bool IsPartial(INamedTypeSymbol type)
    {
        foreach (var r in type.DeclaringSyntaxReferences)
        {
            if (r.GetSyntax() is ClassDeclarationSyntax c &&
                c.Modifiers.Any(m => m.Text == "partial")) return true;
        }
        return false;
    }

    /// <summary>An overridable GetExtraTrainableTensors() is reachable on a base type.</summary>
    private static bool InheritsExtraTensorsHook(INamedTypeSymbol type)
    {
        for (var c = type.BaseType; c is not null; c = c.BaseType)
        {
            foreach (var m in c.GetMembers("GetExtraTrainableTensors"))
            {
                if (m is IMethodSymbol ms && ms.Parameters.Length == 0 &&
                    (ms.IsVirtual || ms.IsOverride || ms.IsAbstract)) return true;
            }
        }
        return false;
    }

    /// <summary>Both the registry call and an overridable hook must be inherited.</summary>
    private static bool InheritsRegistry(INamedTypeSymbol type)
    {
        bool call = false, hook = false;
        for (var c = type.BaseType; c is not null; c = c.BaseType)
        {
            foreach (var m in c.GetMembers())
            {
                if (m is not IMethodSymbol ms) continue;
                if (ms.Name == "RegisterParameterComponent" && ms.Parameters.Length == 1) call = true;
                else if (ms.Name == "RegisterComponents" && ms.Parameters.Length == 0 &&
                         (ms.IsVirtual || ms.IsOverride || ms.IsAbstract)) hook = true;
            }
            if (call && hook) return true;
        }
        return false;
    }

    /// <summary>
    /// Mirrors ModelParameterGenerator's scalar, indexed-collection and keyed-collection support.
    /// </summary>
    private static bool GeneratorHandles(IFieldSymbol f, INamedTypeSymbol type)
    {
        if (f.Name.EndsWith("Gradient", System.StringComparison.Ordinal) ||
            f.Name.EndsWith("Gradients", System.StringComparison.Ordinal)) return false;

        string? elementType = null;
        for (var current = type; current is not null && elementType is null; current = current.ContainingType)
        {
            foreach (var typeParameter in current.TypeParameters)
            {
                if (typeParameter.Name != "T") continue;
                elementType = typeParameter.Name;
                break;
            }
        }
        if (elementType is null && type.TypeParameters.Length > 0)
            elementType = type.TypeParameters[0].Name;
        if (elementType is null)
        {
            for (var baseType = type.BaseType; baseType is not null; baseType = baseType.BaseType)
            {
                string definition = baseType.OriginalDefinition.ToDisplayString();
                bool parameterRoot =
                    definition.StartsWith("AiDotNet.Models.ModelBase<", System.StringComparison.Ordinal) ||
                    definition.StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal);
                if (!parameterRoot || baseType.TypeArguments.Length == 0 ||
                    baseType.TypeArguments[0].TypeKind == TypeKind.TypeParameter) continue;

                elementType = baseType.TypeArguments[0].ToDisplayString();
                break;
            }
        }
        return elementType is not null && GeneratorNumericType(f.Type, elementType);
    }

    private static bool GeneratorNumericType(ITypeSymbol type, string elementType)
    {
        var normalized = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        if (normalized is IArrayTypeSymbol arrayType)
            return GeneratorNumericType(arrayType.ElementType, elementType);
        if (normalized is not INamedTypeSymbol namedType)
            return false;

        string definition = namedType.OriginalDefinition.ToDisplayString();
        if (namedType.TypeArguments.Length == 1)
        {
            bool numericContainer = namedType.TypeArguments[0].ToDisplayString() == elementType &&
                (definition.StartsWith("AiDotNet.Tensors.LinearAlgebra.Tensor<", System.StringComparison.Ordinal) ||
                 definition.StartsWith("AiDotNet.Tensors.LinearAlgebra.Matrix<", System.StringComparison.Ordinal) ||
                 definition.StartsWith("AiDotNet.Tensors.LinearAlgebra.Vector<", System.StringComparison.Ordinal));
            if (numericContainer) return true;

            bool indexedCollection =
                definition.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal) ||
                definition.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal) ||
                definition.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal) ||
                definition.StartsWith("System.Collections.Generic.IEnumerable<", System.StringComparison.Ordinal) ||
                definition.StartsWith("System.Collections.Generic.IReadOnlyCollection<", System.StringComparison.Ordinal);
            return indexedCollection && GeneratorNumericType(namedType.TypeArguments[0], elementType);
        }

        if (namedType.TypeArguments.Length == 2)
        {
            bool keyedCollection =
                definition.StartsWith("System.Collections.Generic.Dictionary<", System.StringComparison.Ordinal) ||
                definition.StartsWith("System.Collections.Generic.IDictionary<", System.StringComparison.Ordinal) ||
                definition.StartsWith("System.Collections.Generic.IReadOnlyDictionary<", System.StringComparison.Ordinal);
            return keyedCollection && GeneratorNumericType(namedType.TypeArguments[1], elementType);
        }
        return false;
    }

    /// <summary>
    /// True for <c>X.ParameterCount == 0</c> / <c>!= 0</c> in either operand order, including
    /// <c>ParameterCount</c> reached without a receiver.
    /// </summary>
    private static bool IsParameterCountZeroComparison(SyntaxNode node)
    {
        if (node is not BinaryExpressionSyntax binary) return false;
        if (!binary.IsKind(SyntaxKind.EqualsExpression) && !binary.IsKind(SyntaxKind.NotEqualsExpression))
            return false;

        return (NamesParameterCount(binary.Left) && IsZero(binary.Right))
            || (NamesParameterCount(binary.Right) && IsZero(binary.Left));
    }

    private static bool NamesParameterCount(ExpressionSyntax expression) => expression switch
    {
        MemberAccessExpressionSyntax member => member.Name.Identifier.ValueText == "ParameterCount",
        IdentifierNameSyntax identifier => identifier.Identifier.ValueText == "ParameterCount",
        _ => false,
    };

    private static bool IsZero(ExpressionSyntax expression)
        => expression is LiteralExpressionSyntax literal
           && literal.IsKind(SyntaxKind.NumericLiteralExpression)
           && literal.Token.ValueText == "0";
}
