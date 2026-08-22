using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Registers a model's explicitly classified numeric state with its parameter component registry, so a model
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
/// Numeric storage has no inferred default. The declaration supplies exactly one semantic role;
/// the shared semantic model is consumed by this generator and by compiler diagnostics, so a
/// nullable tensor, a field name, or a CLR type can never silently make state trainable.
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
    private const string TensorTypeName = "AiDotNet.Tensors.LinearAlgebra.Tensor";
    private const string MatrixTypeName = "AiDotNet.Tensors.LinearAlgebra.Matrix";
    private const string VectorTypeName = "AiDotNet.Tensors.LinearAlgebra.Vector";

    private const string RegisterHook = "RegisterComponents";
    private const string RegisterCall = "RegisterParameterComponent";
    private const string ExtraTensorsHook = "GetExtraTrainableTensors";
    private const string ExtraLayersHook = "GetExtraTrainableLayers";
    private const string RebindLayerAliasesHook = "RebindLayerAliases";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // The pipeline used to cache ClassDeclarationSyntax. A syntax node is the same class of
        // leak as a symbol: it holds its SyntaxTree, which roots the entire Compilation, so every
        // cached entry pinned a compilation in memory.
        //
        // It now carries only each candidate's metadata name (a string), and the symbol is
        // re-resolved from the compilation at the point of use.
        //
        // DELIBERATE SCOPE LIMIT: this fixes the retention, not the re-execution. The per-class
        // analysis below is ~1000 lines of symbol walking that would have to move into the
        // transform to make the pipeline genuinely cacheable, and restructuring it carries far more
        // risk than the incremental win is worth. CompilationProvider therefore stays, and the
        // generator still re-runs on every compilation -- but it no longer holds compilations alive.
        var classNames = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax cds &&
                    cds.Modifiers.Any(m => m.Text == "partial"),
                transform: static (ctx, _) =>
                    ctx.SemanticModel.GetDeclaredSymbol(ctx.Node) is INamedTypeSymbol symbol
                        ? GeneratorHelpers.MetadataNameOf(symbol)
                        : null)
            .Where(static n => n is not null)
            .Select(static (n, _) => n ?? string.Empty);

        var compilationAndClasses = context.CompilationProvider.Combine(classNames.Collect());
        context.RegisterSourceOutput(compilationAndClasses,
            static (spc, source) => Execute(source.Left, source.Right, spc));
    }


    private static void Execute(Compilation compilation,
                                ImmutableArray<string> classMetadataNames,
                                SourceProductionContext context)
    {
        if (classMetadataNames.IsDefaultOrEmpty) return;

        var processed = new HashSet<string>();
        var resolved = new HashSet<string>();

        foreach (var metadataName in classMetadataNames)
        {
            if (metadataName.Length == 0) continue;

            // Partial classes contribute one name per declaration; resolve each distinct name once.
            if (!resolved.Add(metadataName)) continue;

            if (GeneratorHelpers.ResolveSourceType(compilation, metadataName) is not INamedTypeSymbol classSymbol) continue;
            var elem = ElementTypeParam(classSymbol);
            if (elem is null) continue;

            // Two trunks, two hooks.
            //
            // ModelBase and its descendants have the component registry, which takes a source per
            // field and so can carry tensors, matrices and vectors alike.
            //
            // NeuralNetworkBase uses its tensor/layer hooks for trainable storage because those hooks
            // are already consumed by flat restore, gradient collection and GPU mirroring. It also has
            // a registry for persistent non-trainable state. Keeping the two roles separate prevents a
            // buffer from entering the optimizer while still making it checkpoint-visible.
            bool onNetworkTrunk = InheritsExtraTensorsHook(classSymbol);
            bool hasRegistry = InheritsRegistry(classSymbol);

            // The two hooks are suppressed INDEPENDENTLY. Coupling them was a bug: Flamingo declares
            // its own GetExtraTrainableTensors, which silently also suppressed the layers hook, so
            // its vision tower stayed invisible -- the very defect being automated away. Declaring
            // one hook is a claim about that hook only.
            bool emitTensors = onNetworkTrunk && !DeclaresOwn(classSymbol, ExtraTensorsHook);
            bool emitLayers = onNetworkTrunk && !DeclaresOwn(classSymbol, ExtraLayersHook);
            bool emitLayerAliasRebinding = onNetworkTrunk && !DeclaresLayerAliasRebinding(classSymbol);
            if (!hasRegistry && !emitTensors && !emitLayers && !emitLayerAliasRebinding) continue;

            if (!processed.Add(classSymbol.ToDisplayString())) continue;

            if (onNetworkTrunk)
            {
                var tensors = new List<string>();
                var layerGroups = new List<string>();
                var layerAliasRebinders = new List<string>();
                var persistentFields = new List<(string Name, string SourceExpression, string Role, string Availability)>();
                foreach (var member in classSymbol.GetMembers())
                {
                    if (member is IFieldSymbol tf)
                    {
                        if (tf.IsStatic || tf.IsConst || tf.IsImplicitlyDeclared || tf.AssociatedSymbol is not null)
                            continue;
                        if (emitLayerAliasRebinding)
                        {
                            var rebinder = LayerAliasRebinderFor(tf, elem);
                            if (rebinder is not null) layerAliasRebinders.Add(rebinder);
                        }
                        var classification = ParameterMemberSemanticModel.Classify(tf);
                        if (IsNonOptimizerPersistentState(classification.Kind) && hasRegistry)
                        {
                            var persistentSource = SourceExpressionFor(
                                tf, elem, allowPrimitive: true,
                                allowDeferredVectorReplacement: HasFitAvailability(
                                    tf, classification.Kind),
                                allowSerializedObject:
                                    classification.Kind == ParameterMemberSemanticModel.Kind.Fitted);
                            if (persistentSource is not null)
                            {
                                persistentFields.Add((tf.Name, persistentSource,
                                    RoleExpression(classification.Kind),
                                    AvailabilityExpression(tf, classification.Kind)));
                                continue;
                            }
                        }
                        if (emitTensors)
                        {
                            var nestedTensors = NestedNetworkTensorAccessorFor(tf.Type, tf.Name, elem);
                            if (nestedTensors is not null) tensors.Add(nestedTensors);
                        }
                        var tensorAccessor = classification.Kind == ParameterMemberSemanticModel.Kind.Trainable
                            ? TensorAccessorFor(tf.Type, tf.Name, elem)
                            : null;
                        if (tensorAccessor is not null)
                        {
                            if (emitTensors) tensors.Add(tensorAccessor);
                            continue;
                        }
                        if (!emitLayers) continue;
                        var acc = LayerAccessorFor(tf.Type, tf.Name, elem);
                        if (acc is not null) layerGroups.Add(acc);
                    }
                    else if (member is IPropertySymbol tp)
                    {
                        // Sub-networks are conventionally exposed as properties (GAN's Generator and
                        // Discriminator, StyleGAN's MappingNetwork). Fields alone would miss them.
                        if (tp.IsStatic || tp.IsImplicitlyDeclared || tp.GetMethod is null) continue;
                        if (emitLayerAliasRebinding)
                        {
                            var rebinder = LayerAliasRebinderFor(tp, elem);
                            if (rebinder is not null) layerAliasRebinders.Add(rebinder);
                        }
                        if (!emitLayers) continue;
                        var classification = ParameterMemberSemanticModel.Classify(tp);
                        if (IsNonOptimizerPersistentState(classification.Kind) && hasRegistry)
                        {
                            var persistentSource = SourceExpressionFor(
                                tp, elem,
                                allowDeferredVectorReplacement: HasFitAvailability(
                                    tp, classification.Kind),
                                allowSerializedObject:
                                    classification.Kind == ParameterMemberSemanticModel.Kind.Fitted);
                            if (persistentSource is not null)
                            {
                                persistentFields.Add((tp.Name, persistentSource,
                                    RoleExpression(classification.Kind),
                                    AvailabilityExpression(tp, classification.Kind)));
                                continue;
                            }
                        }
                        if (emitTensors)
                        {
                            var nestedTensors = NestedNetworkTensorAccessorFor(tp.Type, tp.Name, elem);
                            if (nestedTensors is not null) tensors.Add(nestedTensors);
                        }
                        if (classification.Kind == ParameterMemberSemanticModel.Kind.Trainable)
                        {
                            var tensorAccessor = TensorAccessorFor(tp.Type, tp.Name, elem);
                            if (tensorAccessor is not null && emitTensors)
                            {
                                tensors.Add(tensorAccessor);
                                continue;
                            }
                        }
                        if (classification.IsDeclared) continue;
                        var acc = LayerAccessorFor(tp.Type, tp.Name, elem);
                        if (acc is not null) layerGroups.Add(acc);
                    }
                }

                if (tensors.Count > 0 || layerGroups.Count > 0 || layerAliasRebinders.Count > 0)
                {
                    context.AddSource(
                        HintName(classSymbol) + ".ModelExtraTensors.g.cs",
                        GenerateExtraTensorsSource(
                            classSymbol, elem, tensors, layerGroups, layerAliasRebinders));
                }
                if (persistentFields.Count > 0)
                {
                    context.AddSource(
                        HintName(classSymbol) + ".ModelPersistentState.g.cs",
                        GenerateSource(classSymbol, elem, persistentFields,
                            new List<(string Name, string SourceExpression, string Role, string Availability)>()));
                }
                continue;
            }

            var fields = new List<(string Name, string SourceExpression, string Role, string Availability)>();
            var components = new List<(string Name, string SourceExpression, string Role, string Availability)>();
            foreach (var member in classSymbol.GetMembers())
            {
                // A member that IS a parameterized component, or a collection of them. Every
                // IFullModel is an IParameterSource<T> already -- IParameterizable derives from it --
                // so an ensemble, a mixture of experts or a stacked model needs no adapter, only
                // discovery. The collection form is re-read on each access rather than snapshotted,
                // because members are routinely added after the one lazy registration has run.
                var classification = ParameterMemberSemanticModel.Classify(member);
                if (member is IFieldSymbol or IPropertySymbol
                    && !member.IsStatic && !member.IsImplicitlyDeclared
                    && classification.Kind is not ParameterMemberSemanticModel.Kind.Scratch
                        and not ParameterMemberSemanticModel.Kind.Alias
                        and not ParameterMemberSemanticModel.Kind.External
                        and not ParameterMemberSemanticModel.Kind.Conflicting)
                {
                    var kind = ComponentKindFor(MemberType(member), elem);
                    if (kind == "one")
                    {
                        components.Add((member.Name,
                            $"new ComponentAccessorParameterSource<{elem}>(() => {member.Name})",
                            RoleExpression(classification.Kind),
                            AvailabilityExpression(member, classification.Kind)));
                        continue;
                    }
                    if (kind == "many")
                    {
                        components.Add((member.Name,
                            $"new ComponentCollectionParameterSource<{elem}>(() => {member.Name})",
                            RoleExpression(classification.Kind),
                            AvailabilityExpression(member, classification.Kind)));
                        continue;
                    }
                }

                if (member is not IFieldSymbol and not IPropertySymbol) continue;
                if (member.IsStatic || member.IsImplicitlyDeclared || !classification.IsDeclared
                    || !IsPersistentState(classification.Kind)) continue;
                if (member is IFieldSymbol field && (field.IsConst || field.AssociatedSymbol is not null)) continue;
                if (member is IPropertySymbol property && property.GetMethod is null) continue;

                // Primitive CLR storage is ambiguous: a double may be a trainable bias, a threshold,
                // a tolerance, or a hyperparameter. Unlike Tensor/Matrix/Vector fields, it is only
                // automated when the author supplies an explicit semantic role.
                bool allowPrimitive = true;
                var sourceExpression = SourceExpressionFor(
                    member, elem, allowPrimitive,
                    allowDeferredVectorReplacement: HasFitAvailability(
                        member, classification.Kind),
                    allowSerializedObject:
                        classification.Kind == ParameterMemberSemanticModel.Kind.Fitted);
                if (sourceExpression is null) continue;
                fields.Add((member.Name, sourceExpression, RoleExpression(classification.Kind),
                    AvailabilityExpression(member, classification.Kind)));
            }

            if (fields.Count == 0 && components.Count == 0) continue;

            context.AddSource(HintName(classSymbol) + ".ModelParameters.g.cs",
                              GenerateSource(classSymbol, elem, fields, components));
        }
    }

    private static string HintName(INamedTypeSymbol t) =>
        t.ToDisplayString().Replace('.', '_').Replace('<', '_').Replace('>', '_');

    private static ITypeSymbol? MemberType(ISymbol m) => m switch
    {
        IFieldSymbol f when f.AssociatedSymbol is null => f.Type,
        IPropertySymbol p when p.GetMethod is not null => p.Type,
        _ => null,
    };

    /// <summary>
    /// "one" when the member IS a parameterized component, "many" when it is a collection of them,
    /// null otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the shape an ensemble, a mixture of experts, a stacked or boosted model has: the
    /// parameters are not in fields at all, they are in sub-models. Nothing needed adapting for it
    /// -- <c>IParameterizable&lt;T, TInput, TOutput&gt;</c> derives from
    /// <c>IParameterSource&lt;T&gt;</c>, so every <c>IFullModel</c> can already be registered. What
    /// was missing was discovery.
    /// </para>
    /// <para>
    /// Deliberately does NOT match a sub-network on the NeuralNetworkBase trunk: those are surfaced
    /// as LAYERS through GetExtraTrainableLayers, and matching them here as well would register the
    /// same weights twice through two different routes.
    /// </para>
    /// </remarks>
    private static string? ComponentKindFor(ITypeSymbol? type, string elem)
    {
        if (type is null) return null;
        var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);

        if (IsParameterSourceOf(bare, elem) && !IsNeuralNetworkBase(bare)) return "one";

        ITypeSymbol? element = null;
        if (bare is IArrayTypeSymbol arr) element = arr.ElementType;
        else if (bare is INamedTypeSymbol named && named.TypeArguments.Length == 1)
        {
            var open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IEnumerable<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IReadOnlyCollection<", System.StringComparison.Ordinal))
                element = named.TypeArguments[0];
        }
        if (element is null) return null;
        element = element.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        if (IsParameterSourceOf(element, elem) && !IsNeuralNetworkBase(element)) return "many";
        return null;
    }

    private static bool IsParameterSourceOf(ITypeSymbol type, string elem)
    {
        foreach (var i in type.AllInterfaces)
        {
            if (i.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.Interfaces.IParameterSource<", System.StringComparison.Ordinal)
                && i.TypeArguments.Length == 1
                && i.TypeArguments[0].ToDisplayString() == elem)
                return true;
        }
        if (type is INamedTypeSymbol n
            && n.OriginalDefinition.ToDisplayString()
                .StartsWith("AiDotNet.Interfaces.IParameterSource<", System.StringComparison.Ordinal)
            && n.TypeArguments.Length == 1
            && n.TypeArguments[0].ToDisplayString() == elem)
            return true;
        return false;
    }

    private static bool IsNeuralNetworkBase(ITypeSymbol type)
    {
        for (var c = type as INamedTypeSymbol; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
                return true;
        }
        return false;
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
                                                     List<string> tensors, List<string> layerGroups,
                                                     List<string> layerAliasRebinders)
    {
        var sb = OpenPartial(classSymbol, out var closers);

        if (tensors.Count > 0)
        {
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
            foreach (var accessor in tensors)
            {
                sb.AppendLine($"        foreach (var __extra in {accessor})");
                sb.AppendLine("        {");
                sb.AppendLine("            if (__extra is not null) yield return __extra;");
                sb.AppendLine("        }");
            }
            sb.AppendLine("    }");
        }

        if (layerGroups.Count > 0)
        {
            if (tensors.Count > 0) sb.AppendLine();
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Auto-generated: surfaces this model's trainable layers that live outside");
            sb.AppendLine("    /// <c>Layers</c> -- its own layer collections and its sub-networks' layers.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    /// <remarks>");
            sb.AppendLine("    /// <para>");
            sb.AppendLine("    /// Discovery order is declaration order, which is the serialization order.");
            sb.AppendLine("    /// Members marked [Scratch] or [Buffer] are excluded -- use [Buffer] for a frozen");
            sb.AppendLine("    /// teacher or target copy, which is not an independent parameter and would");
            sb.AppendLine("    /// otherwise inflate the count and be handed to an optimizer.");
            sb.AppendLine("    /// </para>");
            sb.AppendLine("    /// <para>");
            sb.AppendLine("    /// Each layer is yielded at most once, and never if it is already in <c>Layers</c>.");
            sb.AppendLine("    /// The base folds Layers and this hook back to back WITHOUT deduplicating, so a");
            sb.AppendLine("    /// model that both adds a sub-network's layers to Layers and owns the sub-network");
            sb.AppendLine("    /// -- WGANGP does exactly that -- would otherwise count every one of those weights");
            sb.AppendLine("    /// twice in ParameterCount and emit them twice from GetParameters.");
            sb.AppendLine("    /// </para>");
            sb.AppendLine("    /// </remarks>");
            sb.AppendLine("    protected override global::System.Collections.Generic.IEnumerable<"
                          + "global::AiDotNet.NeuralNetworks.Layers.LayerBase<" + elem + ">?> GetExtraTrainableLayers()");
            sb.AppendLine("    {");
            // A reference-identity list rather than HashSet + ReferenceEqualityComparer: that
            // comparer does not exist on net471, which this library still targets. Layer counts are
            // in the tens, so the linear scan costs nothing and the code compiles on every target.
            sb.AppendLine("        var __seen = new global::System.Collections.Generic.List<object>();");
            sb.AppendLine("        bool __IsNew(object __c)");
            sb.AppendLine("        {");
            sb.AppendLine("            for (int __k = 0; __k < __seen.Count; __k++)");
            sb.AppendLine("            {");
            sb.AppendLine("                if (ReferenceEquals(__seen[__k], __c)) return false;");
            sb.AppendLine("            }");
            sb.AppendLine("            __seen.Add(__c);");
            sb.AppendLine("            return true;");
            sb.AppendLine("        }");
            sb.AppendLine("        foreach (var __l in base.GetExtraTrainableLayers())");
            sb.AppendLine("        {");
            sb.AppendLine("            if (__l is not null) __IsNew(__l);");
            sb.AppendLine("            yield return __l;");
            sb.AppendLine("        }");
            sb.AppendLine("        for (int __i = 0; __i < Layers.Count; __i++)");
            sb.AppendLine("        {");
            sb.AppendLine("            if (Layers[__i] is object __own) __IsNew(__own);");
            sb.AppendLine("        }");
            foreach (var group in layerGroups)
            {
                sb.AppendLine($"        foreach (var __layer in {group})");
                sb.AppendLine("        {");
                sb.AppendLine($"            if (__layer is global::AiDotNet.NeuralNetworks.Layers.LayerBase<{elem}> __lb && __IsNew(__layer)) yield return __lb;");
                sb.AppendLine("        }");
            }
            sb.AppendLine("    }");
        }

        if (layerAliasRebinders.Count > 0)
        {
            if (tensors.Count > 0 || layerGroups.Count > 0) sb.AppendLine();
            sb.AppendLine("    /// <summary>");
            sb.AppendLine("    /// Auto-generated: rebinds named fields and collection views when the canonical");
            sb.AppendLine("    /// <c>Layers</c> graph is replaced by deserialization or eager cloning.");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine($"    protected override void {RebindLayerAliasesHook}(");
            sb.AppendLine($"        global::System.Collections.Generic.IReadOnlyList<global::AiDotNet.Interfaces.ILayer<{elem}>> previousLayers,");
            sb.AppendLine($"        global::System.Collections.Generic.IReadOnlyList<global::AiDotNet.Interfaces.ILayer<{elem}>> replacementLayers)");
            sb.AppendLine("    {");
            sb.AppendLine($"        base.{RebindLayerAliasesHook}(previousLayers, replacementLayers);");
            foreach (var rebinder in layerAliasRebinders)
                sb.AppendLine("        " + rebinder);
            sb.AppendLine("    }");
        }

        sb.AppendLine("}");
        for (int i = 0; i < closers; i++) sb.AppendLine("}");
        return sb.ToString();
    }

    /// <summary>
    /// An expression yielding the trainable layers a member contributes, or null when the member is
    /// not layer-bearing.
    /// </summary>
    /// <remarks>
    /// Two shapes cover what models in this library actually hold outside <c>Layers</c>: a collection
    /// of layers (SileroVad's conv and LSTM stacks, Flamingo's vision tower, BLIP3's Q-Former), and a
    /// whole sub-network (every GAN's generator and discriminator). Both were hand-written hooks
    /// before this, which is the same override in a different name -- an author writing the next one
    /// still has to know, and still forgets.
    /// </remarks>
    private static string? LayerAccessorFor(ITypeSymbol type, string name, string elem)
    {
        var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);

        // A sub-network: yield the layers it owns.
        for (var c = bare as INamedTypeSymbol; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
            {
                return $"EnumerateNestedNetworkLayers({name})";
            }
        }

        // A single layer held in its own member -- a lazily-built patch-embedding conv, a token
        // embedding, a projection head. NeuralNetworkBase's own documentation names this shape as
        // the reason GetExtraTrainableLayers exists, and it is the commonest way one weight goes
        // missing: it is not in Layers, it is not a collection, and nothing walks it.
        if (IsLayerOf(bare, elem))
        {
            // Nullable element type: these fields are frequently built lazily, so a non-nullable
            // array would be CS8601 at every site. The loop's `is LayerBase<T>` test drops nulls.
            return $"new global::AiDotNet.Interfaces.ILayer<{elem}>?[] {{ {name} }}";
        }

        // A collection of layers.
        ITypeSymbol? element = null;
        if (bare is IArrayTypeSymbol arr) element = arr.ElementType;
        else if (bare is INamedTypeSymbol named && named.TypeArguments.Length == 1)
        {
            var open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal) ||
                open.StartsWith("System.Collections.Generic.IEnumerable<", System.StringComparison.Ordinal))
            {
                element = named.TypeArguments[0];
            }
        }
        if (element is null) return null;

        // A collection of sub-networks owns a collection of layer collections. Flatten those in
        // the author's stable collection order so multi-scale networks and expert banks do not
        // disappear merely because the network boundary is one level deeper.
        element = element.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        for (var c = element as INamedTypeSymbol; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
            {
                var networkType = element.ToDisplayString();
                return $"({name} ?? (global::System.Collections.Generic.IEnumerable<{networkType}>)global::System.Array.Empty<{networkType}>()).SelectMany(__n => EnumerateNestedNetworkLayers(__n))";
            }
        }

        if (!IsLayerOf(element, elem)) return null;
        var et = element.WithNullableAnnotation(NullableAnnotation.NotAnnotated).ToDisplayString();
        return $"{name} ?? (global::System.Collections.Generic.IEnumerable<{et}>)global::System.Array.Empty<{et}>()";
    }

    /// <summary>
    /// An expression yielding raw trainable tensors owned by a nested network. Nested models are a
    /// graph boundary, not a layer-only boundary: omitting their model-owned tensors makes a parent
    /// checkpoint and optimizer view incomplete even when all child layers are discovered.
    /// </summary>
    private static string? NestedNetworkTensorAccessorFor(ITypeSymbol type, string name, string elem)
    {
        var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        for (var c = bare as INamedTypeSymbol; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
            {
                return $"EnumerateNestedNetworkTensors({name})";
            }
        }

        var element = CollectionElementType(bare);
        if (element is null) return null;
        element = element.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        for (var c = element as INamedTypeSymbol; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
            {
                var networkType = element.ToDisplayString();
                return $"({name} ?? (global::System.Collections.Generic.IEnumerable<{networkType}>)global::System.Array.Empty<{networkType}>()).SelectMany(__n => EnumerateNestedNetworkTensors(__n))";
            }
        }

        return null;
    }

    /// <summary>
    /// Emits type-safe lifecycle repair for a field/property that may be a view into Layers.
    /// Independent layer ownership is preserved because the base helpers only replace references
    /// found in the previous canonical graph.
    /// </summary>
    private static string? LayerAliasRebinderFor(ISymbol member, string elem)
    {
        var type = MemberType(member);
        if (type is null) return null;
        var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);

        if (IsLayerOf(bare, elem))
        {
            bool writable = member switch
            {
                IFieldSymbol field => !field.IsReadOnly,
                IPropertySymbol property => property.SetMethod is not null && !property.SetMethod.IsInitOnly,
                _ => false,
            };
            bool nullable = ParameterMemberSemanticModel.IsNullable(member);
            return writable
                ? nullable
                    ? $"{member.Name} = RebindLayerAlias({member.Name}, previousLayers, replacementLayers, nameof({member.Name}));"
                    : $"{member.Name} = RebindRequiredLayerAlias({member.Name}, previousLayers, replacementLayers, nameof({member.Name}));"
                : $"ValidateReadonlyLayerAlias({member.Name}, previousLayers, replacementLayers, nameof({member.Name}));";
        }

        var element = LayerCollectionElementType(bare);
        if (element is null || !IsLayerOf(
                element.WithNullableAnnotation(NullableAnnotation.NotAnnotated), elem))
            return null;

        return $"RebindLayerAliasCollection({member.Name}, previousLayers, replacementLayers, nameof({member.Name}));";
    }

    /// <summary>Returns the element type for a supported layer collection shape.</summary>
    private static ITypeSymbol? LayerCollectionElementType(ITypeSymbol type)
    {
        if (type is IArrayTypeSymbol array) return array.ElementType;
        if (type is not INamedTypeSymbol named || named.TypeArguments.Length != 1) return null;

        var open = named.OriginalDefinition.ToDisplayString();
        return open.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal)
            || open.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal)
            || open.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal)
            || open.StartsWith("System.Collections.Generic.ICollection<", System.StringComparison.Ordinal)
            || open.StartsWith("System.Collections.Generic.IReadOnlyCollection<", System.StringComparison.Ordinal)
            || open.StartsWith("System.Collections.Generic.IEnumerable<", System.StringComparison.Ordinal)
            ? named.TypeArguments[0]
            : null;
    }

    /// <summary>ILayer&lt;T&gt; or a LayerBase&lt;T&gt; subclass over the model's element type.</summary>
    private static bool IsLayerOf(ITypeSymbol type, string elem)
    {
        if (type is not INamedTypeSymbol named) return false;
        var open = named.OriginalDefinition.ToDisplayString();
        if (open.StartsWith("AiDotNet.Interfaces.ILayer<", System.StringComparison.Ordinal))
            return named.TypeArguments.Length == 1 && named.TypeArguments[0].ToDisplayString() == elem;
        for (var c = named; c is not null; c = c.BaseType)
        {
            if (c.OriginalDefinition.ToDisplayString()
                 .StartsWith("AiDotNet.NeuralNetworks.Layers.LayerBase<", System.StringComparison.Ordinal))
            {
                return c.TypeArguments.Length == 1 && c.TypeArguments[0].ToDisplayString() == elem;
            }
        }
        return false;
    }

    private static bool HasAttr2(ISymbol s, INamedTypeSymbol? attr) =>
        attr is not null && s.GetAttributes()
            .Any(a => SymbolEqualityComparer.Default.Equals(a.AttributeClass, attr));

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
                if (ms.Name == RegisterCall && ms.Parameters.Length >= 1 &&
                    ms.Parameters.Skip(1).All(parameter => parameter.IsOptional)) call = true;
                else if (ms.Name == RegisterHook && ms.Parameters.Length == 0 &&
                         (ms.IsVirtual || ms.IsOverride || ms.IsAbstract)) hook = true;
            }
            if (call && hook) return true;
        }
        return false;
    }

    private static bool DeclaresOwn(INamedTypeSymbol type, string name) =>
        type.GetMembers(name).OfType<IMethodSymbol>().Any(m => m.Parameters.Length == 0);

    private static bool DeclaresLayerAliasRebinding(INamedTypeSymbol type) =>
        type.GetMembers(RebindLayerAliasesHook).OfType<IMethodSymbol>()
            .Any(method => method.Parameters.Length == 2);

    /// <summary>
    /// The numeric element type. Conventionally the parameter named <c>T</c>: models in this
    /// library are <c>Foo&lt;T&gt;</c> or descend from <c>ModelBase&lt;T, TInput, TOutput&gt;</c>,
    /// where the second and third are the input and output shapes rather than the scalar type.
    /// </summary>
    private static string? ElementTypeParam(INamedTypeSymbol type)
    {
        // Preserve the fallback before walking the containing-type chain. Keeping the
        // dereference after that nullable traversal confuses null-flow analysis and is
        // unnecessary: both fallbacks belong to the original model type, never to a
        // containing type visited by the traversal.
        string? firstDeclaredTypeParameter = type.TypeParameters.Length > 0
            ? type.TypeParameters[0].Name
            : null;
        INamedTypeSymbol? firstBaseType = type.BaseType;

        for (var c = type; c is not null; c = c.ContainingType)
        {
            foreach (var tp in c.TypeParameters)
            {
                if (tp.Name == "T") return tp.Name;
            }
        }
        if (firstDeclaredTypeParameter is not null) return firstDeclaredTypeParameter;

        // A model that CLOSES over a concrete numeric type has no type parameter to find, and
        // returning null here skipped it from automation entirely and silently -- LinearVectorModel
        // is `ModelBase<double, Matrix<double>, Vector<double>>` and got nothing at all. The element
        // type is still perfectly well known: it is the base's first type ARGUMENT. Reading it there
        // means a model is automated whether it is generic over its scalar or fixed to one, which is
        // a property future models should not have to know about.
        for (var b = firstBaseType; b is not null; b = b.BaseType)
        {
            var open = b.OriginalDefinition.ToDisplayString();
            if ((open.StartsWith("AiDotNet.Models.ModelBase<", System.StringComparison.Ordinal)
                 || open.StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
                && b.TypeArguments.Length > 0)
            {
                var arg = b.TypeArguments[0];
                // Only a CLOSED type is usable: an unsubstituted parameter is already handled above,
                // and emitting its name here would bind to a parameter this class does not declare.
                if (arg.TypeKind != TypeKind.TypeParameter) return arg.ToDisplayString();
            }
        }
        return null;
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

    private static string? SourceExpressionFor(
        ISymbol member,
        string elem,
        bool allowPrimitive = false,
        bool allowDeferredVectorReplacement = false,
        bool allowSerializedObject = false)
    {
        var type = MemberType(member);
        if (type is null) return null;
        string name = member.Name;
        var scalar = SourceFor(type, elem);
        if (scalar is not null)
        {
            if (allowDeferredVectorReplacement
                && scalar == "TensorFieldParameterSource"
                && CanAssign(member))
            {
                return $"new ResizableTensorFieldParameterSource<{elem}>(() => {name}, value => {name} = value)";
            }
            if (allowDeferredVectorReplacement
                && scalar == "VectorFieldWriteThroughSource"
                && CanAssign(member))
            {
                return $"new VectorFieldParameterSource<{elem}>(() => {name}, value => {name} = value)";
            }
            return $"new {scalar}<{elem}>(() => {name})";
        }

        if (allowPrimitive)
        {
            var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
            if (bare.ToDisplayString() == elem && CanAssign(member))
                return $"new ScalarParameterSource<{elem}>(() => {name}, value => {name} = value)";
            if (bare.SpecialType == SpecialType.System_Double)
            {
                bool writable = member switch
                {
                    IFieldSymbol field => !field.IsReadOnly,
                    IPropertySymbol property => property.SetMethod is not null,
                    _ => false
                };
                if (!writable) return null;
                return $"new DoubleScalarParameterSource<{elem}>(() => {name}, value => {name} = value)";
            }
            if (bare is IArrayTypeSymbol array && array.ElementType.SpecialType == SpecialType.System_Double)
                return $"new DoubleArrayParameterSource<{elem}>(() => {name})";
            if (bare is IArrayTypeSymbol outer && outer.ElementType is IArrayTypeSymbol inner &&
                inner.ElementType.SpecialType == SpecialType.System_Double)
                return $"new DoubleJaggedParameterSource<{elem}>(() => {name})";
        }

        var element = CollectionElementType(type);
        var family = element is null ? null : NumericFamilyFor(element, elem);
        if (family is not null)
            return $"new {family}CollectionParameterSource<{elem}>(() => {name})";

        if (DictionaryTypes(type, out var key, out var value))
        {
            family = NumericFamilyFor(value!, elem);
            if (family is not null)
            {
                var keyType = key!.WithNullableAnnotation(NullableAnnotation.NotAnnotated).ToDisplayString();
                return $"new Keyed{family}CollectionParameterSource<{elem}, {keyType}>(() => {name})";
            }
        }

        // A fitted graph can carry learned topology rather than tensor storage (tree ensembles are
        // the canonical example). The semantic declaration is the opt-in: never infer persistence
        // from an arbitrary CLR object, but once the author says [FittedParameter], generate the
        // same count/read/restore contract numeric fields receive. Assignability is required so a
        // fresh instance can accept the deserialized graph.
        if (allowSerializedObject && CanAssign(member))
        {
            string stateType = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated)
                .ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            return $"new SerializedObjectParameterSource<{elem}>(() => {name}, " +
                   $"value => {name} = ({stateType})value!, typeof({stateType}))";
        }
        return null;
    }

    private static bool CanAssign(ISymbol member) => member switch
    {
        IFieldSymbol field => !field.IsReadOnly,
        IPropertySymbol property => property.SetMethod is { IsInitOnly: false },
        _ => false
    };

    private static bool HasFitAvailability(
        ISymbol member,
        ParameterMemberSemanticModel.Kind kind)
    {
        if (kind == ParameterMemberSemanticModel.Kind.Fitted) return true;
        foreach (var attribute in member.GetAttributes())
        {
            if (!ParameterMemberSemanticModel.TryGetKind(attribute, out var declaredKind)
                || declaredKind != kind) continue;
            foreach (var argument in attribute.NamedArguments)
            {
                // ParameterAvailability.Fit is the third enum member (value 2). Comparing the
                // typed constant keeps this generator independent of the runtime assembly.
                if (argument.Key == "Availability" && argument.Value.Value is int value
                    && value == 2)
                    return true;
            }
        }
        return false;
    }

    private static string? TensorAccessorFor(ITypeSymbol type, string name, string elem)
    {
        if (NumericFamilyFor(type, elem) == "Tensor")
            return $"new Tensor<{elem}>?[] {{ {name} }}";

        var element = CollectionElementType(type);
        if (element is not null && NumericFamilyFor(element, elem) == "Tensor")
            return $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull({name})";

        if (DictionaryTypes(type, out _, out var value) &&
            value is not null && NumericFamilyFor(value, elem) == "Tensor")
            return $"global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.OrderedValues({name})";

        return null;
    }

    private static string? NumericFamilyFor(ITypeSymbol type, string elem)
    {
        var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        if (bare is not INamedTypeSymbol named || named.TypeArguments.Length != 1) return null;
        if (named.TypeArguments[0].ToDisplayString() != elem) return null;
        var open = named.OriginalDefinition.ToDisplayString();
        if (open.StartsWith(TensorTypeName + "<", System.StringComparison.Ordinal)) return "Tensor";
        if (open.StartsWith(MatrixTypeName + "<", System.StringComparison.Ordinal)) return "Matrix";
        if (open.StartsWith(VectorTypeName + "<", System.StringComparison.Ordinal)) return "Vector";
        return null;
    }

    private static ITypeSymbol? CollectionElementType(ITypeSymbol type)
    {
        var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        if (bare is IArrayTypeSymbol array) return array.ElementType;
        if (bare is not INamedTypeSymbol named || named.TypeArguments.Length != 1) return null;
        var open = named.OriginalDefinition.ToDisplayString();
        return open.StartsWith("System.Collections.Generic.List<", System.StringComparison.Ordinal) ||
               open.StartsWith("System.Collections.Generic.IList<", System.StringComparison.Ordinal) ||
               open.StartsWith("System.Collections.Generic.IReadOnlyList<", System.StringComparison.Ordinal) ||
               open.StartsWith("System.Collections.Generic.IEnumerable<", System.StringComparison.Ordinal) ||
               open.StartsWith("System.Collections.Generic.IReadOnlyCollection<", System.StringComparison.Ordinal)
            ? named.TypeArguments[0]
            : null;
    }

    private static bool DictionaryTypes(ITypeSymbol type, out ITypeSymbol? key, out ITypeSymbol? value)
    {
        key = null;
        value = null;
        var bare = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        if (bare is not INamedTypeSymbol named || named.TypeArguments.Length != 2) return false;
        var open = named.OriginalDefinition.ToDisplayString();
        if (!open.StartsWith("System.Collections.Generic.Dictionary<", System.StringComparison.Ordinal) &&
            !open.StartsWith("System.Collections.Generic.IDictionary<", System.StringComparison.Ordinal) &&
            !open.StartsWith("System.Collections.Generic.IReadOnlyDictionary<", System.StringComparison.Ordinal))
            return false;
        key = named.TypeArguments[0];
        value = named.TypeArguments[1];
        return true;
    }
    private static bool IsPersistentState(ParameterMemberSemanticModel.Kind kind) => kind is
        ParameterMemberSemanticModel.Kind.Trainable
        or ParameterMemberSemanticModel.Kind.Fitted
        or ParameterMemberSemanticModel.Kind.Frozen
        or ParameterMemberSemanticModel.Kind.Buffer;

    private static bool IsNonOptimizerPersistentState(ParameterMemberSemanticModel.Kind kind) => kind is
        ParameterMemberSemanticModel.Kind.Fitted
        or ParameterMemberSemanticModel.Kind.Frozen
        or ParameterMemberSemanticModel.Kind.Buffer;

    private static string RoleExpression(ParameterMemberSemanticModel.Kind kind) => kind switch
    {
        ParameterMemberSemanticModel.Kind.Fitted =>
            "global::AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState",
        ParameterMemberSemanticModel.Kind.Frozen =>
            "global::AiDotNet.Models.Parameters.ParameterSlotRole.Frozen",
        ParameterMemberSemanticModel.Kind.Buffer =>
            "global::AiDotNet.Models.Parameters.ParameterSlotRole.Buffer",
        _ => "global::AiDotNet.Models.Parameters.ParameterSlotRole.Trainable"
    };

    private static string AvailabilityExpression(
        ISymbol member,
        ParameterMemberSemanticModel.Kind kind)
    {
        foreach (var attribute in member.GetAttributes())
        {
            if (!ParameterMemberSemanticModel.TryGetKind(attribute, out var declaredKind)
                || declaredKind != kind) continue;
            foreach (var argument in attribute.NamedArguments)
            {
                if (argument.Key == "Optional" && argument.Value.Value is bool optional && optional)
                    return "global::AiDotNet.Models.Parameters.ParameterAvailability.Conditional";
                if (argument.Key == "Availability" && argument.Value.Value is int value)
                    return $"(global::AiDotNet.Models.Parameters.ParameterAvailability){value}";
            }
        }

        return kind == ParameterMemberSemanticModel.Kind.Fitted
            ? "global::AiDotNet.Models.Parameters.ParameterAvailability.Fit"
            : "global::AiDotNet.Models.Parameters.ParameterAvailability.Construction";
    }

    private static string GenerateSource(INamedTypeSymbol classSymbol, string elem,
                                         List<(string Name, string SourceExpression, string Role, string Availability)> fields,
                                         List<(string Name, string SourceExpression, string Role, string Availability)> components)
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

        sb.AppendLine($"partial class {classSymbol.Name}{typeParams} : global::AiDotNet.Models.Parameters.IGeneratedParameterRegistrar<{elem}>");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>");
        sb.AppendLine("    /// Auto-generated stable-ID registration for this model's weight-bearing members.");
        sb.AppendLine("    /// </summary>");
        sb.AppendLine($"    void global::AiDotNet.Models.Parameters.IGeneratedParameterRegistrar<{elem}>.RegisterGeneratedParameters(");
        sb.AppendLine($"        global::AiDotNet.Models.Parameters.ParameterComponentRegistry<{elem}> registry)");
        sb.AppendLine("    {");
        sb.AppendLine("        RegisterGeneratedParameterComponents(registry);");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Composes this type's generated parameter fields with inherited fields.</summary>");
        sb.AppendLine("    protected override void RegisterGeneratedParameterComponents(");
        sb.AppendLine($"        global::AiDotNet.Models.Parameters.ParameterComponentRegistry<{elem}> registry)");
        sb.AppendLine("    {");
        sb.AppendLine("        base.RegisterGeneratedParameterComponents(registry);");
        string ownerId = classSymbol.ToDisplayString().Replace("\\", "\\\\").Replace("\"", "\\\"");
        foreach (var f in fields.OrderBy(item => item.Name, System.StringComparer.Ordinal))
        {
            sb.AppendLine($"        registry.Register(\"{ownerId}::{f.Name}\", {f.SourceExpression}, {f.Role}, {f.Availability});");
        }
        foreach (var c in components.OrderBy(item => item.Name, System.StringComparer.Ordinal))
        {
            sb.AppendLine($"        registry.Register(\"{ownerId}::{c.Name}\", {c.SourceExpression}, {c.Role}, {c.Availability});");
        }
        sb.AppendLine("    }");
        sb.AppendLine("}");

        for (int i = 0; i < containing.Count; i++) sb.AppendLine("}");

        return sb.ToString();
    }

}
