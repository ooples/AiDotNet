using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Emits each model's state declarations into the model, so a model author writes none.
/// </summary>
/// <remarks>
/// <para>
/// A hand-written <c>RegisterState</c> that lists every field is the same boilerplate as a
/// hand-written <c>Serialize</c> wearing a different hat: it is common behaviour living in the model,
/// and it is one more place to forget the field somebody adds next year. The generator already knows
/// the type's members and already classifies them, so it can write both halves and the author can
/// write nothing at all.
/// </para>
/// <para>
/// Reuses <see cref="ParameterMemberSemanticModel"/> rather than inventing a second opinion about
/// what a member is. Its vocabulary already answers the question this generator asks:
/// <c>Trainable</c> is in the parameter vector and must NOT be written twice; <c>Fitted</c>,
/// <c>Frozen</c> and <c>Buffer</c> are learned state that the vector does not carry; <c>Scratch</c>
/// is recomputable; <c>Alias</c> is a view of something else; <c>External</c> belongs to another
/// runtime. Unclassified numeric state is already an error under AIDN088, which is what makes
/// "persist everything I can place" safe -- there is nothing it cannot place and silently skip.
/// </para>
/// <para>
/// Emits into a <c>partial</c> declaration, the same way the parameter generator does, because that
/// is the only way generated code can reach a private field. AIDN085 already establishes that
/// convention for weights and 1099 types in this library are already partial.
/// </para>
/// </remarks>
[Generator]
public class ModelStateGenerator : IIncrementalGenerator
{
    /// <summary>A model owns state the generator can persist, but is not partial.</summary>
    private static readonly DiagnosticDescriptor MustBePartial = new(
        "ADN0061",
        "Model must be partial for its state to be persisted automatically",
        "'{0}' owns state that is not in its parameter vector ({1}) but is not declared 'partial', so "
            + "the state generator cannot reach it and nothing persists it. Add 'partial' and the "
            + "declarations are written for you",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <summary>State was annotated, and the registry has no way to carry its type.</summary>
    private static readonly DiagnosticDescriptor UnsupportedStateShape = new(
        "ADN0062",
        "Declared state has a shape the registry cannot carry",
        "'{0}.{1}' is annotated as state but its type '{2}' has no ModelStateRegistry declaration, so "
            + "nothing would persist it. Add an overload for that shape, or hold the value in one the "
            + "registry already carries",
        "AiDotNet.Serialization",
        DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    /// <inheritdoc/>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax { BaseList: not null },
                transform: static (ctx, _) =>
                    ctx.SemanticModel.GetDeclaredSymbol((ClassDeclarationSyntax)ctx.Node) as INamedTypeSymbol)
            .Where(static symbol => symbol is not null);

        context.RegisterSourceOutput(candidates, static (spc, symbol) => Emit(spc, symbol));
    }

    private static void Emit(SourceProductionContext spc, INamedTypeSymbol? type)
    {
        // A semantic model can legitimately return no declared symbol while the user's
        // compilation is incomplete. Treat that as no candidate; never assert it away with !.
        if (type is null) return;

        // ABSTRACT BASES ARE INCLUDED. Skipping them meant state declared on a shared base was never
        // generated for anyone: every decision-tree model keeps its structure in
        // DecisionTreeRegressionBase.Root, and no concrete model declares it, so nothing persisted it
        // and each model wrote its own Serialize to walk the tree by hand. An abstract class can carry
        // an override perfectly well, and RegisterGeneratedState chains through base calls, so putting
        // the declaration where the member actually lives is both correct and the only place it can go.
        if (type.IsStatic) return;

        // The numeric type comes from the base hook's own signature rather than from a guess about
        // which type parameter is "the" numeric one.
        var hook = FindHook(type);
        if (hook is null) return;
        if (hook.Parameters.Length != 1) return;
        if (hook.Parameters[0].Type is not INamedTypeSymbol { TypeArguments.Length: 1 } registry) return;

        var numeric = registry.TypeArguments[0].ToDisplayString();
        bool persistsParametersSeparately = PersistsParametersSeparately(type);
        bool onNeuralNetworkTrunk = InheritsNeuralNetworkBase(type);
        bool emitSerializationSurface = NeedsGeneratedSerializationSurface(type);

        // The type that DECLARES the hook cannot also override it. It gets a Core method instead,
        // which its own hand-written hook calls -- so the class holding the state finally gets
        // declarations generated for it without disturbing the override chain below it.
        var declaresHook = SymbolEqualityComparer.Default.Equals(hook.ContainingType, type);

        var members = new List<(string Name, string Call)>();
        bool hasExplicitState = false;
        bool hasScratchState = false;
        var registrations = ParameterMemberSemanticModel.GetRegistrationClassifications(type);

        foreach (var member in type.GetMembers())
        {
            if (member.IsStatic || member.IsImplicitlyDeclared) continue;

            var memberType = member switch
            {
                IFieldSymbol f when !f.IsConst => f.Type,
                IPropertySymbol { IsIndexer: false } p when p.GetMethod is not null && p.SetMethod is not null => p.Type,
                _ => null,
            };
            if (memberType is null) continue;

            // NeuralNetworkBase already owns its canonical layer graph and serializes each layer's
            // layout, parameters and buffers. ModelParameterGenerator also emits the alias rebinding
            // and extra-layer traversal used by clone/parameter operations. Declaring those same
            // layer fields here gives them a SECOND persistence owner: canonical aliases are restored
            // twice, large stacks are duplicated into the state envelope, and a destination whose
            // runtime layout differs can receive a flat vector meant for the old alias. RWKV exposed
            // this as an exact 21,120-vs-21,440 parameter mismatch; large diffusion models paid for
            // the duplicate graph with shard-wide timeouts.
            //
            // This exclusion is specific to the neural-network trunk. A sibling model base that owns
            // a plain list of layers has no canonical graph serializer, so DeclareLayerList remains
            // its generated persistence mechanism.
            if (onNeuralNetworkTrunk
                && (IsLayer(memberType) || IsLayerCollection(memberType)))
            {
                continue;
            }

            // These booleans are lazy-registration LATCHES owned by the framework plumbing, not
            // model state. Restoring `_componentsRegistered = true` into a fresh instance leaves its
            // new registry empty while preventing the registration callback from ever running. The
            // same pattern exists on every model-base trunk, so exclude it once here rather than
            // requiring every base and every future model family to annotate identical machinery.
            if (IsRegistryLifecycleLatch(member)) continue;

            // Readonly storage cannot be REASSIGNED on restore, so declaring it would produce a
            // payload nothing could apply -- true of a vector or a matrix, and false of anything
            // restored IN PLACE. DeclareChild already fills a readonly child by calling Deserialize
            // on the instance the constructor built, and DeclareOptions does the same for settings.
            //
            // Excluding on mutability alone hid a real defect: KNearestNeighborsRegression holds
            // `private readonly KNearestNeighborsOptions _options` and answers with _options.K, and
            // the field was dropped here before anything could ask what it was -- so the payload
            // carried the training data, not the K, and the model restored and answered differently.
            // A LIST OF LAYERS belongs with them: DeclareLayerList restores each layer through its own
            // Deserialize, on the instance the constructor built, so the list reference is never
            // reassigned and readonly is no obstacle. Excluding it dropped DeepANT's `private readonly
            // List<ConvLayerTensor<T>> _convLayers` before the type was ever consulted, which is the
            // same shape as the KNearestNeighbors defect above: the payload carried everything except
            // the part that decides the answer.
            // Imperative parameter-component registration is a state-ownership declaration too.
            // Treating it as unclassified makes the declared-state envelope serialize the same
            // child a second time after the parameter/clone path has already restored it. That is
            // both redundant and unsafe for a materialized lazy child whose constructor clone owns
            // the exact runtime layout.
            var classification = ParameterMemberSemanticModel.ClassifyWithRegistrations(
                member, registrations);
            hasScratchState |= classification.Kind == ParameterMemberSemanticModel.Kind.Scratch;
            if (member is IFieldSymbol { IsReadOnly: true }
                && !IsModelOptions(memberType)
                && !IsSerializableModel(memberType)
                && !IsSerializableModelList(memberType)
                && !IsObjectCollection(memberType)
                && !IsLayerList(memberType)
                && !CanRestoreReadonlyNumericArray(memberType))
            {
                continue;
            }

            // ONE OWNER PER PIECE OF STATE. A model that still hand-writes its serialization already
            // carries its layers, so declaring them too would write the same state twice and restore it
            // twice. The two halves cannot be assumed to agree: a hand-written DeserializeCore
            // typically rebuilds its layers through a placeholder constructor, so a declared restore
            // landing on those same layers would be applying trained values to whatever shape the
            // placeholder happened to have.
            //
            // Skipping here makes the migration INCREMENTAL rather than a flag day: deleting a model's
            // hand-written pair is the single act that switches it onto declared state, with no other
            // edit and no window in which both mechanisms own the same fields. ADN0060 is what makes
            // that deletion happen; this is what makes it safe.
            if (IsLayerList(memberType) && DeclaresHandWrittenSerialization(type))
            {
                continue;
            }

            // OPT-OUT, NOT OPT-IN, and this is the whole reason 330 hand-written Serialize/Deserialize
            // pairs exist. Requiring [Fitted], [Frozen] or [Buffer] before a member is persisted means
            // an author who adds a field and annotates nothing gets a model that serialises
            // incompletely and no error -- so the only way to be sure was to write the pair by hand,
            // which is two places to forget the same field instead of one. Storage is now persisted by
            // DEFAULT and a member has to say why it should not be.
            //
            // The four exclusions are the ones that would be wrong to persist, not the ones nobody
            // annotated:
            //   Trainable  already in the parameter vector; writing it again would restore it twice
            //   Scratch    a work buffer whose value between calls means nothing
            //   Alias      another name for a member already carried
            //   External   not this model's to save
            // Conflicting is excluded too, because a member carrying contradictory annotations is a
            // question for AIDN089 to answer rather than something to guess at here.
            // ModelBase and NeuralNetworkBase have a separate generated parameter registry, so
            // trainable storage on those trunks must not be written twice. Their legacy sibling
            // bases do not: their ordinary payload knows only the base fields. On those trunks the
            // declared-state envelope is the generated persistence mechanism for trainable storage
            // too. Treating every trunk as if it owned a parameter registry dropped the learned
            // coefficients from GAMLSS and ZeroInflatedRegression while their clones appeared to
            // deserialize successfully.
            bool carryTrainableAsState = classification.Kind == ParameterMemberSemanticModel.Kind.Trainable
                && !persistsParametersSeparately;
            bool carryNativePrecisionShadow =
                (classification.Kind is ParameterMemberSemanticModel.Kind.Trainable
                    or ParameterMemberSemanticModel.Kind.Fitted
                    or ParameterMemberSemanticModel.Kind.Frozen
                    or ParameterMemberSemanticModel.Kind.Buffer)
                && persistsParametersSeparately
                && RequiresNativePrecisionShadow(memberType, numeric);
            if ((classification.Kind == ParameterMemberSemanticModel.Kind.Trainable
                    && persistsParametersSeparately
                    && !carryNativePrecisionShadow)
                || classification.Kind is ParameterMemberSemanticModel.Kind.Scratch
                or ParameterMemberSemanticModel.Kind.Alias
                or ParameterMemberSemanticModel.Kind.External
                or ParameterMemberSemanticModel.Kind.Conflicting)
            {
                continue;
            }

            // Whether somebody ASKED for this member to be state, as opposed to it being swept in by
            // the default. It decides how loudly an unsupported shape is reported, below.
            var annotated = classification.Kind is ParameterMemberSemanticModel.Kind.Fitted
                or ParameterMemberSemanticModel.Kind.Frozen
                or ParameterMemberSemanticModel.Kind.Buffer
                || carryTrainableAsState
                || carryNativePrecisionShadow;

            // Keyed by DECLARING TYPE and member, not by member alone. A name is unique within one
            // class and nothing more: VectorAutoRegressionModel and VARMAModel each keep a private
            // Matrix<T> _residuals, which is ordinary C# and means the derived model's generated
            // registration met the base's under the same key and threw "State '_residuals' is
            // already declared". Every model with a field that shares a name with one further up its
            // own hierarchy had the same fault waiting in it.
            var call = DeclareCall(member.Name, $"{type.Name}.{member.Name}", memberType, numeric, annotated,
                nullableTarget: memberType.NullableAnnotation == NullableAnnotation.Annotated
                    || memberType.IsValueType,
                restoreInPlace: member is IFieldSymbol { IsReadOnly: true },
                exactPrecisionShadow: carryNativePrecisionShadow,
                childFactory: ChildFactoryExpression(type, memberType));

            if (call is null)
            {
                // A member the default swept in whose type the registry cannot carry is passed over in
                // silence, because that is exactly what happened to it before persistence became the
                // default -- reporting it would turn "no change" into hundreds of new build errors
                // about members nobody claimed were state. An ANNOTATED member is the opposite case and
                // is still reported below.
                if (!annotated) continue;

                // LOUD, NOT SILENT. This member was CLASSIFIED as state -- somebody annotated it --
                // and the generator cannot express its type. Skipping quietly would drop annotated
                // state from the payload and produce a model that restores almost everything, which
                // is the exact failure this work exists to remove. GeneralizedAdditiveModel proved
                // it: its List<Vector<T>> knots were annotated, silently skipped, and Predict then
                // refused with "loaded without its fitted knot vectors".
                spc.ReportDiagnostic(Diagnostic.Create(
                    UnsupportedStateShape,
                    member.Locations.FirstOrDefault() ?? type.Locations.FirstOrDefault(),
                    type.Name,
                    member.Name,
                    memberType.ToDisplayString()));
                continue;
            }

            members.Add((member.Name, call));
            hasExplicitState |= annotated;
        }

        var fittedInfrastructureRepairs = FittedInfrastructureRepairs(type);
        if (fittedInfrastructureRepairs.Count > 0)
        {
            var repair = new StringBuilder();
            repair.Append($"state.DeclareAfterRestore(\"{type.Name}.$fittedInfrastructure\", () => {{ if (IsFitted) {{ ");
            foreach (var (field, expression) in fittedInfrastructureRepairs)
            {
                repair.Append($"if ({field} is null) {field} = {expression}; ");
            }
            repair.Append("} });");
            members.Add(("$fittedInfrastructure", repair.ToString()));
        }

        // Scratch caches are deliberately absent from persisted state, but a constructor can fill
        // them from its fresh parameters before a restore replaces the authoritative fields. A
        // conventional zero-argument Refresh*Cache(s) method is an existing declaration of how to
        // rebuild those derived values. Register it after the parameter phase so clone/checkpoint
        // restoration cannot leave the cache describing the constructor's discarded weights.
        if (hasScratchState)
        {
            foreach (var refresh in type.GetMembers().OfType<IMethodSymbol>()
                         .Where(method => !method.IsStatic
                                          && method.Parameters.Length == 0
                                          && method.TypeParameters.Length == 0
                                          && method.ReturnsVoid
                                          && method.Name.StartsWith("Refresh", System.StringComparison.Ordinal)
                                          && (method.Name.EndsWith("Cache", System.StringComparison.Ordinal)
                                              || method.Name.EndsWith("Caches", System.StringComparison.Ordinal)))
                         .OrderBy(method => method.Name, System.StringComparer.Ordinal))
            {
                string name = $"{type.Name}.$derivedCache.{refresh.Name}";
                members.Add(($"$derivedCache.{refresh.Name}",
                    $"state.DeclareAfterParameterRestore(\"{name}\", {refresh.Name});"));
            }
        }

        // Canonical layer replacement and declared-state restore can leave a model's derived views
        // describing its constructor graph. A conventional zero-argument Rebind* method is the
        // model's existing declaration of how to repair those views. Run it only after parameters
        // and generated aliases are final; invoking it during the structural phase would merely
        // rebind the constructor objects that the base is about to replace.
        foreach (var rebind in type.GetMembers().OfType<IMethodSymbol>()
                     .Where(method => !method.IsStatic
                                      && method.Parameters.Length == 0
                                      && method.TypeParameters.Length == 0
                                      && method.ReturnsVoid
                                      && method.Name.StartsWith("Rebind", System.StringComparison.Ordinal))
                     .OrderBy(method => method.Name, System.StringComparer.Ordinal))
        {
            string name = $"{type.Name}.$derivedRebind.{rebind.Name}";
            members.Add(($"$derivedRebind.{rebind.Name}",
                $"state.DeclareAfterParameterRestore(\"{name}\", {rebind.Name});"));
        }

        // A declaring type ALWAYS gets its Core method, even empty: its hand-written hook calls it
        // unconditionally, so omitting it would leave that call with no target.
        if (members.Count == 0 && !declaresHook && !emitSerializationSurface) return;

        // The type AND everything containing it. A nested partial can only be reopened inside partial
        // outers, so reporting only the inner one would name a fix that does not compile on its own.
        for (var scope = type; scope is not null; scope = scope.ContainingType)
        {
            if (IsPartial(scope)) continue;

            // The opt-out sweep is an automation benefit for types already participating in source
            // generation, not a flag-day migration for every legacy type in the assembly. An
            // explicitly annotated state member must still fail loudly when generation is
            // impossible; an unannotated collection on a non-partial legacy type keeps its previous
            // behavior until that type opts in by becoming partial.
            if (!hasExplicitState) return;

            spc.ReportDiagnostic(Diagnostic.Create(
                MustBePartial,
                scope.Locations.FirstOrDefault(),
                scope.Name,
                string.Join(", ", members.Select(m => m.Name))));
            return;
        }

        spc.AddSource($"{type.ToDisplayString().Replace('<', '_').Replace('>', '_').Replace(',', '_')}.State.g.cs",
            Render(type, numeric, members, declaresHook, emitSerializationSurface));
    }

    /// <summary>Whether a field is one of the framework's lazy registry initialization latches.</summary>
    private static bool IsRegistryLifecycleLatch(ISymbol member)
        => member is IFieldSymbol { Type.SpecialType: SpecialType.System_Boolean }
           && member.Name is "_componentsRegistered" or "_declaredStateRegistered" or "_stateRegistered";

    /// <summary>Whether this hierarchy persists trainable storage outside declared model state.</summary>
    private static bool PersistsParametersSeparately(INamedTypeSymbol type)
    {
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (current.GetMembers("RegisterGeneratedParameterComponents")
                .OfType<IMethodSymbol>()
                .Any(method => method.Parameters.Length == 1))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Whether the type inherits the canonical neural-network graph owner.</summary>
    private static bool InheritsNeuralNetworkBase(INamedTypeSymbol type)
    {
        for (var current = type; current is not null; current = current.BaseType)
        {
            if (current.OriginalDefinition.ToDisplayString()
                .StartsWith("AiDotNet.NeuralNetworks.NeuralNetworkBase<", System.StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Whether an abstract public serializer can be implemented by delegating to common protected
    /// base helpers. The capability is detected structurally, so the generator does not know or care
    /// which model-family base supplies it.
    /// </summary>
    private static bool NeedsGeneratedSerializationSurface(INamedTypeSymbol type)
    {
        if (type.IsAbstract) return false;
        if (type.GetMembers().OfType<IMethodSymbol>().Any(method =>
            method.Name is "Serialize" or "Deserialize"))
        {
            return false;
        }

        bool hasSerializeHelper = false;
        bool hasDeserializeHelper = false;
        bool hasAbstractSerialize = false;
        bool hasAbstractDeserialize = false;

        for (var current = type.BaseType; current is not null; current = current.BaseType)
        {
            foreach (var method in current.GetMembers().OfType<IMethodSymbol>())
            {
                hasSerializeHelper |= method.Name == "SerializeGeneratedModelState"
                    && method.Parameters.Length == 0
                    && method.ReturnType is IArrayTypeSymbol
                    {
                        ElementType.SpecialType: SpecialType.System_Byte,
                    };
                hasDeserializeHelper |= method.Name == "DeserializeGeneratedModelState"
                    && method.Parameters.Length == 1
                    && method.Parameters[0].Type is IArrayTypeSymbol
                    {
                        ElementType.SpecialType: SpecialType.System_Byte,
                    };
                hasAbstractSerialize |= method.Name == "Serialize"
                    && method.IsAbstract
                    && method.Parameters.Length == 0;
                hasAbstractDeserialize |= method.Name == "Deserialize"
                    && method.IsAbstract
                    && method.Parameters.Length == 1
                    && method.Parameters[0].Type is IArrayTypeSymbol
                    {
                        ElementType.SpecialType: SpecialType.System_Byte,
                    };
            }
        }

        return hasSerializeHelper && hasDeserializeHelper
            && hasAbstractSerialize && hasAbstractDeserialize;
    }

    /// <summary>True when the member is itself something that can serialize its own state.</summary>
    private static bool IsSerializableModel(ITypeSymbol type)
    {
        if (type.AllInterfaces.Any(i => i.Name is "IModelSerializer" or "IFullModel" or "INeuralNetwork")
            || type.Name is "IModelSerializer" or "IFullModel" or "INeuralNetwork")
        {
            return true;
        }

        return false;
    }

    /// <summary>Whether the type still persists state by hand, and so already owns its layers.</summary>
    /// <remarks>
    /// Checks the type's OWN members, not inherited ones: an inherited hook is the base doing the work,
    /// which is exactly the state this asks about being declared rather than hand-written.
    /// </remarks>
    private static bool DeclaresHandWrittenSerialization(INamedTypeSymbol type)
        => type.GetMembers().Any(m => m is IMethodSymbol
        {
            Name: "SerializeCore" or "DeserializeCore"
                or "SerializeModelSpecificData" or "DeserializeModelSpecificData"
                or "SerializeNetworkSpecificData" or "DeserializeNetworkSpecificData",
        });

    /// <summary>Whether a type is a <c>List</c> of layers, which restores in place.</summary>
    private static bool IsLayerList(ITypeSymbol type)
        => type is INamedTypeSymbol { Name: "List", TypeArguments.Length: 1 } list
           && IsLayer(list.TypeArguments[0]);

    /// <summary>Whether a supported collection carries layers.</summary>
    private static bool IsLayerCollection(ITypeSymbol type)
    {
        if (type is IArrayTypeSymbol array) return IsLayer(array.ElementType);
        if (type is not INamedTypeSymbol { TypeArguments.Length: 1 } collection) return false;

        if (collection.Name is not ("List" or "IList" or "IReadOnlyList" or "IEnumerable"
            or "ICollection" or "IReadOnlyCollection"))
        {
            return false;
        }

        return IsLayer(collection.TypeArguments[0]);
    }

    /// <summary>Whether a list carries nested models through their own serialization contract.</summary>
    private static bool IsSerializableModelList(ITypeSymbol type)
        => type is INamedTypeSymbol { Name: "List", TypeArguments.Length: 1 } list
           && IsSerializableModel(list.TypeArguments[0]);

    /// <summary>Whether a collection can be cleared and refilled through its readonly reference.</summary>
    private static bool IsObjectCollection(ITypeSymbol type)
        => type is INamedTypeSymbol { TypeArguments.Length: > 0 } named
           && named.Name is "List" or "Dictionary";

    /// <summary>
    /// Numeric arrays own mutable contents even when their field reference is readonly. The state
    /// registry has explicit in-place readers for these shapes, so readonly is not a reason to drop
    /// them from generated persistence.
    /// </summary>
    private static bool CanRestoreReadonlyNumericArray(ITypeSymbol type)
        => IsDoubleArray(type) || IsJaggedDoubleArray(type);

    /// <summary>
    /// A flat Vector&lt;T&gt; checkpoint cannot preserve a double-backed working value when T is float.
    /// Emit a post-vector precision shadow for the CLR-double shapes the parameter generator owns.
    /// Closed double models need no duplicate because their public vector is already lossless.
    /// </summary>
    private static bool RequiresNativePrecisionShadow(ITypeSymbol type, string numeric)
        => numeric != "double"
           && (type.SpecialType == SpecialType.System_Double
               || IsDoubleArray(type)
               || IsJaggedDoubleArray(type));

    private static bool IsDoubleArray(ITypeSymbol type)
        => type is IArrayTypeSymbol
        {
            Rank: 1,
            ElementType.SpecialType: SpecialType.System_Double
        };

    private static bool IsJaggedDoubleArray(ITypeSymbol type)
        => type is IArrayTypeSymbol
        {
            Rank: 1,
            ElementType: IArrayTypeSymbol
            {
                Rank: 1,
                ElementType.SpecialType: SpecialType.System_Double
            }
        };

    /// <summary>Whether a type is a layer, i.e. derives from LayerBase.</summary>
    /// <remarks>
    /// Tested by walking the base chain rather than by interface, because a layer's identity is its
    /// base class: ILayer is implemented by wrappers and adapters that are not themselves storage, and
    /// DeclareLayerList restores THROUGH LayerBase.Serialize/Deserialize, so the declaration is only
    /// sound for something that actually inherits that pair.
    /// </remarks>
    private static bool IsLayer(ITypeSymbol type)
    {
        for (var t = type as INamedTypeSymbol; t is not null; t = t.BaseType)
        {
            if (t.Name == "LayerBase") return true;
        }

        return false;
    }

    /// <summary>Finds the inherited hook this generator overrides, and with it the numeric type.</summary>
    private static IMethodSymbol? FindHook(INamedTypeSymbol type)
    {
        // Starts at the TYPE so a class that declares the hook is recognised as the root of its
        // hierarchy. Safe because every hook is HAND-WRITTEN and therefore visible here; a marker
        // that lived only in generated source would be invisible to the generator, which is why
        // deleting the hooks made every derived type look like a root and broke the chain.
        for (var current = type; current is not null; current = current.BaseType)
        {
            var hook = current.GetMembers("RegisterGeneratedState").OfType<IMethodSymbol>().FirstOrDefault();
            if (hook is not null) return hook;
        }

        return null;
    }

    /// <summary>Maps a member's type onto the registry call that persists it.</summary>
    /// <remarks>
    /// Returns null for a shape the registry cannot express. That is deliberately silent HERE and
    /// loud elsewhere: AIDN088 already refuses to let numeric state go unclassified, so a member that
    /// reaches this point and has no mapping is a container the registry has not learned yet, and the
    /// model keeps its own declaration until it does.
    /// </remarks>
    private static string? DeclareCall(
        string name,
        string id,
        ITypeSymbol memberType,
        string numeric,
        bool annotated,
        bool nullableTarget,
        bool restoreInPlace,
        bool exactPrecisionShadow,
        string? childFactory)
    {
        // A nullable value type has a state the registry cannot express: "not set" is not a number,
        // and the getter cannot hand an int? to something expecting an int. MOMENT proved it -- the
        // display string had its '?' trimmed before matching, so an int? matched the int case and the
        // generated lambda would not compile. Declining is the honest answer; inventing a zero for it
        // would silently turn "never configured" into "configured to zero" on every round trip.
        if (memberType is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T })
        {
            return null;
        }

        // A NULLABLE `T?` IS THE SAME SITUATION. DeclareScalar takes Func<T>, and no Func<T?> overload
        // can sit beside it because for an unconstrained T the two differ only by nullability. A getter
        // for a `T?` member therefore returns a possible null into a non-null contract, and the registry
        // has no way to say "not set" for it. Declining is the same honest answer given to Nullable<T>
        // above: ClusteringBase.Inertia and NeuralNetworkBase.LastLoss are exactly this shape and
        // neither was persisted before.
        if (memberType is ITypeParameterSymbol
            && memberType.NullableAnnotation == NullableAnnotation.Annotated)
        {
            return null;
        }

        // Namespaces stripped before matching. The display string is fully qualified, so
        // List<AiDotNet.Tensors.LinearAlgebra.Vector<T>> does not end with "List<Vector<T>>" and a
        // naive suffix test silently declined it -- which is how the knots got dropped.
        var display = memberType.ToDisplayString().TrimEnd('?');
        var key = System.Text.RegularExpressions.Regex
            .Replace(display, @"\b[A-Za-z_][A-Za-z0-9_]*\.", string.Empty)
            .Replace($"<{numeric}>", "<T>");

        // A non-nullable field cannot be handed a null, and the null-forgiving operator is not an
        // option here -- AIDN071 rejects it precisely because it suppresses the question rather than
        // answering it. So a null in the payload leaves the constructed value in place, which is the
        // honest reading: the saving model had nothing there to restore.
        var setter = nullableTarget
            ? $"v => {name} = v"
            : $"v => {{ if (v is not null) {name} = v; }}";
        var getter = $"() => {name}";

        // The public parameter vector is typed as T. A float model whose implementation keeps
        // double working weights therefore cannot make a bit-identical checkpoint through that
        // vector alone. These declarations are a precision shadow, restored in a distinct phase
        // after the vector; they are generated from the storage type and require no model hook.
        if (exactPrecisionShadow)
        {
            return key switch
            {
                "double" => $"state.DeclareExactDouble(\"{id}\", {getter}, {setter});",
                "double[]" when restoreInPlace =>
                    $"state.DeclareExactInPlace(\"{id}\", {getter});",
                "double[]" => $"state.DeclareExact(\"{id}\", {getter}, {setter});",
                "double[][]" when restoreInPlace =>
                    $"state.DeclareExactInPlace(\"{id}\", {getter});",
                "double[][]" => $"state.DeclareExact(\"{id}\", {getter}, {setter});",
                _ => null,
            };
        }

        // Numeric collections have purpose-built binary declarations. A readonly field still owns
        // mutable contents, so select their in-place counterparts before the ordinary switch can
        // emit a setter that cannot compile. Keeping these on the binary path also avoids routing
        // Tensor/Matrix/Vector through JSON, which cannot reconstruct their internal storage.
        if (restoreInPlace)
        {
            var inPlaceNumericCollection = key switch
            {
                "List<Vector<T>>" or "List<Matrix<T>>" or "List<Tensor<T>>"
                    or "Dictionary<string, Vector<T>>" or "Dictionary<int, Vector<T>>" =>
                    $"state.DeclareInPlace(\"{id}\", {getter});",
                "double[]" or "double[][]" =>
                    $"state.DeclareInPlace(\"{id}\", {getter});",
                _ => null,
            };
            if (inPlaceNumericCollection is not null) return inPlaceNumericCollection;
        }

        // A readonly collection owns fitted CONTENTS even though its reference cannot be assigned.
        // Lists of models and layers retain their purpose-built restore paths; every other list or
        // dictionary is reconstructed by the registry and copied into the constructor-created
        // instance. This is what carries known-class tables, nested tree records and per-class
        // statistics without making the model author write a hook.
        if (restoreInPlace && IsSerializableModelList(memberType)
            && memberType is INamedTypeSymbol { TypeArguments.Length: 1 } inPlaceChildList)
        {
            return $"state.DeclareChildList<{inPlaceChildList.TypeArguments[0].ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter});";
        }

        if (restoreInPlace && IsLayerList(memberType)
            && memberType is INamedTypeSymbol { TypeArguments.Length: 1 } inPlaceLayerList)
        {
            return $"state.DeclareLayerList<{inPlaceLayerList.TypeArguments[0].ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter});";
        }

        if (restoreInPlace && IsObjectCollection(memberType)
            && IsGeneratedObjectState(memberType, numeric))
        {
            return $"state.DeclareObjectInPlace(\"{id}\", {getter});";
        }

        return key switch
        {
            // A DECISION TREE, carried whole instead of walked by hand: the shared hand-written
            // walk dropped Threshold and the per-leaf LinearModel.
            var k when k.EndsWith(".DecisionTreeNode<T>") || k == "DecisionTreeNode<T>" =>
                $"state.DeclareTree(\"{id}\", {getter}, {setter});",

            var k when k.EndsWith(".Vector<T>") || k == "Vector<T>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            var k when k.EndsWith(".Matrix<T>") || k == "Matrix<T>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            var k when k.EndsWith(".Tensor<T>") || k == "Tensor<T>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Vector<byte>" => $"state.DeclareByteVector(\"{id}\", {getter}, {setter});",
            "Vector<double>" => $"state.DeclareDoubleVector(\"{id}\", {getter}, {setter});",
            "List<Vector<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "List<Matrix<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "List<Tensor<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Matrix<T>[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Vector<int>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Dictionary<string, Vector<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Dictionary<int, Vector<T>>" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "Vector<T>[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "int[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "double[]" => $"state.Declare(\"{id}\", {getter}, {setter});",
            "T[]" => $"state.DeclareArray(\"{id}\", {getter}, {setter});",
            "int" => $"state.DeclareInt32(\"{id}\", {getter}, {setter});",
            "long" => $"state.DeclareInt64(\"{id}\", {getter}, {setter});",
            "double" => $"state.DeclareDouble(\"{id}\", {getter}, {setter});",
            "bool" => $"state.DeclareBoolean(\"{id}\", {getter}, {setter});",
            "string" => $"state.DeclareString(\"{id}\", {getter}, {setter});",
            "T" => $"state.DeclareScalar(\"{id}\", {getter}, {setter});",

            // A nested model carries its own state through its own Serialize, so the parent only has
            // to say that it is there. Restored IN PLACE, because the parent builds it and what
            // travels is its state rather than its identity.
            // A parameter source keeps its state in a vector rather than a payload.
            // A RECURSIVE NODE GRAPH -- a decision tree. The registry has carried these all along
            // through DeclareGraph; nothing could reach it, because describing a node meant writing the
            // description by hand, which is the boilerplate this work removes rather than relocates.
            // The node type says everything needed: its own properties give the fields, the ones typed
            // as itself give the children, and its parameterless constructor gives Create. Eight
            // tree and ensemble model families hand-wrote a Serialize to walk this structure, and
            // deleting those pairs without this failed 26 tests -- exactly the silent state loss the
            // deleter's own design warns about.
            _ when IsRecursiveNode(memberType, numeric) is { } node => GraphCall(id, name, node, numeric, setter),

            // A fitted forest is the same recursive shape repeated. Its element type describes the
            // walk; the list count and roots are registry concerns. This carries private tree records
            // such as DART's without a model-specific SerializeTree/DeserializeTree pair.
            _ when !restoreInPlace && memberType is INamedTypeSymbol { Name: "List", TypeArguments.Length: 1 } graphList
                   && IsRecursiveNode(graphList.TypeArguments[0], numeric) is { } graphNode =>
                GraphListCall(id, name, graphNode, numeric, setter),

            // A node derived from the library's common DecisionTreeNode has children typed as the
            // base node rather than as its own derived type, so it is not self-recursive in Roslyn's
            // exact-type sense. Generate the predictive base fields plus derived scalar fields and
            // cast the child links back to the concrete node type.
            _ when IsDerivedDecisionTreeNode(memberType) is { } derivedTree =>
                DerivedDecisionTreeGraphCall(id, name, derivedTree, numeric, setter),

            // THE CHILD PATHS STAY OPT-IN even though storage is now opt-out, and the difference is
            // real rather than cautious. Storage is state by its nature -- a Matrix<T> a model holds
            // between calls is something it learned. An object is not: a model also holds its
            // optimizer, its scheduler, its loss function, and none of those are state to restore.
            // Sweeping them in on the default is how three networks came to persist a _trainOptimizer
            // that is never assigned, which the compiler reported as CS0649 and which would have
            // travelled in every payload as a null. So a nested model is carried when somebody says it
            // is state, and otherwise left alone.
            _ when !IsInfrastructure(memberType) && memberType.AllInterfaces.Any(i => i.Name == "IParameterSource")
                   && !IsSerializableModel(memberType) =>
                $"state.DeclareParameterSource(\"{id}\", {getter});",

            _ when (!IsInfrastructure(memberType) || IsFittedSerializer(memberType))
                   && IsSerializableModel(memberType)
                   && nullableTarget && !restoreInPlace =>
                childFactory is null
                    ? $"state.DeclareChild<{memberType.ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter}, {setter});"
                    : $"state.DeclareChild<{memberType.ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter}, {setter}, {childFactory});",

            _ when (!IsInfrastructure(memberType) || IsFittedSerializer(memberType))
                   && IsSerializableModel(memberType) =>
                $"state.DeclareChild<{memberType.ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter});",

            // A list of nested models -- an agent's per-actor target networks, a mixer's per-agent
            // heads. Same rule as a single child: each carries its own state, restored in place.
            // A LIST of models is carried by default, unlike a single one, and the split is not
            // arbitrary. The reason single children stay opt-in is the optimizer a model holds, and an
            // optimizer is held in a field of its own -- nobody keeps a list of them. A list of models
            // is an ensemble's members: a random forest IS its trees, and dropping them leaves a
            // forest that restores with nothing to predict from. RandomForest, DART and
            // ExtremelyRandomizedTrees all failed their round trip on exactly that.
            _ when memberType is INamedTypeSymbol { Name: "List", TypeArguments.Length: 1 } list
                   && IsSerializableModel(list.TypeArguments[0]) =>
                $"state.DeclareChildList<{list.TypeArguments[0].ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter});",

            // A list of LAYERS the model owns directly. Networks never reach this arm -- their layers
            // belong to the network base -- but a model on another base that keeps a conv stack or an
            // encoder stack in a plain List had NO declaration available at all: every other arm wants
            // a vector, a matrix, a tensor or an IModelSerializer, and a layer is none of those. So the
            // member was skipped in silence and the layers' learned values travelled nowhere. DeepANT
            // came back holding the placeholder-shaped convolutions its deserialization constructor
            // builds -- 96 kernel values collapsed to 1 -- and its prediction changed sign across a
            // round trip while every other declared member restored perfectly.
            //
            // Restored in place, like the child list above: the constructor already builds these at
            // their configured widths, so only the learned values need to travel.
            _ when memberType is INamedTypeSymbol { Name: "List", TypeArguments.Length: 1 } layerList
                   && IsLayer(layerList.TypeArguments[0]) =>
                $"state.DeclareLayerList<{layerList.TypeArguments[0].ToDisplayString().TrimEnd('?')}>(\"{id}\", {getter});",

            // THE SETTINGS A MODEL PREDICTS WITH, carried for the same reason a list of children is:
            // they decide the answer. KNearestNeighborsRegression predicts with _options.K, so a
            // payload holding its training data but not its K restored a model that ran and answered
            // differently -- the silent kind of wrong. Only scalar settings travel, which
            // DeclareOptions states and enforces; anything object-shaped is rebuilt by the
            // constructor the clone plan already replays.
            _ when IsModelOptions(memberType) =>
                $"state.DeclareOptions(\"{id}\", {getter});",

            // General fitted object state. The boundary is intentionally structural: arrays,
            // lists, dictionaries and a model's own nested record/node types are state-shaped;
            // arbitrary services are not. Nested IModelSerializer values inside these objects use
            // their canonical byte payload rather than being reduced to public JSON properties.
            _ when !IsInfrastructure(memberType) && IsGeneratedObjectState(memberType, numeric) =>
                $"state.DeclareObject(\"{id}\", {getter}, {setter});",

            _ => null,
        };
    }

    /// <summary>Whether an assignable member has a generated general-object state representation.</summary>
    private static bool IsGeneratedObjectState(ITypeSymbol type, string numeric)
    {
        if (type is not IArrayTypeSymbol
            && !IsObjectCollection(type)
            && type is not INamedTypeSymbol { TypeKind: TypeKind.Class })
        {
            return false;
        }

        return CanCarryObjectState(type, numeric, new HashSet<string>(), depth: 0);
    }

    /// <summary>
    /// Proves the JSON-backed fallback can reconstruct the complete reachable public shape.
    /// </summary>
    /// <remarks>
    /// This is deliberately a proof, not a guess. A broad "every List is JSON" rule captured the
    /// neural-network layer graph, tensor-keyed gradient dictionaries and POCOs containing Matrix,
    /// all of which Json.NET can write but cannot reconstruct. Declining an unproven shape lets its
    /// purpose-built base serialization remain the sole owner instead of adding a broken second copy.
    /// </remarks>
    private static bool CanCarryObjectState(
        ITypeSymbol type,
        string numeric,
        HashSet<string> visiting,
        int depth)
    {
        if (depth > 24) return false;
        if (type.NullableAnnotation == NullableAnnotation.Annotated)
            type = type.WithNullableAnnotation(NullableAnnotation.NotAnnotated);

        if (type is ITypeParameterSymbol) return true;
        if (type.TypeKind == TypeKind.Enum) return true;
        if (type.SpecialType is SpecialType.System_Boolean
            or SpecialType.System_Byte or SpecialType.System_SByte
            or SpecialType.System_Int16 or SpecialType.System_UInt16
            or SpecialType.System_Int32 or SpecialType.System_UInt32
            or SpecialType.System_Int64 or SpecialType.System_UInt64
            or SpecialType.System_Single or SpecialType.System_Double
            or SpecialType.System_Decimal or SpecialType.System_Char
            or SpecialType.System_String)
        {
            return true;
        }

        if (type is IArrayTypeSymbol array)
            return CanCarryObjectState(array.ElementType, numeric, visiting, depth + 1);

        if (type is not INamedTypeSymbol named) return false;
        if (named.OriginalDefinition.SpecialType == SpecialType.System_Nullable_T)
            return CanCarryObjectState(named.TypeArguments[0], numeric, visiting, depth + 1);

        if (named.IsTupleType)
        {
            return named.TupleElements.All(element =>
                CanCarryObjectState(element.Type, numeric, visiting, depth + 1));
        }

        if (ParameterMemberSemanticModel.IsNumericStateStorage(type)) return false;
        if (IsInfrastructure(type)) return false;
        if (IsSerializableModel(type)) return true;

        if (named.Name is "List" or "IList" or "IReadOnlyList" or "IEnumerable"
                or "ICollection" or "IReadOnlyCollection"
                or "HashSet" or "ISet" or "IReadOnlySet"
            && named.TypeArguments.Length == 1)
            return CanCarryObjectState(named.TypeArguments[0], numeric, visiting, depth + 1);

        if (named.Name == "Dictionary" && named.TypeArguments.Length == 2)
        {
            var key = named.TypeArguments[0];
            bool safeKey = key.TypeKind == TypeKind.Enum
                || key.SpecialType is SpecialType.System_Boolean
                    or SpecialType.System_Byte or SpecialType.System_SByte
                    or SpecialType.System_Int16 or SpecialType.System_UInt16
                    or SpecialType.System_Int32 or SpecialType.System_UInt32
                    or SpecialType.System_Int64 or SpecialType.System_UInt64
                    or SpecialType.System_Char or SpecialType.System_String;
            return safeKey
                && CanCarryObjectState(named.TypeArguments[1], numeric, visiting, depth + 1);
        }

        if (named.TypeKind != TypeKind.Class || named.IsAbstract) return false;
        bool hasJsonConstructor = named.InstanceConstructors.Any(c => c.GetAttributes().Any(a =>
            a.AttributeClass?.ToDisplayString() == "Newtonsoft.Json.JsonConstructorAttribute"));
        if (!hasJsonConstructor
            && !named.InstanceConstructors.Any(c => c.Parameters.Length == 0
                || c.Parameters.All(p => p.IsOptional))
            && !HasSingleJsonMappableConstructor(named))
        {
            return false;
        }

        string identity = named.ToDisplayString();
        if (!visiting.Add(identity)) return true;

        for (var current = named; current is not null && current.SpecialType != SpecialType.System_Object;
            current = current.BaseType)
        {
            foreach (var property in current.GetMembers().OfType<IPropertySymbol>())
            {
                if (property.IsStatic || property.IsIndexer || property.GetMethod is null
                    || property.GetMethod.DeclaredAccessibility != Accessibility.Public)
                {
                    continue;
                }

                if (!CanCarryObjectState(property.Type, numeric, visiting, depth + 1))
                    return false;
            }

            foreach (var field in current.GetMembers().OfType<IFieldSymbol>())
            {
                if (field.IsStatic || field.DeclaredAccessibility != Accessibility.Public) continue;
                if (!CanCarryObjectState(field.Type, numeric, visiting, depth + 1))
                    return false;
            }
        }

        visiting.Remove(identity);
        return true;
    }

    /// <summary>
    /// Whether Json.NET can reconstruct a class through its single public value constructor.
    /// </summary>
    /// <remarks>
    /// Json.NET binds a lone public parameterized constructor by member name. Requiring every
    /// object-state type to also expose a parameterless or explicitly attributed constructor
    /// excluded immutable value records such as NEAT Genome/Connection even though their complete
    /// public shape is constructor-mappable. Keep the proof narrow: exactly one public constructor,
    /// and every required argument must have a same-typed readable public property or field.
    /// Remaining writable properties and constructor-created collections are populated by Json.NET
    /// after construction and are validated by the ordinary public-shape walk below.
    /// </remarks>
    private static bool HasSingleJsonMappableConstructor(INamedTypeSymbol type)
    {
        var constructors = type.InstanceConstructors
            .Where(c => c.DeclaredAccessibility == Accessibility.Public)
            .ToList();
        if (constructors.Count != 1 || constructors[0].Parameters.Length == 0) return false;

        foreach (var parameter in constructors[0].Parameters)
        {
            bool matched = false;
            for (var current = type; current is not null
                && current.SpecialType != SpecialType.System_Object; current = current.BaseType)
            {
                foreach (var member in current.GetMembers())
                {
                    if (!string.Equals(member.Name, parameter.Name,
                            System.StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    ITypeSymbol? memberType = member switch
                    {
                        IPropertySymbol { IsStatic: false, IsIndexer: false,
                            GetMethod.DeclaredAccessibility: Accessibility.Public } property => property.Type,
                        IFieldSymbol { IsStatic: false,
                            DeclaredAccessibility: Accessibility.Public } field => field.Type,
                        _ => null,
                    };
                    if (memberType is null) continue;
                    if (!SymbolEqualityComparer.Default.Equals(
                            memberType.WithNullableAnnotation(NullableAnnotation.NotAnnotated),
                            parameter.Type.WithNullableAnnotation(NullableAnnotation.NotAnnotated)))
                    {
                        continue;
                    }

                    matched = true;
                    break;
                }

                if (matched) break;
            }

            if (!matched && !parameter.IsOptional) return false;
        }

        return true;
    }

    /// <summary>Whether a member holds a model's options.</summary>
    /// <param name="type">The member's type.</param>
    /// <returns><see langword="true"/> when it derives from <c>ModelOptions</c>.</returns>
    private static bool IsModelOptions(ITypeSymbol type)
    {
        for (var current = type as INamedTypeSymbol; current is not null; current = current.BaseType)
        {
            if (current.Name == "ModelOptions") return true;
        }

        return false;
    }

    /// <summary>
    /// Finds a configured factory already owned by the parent for an assignable fitted child.
    /// </summary>
    private static string? ChildFactoryExpression(INamedTypeSymbol owner, ITypeSymbol childType)
    {
        var expected = childType.WithNullableAnnotation(NullableAnnotation.NotAnnotated);
        foreach (var member in owner.GetMembers())
        {
            ITypeSymbol? factoryType = member switch
            {
                IFieldSymbol { IsStatic: false } field => field.Type,
                IPropertySymbol { IsStatic: false, GetMethod: not null } property => property.Type,
                _ => null,
            };
            if (factoryType is not INamedTypeSymbol { IsGenericType: true, TypeArguments.Length: 1 } factory
                || factory.OriginalDefinition.ToDisplayString() != "System.Func<TResult>")
            {
                continue;
            }

            var produced = factory.TypeArguments[0]
                .WithNullableAnnotation(NullableAnnotation.NotAnnotated);
            if (!SymbolEqualityComparer.Default.Equals(produced, expected)) continue;

            return $"() => {member.Name} is null "
                + $"? throw new global::System.InvalidOperationException(\"Configured child factory '{member.Name}' is not available during restore.\") "
                + $": {member.Name}()";
        }

        return null;
    }

    /// <summary>
    /// True when an infrastructure component is itself fitted state with a canonical serializer.
    /// </summary>
    private static bool IsFittedSerializer(ITypeSymbol type)
    {
        if (!IsSerializableModel(type)) return false;
        return type.GetMembers("IsFitted").OfType<IPropertySymbol>().Any(property =>
            property.GetMethod is not null
            && property.Type.SpecialType == SpecialType.System_Boolean);
    }

    /// <summary>
    /// Recovers constructor expressions already used by Fit for derived helper components.
    /// </summary>
    /// <remarks>
    /// The generator reuses source construction rather than reverse-engineering constructor
    /// arguments. Expressions that read a method local are rejected; only owner members, type
    /// parameters, literals and member names rooted in those owner members are safe to replay.
    /// </remarks>
    private static List<(string Field, string Expression)> FittedInfrastructureRepairs(
        INamedTypeSymbol owner)
    {
        bool hasFittedLatch = false;
        for (var current = owner; current is not null; current = current.BaseType)
        {
            hasFittedLatch |= current.GetMembers("IsFitted").OfType<IPropertySymbol>().Any(property =>
                property.GetMethod is not null
                && property.Type.SpecialType == SpecialType.System_Boolean);
        }
        if (!hasFittedLatch) return new List<(string, string)>();

        var ownerNames = new HashSet<string>(owner.GetMembers()
            .Where(member => !member.IsStatic)
            .Select(member => member.Name), System.StringComparer.Ordinal);
        foreach (var parameter in owner.TypeParameters) ownerNames.Add(parameter.Name);

        var repairs = new List<(string Field, string Expression)>();
        foreach (var field in owner.GetMembers().OfType<IFieldSymbol>())
        {
            if (field.IsStatic || field.IsReadOnly || field.IsImplicitlyDeclared
                || field.NullableAnnotation != NullableAnnotation.Annotated
                || field.Type is not INamedTypeSymbol { TypeKind: TypeKind.Class, IsAbstract: false } fieldType
                || IsSerializableModel(field.Type) || IsLayer(field.Type)
                || !IsDerivedFittedHelperType(fieldType))
            {
                continue;
            }

            string? construction = null;
            foreach (var syntaxReference in owner.DeclaringSyntaxReferences)
            {
                if (syntaxReference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
                foreach (var assignment in declaration.DescendantNodes().OfType<AssignmentExpressionSyntax>())
                {
                    string assignedName = assignment.Left switch
                    {
                        IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
                        MemberAccessExpressionSyntax { Name: IdentifierNameSyntax identifier } =>
                            identifier.Identifier.ValueText,
                        _ => string.Empty,
                    };
                    if (assignedName != field.Name
                        || assignment.Right is not ObjectCreationExpressionSyntax creation
                        || !CanReplayConstructionExpression(creation, ownerNames))
                    {
                        continue;
                    }

                    construction = creation.ToString();
                    break;
                }
                if (construction is not null) break;
            }

            if (construction is not null) repairs.Add((field.Name, construction));
        }

        return repairs;
    }

    private static bool IsDerivedFittedHelperType(INamedTypeSymbol type)
        => IsInfrastructure(type)
           || type.GetAttributes().Any(attribute => attribute.AttributeClass?.Name is
               "ComponentTypeAttribute" or "PipelineStageAttribute");

    private static bool CanReplayConstructionExpression(
        ObjectCreationExpressionSyntax creation,
        HashSet<string> ownerNames)
    {
        foreach (var identifier in creation.DescendantNodes().OfType<IdentifierNameSyntax>())
        {
            // The constructed type and the right-hand names of member accesses are type/property
            // syntax, not captured locals. Only unqualified value roots need ownership proof.
            if (identifier.Parent is GenericNameSyntax
                || identifier.Parent is QualifiedNameSyntax
                || identifier.Parent is MemberAccessExpressionSyntax access
                    && ReferenceEquals(access.Name, identifier))
            {
                continue;
            }

            if (!ownerNames.Contains(identifier.Identifier.ValueText)) return false;
        }

        return true;
    }

    private static int StateRestorePriority(string call)
    {
        if (call.IndexOf(".DeclareOptions(", System.StringComparison.Ordinal) >= 0) return -100;
        if (call.IndexOf(".DeclareAfterRestore(", System.StringComparison.Ordinal) >= 0
            || call.IndexOf(".DeclareAfterParameterRestore(", System.StringComparison.Ordinal) >= 0)
            return 100;
        return 0;
    }

    private static string Render(
        INamedTypeSymbol type,
        string numeric,
        List<(string Name, string Call)> members,
        bool declaresHook,
        bool emitSerializationSurface)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();

        var ns = type.ContainingNamespace.IsGlobalNamespace ? null : type.ContainingNamespace.ToDisplayString();
        if (ns is not null)
        {
            sb.AppendLine($"namespace {ns};");
            sb.AppendLine();
        }

        // A NESTED type has to be reopened through the types that contain it. Emitting `partial class
        // Inner` at namespace level declares a DIFFERENT type -- one with no base -- and the override
        // then has nothing to override, which is what CS0115 was reporting for five model classes
        // declared inside their test fixtures. Outermost first, so the chain reads the way it is
        // written in the source.
        var chain = new List<INamedTypeSymbol>();
        for (var outer = type.ContainingType; outer is not null; outer = outer.ContainingType)
        {
            chain.Insert(0, outer);
        }

        var indent = string.Empty;

        foreach (var outer in chain)
        {
            sb.AppendLine($"{indent}partial class {outer.Name}{TypeParametersOf(outer)}");
            sb.AppendLine($"{indent}{{");
            indent += "    ";
        }

        sb.AppendLine($"{indent}partial class {type.Name}{TypeParametersOf(type)}");
        sb.AppendLine($"{indent}{{");
        sb.AppendLine($"{indent}    /// <summary>Auto-generated state declarations for this model's own members.</summary>");
        if (declaresHook)
        {
            sb.AppendLine($"{indent}    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.ModelStateGenerator\", \"1.0.0\")]");
            sb.AppendLine($"{indent}    private void RegisterGeneratedStateCore(global::AiDotNet.Models.ModelStateRegistry<{numeric}> state)");
            sb.AppendLine($"{indent}    {{");
        }
        else
        {
            sb.AppendLine($"{indent}    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.ModelStateGenerator\", \"1.0.0\")]");
            sb.AppendLine($"{indent}    protected override void RegisterGeneratedState(global::AiDotNet.Models.ModelStateRegistry<{numeric}> state)");
            sb.AppendLine($"{indent}    {{");
            sb.AppendLine($"{indent}        base.RegisterGeneratedState(state);");
        }

        // Ordered by name so the payload does not depend on declaration order, which a refactor can
        // change without anybody meaning to.
        foreach (var member in members
            .OrderBy(m => StateRestorePriority(m.Call))
            .ThenBy(m => m.Name, System.StringComparer.Ordinal))
        {
            sb.AppendLine($"{indent}        {member.Call}");
        }

        sb.AppendLine($"{indent}    }}");

        if (emitSerializationSurface)
        {
            sb.AppendLine();
            sb.AppendLine($"{indent}    /// <summary>Auto-generated common model serialization surface.</summary>");
            sb.AppendLine($"{indent}    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.ModelStateGenerator\", \"1.0.0\")]");
            sb.AppendLine($"{indent}    public override byte[] Serialize() => SerializeGeneratedModelState();");
            sb.AppendLine();
            sb.AppendLine($"{indent}    /// <summary>Auto-generated common model deserialization surface.</summary>");
            sb.AppendLine($"{indent}    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.ModelStateGenerator\", \"1.0.0\")]");
            sb.AppendLine($"{indent}    public override void Deserialize(byte[] data) => DeserializeGeneratedModelState(data);");
        }

        sb.AppendLine($"{indent}}}");

        for (var i = chain.Count - 1; i >= 0; i--)
        {
            indent = indent.Substring(0, indent.Length - 4);
            sb.AppendLine($"{indent}}}");
        }

        return sb.ToString();
    }

    /// <summary>The node type, when a member holds the root of a graph whose nodes point at their own kind.</summary>
    /// <remarks>
    /// Recognised by shape rather than by name, so a consumer's own tree is carried on the same terms
    /// as ours. Three things have to hold: it is a class, it can be built with no arguments -- the
    /// registry has to make one per node on restore -- and at least one settable property is typed as
    /// the node itself, which is what makes it a graph rather than a plain object.
    /// </remarks>
    private static INamedTypeSymbol? IsRecursiveNode(ITypeSymbol memberType, string numeric)
    {
        if (memberType is not INamedTypeSymbol { TypeKind: TypeKind.Class } named) return null;
        if (named.IsAbstract) return null;

        var self = named.ToDisplayString().TrimEnd('?');

        if (GraphNodeFactory(named, numeric) is null)
        {
            return null;
        }

        var properties = named.GetMembers().OfType<IPropertySymbol>()
            .Where(p => !p.IsStatic && p.GetMethod is not null)
            .ToList();

        // The typed graph path must be COMPLETE. Its former best-effort behavior recognized a
        // recursive node and then silently skipped dictionaries, long counters and getter-only
        // collections. HoeffdingTree consequently restored the shape of its tree but none of the
        // class statistics that decide a leaf prediction. When even one readable property cannot be
        // represented, decline the typed path so the general object-state declaration carries the
        // whole node instead.
        foreach (var property in properties)
        {
            if (property.SetMethod is null) return null;

            var propertyType = property.Type.ToDisplayString().TrimEnd('?');
            var bare = System.Text.RegularExpressions.Regex
                .Replace(propertyType, @"\b[A-Za-z_][A-Za-z0-9_]*\.", string.Empty)
                .Replace($"<{numeric}>", "<T>");

            if (propertyType != self
                && bare is not ("int" or "long" or "double" or "double[]" or "bool" or "T" or "Vector<T>"))
            {
                return null;
            }
        }

        var recursive = properties.Any(p => p.Type.ToDisplayString().TrimEnd('?') == self);

        return recursive ? named : null;
    }

    /// <summary>A recursive node factory expressible without model-specific code.</summary>
    private static string? GraphNodeFactory(INamedTypeSymbol node, string numeric)
    {
        var qualified = "global::" + node.ToDisplayString().TrimEnd('?');
        if (node.InstanceConstructors.Any(c => c.Parameters.Length == 0
            && c.DeclaredAccessibility == Accessibility.Public))
        {
            return $"new {qualified}()";
        }

        // Tree records often take the model's numeric zero solely to initialize generic scalar
        // properties. default(T) is exactly numeric zero for the supported numeric types and lets
        // the generated graph factory rebuild them without requiring a ceremonial parameterless
        // constructor on every nested node type.
        if (node.InstanceConstructors.Any(c => c.DeclaredAccessibility == Accessibility.Public
            && c.Parameters.Length == 1
            && c.Parameters[0].Type.ToDisplayString() == numeric))
        {
            return $"new {qualified}(default!)";
        }

        return null;
    }

    private static INamedTypeSymbol? IsDerivedDecisionTreeNode(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol { TypeKind: TypeKind.Class } named) return null;
        for (var current = named.BaseType; current is not null; current = current.BaseType)
        {
            if (current.OriginalDefinition.ToDisplayString()
                .StartsWith("AiDotNet.LinearAlgebra.DecisionTreeNode<", System.StringComparison.Ordinal))
            {
                return named.InstanceConstructors.Any(c => c.Parameters.Length == 0
                    && c.DeclaredAccessibility == Accessibility.Public)
                    ? named
                    : null;
            }
        }

        return null;
    }

    /// <summary>Builds the DeclareGraph call that carries a node graph.</summary>
    /// <remarks>
    /// Only the property shapes NodeShape can carry are described. What is left out is left out on
    /// purpose: a decision node also holds the training samples that produced it and, in a model-tree,
    /// a fitted sub-model, and neither is needed to reproduce a prediction. Carrying the samples would
    /// put the training set inside every saved model.
    /// </remarks>
    private static string GraphCall(
        string id,
        string name,
        INamedTypeSymbol node,
        string numeric,
        string setter)
    {
        var qualified = "global::" + node.ToDisplayString().TrimEnd('?');
        var shape = GraphShape(node, numeric);
        return $"state.DeclareGraph<{qualified}>(\"{id}\", () => {name}, {setter}, n => n{shape});";
    }

    private static string GraphListCall(
        string id,
        string name,
        INamedTypeSymbol node,
        string numeric,
        string setter)
    {
        var qualified = "global::" + node.ToDisplayString().TrimEnd('?');
        var shape = GraphShape(node, numeric);
        return $"state.DeclareGraphList<{qualified}>(\"{id}\", () => {name}, {setter}, n => n{shape});";
    }

    private static string GraphShape(INamedTypeSymbol node, string numeric)
    {
        var self = node.ToDisplayString().TrimEnd('?');
        var shape = new StringBuilder();

        shape.Append($".Create(() => {GraphNodeFactory(node, numeric)})");

        foreach (var property in node.GetMembers().OfType<IPropertySymbol>()
            .Where(p => !p.IsStatic && p.GetMethod is not null && p.SetMethod is not null)
            .OrderBy(p => p.Name, System.StringComparer.Ordinal))
        {
            var propertyType = property.Type.ToDisplayString().TrimEnd('?');
            var bare = System.Text.RegularExpressions.Regex
                .Replace(propertyType, @"\b[A-Za-z_][A-Za-z0-9_]*\.", string.Empty)
                .Replace($"<{numeric}>", "<T>");

            var call = propertyType == self ? "Child"
                : bare switch
                {
                    "int" => "Int32",
                    "long" => "Int64",
                    "double" => "Double",
                    "double[]" => "DoubleArray",
                    "bool" => "Boolean",
                    "T" => "Scalar",
                    "Vector<T>" => "Vector",
                    _ => null,
                };

            if (call is null) continue;

            shape.Append($".{call}(n => n.{property.Name}, (n, v) => n.{property.Name} = v)");
        }

        return shape.ToString();
    }

    private static string DerivedDecisionTreeGraphCall(
        string id,
        string name,
        INamedTypeSymbol node,
        string numeric,
        string setter)
    {
        var qualified = "global::" + node.ToDisplayString().TrimEnd('?');
        var shape = new StringBuilder()
            .Append($".Create(() => new {qualified}())")
            .Append(".Int32(n => n.FeatureIndex, (n, v) => n.FeatureIndex = v)")
            .Append(".Scalar(n => n.SplitValue, (n, v) => n.SplitValue = v)")
            .Append(".Scalar(n => n.Threshold, (n, v) => n.Threshold = v)")
            .Append(".Scalar(n => n.Prediction, (n, v) => n.Prediction = v)")
            .Append(".Boolean(n => n.IsLeaf, (n, v) => n.IsLeaf = v)");

        foreach (var property in node.GetMembers().OfType<IPropertySymbol>()
            .Where(p => !p.IsStatic && p.GetMethod is not null && p.SetMethod is not null)
            .OrderBy(p => p.Name, System.StringComparer.Ordinal))
        {
            var bare = System.Text.RegularExpressions.Regex
                .Replace(property.Type.ToDisplayString().TrimEnd('?'), @"\b[A-Za-z_][A-Za-z0-9_]*\.", string.Empty)
                .Replace($"<{numeric}>", "<T>");
            var call = bare switch
            {
                "int" => "Int32",
                "long" => "Int64",
                "double" => "Double",
                "double[]" => "DoubleArray",
                "bool" => "Boolean",
                "T" => "Scalar",
                "Vector<T>" => "Vector",
                _ => null,
            };
            if (call is not null)
                shape.Append($".{call}(n => n.{property.Name}, (n, v) => n.{property.Name} = v)");
        }

        shape
            .Append($".Child(n => ({qualified}?)n.Left, (n, v) => n.Left = v)")
            .Append($".Child(n => ({qualified}?)n.Right, (n, v) => n.Right = v)");

        return $"state.DeclareGraph<{qualified}>(\"{id}\", () => {name}, {setter}, n => n{shape});";
    }

    /// <summary>True for a member that is training machinery rather than state to restore.</summary>
    /// <remarks>
    /// THE RIGHT DISCRIMINATOR, replacing "single children are opt-in". That gate was written to keep a
    /// model's optimizer out of the payload, and it did -- along with every legitimate sub-model held in
    /// a field of its own. SiameseNetwork keeps its twin in <c>_subnetwork</c> and its head in
    /// <c>_outputLayer</c>, both single, and lost both: clone output moved 0.585 -> 0.502. Its optimizer
    /// sits in the very next field, which is what makes the distinction clear -- it is not arity that
    /// separates them, it is WHAT THEY ARE.
    /// <para>
    /// Matched by interface name so a consumer's own optimizer is excluded on the same terms as ours,
    /// and kept short on purpose: everything not on this list is state, which is the direction the
    /// default should fail in.
    /// </para>
    /// </remarks>
    private static bool IsInfrastructure(ITypeSymbol type)
    {
        static bool Machinery(string name)
            => name is "IOptimizer" or "IGradientBasedOptimizer" or "ILossFunction"
                or "ILearningRateScheduler" or "IRegularization" or "IActivationFunction"
                or "IAudioFeatureExtractor" or "Random";

        return Machinery(type.Name) || type.AllInterfaces.Any(i => Machinery(i.Name));
    }

    private static bool IsPartial(INamedTypeSymbol type)
        => type.DeclaringSyntaxReferences
            .Select(r => r.GetSyntax())
            .OfType<ClassDeclarationSyntax>()
            .Any(d => d.Modifiers.Any(m => m.ValueText == "partial"));

    private static string TypeParametersOf(INamedTypeSymbol type)
    {
        return type.TypeParameters.Length > 0
            ? "<" + string.Join(", ", type.TypeParameters.Select(p => p.Name)) + ">"
            : string.Empty;
    }
}
