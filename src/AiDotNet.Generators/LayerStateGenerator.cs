using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Generates the save/restore halves of layer construction state from
/// <c>[LayerState]</c>-marked constructor parameters.
/// </summary>
/// <remarks>
/// <para>
/// Replaces a 4811-line hand-written switch in <c>DeserializationHelper</c> in which 93 of 94
/// branches reverse-engineered constructor arguments out of the saved input shape. That inference
/// silently depended on layers encoding their capacity in their declared shape; when a layer
/// correctly declared an axis dynamic, the branch handed its constructor a <c>-1</c>.
/// </para>
/// <para>
/// For each annotated constructor this emits (a) a <c>protected override void WriteConstructionState</c>
/// on the layer writing every marked parameter, and (b) an entry in a central factory keyed by open
/// generic type that reconstructs the layer by calling that same constructor. Because both halves are
/// derived from one declaration, they cannot drift apart — which is the failure mode Keras's
/// hand-written <c>get_config</c>/<c>from_config</c> pairs are subject to and cannot detect.
/// </para>
/// <para>
/// WHERE THE GENERATED WRITER IS CALLED FROM: <c>LayerBase.GetMetadata</c> invokes
/// <c>WriteConstructionState</c> as its last step, and <c>LayerBase</c> supplies the empty virtual
/// base that the generated member overrides. That chain is what makes ADN0054's advice correct — an
/// author who overrides <c>GetMetadata</c> without calling <c>base.GetMetadata()</c> skips the
/// generated writer entirely, and every <c>[LayerState]</c> value is silently absent from the save.
/// A separate member rather than generating into <c>GetMetadata</c> itself, because a generated
/// partial cannot merge with a hand-written override of the same method.
/// </para>
/// </remarks>
[Generator]
public class LayerStateGenerator : IIncrementalGenerator
{
    private const string StateAttribute = "AiDotNet.Attributes.LayerStateAttribute";
    private const string CloneRandomSeedKey = "__aidotnet_clone_random_seed";

    private static readonly DiagnosticDescriptor NotPartial = new(
        "ADN0050",
        "Layer with [LayerState] must be partial",
        "'{0}' has [LayerState] constructor parameters, so its saved-state writer is generated into it; mark the class 'partial'",
        "AiDotNet.Serialization", DiagnosticSeverity.Error, true);

    private static readonly DiagnosticDescriptor NoBackingMember = new(
        "ADN0051",
        "[LayerState] parameter has no readable backing member",
        "'{0}' marks constructor parameter '{1}' as [LayerState], but no field or property named '{1}', '_{1}' or '{2}' of type '{3}' exists to read it back at save time; store the parameter in a field or drop the attribute",
        "AiDotNet.Serialization", DiagnosticSeverity.Error, true);

    private static readonly DiagnosticDescriptor UnsupportedType = new(
        "ADN0052",
        "[LayerState] parameter type cannot be serialized",
        "'{0}' marks parameter '{1}' of type '{2}' as [LayerState], but only int, long, float, double, bool, "
            + "string, enum, int[] and interface values can round-trip through layer metadata",
        "AiDotNet.Serialization", DiagnosticSeverity.Error, true);

    // Informational by design: pinning an optional argument is sometimes intentional, but it must
    // never be invisible because a non-default value would otherwise be lost during reconstruction.
    private static readonly DiagnosticDescriptor PinnedDefault = new(
        "ADN0057",
        "Optional constructor parameter is pinned to its default in the generated factory",
        "'{0}' pins optional constructor parameter '{1}' to its declared default, so a rebuilt layer "
            + "will not preserve a non-default value. Store it in a field named '{1}', '_{1}' or its "
            + "PascalCase form and the generator will round-trip it",
        "AiDotNet.Serialization", DiagnosticSeverity.Info, true);

    private static readonly DiagnosticDescriptor Unsuppliable = new(
        "ADN0053",
        "Required constructor parameter cannot be restored",
        "'{0}' cannot be rebuilt: parameter '{1}' of type '{2}' is required but is neither marked [LayerState], an activation function, nor optional; mark it, give it a default, or exclude this constructor",
        "AiDotNet.Serialization", DiagnosticSeverity.Error, true);

    private static readonly DiagnosticDescriptor NotALayer = new(
        "ADN0056",
        "[LayerState] is only supported on a class deriving from LayerBase",
        "'{0}' is a {1} and marks constructor parameters [LayerState], but the generated writer is "
            + "an `internal override` that only exists on LayerBase. Emitting it here produces compiler "
            + "errors inside generated code; move the type under LayerBase or drop the attribute",
        "AiDotNet.Serialization", DiagnosticSeverity.Error, true);

    private static readonly DiagnosticDescriptor UnsupportedArity = new(
        "ADN0055",
        "[LayerState] layer cannot be registered in the generated factory",
        "'{0}' has [LayerState] parameters and {1} type parameter(s), but the generated factory "
            + "only registers non-generic layers and layers with exactly one (the numeric type). Its state IS saved and "
            + "nothing can rebuild it, so deserialization falls back to the shape-inference path "
            + "this generator exists to replace; give the layer a single type parameter or exclude it",
        "AiDotNet.Serialization", DiagnosticSeverity.Warning, true);

    // TEMPORARY-DIAGNOSTIC (keep if it proves its worth): a layer that gets no factory, no clone
    // and no message is exactly the invisible-coverage failure this generator exists to remove.
    private static readonly DiagnosticDescriptor FactoryDeclined = new(
        "ADN0064",
        "Layer declined a generated clone factory",
        "'{0}' gets no generated factory: {1}. It will fail to clone with 'cannot be rebuilt'.",
        "AiDotNet.Serialization", DiagnosticSeverity.Warning, true);

    private static readonly DiagnosticDescriptor HandWrittenMetadata = new(
        "ADN0054",
        "Hand-written GetMetadata may drift from [LayerState]",
        "'{0}' overrides GetMetadata without calling base.GetMetadata(), so its generated [LayerState] values ({1}) are never written and a rebuild will fail; call base.GetMetadata() first",
        "AiDotNet.Serialization", DiagnosticSeverity.Warning, true);

    /// <inheritdoc/>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                // Any constructor that takes arguments. It used to require a parameter carrying an
                // ATTRIBUTE, which is the first of the two gates that made this generator only
                // able to see layers which had already opted in -- a rule that by construction
                // cannot report the layers that did not. Inference in Analyze is unreachable
                // without widening this, because a layer with no attributes never arrives here.
                //
                // Analyze does the real filtering, and it needs the semantic model to do it: it
                // rejects a host type that is not a LayerBase-derived class, and declines any
                // constructor with nothing restorable. A parameterless constructor is excluded
                // here because it carries no construction state by definition.
                static (node, _) => node is ConstructorDeclarationSyntax c
                    && c.ParameterList.Parameters.Count > 0,
                static (ctx, _) => Analyze(ctx))
            .Where(static m => m is not null)
            .Select(static (m, _) => m!);

        context.RegisterSourceOutput(candidates.Collect(), Emit);
    }

    private static LayerModel? Analyze(GeneratorSyntaxContext ctx)
    {
        var syntax = (ConstructorDeclarationSyntax)ctx.Node;
        if (ctx.SemanticModel.GetDeclaredSymbol(syntax) is not IMethodSymbol ctor) return null;

        // An unannotated convenience constructor that delegates with `: this(...)` holds no
        // construction-state claim of its own; the target does. An explicitly [LayerState]-marked
        // delegating constructor is different: it can assign additional state in its body after the
        // delegation (FeatureTokenizerLayer's runtime feature count is the representative case).
        // Discarding it before inspecting the attributes silently selects the narrower constructor.
        if (syntax.Initializer is { } selfInit
            && selfInit.IsKind(SyntaxKind.ThisConstructorInitializer)
            && !ctor.Parameters.Any(HasStateAttribute))
        {
            return null;
        }

        var marked = ctor.Parameters.Where(HasStateAttribute).ToList();

        // Captured BEFORE the inference block below reassigns `marked`, because afterwards the two
        // origins are indistinguishable. An author who wrote [LayerState] on a constructor stated
        // that this is the one to rebuild through; inference only guesses. Selection uses that.
        bool explicitlyMarked = marked.Count > 0;

        var safelyInferred = ctor.Parameters.Where(p =>
            // Activations use LayerBase's ordered scalar/vector construction channel. Treating
            // vectorActivation as ordinary component state merely because the inherited
            // VectorActivation property has the same name made that overload outrank its scalar
            // twin, even when the live layer was built with a scalar activation.
            !IsActivation(p.Type, out _)
            && Classify(p.Type) is not ValueKind.Unsupported
            // Interface-valued components are state too. The in-memory path supplies the live
            // configured object through WriteConstructionObjects; durable restoration records the
            // concrete type and either constructs it or fails loudly. Pinning a strategy/activation
            // to null is never safer: it silently changes the rebuilt layer.
            && FindBackingMember(
                ctor.ContainingType, p, ctx.SemanticModel, syntax,
                out _, out _) is not null).ToList();

        if (marked.Count == 0)
        {
            // INFERENCE. A constructor argument the layer stores in a field of the same name IS
            // construction state, whether or not anyone wrote the attribute. Gating on the
            // attribute alone meant a layer that stored every argument correctly was discarded
            // with no factory, no clone and no error -- a green build that had silently opted the
            // layer out. Requiring an opt-in is precisely what cannot report the layers that did
            // not opt in.
            //
            // Restricted to parameters that are BOTH restorable and backed by a non-nullable
            // member, which is not an optimisation: those three conditions are exactly the three
            // that raise ADN0053 / NoBackingMember below, so an inferred parameter cannot reach a
            // diagnostic. Reporting stays the exclusive province of an explicit [LayerState]
            // claim, which is the same narrowing ADN0056 already needed.
            marked = safelyInferred;

            // A factory can only call this constructor if EVERY required argument can be supplied.
            // Layers whose constructor takes another layer (DenseLoRAAdapter's baseLayer,
            // QuantizedDenseLayer's source) cannot be rebuilt from string metadata yet, so infer
            // nothing for them and leave the existing path in place. Emitting a partial factory
            // would trade a clear "no factory" for a call that cannot compile, and REPORTING it
            // would be a diagnostic against a layer whose author claimed nothing.
            // Activations are EXCLUDED from `marked` above on purpose -- LayerBase supplies them
            // through its ordered scalar/vector construction channel, not as component state. The
            // completeness check must therefore excuse them too, or a constructor whose every other
            // argument is stored is still declined for the one argument that was never meant to be
            // counted. That declined CSPBlock, which stores all four of its ints, and surfaced only
            // as "cannot be rebuilt: no generated factory" at clone time.
            if (marked.Count > 0
                && ctor.Parameters.Any(p => !p.IsOptional
                    && !IsActivation(p.Type, out _)
                    && !marked.Contains(p, SymbolEqualityComparer.Default)))
            {
                if (DerivesFromLayerBase(ctor.ContainingType)
                    && !ctor.ContainingType.IsAbstract
                    && ctor.ContainingType.TypeKind == TypeKind.Class
                    && !ctor.ContainingType.IsRecord)
                {
                    return new LayerModel
                {
                    TypeName = ctor.ContainingType.Name,
                    Location = new SourceSpan(syntax.Identifier.GetLocation()),
                    Diagnostics =
                    {
                        new PendingDiagnostic(
                            FactoryDeclined, new SourceSpan(syntax.Identifier.GetLocation()),
                            ctor.ContainingType.Name, "required parameter '" + ctor.Parameters.First(q => !q.IsOptional && !IsActivation(q.Type, out _) && !marked.Contains(q, SymbolEqualityComparer.Default)).Name + "' has no backing member the derived type can read"),
                    },
                };
                }

                return null;
            }
        }
        else
        {
            // An explicit attribute selects the constructor; it must not disable safe inference for
            // its other arguments. The old either/or rule preserved the marked dimensions while
            // silently pinning an unmarked padding, epsilon, momentum, or enum in the same call.
            foreach (var parameter in safelyInferred)
            {
                if (!marked.Contains(parameter, SymbolEqualityComparer.Default)) marked.Add(parameter);
            }
        }

        // ACTIVATION-ONLY CONSTRUCTORS ARE REBUILDABLE. ActivationLayer's only argument is its
        // activation, which is excluded from `marked` on purpose because LayerBase supplies it
        // through the ordered construction channel. Declining for "nothing restorable" confused
        // an empty state set with an unrebuildable layer: there is simply no state BEYOND the
        // activation, and the channel already carries that.
        bool activationOnly = marked.Count == 0
            && ctor.Parameters.Length > 0
            && ctor.Parameters.All(q => q.IsOptional || IsActivation(q.Type, out _))
            && ctor.Parameters.Any(q => IsActivation(q.Type, out _));

        // A constructor with nothing restorable is still declined: emitting a factory that cannot
        // rebuild the layer would replace a clear "no factory" with a silent wrong reconstruction.
        if (marked.Count == 0 && !activationOnly)
        {
                if (DerivesFromLayerBase(ctor.ContainingType)
                    && !ctor.ContainingType.IsAbstract
                    && ctor.ContainingType.TypeKind == TypeKind.Class
                    && !ctor.ContainingType.IsRecord)
                {
                    return new LayerModel
                {
                    TypeName = ctor.ContainingType.Name,
                    Location = new SourceSpan(syntax.Identifier.GetLocation()),
                    Diagnostics =
                    {
                        new PendingDiagnostic(
                            FactoryDeclined, new SourceSpan(syntax.Identifier.GetLocation()),
                            ctor.ContainingType.Name, "no constructor parameter resolves to restorable state"),
                    },
                };
                }

                return null;
            }

        var inferredState = new HashSet<string>(marked.Select(p => p.Name), System.StringComparer.Ordinal);

        var type = ctor.ContainingType;

        // An abstract layer has a constructor but cannot be instantiated, so a factory naming it
        // emits `new AbstractLayer<T>(...)` and fails with CS0144 inside generated source. Declined
        // silently and before the diagnostic below: an abstract base is not a mistake the author
        // made, it is a type that is only ever built through a derived class -- and that derived
        // class gets its own factory.
        if (type.IsAbstract) return null;

        // A type the generated table cannot NAME. GeneratedLayerFactories is a separate static
        // class, so a private nested layer (STCConnectorLayer's RegStageBlock) is unreachable from
        // it however correct its state declarations are. Declined rather than reported: the author
        // of a private helper layer has done nothing wrong, and ADN0055 firing on it is a
        // diagnostic about the table's reach rather than about their code.
        for (var scope = type; scope is not null; scope = scope.ContainingType)
        {
            if (scope.DeclaredAccessibility is Accessibility.Private or Accessibility.ProtectedAndInternal)
                return null;
        }

        // THE HOST TYPE MUST BE ABLE TO CARRY THE GENERATED MEMBER. Analyze accepted any
        // constructor whose parameters carried [LayerState] and then emitted
        // `partial class {TypeName}` with `protected override void WriteConstructionState`.
        // A struct, a record, or a class not derived from LayerBase produced a raw C#
        // compiler error pointing INTO generated source -- an error about code the author
        // never wrote and cannot open. Refused here with a diagnostic on the declaration
        // instead.
        if (type.TypeKind != TypeKind.Class || type.IsRecord || !DerivesFromLayerBase(type))
        {
            // Report ONLY an explicit claim. Now that every parameterized constructor is analysed
            // rather than only attributed ones, an unguarded report here tells every ordinary
            // class in the compilation to stop marking parameters it never marked -- measured at
            // 4,716 errors on the first build after widening the predicate. Declining silently is
            // the correct answer for a type that made no claim; the diagnostic exists for an
            // author who wrote [LayerState] somewhere it cannot work.
            if (!ctor.Parameters.Any(HasStateAttribute)) return null;

            return new LayerModel
            {
                TypeName = type.Name,
                Location = new SourceSpan(syntax.Identifier.GetLocation()),
                Diagnostics =
                {
                    new PendingDiagnostic(
                        NotALayer, new SourceSpan(syntax.Identifier.GetLocation()), type.Name, type.TypeKind.ToString().ToLowerInvariant()),
                },
            };
        }
        // Materialized so the "no override at all" case can be answered separately below: All over
        // an empty sequence is true, which would report a layer that never overrides GetMetadata as
        // one whose override fails to call base.
        var metadataOverrides = type.GetMembers("GetMetadata")
            .OfType<IMethodSymbol>()
            .Where(m => m.Parameters.Length == 0)
            .SelectMany(m => m.DeclaringSyntaxReferences)
            .Select(r => r.GetSyntax())
            .OfType<MethodDeclarationSyntax>()
            .ToList();

        var model = new LayerModel
        {
            Namespace = type.ContainingNamespace.IsGlobalNamespace
                ? null
                : type.ContainingNamespace.ToDisplayString(),
            TypeName = type.Name,
            HasExplicitState = explicitlyMarked,
            ContainingTypes = ContainingChain(type),
            TypeParameters = type.TypeParameters.Select(tp => tp.Name).ToList(),
            BaseFqn = type.ConstructedFrom.ToDisplayString(UnqualifiedGenerics),
            Location = new SourceSpan(syntax.Identifier.GetLocation()),
            IsPartial = type.DeclaringSyntaxReferences
                .Select(r => r.GetSyntax())
                .OfType<TypeDeclarationSyntax>()
                .Any(d => d.Modifiers.Any(SyntaxKind.PartialKeyword)),
            HasHandWrittenMetadata = metadataOverrides.Count > 0
                // SEMANTIC, NOT SUBSTRING. Scanning the method's full text for
                // "base.GetMetadata" fired on the string appearing in a comment or a string
                // literal, and MISSED a real call written `base . GetMetadata()`. Both
                // directions are wrong for a diagnostic whose whole job is telling the author
                // their metadata will silently not be written. Now an actual invocation of a
                // member access on `base` named GetMetadata.
                && metadataOverrides.All(d => !d.DescendantNodes()
                    .OfType<InvocationExpressionSyntax>()
                    .Any(inv => inv.Expression is MemberAccessExpressionSyntax
                    {
                        Expression: BaseExpressionSyntax,
                        Name.Identifier.ValueText: "GetMetadata",
                    })),
        };

        // Activation slots are ordinal per kind. LayerBase records the distinct activation objects
        // exposed by the layer and its immediate registered children in the same order, which lets
        // a composite preserve both hidden and output activations without per-layer backing fields.
        // The old one-per-kind gate silently defaulted every second activation.
        var scalarActivationIndex = 0;
        var vectorActivationIndex = 0;

        foreach (var p in ctor.Parameters)
        {
            var info = new ParamModel { Name = p.Name, TypeFqn = p.Type.ToDisplayString(FullyQualified) };

            // Membership rather than the attribute: `marked` is the attributed set when there is
            // one and the inferred set otherwise, so this one condition serves both. Testing
            // HasStateAttribute here instead would collect inferred parameters above and then emit
            // none of them -- the change would appear applied and do nothing.
            if (inferredState.Contains(p.Name))
            {
                info.IsState = true;
                info.IsOptionalState = p.IsOptional;
                if (p.IsOptional) info.DefaultExpression = RenderDefault(p);
                info.Key = StateKey(p) ?? p.Name;
                info.OmitWhenNonPositive = OmitWhenNonPositive(p);
                info.Kind = Classify(p.Type);
                if (info.Kind == ValueKind.Unsupported)
                {
                    model.Diagnostics.Add(new PendingDiagnostic(
                        UnsupportedType, SpanFor(p, model),
                        type.Name, p.Name, p.Type.ToDisplayString()));
                    return model;
                }

                info.BackingMember = FindBackingMember(
                    type, p, ctx.SemanticModel, syntax,
                    out var needsConvert, out var memberIsNullable);
                info.OwnerName = ctor.ContainingType.Name;
                if (p.Type is INamedTypeSymbol { IsGenericType: true, TypeArguments.Length: 1 } exprType
                    && exprType.ConstructedFrom.ToDisplayString(UnqualifiedGenerics)
                        == "System.Linq.Expressions.Expression")
                {
                    info.DelegateFqn = exprType.TypeArguments[0]
                        .ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                }
                info.NeedsConvert = needsConvert;
                info.IsNullable = IsNullableType(p.Type);
                info.BackingMemberIsNullable = memberIsNullable;
                if (info.BackingMember is null)
                {
                    model.Diagnostics.Add(new PendingDiagnostic(
                        NoBackingMember, SpanFor(p, model),
                        type.Name, p.Name, Pascal(p.Name), p.Type.ToDisplayString()));
                    return model;
                }

            }
            else if (IsActivation(p.Type, out var vector))
            {
                info.IsActivation = true;
                info.IsVectorActivation = vector;
                info.NumericTypeName = NumericTypeNameOf(ctor.ContainingType);
                info.OwnerName = ctor.ContainingType.Name;
                if (p.IsOptional) info.DefaultExpression = RenderDefault(p);

                // Prefer the constructor argument's own stored member when one exists. Composite
                // layers frequently expose LayerBase.ScalarActivation as Identity while storing a
                // different activation for an internal FFN (PreLNTransformerBlock is the canonical
                // case). Binding such an argument to ordered slot zero silently reconstructed GELU
                // as Identity. The ordered channel remains the fallback for composites whose
                // constructor activation is represented only by child layers.
                info.BackingMember = FindBackingMember(
                    type, p, ctx.SemanticModel, syntax,
                    out _, out _);
                if (info.BackingMember is not null)
                {
                    info.UseBackedActivation = true;
                    info.Key = StateKey(p) ?? p.Name;
                }
                if (vector)
                {
                    info.ActivationIndex = vectorActivationIndex++;
                }
                else
                {
                    info.ActivationIndex = scalarActivationIndex++;
                }
            }
            else if (p.IsOptional)
            {
                info.UseDefault = true;
                // Taken from the symbol, so the emitted argument is the value the
                // constructor signature actually promises.
                info.DefaultExpression = RenderDefault(p);

                // Entropy is deliberately NOT construction configuration. A Full clone installs
                // the original tensors and then applies the requested random-stream semantics;
                // an Architecture clone needs a fresh initialization. LayerCloning supplies its
                // derived seed under the reserved key below, so replaying the source constructor's
                // literal seed would be the wrong behaviour and reporting it as lost state would
                // tell authors to persist something the clone contract explicitly replaces.
                if (IsEntropyParameter(p))
                {
                    info.UseCloneRandomSeed = true;
                }
                else
                {
                    model.Diagnostics.Add(new PendingDiagnostic(
                        PinnedDefault, SpanFor(p, model), type.Name, p.Name));
                }
            }
            else
            {
                model.Diagnostics.Add(new PendingDiagnostic(
                    Unsuppliable, SpanFor(p, model),
                    type.Name, p.Name, p.Type.ToDisplayString()));
                return model;
            }

            model.Parameters.Add(info);
        }

        if (!model.IsPartial)
        {
            model.Diagnostics.Add(new PendingDiagnostic(NotPartial, model.Location, type.Name));
        }

        if (model.HasHandWrittenMetadata)
        {
            model.Diagnostics.Add(new PendingDiagnostic(
                HandWrittenMetadata, model.Location, type.Name,
                string.Join(", ", marked.Select(p => p.Name))));
        }

        model.IsValid = true;
        return model;
    }

    private static bool HasStateAttribute(IParameterSymbol p)
        => p.GetAttributes().Any(a => a.AttributeClass?.ToDisplayString() == StateAttribute);

    private static string? StateKey(IParameterSymbol p)
    {
        var attr = p.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == StateAttribute);
        var named = attr?.NamedArguments.FirstOrDefault(n => n.Key == "Key").Value.Value as string;
        return string.IsNullOrWhiteSpace(named) ? null : named;
    }

    /// <summary>
    /// Whether the author declared this parameter's zero to mean "not resolved yet", so the writer
    /// must skip it rather than save a value the constructor would reject.
    /// </summary>
    private static bool OmitWhenNonPositive(IParameterSymbol p)
    {
        var attr = p.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == StateAttribute);
        return attr?.NamedArguments.FirstOrDefault(n => n.Key == "OmitWhenNonPositive").Value.Value is true;
    }

    private static ValueKind Classify(ITypeSymbol type)
    {
        type = Unwrap(type);

        if (type.TypeKind == TypeKind.Enum) return ValueKind.Enum;
        if (type is ITypeParameterSymbol) return ValueKind.NumericTypeParameter;

        // Owned layer/delegate constructor values exist only on the in-memory channel. Layers must
        // be cloned recursively (not returned as aliases), while delegates are immutable callable
        // construction state. Classify these before the general interface component case so an
        // ILayer<T> parameter cannot quietly reuse the source child.
        // An expression TREE is the function as data, and AiDotNet.Serialization.ExpressionState
        // already saves and loads one against a host-approved method allowlist, rejecting anything
        // else BEFORE Compile() is called. It was written for exactly this and wired to nothing:
        // LambdaLayer stores Expression<Func<Tensor<T>,Tensor<T>>> deliberately ("the tree is what
        // lets a closure survive a save without naming a method") and still got no clone factory.
        if (type is INamedTypeSymbol { IsGenericType: true } expr
            && expr.ConstructedFrom.ToDisplayString(UnqualifiedGenerics)
                == "System.Linq.Expressions.Expression")
        {
            return ValueKind.Expression;
        }

        // THE EXTENSION POINT. Asked before the built-in kinds so a type can override how it is
        // carried, and asked as ONE question so a novel state type needs no generator change at
        // all -- which is the whole reason it exists.
        if (ImplementsPersistableState(type)) return ValueKind.PersistableState;

        if (IsCloneObject(type)) return ValueKind.CloneObject;

        // A pluggable strategy: record which implementation was used and rebuild that one.
        // A COLLECTION OF LAYERS is clone state, and must be tested BEFORE the interface case.
        // LayerStateBag already round-trips one end to end -- FormatCloneObject detects a layer
        // collection and writes it via FormatLayerCollection, and CloneObject<T> reads back the
        // "aidotnet-layer-list-v1:" prefix. Falling through to Component instead wrote the value
        // with FormatType, i.e. the TYPE NAME only: HybridBlockScheduler's ILayer<T>[] was declined
        // outright, and ParallelStreamsLayer's IEnumerable<ILayer<T>> was worse -- it got a factory
        // and would have cloned "successfully" with its stream layers silently gone.
        if (IsLayerCollection(type)) return ValueKind.CloneObject;

        if (type.TypeKind == TypeKind.Interface) return ValueKind.Component;

        // The runtime has read BooleanArray/StringArray all along; Classify simply never routed
        // to them, so a layer taking bool[] or string[] was declined as if the type were exotic.
        if (type is IArrayTypeSymbol { Rank: 1 } bl && bl.ElementType.SpecialType == SpecialType.System_Boolean)
            return ValueKind.BooleanArray;

        if (type is IArrayTypeSymbol { Rank: 1 } st && st.ElementType.SpecialType == SpecialType.System_String)
            return ValueKind.StringArray;

        if (type is IArrayTypeSymbol { Rank: 1 } dbl && dbl.ElementType.SpecialType == SpecialType.System_Double)
            return ValueKind.DoubleArray;

        if (type is IArrayTypeSymbol { Rank: 1 } arr && arr.ElementType.SpecialType == SpecialType.System_Int32)
            return ValueKind.Int32Array;

        if (type is IArrayTypeSymbol { Rank: 1, ElementType: IArrayTypeSymbol { Rank: 1 } row }
            && row.ElementType.SpecialType == SpecialType.System_Int32)
            return ValueKind.Int32Jagged;

        if (type is IArrayTypeSymbol { Rank: 1 } enumArray
            && enumArray.ElementType.TypeKind == TypeKind.Enum)
            return ValueKind.EnumArray;

        if (IsJsonConfiguration(type)) return ValueKind.JsonObject;

        return type.SpecialType switch
        {
            SpecialType.System_Int32 => ValueKind.Int32,
            SpecialType.System_Int64 => ValueKind.Int64,
            SpecialType.System_Double => ValueKind.Double,
            SpecialType.System_Single => ValueKind.Single,
            SpecialType.System_Boolean => ValueKind.Boolean,
            SpecialType.System_String => ValueKind.String,
            _ => ValueKind.Unsupported,
        };
    }

    /// <summary>True when the parameter is one of AiDotNet's activation interfaces.</summary>
    /// <remarks>
    /// MATCHED FULLY QUALIFIED. Comparing the bare simple name meant ANY interface called
    /// IActivationFunction, from any namespace or any referenced package, was treated as a
    /// restorable activation -- and the generated factory then emitted
    /// `scalarActivation as global::AiDotNet.Interfaces.IActivationFunction&lt;T&gt;`, a cast
    /// that yields null for the foreign type. The layer rebuilds with NO activation and
    /// nothing reports it.
    /// </remarks>
    private static bool IsActivation(ITypeSymbol type, out bool vector)
    {
        var fqn = ((type as INamedTypeSymbol)?.ConstructedFrom ?? type)
            .ToDisplayString(UnqualifiedGenerics);
        vector = fqn == "AiDotNet.Interfaces.IVectorActivationFunction";
        return vector || fqn == "AiDotNet.Interfaces.IActivationFunction";
    }

    /// <summary>Whether a fixed compile-time type is safe to round-trip as JSON configuration.</summary>
    private static bool IsJsonConfiguration(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol named || named.TypeKind != TypeKind.Class
            || !(named.Name.EndsWith("Config", System.StringComparison.Ordinal)
                 || named.Name.EndsWith("Options", System.StringComparison.Ordinal)))
            return false;

        var properties = named.GetMembers().OfType<IPropertySymbol>()
            .Where(p => !p.IsStatic && !p.IsIndexer && p.DeclaredAccessibility == Accessibility.Public)
            .Where(p => p.GetMethod is not null && p.SetMethod is not null)
            .ToList();
        return properties.Count > 0 && properties.All(p => IsJsonScalar(p.Type));
    }

    private static bool IsJsonScalar(ITypeSymbol type)
    {
        type = Unwrap(type);
        return type.TypeKind == TypeKind.Enum
            || type.SpecialType is SpecialType.System_Int32
                or SpecialType.System_Int64
                or SpecialType.System_Double
                or SpecialType.System_Single
                or SpecialType.System_Boolean
                or SpecialType.System_String;
    }

    /// <summary>
    /// Whether the in-memory construction-object channel can make an independent structural copy.
    /// </summary>
    /// <summary>An array or IEnumerable whose elements are layers, matching the runtime's
    /// TryGetLayerCollection.</summary>
    private static bool IsLayerCollection(ITypeSymbol type)
    {
        ITypeSymbol? element = type switch
        {
            IArrayTypeSymbol { Rank: 1 } a => a.ElementType,
            INamedTypeSymbol { IsGenericType: true } n
                when n.AllInterfaces.Any(i => i.ConstructedFrom.SpecialType
                        == SpecialType.System_Collections_Generic_IEnumerable_T)
                    || n.ConstructedFrom.SpecialType
                        == SpecialType.System_Collections_Generic_IEnumerable_T
                => n.TypeArguments.Length == 1 ? n.TypeArguments[0] : null,
            _ => null,
        };

        return element is not null && IsCloneObject(element);
    }

    /// <summary>Whether the type opts into ILayerStatePersistable.</summary>
    private static bool ImplementsPersistableState(ITypeSymbol type)
        => type is INamedTypeSymbol named
            && named.AllInterfaces.Any(i => i.ToDisplayString(UnqualifiedGenerics)
                == "AiDotNet.Serialization.ILayerStatePersistable");

    private static bool IsCloneObject(ITypeSymbol type)
    {
        if (type.TypeKind == TypeKind.Delegate) return true;
        if (type is not INamedTypeSymbol named) return false;

        string open = named.ConstructedFrom.ToDisplayString(UnqualifiedGenerics);
        if (open == "AiDotNet.Tensors.LinearAlgebra.Tensor") return true;
        if (open is "AiDotNet.Interfaces.ILayer" or "AiDotNet.NeuralNetworks.Layers.LayerBase"
            || named.AllInterfaces.Any(i =>
                i.ConstructedFrom.ToDisplayString(UnqualifiedGenerics) == "AiDotNet.Interfaces.ILayer")
            || DerivesFromLayerBase(named))
            return true;

        // Composite layer constructors commonly accept IEnumerable<ILayer<T>> while retaining a
        // private List<ILayer<T>>. Treat every one-argument layer collection abstraction as owned
        // construction topology, before the general interface classification can reduce it to a
        // type name. LayerStateBag clones the elements independently in memory and persists the
        // same allowlisted layer payload for durable restoration.
        if (named.TypeArguments.Length == 1
            && IsLayerValue(named.TypeArguments[0])
            && open is "System.Collections.Generic.IEnumerable"
                or "System.Collections.Generic.ICollection"
                or "System.Collections.Generic.IList"
                or "System.Collections.Generic.IReadOnlyCollection"
                or "System.Collections.Generic.IReadOnlyList")
        {
            return true;
        }

        // A list supplied to a composite constructor represents owned/shared child structure. The
        // base cloner duplicates every element (layers through Clone, other stateful components
        // through their generated/reflected configuration plan) and copies attributed state.
        return open == "System.Collections.Generic.List";
    }

    private static bool IsLayerValue(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol named) return false;
        string open = named.ConstructedFrom.ToDisplayString(UnqualifiedGenerics);
        return open is "AiDotNet.Interfaces.ILayer" or "AiDotNet.NeuralNetworks.Layers.LayerBase"
            || named.AllInterfaces.Any(i =>
                i.ConstructedFrom.ToDisplayString(UnqualifiedGenerics) == "AiDotNet.Interfaces.ILayer")
            || DerivesFromLayerBase(named);
    }

    /// <summary>Whether cloning intentionally replaces this optional entropy source.</summary>
    private static bool IsEntropyParameter(IParameterSymbol parameter)
    {
        if (string.Equals(parameter.Name, "seed", System.StringComparison.OrdinalIgnoreCase)
            && Unwrap(parameter.Type).SpecialType == SpecialType.System_Int32)
        {
            return true;
        }

        return string.Equals(parameter.Name, "random", System.StringComparison.OrdinalIgnoreCase)
            && Unwrap(parameter.Type).ToDisplayString(FullyQualified)
                == "global::System.Random";
    }

    private static string? FindBackingMember(
        INamedTypeSymbol type,
        IParameterSymbol p,
        SemanticModel semanticModel,
        ConstructorDeclarationSyntax constructor,
        out bool needsConvert,
        out bool memberIsNullable)
    {
        needsConvert = false;
        memberIsNullable = false;
        var candidates = new[] { p.Name, "_" + p.Name, "m_" + p.Name, Pascal(p.Name), "_" + Pascal(p.Name) };

        for (var t = type; t is not null; t = t.BaseType)
        {
            foreach (var name in candidates)
            {
                foreach (var member in t.GetMembers(name))
                {
                    // Must be readable from generated code inside the layer itself, and must hold
                    // the same type the constructor took -- otherwise the value written at save
                    // time is not the value the constructor was given.
                    if (member.DeclaredAccessibility == Accessibility.Private && !SymbolEqualityComparer.Default.Equals(t, type))
                        continue;

                    switch (member)
                    {
                        case IFieldSymbol f when MemberCanHold(f.Type, p.Type):
                            memberIsNullable = IsNullableType(f.Type);
                            return f.Name;
                        case IPropertySymbol { GetMethod: not null } prop when MemberCanHold(prop.Type, p.Type):
                            memberIsNullable = IsNullableType(prop.Type);
                            return prop.Name;

                        // Layers routinely keep a numeric constructor argument converted to their
                        // own numeric type (a double rate stored as T). That is still the value the
                        // constructor was given, so read it back through a conversion.
                        case IFieldSymbol f2 when IsNumericTypeParameter(f2.Type, p.Type, type):
                            needsConvert = true;
                            return f2.Name;
                        case IPropertySymbol { GetMethod: not null } prop2 when IsNumericTypeParameter(prop2.Type, p.Type, type):
                            needsConvert = true;
                            return prop2.Name;
                    }
                }
            }
        }

        // Names are a convention, not a contract. Resolve the member the constructor actually
        // assigns from this parameter so `_epsilon = Normalize(epsilon)`, `_config = config ??
        // Default`, and differently-named legacy fields remain generator-owned state rather than
        // requiring one-off annotations in every layer. The left side must be a readable member on
        // the layer and its type must be the same after unwrapping Nullable<T>; assignments hidden
        // in lambdas are ignored because they need not run during construction.
        foreach (var assignment in constructor.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            if (!assignment.IsKind(SyntaxKind.SimpleAssignmentExpression)
                || assignment.Ancestors().Any(a => a is AnonymousFunctionExpressionSyntax))
                continue;

            bool readsParameter = assignment.Right.DescendantNodesAndSelf()
                .OfType<IdentifierNameSyntax>()
                .Any(id => SymbolEqualityComparer.Default.Equals(
                    semanticModel.GetSymbolInfo(id).Symbol, p));
            if (!readsParameter) continue;

            var assigned = semanticModel.GetSymbolInfo(assignment.Left).Symbol;
            switch (assigned)
            {
                case IFieldSymbol field
                    when IsMemberOnLayerHierarchy(field.ContainingType, type)
                        && SameType(field.Type, p.Type):
                    memberIsNullable = IsNullableType(field.Type);
                    return field.Name;
                case IPropertySymbol { GetMethod: not null, IsIndexer: false } property
                    when IsMemberOnLayerHierarchy(property.ContainingType, type)
                        && SameType(property.Type, p.Type):
                    memberIsNullable = IsNullableType(property.Type);
                    return property.Name;
                case IFieldSymbol numericField
                    when IsMemberOnLayerHierarchy(numericField.ContainingType, type)
                        && IsNumericTypeParameter(numericField.Type, p.Type, type):
                    needsConvert = true;
                    return numericField.Name;
                case IPropertySymbol { GetMethod: not null, IsIndexer: false } numericProperty
                    when IsMemberOnLayerHierarchy(numericProperty.ContainingType, type)
                        && IsNumericTypeParameter(numericProperty.Type, p.Type, type):
                    needsConvert = true;
                    return numericProperty.Name;
            }
        }


        // FOLLOW `: base(...)`. A layer that forwards a constructor argument straight to a base
        // constructor parameter IS restorable through the base's own declaration, but inference
        // missed it on two counts: the names differ (SwiGLU's `outputSize` -> GatedLinearUnitLayer's
        // `outputDimension`), and the loop above skips base members it cannot see. The name mapping
        // is resolved here rather than by renaming the derived parameter, which would break callers
        // written as `new SwiGLUFeedForwardLayer(outputSize: 64)`.
        //
        // ACCESSIBILITY IS STILL ENFORCED: the result is emitted as `this.<member>` inside the
        // DERIVED type's generated partial, so returning a private base field would produce
        // generated code that does not compile. Only members the derived type can actually read
        // are accepted -- a private base field must be widened to protected at the base instead.
        if (constructor.Initializer is { } baseInit
            && baseInit.IsKind(SyntaxKind.BaseConstructorInitializer)
            && semanticModel.GetSymbolInfo(baseInit).Symbol is IMethodSymbol baseCtor)
        {
            var forwarded = baseInit.ArgumentList.Arguments;
            for (int i = 0; i < forwarded.Count; i++)
            {
                if (forwarded[i].Expression is not IdentifierNameSyntax forwardedId) continue;
                if (!SymbolEqualityComparer.Default.Equals(
                        semanticModel.GetSymbolInfo(forwardedId).Symbol, p)) continue;

                IParameterSymbol? target = forwarded[i].NameColon is { } nameColon
                    ? baseCtor.Parameters.FirstOrDefault(
                        bp => bp.Name == nameColon.Name.Identifier.ValueText)
                    : (i < baseCtor.Parameters.Length ? baseCtor.Parameters[i] : null);
                if (target is null) continue;

                var baseCandidates = new[]
                {
                    target.Name, "_" + target.Name, "m_" + target.Name,
                    Pascal(target.Name), "_" + Pascal(target.Name)
                };

                for (var bt = baseCtor.ContainingType; bt is not null; bt = bt.BaseType)
                {
                    foreach (var name in baseCandidates)
                    {
                        foreach (var member in bt.GetMembers(name))
                        {
                            if (member.DeclaredAccessibility == Accessibility.Private) continue;

                            switch (member)
                            {
                                case IFieldSymbol bf when SameType(bf.Type, p.Type):
                                    memberIsNullable = IsNullableType(bf.Type);
                                    return bf.Name;
                                case IPropertySymbol { GetMethod: not null } bp2
                                    when SameType(bp2.Type, p.Type):
                                    memberIsNullable = IsNullableType(bp2.Type);
                                    return bp2.Name;
                            }
                        }
                    }
                }
            }
        }

        return null;
    }

    private static bool IsMemberOnLayerHierarchy(INamedTypeSymbol? memberType, INamedTypeSymbol layerType)
    {
        for (var current = layerType; current is not null; current = current.BaseType)
        {
            if (SymbolEqualityComparer.Default.Equals(current, memberType)) return true;
        }
        return false;
    }

    /// <summary>True when the member is held as THE LAYER'S numeric type parameter.</summary>
    /// <remarks>
    /// WHICH type parameter is the whole question, and the old test did not ask it:
    /// `member is ITypeParameterSymbol` matched ANY type parameter on the type. A field typed
    /// as an unrelated parameter -- TState, TKey, a second generic -- was classified as the
    /// numeric one and the emitted Convert threw at SAVE time, the worst moment to find it:
    /// the model is already trained. The layer's numeric type is by convention its FIRST type
    /// parameter, so the member must be exactly that one.
    /// </remarks>
    private static bool IsNumericTypeParameter(ITypeSymbol member, ITypeSymbol parameter, INamedTypeSymbol containingType)
    {
        if (member is not ITypeParameterSymbol tp) return false;

        var numeric = containingType.TypeParameters.Length > 0 ? containingType.TypeParameters[0] : null;
        if (numeric is null || !SymbolEqualityComparer.Default.Equals(tp, numeric)) return false;

        return parameter.SpecialType is SpecialType.System_Double
            or SpecialType.System_Single
            or SpecialType.System_Int32;
    }

    /// <summary>
    /// Whether <paramref name="memberType"/> can hold a value of <paramref name="paramType"/>.
    /// </summary>
    /// <remarks>
    /// Strict symbol equality is too narrow for layer-valued state. DenseLoRAAdapter has an
    /// overload taking <c>LayerBase&lt;T&gt; baseLayer</c> while LoRAAdapterBase stores it in
    /// <c>protected readonly ILayer&lt;T&gt; _baseLayer</c> -- a perfectly ordinary widening that
    /// left the layer with no factory at all. Restoration records the CONCRETE type and the read
    /// is typed by the parameter, so a member declared as a base class or implemented interface
    /// of the parameter is a correct home for it. Kept to clone/component kinds: numeric and
    /// array kinds still require an exact match.
    /// </remarks>
    private static bool MemberCanHold(ITypeSymbol memberType, ITypeSymbol paramType)
    {
        if (SameType(memberType, paramType)) return true;
        if (!IsCloneObject(paramType) && paramType.TypeKind != TypeKind.Interface) return false;

        var target = Unwrap(memberType).WithNullableAnnotation(NullableAnnotation.None);
        var source = Unwrap(paramType).WithNullableAnnotation(NullableAnnotation.None);

        for (var b = (source as INamedTypeSymbol)?.BaseType; b is not null; b = b.BaseType)
        {
            if (SymbolEqualityComparer.Default.Equals(
                    b.WithNullableAnnotation(NullableAnnotation.None), target))
                return true;
        }

        return source.AllInterfaces.Any(i => SymbolEqualityComparer.Default.Equals(
            i.WithNullableAnnotation(NullableAnnotation.None), target));
    }

    private static bool SameType(ITypeSymbol a, ITypeSymbol b)
        => SymbolEqualityComparer.Default.Equals(
            Unwrap(a).WithNullableAnnotation(NullableAnnotation.None),
            Unwrap(b).WithNullableAnnotation(NullableAnnotation.None));

    /// <summary>True when the saved member can carry null and the format must preserve that fact.</summary>
    private static bool IsNullableType(ITypeSymbol type)
        => type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T }
            || (type.IsReferenceType && type.NullableAnnotation == NullableAnnotation.Annotated);

    /// <summary>Strips <c>Nullable&lt;T&gt;</c> so an <c>int?</c> parameter matches an <c>int</c> field.</summary>
    private static ITypeSymbol Unwrap(ITypeSymbol type)
        => type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } n
            ? n.TypeArguments[0]
            : type;

    private static string Pascal(string name)
        => name.Length == 0 ? name : char.ToUpperInvariant(name[0]) + name.Substring(1);

    private static void Emit(SourceProductionContext spc, ImmutableArray<LayerModel> models)
    {
        // One constructor per type: if a layer offers several, the one that RESTORES THE MOST wins,
        // with source order kept only as the final tie-break.
        //
        // Source order alone chose the constructor that happened to be written first, which is not a
        // statement about fidelity. A layer whose narrow convenience overload precedes its fuller one
        // had the narrow one selected and every parameter only the fuller one carries was dropped from
        // the save with no diagnostic -- the factory still compiled and still returned a layer, just a
        // differently-configured one. FeatureTokenizerLayer(embeddingDim) at line 102 beat
        // FeatureTokenizerLayer([LayerState] numFeatures, [LayerState] embeddingDim) at line 116, so a
        // restored tokenizer kept numFeatures = -1, never allocated its [F,E] weights, reported
        // ParameterCount 0, and let SetParameters discard 512 trained values in silence.
        //
        // Explicit [LayerState] outranks inference because it is an author's claim about which
        // constructor rebuilds the layer, and inference is only this generator's guess.
        var candidatesByType = models
            .Where(m => m.IsValid)
            .GroupBy(TypeKey, System.StringComparer.Ordinal)
            // DETERMINISTIC. `g.First()` took whatever order Collect() yielded, and Roslyn
            // does not document that collected results keep source order -- for a type split
            // across partial files the per-file order is undefined, so a layer annotating two
            // constructors could generate a different factory between builds. Ordering on the
            // constructor's own location also makes "first by source order" true.
            .Select(g => g
                .OrderByDescending(m => m.HasExplicitState)
                .ThenByDescending(m => m.StateCount)
                .ThenBy(m => m.Location.FilePath, System.StringComparer.Ordinal)
                .ThenBy(m => m.Location.Start)
                .ToList())
            .OrderBy(g => TypeKey(g[0]), System.StringComparer.Ordinal)
            .ToList();
        var byType = candidatesByType.Select(g => g[0]).ToList();

        // A required positive dimension can coexist with a lazy convenience constructor that
        // omits it. The live object then legitimately stores the dimension sentinel as zero even
        // after its tensors have materialized (BatchNormalizationLayer is the canonical case).
        // Infer the existing OmitWhenNonPositive contract only when another generated constructor
        // for the same type can rebuild without that key; ordinary zero-valued state such as axis
        // remains serialized exactly.
        foreach (var candidates in candidatesByType)
        {
            var writer = candidates[0];
            foreach (var parameter in writer.Parameters.Where(p =>
                         p.IsState && p.Kind == ValueKind.Int32 && IsPositiveDimensionName(p.Name)))
            {
                if (candidates.Any(candidate => candidate.Parameters.All(p =>
                        !p.IsState || !string.Equals(p.Key, parameter.Key, System.StringComparison.Ordinal))))
                {
                    parameter.OmitWhenNonPositive = true;
                }
            }
        }

        // Diagnostics on a valid constructor describe the factory that is actually emitted, not
        // every convenience overload on the type. Reporting them before selection made a correct
        // wide constructor fail ADN0057 because a narrower forwarding overload pinned a value the
        // generated factory never consumed (PatchGANDiscriminator.receptiveField is one example).
        // Invalid candidates still report: an explicit [LayerState] claim must not disappear just
        // because another overload happens to be usable.
        var selected = new HashSet<LayerModel>(byType);
        foreach (var model in models.Where(m => !m.IsValid || selected.Contains(m)))
        {
            foreach (var diagnostic in model.Diagnostics)
            {
                spc.ReportDiagnostic(diagnostic.ToDiagnostic());
            }
        }

        foreach (var model in byType)
        {
            // UNIQUE OR THE BUILD THROWS. Built from the SIMPLE name (which also drops
            // arity) while the grouping key is namespace-qualified, so two annotated layers
            // both named DenseLayer in different namespaces emitted the same file name and
            // AddSource threw on the duplicate. Derived from the same key the grouping uses.
            // A LAYER THAT SAVES BUT CANNOT BE REBUILT IS REPORTED, not skipped in silence.
            // The factory registers non-generic layers and single-type-parameter layers. A type
            // declared Foo<T, TState> wrote its [LayerState] values to metadata and had
            // no TryCreate entry: deserialization silently fell back to the shape-inference
            // path this generator was built to replace, which is the -1 bug it fixes.
            if (model.TypeParameters.Count > 1)
            {
                spc.ReportDiagnostic(Diagnostic.Create(
                    UnsupportedArity, model.Location.ToLocation(), model.TypeName, model.TypeParameters.Count));
            }

            spc.AddSource($"{HintName(model)}.LayerState.g.cs", SourceText(EmitWriter(model)));
        }

        spc.AddSource(
            "GeneratedLayerFactories.g.cs",
            SourceText(EmitFactories(candidatesByType.SelectMany(group => group).ToList())));
    }

    private static bool IsPositiveDimensionName(string name)
    {
        string lowered = name.ToLowerInvariant();
        return lowered.Contains("size") || lowered.Contains("count")
            || lowered.Contains("feature") || lowered.Contains("dimension")
            || lowered.EndsWith("dim", System.StringComparison.Ordinal)
            || lowered.Contains("width") || lowered.Contains("height")
            || lowered.Contains("depth") || lowered.Contains("channels")
            || lowered.Contains("heads");
    }

    private static Microsoft.CodeAnalysis.Text.SourceText SourceText(string text)
        => Microsoft.CodeAnalysis.Text.SourceText.From(text, Encoding.UTF8);

    private static string EmitWriter(LayerModel model)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/> Generated by LayerStateGenerator. Do not edit.");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        if (model.Namespace is not null)
        {
            sb.AppendLine($"namespace {model.Namespace};");
            sb.AppendLine();
        }

        var generics = model.TypeParameters.Count == 0
            ? string.Empty
            : "<" + string.Join(", ", model.TypeParameters) + ">";

        // REOPEN EVERY CONTAINING TYPE. Writing the partial straight at namespace scope
        // put a nested layer's generated member on a DIFFERENT type than the one it belongs
        // to, so the `internal override` had no base member and the build failed inside
        // generated code -- an error about source the author never wrote. Each outer type is
        // reopened as a partial carrying its own generics.
        foreach (var outer in model.ContainingTypes)
        {
            sb.AppendLine($"partial class {outer}");
            sb.AppendLine("{");
        }

        sb.AppendLine($"partial class {model.TypeName}{generics}");
        sb.AppendLine("{");
        sb.AppendLine("    /// <inheritdoc/>");
        sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.LayerStateGenerator\", \"1.0.0\")]");
        sb.AppendLine("    protected override void WriteConstructionState(global::System.Collections.Generic.Dictionary<string, string> __metadata)");
        sb.AppendLine("    {");
        sb.AppendLine("        base.WriteConstructionState(__metadata);");
        foreach (var p in model.Parameters.Where(p => p.UseBackedActivation))
        {
            sb.AppendLine($"        if (this.{p.BackingMember} is not null)");
            sb.AppendLine("        {");
            sb.AppendLine($"            __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.FormatType(this.{p.BackingMember});");
            sb.AppendLine("        }");
        }
        foreach (var p in model.Parameters.Where(p => p.IsState))
        {
            if (p.Kind is ValueKind.Component or ValueKind.CloneObject)
            {
                // A null component is ABSENT state, not an empty type name. Keeping an empty key
                // made HasAll succeed and selected a vector-activation constructor for a scalar
                // layer, which then failed while resolving the empty component.
                sb.AppendLine($"        if (this.{p.BackingMember} is not null)");
                sb.AppendLine("        {");
                string componentFormatter = p.Kind == ValueKind.CloneObject
                    ? "FormatCloneObject"
                    : "FormatType";
                sb.AppendLine($"            __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.{componentFormatter}(this.{p.BackingMember});");
                sb.AppendLine("        }");
                continue;
            }

            if (p.Kind == ValueKind.JsonObject)
            {
                sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.FormatJson(this.{p.BackingMember});");
                continue;
            }

            if (p.Kind == ValueKind.PersistableState)
            {
                sb.AppendLine($"        if (this.{p.BackingMember} is not null)");
                sb.AppendLine("        {");
                sb.AppendLine($"            __metadata[\"{p.Key}\"] = this.{p.BackingMember}.SaveState();");
                sb.AppendLine("        }");
                continue;
            }
            if (p.Kind == ValueKind.Expression)
            {
                sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.ExpressionState.Save(this.{p.BackingMember});");
                continue;
            }
            if (p.Kind == ValueKind.EnumArray)
            {
                sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.FormatEnumArray(this.{p.BackingMember});");
                continue;
            }

            if (p.Kind == ValueKind.NumericTypeParameter)
            {
                string numeric = model.TypeParameters.Count > 0 ? model.TypeParameters[0] : "T";
                sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.Format(global::AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<{numeric}>().ToDouble(this.{p.BackingMember}));");
                continue;
            }

            var read = p.NeedsConvert
                ? ConvertExpression(p, model.TypeParameters.Count > 0 ? model.TypeParameters[0] : "T")
                : $"this.{p.BackingMember}";

            // A size the layer has not resolved yet is written as NOTHING, not as 0. Saving the 0
            // produced a state the layer's own constructor rejects ("featureSize must be positive,
            // got 0"), because for a lazily-shaped layer 0 is the truth and the constructor still
            // refuses it. Omitting the key makes the generated factory's state.HasAll(...) check
            // fail, so TryCreate returns false and the caller falls through to the lazy build path,
            // which is exactly right for a layer with no width yet.
            if (p.OmitWhenNonPositive)
            {
                var positive = p.BackingMemberIsNullable
                    ? $"this.{p.BackingMember}.HasValue && this.{p.BackingMember}.Value > 0"
                    : $"this.{p.BackingMember} > 0";
                var omitFormatter = p.BackingMemberIsNullable ? "FormatNullable" : "Format";
                sb.AppendLine($"        if ({positive})");
                sb.AppendLine("        {");
                sb.AppendLine($"            __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.{omitFormatter}({read});");
                sb.AppendLine("        }");
                continue;
            }

            var formatter = p.BackingMemberIsNullable ? "FormatNullable" : "Format";
            sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.{formatter}({read});");
        }
        sb.AppendLine("    }");

        var components = model.Parameters
            .Where(p => p.UseBackedActivation
                || (p.IsState && p.Kind is ValueKind.Component or ValueKind.JsonObject or ValueKind.CloneObject))
            .ToList();
        if (components.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("    /// <inheritdoc/>");
            sb.AppendLine("    [global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.LayerStateGenerator\", \"1.0.0\")]");
            sb.AppendLine("    protected override void WriteConstructionObjects(global::System.Collections.Generic.Dictionary<string, object> __values)");
            sb.AppendLine("    {");
            sb.AppendLine("        base.WriteConstructionObjects(__values);");
            foreach (var p in components)
            {
                sb.AppendLine($"        if (this.{p.BackingMember} is object __component_{p.Name})");
                sb.AppendLine("        {");
                sb.AppendLine($"            __values[\"{p.Key}\"] = __component_{p.Name};");
                sb.AppendLine("        }");
            }
            sb.AppendLine("    }");
        }
        sb.AppendLine("}");
        // Close the outer types opened above.
        for (int i = 0; i < model.ContainingTypes.Count; i++)
        {
            sb.AppendLine("}");
        }
        return sb.ToString();
    }

    /// <summary>Reads a value stored in the layer's numeric type parameter back as a primitive.</summary>
    /// <remarks>
    /// THROUGH THE LIBRARY'S NUMERIC ABSTRACTION, NOT System.Convert.
    /// <c>System.Convert.ToDouble(object, IFormatProvider)</c> throws <c>InvalidCastException</c> when
    /// the boxed value does not implement <c>IConvertible</c>. This library is generic over its numeric
    /// type through <c>INumericOperations&lt;T&gt;</c>, NOT through <c>IConvertible</c>, so any custom
    /// numeric struct failed here -- at SAVE time, on a trained model, inside generated code the author
    /// never wrote and cannot open. <c>MathHelper.GetNumericOperations&lt;T&gt;().ToDouble</c> is the
    /// path the rest of the codebase uses and carries no such requirement.
    ///
    /// Only <c>ToDouble</c> is used, with a C# conversion to the declared parameter type on top: it is
    /// the one conversion every <c>INumericOperations&lt;T&gt;</c> implementation provides. The integral
    /// cases go through <c>Math.Round</c> rather than a bare cast so they keep <c>Convert.ToInt32</c>'s
    /// round-half-to-even behaviour instead of silently truncating a value that floating-point error
    /// left at 63.9999999.
    /// </remarks>
    private static string ConvertExpression(ParamModel p, string numericTypeParameter)
    {
        var asDouble = $"global::AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<{numericTypeParameter}>()" +
                       $".ToDouble(this.{p.BackingMember})";

        return p.Kind switch
        {
            ValueKind.Int32 => $"(int)global::System.Math.Round({asDouble}, global::System.MidpointRounding.ToEven)",
            ValueKind.Int64 => $"(long)global::System.Math.Round({asDouble}, global::System.MidpointRounding.ToEven)",
            ValueKind.Single => $"(float)({asDouble})",
            _ => asDouble,
        };
    }

    private static string EmitFactories(List<LayerModel> models)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/> Generated by LayerStateGenerator. Do not edit.");
        sb.AppendLine("#nullable enable");
        sb.AppendLine();
        sb.AppendLine("namespace AiDotNet.Serialization;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>");
        sb.AppendLine("/// Rebuilds layers from their saved [LayerState] constructor arguments.");
        sb.AppendLine("/// </summary>");
        sb.AppendLine("/// <remarks>");
        sb.AppendLine("/// Every entry calls the layer's real constructor with the values it was originally given, so");
        sb.AppendLine("/// no dimension is inferred from the saved shape and a dynamic axis cannot corrupt a rebuild.");
        sb.AppendLine("/// </remarks>");
        sb.AppendLine("/// <typeparam name=\"T\">The layer's numeric type.</typeparam>");
        sb.AppendLine("[global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.LayerStateGenerator\", \"1.0.0\")]");
        sb.AppendLine("internal static class GeneratedLayerFactories<T>");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Number of layer types with generated factories.</summary>");
        sb.AppendLine($"    internal const int Count = {models.Where(m => m.TypeParameters.Count <= 1).Select(TypeKey).Distinct(System.StringComparer.Ordinal).Count()};");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Attempts to rebuild a layer of the given open generic type.</summary>");
        sb.AppendLine("    /// <param name=\"genericDefinition\">The layer's open generic type, e.g. <c>typeof(DenseLayer&lt;&gt;)</c>.</param>");
        sb.AppendLine("    /// <param name=\"state\">The layer's saved metadata.</param>");
        sb.AppendLine("    /// <param name=\"scalarActivation\">Restored scalar activation, when the constructor takes one.</param>");
        sb.AppendLine("    /// <param name=\"vectorActivation\">Restored vector activation, when the constructor takes one.</param>");
        sb.AppendLine("    /// <param name=\"layer\">The rebuilt layer.</param>");
        sb.AppendLine("    /// <returns><c>true</c> when a factory exists for the type.</returns>");
        sb.AppendLine("    internal static bool TryCreate(");
        sb.AppendLine("        global::System.Type genericDefinition,");
        sb.AppendLine("        global::AiDotNet.Serialization.LayerStateBag state,");
        sb.AppendLine("        object? scalarActivation,");
        sb.AppendLine("        object? vectorActivation,");
        sb.AppendLine("        out object? layer)");
        sb.AppendLine("    {");

        foreach (var candidates in models
                     .Where(m => m.TypeParameters.Count <= 1)
                     .GroupBy(TypeKey, System.StringComparer.Ordinal)
                     .OrderBy(g => g.Key, System.StringComparer.Ordinal))
        {
            var first = candidates.First();
            var ordered = candidates
                .OrderByDescending(m => m.HasExplicitState)
                .ThenByDescending(m => m.StateCount)
                .ThenBy(m => m.Location.FilePath, System.StringComparer.Ordinal)
                .ThenBy(m => m.Location.Start)
                .ToList();
            var closed = first.ClosedFqn;

            sb.AppendLine($"        if (genericDefinition == typeof({first.OpenGenericFqn}))");
            sb.AppendLine("        {");

            foreach (var model in ordered)
            {
                var args = string.Join(", ", model.Parameters.Select(p => Argument(p)));
                var required = model.Parameters
                    // An optional state slot is absent when its live backing member is null. Requiring
                    // that omitted key made the factory reject the very constructor whose declared
                    // default can rebuild it (LambdaLayer's optional backward delegate is the minimal
                    // example). Required state remains fail-closed.
                    .Where(p => p.IsState && !p.IsOptionalState)
                    .Select(p => "state.Has(\"" + p.Key + "\")")
                    .ToList();

                bool scalar = model.Parameters.Any(p => p.IsActivation && !p.IsVectorActivation);
                bool vector = model.Parameters.Any(p => p.IsActivation && p.IsVectorActivation);
                if (scalar)
                {
                    required.Add("vectorActivation is null");
                    var slot = model.Parameters.First(p => p.IsActivation && !p.IsVectorActivation);
                    if (slot.DefaultExpression is null)
                        required.Add(slot.UseBackedActivation
                            ? $"state.Has(\"{slot.Key}\")"
                            : "scalarActivation is not null || state.Has(\"__aidotnet_scalar_activation_0\")");
                }
                else if (vector)
                {
                    required.Add("scalarActivation is null");
                    var slot = model.Parameters.First(p => p.IsActivation && p.IsVectorActivation);
                    if (slot.DefaultExpression is null)
                        required.Add("vectorActivation is not null || state.Has(\"__aidotnet_vector_activation_0\")");
                }
                else
                {
                    required.Add("scalarActivation is null");
                    required.Add("vectorActivation is null");
                }

                string condition = required.Count == 0 ? "true" : string.Join(" && ", required);
                sb.AppendLine($"            if ({condition})");
                sb.AppendLine("            {");
                sb.AppendLine($"                layer = new {closed}({args});");
                sb.AppendLine("                return true;");
                sb.AppendLine("            }");
            }

            sb.AppendLine();
            sb.AppendLine("            layer = null;");
            sb.AppendLine("            return false;");
            sb.AppendLine("        }");
            sb.AppendLine();
        }

        sb.AppendLine("        layer = null;");
        sb.AppendLine("        return false;");
        sb.AppendLine("    }");
        sb.AppendLine("}");
        return sb.ToString();
    }

    private static string Argument(ParamModel p)
    {
        if (p.IsActivation)
        {
            var iface = p.IsVectorActivation
                ? $"global::AiDotNet.Interfaces.IVectorActivationFunction<{p.NumericTypeName}>"
                : $"global::AiDotNet.Interfaces.IActivationFunction<{p.NumericTypeName}>";
            var source = p.IsVectorActivation ? "vectorActivation" : "scalarActivation";
            var kind = p.IsVectorActivation ? "vector" : "scalar";
            var key = $"__aidotnet_{kind}_activation_{p.ActivationIndex}";
            if (p.UseBackedActivation) key = p.Key;
            var fallback = p.ActivationIndex == 0
                ? $"{source} as {iface}"
                : p.DefaultExpression ?? "default";
            var expression = $"state.Has(\"{key}\") "
                + $"? state.Component<{iface}>(\"{key}\") : {fallback}";

            // The factory predicate proves a required activation exists, but nullable flow state
            // does not cross the generated if-condition into this separately rendered argument.
            // Assert only for a required constructor slot; optional/nullable slots retain null.
            if (p.DefaultExpression is null) expression = $"({expression})!";
            return $"{p.Name}: {expression}";
        }

        if (p.UseCloneRandomSeed)
        {
            var fallback = p.DefaultExpression ?? "default!";
            return p.TypeFqn.TrimEnd('?') == "global::System.Random"
                ? $"{p.Name}: state.Has(\"{CloneRandomSeedKey}\") "
                    + $"? global::AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(state.Int32(\"{CloneRandomSeedKey}\")) "
                    + $": {fallback}"
                : $"{p.Name}: state.Has(\"{CloneRandomSeedKey}\") "
                    + $"? state.Int32(\"{CloneRandomSeedKey}\") : {fallback}";
        }

        // THE PARAMETER'S DEFAULT, NOT THE TYPE'S. Falls back to `default!` only when the
        // declaration genuinely has no value to render.
        if (p.UseDefault) return $"{p.Name}: {p.DefaultExpression ?? "default!"}";

        var read = (p.Kind, p.IsNullable) switch
        {
            (ValueKind.Int32, false) => $"state.Int32(\"{p.Key}\")",
            (ValueKind.Int32, true) => $"state.NullableInt32(\"{p.Key}\")",
            (ValueKind.Int64, false) => $"state.Int64(\"{p.Key}\")",
            (ValueKind.Int64, true) => $"state.NullableInt64(\"{p.Key}\")",
            (ValueKind.Double, false) => $"state.Double(\"{p.Key}\")",
            (ValueKind.Double, true) => $"state.NullableDouble(\"{p.Key}\")",
            (ValueKind.Single, false) => $"state.Single(\"{p.Key}\")",
            (ValueKind.Single, true) => $"state.NullableSingle(\"{p.Key}\")",
            (ValueKind.Boolean, false) => $"state.Boolean(\"{p.Key}\")",
            (ValueKind.Boolean, true) => $"state.NullableBoolean(\"{p.Key}\")",
            (ValueKind.String, false) => $"state.String(\"{p.Key}\")",
            (ValueKind.String, true) => $"state.NullableString(\"{p.Key}\")",
            (ValueKind.PersistableState, _) =>
                $"state.PersistableState<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            (ValueKind.Expression, _) =>
                $"global::AiDotNet.Serialization.ExpressionState.Load<{p.DelegateFqn}>("
                    + $"state.String(\"{p.Key}\"), \"{p.OwnerName}\", \"{p.Key}\")",
            (ValueKind.BooleanArray, _) => $"state.BooleanArray(\"{p.Key}\")",
            (ValueKind.StringArray, _) => $"state.StringArray(\"{p.Key}\")",
            (ValueKind.DoubleArray, false) => $"state.DoubleArray(\"{p.Key}\")",
            (ValueKind.DoubleArray, true) => $"state.NullableDoubleArray(\"{p.Key}\")",
            (ValueKind.Int32Array, false) => $"state.Int32Array(\"{p.Key}\")",
            (ValueKind.Int32Array, true) => $"state.NullableInt32Array(\"{p.Key}\")",
            (ValueKind.Int32Jagged, _) => $"state.Int32Jagged(\"{p.Key}\")",
            (ValueKind.EnumArray, _) => $"state.EnumArray<{p.TypeFqn.TrimEnd('?', '[', ']')}>(\"{p.Key}\")",
            (ValueKind.Enum, false) => $"state.Enum<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            (ValueKind.Enum, true) => $"state.NullableEnum<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            (ValueKind.JsonObject, _) => $"state.JsonObject<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            (ValueKind.CloneObject, _) => $"state.CloneObject<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            (ValueKind.NumericTypeParameter, _) => $"global::AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(state.Double(\"{p.Key}\"))",
            // Component returns null when the key is absent, which is correct for an optional
            // slot and a nullable-warning error (CS8604) when the constructor parameter is not
            // nullable. The factory already guards the whole call with state.HasAll(...), so a
            // null here means the payload disagreed with the layer AFTER that check passed --
            // worth an exception naming the key rather than a NullReferenceException from inside
            // the constructor. Not suppressed with `!`: that would hand the constructor a null and
            // fail somewhere less informative.
            (ValueKind.Component, false) when !p.TypeFqn.EndsWith("?", System.StringComparison.Ordinal) =>
                $"state.Component<{p.TypeFqn}>(\"{p.Key}\") ?? throw new global::System.InvalidOperationException("
                    + $"\"Saved state for '{p.Key}' is missing or names a type that could not be loaded.\")",
            (ValueKind.Component, _) => $"state.Component<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            _ => "default!",
        };

        if (p.IsOptionalState)
        {
            read = $"state.Has(\"{p.Key}\") ? {read} : {p.DefaultExpression ?? "default!"}";
        }

        return $"{p.Name}: {read}";
    }

    private static readonly SymbolDisplayFormat FullyQualified =
        SymbolDisplayFormat.FullyQualifiedFormat;

    /// <summary>Namespace-qualified, no generics, and NO <c>global::</c> prefix.</summary>
    /// <remarks>
    /// THE global:: PREFIX MADE EVERY COMPARISON AGAINST THIS FORMAT FALSE.
    /// <c>SymbolDisplayFormat.FullyQualifiedFormat</c> sets
    /// <c>SymbolDisplayGlobalNamespaceStyle.Included</c>, so a symbol rendered as
    /// <c>global::AiDotNet.NeuralNetworks.Layers.LayerBase</c> never equalled the
    /// <c>"AiDotNet.NeuralNetworks.Layers.LayerBase"</c> it was compared with. Two checks were
    /// silently inert as a result: DerivesFromLayerBase always returned false, so ADN0056 fired on
    /// EVERY [LayerState] layer (120 of them on the full branch), and IsActivation never matched, so
    /// no activation parameter was ever classified as a component.
    /// </remarks>
    private static readonly SymbolDisplayFormat UnqualifiedGenerics =
        SymbolDisplayFormat.FullyQualifiedFormat
            .WithGlobalNamespaceStyle(SymbolDisplayGlobalNamespaceStyle.Omitted)
            .WithGenericsOptions(SymbolDisplayGenericsOptions.None);

    private enum ValueKind
    {
        Unsupported,
        Int32,
        Int64,
        Double,
        Single,
        Boolean,
        String,
        Enum,
        Int32Array,
        DoubleArray,
        BooleanArray,
        StringArray,
        Expression,
        PersistableState,
        Int32Jagged,
        EnumArray,
        Component,
        JsonObject,
        CloneObject,
        NumericTypeParameter,
    }

    /// <summary>The symbol's own span when it has one, else the model's.</summary>
    private static SourceSpan SpanFor(ISymbol symbol, LayerModel model)
    {
        var loc = symbol.Locations.FirstOrDefault();
        return loc is null || loc == Location.None ? model.Location : new SourceSpan(loc);
    }

    /// <summary>True when the type derives from AiDotNet's LayerBase.</summary>
    /// <summary>The numeric type a layer is closed over: "T" when generic, else e.g. "float".</summary>
    private static string NumericTypeNameOf(INamedTypeSymbol type)
    {
        if (type.TypeParameters.Length > 0) return "T";
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.ConstructedFrom.ToDisplayString(UnqualifiedGenerics) == "AiDotNet.NeuralNetworks.Layers.LayerBase"
                && b.TypeArguments.Length == 1)
            {
                return b.TypeArguments[0].ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            }
        }
        return "T";
    }

    private static bool DerivesFromLayerBase(INamedTypeSymbol type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.ConstructedFrom.ToDisplayString(UnqualifiedGenerics) == "AiDotNet.NeuralNetworks.Layers.LayerBase")
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>The outer types a nested declaration must reopen, outermost first.</summary>
    private static List<string> ContainingChain(INamedTypeSymbol type)
    {
        var chain = new List<string>();
        for (var outer = type.ContainingType; outer is not null; outer = outer.ContainingType)
        {
            var generics = outer.TypeParameters.Length == 0
                ? string.Empty
                : "<" + string.Join(", ", outer.TypeParameters.Select(tp => tp.Name)) + ">";
            chain.Insert(0, outer.Name + generics);
        }
        return chain;
    }

    /// <summary>A file-name-safe, collision-free stem for the type.</summary>
    /// <remarks>
    /// FULL NAME PLUS ARITY. AddSource throws on a duplicate hint name, so this must be unique
    /// across every emitted model. The simple name is not: two layers with the same name in
    /// different namespaces collide, and ISymbol.Name also excludes arity, so a non-generic type
    /// and a generic one of the same name collide too. BaseFqn carries the namespace and the
    /// containing types; the arity suffix carries the rest. It is the same key the models are
    /// grouped by, so one group can never produce two hint names or two groups one.
    /// </remarks>
    private static string HintName(LayerModel model)
    {
        var stem = TypeKey(model);
        var sb = new StringBuilder(stem.Length);
        foreach (var ch in stem)
        {
            sb.Append(char.IsLetterOrDigit(ch) || ch == '_' ? ch : '_');
        }
        return sb.ToString();
    }

    /// <summary>Identity of the emitted type: full name plus arity.</summary>
    private static string TypeKey(LayerModel model)
        => model.TypeParameters.Count == 0
            ? model.BaseFqn
            : model.BaseFqn + "`" + model.TypeParameters.Count;

    /// <summary>Renders an optional parameter's declared default as a C# expression.</summary>
    /// <remarks>
    /// Returns null when the default cannot be rendered faithfully rather than guessing: a
    /// wrong default is worse than an explicit `default!`, because it is a value the
    /// constructor never promised and nothing downstream can tell it from a real one.
    /// </remarks>
    private static string? RenderDefault(IParameterSymbol p)
    {
        if (!p.HasExplicitDefaultValue) return null;
        var v = p.ExplicitDefaultValue;

        if (v is null)
        {
            // For an unconstrained T? Roslyn reports the explicit `default` value as null, but
            // `T-value : null` cannot be target-typed back to T?. Spell the type's default out so
            // both branches of the generated conditional have the same T type. Reference types
            // retain the literal null promised by their declarations.
            return p.Type.TypeKind == TypeKind.TypeParameter
                ? $"default({p.Type.ToDisplayString(FullyQualified).TrimEnd('?')})"
                : p.Type.IsValueType ? "default" : "null";
        }

        // Enums arrive as their underlying integral value, so the declared enum type is cast
        // back on -- a bare number does not compile against an enum-typed parameter.
        if (p.Type.TypeKind == TypeKind.Enum)
        {
            return $"({p.Type.ToDisplayString(FullyQualified)}){System.Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture)}";
        }

        // INVARIANT CULTURE and explicit suffixes: on a comma-decimal machine ToString()
        // renders 0.5 as "0,5", which is not valid C#, and an unsuffixed 0.5 does not compile
        // against a float parameter.
        return v switch
        {
            bool b => b ? "true" : "false",
            string str => SymbolDisplay.FormatLiteral(str, quote: true),
            char c => SymbolDisplay.FormatLiteral(c, quote: true),
            float f when float.IsNaN(f) => "global::System.Single.NaN",
            float f when float.IsPositiveInfinity(f) => "global::System.Single.PositiveInfinity",
            float f when float.IsNegativeInfinity(f) => "global::System.Single.NegativeInfinity",
            float f => f.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "f",
            double d when double.IsNaN(d) => "global::System.Double.NaN",
            double d when double.IsPositiveInfinity(d) => "global::System.Double.PositiveInfinity",
            double d when double.IsNegativeInfinity(d) => "global::System.Double.NegativeInfinity",
            double d => d.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "d",
            decimal m => m.ToString(System.Globalization.CultureInfo.InvariantCulture) + "m",
            long l => l.ToString(System.Globalization.CultureInfo.InvariantCulture) + "L",
            _ => System.Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture),
        };
    }

    private sealed class ParamModel
    {
        public string Name = string.Empty;
        public string TypeFqn = string.Empty;
        public string Key = string.Empty;
        public string? BackingMember;
        public bool IsState;
        /// <summary>Whether an inferred/declared state slot has a constructor default.</summary>
        public bool IsOptionalState;
        public bool IsActivation;
        public bool IsVectorActivation;
        // "T" for a generic layer; the CONCRETE type for a non-generic one.
        // QuantizedDenseLayer : LayerBase<float> has no T, so emitting
        // IVectorActivationFunction<T> for it produced generated code that would not compile.
        public string NumericTypeName = "T";
        // For ValueKind.Expression: the TDelegate of Expression<TDelegate>, and the owning layer
        // name that ExpressionState.Load reports in its rejection message.
        public string DelegateFqn = string.Empty;
        public string OwnerName = string.Empty;
        /// <summary>Whether this activation has an exact constructor-argument backing member.</summary>
        public bool UseBackedActivation;
        /// <summary>Zero-based position among scalar or vector activation constructor slots.</summary>
        public int ActivationIndex;
        public bool UseDefault;
        /// <summary>The parameter's DECLARED default rendered as C#, or null if it has none.</summary>
        /// <remarks>
        /// `default!` is the default of the TYPE, not of the PARAMETER. Every optional
        /// parameter with a non-zero default was rebuilt wrong and silently: `useBias = true`
        /// came back false, `dropoutRate = 0.5` came back 0.0, `heads = 8` came back 0. The
        /// layer loads without error and behaves differently from the one that was saved.
        /// </remarks>
        public string? DefaultExpression;
        public bool NeedsConvert;
        /// <summary>Whether the clone adapter supplies a derived entropy seed for this slot.</summary>
        public bool UseCloneRandomSeed;
        /// <summary>Whether the readable backing member itself can carry null.</summary>
        public bool IsNullable;
        /// <summary>Whether the writer's member, as opposed to the constructor parameter, can be null.</summary>
        public bool BackingMemberIsNullable;
        public ValueKind Kind;

        /// <summary>
        /// The author declared that a zero here means "not resolved yet", so the writer guards it.
        /// </summary>
        public bool OmitWhenNonPositive;
    }

    /// <summary>A location reduced to primitives, so it neither roots a Compilation nor breaks equality.</summary>
    private readonly struct SourceSpan : System.IEquatable<SourceSpan>
    {
        public static readonly SourceSpan None = default;

        public SourceSpan(Location location)
        {
            var lineSpan = location.GetLineSpan();
            FilePath = lineSpan.Path ?? string.Empty;
            Start = location.SourceSpan.Start;
            Length = location.SourceSpan.Length;
            StartLine = lineSpan.StartLinePosition.Line;
            StartChar = lineSpan.StartLinePosition.Character;
            EndLine = lineSpan.EndLinePosition.Line;
            EndChar = lineSpan.EndLinePosition.Character;
        }

        public string FilePath { get; }
        public int Start { get; }
        public int Length { get; }
        public int StartLine { get; }
        public int StartChar { get; }
        public int EndLine { get; }
        public int EndChar { get; }

        /// <summary>Rebuilds a reportable Location at emit time, where rooting no longer matters.</summary>
        public Location ToLocation()
            => string.IsNullOrEmpty(FilePath)
                ? Location.None
                : Location.Create(
                    FilePath,
                    new Microsoft.CodeAnalysis.Text.TextSpan(Start, Length),
                    new Microsoft.CodeAnalysis.Text.LinePositionSpan(
                        new Microsoft.CodeAnalysis.Text.LinePosition(StartLine, StartChar),
                        new Microsoft.CodeAnalysis.Text.LinePosition(EndLine, EndChar)));

        public bool Equals(SourceSpan other)
            => FilePath == other.FilePath && Start == other.Start && Length == other.Length
               && StartLine == other.StartLine && StartChar == other.StartChar
               && EndLine == other.EndLine && EndChar == other.EndChar;

        public override bool Equals(object? obj) => obj is SourceSpan o && Equals(o);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = FilePath?.GetHashCode() ?? 0;
                h = (h * 397) ^ Start;
                h = (h * 397) ^ Length;
                return h;
            }
        }
    }

    /// <summary>A diagnostic as descriptor id + span + arguments, rebuilt only at emit time.</summary>
    private readonly struct PendingDiagnostic : System.IEquatable<PendingDiagnostic>
    {
        public PendingDiagnostic(DiagnosticDescriptor descriptor, SourceSpan span, params object?[] args)
        {
            Descriptor = descriptor;
            Span = span;
            // Joined into one string so equality is a string comparison rather than a
            // reference comparison over an array, which would never compare equal and would
            // defeat the caching this exists to restore.
            Args = string.Join("\u001f", args.Select(a => a?.ToString() ?? string.Empty));
        }

        public DiagnosticDescriptor Descriptor { get; }
        public SourceSpan Span { get; }
        public string Args { get; }

        public Diagnostic ToDiagnostic()
            => Diagnostic.Create(
                Descriptor,
                Span.ToLocation(),
                Args.Length == 0 ? new object[0] : Args.Split('\u001f'));

        public bool Equals(PendingDiagnostic other)
            => ReferenceEquals(Descriptor, other.Descriptor) && Span.Equals(other.Span) && Args == other.Args;

        public override bool Equals(object? obj) => obj is PendingDiagnostic o && Equals(o);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = Descriptor?.Id?.GetHashCode() ?? 0;
                h = (h * 397) ^ Span.GetHashCode();
                h = (h * 397) ^ (Args?.GetHashCode() ?? 0);
                return h;
            }
        }
    }

    private sealed class LayerModel : System.IEquatable<LayerModel>
    {
        public string? Namespace;
        public string TypeName = string.Empty;
        public List<string> TypeParameters = new();
        public string BaseFqn = string.Empty;

        /// <summary>Outer types, outermost first, each as it must be REOPENED in generated code.</summary>
        /// <remarks>
        /// A layer declared inside another type had its `partial class` emitted at NAMESPACE
        /// scope, so `WriteConstructionState` never joined the real class and `internal
        /// override` on a namespace-level type with no such base member failed to compile --
        /// a generator error surfacing as a raw C# error in code the author never wrote.
        /// </remarks>
        public List<string> ContainingTypes = new();

        /// <summary>The type as <c>typeof(X&lt;&gt;)</c> renders it.</summary>
        public string OpenGenericFqn => TypeParameters.Count == 0
            ? BaseFqn
            : BaseFqn + "<" + new string(',', TypeParameters.Count - 1) + ">";

        /// <summary>The type closed over the factory's single numeric parameter.</summary>
        public string ClosedFqn => TypeParameters.Count == 0 ? BaseFqn : BaseFqn + "<T>";
        public List<ParamModel> Parameters = new();
        /// <summary>Diagnostics as DATA, not as live Diagnostic instances.</summary>
        /// <remarks>
        /// A Diagnostic holds a Location, a Location holds a SyntaxTree, and a SyntaxTree roots
        /// the whole Compilation. Holding them as incremental pipeline state meant Roslyn could
        /// not release the previous compilation between builds, and -- because neither type has
        /// value equality -- could not cache the step either, so the generator re-ran in full on
        /// every keystroke while pinning the old compilation in memory. Both halves of that are
        /// fixed by carrying only what is needed to REBUILD the diagnostic at emit time.
        /// </remarks>
        public List<PendingDiagnostic> Diagnostics = new();
        public SourceSpan Location = SourceSpan.None;
        public bool IsPartial;
        public bool HasHandWrittenMetadata;
        public bool IsValid;

        /// <summary>Whether this constructor's state came from [LayerState], not from inference.</summary>
        public bool HasExplicitState;

        /// <summary>How much construction state rebuilding through this constructor restores.</summary>
        public int StateCount => Parameters.Count(p => p.IsState);

        /// <summary>Value equality, which is what lets Roslyn cache this pipeline step.</summary>
        /// <remarks>
        /// Reference equality on a mutable class means two structurally identical models from
        /// consecutive builds never compare equal, so the incremental pipeline treats every
        /// build as a change and re-runs the whole generator. Every member compared here is a
        /// string, bool, or a sequence of them -- nothing that roots a Compilation.
        /// </remarks>
        public bool Equals(LayerModel? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;
            return Namespace == other.Namespace
                && TypeName == other.TypeName
                && BaseFqn == other.BaseFqn
                && IsPartial == other.IsPartial
                && HasExplicitState == other.HasExplicitState
                && HasHandWrittenMetadata == other.HasHandWrittenMetadata
                && IsValid == other.IsValid
                && Location.Equals(other.Location)
                && TypeParameters.SequenceEqual(other.TypeParameters)
                && ContainingTypes.SequenceEqual(other.ContainingTypes)
                && Diagnostics.SequenceEqual(other.Diagnostics)
                && Parameters.SequenceEqual(other.Parameters);
        }

        public override bool Equals(object? obj) => Equals(obj as LayerModel);

        public override int GetHashCode()
        {
            unchecked
            {
                int h = TypeName?.GetHashCode() ?? 0;
                h = (h * 397) ^ (BaseFqn?.GetHashCode() ?? 0);
                h = (h * 397) ^ Parameters.Count;
                h = (h * 397) ^ Diagnostics.Count;
                return h;
            }
        }
    }
}
