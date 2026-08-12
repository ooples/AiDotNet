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
/// For each annotated constructor this emits (a) an <c>internal override void WriteConstructionState</c>
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
    // INFO, not a warning. Pinning is often correct -- an optional argument nobody varies really is
    // its default forever -- so this must not fail a build or drown the log. What it must do is stop
    // being INVISIBLE. Until now the generator silently substituted the literal default for any
    // optional parameter it could not read back, so a layer whose clone quietly reverts a
    // hyperparameter looked exactly like one that round-trips perfectly, and the only way to find out
    // which was to sweep every model and read the wreckage. ContinuumMemorySystemLayer is the worked
    // example: numFrequencyLevels was pinned to 3 while updateFrequencies WAS recorded, so a layer
    // built with five levels came back as (3 levels, 5 frequencies) and its own constructor rejected
    // the pair -- HopeNetwork could not be cloned at all, and nothing said so until a sweep did.
    private static readonly DiagnosticDescriptor PinnedDefault = new(
        "ADN0057",
        "Optional constructor parameter is pinned to its default in the generated factory",
        "'{0}' pins optional constructor parameter '{1}' to its declared default, so a rebuilt layer "
            + "will not preserve a non-default value. Store it in a field named '{1}', '_{1}' or its "
            + "PascalCase form and the generator will round-trip it",
        // INFO, and it has to be. This project builds with warnings-as-errors, so shipping this at
        // Warning turned 80 pinned parameters into 84 build ERRORS and failed the build outright --
        // a reporting diagnostic must not be able to do that. Info does not surface at normal
        // verbosity, and detailed verbosity is not an option (it logs all 72k discovered cases and
        // has filled the system drive), so the way to COUNT them is to flip this one word to
        // Warning, build, and grep ADN0057. That is how 80 -> 38 was measured.
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
            + "only registers layers with exactly one (the numeric type). Its state IS saved and "
            + "nothing can rebuild it, so deserialization falls back to the shape-inference path "
            + "this generator exists to replace; give the layer a single type parameter or exclude it",
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
                // EVERY constructor on a type with a base list, not just ones that already carry an
                // attribute. Requiring an attributed parameter here meant a layer with no
                // [LayerState] at all never reached Analyze -- so inference could not see it, no
                // factory was generated for it, and the ADN0053 rule that exists to reject an
                // unrestorable layer could only fire on layers that had already opted in.
                static (node, _) => node is ConstructorDeclarationSyntax
                {
                    Parent: ClassDeclarationSyntax { BaseList: not null },
                },
                static (ctx, _) => Analyze(ctx))
            .Where(static m => m is not null)
            .Select(static (m, _) => m!);

        context.RegisterSourceOutput(candidates.Collect(), Emit);
    }

    private static LayerModel? Analyze(GeneratorSyntaxContext ctx)
    {
        var syntax = (ConstructorDeclarationSyntax)ctx.Node;
        if (ctx.SemanticModel.GetDeclaredSymbol(syntax) is not IMethodSymbol ctor) return null;

        var marked = ctor.Parameters.Where(HasStateAttribute).ToList();

        // NO marked-count gate. Dropping a constructor with no [LayerState] made the whole feature
        // opt-IN a second time, below the syntax predicate: inference lives further down this
        // method, so a layer that stored every argument in a field but wrote no attribute was
        // discarded here, got no factory, and could not be cloned -- while the build stayed green,
        // because ADN0053 also never ran for it. Marked parameters are still read below; they are
        // just no longer the price of admission.

        var type = ctor.ContainingType;

        // THE HOST TYPE MUST BE ABLE TO CARRY THE GENERATED MEMBER. Analyze accepted any
        // constructor whose parameters carried [LayerState] and then emitted
        // `partial class {TypeName}` with `internal override void WriteConstructionState`.
        // A struct, a record, or a class deriving from neither LayerBase nor
        // NeuralNetworkBase produced a raw C#
        // compiler error pointing INTO generated source -- an error about code the author
        // never wrote and cannot open. Refused here with a diagnostic on the declaration
        // instead.
        if (type.TypeKind != TypeKind.Class || type.IsRecord || !DerivesFromLayerBase(type))
        {
            // Only an EXPLICIT claim is worth a diagnostic here. Now that every constructor on a
            // type with a base list is analysed, this branch also sees ordinary classes that never
            // mentioned [LayerState] -- reporting those told 7,525 authors their type "marks
            // constructor parameters [LayerState]" when it does no such thing.
            if (marked.Count == 0) return null;
        }

        // An ABSTRACT layer is never constructed, so a factory naming it emits `new Abstract<T>(...)`
        // and the generated code will not compile. This only began to matter once every
        // constructor was analysed rather than only attributed ones.
        //
        // Likewise an arity the factory cannot register: reporting it is right when the author
        // asked for state explicitly, and noise when the parameter was merely inferred.
        if (type.IsAbstract && marked.Count == 0) return null;
        if (type.TypeParameters.Length != 1 && marked.Count == 0) return null;

        if (type.TypeKind != TypeKind.Class || type.IsRecord || !DerivesFromLayerBase(type))
        {
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
            ContainingTypes = ContainingChain(type),
            TypeParameters = type.TypeParameters.Select(tp => tp.Name).ToList(),
            BaseFqn = type.ConstructedFrom.ToDisplayString(UnqualifiedGenerics),
            Location = new SourceSpan(syntax.Identifier.GetLocation()),
            IsPartial = type.DeclaringSyntaxReferences
                .Select(r => r.GetSyntax())
                .OfType<TypeDeclarationSyntax>()
                .Any(d => d.Modifiers.Any(SyntaxKind.PartialKeyword)),
            HasExplicitState = marked.Count > 0,
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

        // One restored activation per kind, because TryCreate only receives one of each: the
        // activation metadata records the function handed to base(...), and there is only ever one
        // of those. So the FIRST activation parameter of a kind takes the restored value and any
        // further one falls back to its own default -- LSTMLayer's `recurrentActivation` (sigmoid
        // gates) must not be handed `activation` (tanh cell state). A required second activation
        // slot has no value to fall back to and is reported by ADN0053.
        var scalarActivationBound = false;
        var vectorActivationBound = false;

        foreach (var p in ctor.Parameters)
        {
            var info = new ParamModel
            {
                Name = p.Name,
                TypeFqn = p.Type.ToDisplayString(FullyQualified),
                Owner = type.Name,
            };

            if (IsLayer(Unwrap(p.Type), out var childNumeric)
                || IsLayerSequence(Unwrap(p.Type), out childNumeric))
            {
                info.LayerNumeric = childNumeric?.ToDisplayString(FullyQualified);
            }

            info.AcceptsNull = p.Type.NullableAnnotation == NullableAnnotation.Annotated
                || p.NullableAnnotation == NullableAnnotation.Annotated;
            info.TraceableNumeric = TraceableNumericOf(Unwrap(p.Type))?.ToDisplayString(FullyQualified);
            info.ExpressionDelegate = ExpressionDelegateOf(Unwrap(p.Type));
            info.Settings = SettingsOf(Unwrap(p.Type)) ?? new List<SettingModel>();


            if (HasStateAttribute(p))
            {
                info.IsState = true;
                info.Key = StateKey(p) ?? p.Name;
                info.Kind = Classify(p.Type);
                if (info.Kind == ValueKind.Unsupported)
                {
                    model.Diagnostics.Add(new PendingDiagnostic(
                        UnsupportedType, SpanFor(p, model),
                        type.Name, p.Name, p.Type.ToDisplayString()));
                    return model;
                }

                info.BackingMember = FindBackingMember(type, p, out var needsConvert, out var memberIsNullable);
                info.NeedsConvert = needsConvert;
                if (info.BackingMember is null)
                {
                    model.Diagnostics.Add(new PendingDiagnostic(
                        NoBackingMember, SpanFor(p, model),
                        type.Name, p.Name, Pascal(p.Name), p.Type.ToDisplayString()));
                    return model;
                }

                // An `int?` parameter is fine when the layer stores it as a plain `int`, which is the
                // common case. A nullable BACKING member is not: LayerStateBag.Format has no nullable
                // overload and the format cannot express null.
                if (memberIsNullable)
                {
                    model.Diagnostics.Add(new PendingDiagnostic(
                        UnsupportedType, SpanFor(p, model),
                        type.Name, p.Name, p.Type.ToDisplayString()));
                    return model;
                }
            }

            // INFERRED, not merely marked. A constructor argument the layer stores in a field is
            // construction state whether or not anyone wrote the attribute -- that is what the
            // field is for. Requiring the attribute made correctness opt-IN, which is why 76 of
            // 321 layers had no factory and nothing reported it.
            //
            // The attribute is still honoured, and is still how to override the metadata key, name
            // the backing member, or claim a parameter inference would decline.
            else if (!IsActivation(p.Type, out _)
                && Classify(p.Type) is var inferredKind and not ValueKind.Unsupported
                && FindBackingMember(type, p, out var inferredConvert, out var inferredNullable) is { } inferredMember
                // A nullable backing member DECLINES inference rather than erroring, unlike the
                // marked path: an unmarked parameter that cannot round-trip is simply not state.
                && !inferredNullable)
            {
                info.IsState = true;
                info.Key = p.Name;
                info.Kind = inferredKind;
                info.BackingMember = inferredMember;
                info.NeedsConvert = inferredConvert;
            }
            else if (IsActivation(p.Type, out var vector)
                && !(vector ? vectorActivationBound : scalarActivationBound))
            {
                info.IsActivation = true;
                info.IsVectorActivation = vector;
                if (vector)
                {
                    vectorActivationBound = true;
                }
                else
                {
                    scalarActivationBound = true;
                }
            }
            else if (p.IsOptional)
            {
                info.UseDefault = true;
                // Taken from the symbol, so the emitted argument is the value the
                // constructor signature actually promises.
                info.DefaultExpression = RenderDefault(p);

                // Say so. Pinning is a real loss of construction state whenever the value was not
                // the default, and it used to happen with nothing in the build to show for it.
                model.Diagnostics.Add(new PendingDiagnostic(
                    PinnedDefault, SpanFor(p, model), type.Name, p.Name));
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
            // REPORT, THEN STOP. Emitting `partial class X` against a non-partial declaration is
            // CS0260 -- a raw compiler error pointing INTO generated source, about code the author
            // never wrote. This is the same principle the host-type check above already applies;
            // it only began to matter when models joined, because every layer was already partial
            // and so the emit-anyway path was never exercised.
            model.Diagnostics.Add(new PendingDiagnostic(NotPartial, model.Location, type.Name));
            return model;
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

    /// <summary>The member named by <c>[LayerState(Member = "...")]</c>, if any.</summary>
    private static string? StateMember(IParameterSymbol p)
    {
        var attr = p.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == StateAttribute);
        var named = attr?.NamedArguments.FirstOrDefault(n => n.Key == "Member").Value.Value as string;
        return string.IsNullOrWhiteSpace(named) ? null : named;
    }

    /// <summary>
    /// Whether the type is a layer, and if so its numeric type argument. Covers the base class,
    /// the interface, and a concrete layer named directly.
    /// </summary>
    private static bool IsLayer(ITypeSymbol type, out ITypeSymbol? numeric)
    {
        numeric = null;
        if (type is not INamedTypeSymbol named) return false;

        if (named.ConstructedFrom.Name == "ILayer" && named.TypeArguments.Length == 1)
        {
            numeric = named.TypeArguments[0];
            return true;
        }

        for (var b = named; b is not null; b = b.BaseType)
        {
            if (b.Name == "LayerBase" && b.TypeArguments.Length == 1)
            {
                numeric = b.TypeArguments[0];
                return true;
            }
        }

        foreach (var i in named.AllInterfaces)
        {
            if (i.Name == "ILayer" && i.TypeArguments.Length == 1)
            {
                numeric = i.TypeArguments[0];
                return true;
            }
        }

        return false;
    }

    /// <summary>Whether the type is a sequence of layers, and if so their numeric type.</summary>
    private static bool IsLayerSequence(ITypeSymbol type, out ITypeSymbol? numeric)
    {
        numeric = null;

        if (type is IArrayTypeSymbol { Rank: 1 } arr) return IsLayer(arr.ElementType, out numeric);
        if (type is not INamedTypeSymbol named) return false;

        if (named.TypeArguments.Length == 1
            && named.Name is "List" or "IList" or "IEnumerable" or "IReadOnlyList"
                or "IReadOnlyCollection" or "ICollection")
            return IsLayer(named.TypeArguments[0], out numeric);

        return false;
    }
    /// <summary>
    /// The member holds the layer the parameter was given: both are layers over the same numeric
    /// type, AND the parameter is assignable to the member's declared type.
    /// </summary>
    /// <remarks>
    /// Assignability is what keeps this from being too loose. "Both are layers over T" alone would
    /// let a GroupedQueryAttentionLayer&lt;float&gt; parameter bind to a
    /// MultiHeadAttentionLayer&lt;float&gt; field -- a DIFFERENT child, silently read back in place
    /// of the one the constructor was given. The case this exists for is the reverse and is safe:
    /// a LayerBase&lt;T&gt; parameter stored in an ILayer&lt;T&gt; field is the same object seen
    /// through a wider type.
    /// </remarks>
    private static bool IsSameLayer(ITypeSymbol member, ITypeSymbol parameter)
        => IsLayer(Unwrap(parameter), out var pn)
           && IsLayer(Unwrap(member), out var mn)
           && pn is not null && mn is not null
           && SymbolEqualityComparer.Default.Equals(pn, mn)
           && IsAssignableTo(Unwrap(parameter), Unwrap(member));

    /// <summary>Both types are sequences of layers over the same numeric type.</summary>
    private static bool IsSameLayerSequence(ITypeSymbol member, ITypeSymbol parameter)
        => IsLayerSequence(Unwrap(parameter), out var pn)
           && IsLayerSequence(Unwrap(member), out var mn)
           && pn is not null && mn is not null
           && SymbolEqualityComparer.Default.Equals(pn, mn);

    /// <summary>Whether a value of <paramref name="from"/> can be stored in <paramref name="to"/>.</summary>
    private static bool IsAssignableTo(ITypeSymbol from, ITypeSymbol to)
    {
        if (SameType(from, to)) return true;

        for (var b = (from as INamedTypeSymbol)?.BaseType; b is not null; b = b.BaseType)
        {
            if (SameType(b, to)) return true;
        }

        return from is INamedTypeSymbol named && named.AllInterfaces.Any(i => SameType(i, to));
    }

    /// <summary>
    /// For <c>Func&lt;ComputationNode&lt;X&gt;, ComputationNode&lt;X&gt;&gt;</c>, returns X.
    /// </summary>
    private static ITypeSymbol? TraceableNumericOf(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol { TypeKind: TypeKind.Delegate } named) return null;

        var invoke = named.DelegateInvokeMethod;
        if (invoke is null || invoke.Parameters.Length != 1) return null;

        var argument = Node(invoke.Parameters[0].Type);
        var result = Node(invoke.ReturnType);
        return argument is not null && SymbolEqualityComparer.Default.Equals(argument, result) ? argument : null;

        static ITypeSymbol? Node(ITypeSymbol t)
            => t is INamedTypeSymbol { Name: "ComputationNode", TypeArguments.Length: 1 } n ? n.TypeArguments[0] : null;
    }

    /// <summary>For <c>Expression&lt;TDelegate&gt;</c>, returns TDelegate as C# source.</summary>
    private static string? ExpressionDelegateOf(ITypeSymbol type)
        => type is INamedTypeSymbol { Name: "Expression", TypeArguments.Length: 1 } expression
           && expression.TypeArguments[0].TypeKind == TypeKind.Delegate
            ? expression.TypeArguments[0].ToDisplayString(FullyQualified)
            : null;

    /// <summary>
    /// The settable public properties of a plain settings object, when it can be constructed
    /// without arguments and every one of them is a value the metadata can carry. Anything else is
    /// not a settings object and keeps reporting ADN0053 rather than round-tripping partially.
    /// </summary>
    private static List<SettingModel>? SettingsOf(ITypeSymbol type)
    {
        if (type is not INamedTypeSymbol { TypeKind: TypeKind.Class } named || named.IsAbstract) return null;

        if (!named.InstanceConstructors.Any(c => c.Parameters.Length == 0
            && c.DeclaredAccessibility is Accessibility.Public or Accessibility.Internal))
            return null;

        var settings = new List<SettingModel>();
        foreach (var property in named.GetMembers().OfType<IPropertySymbol>())
        {
            if (property.IsStatic || property.IsIndexer) continue;
            if (property.DeclaredAccessibility != Accessibility.Public) continue;
            if (property.GetMethod is null || property.SetMethod is null) continue;
            if (property.SetMethod.DeclaredAccessibility != Accessibility.Public) continue;

            var kind = Classify(property.Type);
            if (kind is ValueKind.Unsupported or ValueKind.Settings or ValueKind.Layer
                or ValueKind.LayerList or ValueKind.Delegate or ValueKind.Expression or ValueKind.Component)
                return null;

            settings.Add(new SettingModel
            {
                Name = property.Name,
                Kind = kind,
                TypeFqn = property.Type.ToDisplayString(FullyQualified),
            });
        }

        return settings.Count > 0 ? settings : null;
    }



    private static string? StateKey(IParameterSymbol p)
    {
        var attr = p.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == StateAttribute);
        var named = attr?.NamedArguments.FirstOrDefault(n => n.Key == "Key").Value.Value as string;
        return string.IsNullOrWhiteSpace(named) ? null : named;
    }

    private static ValueKind Classify(ITypeSymbol type)
    {
        type = Unwrap(type);

        if (type.TypeKind == TypeKind.Enum) return ValueKind.Enum;

        // A pluggable strategy: record which implementation was used and rebuild that one.
        // Before the interface check: ILayer<T> would otherwise be claimed as a Component, which
        // rebuilds by parameterless Activator.CreateInstance -- losing every argument the child
        // was built with, silently and without failing.
        // A delegate and an expression tree each have their own description path, and an
        // Expression<T> is a class, so both are decided before the checks below.
        if (type.TypeKind == TypeKind.Delegate) return ValueKind.Delegate;
        if (ExpressionDelegateOf(type) is not null) return ValueKind.Expression;

        if (IsLayer(type, out _)) return ValueKind.Layer;
        if (IsLayerSequence(type, out _)) return ValueKind.LayerList;

        // A plain configuration object: no behaviour, just settable values. Its properties are
        // construction state as much as a scalar is, and rebuilding it by parameterless
        // construction alone would silently restore every one of them to its default.
        if (SettingsOf(type) is not null) return ValueKind.Settings;

        if (type.TypeKind == TypeKind.Interface) return ValueKind.Component;

        if (type is IArrayTypeSymbol { Rank: 1 } arr)
        {
            // A shape list is as much construction state as a scalar dimension is, and rebuilding
            // one from the declared input shape is exactly the inference this generator removes.
            if (arr.ElementType.SpecialType == SpecialType.System_Int32) return ValueKind.Int32Array;
            if (arr.ElementType.SpecialType == SpecialType.System_Double) return ValueKind.DoubleArray;
            if (arr.ElementType.SpecialType == SpecialType.System_Boolean) return ValueKind.BooleanArray;
            if (arr.ElementType.SpecialType == SpecialType.System_String) return ValueKind.StringArray;

            // int[][]: the per-input shapes a merge layer (Add, Concatenate, Multiply) was built
            // with, where the outer length is the number of inputs.
            if (arr.ElementType is IArrayTypeSymbol { Rank: 1 } inner
                && inner.ElementType.SpecialType == SpecialType.System_Int32)
                return ValueKind.Int32Jagged;

            return ValueKind.Unsupported;
        }

        // A graph layer's schema: which node and edge types it was built over. A layer built over
        // different node types is a different layer, so this is construction state.
        if (IsMap(type, SpecialType.System_Int32)) return ValueKind.StringInt32Map;
        if (IsPairMap(type)) return ValueKind.StringPairMap;

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

    private static string? FindBackingMember(INamedTypeSymbol type, IParameterSymbol p, out bool needsConvert, out bool memberIsNullable)
    {
        needsConvert = false;
        memberIsNullable = false;
        // [LayerState(Member = "...")] wins outright. Without it the attribute could not rescue a
        // layer that stored the argument under any other name, because ADN0051 applied the same
        // five-name rule to marked parameters too.
        var named = StateMember(p);
        var candidates = named is not null
            ? new[] { named }
            : new[] { p.Name, "_" + p.Name, "m_" + p.Name, Pascal(p.Name), "_" + Pascal(p.Name) };

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
                        // A child layer is recorded by walking the object, so the member only has
                        // to BE that layer, not to be declared as the same type. LoRAAdapterBase
                        // keeps its child as ILayer<T> while a shape-resolving overload takes
                        // LayerBase<T>; the stored object is the same one either way.
                        case IFieldSymbol lf when IsSameLayer(lf.Type, p.Type):
                            return lf.Name;
                        case IPropertySymbol { GetMethod: not null } lp when IsSameLayer(lp.Type, p.Type):
                            return lp.Name;

                        // A sequence of children is written by enumerating it and read back as a
                        // fresh list, so the member only has to hold layers of the same numeric
                        // type -- List<ILayer<T>> backs an IEnumerable<ILayer<T>> parameter.
                        case IFieldSymbol sf when IsSameLayerSequence(sf.Type, p.Type):
                            return sf.Name;
                        case IPropertySymbol { GetMethod: not null } sp when IsSameLayerSequence(sp.Type, p.Type):
                            return sp.Name;

                        case IFieldSymbol f when SameType(f.Type, p.Type):
                            memberIsNullable = IsNullableValueType(f.Type);
                            return f.Name;
                        case IPropertySymbol { GetMethod: not null } prop when SameType(prop.Type, p.Type):
                            memberIsNullable = IsNullableValueType(prop.Type);
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

        return null;
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

    /// <summary>A <c>Dictionary&lt;string, TValue&gt;</c> over the given value type.</summary>
    private static bool IsMap(ITypeSymbol type, SpecialType value)
        => type is INamedTypeSymbol { Name: "Dictionary", TypeArguments.Length: 2 } map
           && map.TypeArguments[0].SpecialType == SpecialType.System_String
           && map.TypeArguments[1].SpecialType == value;

    /// <summary>A <c>Dictionary&lt;string, (string, string)&gt;</c>, however the tuple is named.</summary>
    private static bool IsPairMap(ITypeSymbol type)
        => type is INamedTypeSymbol { Name: "Dictionary", TypeArguments.Length: 2 } map
           && map.TypeArguments[0].SpecialType == SpecialType.System_String
           && map.TypeArguments[1] is INamedTypeSymbol { IsTupleType: true } tuple
           && tuple.TupleElements.Length == 2
           && tuple.TupleElements.All(e => e.Type.SpecialType == SpecialType.System_String);

    private static bool SameType(ITypeSymbol a, ITypeSymbol b)
        => Unwrap(a).ToDisplayString(FullyQualified) == Unwrap(b).ToDisplayString(FullyQualified);

    /// <summary>True for <c>Nullable&lt;T&gt;</c>, whose null the metadata format cannot represent.</summary>
    private static bool IsNullableValueType(ITypeSymbol type)
        => type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T };

    /// <summary>Strips <c>Nullable&lt;T&gt;</c> so an <c>int?</c> parameter matches an <c>int</c> field.</summary>
    private static ITypeSymbol Unwrap(ITypeSymbol type)
        => type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } n
            ? n.TypeArguments[0]
            : type;

    private static string Pascal(string name)
        => name.Length == 0 ? name : char.ToUpperInvariant(name[0]) + name.Substring(1);

    private static void Emit(SourceProductionContext spc, ImmutableArray<LayerModel> models)
    {
        foreach (var d in models.SelectMany(m => m.Diagnostics).Select(d => d.ToDiagnostic()))
        {
            spc.ReportDiagnostic(d);
        }

        // One constructor per type: if a layer annotates several, the first by source order wins so
        // the generated factory is deterministic.
        var byType = models
            .Where(m => m.IsValid)
            .GroupBy(TypeKey, System.StringComparer.Ordinal)
            // DETERMINISTIC. `g.First()` took whatever order Collect() yielded, and Roslyn
            // does not document that collected results keep source order -- for a type split
            // across partial files the per-file order is undefined, so a layer annotating two
            // constructors could generate a different factory between builds. Ordering on the
            // constructor's own location also makes "first by source order" true.
            .Select(g => g
                // Recoverable state first. Ranking by state-parameter count alone picked
                // LambdaLayer's two-opaque-delegate constructor over its traceable one: an opaque
                // delegate is the parameter LEAST likely to survive a save, so the constructor that
                // rebuilt worst scored highest. Source order still breaks the remaining ties, so the
                // choice stays deterministic across partial files.
                .OrderByDescending(m => m.Parameters.Count(
                    p => p.TraceableNumeric is not null || p.ExpressionDelegate is not null))
                .ThenByDescending(m => m.Parameters.Count(p => p.IsState))
                .ThenBy(m => m.Location.FilePath, System.StringComparer.Ordinal)
                .OrderBy(m => m.Location.FilePath, System.StringComparer.Ordinal)
                .ThenBy(m => m.Location.Start)
                .First())
            .OrderBy(TypeKey, System.StringComparer.Ordinal)
            .ToList();

        foreach (var model in byType)
        {
            // UNIQUE OR THE BUILD THROWS. Built from the SIMPLE name (which also drops
            // arity) while the grouping key is namespace-qualified, so two annotated layers
            // both named DenseLayer in different namespaces emitted the same file name and
            // AddSource threw on the duplicate. Derived from the same key the grouping uses.
            // A LAYER THAT SAVES BUT CANNOT BE REBUILT IS REPORTED, not skipped in silence.
            // The factory registers only single-type-parameter layers, so a non-generic layer
            // or one declared Foo<T, TState> wrote its [LayerState] values to metadata and had
            // no TryCreate entry: deserialization silently fell back to the shape-inference
            // path this generator was built to replace, which is the -1 bug it fixes.
            if (model.TypeParameters.Count != 1)
            {
                // Only an EXPLICIT claim is reported. With state inferred, this also sees
                // non-generic layers that never asked to participate; writing a writer they
                // can never pair with a factory, then reporting it, is noise the author
                // cannot act on. An inferred parameter on an unsupported arity is simply not
                // construction state.
                if (!model.HasExplicitState) continue;

                spc.ReportDiagnostic(Diagnostic.Create(
                    UnsupportedArity, model.Location.ToLocation(), model.TypeName, model.TypeParameters.Count));
            }

            spc.AddSource($"{HintName(model)}.LayerState.g.cs", SourceText(EmitWriter(model)));
        }

        spc.AddSource("GeneratedLayerFactories.g.cs", SourceText(EmitFactories(byType)));
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
        sb.AppendLine("    internal override void WriteConstructionState(global::System.Collections.Generic.Dictionary<string, string> __metadata)");
        sb.AppendLine("    {");
        sb.AppendLine("        base.WriteConstructionState(__metadata);");
        foreach (var p in model.Parameters.Where(p => p.IsState))
        {
            // A child layer records its OWN construction state under a nested key namespace, so a
            // composite layer is rebuildable exactly as far as its children are. Writing only the
            // child's type name would rebuild it by parameterless construction, silently dropping
            // every argument it was given.
            if (p.Kind == ValueKind.Layer)
            {
                var num = p.LayerNumeric ?? "T";
                sb.AppendLine($"        global::AiDotNet.Serialization.LayerStateBag.WriteNested<{num}>("
                    + $"__metadata, \"{p.Key}\", this.{p.BackingMember} as global::AiDotNet.NeuralNetworks.Layers.LayerBase<{num}>);");
                continue;
            }

            if (p.Kind == ValueKind.LayerList)
            {
                var num = p.LayerNumeric ?? "T";
                sb.AppendLine($"        global::AiDotNet.Serialization.LayerStateBag.WriteNestedRange<{num}>("
                    + $"__metadata, \"{p.Key}\", this.{p.BackingMember});");
                continue;
            }

            // A delegate is described rather than written down -- a traced graph, an expression
            // tree, or a named method by reference. Never marshalled code: Keras writes the Lambda
            // layer's bytecode and made loading a model arbitrary code execution (CVE-2025-9906).
            if (p.Kind == ValueKind.Delegate)
            {
                sb.AppendLine(p.TraceableNumeric is null
                    ? $"        __metadata[\"{p.Key}\"] = "
                      + $"global::AiDotNet.Serialization.DelegateState.Save(this.{p.BackingMember});"
                    : $"        __metadata[\"{p.Key}\"] = "
                      + $"global::AiDotNet.Serialization.DelegateState.SaveTraceable<{p.TraceableNumeric}>("
                      + $"this.{p.BackingMember}, this.GetInputShape());");
                continue;
            }

            if (p.Kind == ValueKind.Expression)
            {
                sb.AppendLine($"        __metadata[\"{p.Key}\"] = "
                    + $"global::AiDotNet.Serialization.ExpressionState.Save(this.{p.BackingMember});");
                continue;
            }

            // A settings object is written property by property under its own key namespace, so it
            // rebuilds through an object initializer the compiler checks rather than by reflection.
            if (p.Kind == ValueKind.Settings)
            {
                sb.AppendLine($"        if (this.{p.BackingMember} is not null)");
                sb.AppendLine("        {");
                sb.AppendLine($"            __metadata[\"{p.Key}.$set\"] = \"true\";");
                foreach (var s in p.Settings)
                {
                    sb.AppendLine($"            __metadata[\"{p.Key}.{s.Name}\"] = "
                        + $"global::AiDotNet.Serialization.LayerStateBag.Format(this.{p.BackingMember}.{s.Name});");
                }
                sb.AppendLine("        }");
                continue;
            }

            if (p.Kind == ValueKind.Component)
            {
                sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.FormatType(this.{p.BackingMember});");
                continue;
            }

            var read = p.NeedsConvert
                ? ConvertExpression(p, model.TypeParameters.Count > 0 ? model.TypeParameters[0] : "T")
                : $"this.{p.BackingMember}";
            sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.Format({read});");
        }
        sb.AppendLine("    }");
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
        sb.AppendLine("internal static class GeneratedLayerFactories<T>");
        sb.AppendLine("{");
        sb.AppendLine("    /// <summary>Number of layer types with generated factories.</summary>");
        sb.AppendLine($"    internal const int Count = {models.Count(m => m.TypeParameters.Count == 1)};");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Attempts to rebuild a layer of the given open generic type.</summary>");
        sb.AppendLine("    /// <param name=\"genericDefinition\">The layer's open generic type, e.g. <c>typeof(DenseLayer&lt;&gt;)</c>.</param>");
        sb.AppendLine("    /// <param name=\"state\">The layer's saved metadata.</param>");
        sb.AppendLine("    /// <param name=\"scalarActivation\">Restored scalar activation, when the constructor takes one.</param>");
        sb.AppendLine("    /// <param name=\"vectorActivation\">Restored vector activation, when the constructor takes one.</param>");
        sb.AppendLine("    /// <param name=\"layer\">The rebuilt layer.</param>");
        sb.AppendLine("    /// <returns><c>true</c> when a factory exists for the type.</returns>");
        // The recursion lives in generated code because only generated code can call TryCreate;
        // putting it in LayerStateBag would make a hand-written type depend on generated output.
        sb.AppendLine("    /// <summary>Rebuilds the child layer saved under <paramref name=\"key\"/>.</summary>");
        sb.AppendLine("    /// <param name=\"state\">The parent's saved metadata.</param>");
        sb.AppendLine("    /// <param name=\"key\">The constructor parameter the child was passed as.</param>");
        sb.AppendLine("    /// <returns>The rebuilt child, or <c>null</c> when none was saved.</returns>");
        sb.AppendLine("    internal static object? RebuildNested(global::AiDotNet.Serialization.LayerStateBag state, string key)");
        sb.AppendLine("    {");
        sb.AppendLine("        var type = state.NestedType(key);");
        sb.AppendLine("        if (type is null) return null;");
        sb.AppendLine();
        sb.AppendLine("        var nested = state.Nested(key);");
        sb.AppendLine("        var definition = type.IsGenericType ? type.GetGenericTypeDefinition() : type;");
        sb.AppendLine("        return TryCreate(");
        sb.AppendLine("            definition,");
        sb.AppendLine("            nested,");
        sb.AppendLine("            nested.Component<global::AiDotNet.Interfaces.IActivationFunction<T>>(\"ScalarActivationType\"),");
        sb.AppendLine("            nested.Component<global::AiDotNet.Interfaces.IVectorActivationFunction<T>>(\"VectorActivationType\"),");
        sb.AppendLine("            out var child) ? child : null;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Rebuilds the child layers saved under <paramref name=\"key\"/>, in order.</summary>");
        sb.AppendLine("    /// <typeparam name=\"TChild\">The element type the constructor takes.</typeparam>");
        sb.AppendLine("    /// <param name=\"state\">The parent's saved metadata.</param>");
        sb.AppendLine("    /// <param name=\"key\">The constructor parameter the children were passed as.</param>");
        sb.AppendLine("    /// <returns>The rebuilt children.</returns>");
        sb.AppendLine("    internal static global::System.Collections.Generic.List<TChild> RebuildNestedRange<TChild>(");
        sb.AppendLine("        global::AiDotNet.Serialization.LayerStateBag state, string key) where TChild : class");
        sb.AppendLine("    {");
        sb.AppendLine("        var count = state.NestedCount(key);");
        sb.AppendLine("        var items = new global::System.Collections.Generic.List<TChild>(count < 0 ? 0 : count);");
        sb.AppendLine("        for (var i = 0; i < count; i++)");
        sb.AppendLine("        {");
        sb.AppendLine("            // A child that will not rebuild is dropped rather than left as a null element:");
        sb.AppendLine("            // a null layer in an expert list faults on the first forward pass, far from here.");
        sb.AppendLine("            if (RebuildNested(state, key + \".\" + i.ToString(global::System.Globalization.CultureInfo.InvariantCulture)) is TChild c)");
        sb.AppendLine("                items.Add(c);");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        return items;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    internal static bool TryCreate(");
        sb.AppendLine("        global::System.Type genericDefinition,");
        sb.AppendLine("        global::AiDotNet.Serialization.LayerStateBag state,");
        sb.AppendLine("        object? scalarActivation,");
        sb.AppendLine("        object? vectorActivation,");
        sb.AppendLine("        out object layer)");
        sb.AppendLine("    {");

        foreach (var model in models.Where(m => m.TypeParameters.Count == 1))
        {
            var args = string.Join(", ", model.Parameters.Select(p => Argument(p)));
            var closed = model.ClosedFqn;
            var required = model.Parameters
                .Where(p => p.IsState)
                // Nested state is not stored under the bare parameter name, so the "is this state
                // even mine" guard has to look for the key that is actually written.
                .Select(p => "\"" + p.Key + p.Kind switch
                {
                    ValueKind.Layer => ".$type",
                    ValueKind.LayerList => ".count",
                    _ => string.Empty,
                } + "\"")
                .ToList();

            sb.AppendLine($"        if (genericDefinition == typeof({model.OpenGenericFqn}))");
            sb.AppendLine("        {");
            if (required.Count > 0)
            {
                sb.AppendLine($"            if (!state.HasAll({string.Join(", ", required)}))");
                sb.AppendLine("            {");
                sb.AppendLine("                layer = null!;");
                sb.AppendLine("                return false;");
                sb.AppendLine("            }");
                sb.AppendLine();
            }
            sb.AppendLine($"            layer = new {closed}({args});");
            sb.AppendLine("            return true;");
            sb.AppendLine("        }");
            sb.AppendLine();
        }

        sb.AppendLine("        layer = null!;");
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
                ? "global::AiDotNet.Interfaces.IVectorActivationFunction<T>"
                : "global::AiDotNet.Interfaces.IActivationFunction<T>";
            var source = p.IsVectorActivation ? "vectorActivation" : "scalarActivation";
            // `as` yields null both when no activation was saved and when the wrong kind was. A
            // parameter that does not accept null is told which, rather than handed the null.
            return p.AcceptsNull
                ? $"{p.Name}: {source} as {iface}"
                : $"{p.Name}: global::AiDotNet.Serialization.LayerStateBag.RequireActivation<{iface}>"
                  + $"({source}, \"{p.Name}\", \"{p.Owner}\")";
        }

        // THE PARAMETER'S DEFAULT, NOT THE TYPE'S. Falls back to `default!` only when the
        // declaration genuinely has no value to render.
        // `null` does not convert to an unconstrained `T?`, so a null default is emitted as
        // `default!` -- the same value, in a form that compiles for a type parameter.
        if (p.UseDefault)
        {
            var rendered = p.DefaultExpression is null or "null" ? "default!" : p.DefaultExpression;
            return $"{p.Name}: {rendered}";
        }
        var read = ReadExpression(p);
        return $"{p.Name}: {read}";
    }

    /// <summary>
    /// Rebuilds a sequence of child layers into the collection type the constructor declared.
    /// </summary>
    /// <remarks>
    /// The children come back as a <c>List&lt;TChild&gt;</c>; an array parameter needs ToArray() and
    /// an IEnumerable/IReadOnlyList parameter takes the list as it is.
    /// </remarks>
    /// <summary>
    /// Rebuilds a settings object through an object initializer, or leaves it null when none was
    /// saved. Compile-checked property assignments rather than reflection, so a renamed property
    /// breaks the build instead of silently restoring a default.
    /// </summary>
    private static string SettingsArgument(ParamModel p)
    {
        var assignments = string.Join(", ", p.Settings.Select(s => $"{s.Name} = {s.Read(p.Key)}"));
        var created = $"new {p.TypeFqn.TrimEnd('?')} {{ {assignments} }}";
        return p.AcceptsNull ? $"state.Has(\"{p.Key}.$set\") ? {created} : null" : created;
    }

    private static string LayerListArgument(ParamModel p)
    {
        var declared = p.TypeFqn.TrimEnd('?');
        var rebuilt = $"RebuildNestedRange<{ElementFqn(declared)}>(state, \"{p.Key}\")";
        return declared.EndsWith("[]", System.StringComparison.Ordinal) ? $"{rebuilt}.ToArray()" : rebuilt;
    }

    /// <summary>The element type of a declared array or single-argument generic collection.</summary>
    private static string ElementFqn(string declared)
    {
        if (declared.EndsWith("[]", System.StringComparison.Ordinal))
            return declared.Substring(0, declared.Length - 2);

        var open = declared.IndexOf('<');
        return open < 0 ? declared : declared.Substring(open + 1, declared.Length - open - 2);
    }

    private static string ReadExpression(ParamModel p)
    {
        var read = p.Kind switch
        {
            ValueKind.Int32 => $"state.Int32(\"{p.Key}\")",
            ValueKind.Int64 => $"state.Int64(\"{p.Key}\")",
            ValueKind.Double => $"state.Double(\"{p.Key}\")",
            ValueKind.Single => $"state.Single(\"{p.Key}\")",
            ValueKind.Boolean => $"state.Boolean(\"{p.Key}\")",
            ValueKind.String => $"state.String(\"{p.Key}\")",
            ValueKind.Int32Array => $"state.Int32Array(\"{p.Key}\")",
            ValueKind.DoubleArray => $"state.DoubleArray(\"{p.Key}\")",
            ValueKind.BooleanArray => $"state.BooleanArray(\"{p.Key}\")",
            ValueKind.StringArray => $"state.StringArray(\"{p.Key}\")",
            ValueKind.Int32Jagged => $"state.Int32Jagged(\"{p.Key}\")",
            ValueKind.StringInt32Map => $"state.StringInt32Map(\"{p.Key}\")",
            ValueKind.StringPairMap => $"state.StringPairMap(\"{p.Key}\")",
            ValueKind.Layer => $"({p.TypeFqn.TrimEnd('?')})RebuildNested(state, \"{p.Key}\")!",
            ValueKind.LayerList => LayerListArgument(p),
            ValueKind.Enum => $"state.Enum<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            ValueKind.Delegate => $"global::AiDotNet.Serialization.DelegateState.Load<{p.TypeFqn.TrimEnd('?')}>("
                + $"state.String(\"{p.Key}\"), \"{p.Owner}\", \"{p.Key}\")",
            ValueKind.Expression => $"global::AiDotNet.Serialization.ExpressionState.Load<{p.ExpressionDelegate}>("
                + $"state.String(\"{p.Key}\"), \"{p.Owner}\", \"{p.Key}\")",
            ValueKind.Settings => SettingsArgument(p),

            // A parameter that does not accept null must not be handed one: Component() returns
            // null both when nothing was saved and when the saved type will not load.
            ValueKind.Component => p.AcceptsNull
                ? $"state.Component<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")"
                : $"state.ComponentRequired<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            _ => "default!",
        };

        return read;
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
        Int32Jagged,
        StringInt32Map,
        StringPairMap,
        Layer,
        LayerList,
        Delegate,
        Expression,
        Settings,
        Component,
    }

    /// <summary>The symbol's own span when it has one, else the model's.</summary>
    private static SourceSpan SpanFor(ISymbol symbol, LayerModel model)
    {
        var loc = symbol.Locations.FirstOrDefault();
        return loc is null || loc == Location.None ? model.Location : new SourceSpan(loc);
    }

    /// <summary>True when the type derives from AiDotNet's LayerBase.</summary>
    private static bool DerivesFromLayerBase(INamedTypeSymbol type)
        => DerivesFrom(type, "AiDotNet.NeuralNetworks.Layers.LayerBase");

    /// <summary>True when the type derives from AiDotNet's NeuralNetworkBase.</summary>
    private static bool DerivesFromNeuralNetworkBase(INamedTypeSymbol type)
        => DerivesFrom(type, "AiDotNet.NeuralNetworks.NeuralNetworkBase");

    /// <summary>
    /// True when the type has a base that declares the virtual this generator overrides.
    /// </summary>
    /// <remarks>
    /// NOT CURRENTLY USED AS A GATE, and the reason is worth recording. Widening the gate to
    /// NeuralNetworkBase compiles, but every model then fails ADN0053 on its `architecture`
    /// parameter: this generator rebuilds from a Dictionary&lt;string, string&gt;, so a constructor
    /// argument has to survive a round trip through text. Layers take scalars, enums and child
    /// layers, all of which do. A model takes a NeuralNetworkArchitecture&lt;T&gt; -- a graph, which
    /// does not.
    ///
    /// Cloning does not actually need it to. A clone has the LIVE source object in hand, so rich
    /// arguments can be passed straight across (or cloned in their own right) rather than
    /// serialized and rebuilt. Deserialization is the case that needs text, and it already has
    /// SerializeNetworkSpecificData. So the model factory wants a different signature from the
    /// layer one -- source-aware, not metadata-only -- which is why models do not simply join this
    /// gate.
    /// </remarks>
    /// <remarks>
    /// Both LayerBase and NeuralNetworkBase declare
    /// <c>internal virtual void WriteConstructionState(Dictionary&lt;string, string&gt;)</c>, so the
    /// generated override compiles against either. Gating on LayerBase alone was why models had no
    /// generated factory and therefore could not be rebuilt — the same shape of omission this
    /// generator's own history records, where gating on an already-present attribute meant the
    /// types that most needed a diagnostic were the ones excluded from receiving one.
    /// </remarks>
    private static bool DerivesFromStatefulBase(INamedTypeSymbol type)
        => DerivesFromLayerBase(type) || DerivesFromNeuralNetworkBase(type);

    private static bool DerivesFrom(INamedTypeSymbol type, string baseDisplayName)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.ConstructedFrom.ToDisplayString(UnqualifiedGenerics) == baseDisplayName)
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

        if (v is null) return p.Type.IsValueType ? "default" : "null";

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
            float f => f.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "f",
            double d => d.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "d",
            decimal m => m.ToString(System.Globalization.CultureInfo.InvariantCulture) + "m",
            long l => l.ToString(System.Globalization.CultureInfo.InvariantCulture) + "L",
            _ => System.Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture),
        };
    }

    /// <summary>One property of a settings object, and how a rebuild reads it back.</summary>
    private sealed class SettingModel
    {
        public string Name = string.Empty;
        public ValueKind Kind;
        public string TypeFqn = string.Empty;

        /// <summary>The accessor call that reads this property out of the saved state.</summary>
        public string Read(string parameterKey)
        {
            var key = parameterKey + "." + Name;
            return Kind switch
            {
                ValueKind.Int32 => $"state.Int32(\"{key}\")",
                ValueKind.Int64 => $"state.Int64(\"{key}\")",
                ValueKind.Double => $"state.Double(\"{key}\")",
                ValueKind.Single => $"state.Single(\"{key}\")",
                ValueKind.Boolean => $"state.Boolean(\"{key}\")",
                ValueKind.String => $"state.String(\"{key}\")",
                ValueKind.Int32Array => $"state.Int32Array(\"{key}\")",
                ValueKind.DoubleArray => $"state.DoubleArray(\"{key}\")",
                ValueKind.BooleanArray => $"state.BooleanArray(\"{key}\")",
                ValueKind.StringArray => $"state.StringArray(\"{key}\")",
                ValueKind.Int32Jagged => $"state.Int32Jagged(\"{key}\")",
                ValueKind.StringInt32Map => $"state.StringInt32Map(\"{key}\")",
                ValueKind.StringPairMap => $"state.StringPairMap(\"{key}\")",
                _ => $"state.Enum<{TypeFqn.TrimEnd('?')}>(\"{key}\")",
            };
        }
    }

    private sealed class ParamModel
    {
        public string Name = string.Empty;
        public string TypeFqn = string.Empty;
        public string Key = string.Empty;
        public string? BackingMember;
        public bool IsState;
        public bool IsActivation;
        public bool IsVectorActivation;
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
        public ValueKind Kind;

        /// <summary>
        /// For a child layer or a sequence of them, the numeric type argument to write and rebuild
        /// through. Usually the parent's own T, but a layer may name a closed child type.
        /// </summary>
        public string? LayerNumeric;

        /// <summary>The layer that declares this parameter, named in any rebuild failure.</summary>
        public string Owner = string.Empty;

        /// <summary>Whether the parameter itself accepts a null argument.</summary>
        public bool AcceptsNull;

        /// <summary>
        /// For a <c>Func&lt;ComputationNode&lt;X&gt;, ComputationNode&lt;X&gt;&gt;</c>, the X. Such a
        /// delegate is described by running it once and recording what it did, which is the only
        /// description that survives a closure.
        /// </summary>
        public string? TraceableNumeric;

        /// <summary>For an <c>Expression&lt;TDelegate&gt;</c>, the TDelegate.</summary>
        public string? ExpressionDelegate;

        /// <summary>For a settings object, the properties that carry its state.</summary>
        public List<SettingModel> Settings = new();
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

        /// <summary>Whether any parameter was EXPLICITLY marked [LayerState].</summary>
        public bool HasExplicitState;
        public bool IsValid;

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
