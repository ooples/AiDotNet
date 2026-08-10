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
/// For each annotated constructor this emits (a) a <c>GetMetadata</c> override on the layer writing
/// every marked parameter, and (b) an entry in a central factory keyed by open generic type that
/// reconstructs the layer by calling that same constructor. Because both halves are derived from one
/// declaration, they cannot drift apart — which is the failure mode Keras's hand-written
/// <c>get_config</c>/<c>from_config</c> pairs are subject to and cannot detect.
/// </para>
/// </remarks>
[Generator]
public class LayerStateGenerator : IIncrementalGenerator
{
    private const string StateAttribute = "AiDotNet.Attributes.LayerStateAttribute";

    /// <summary>Mirrors <c>LayerStateBag.TypeKey</c>; the generator cannot reference that assembly.</summary>
    private const string NestedTypeKey = "$type";

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
        "'{0}' marks parameter '{1}' of type '{2}' as [LayerState], but only integral, floating-point, bool, string, enum and int[] values can round-trip through layer metadata",
        "AiDotNet.Serialization", DiagnosticSeverity.Error, true);

    private static readonly DiagnosticDescriptor Unsuppliable = new(
        "ADN0053",
        "Required constructor parameter cannot be restored",
        "'{0}' cannot be rebuilt: parameter '{1}' of type '{2}' is required but is neither marked [LayerState], an activation function, nor optional; mark it, give it a default, or exclude this constructor",
        "AiDotNet.Serialization", DiagnosticSeverity.Error, true);

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
                // Every constructor on a type with a base list. Requiring an attributed parameter
                // here meant a layer with no [LayerState] at all never reached Analyze, so the
                // ADN0053 rule that is supposed to reject an unrestorable layer could only fire on
                // layers that had already opted in.
                static (node, _) => node is ConstructorDeclarationSyntax { Parent: ClassDeclarationSyntax { BaseList: not null } },
                static (ctx, _) => Analyze(ctx))
            .Where(static m => m is not null)
            .Select(static (m, _) => m!);

        context.RegisterSourceOutput(candidates.Collect(), Emit);
    }

    /// <summary>Determines whether a type is a layer, by base chain.</summary>
    private static bool DerivesFromLayerBase(INamedTypeSymbol type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.Name == "LayerBase") return true;
        }

        return false;
    }

    private static LayerModel? Analyze(GeneratorSyntaxContext ctx)
    {
        var syntax = (ConstructorDeclarationSyntax)ctx.Node;
        if (ctx.SemanticModel.GetDeclaredSymbol(syntax) is not IMethodSymbol ctor) return null;

        var type = ctor.ContainingType;
        if (type.IsAbstract || !DerivesFromLayerBase(type)) return null;

        // A private constructor is not how the layer is built, so it is not the one to rebuild it.
        if (ctor.DeclaredAccessibility is not (Accessibility.Public or Accessibility.Internal)) return null;

        var model = new LayerModel
        {
            Namespace = type.ContainingNamespace.IsGlobalNamespace
                ? null
                : type.ContainingNamespace.ToDisplayString(),
            TypeName = type.Name,
            TypeParameters = type.TypeParameters.Select(tp => tp.Name).ToList(),
            BaseFqn = type.ConstructedFrom.ToDisplayString(UnqualifiedGenerics),
            Location = syntax.Identifier.GetLocation(),
            ContainingTypes = ContainingTypeChain(type),
            FactoryAccessible = IsFactoryAccessible(type),
            // A nested layer's writer lands inside its container, so EVERY enclosing declaration
            // has to be partial too, not just the layer's own.
            IsPartial = EnclosingChain(type).All(t => t.DeclaringSyntaxReferences
                .Select(r => r.GetSyntax())
                .OfType<TypeDeclarationSyntax>()
                .Any(d => d.Modifiers.Any(SyntaxKind.PartialKeyword))) &&
                type.DeclaringSyntaxReferences
                .Select(r => r.GetSyntax())
                .OfType<TypeDeclarationSyntax>()
                .Any(d => d.Modifiers.Any(SyntaxKind.PartialKeyword)),
            HasHandWrittenMetadata = type.GetMembers("GetMetadata")
                .OfType<IMethodSymbol>()
                .Where(m => m.Parameters.Length == 0)
                .SelectMany(m => m.DeclaringSyntaxReferences)
                .Select(r => r.GetSyntax())
                .OfType<MethodDeclarationSyntax>()
                .Any(d => d.ToString().IndexOf("base.GetMetadata", System.StringComparison.Ordinal) < 0),
        };

        foreach (var p in ctor.Parameters)
        {
            var info = new ParamModel
            {
                Name = p.Name,
                TypeFqn = p.Type.ToDisplayString(FullyQualified),
                AcceptsNull = p.Type.NullableAnnotation == NullableAnnotation.Annotated
                              || p.NullableAnnotation == NullableAnnotation.Annotated,
            };

            if (IsLayer(Unwrap(p.Type), out var childNumeric)
                || IsLayerSequence(Unwrap(p.Type), out childNumeric))
            {
                info.LayerNumeric = childNumeric?.ToDisplayString(FullyQualified);
            }

            info.TraceableNumeric = TraceableNumericOf(Unwrap(p.Type))?.ToDisplayString(FullyQualified);
            info.Settings = SettingsOf(Unwrap(p.Type)) ?? new List<SettingModel>();

            var isMarked = HasStateAttribute(p);

            // INFERRED, not merely marked. A constructor argument the layer stores in a field is
            // construction state whether or not anyone wrote the attribute -- that is what the
            // field is for. Requiring the attribute made correctness opt-in, which is why 76 of
            // 321 layers had no factory and nothing reported it.
            //
            // The attribute is still honoured, and still the way to override the metadata key or
            // to claim a parameter inference would decline.
            var inferredKind = isMarked ? ValueKind.Unsupported : Classify(p.Type);
            string? inferredMember = null;
            bool inferredConvert = false;
            if (!isMarked && inferredKind != ValueKind.Unsupported && !IsActivation(p.Type, out _))
            {
                inferredMember = FindBackingMember(type, p, out inferredConvert);
            }

            if (isMarked)
            {
                info.IsState = true;
                info.Key = StateKey(p) ?? p.Name;
                info.Kind = Classify(p.Type);
                if (info.Kind == ValueKind.Unsupported)
                {
                    model.Diagnostics.Add(Diagnostic.Create(
                        UnsupportedType, p.Locations.FirstOrDefault() ?? model.Location,
                        type.Name, p.Name, p.Type.ToDisplayString()));
                    return model;
                }

                info.BackingMember = FindBackingMember(type, p, out var needsConvert);
                info.NeedsConvert = needsConvert;
                if (info.BackingMember is null)
                {
                    model.Diagnostics.Add(Diagnostic.Create(
                        NoBackingMember, p.Locations.FirstOrDefault() ?? model.Location,
                        type.Name, p.Name, Pascal(p.Name), p.Type.ToDisplayString()));
                    return model;
                }
            }
            else if (inferredMember is not null)
            {
                // Inferred state covers optional parameters too. An optional argument that the
                // layer stored was previously rebuilt from its DEFAULT, silently discarding a
                // configured value -- the same silent-loss failure this work exists to remove.
                info.IsState = true;
                info.Key = p.Name;
                info.Kind = inferredKind;
                info.BackingMember = inferredMember;
                info.NeedsConvert = inferredConvert;
            }
            else if (IsActivation(p.Type, out var vector))
            {
                info.IsActivation = true;
                info.IsVectorActivation = vector;
            }
            else if (p.IsOptional)
            {
                info.UseDefault = true;
                info.DefaultLiteral = DefaultLiteralOf(p);
            }
            else
            {
                model.Diagnostics.Add(Diagnostic.Create(
                    Unsuppliable, p.Locations.FirstOrDefault() ?? model.Location,
                    type.Name, p.Name, p.Type.ToDisplayString()));
                return model;
            }

            model.Parameters.Add(info);
        }

        if (!model.IsPartial)
        {
            model.Diagnostics.Add(Diagnostic.Create(NotPartial, model.Location, type.Name));
        }

        if (model.HasHandWrittenMetadata)
        {
            model.Diagnostics.Add(Diagnostic.Create(
                HandWrittenMetadata, model.Location, type.Name,
                string.Join(", ", model.Parameters.Where(p => p.IsState).Select(p => p.Name))));
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

    /// <summary>The member named by <c>[LayerState(Member = "...")]</c>, if any.</summary>
    private static string? StateMember(IParameterSymbol p)
    {
        var attr = p.GetAttributes().FirstOrDefault(a => a.AttributeClass?.ToDisplayString() == StateAttribute);
        var named = attr?.NamedArguments.FirstOrDefault(n => n.Key == "Member").Value.Value as string;
        return string.IsNullOrWhiteSpace(named) ? null : named;
    }
    /// <summary>
    /// Whether the type is a layer, and if so its numeric type argument. Covers the base class, the
    /// interface and a concrete layer named directly (QuantizedDenseLayer takes a DenseLayer&lt;float&gt;).
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

    /// <summary>Whether the type is a sequence of layers, and if so their numeric type argument.</summary>
    private static bool IsLayerSequence(ITypeSymbol type, out ITypeSymbol? numeric)
    {
        numeric = null;

        if (type is IArrayTypeSymbol { Rank: 1 } arr) return IsLayer(arr.ElementType, out numeric);

        if (type is not INamedTypeSymbol named) return false;

        // List<ILayer<T>>, IEnumerable<ILayer<T>>, IReadOnlyList<...> and friends.
        if (named.TypeArguments.Length == 1
            && (named.Name is "List" or "IList" or "IEnumerable" or "IReadOnlyList" or "IReadOnlyCollection" or "ICollection"))
            return IsLayer(named.TypeArguments[0], out numeric);

        return false;
    }



    private static ValueKind Classify(ITypeSymbol type)
    {
        type = Unwrap(type);

        if (type.TypeKind == TypeKind.Enum) return ValueKind.Enum;

        // Before the interface check below, which would otherwise claim ILayer<T> as a Component and
        // rebuild a child layer by parameterless Activator.CreateInstance -- losing every argument
        // it was built with.
        // Before the interface check: a delegate is a reference type with its own save path.
        if (type.TypeKind == TypeKind.Delegate) return ValueKind.Delegate;
        if (type is IArrayTypeSymbol { Rank: 1 } strings
            && strings.ElementType.SpecialType == SpecialType.System_String)
            return ValueKind.StringArray;


        if (IsLayer(type, out _)) return ValueKind.Layer;

        // A plain configuration object: no behaviour, just settable values. Its properties are
        // construction state as much as a scalar parameter is, and rebuilding it by parameterless
        // construction alone would silently restore every one of them to its default.
        if (SettingsOf(type) is not null) return ValueKind.Settings;
        if (IsLayerSequence(type, out _)) return ValueKind.LayerList;

        // A pluggable strategy: record which implementation was used and rebuild that one.
        if (type.TypeKind == TypeKind.Interface) return ValueKind.Component;

        if (type is IArrayTypeSymbol { Rank: 1 } arr)
        {
            // A shape list is as much construction state as a scalar dimension, and rebuilding one
            // from the declared input shape is exactly the inference this generator removes.
            if (arr.ElementType.SpecialType == SpecialType.System_Int32) return ValueKind.Int32Array;
            if (arr.ElementType.SpecialType == SpecialType.System_Double) return ValueKind.DoubleArray;
            if (arr.ElementType.SpecialType == SpecialType.System_Boolean) return ValueKind.BooleanArray;


            // int[][]: the per-input shapes a merge layer (Add, Concatenate, Multiply) was built
            // with, where the outer length is the number of inputs.
            if (arr.ElementType is IArrayTypeSymbol { Rank: 1 } inner
                && inner.ElementType.SpecialType == SpecialType.System_Int32)
                return ValueKind.Int32Jagged;

            return ValueKind.Unsupported;
        }

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
    /// <summary>
    /// For <c>Func&lt;ComputationNode&lt;X&gt;, ComputationNode&lt;X&gt;&gt;</c>, returns X. Such a
    /// delegate can be described by running it once and recording the operations it performed,
    /// which is the only description that survives a closure.
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
            => t is INamedTypeSymbol { Name: "ComputationNode", TypeArguments.Length: 1 } n
                ? n.TypeArguments[0]
                : null;
    }



    private static bool IsActivation(ITypeSymbol type, out bool vector)
    {
        var name = (type as INamedTypeSymbol)?.ConstructedFrom.Name ?? type.Name;
        vector = name == "IVectorActivationFunction";
        return vector || name == "IActivationFunction";
    }

    /// <summary>Both types are layers over the same numeric type.</summary>
    private static bool IsSameLayer(ITypeSymbol member, ITypeSymbol parameter)
        => IsLayer(Unwrap(parameter), out var pn)
           && IsLayer(Unwrap(member), out var mn)
           && pn is not null && mn is not null
           && SymbolEqualityComparer.Default.Equals(pn, mn);

    private static string? FindBackingMember(INamedTypeSymbol type, IParameterSymbol p, out bool needsConvert)
    {
        needsConvert = false;

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
                        // A child layer is recorded by walking the object, so the member only has to
                        // BE that layer -- not to be declared as the same type. LoRAAdapterBase keeps
                        // its child as ILayer<T> while a convenience overload takes LayerBase<T>; the
                        // stored object is the same one either way.
                        case IFieldSymbol lf when IsSameLayer(lf.Type, p.Type):
                            return lf.Name;
                        case IPropertySymbol { GetMethod: not null } lp when IsSameLayer(lp.Type, p.Type):
                            return lp.Name;

                        case IFieldSymbol f when SameType(f.Type, p.Type):
                            return f.Name;
                        case IPropertySymbol { GetMethod: not null } prop when SameType(prop.Type, p.Type):
                            return prop.Name;

                        // Layers routinely keep a numeric constructor argument converted to their
                        // own numeric type (a double rate stored as T). That is still the value the
                        // constructor was given, so read it back through a conversion.
                        case IFieldSymbol f2 when IsNumericTypeParameter(f2.Type, p.Type):
                            needsConvert = true;
                            return f2.Name;
                        case IPropertySymbol { GetMethod: not null } prop2 when IsNumericTypeParameter(prop2.Type, p.Type):
                            needsConvert = true;
                            return prop2.Name;
                    }
                }
            }
        }

        return null;
    }

    /// <summary>True when the member is held as the layer's generic numeric type.</summary>
    private static bool IsNumericTypeParameter(ITypeSymbol member, ITypeSymbol parameter)
        => member is ITypeParameterSymbol
           && parameter.SpecialType is SpecialType.System_Double
              or SpecialType.System_Single
              or SpecialType.System_Int32;

    private static bool SameType(ITypeSymbol a, ITypeSymbol b)
        => Unwrap(a).ToDisplayString(FullyQualified) == Unwrap(b).ToDisplayString(FullyQualified);

    /// <summary>Strips <c>Nullable&lt;T&gt;</c> so an <c>int?</c> parameter matches an <c>int</c> field.</summary>
    private static ITypeSymbol Unwrap(ITypeSymbol type)
        => type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } n
            ? n.TypeArguments[0]
            : type;

    /// <summary>Enclosing types, outermost first, rendered with their type parameters.</summary>
    private static List<string> ContainingTypeChain(INamedTypeSymbol type)
    {
        var chain = new List<string>();
        for (var t = type.ContainingType; t is not null; t = t.ContainingType)
        {
            var generics = t.TypeParameters.Length == 0
                ? string.Empty
                : "<" + string.Join(", ", t.TypeParameters.Select(tp => tp.Name)) + ">";
            chain.Insert(0, t.Name + generics);
        }

        return chain;
    }

    /// <summary>The type and every type it is nested in, innermost first.</summary>
    private static IEnumerable<INamedTypeSymbol> EnclosingChain(INamedTypeSymbol type)
    {
        for (var t = type; t is not null; t = t.ContainingType) yield return t;
    }

    /// <summary>
    /// Whether <c>GeneratedLayerFactories</c>, which lives in another namespace, can name this
    /// type. A private or protected nested layer is reachable only from inside its container, so
    /// an entry for it would not compile -- it still gets a writer, just no factory.
    /// </summary>
    private static bool IsFactoryAccessible(INamedTypeSymbol type)
        => EnclosingChain(type).All(t => t.DeclaredAccessibility
            is Accessibility.Public or Accessibility.Internal or Accessibility.ProtectedOrInternal);

    /// <summary>
    /// The parameter's declared default as C# source, so an omitted optional argument is rebuilt
    /// as the value the signature promises rather than as <c>default(T)</c>.
    /// </summary>
    private static string? DefaultLiteralOf(IParameterSymbol p)
    {
        if (!p.HasExplicitDefaultValue) return null;

        var v = p.ExplicitDefaultValue;
        if (v is null) return "default!";

        // An enum default arrives as its underlying integral value; cast it back so the emitted
        // argument keeps the parameter's own type.
        var t = Unwrap(p.Type);
        if (t.TypeKind == TypeKind.Enum)
            return $"({t.ToDisplayString(FullyQualified)})({SymbolDisplay.FormatPrimitive(v, true, false)})";

        return SymbolDisplay.FormatPrimitive(v, quoteStrings: true, useHexadecimalNumbers: false);
    }


    private static string Pascal(string name)
        => name.Length == 0 ? name : char.ToUpperInvariant(name[0]) + name.Substring(1);

    private static void Emit(SourceProductionContext spc, ImmutableArray<LayerModel> models)
    {
        foreach (var d in models.SelectMany(m => m.Diagnostics))
        {
            spc.ReportDiagnostic(d);
        }

        // One constructor per type, choosing the one that carries the MOST construction state. With
        // state inferred rather than annotated, a layer commonly has several constructors and taking
        // the first by source order would pick whichever happened to be declared first -- often a
        // convenience overload that omits the very arguments a rebuild needs. Ties break on source
        // order so the output stays deterministic.
        var byType = models
            .Where(m => m.IsValid)
            .GroupBy(m => m.BaseFqn)
            .Select(g => g
                // Recoverable state first, then how much of it. Counting parameters alone picked
                // LambdaLayer's two-opaque-delegate constructor over the traceable one, and an
                // opaque delegate is the parameter LEAST likely to survive a save -- so the
                // constructor with more of them scored higher while rebuilding worse.
                .OrderByDescending(m => m.Parameters.Count(p => p.TraceableNumeric is not null))
                .ThenByDescending(m => m.Parameters.Count(p => p.IsState))
                .First())
            .OrderBy(m => m.BaseFqn, System.StringComparer.Ordinal)
            .ToList();

        foreach (var model in byType)
        {
            // Qualified by namespace: the short name is not unique. Two distinct MaxPoolingLayer
            // types live in different namespaces, and keying the generated file on TypeName alone
            // made the second one collide with the first and abort the whole generator.
            var hint = model.Namespace is null
                ? model.TypeName
                : model.Namespace.Replace('.', '_') + "_" + model.TypeName;

            spc.AddSource($"{hint}.LayerState.g.cs", SourceText(EmitWriter(model)));
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

        // Nested layers are declared inside their container, so the writer has to be too. Emitting
        // `partial class RegStageBlock` at namespace level declared a NEW top-level type that
        // derived from nothing, and the override bound to nothing (CS0115).
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

            // A delegate is described rather than written down: a named static method by reference,
            // never as marshalled code. Keras marshals the Lambda layer's bytecode and made loading
            // a model arbitrary code execution (CVE-2025-9906); this refuses to.
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

            if (p.Kind == ValueKind.Delegate)
            {
                // A traceable expression is recorded by running it, so a closure survives; anything
                // else falls back to naming the method. Never as marshalled code: Keras writes the
                // Lambda layer's bytecode and made loading a model arbitrary code execution
                // (CVE-2025-9906).
                sb.AppendLine(p.TraceableNumeric is null
                    ? $"        __metadata[\"{p.Key}\"] = "
                      + $"global::AiDotNet.Serialization.DelegateState.Save(this.{p.BackingMember});"
                    : $"        __metadata[\"{p.Key}\"] = "
                      + $"global::AiDotNet.Serialization.DelegateState.SaveTraceable<{p.TraceableNumeric}>("
                      + $"this.{p.BackingMember}, this.GetInputShape());");
                continue;
            }

            if (p.Kind == ValueKind.Component)
            {
                sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.FormatType(this.{p.BackingMember});");
                continue;
            }

            var read = p.NeedsConvert
                ? ConvertExpression(p)
                : $"this.{p.BackingMember}";
            sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.Format({read});");
        }
        sb.AppendLine("    }");
        sb.AppendLine("}");

        for (var i = 0; i < model.ContainingTypes.Count; i++)
        {
            sb.AppendLine("}");
        }

        return sb.ToString();
    }

    private static string ConvertExpression(ParamModel p)
    {
        var converter = p.Kind switch
        {
            ValueKind.Int32 => "ToInt32",
            ValueKind.Int64 => "ToInt64",
            ValueKind.Single => "ToSingle",
            _ => "ToDouble",
        };

        return $"global::System.Convert.{converter}((object)this.{p.BackingMember}!, " +
               "global::System.Globalization.CultureInfo.InvariantCulture)";
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
        sb.AppendLine($"    internal const int Count = {models.Count(m => m.TypeParameters.Count == 1 && m.FactoryAccessible)};");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Attempts to rebuild a layer of the given open generic type.</summary>");
        sb.AppendLine("    /// <param name=\"genericDefinition\">The layer's open generic type, e.g. <c>typeof(DenseLayer&lt;&gt;)</c>.</param>");
        sb.AppendLine("    /// <param name=\"state\">The layer's saved metadata.</param>");
        sb.AppendLine("    /// <param name=\"scalarActivation\">Restored scalar activation, when the constructor takes one.</param>");
        sb.AppendLine("    /// <param name=\"vectorActivation\">Restored vector activation, when the constructor takes one.</param>");
        sb.AppendLine("    /// <param name=\"layer\">The rebuilt layer.</param>");
        // Enumerating the saved children back into whatever collection the constructor asked for.
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

        // Recursion lives here rather than in LayerStateBag because only generated code can call
        // TryCreate; putting it in the bag would make a hand-written type depend on generated output.
        sb.AppendLine("    /// <summary>Rebuilds a child layer from the state nested under <paramref name=\"key\"/>.</summary>");
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

        sb.AppendLine("    /// <returns><c>true</c> when a factory exists for the type.</returns>");
        sb.AppendLine("    internal static bool TryCreate(");
        sb.AppendLine("        global::System.Type genericDefinition,");
        sb.AppendLine("        global::AiDotNet.Serialization.LayerStateBag state,");
        sb.AppendLine("        object? scalarActivation,");
        sb.AppendLine("        object? vectorActivation,");
        sb.AppendLine("        out object layer)");
        sb.AppendLine("    {");

        foreach (var model in models.Where(m => m.TypeParameters.Count == 1 && m.FactoryAccessible))
        {
            var args = string.Join(", ", model.Parameters.Select(p => Argument(p, model.TypeName)));
            var closed = model.ClosedFqn;
            var required = model.Parameters
                .Where(p => p.IsState)
                // Nested state is not stored under the bare parameter name, so the "is this state
                // even mine" guard has to look for the key that is actually written.
                .Select(p => "\"" + p.Key + p.Kind switch
                {
                    ValueKind.Layer => "." + NestedTypeKey,
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

    /// <summary>
    /// Rebuilds a sequence of child layers into the exact collection type the constructor declared.
    /// </summary>
    /// <remarks>
    /// The children come back as a <c>List&lt;TChild&gt;</c>; an array parameter needs ToArray() and
    /// an IEnumerable/IReadOnlyList parameter takes the list as-is.
    /// </remarks>
    private static string LayerListArgument(ParamModel p)
    {
        var declared = p.TypeFqn.TrimEnd('?');
        var element = ElementFqn(declared);
        var rebuilt = $"RebuildNestedRange<{element}>(state, \"{p.Key}\")";
        return declared.EndsWith("[]", System.StringComparison.Ordinal)
            ? $"{rebuilt}.ToArray()"
            : rebuilt;
    }

    /// <summary>The element type of a declared array or single-argument generic collection.</summary>
    private static string ElementFqn(string declared)
    {
        if (declared.EndsWith("[]", System.StringComparison.Ordinal))
            return declared.Substring(0, declared.Length - 2);

        var open = declared.IndexOf('<');
        return open < 0
            ? declared
            : declared.Substring(open + 1, declared.Length - open - 2);
    }


    private static string Argument(ParamModel p, string layerName)
    {
        if (p.IsActivation)
        {
            var iface = p.IsVectorActivation
                ? "global::AiDotNet.Interfaces.IVectorActivationFunction<T>"
                : "global::AiDotNet.Interfaces.IActivationFunction<T>";
            var source = p.IsVectorActivation ? "vectorActivation" : "scalarActivation";
            // `as` yields null both when no activation was saved and when the wrong kind was.
            // A parameter that does not accept null is told which, rather than handed the null.
            return p.AcceptsNull
                ? $"{p.Name}: {source} as {iface}"
                : $"{p.Name}: global::AiDotNet.Serialization.LayerStateBag.RequireActivation<{iface}>"
                  + $"({source}, \"{p.Name}\", \"{layerName}\")";
        }

        // The parameter's OWN default, not default(T). `bool useBias = true` rebuilt through
        // `default!` came back false -- the same silent-value-loss this generator replaced a
        // 4811-line switch to remove, reintroduced one layer down.
        if (p.UseDefault) return $"{p.Name}: {p.DefaultLiteral ?? "default!"}";

        var read = p.Kind switch
        {
            ValueKind.Int32 => $"state.Int32(\"{p.Key}\")",
            ValueKind.Int64 => $"state.Int64(\"{p.Key}\")",
            ValueKind.Double => $"state.Double(\"{p.Key}\")",
            ValueKind.Single => $"state.Single(\"{p.Key}\")",
            ValueKind.Boolean => $"state.Boolean(\"{p.Key}\")",
            ValueKind.String => $"state.String(\"{p.Key}\")",
            ValueKind.Int32Array => $"state.Int32Array(\"{p.Key}\")",
            ValueKind.BooleanArray => $"state.BooleanArray(\"{p.Key}\")",
            ValueKind.StringArray => $"state.StringArray(\"{p.Key}\")",
            ValueKind.Settings => SettingsArgument(p),
            ValueKind.Delegate => $"global::AiDotNet.Serialization.DelegateState.Load<{p.TypeFqn.TrimEnd('?')}>("
                + $"state.String(\"{p.Key}\"), \"{layerName}\", \"{p.Key}\")",
            ValueKind.DoubleArray => $"state.DoubleArray(\"{p.Key}\")",
            ValueKind.Int32Jagged => $"state.Int32Jagged(\"{p.Key}\")",
            ValueKind.Layer => $"({p.TypeFqn.TrimEnd('?')})RebuildNested(state, \"{p.Key}\")!",
            ValueKind.LayerList => LayerListArgument(p),

            ValueKind.Enum => $"state.Enum<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            // A parameter that does not accept null must not be handed one. Component() returns
            // null both when nothing was saved and when the saved type will not load, so a
            // non-nullable parameter reads through the variant that says so instead.
            ValueKind.Component => p.AcceptsNull
                ? $"state.Component<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")"
                : $"state.ComponentRequired<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            _ => "default!",
        };

        return $"{p.Name}: {read}";
    }

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

    /// <summary>
    /// The settable public properties of a plain settings object, when every one of them is a value
    /// the metadata can carry and the type can be constructed without arguments. Anything else is
    /// not a settings object and keeps reporting ADN0053, rather than round-tripping partially.
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
                or ValueKind.LayerList or ValueKind.Delegate or ValueKind.Component)
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

    private static readonly SymbolDisplayFormat FullyQualified =
        SymbolDisplayFormat.FullyQualifiedFormat;

    private static readonly SymbolDisplayFormat UnqualifiedGenerics =
        SymbolDisplayFormat.FullyQualifiedFormat
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
        Delegate,
        StringArray,
        Settings,
        Int32Jagged,
        Layer,
        LayerList,
        Component,
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
                _ => $"state.Enum<{TypeFqn.TrimEnd('?')}>(\"{key}\")",
            };
        }
    }


    private sealed class ParamModel
    {
        public string Name = string.Empty;
        public string TypeFqn = string.Empty;

        /// <summary>Whether the parameter itself accepts a null argument.</summary>
        public bool AcceptsNull;

        /// <summary>
        /// For a child layer or sequence of them, the numeric type argument to write and rebuild
        /// through. Usually the parent's own T, but a layer may name a closed child type --
        /// QuantizedDenseLayer takes a DenseLayer&lt;float&gt; whatever its own T is.
        /// </summary>
        public string? LayerNumeric;

        /// <summary>
        /// For a <c>Func&lt;ComputationNode&lt;X&gt;, ComputationNode&lt;X&gt;&gt;</c>, the X. Such a
        /// delegate can be recorded by running it once over autodiff nodes, which is the only
        /// description that survives a closure.
        /// </summary>
        public string? TraceableNumeric;

        /// <summary>For a settings object, the properties that carry its state.</summary>
        public List<SettingModel> Settings = new();



        public string Key = string.Empty;
        public string? BackingMember;
        public bool IsState;
        public bool IsActivation;
        public bool IsVectorActivation;
        public bool UseDefault;
        public bool NeedsConvert;
        public ValueKind Kind;

        /// <summary>
        /// The parameter's own declared default, rendered as C# source. Rebuilding an omitted
        /// optional parameter as <c>default!</c> is not the same as rebuilding it as its declared
        /// default: <c>bool useBias = true</c> came back <c>false</c>. That is the silent-value-loss
        /// this generator exists to remove, reintroduced inside the replacement.
        /// </summary>
        public string? DefaultLiteral;

    }

    private sealed class LayerModel
    {
        public string? Namespace;
        public string TypeName = string.Empty;
        public List<string> TypeParameters = new();
        public string BaseFqn = string.Empty;

        /// <summary>The type as <c>typeof(X&lt;&gt;)</c> renders it.</summary>
        public string OpenGenericFqn => TypeParameters.Count == 0
            ? BaseFqn
            : BaseFqn + "<" + new string(',', TypeParameters.Count - 1) + ">";

        /// <summary>The type closed over the factory's single numeric parameter.</summary>
        public string ClosedFqn => TypeParameters.Count == 0 ? BaseFqn : BaseFqn + "<T>";
        public List<ParamModel> Parameters = new();
        public List<Diagnostic> Diagnostics = new();
        public Location Location = Location.None;
        /// <summary>
        /// Enclosing types, outermost first, each already rendered with its type parameters. A
        /// nested layer's writer has to be emitted inside those same declarations; emitting it at
        /// namespace level declared an unrelated top-level type deriving from nothing, so the
        /// override bound to nothing -- CS0115 on STCConnectorLayer's private RegStageBlock.
        /// </summary>
        public List<string> ContainingTypes = new();

        /// <summary>
        /// Whether the central factory can name this type at all. A private nested layer is
        /// visible only inside its container, so registering it would not compile.
        /// </summary>
        public bool FactoryAccessible = true;

        public bool IsPartial;
        public bool HasHandWrittenMetadata;
        public bool IsValid;
    }
}
