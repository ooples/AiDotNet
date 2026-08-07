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
                static (node, _) => node is ConstructorDeclarationSyntax c
                    && c.ParameterList.Parameters.Any(p => p.AttributeLists.Count > 0),
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
        if (marked.Count == 0) return null;

        var type = ctor.ContainingType;

        // THE HOST TYPE MUST BE ABLE TO CARRY THE GENERATED MEMBER. Analyze accepted any
        // constructor whose parameters carried [LayerState] and then emitted
        // `partial class {TypeName}` with `internal override void WriteConstructionState`.
        // A struct, a record, or a class not derived from LayerBase produced a raw C#
        // compiler error pointing INTO generated source -- an error about code the author
        // never wrote and cannot open. Refused here with a diagnostic on the declaration
        // instead.
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
            HasHandWrittenMetadata = type.GetMembers("GetMetadata")
                .OfType<IMethodSymbol>()
                .Where(m => m.Parameters.Length == 0)
                .SelectMany(m => m.DeclaringSyntaxReferences)
                .Select(r => r.GetSyntax())
                .OfType<MethodDeclarationSyntax>()
                // SEMANTIC, NOT SUBSTRING. Scanning the method's full text for
                // "base.GetMetadata" fired on the string appearing in a comment or a string
                // literal, and MISSED a real call written `base . GetMetadata()`. Both
                // directions are wrong for a diagnostic whose whole job is telling the author
                // their metadata will silently not be written. Now an actual invocation of a
                // member access on `base` named GetMetadata.
                .All(d => !d.DescendantNodes()
                    .OfType<InvocationExpressionSyntax>()
                    .Any(inv => inv.Expression is MemberAccessExpressionSyntax
                    {
                        Expression: BaseExpressionSyntax,
                        Name.Identifier.ValueText: "GetMetadata",
                    })),
        };

        foreach (var p in ctor.Parameters)
        {
            var info = new ParamModel { Name = p.Name, TypeFqn = p.Type.ToDisplayString(FullyQualified) };

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

                info.BackingMember = FindBackingMember(type, p, out var needsConvert);
                info.NeedsConvert = needsConvert;
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
            }
            else if (p.IsOptional)
            {
                info.UseDefault = true;
                // Taken from the symbol, so the emitted argument is the value the
                // constructor signature actually promises.
                info.DefaultExpression = RenderDefault(p);
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

    private static ValueKind Classify(ITypeSymbol type)
    {
        type = Unwrap(type);

        if (type.TypeKind == TypeKind.Enum) return ValueKind.Enum;

        // A pluggable strategy: record which implementation was used and rebuild that one.
        if (type.TypeKind == TypeKind.Interface) return ValueKind.Component;

        if (type is IArrayTypeSymbol { Rank: 1 } arr && arr.ElementType.SpecialType == SpecialType.System_Int32)
            return ValueKind.Int32Array;

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

    private static string? FindBackingMember(INamedTypeSymbol type, IParameterSymbol p, out bool needsConvert)
    {
        needsConvert = false;
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
                        case IFieldSymbol f when SameType(f.Type, p.Type):
                            return f.Name;
                        case IPropertySymbol { GetMethod: not null } prop when SameType(prop.Type, p.Type):
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

    private static bool SameType(ITypeSymbol a, ITypeSymbol b)
        => Unwrap(a).ToDisplayString(FullyQualified) == Unwrap(b).ToDisplayString(FullyQualified);

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
                .Select(p => "\"" + p.Key + "\"")
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
            return $"{p.Name}: {source} as {iface}";
        }

        // THE PARAMETER'S DEFAULT, NOT THE TYPE'S. Falls back to `default!` only when the
        // declaration genuinely has no value to render.
        if (p.UseDefault) return $"{p.Name}: {p.DefaultExpression ?? "default!"}";

        var read = p.Kind switch
        {
            ValueKind.Int32 => $"state.Int32(\"{p.Key}\")",
            ValueKind.Int64 => $"state.Int64(\"{p.Key}\")",
            ValueKind.Double => $"state.Double(\"{p.Key}\")",
            ValueKind.Single => $"state.Single(\"{p.Key}\")",
            ValueKind.Boolean => $"state.Boolean(\"{p.Key}\")",
            ValueKind.String => $"state.String(\"{p.Key}\")",
            ValueKind.Int32Array => $"state.Int32Array(\"{p.Key}\")",
            ValueKind.Enum => $"state.Enum<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            ValueKind.Component => $"state.Component<{p.TypeFqn.TrimEnd('?')}>(\"{p.Key}\")",
            _ => "default!",
        };

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
