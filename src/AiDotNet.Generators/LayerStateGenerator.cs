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
                .Any(d => d.ToString().IndexOf("base.GetMetadata", System.StringComparison.Ordinal) < 0),
        };

        foreach (var p in ctor.Parameters)
        {
            var info = new ParamModel { Name = p.Name, TypeFqn = p.Type.ToDisplayString(FullyQualified) };

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

    private static bool IsActivation(ITypeSymbol type, out bool vector)
    {
        var name = (type as INamedTypeSymbol)?.ConstructedFrom.Name ?? type.Name;
        vector = name == "IVectorActivationFunction";
        return vector || name == "IActivationFunction";
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
                .OrderByDescending(m => m.Parameters.Count(p => p.IsState))
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
                ? ConvertExpression(p)
                : $"this.{p.BackingMember}";
            sb.AppendLine($"        __metadata[\"{p.Key}\"] = global::AiDotNet.Serialization.LayerStateBag.Format({read});");
        }
        sb.AppendLine("    }");
        sb.AppendLine("}");
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

        if (p.UseDefault) return $"{p.Name}: default!";

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
        Component,
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
        public bool NeedsConvert;
        public ValueKind Kind;
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
        public bool IsPartial;
        public bool HasHandWrittenMetadata;
        public bool IsValid;
    }
}
