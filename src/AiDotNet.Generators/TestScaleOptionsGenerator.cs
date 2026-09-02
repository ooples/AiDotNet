using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Emits <c>ModelTestScale.CreateBoundedOptions</c>, which builds a size-bounded instance of any
/// model options type without a line of hand-written per-model code.
/// </summary>
/// <remarks>
/// <para>
/// WHY THIS EXISTS. Constructing models at their paper-faithful defaults is correct and must stay
/// that way, but a diagnostic sweep that builds every model cannot afford it: an LLM-scale model is
/// tens of gigabytes of weights, and six of them were killing the clone sweep outright. The
/// responses to that had all been hand-written and were multiplying -- four per-family branches in
/// AllModelsCloneTests.CreateBoundedOptions (Nemotron, codec TTS, foundational VLM), and seven
/// ForTesting/CreateForTesting factories across ResNet, EfficientNet and DenseNet plus their
/// configuration classes. Every new large model added another. This replaces all of them.
/// </para>
/// <para>
/// HOW A KNOB IS RECOGNISED. A settable numeric property whose name matches the size vocabulary
/// below, on a type whose name ends in Options or Configuration. Names, not values, because the
/// value IS the paper default and carries no marker distinguishing "hidden width" from "vocabulary
/// id" -- the name is the only signal the source carries. The vocabulary is deliberately narrow and
/// listed in one place so it can be read and argued with.
/// </para>
/// <para>
/// WHY GUESSING WRONG IS CHEAP HERE. This code path is reachable only from test-scale construction.
/// A missed knob leaves that model large, which is the status quo; a wrongly matched property makes
/// one test fixture smaller than intended. Neither can affect a model a user constructs, because
/// nothing in the shipping construction path calls it.
/// </para>
/// <para>
/// CLAMP, NEVER RAISE. A knob already at or below its bound is left alone, so a model whose defaults
/// are already small is untouched rather than being rewritten to a different small value.
/// </para>
/// </remarks>
[Generator]
public class TestScaleOptionsGenerator : IIncrementalGenerator
{
    private const string DimensionDivisibilityAttributeName =
        "AiDotNet.Attributes.DimensionDivisibilityAttribute";

    /// <summary>Names that must KEEP their value even though they are ints above the cap.</summary>
    /// <remarks>
    /// An allow-list of size words cannot work here and the attempt is instructive: bounding CSM by
    /// name matched its dims but missed NumCodebooks, CodecFrameRate and MaxCodecFrames, leaving
    /// small widths driving a huge frame count. That is WORSE than leaving it alone -- the model
    /// stopped OOMing and started stalling past the drain budget instead. Every new family would
    /// add more words to chase.
    ///
    /// So the rule is inverted: an int knob above the cap is shrunk unless it is semantically fixed.
    /// A missed knob is then the safe direction (a value stays large), and the deny-list holds only
    /// things whose MEANING is the number -- a sample rate is 16000 Hz because audio is 16 kHz, not
    /// because someone chose a big model.
    /// </remarks>
    private static readonly string[] SemanticallyFixed =
    {
        // INVERSE KNOBS: shrinking these INCREASES work, so they must never be capped. A smaller
        // hop or patch produces MORE frames or patches from the same input, and a smaller stride
        // downsamples less. Capping HopSize to 16 is what made CSM stall even though every other
        // number was smaller than the hand-written config -- the model was doing far more work at
        // "reduced" scale. This is the one direction where a wrong guess is expensive, which is why
        // they live in the leave-alone list rather than getting a cap of their own.
        "HopSize",
        "FftSize",      // a SMALLER analysis window yields more frames, exactly like HopSize
        "NFft",
        "WinLength",
        "Stride",
        "PatchSize",
        "Downsample",
        "PoolSize",
        "WindowSize",
        "SampleRate",
        "Seed",
        "RandomState",
        "Version",
        "Rank",         // LoRA rank is already small and load-bearing
        "Axis",
        "Precision",
        "DeviceId",
        "Port",
        "Timeout",
        "Year",
    };




    /// <summary>Factor applied to any declared integer above <see cref="SmallEnough"/>.</summary>
    /// <remarks>
    /// 32 turns paper-scale widths into test-scale ones (1536 -> 48, 768 -> 24, 4096 -> 128) while
    /// leaving the structure recognisable. It is a ratio rather than a target, so nothing has to
    /// know what the number MEANS.
    /// </remarks>
    private const int ScaleDivisor = 32;

    /// <summary>Floor for a scaled integer, and the value at or below which nothing changes.</summary>
    /// <remarks>
    /// A pure ratio is not enough on its own. A classifier backbone downsamples by 32x, so scaling
    /// a 224-wide input to 7 collapses the spatial extent before the classifier and 35 vision tests
    /// fail -- the same collapse an earlier absolute "spatial" tier existed to prevent. A floor
    /// achieves it without classifying names: 224 becomes 32, 1536 becomes 48, and anything already
    /// at or below 32 is untouched, so nothing is ever RAISED.
    /// </remarks>
    private const int MinimumScaled = 32;

    /// <summary>Legacy alias retained for the emitted-constant name only.</summary>
    private const int SmallEnough = MinimumScaled;




    /// <inheritdoc/>
    /// <remarks>
    /// Driven from the COMPILATION, not a syntax provider. The syntax-provider form yielded 742
    /// types when the project was built directly and 0 when it was built as a dependency of the test
    /// project -- an empty table that still compiled, so every caller silently got unbounded
    /// options and four rounds of investigation were run against bounds that were never applied.
    /// Walking the compilation's namespaces is deterministic and cannot half-populate.
    /// </remarks>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        context.RegisterSourceOutput(
            context.CompilationProvider,
            static (spc, compilation) => Execute(spc, CollectOptionTypes(compilation)));
    }

    private static ImmutableArray<INamedTypeSymbol?> CollectOptionTypes(Compilation compilation)
    {
        var found = ImmutableArray.CreateBuilder<INamedTypeSymbol?>();
        var queue = new System.Collections.Generic.Queue<INamespaceOrTypeSymbol>();
        queue.Enqueue(compilation.Assembly.GlobalNamespace);

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            foreach (var member in current.GetMembers())
            {
                if (member is INamespaceSymbol childNamespace)
                {
                    queue.Enqueue(childNamespace);
                    continue;
                }

                if (member is not INamedTypeSymbol type) continue;

                // Nested types can carry options too; keep walking.
                queue.Enqueue(type);

                if (type.TypeKind != TypeKind.Class) continue;
                if (!type.Name.EndsWith("Options") && !type.Name.EndsWith("Configuration")) continue;
                if (!HasParameterlessConstructor(type) && !HasSynthesizableConstructor(type)) continue;

                found.Add(type);
            }
        }

        return found.ToImmutable();
    }

    private static bool HasParameterlessConstructor(INamedTypeSymbol? symbol)
    {
        if (symbol is null || symbol.IsAbstract || symbol.IsStatic) return false;
        if (symbol.DeclaredAccessibility != Accessibility.Public) return false;

        // Only a type the harness can actually new up is worth an entry.
        return symbol.InstanceConstructors.Any(c =>
            c.Parameters.Length == 0 && c.DeclaredAccessibility == Accessibility.Public);
    }

    /// <summary>Whether a constructor-only type can be built from synthesized arguments.</summary>
    /// <remarks>
    /// The three vision configurations (ResNet, EfficientNet, DenseNet) are IMMUTABLE: constructor
    /// only, zero settable properties, so the clamp-a-property mechanism cannot touch them. They are
    /// the last hand-written test-scale logic in the library, and they all share one shape --
    /// (SomeVariant variant, int numClasses, ...optionals). Named arguments let the generator supply
    /// only the parameters worth bounding and leave every default alone.
    ///
    /// Deliberately narrow: every REQUIRED parameter must be an int or an enum. Anything else (a
    /// path, a delegate, another options object) means the generator cannot honestly invent a value,
    /// and the type is skipped rather than guessed at.
    /// </remarks>
    private static bool HasSynthesizableConstructor(INamedTypeSymbol type)
        => PickSynthesizableConstructor(type) is not null;

    /// <summary>Gets inherited, declarative integer divisibility relationships for an options type.</summary>
    private static System.Collections.Generic.List<(string Dimension, string Divisor)>
        GetDivisibilityConstraints(INamedTypeSymbol type)
    {
        var constraints = new System.Collections.Generic.List<(string Dimension, string Divisor)>();
        var seen = new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal);

        for (var walk = type; walk is not null && walk.SpecialType != SpecialType.System_Object; walk = walk.BaseType)
        {
            foreach (var attribute in walk.GetAttributes())
            {
                if (attribute.AttributeClass?.ToDisplayString() != DimensionDivisibilityAttributeName)
                    continue;
                if (attribute.ConstructorArguments.Length != 2)
                    continue;
                if (attribute.ConstructorArguments[0].Value is not string dimension
                    || attribute.ConstructorArguments[1].Value is not string divisor)
                    continue;
                if (!seen.Add(dimension + "\0" + divisor))
                    continue;

                constraints.Add((dimension, divisor));
            }
        }

        return constraints;
    }

    private static IMethodSymbol? PickSynthesizableConstructor(INamedTypeSymbol type)
    {
        if (type.IsAbstract || type.IsStatic || type.IsGenericType) return null;
        if (type.DeclaredAccessibility != Accessibility.Public) return null;

        IMethodSymbol? best = null;
        foreach (var ctor in type.InstanceConstructors)
        {
            if (ctor.DeclaredAccessibility != Accessibility.Public) continue;
            if (ctor.Parameters.Length == 0) continue;

            bool usable = true;
            foreach (var parameter in ctor.Parameters)
            {
                if (parameter.HasExplicitDefaultValue) continue;
                if (parameter.Type.TypeKind == TypeKind.Enum) continue;
                if (parameter.Type.SpecialType == SpecialType.System_Int32) continue;
                usable = false;
                break;
            }

            if (!usable) continue;

            // Fewest required parameters wins: the least invented state.
            int required = ctor.Parameters.Count(x => !x.HasExplicitDefaultValue);
            int bestRequired = best is null
                ? int.MaxValue
                : best.Parameters.Count(x => !x.HasExplicitDefaultValue);
            if (required < bestRequired) best = ctor;
        }

        return best;
    }

    /// <summary>Renders the named arguments for a synthesized constructor call.</summary>
    private static string? RenderSynthesizedArguments(IMethodSymbol constructor)
    {
        var parts = new System.Collections.Generic.List<string>();

        foreach (var parameter in constructor.Parameters)
        {
            if (parameter.Type.TypeKind == TypeKind.Enum)
            {
                // FIRST declared member. Variant enums are conventionally ordered smallest-first
                // (ResNet18, EfficientNetB0, DenseNet121), which is exactly the test-scale choice
                // the hand-written CreateForTesting made by hand.
                // PREFER A "Custom" MEMBER. A variant enum that offers Custom means "use the
                // dimensions I pass rather than the paper preset", which is exactly what test-scale
                // construction wants -- EfficientNet's hand-written factory chose Custom with a
                // 32x32 input, and taking B0 instead both changed the reported variant and left the
                // model too large for the mini-network size gate. Falling back to the first member
                // keeps ResNet18 and DenseNet121, where first IS smallest.
                var members = parameter.Type.GetMembers()
                    .OfType<IFieldSymbol>()
                    .Where(f => f.HasConstantValue)
                    .ToArray();
                // A "Custom" member is TEMPTING and wrong on its own: it means "use the multipliers
                // I pass", and supplying the variant without its companion custom* parameters
                // produced a degenerate EfficientNet -- 35 failures against 2. Until the generator
                // can supply the companions too, take the first member, which is smallest for the
                // variant enums that exist (ResNet18, EfficientNetB0, DenseNet121).
                // A "Custom" member means "use the values I pass" -- exactly test-scale intent --
                // but ONLY if the companion custom* parameters are supplied with it. Choosing it
                // bare produced a degenerate EfficientNet (35 failures against 2), so it is taken
                // only when this constructor actually offers those companions.
                // EVERY companion must be supplyable, not merely present. EfficientNet offers
                // customInputHeight and two multipliers, all int?/double? the generator can fill.
                // DenseNet offers customBlockLayers, an array it cannot invent -- taking Custom
                // there produced a model that missed its size thresholds. Partial custom state is
                // worse than the preset.
                var companions = constructor.Parameters
                    .Where(x => x.Name.StartsWith("custom", System.StringComparison.OrdinalIgnoreCase))
                    .ToArray();
                var hasCompanions = companions.Length > 0 && companions.All(IsSupplyableCompanion);
                var firstMember = (hasCompanions
                        ? members.FirstOrDefault(f =>
                            string.Equals(f.Name, "Custom", System.StringComparison.Ordinal))
                        : null)
                    ?? members.FirstOrDefault();
                if (firstMember is null)
                {
                    if (parameter.HasExplicitDefaultValue) continue;
                    return null;
                }

                var enumName = parameter.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
                parts.Add($"{parameter.Name}: {enumName}.{firstMember.Name}");
                continue;
            }

            // Nullable companions to a Custom variant. Without these the variant has nothing to
            // work from; with them it is the smallest honest configuration the type can express.
            if (parameter.Type is INamedTypeSymbol nullable
                && nullable.OriginalDefinition.SpecialType == SpecialType.System_Nullable_T
                && parameter.Name.StartsWith("custom", System.StringComparison.OrdinalIgnoreCase))
            {
                var inner = nullable.TypeArguments[0];
                if (inner.SpecialType == SpecialType.System_Int32)
                {
                    // Scaled from the declared default, or a small concrete value when there is
                    // none to scale. No cap tier, so no name has to be classified.
                    var companionDefault = parameter.HasExplicitDefaultValue
                        && parameter.ExplicitDefaultValue is int companionDeclared
                            ? companionDeclared
                            : 32;
                    parts.Add($"{parameter.Name}: ScaleDeclaredInteger({companionDefault})");
                    continue;
                }

                if (inner.SpecialType == SpecialType.System_Double)
                {
                    // A multiplier of one keeps the variant's own proportions.
                    parts.Add($"{parameter.Name}: 1.0");
                    continue;
                }
            }

            // An int[] companion describes a per-STAGE structure -- DenseNet's customBlockLayers
            // is layers-per-dense-block. Four stages is the near-universal backbone convention
            // (ResNet, DenseNet and EfficientNet all use four), and each stage takes the count cap,
            // which reproduces the hand-written [2, 2, 2, 2] exactly.
            if (parameter.Type is IArrayTypeSymbol array
                && array.ElementType.SpecialType == SpecialType.System_Int32
                && parameter.Name.StartsWith("custom", System.StringComparison.OrdinalIgnoreCase))
            {
                var stages = string.Join(", ", Enumerable.Repeat("2", BackboneStages));
                parts.Add($"{parameter.Name}: new int[] {{ {stages} }}");
                continue;
            }

            if (parameter.Type.SpecialType != SpecialType.System_Int32) continue;

            bool scalable = IsScalable(parameter.Name);

            // CLAMP, NEVER RAISE -- for constructor arguments too. A property is compared against
            // its live value; a constructor parameter has only its DECLARED DEFAULT, and skipping
            // that comparison is what made this raise values instead of lowering them. inputChannels
            // defaults to 3 for RGB, and passing the cap of 16 built a stem expecting 16-channel
            // input that every 3-channel image then failed against: "Expected input depth 16, but
            // got 3", 41 tests. When the default is already at or below the bound, say nothing and
            // let the default stand.
            // A declared default is scaled in place; anything not scalable keeps its own value,
            // which for an optional parameter means saying nothing at all.
            if (parameter.HasExplicitDefaultValue
                && parameter.ExplicitDefaultValue is int declaredDefault)
            {
                if (!scalable) continue;
                parts.Add($"{parameter.Name}: ScaleDeclaredInteger({declaredDefault})");
                continue;
            }

            if (!scalable)
            {
                // Semantically fixed: only supply it when it is required, and then with its own
                // declared default if it has one.
                if (parameter.HasExplicitDefaultValue) continue;
                parts.Add(
                    $"{parameter.Name}: Override(overrides, \"{parameter.Name}\", "
                        + $"{DefaultRequiredInt(parameter.Name)})");
                continue;
            }

            parts.Add(
                $"{parameter.Name}: Override(overrides, \"{parameter.Name}\", "
                    + $"{DefaultRequiredInt(parameter.Name)})");
        }

        return parts.Count == 0 ? null : string.Join(", ", parts);
    }

    /// <summary>Value for a required int the bounds vocabulary has nothing to say about.</summary>
    /// <remarks>Ten, matching the class count the hand-written vision fixtures used.</remarks>
    private static int DefaultRequiredInt(string parameterName) => 10;

    /// <summary>Stages in a conventional convolutional backbone.</summary>
    /// <remarks>Four, as ResNet, DenseNet and EfficientNet all use.</remarks>
    private const int BackboneStages = 4;

    /// <summary>Whether the generator can honestly invent a value for a Custom companion.</summary>
    private static bool IsSupplyableCompanion(IParameterSymbol parameter)
    {
        if (parameter.Type is IArrayTypeSymbol array)
            return array.ElementType.SpecialType == SpecialType.System_Int32;

        if (parameter.Type is not INamedTypeSymbol named) return false;
        if (named.OriginalDefinition.SpecialType != SpecialType.System_Nullable_T) return false;

        var inner = named.TypeArguments[0].SpecialType;
        return inner == SpecialType.System_Int32 || inner == SpecialType.System_Double;
    }

    private static void Execute(SourceProductionContext context, ImmutableArray<INamedTypeSymbol?> types)
    {
        // EMIT NOTHING WHEN THERE IS NOTHING TO BOUND. This generator is attached to more than one
        // compilation, and the test assembly declares no options types, so it produced a SECOND,
        // EMPTY ModelTestScale there. Being in the test assembly itself, that copy won every lookup
        // and CreateBoundedOptions silently returned unbounded options -- while the real table in
        // AiDotNet carried 1193 types. The symptom was invisible: it compiled, returned non-null,
        // and every value came back at its paper default. Four investigations into "which knob is
        // wrong" were run against bounds that were never applied.
        if (types.IsDefaultOrEmpty) return;

        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine("#nullable enable");
        sb.AppendLine("namespace AiDotNet.Testing;");
        sb.AppendLine();
        sb.AppendLine("/// <summary>Size-bounded construction of model options, for diagnostic sweeps.</summary>");
        sb.AppendLine("/// <remarks>");
        sb.AppendLine("/// Generated by TestScaleOptionsGenerator. Do not hand-edit, and do not add per-model");
        sb.AppendLine("/// branches anywhere else -- extend the size vocabulary in the generator instead.");
        sb.AppendLine("/// The knobs are a TABLE applied reflectively rather than emitted as typed property");
        sb.AppendLine("/// assignments, because many options types are generic: a typed assignment would need");
        sb.AppendLine("/// typeof(Foo&lt;T&gt;) with no T in scope, while the open definition keys a table fine.");
        sb.AppendLine("/// </remarks>");
        sb.AppendLine("public static class ModelTestScale");
        sb.AppendLine("{");
        // Constructor-only types get a direct, strongly-typed branch: there is no property to clamp,
        // so the bound has to be applied at construction.
        var synthesized = new StringBuilder();
        int synthesizedCount = 0;
        foreach (var type in types)
        {
            if (type is null || HasParameterlessConstructor(type)) continue;

            var ctor = PickSynthesizableConstructor(type);
            if (ctor is null) continue;

            var arguments = RenderSynthesizedArguments(ctor);
            if (arguments is null) continue;

            var typeName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            synthesized.AppendLine($"        if (optionsType == typeof({typeName}))");
            synthesized.AppendLine("        {");
            synthesized.AppendLine("            try");
            synthesized.AppendLine("            {");
            synthesized.AppendLine($"                return new {typeName}({arguments});");
            synthesized.AppendLine("            }");
            synthesized.AppendLine("            catch (global::System.Exception)");
            synthesized.AppendLine("            {");
            synthesized.AppendLine("                // A configuration that validates its arguments may reject a bounded");
            synthesized.AppendLine("                // combination; that is not a failure, the caller falls back.");
            synthesized.AppendLine("                return null;");
            synthesized.AppendLine("            }");
            synthesized.AppendLine("        }");
            synthesized.AppendLine();
            synthesizedCount++;
        }

        sb.AppendLine("    private static readonly (global::System.Type Type, string Property, int Bound)[] Knobs =");
        sb.AppendLine("    {");

        var seen = new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal);
        int typeCount = 0;
        int knobCount = 0;

        foreach (var type in types)
        {
            if (type is null) continue;

            var fullName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!seen.Add(fullName)) continue;

            // INHERITED KNOBS COUNT. GetMembers() returns only directly declared members, and an
            // options type routinely inherits its dimensions from a family base: CSMOptions declares
            // almost nothing and takes its sizes from CodecTtsOptions. The hand-written branch this
            // replaces used IsAssignableFrom, so it covered the whole family; keying on the exact
            // type without walking the base chain silently bounded nothing for every derived type,
            // and CSM went from passing to stalling past the drain budget.
            var allProperties = new System.Collections.Generic.List<IPropertySymbol>();
            var declaredNames = new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal);
            for (var walk = type; walk is not null && walk.SpecialType != SpecialType.System_Object; walk = walk.BaseType)
            {
                foreach (var property in walk.GetMembers().OfType<IPropertySymbol>())
                {
                    // A derived override shadows the base declaration; first one wins.
                    if (declaredNames.Add(property.Name)) allProperties.Add(property);
                }
            }

            var knobs = allProperties
                .Where(p => p.DeclaredAccessibility == Accessibility.Public
                    && p.SetMethod is { DeclaredAccessibility: Accessibility.Public }
                    && !p.IsStatic
                    && p.Type.SpecialType == SpecialType.System_Int32)
                .Select(p => (Property: p.Name, Bound: IsScalable(p.Name) ? 1 : 0))
                .Where(k => k.Bound > 0)
                .ToArray();

            if (knobs.Length == 0) continue;

            // Open generic definition for a generic type; the closed type keys back to it at runtime.
            var typeOfArgument = type.IsGenericType
                ? type.ConstructUnboundGenericType().ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                : fullName;

            foreach (var (property, bound) in knobs)
            {
                sb.AppendLine($"        (typeof({typeOfArgument}), \"{property}\", {bound}),");
                knobCount++;
            }

            typeCount++;
        }

        sb.AppendLine("    };");
        sb.AppendLine();
        sb.AppendLine("    private static readonly (global::System.Type Type, string Dimension, string Divisor)[] DivisibilityConstraints =");
        sb.AppendLine("    {");

        var constraintSeen = new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal);
        int constraintCount = 0;
        foreach (var type in types)
        {
            if (type is null) continue;

            var fullName = type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            if (!constraintSeen.Add(fullName)) continue;

            var typeOfArgument = type.IsGenericType
                ? type.ConstructUnboundGenericType().ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)
                : fullName;
            foreach (var (dimension, divisor) in GetDivisibilityConstraints(type))
            {
                sb.AppendLine($"        (typeof({typeOfArgument}), \"{dimension}\", \"{divisor}\"),");
                constraintCount++;
            }
        }

        sb.AppendLine("    };");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Creates an options instance with its size knobs clamped down.</summary>");
        sb.AppendLine("    /// <param name=\"optionsType\">The options or configuration type to build.</param>");
        sb.AppendLine("    /// <returns>A bounded instance, or null when the type cannot be bounded.</returns>");
        sb.AppendLine("    /// <param name=\"overrides\">");
        sb.AppendLine("    /// Caller-supplied values by parameter or property name, applied INSTEAD of the");
        sb.AppendLine("    /// generated bound. Some values are the caller's intent, not a size the generator");
        sb.AppendLine("    /// can infer -- numClasses is the obvious one: a hand-written test factory took it");
        sb.AppendLine("    /// as an argument, and inventing 16 where the caller meant 10 failed 42 tests.");
        sb.AppendLine("    /// </param>");
        sb.AppendLine("    public static object? CreateBoundedOptions(");
        sb.AppendLine("        global::System.Type optionsType,");
        sb.AppendLine("        global::System.Collections.Generic.IReadOnlyDictionary<string, int>? overrides = null)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (optionsType is null) return null;");
        sb.AppendLine();
        if (synthesizedCount > 0)
        {
            sb.Append(synthesized.ToString());
        }
        sb.AppendLine("        var key = optionsType.IsGenericType");
        sb.AppendLine("            ? optionsType.GetGenericTypeDefinition()");
        sb.AppendLine("            : optionsType;");
        sb.AppendLine();
        sb.AppendLine("        // ONLY ANSWER FOR TYPES THIS TABLE KNOWS. Constructing whatever it is handed");
        sb.AppendLine("        // made this return boxed defaults for primitives: CreateBoundedOptions(typeof(int))");
        sb.AppendLine("        // produced 0, and the clone sweep -- which asks per CONSTRUCTOR PARAMETER type --");
        sb.AppendLine("        // used that 0 in place of the parameter's own default. MATCHA then built a");
        sb.AppendLine("        // PatchEmbeddingLayer with embeddingDim 0 and threw. Every model with an int");
        sb.AppendLine("        // constructor parameter was exposed to this.");
        sb.AppendLine("        bool known = false;");
        sb.AppendLine("        for (int i = 0; i < Knobs.Length; i++)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (Knobs[i].Type != key) continue;");
        sb.AppendLine("            known = true;");
        sb.AppendLine("            break;");
        sb.AppendLine("        }");
        sb.AppendLine("        if (!known)");
        sb.AppendLine("        {");
        sb.AppendLine("            for (int i = 0; i < DivisibilityConstraints.Length; i++)");
        sb.AppendLine("            {");
        sb.AppendLine("                if (DivisibilityConstraints[i].Type != key) continue;");
        sb.AppendLine("                known = true;");
        sb.AppendLine("                break;");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        if (!known) return null;");
        sb.AppendLine();
        sb.AppendLine("        object? instance;");
        sb.AppendLine("        try");
        sb.AppendLine("        {");
        sb.AppendLine("            instance = global::System.Activator.CreateInstance(optionsType);");
        sb.AppendLine("        }");
        sb.AppendLine("        catch (global::System.Exception)");
        sb.AppendLine("        {");
        sb.AppendLine("            // A type whose parameterless constructor throws under a test");
        sb.AppendLine("            // configuration is not a bounding failure; the caller falls back.");
        sb.AppendLine("            return null;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        if (instance is null) return null;");
        sb.AppendLine();
        sb.AppendLine("        bool bounded = false;");
        sb.AppendLine("        for (int i = 0; i < Knobs.Length; i++)");
        sb.AppendLine("        {");
        sb.AppendLine("            if (Knobs[i].Type != key) continue;");
        sb.AppendLine();
        sb.AppendLine("            var property = optionsType.GetProperty(Knobs[i].Property);");
        sb.AppendLine("            if (property is null || !property.CanRead || !property.CanWrite) continue;");
        sb.AppendLine("            if (property.GetValue(instance) is not int current) continue;");
        sb.AppendLine();
        sb.AppendLine("            // SCALED, not clamped. Dividing preserves the ratios between knobs, so a");
        sb.AppendLine("            // width and a head count shrink together and width % heads still divides.");
        sb.AppendLine("            // An absolute cap set them independently and broke exactly that.");
        sb.AppendLine("            var scaled = ScaleDeclaredInteger(current);");
        sb.AppendLine("            if (scaled == current) continue;");
        sb.AppendLine();
        sb.AppendLine("            property.SetValue(instance, scaled);");
        sb.AppendLine("            bounded = true;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        if (overrides is not null)");
        sb.AppendLine("        {");
        sb.AppendLine("            foreach (var pair in overrides)");
        sb.AppendLine("            {");
        sb.AppendLine("                var target = optionsType.GetProperty(pair.Key);");
        sb.AppendLine("                if (target is null || !target.CanWrite) continue;");
        sb.AppendLine("                if (target.PropertyType != typeof(int)) continue;");
        sb.AppendLine("                target.SetValue(instance, pair.Value);");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        // Restore relationships after both independent scaling and explicit overrides.");
        sb.AppendLine("        // A dimension may cross the scaling floor while its small head count does not");
        sb.AppendLine("        // (768 / 12 becomes 32 / 12), so ratio-preserving division alone is insufficient.");
        sb.AppendLine("        bounded |= AlignDeclaredConstraints(optionsType, key, instance);");
        sb.AppendLine();
        sb.AppendLine("        return bounded ? instance : instance;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Checks every divisibility relationship declared for an options instance.</summary>");
        sb.AppendLine("    public static bool SatisfiesDeclaredConstraints(object? instance, out string? failure)");
        sb.AppendLine("    {");
        sb.AppendLine("        failure = null;");
        sb.AppendLine("        if (instance is null)");
        sb.AppendLine("        {");
        sb.AppendLine("            failure = \"The bounded options instance is null.\";");
        sb.AppendLine("            return false;");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        var optionsType = instance.GetType();");
        sb.AppendLine("        var key = optionsType.IsGenericType ? optionsType.GetGenericTypeDefinition() : optionsType;");
        sb.AppendLine("        for (int i = 0; i < DivisibilityConstraints.Length; i++)");
        sb.AppendLine("        {");
        sb.AppendLine("            var constraint = DivisibilityConstraints[i];");
        sb.AppendLine("            if (constraint.Type != key) continue;");
        sb.AppendLine("            var dimensionProperty = optionsType.GetProperty(constraint.Dimension);");
        sb.AppendLine("            var divisorProperty = optionsType.GetProperty(constraint.Divisor);");
        sb.AppendLine("            if (dimensionProperty?.GetValue(instance) is not int dimension");
        sb.AppendLine("                || divisorProperty?.GetValue(instance) is not int divisor)");
        sb.AppendLine("            {");
        sb.AppendLine("                failure = $\"{optionsType.FullName} declares {constraint.Dimension} / {constraint.Divisor}, but both are not readable integer properties.\";");
        sb.AppendLine("                return false;");
        sb.AppendLine("            }");
        sb.AppendLine("            if (dimension <= 0 || divisor <= 0 || dimension % divisor != 0)");
        sb.AppendLine("            {");
        sb.AppendLine("                failure = $\"{optionsType.FullName}.{constraint.Dimension} ({dimension}) must be positive and divisible by {constraint.Divisor} ({divisor}).\";");
        sb.AppendLine("                return false;");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine();
        sb.AppendLine("        return true;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    private static bool AlignDeclaredConstraints(global::System.Type optionsType, global::System.Type key, object instance)");
        sb.AppendLine("    {");
        sb.AppendLine("        bool changed = false;");
        sb.AppendLine("        for (int i = 0; i < DivisibilityConstraints.Length; i++)");
        sb.AppendLine("        {");
        sb.AppendLine("            var constraint = DivisibilityConstraints[i];");
        sb.AppendLine("            if (constraint.Type != key) continue;");
        sb.AppendLine("            var dimensionProperty = optionsType.GetProperty(constraint.Dimension);");
        sb.AppendLine("            var divisorProperty = optionsType.GetProperty(constraint.Divisor);");
        sb.AppendLine("            if (dimensionProperty is null || !dimensionProperty.CanRead || !dimensionProperty.CanWrite)");
        sb.AppendLine("                continue;");
        sb.AppendLine("            if (dimensionProperty.GetValue(instance) is not int dimension");
        sb.AppendLine("                || divisorProperty?.GetValue(instance) is not int divisor)");
        sb.AppendLine("                continue;");
        sb.AppendLine("            if (dimension <= 0 || divisor <= 0 || dimension % divisor == 0) continue;");
        sb.AppendLine();
        sb.AppendLine("            int aligned = AlignDimensionToDivisor(dimension, divisor);");
        sb.AppendLine("            if (aligned == dimension) continue;");
        sb.AppendLine("            dimensionProperty.SetValue(instance, aligned);");
        sb.AppendLine("            changed = true;");
        sb.AppendLine("        }");
        sb.AppendLine("        return changed;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Scales any declared integer down proportionally, using no names at all.</summary>");
        sb.AppendLine("    /// <remarks>");
        sb.AppendLine("    /// Name-based rules kept breaking: every new family invents a spelling the lists do not");
        sb.AppendLine("    /// have, and a model that takes its dimensions as plain constructor ints is not matched");
        sb.AppendLine("    /// at all -- MATCHA built at its full 1536-wide default and killed the host.");
        sb.AppendLine("    ///");
        sb.AppendLine("    /// Dividing instead of clamping needs no vocabulary and preserves RELATIONSHIPS. An");
        sb.AppendLine("    /// absolute cap sets width and head count independently and breaks width % heads == 0;");
        sb.AppendLine("    /// scaling by a common factor keeps the ratio, so 1536 wide with 12 heads becomes 48");
        sb.AppendLine("    /// wide with 12 heads and still divides. Values already small are left untouched, which");
        sb.AppendLine("    /// is what keeps head counts, channel counts and other small structural numbers intact.");
        sb.AppendLine("    /// </remarks>");
        sb.AppendLine("    public static int ScaleDeclaredInteger(int declared)");
        sb.AppendLine("        => declared <= MinimumScaled");
        sb.AppendLine("            ? declared");
        sb.AppendLine("            : global::System.Math.Max(MinimumScaled, declared / ScaleDivisor);");
        sb.AppendLine();
        sb.AppendLine("    /// <summary>Aligns a positive dimension upward to a positive architectural divisor.</summary>");
        sb.AppendLine("    public static int AlignDimensionToDivisor(int dimension, int divisor)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (dimension <= 0 || divisor <= 0 || dimension % divisor == 0) return dimension;");
        sb.AppendLine("        long aligned = ((long)dimension + divisor - 1L) / divisor * divisor;");
        sb.AppendLine("        return aligned <= global::System.Int32.MaxValue ? (int)aligned : dimension;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine($"    private const int ScaleDivisor = {ScaleDivisor};");
        sb.AppendLine($"    private const int MinimumScaled = {MinimumScaled};");
        sb.AppendLine();
        sb.AppendLine("    private static int Override(");
        sb.AppendLine("        global::System.Collections.Generic.IReadOnlyDictionary<string, int>? overrides,");
        sb.AppendLine("        string name,");
        sb.AppendLine("        int generated)");
        sb.AppendLine("        => overrides is not null && overrides.TryGetValue(name, out var supplied)");
        sb.AppendLine("            ? supplied");
        sb.AppendLine("            : generated;");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Options types carrying at least one generated bound: {typeCount}.</summary>");
        sb.AppendLine($"    public static int BoundedTypeCount => {typeCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Constructor-only types built from synthesized arguments: {synthesizedCount}.</summary>");
        sb.AppendLine($"    public static int SynthesizedTypeCount => {synthesizedCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Total generated knob entries: {knobCount}.</summary>");
        sb.AppendLine($"    public static int KnobCount => {knobCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Total generated divisibility relationships: {constraintCount}.</summary>");
        sb.AppendLine($"    public static int ConstraintCount => {constraintCount};");
        sb.AppendLine("}");

        context.AddSource("ModelTestScale.g.cs", sb.ToString());
    }

    /// <summary>Bound for a knob name, or 0 to leave it alone.</summary>
    /// <remarks>
    /// Case-INSENSITIVE on purpose. This is asked about PascalCase properties and camelCase
    /// constructor parameters alike, and an ordinal match silently skipped every synthesized
    /// constructor argument -- inputHeight never matched InputHeight, so a spatial extent meant to
    /// floor at 32 was capped to 16 and a 32x-downsampling backbone lost its spatial dims entirely.
    /// </remarks>
    /// <summary>Whether a knob may be scaled at all.</summary>
    /// <remarks>
    /// The only name knowledge left, and it is about MEANING rather than magnitude. Two kinds
    /// survive: values whose number IS the semantics (a sample rate is 16000 because audio is
    /// 16 kHz), and INVERSE knobs where shrinking increases work -- a smaller hop or patch cuts the
    /// same input into more pieces, so scaling those down is the one direction that is expensive.
    ///
    /// The count and spatial tiers that used to live here are gone. They existed to stop absolute
    /// caps from breaking relationships (width % heads) and from collapsing a downsampled input to
    /// nothing, and proportional scaling does not have either problem: ratios are preserved and
    /// anything already small is untouched.
    /// </remarks>
    private static bool IsScalable(string propertyName)
    {
        foreach (var fixedName in SemanticallyFixed)
        {
            if (propertyName.IndexOf(fixedName, System.StringComparison.OrdinalIgnoreCase) >= 0)
                return false;
        }

        return true;
    }
}
