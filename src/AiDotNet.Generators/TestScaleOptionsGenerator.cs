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
        "Dimension",    // an axis INDEX, not a width; widths end in Dim/Size
        "Precision",
        "DeviceId",
        "Port",
        "Timeout",
        "Year",
    };

    /// <summary>Name markers for a knob that is a COUNT of things rather than a width.</summary>
    /// <remarks>
    /// A single uniform cap is not enough, and CSM is the proof: capping everything at 16 bounded
    /// its dimensions correctly but also set NumLLMLayers, NumEncoderLayers and NumDecoderLayers to
    /// SIXTEEN, where the hand-written branch used one. Sixteen transformer layers instead of one is
    /// far more expensive than any width, and the model stalled past the drain budget.
    ///
    /// Counts multiply work; widths mostly scale one matmul. So counts get a much smaller cap. The
    /// distinction is still name-based, but the cost of misreading it is now bounded both ways: a
    /// count mistaken for a width is 16 layers instead of 2 (slow, still correct), and a width
    /// mistaken for a count is a 2-wide matrix (small, still correct).
    /// </remarks>
    private static readonly string[] CountMarkers =
    {
        // SUBSTRINGS, not full spellings. Enumerating names could never keep up: Florence2 calls its
        // head count NumDecoderHeads, which matched none of NumHeads / NumAttentionHeads /
        // NumKeyValueHeads, so it kept its default of 12 while DecoderEmbeddingDim was capped to 16.
        // A head count must DIVIDE the embedding width, and 16/12 truncates to a head dimension of
        // 1, giving 12 * 1 = 12 against an actual width of 16 -- MultiHeadAttentionLayer rejects
        // exactly that. Matching any *Heads* and any *Layers* keeps counts at the count cap, and a
        // cap of 2 divides every width bound this generator produces.
        "Heads", "Layers", "Blocks", "Experts", "Codebooks", "Stages", "Groups",
        "Depth", "LayerCount", "BlockCount",
    };

    /// <summary>Cap for a count of repeated structures.</summary>
    /// <remarks>Two, not one: a single layer cannot exercise inter-layer wiring.</remarks>
    private const int CountCap = 2;

    /// <summary>Every other int knob above this is clamped to it.</summary>
    /// <remarks>
    /// One number rather than a per-name table. A test-scale model needs shapes that exercise the
    /// code path, not shapes that resemble the paper: 16 is wide enough that a head count of 2 or 4
    /// still divides it, and small enough that a dozen layers of it cost nothing.
    /// </remarks>
    private const int KnobCap = 16;

    /// <summary>Name markers for a SPATIAL input extent, which needs a floor rather than the cap.</summary>
    /// <remarks>
    /// A classifier backbone downsamples hard -- ResNet reduces by 32x through its stem and four
    /// stages -- so a 16x16 input collapses to zero spatial extent before the classifier and the
    /// model cannot run at all. The hand-written CreateForTesting chose 32x32 for exactly this
    /// reason. 32 is the smallest power of two that survives a 32x reduction.
    /// </remarks>
    private static readonly string[] SpatialMarkers =
    {
        "InputHeight", "InputWidth", "InputSize", "ImageSize", "Resolution",
    };

    /// <summary>Bound for a spatial extent: small, but large enough to survive downsampling.</summary>
    private const int SpatialCap = 32;

    /// <summary>Below this, a value is left alone entirely.</summary>
    /// <remarks>
    /// Counts that are already tiny (2 codebooks, 4 heads) carry structure worth preserving, and
    /// rewriting them to the cap would make some models LARGER, which this must never do.
    /// </remarks>
    private const int LeaveAloneBelow = 17;

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
                    int companionBound = BoundFor(parameter.Name);
                    parts.Add($"{parameter.Name}: {(companionBound > 0 ? companionBound : SpatialCap)}");
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
                var stages = string.Join(", ", Enumerable.Repeat(CountCap.ToString(), BackboneStages));
                parts.Add($"{parameter.Name}: new int[] {{ {stages} }}");
                continue;
            }

            if (parameter.Type.SpecialType != SpecialType.System_Int32) continue;

            int bound = BoundFor(parameter.Name);

            // CLAMP, NEVER RAISE -- for constructor arguments too. A property is compared against
            // its live value; a constructor parameter has only its DECLARED DEFAULT, and skipping
            // that comparison is what made this raise values instead of lowering them. inputChannels
            // defaults to 3 for RGB, and passing the cap of 16 built a stem expecting 16-channel
            // input that every 3-channel image then failed against: "Expected input depth 16, but
            // got 3", 41 tests. When the default is already at or below the bound, say nothing and
            // let the default stand.
            if (bound > 0
                && parameter.HasExplicitDefaultValue
                && parameter.ExplicitDefaultValue is int declaredDefault
                && declaredDefault <= bound)
            {
                continue;
            }

            if (bound <= 0)
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
                $"{parameter.Name}: Override(overrides, \"{parameter.Name}\", {bound})");
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
                .Select(p => (Property: p.Name, Bound: BoundFor(p.Name)))
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
        sb.AppendLine("            // CLAMP, never raise. A value already at or below the cap keeps");
        sb.AppendLine("            // its own number: small counts carry structure (2 codebooks, 4");
        sb.AppendLine("            // heads) and rewriting them could make a model LARGER.");
        sb.AppendLine("            if (current <= Knobs[i].Bound) continue;");
        sb.AppendLine();
        sb.AppendLine("            property.SetValue(instance, Knobs[i].Bound);");
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
        sb.AppendLine("        return bounded ? instance : instance;");
        sb.AppendLine("    }");
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
    private static int BoundFor(string propertyName)
    {
        foreach (var fixedName in SemanticallyFixed)
        {
            if (propertyName.IndexOf(fixedName, System.StringComparison.OrdinalIgnoreCase) >= 0) return 0;
        }

        foreach (var marker in CountMarkers)
        {
            if (propertyName.IndexOf(marker, System.StringComparison.OrdinalIgnoreCase) >= 0) return CountCap;
        }

        foreach (var marker in SpatialMarkers)
        {
            if (propertyName.IndexOf(marker, System.StringComparison.OrdinalIgnoreCase) >= 0) return SpatialCap;
        }

        return KnobCap;
    }
}
