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
        "NumLayers", "NumHiddenLayers", "NumEncoderLayers", "NumDecoderLayers", "NumVisionLayers",
        "NumLLMLayers", "NumBlocks", "NumExperts", "NumCodebooks", "NumStages", "NumGroups",
        "NumHeads", "NumAttentionHeads", "NumKeyValueHeads", "Depth", "LayerCount", "BlockCount",
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

    /// <summary>Below this, a value is left alone entirely.</summary>
    /// <remarks>
    /// Counts that are already tiny (2 codebooks, 4 heads) carry structure worth preserving, and
    /// rewriting them to the cap would make some models LARGER, which this must never do.
    /// </remarks>
    private const int LeaveAloneBelow = 17;

    /// <inheritdoc/>
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var candidates = context.SyntaxProvider.CreateSyntaxProvider(
                static (node, _) => node is ClassDeclarationSyntax c
                    && !c.Modifiers.Any(m => m.ValueText == "abstract")
                    && !c.Modifiers.Any(m => m.ValueText == "static")
                    && (c.Identifier.ValueText.EndsWith("Options")
                        || c.Identifier.ValueText.EndsWith("Configuration")),
                static (ctx, _) => (INamedTypeSymbol?)ctx.SemanticModel.GetDeclaredSymbol(ctx.Node))
            .Where(static symbol => symbol is not null && HasParameterlessConstructor(symbol));

        context.RegisterSourceOutput(candidates.Collect(), static (spc, types) => Execute(spc, types!));
    }

    private static bool HasParameterlessConstructor(INamedTypeSymbol? symbol)
    {
        if (symbol is null || symbol.IsAbstract || symbol.DeclaredAccessibility != Accessibility.Public)
            return false;

        // Only a type the harness can actually new up is worth an entry.
        return symbol.InstanceConstructors.Any(c =>
            c.Parameters.Length == 0 && c.DeclaredAccessibility == Accessibility.Public);
    }

    private static void Execute(SourceProductionContext context, ImmutableArray<INamedTypeSymbol?> types)
    {
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
        sb.AppendLine("    public static object? CreateBoundedOptions(global::System.Type optionsType)");
        sb.AppendLine("    {");
        sb.AppendLine("        if (optionsType is null) return null;");
        sb.AppendLine();
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
        sb.AppendLine("        return bounded ? instance : instance;");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Options types carrying at least one generated bound: {typeCount}.</summary>");
        sb.AppendLine($"    public static int BoundedTypeCount => {typeCount};");
        sb.AppendLine();
        sb.AppendLine($"    /// <summary>Total generated knob entries: {knobCount}.</summary>");
        sb.AppendLine($"    public static int KnobCount => {knobCount};");
        sb.AppendLine("}");

        context.AddSource("ModelTestScale.g.cs", sb.ToString());
    }

    private static int BoundFor(string propertyName)
    {
        foreach (var fixedName in SemanticallyFixed)
        {
            if (propertyName.IndexOf(fixedName, System.StringComparison.Ordinal) >= 0) return 0;
        }

        foreach (var marker in CountMarkers)
        {
            if (propertyName.IndexOf(marker, System.StringComparison.Ordinal) >= 0) return CountCap;
        }

        return KnobCap;
    }
}
