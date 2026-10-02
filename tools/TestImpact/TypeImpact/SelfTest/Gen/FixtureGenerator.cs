using Microsoft.CodeAnalysis;

namespace Fixture.Gen;

[Generator]
public sealed class FixtureGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var variant = context.AnalyzerConfigOptionsProvider.Select((options, _) =>
            options.GlobalOptions.TryGetValue("build_property.FixtureGenVariant", out var value) ? value : "1");
        context.RegisterSourceOutput(variant, (output, value) =>
        {
            output.AddSource("Stamped.g.cs", $"namespace Fixture;\npublic static class Stamped\n{{\n    public static int Value() => {value};\n}}\n");
            output.AddSource("Steady.g.cs", "namespace Fixture;\npublic static class Steady\n{\n    public static int Value() => 7;\n}\n");
        });
    }
}