using Microsoft.CodeAnalysis;

namespace FixtureGen;

[Generator]
public sealed class GreetingGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context) =>
        context.RegisterPostInitializationOutput(output => output.AddSource("Greeting.g.cs",
            GenHelpers.Header + "\nnamespace Fixture;\n\npublic static class Greeting\n{\n    public static string Hello() => \"hello\";\n}\n"));
}
