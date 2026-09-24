using Xunit;

namespace PrototypeTests;

public sealed class CustomCaseTests
{
    [SkippableFact, Trait("Scenario", "CustomSkip")]
    public void CustomRunnerStillSkips() => Skip.If(true, "Deliberate skip: the attribution runner must preserve the custom case runner.");
}
