using System.Text.Json;
using Xunit;

namespace PrototypeTests;

// This diagnostic is deliberately outside the normal positive workload. Its
// provider has no side effect unless the dedicated probe explicitly opts in.
public sealed class DiscoveryIsolationProbe
{
    private static int discoveries;
    public static IEnumerable<object[]> Rows
    {
        get
        {
            string? output = Environment.GetEnvironmentVariable("ATTRIBUTION_DISCOVERY_PROBE");
            if (!string.IsNullOrWhiteSpace(output))
            {
                discoveries++;
                Record(output, ProbeStage.Discovery);
            }
            yield return new object[] { 1 };
        }
    }

    [Theory, MemberData(nameof(Rows)), Trait("Scenario", "DiscoverySideEffect")]
    public void OutsideTheSelectedWorkload(int value) => Assert.Equal(1, value);

    [Fact, Trait("Scenario", "DiscoveryProbe")]
    public void ObserveExecutionProcess()
    {
        string output = Environment.GetEnvironmentVariable("ATTRIBUTION_DISCOVERY_PROBE")
            ?? throw new InvalidOperationException("The discovery probe requires an explicit output path.");
        Record(output, ProbeStage.Execution);
    }

    private static void Record(string output, ProbeStage stage) => File.AppendAllText(output,
        JsonSerializer.Serialize(new { Stage = stage.ToString(), Process = Environment.ProcessId, Discoveries = discoveries }) + Environment.NewLine);

    private enum ProbeStage { Discovery, Execution }
}
