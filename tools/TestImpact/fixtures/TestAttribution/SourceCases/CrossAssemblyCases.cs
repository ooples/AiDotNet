using Xunit;

[assembly: TestFramework("AiDotNet.TestImpact.Xunit.AttributionTestFramework", "Attribution.Xunit")]
#if METADATA_ALTERNATIVE
[assembly: System.Reflection.AssemblyMetadata("source-proof", "alternative")]
#else
[assembly: System.Reflection.AssemblyMetadata("source-proof", "default")]
#endif

namespace SourceCases;

public sealed class Cases
{
    [Fact]
    public void Left() => Equal(2, SourceLibrary.Subject.Left(1));

    [Fact]
    public void Right() => Equal(4, SourceLibrary.Subject.Right(2));

    [Fact]
    public void Third() => Equal(6, SourceLibrary.Subject.Third(3));

    private static void Equal(int expected, int actual) => _ = 1 / (expected == actual ? 1 : 0);
}
