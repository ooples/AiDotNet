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
    public void Left()
    {
#if SOURCE_ALTERNATIVE
        Equal(4, 2 + 2);
#else
        Equal(2, 1 + 1);
#endif
    }

    [Fact]
    public void Right()
    {
        Equal(4, 2 + 2);
    }

    [Fact]
    public void Third()
    {
        Equal(6, 3 + 3);
    }

    // A closed IL-only oracle: unequal values fail with DivideByZeroException.
    // This deliberately avoids claiming that an external assertion library is
    // pure. Unknown external calls remain conservative selection boundaries.
    private static void Equal(int expected, int actual)
    {
        _ = 1 / (expected == actual ? 1 : 0);
    }
}
