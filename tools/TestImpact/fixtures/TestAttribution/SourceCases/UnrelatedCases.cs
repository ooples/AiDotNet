using Xunit;

namespace SourceCases;

// The proof workload filters to SourceCases.Cases. Neither this fact nor its
// reflection helper may silently become a fixture of that different class.
public sealed class UnrelatedCases
{
    [Fact]
    public void OutsideTheSelectedWorkload() => InspectUnrelatedType();

    private static void InspectUnrelatedType() => _ = typeof(UnrelatedCases).GetMethods();
}
