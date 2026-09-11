using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>Covers rebasing program-specific relative descriptors onto a new reference set.</summary>
public sealed class EvolutionRebaseTests
{
    [Fact]
    public void RebasingProducesANewDescriptorRatherThanChangingTheOldOne()
    {
        var original = new ProgramDiversityDescriptor(new[] { "def a():\n    return 1\n" });
        IRebasableProgramDescriptor rebased = original.Rebase(new[] { Genome("def b():\n    return 2\n") });

        Assert.NotSame(original, rebased);
        Assert.Equal(original.Name, rebased.Name);
        Assert.NotEqual(original.VersionHash, rebased.VersionHash);
        Assert.Single(original.ReferenceSources);
    }

    [Fact]
    public void RebasingAWholeSetMovesEveryRelativeDescriptorAndLeavesTheAbsoluteOnesAlone()
    {
        var set = new ProgramDescriptorSet(
            new ProgramLengthDescriptor(),
            new ProgramDiversityDescriptor(new[] { "def a():\n    return 1\n" }));

        ProgramDescriptorSet rebased = set.Rebase(new[]
        {
            Genome("def b(values):\n    total = 0\n    for value in values:\n        total += value * 3\n    return total\n")
        });

        Assert.Equal(set.Names, rebased.Names);
        Assert.NotEqual(set.VersionHash, rebased.VersionHash);

        ProgramGenome probe = Genome("def c():\n    return 3\n");
        Assert.Equal(set.Compute(probe)["length"], rebased.Compute(probe)["length"], 10);
        Assert.NotEqual(set.Compute(probe)["diversity"], rebased.Compute(probe)["diversity"]);
    }

    [Fact]
    public void ASetOfAbsoluteDescriptorsHasNothingToRebase()
    {
        var set = new ProgramDescriptorSet(new ProgramLengthDescriptor());

        Assert.False(set.HasRebasableDescriptors);
        Assert.Same(set, set.Rebase(new[] { Genome("def b():\n    return 2\n") }));
    }

    private static ProgramGenome Genome(string source) => new(source, ProgramLanguage.Python);
}
