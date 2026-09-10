using AiDotNet.Evolution.Programs;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramEditBoundaryTests
{
    private const string Source = "header\r\n# EVOLVE-BLOCK-START\nold\r# EVOLVE-BLOCK-END\r\nbetween\n# EVOLVE-BLOCK-START\r\nsecond\n# EVOLVE-BLOCK-END\rtail";

    [Theory]
    [InlineData("new")]
    [InlineData("new\r\nlonger\nbody")]
    [InlineData("")]
    public void Multiple_bodies_may_change_without_normalizing_protected_newlines(string body)
    {
        string changed = Source.Replace("old", body).Replace("second\n", "");
        Assert.True(ProgramEditBoundary.PreservesProtectedText(Source, changed, EvolveBlockMarkers.Hash));
        Assert.True(ProgramEditBoundary.PreservesProtectedText(Source, Source, EvolveBlockMarkers.Hash));
    }

    [Theory]
    [InlineData("header", "Header")]
    [InlineData("between", "between ")]
    [InlineData("tail", "tail\n")]
    [InlineData("header\r\n", "header\n")]
    [InlineData("# EVOLVE-BLOCK-START\n", "# EVOLVE-BLOCK-START\r\n")]
    [InlineData("# EVOLVE-BLOCK-END\r\n", "# EVOLVE-BLOCK-END\n")]
    [InlineData("# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-START altered")]
    [InlineData("old", "# EVOLVE-BLOCK-END\n# EVOLVE-BLOCK-START\nold")]
    [InlineData("# EVOLVE-BLOCK-END", "missing")]
    public void Protected_text_marker_structure_and_exact_trivia_cannot_change(string search, string replacement)
    {
        Assert.False(ProgramEditBoundary.PreservesProtectedText(Source, Source.Replace(search, replacement), EvolveBlockMarkers.Hash));
    }

    [Theory]
    [InlineData("ordinary source")]
    [InlineData("# EVOLVE-BLOCK-START\nunterminated")]
    [InlineData("# EVOLVE-BLOCK-END\n# EVOLVE-BLOCK-START")]
    public void Invalid_parent_cannot_be_used_to_claim_a_protected_edit(string source)
    {
        Assert.False(ProgramEditBoundary.PreservesProtectedText(source, Source, EvolveBlockMarkers.Hash));
    }

    [Fact]
    public void Empty_body_and_final_marker_without_newline_support_custom_markers()
    {
        var markers = new EvolveBlockMarkers("// begin", "// finish");
        const string parent = "// begin\n// finish";
        Assert.True(ProgramEditBoundary.PreservesProtectedText(parent, "// begin\nx\n// finish", markers));
        Assert.False(ProgramEditBoundary.PreservesProtectedText(parent, "// begin\nx\n// finish ", markers));
    }
}
