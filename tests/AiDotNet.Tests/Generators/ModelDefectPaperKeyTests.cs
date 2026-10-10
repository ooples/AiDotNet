using AiDotNet.Generators;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// The duplicate-paper key of <see cref="ModelDefectClassAnalyzer"/>: one key per paper whatever link form a model cites.
/// The key used to come from regular expressions with a one-second match timeout, which expired on a saturated build
/// machine and failed the compile with AD0001; these cases pin the string-scanning replacement to the same keys.
/// </summary>
public sealed class ModelDefectPaperKeyTests
{
    [Theory]
    [InlineData("https://arxiv.org/abs/2210.13438", "arxiv:2210.13438")]
    [InlineData("https://arxiv.org/pdf/2210.13438v2", "arxiv:2210.13438")]
    [InlineData("https://arxiv.org/abs/2106.07447/", "arxiv:2106.07447")]
    [InlineData("http://arxiv.org/abs/1703.10135", "arxiv:1703.10135")]
    [InlineData("https://doi.org/10.48550/arXiv.2106.07447", "arxiv:2106.07447")]
    [InlineData("https://arxiv.org/abs/12345.6789", "arxiv.org/abs/12345.6789")]
    [InlineData("https://doi.org/10.1109/TASLP.2021.3129994", "doi:10.1109/taslp.2021.3129994")]
    [InlineData("https://www.isca-speech.org/archive/interspeech_2019/", "isca-speech.org/archive/interspeech_2019")]
    [InlineData("http://example.com/paper.pdf", "example.com/paper.pdf")]
    [InlineData("  Example.org/Paper  ", "example.org/paper")]
    public void NormalizePaper_GivesOneKeyPerPaper(string url, string expected)
        => Assert.Equal(expected, ModelDefectClassAnalyzer.NormalizePaper(url));
}
