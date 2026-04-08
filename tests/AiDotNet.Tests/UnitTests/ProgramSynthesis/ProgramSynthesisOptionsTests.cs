using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class ProgramSynthesisOptionsTests
{
    [Fact]
    public void Defaults_AreIndustryStandardAndStable()
    {
        var options = new ProgramSynthesisOptions();

        Assert.Equal(ProgramSynthesisModelKind.CodeT5, options.ModelKind);
        Assert.Equal(ProgramLanguage.Generic, options.TargetLanguage);
        Assert.Equal(CodeTask.Generation, options.DefaultTask);
        Assert.Equal(SynthesisType.Neural, options.SynthesisType);
        Assert.Equal(512, options.MaxSequenceLength);
        Assert.Equal(50000, options.VocabularySize);
        Assert.Equal(6, options.NumEncoderLayers);
        Assert.Equal(6, options.NumDecoderLayers);
        Assert.Null(options.Tokenizer);
    }
}

