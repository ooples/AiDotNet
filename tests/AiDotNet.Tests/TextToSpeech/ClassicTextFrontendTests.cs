using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class ClassicTextFrontendTests
{
    [Fact]
    public void FastSpeech_UsesCmuPronunciationRatherThanLetters()
    {
        int[] ids = ClassicTtsText.Phonemes("hello", false, null, 51, 100);
        Assert.Equal(new[] { "HH", "AH", "L", "OW", "<eos>" }, ids.Select(i => ClassicTtsText.FastSpeechSymbols[i]));
        Assert.Equal(51, ClassicTtsText.FastSpeechSymbols.Length);
    }

    [Fact]
    public void FastSpeech2_RetainsLexicalStressAndTrainingInventory()
    {
        int[] ids = ClassicTtsText.Phonemes("hello", true, null, 76, 100);
        Assert.Equal(new[] { "HH", "AH0", "L", "OW1", "<eos>" }, ids.Select(i => ClassicTtsText.FastSpeech2Symbols[i]));
        Assert.Equal(76, ClassicTtsText.FastSpeech2Symbols.Length);
    }

    [Fact]
    public void CustomVocabulary_DoesNotSilentlyFoldUnrepresentablePhones()
    {
        Assert.Throws<ArgumentException>(() => ClassicTtsText.Phonemes("hello", true, null, 8, 100));
    }

    [Fact]
    public void AlignTts_NormalizesNumbersAndFramesCharacters()
    {
        int[] ids = ClassicTtsText.Characters("2 apples", null, 256, 100);
        Assert.Equal(" two apples.", string.Concat(ids.Select(i => ClassicTtsText.AlignSymbols[i])));
    }
}
