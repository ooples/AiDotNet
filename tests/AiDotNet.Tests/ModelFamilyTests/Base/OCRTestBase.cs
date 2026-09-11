using AiDotNet.ComputerVision.OCR;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for text recognition / OCR models (CRNN, TrOCR).
/// </summary>
/// <remarks>
/// <para>
/// A recognizer turns a cropped image into a string, so its invariants are about the string:
/// every character it emits must come from the character set it was configured with, the
/// sequence must respect the decoder length bound, and the per-character confidences must line
/// up with the characters they describe. A model that emits a character outside its own
/// vocabulary has decoded an index past the end of its label array - the failure that produces
/// mojibake in output rather than an exception.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type the recognizer is expressed in.</typeparam>
public abstract class OCRTestBase<T> : DetectionModelTestBase<T>
    where T : struct
{
    /// <summary>
    /// OCR crops are wide and short - a line of text, not a square. The recognition height is
    /// what the model resizes to, so the fixture feeds something already in that aspect range.
    /// </summary>
    protected override int[] InputShape => [1, 3, 32, 128];

    /// <summary>
    /// The model under test as a recognizer. Family resolution guarantees the cast.
    /// </summary>
    protected OCRBase<T> CreateRecognizer() => (OCRBase<T>)CreateModel();

    [Fact(Timeout = 120000)]
    public async Task Recognize_ShouldReturnAWellFormedResult()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();

        var result = recognizer.Recognize(CreateRandomImage(rng));

        Assert.NotNull(result);
        Assert.NotNull(result.TextRegions);
        Assert.NotNull(result.FullText);
        foreach (var region in result.TextRegions)
        {
            Assert.NotNull(region.Text);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Recognize_ShouldOnlyEmitCharactersFromItsCharacterSet()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();

        var result = recognizer.Recognize(CreateRandomImage(rng));
        string alphabet = recognizer.CharacterSet;

        Assert.False(string.IsNullOrEmpty(alphabet), "Recognizer exposes an empty character set.");

        foreach (var region in result.TextRegions)
        {
            foreach (char c in region.Text)
            {
                Assert.True(
                    alphabet.IndexOf(c) >= 0,
                    $"Recognized character '{c}' (U+{(int)c:X4}) is not in the model character "
                    + "set. The decoder indexed past the end of its label array.");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Recognize_ShouldRespectMaxSequenceLength()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();

        var result = recognizer.Recognize(CreateRandomImage(rng));

        foreach (var region in result.TextRegions)
        {
            Assert.True(
                region.Text.Length <= recognizer.MaxSequenceLength,
                $"Recognized {region.Text.Length} characters, above the decoder bound of "
                + $"{recognizer.MaxSequenceLength}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Recognize_ConfidencesShouldBeInUnitRange()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();

        var result = recognizer.Recognize(CreateRandomImage(rng));

        foreach (var region in result.TextRegions)
        {
            Assert.InRange(ToD(region.Confidence), 0.0, 1.0);

            foreach (var characterConfidence in region.CharacterConfidences)
            {
                Assert.InRange(ToD(characterConfidence), 0.0, 1.0);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Recognize_CharacterConfidencesShouldAlignWithTheText()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();

        var result = recognizer.Recognize(CreateRandomImage(rng));

        foreach (var region in result.TextRegions)
        {
            if (region.CharacterConfidences.Count == 0)
            {
                continue; // Per-character confidence is optional.
            }

            // When present it is indexed by character position, so a length mismatch means the
            // caller reads a confidence belonging to a different character.
            Assert.Equal(region.Text.Length, region.CharacterConfidences.Count);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Recognize_FullTextShouldAccountForEveryRegion()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();

        var result = recognizer.Recognize(CreateRandomImage(rng));

        foreach (var region in result.TextRegions)
        {
            if (region.Text.Length == 0)
            {
                continue;
            }

            Assert.True(
                result.FullText.Contains(region.Text),
                $"FullText does not contain the recognized region text '{region.Text}', so the "
                + "aggregate view drops content the per-region view reports.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Recognize_ShouldReportTheSourceImageDimensions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();

        var result = recognizer.Recognize(CreateRandomImage(rng));

        Assert.True(result.ImageWidth > 0, "OCRResult.ImageWidth was not populated.");
        Assert.True(result.ImageHeight > 0, "OCRResult.ImageHeight was not populated.");
    }

    [Fact(Timeout = 120000)]
    public async Task Recognize_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();
        var image = CreateRandomImage(rng);

        var first = recognizer.Recognize(image);
        var second = recognizer.Recognize(image);

        Assert.Equal(first.FullText, second.FullText);
        Assert.Equal(first.TextRegions.Count, second.TextRegions.Count);
        for (int i = 0; i < first.TextRegions.Count; i++)
        {
            Assert.Equal(first.TextRegions[i].Text, second.TextRegions[i].Text);
            Assert.Equal(ToD(first.TextRegions[i].Confidence), ToD(second.TextRegions[i].Confidence), 10);
        }
    }
}

/// <summary>Default-precision alias used by the generated fixtures.</summary>
public abstract class OCRTestBase : OCRTestBase<double> { }
