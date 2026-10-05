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

    /// <summary>Training steps the overfit test may take before the transcription must be exact.</summary>
    protected virtual int OverfitStepBudget => 800;

    /// <summary>
    /// The paper objective has to fit: trained through <c>TrainRecognition</c> on one rendered word, the
    /// recogniser must come to read it exactly (CER 0) from a start that misreads it. A CTC alignment or
    /// teacher-forcing shift that is wrong, or labels mapped through the wrong character table, leaves CER above
    /// zero however long it trains. CER is measured on the decoded text, not on the loss.
    /// </summary>
    [Fact(Timeout = 300000)]
    public async Task TrainRecognition_OverfitsOneWord_ToZeroCharacterErrorRate()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var recognizer = CreateRecognizer();
        const string word = "TILE";
        Assert.All(word, ch => Assert.True(recognizer.CharacterSet.IndexOf(ch) >= 0,
            $"'{ch}' is not in {recognizer.GetType().Name}'s character set, so it cannot be a training target."));
        var image = new Tensor<T>(InputShape);
        SyntheticTextImages.Draw(image, 0, word, 4, 5, 3);
        var transcription = new[] { word };

        double CharacterErrorRate()
        {
            recognizer.SetTrainingMode(false);
            // update_bn: eval-mode statistics for the current weights, not the momentum average of earlier ones.
            BatchNormRecalibration.Recalibrate<T>(recognizer, () => recognizer.Predict(image));
            return AiDotNet.Metrics.TextRecognitionMetrics.CharacterErrorRate(word, recognizer.RecognizeText(image).text);
        }

        double initial = CharacterErrorRate();
        var trajectory = new List<string> { $"0: cer {initial:F3}" };
        double final = initial;
        for (int step = 1; step <= OverfitStepBudget && final > 0; step++)
        {
            recognizer.TrainRecognition(image, transcription);
            double loss = ToD(recognizer.GetLastLoss());
            Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"Step {step} reported a non-finite loss {loss}.");
            if (step % 5 == 0 || step == OverfitStepBudget)
            {
                final = CharacterErrorRate();
                trajectory.Add($"{step}: loss {loss:G4} cer {final:F3} '{recognizer.RecognizeText(image).text}'");
            }
        }

        Assert.True(initial > 0, $"The untrained recogniser already reads '{word}', so this fixture cannot show learning.");
        Assert.True(final == 0, $"After {OverfitStepBudget} steps CER is {final:F3}. " + string.Join("; ", trajectory));
    }
    /// <summary>
    /// <c>RecognizeText</c> is what the end-to-end readers call with raw crops, so it must prepare the crop
    /// exactly as <c>Recognize</c> and training do. A recogniser that encoded the raw pixels instead read images
    /// unlike any it was trained on.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task RecognizeText_PreparesTheCropAsRecognizeDoes()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var recognizer = CreateRecognizer();
        var image = CreateRandomImage(rng);
        for (int i = 0; i < image.Length; i++) image[i] = ToT(ToD(image[i]) * 255.0);

        var whole = recognizer.Recognize(image);
        var (text, confidence) = recognizer.RecognizeText(image);

        Assert.Equal(whole.FullText, text);
        // Recognize reports no region for empty text, so the confidences are comparable only when one was read.
        if (text.Length > 0)
            Assert.Equal(ToD(Assert.Single(whole.TextRegions).Confidence), ToD(confidence), 10);
    }

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

        var image = CreateRandomImage(rng);
        var result = recognizer.Recognize(image);

        Assert.Equal(image.Shape[3], result.ImageWidth);
        Assert.Equal(image.Shape[2], result.ImageHeight);
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
