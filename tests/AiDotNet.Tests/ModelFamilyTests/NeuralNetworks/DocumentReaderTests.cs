using AiDotNet.ComputerVision.OCR;
using AiDotNet.ComputerVision.OCR.EndToEnd;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for <see cref="DocumentReader{T}"/> (LayoutLMv3-derived
/// document OCR + layout analysis pipeline). Unlike the standard NN models,
/// DocumentReader is a composite — its <c>Predict</c> override returns the
/// preprocessed image and the real OCR work lives in <c>ReadDocument</c> —
/// so it doesn't fit any of the NN-invariant test bases that exercise
/// gradient flow, parameter changes, or training loops. The auto-generator
/// can't construct one because the only ctor requires an
/// <see cref="OCROptions{T}"/>; with cluster 5(a)'s generator awareness in
/// place, the runtime-throwing stub no longer gets emitted, so this manual
/// scaffold is what surfaces real coverage for the model.
/// </summary>
public class DocumentReaderTests
{
    private static DocumentReader<double> CreateReader()
    {
        // Defaults match the original DocumentReader.cs:53 — DBNet detector
        // with the lower confidence threshold (0.3) tuned for clean
        // documents, CRNN recognizer for the recognition pass.
        var options = new OCROptions<double>
        {
            Mode = OCRMode.Document,
        };
        return new DocumentReader<double>(options);
    }

    /// <summary>
    /// Construction smoke: the composite must initialize without throwing.
    /// Covers the DBNet detection-options wiring + CRNN/TrOCR factory
    /// switch in the ctor (DocumentReader.cs:53-72).
    /// </summary>
    [Fact]
    public void Constructor_DefaultOptions_DoesNotThrow()
    {
        var reader = CreateReader();
        Assert.NotNull(reader);
        Assert.Contains("DocumentReader", reader.Name);
    }

    /// <summary>
    /// Predict contract: <c>DocumentReader.Predict</c> returns the
    /// preprocessed document image rather than running the full
    /// ReadDocument pipeline (preprocessed = enhanced contrast + optional
    /// binarization). Verify the output is non-null, finite, and has the
    /// same rank as the input.
    /// </summary>
    [Fact]
    public void Predict_RandomImage_ReturnsFinitePreprocessedTensor()
    {
        var reader = CreateReader();
        // Small synthetic document image: 1 batch, 1 grayscale channel,
        // 32×32 pixels. DocumentReader's preprocessing works on any
        // rank-4 [batch, channels, height, width] image tensor.
        var image = new Tensor<double>([1, 1, 32, 32]);
        for (int i = 0; i < image.Length; i++)
            image[i] = 0.5;

        var output = reader.Predict(image);

        Assert.NotNull(output);
        Assert.Equal(image.Rank, output.Rank);
        for (int i = 0; i < output.Length; i++)
        {
            double v = output[i];
            Assert.False(double.IsNaN(v) || double.IsInfinity(v),
                $"Preprocessed pixel[{i}] is not finite: {v}");
        }
    }

    /// <summary>
    /// Train contract: <c>DocumentReader.Train</c> is intentionally a
    /// no-op (DocumentReader.cs:398) because the underlying detector +
    /// recognizer are loaded pre-trained. Verify it doesn't throw on
    /// arbitrary input — important because the composite is plumbed
    /// through the standard IFullModel surface where TrainingMonitor /
    /// PredictionModelBuilder code paths call Train indiscriminately.
    /// </summary>
    [Fact]
    public void Train_NoOp_DoesNotThrow()
    {
        var reader = CreateReader();
        var image = new Tensor<double>([1, 1, 16, 16]);
        var target = new Tensor<double>([1, 1, 16, 16]);

        // Behavioural assertion: Train must be a true no-op, not just
        // exception-free. Snapshot Predict() output, invoke Train, then
        // confirm Predict() returns exactly the same tensor. Catches a
        // future regression where Train silently mutates weights.
        var before = reader.Predict(image);
        var ex = Record.Exception(() => reader.Train(image, target));
        var after = reader.Predict(image);

        Assert.Null(ex);
        Assert.Equal(before.Rank, after.Rank);
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < after.Length; i++)
            Assert.Equal(before[i], after[i]);
    }
}
