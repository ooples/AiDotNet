#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Image;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for image safety modules.
/// Tests CLIPImageSafetyClassifier, ViTImageSafetyClassifier, SceneGraphSafetyClassifier,
/// EnsembleImageSafetyClassifier, FrequencyDeepfakeDetector, ConsistencyDeepfakeDetector,
/// and ProvenanceDeepfakeDetector with various image tensors.
/// </summary>
public class ImageSafetyIntegrationTests
{
    #region CLIPImageSafetyClassifier Tests

    [Fact]
    public void CLIP_SmallImage_ProcessesWithoutError()
    {
        var classifier = new CLIPImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 8, 8);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void CLIP_MediumImage_ProcessesWithoutError()
    {
        var classifier = new CLIPImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void CLIP_CustomThresholds_Work()
    {
        var classifier = new CLIPImageSafetyClassifier<double>(
            nsfwThreshold: 0.5, violenceThreshold: 0.5);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region ViTImageSafetyClassifier Tests

    [Fact]
    public void ViT_StandardImage_ProcessesWithoutError()
    {
        var classifier = new ViTImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void ViT_SmallPatchSize_ProcessesWithoutError()
    {
        var classifier = new ViTImageSafetyClassifier<double>(patchSize: 8);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void ViT_CustomThreshold_Works()
    {
        var classifier = new ViTImageSafetyClassifier<double>(threshold: 0.3);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region SceneGraphSafetyClassifier Tests

    [Fact]
    public void SceneGraph_StandardImage_ProcessesWithoutError()
    {
        var classifier = new SceneGraphSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void SceneGraph_CustomGrid_Works()
    {
        var classifier = new SceneGraphSafetyClassifier<double>(threshold: 0.4, gridSize: 4);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region EnsembleImageSafetyClassifier Tests

    [Fact]
    public void Ensemble_StandardImage_ProcessesWithoutError()
    {
        var classifier = new EnsembleImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Ensemble_LargerImage_ProcessesWithoutError()
    {
        var classifier = new EnsembleImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 64, 64);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region FrequencyDeepfakeDetector Tests

    [Fact]
    public void FrequencyDeepfake_StandardImage_ProcessesWithoutError()
    {
        var detector = new FrequencyDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void FrequencyDeepfake_SmallImage_HandlesGracefully()
    {
        var detector = new FrequencyDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 8, 8, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region ConsistencyDeepfakeDetector Tests

    [Fact]
    public void ConsistencyDeepfake_StandardImage_ProcessesWithoutError()
    {
        var detector = new ConsistencyDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void ConsistencyDeepfake_UniformImage_ProcessesWithoutError()
    {
        var detector = new ConsistencyDeepfakeDetector<double>();
        var data = new double[3 * 32 * 32];
        for (int i = 0; i < data.Length; i++) data[i] = 128.0;
        var tensor = new Tensor<double>(data, new[] { 3, 32, 32 });
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region ProvenanceDeepfakeDetector Tests

    [Fact]
    public void ProvenanceDeepfake_StandardImage_ProcessesWithoutError()
    {
        var detector = new ProvenanceDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void ProvenanceDeepfake_DifferentSizes_HandleGracefully()
    {
        var detector = new ProvenanceDeepfakeDetector<double>();

        foreach (var size in new[] { 8, 16, 32, 64 })
        {
            var tensor = CreateRandomImageTensor(3, size, size, scale: 255.0);
            var findings = detector.EvaluateImage(tensor);
            Assert.NotNull(findings);
        }
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllClassifiers_SameImage_ProduceResults()
    {
        var tensor = CreateRandomImageTensor(3, 32, 32);

        Assert.NotNull(new CLIPImageSafetyClassifier<double>().EvaluateImage(tensor));
        Assert.NotNull(new ViTImageSafetyClassifier<double>().EvaluateImage(tensor));
        Assert.NotNull(new SceneGraphSafetyClassifier<double>().EvaluateImage(tensor));
        Assert.NotNull(new EnsembleImageSafetyClassifier<double>().EvaluateImage(tensor));
    }

    [Fact]
    public void AllDeepfakeDetectors_SameImage_ProduceResults()
    {
        var tensor = CreateRandomImageTensor(3, 32, 32, scale: 255.0);

        Assert.NotNull(new FrequencyDeepfakeDetector<double>().EvaluateImage(tensor));
        Assert.NotNull(new ConsistencyDeepfakeDetector<double>().EvaluateImage(tensor));
        Assert.NotNull(new ProvenanceDeepfakeDetector<double>().EvaluateImage(tensor));
    }

    #endregion

    #region Helpers

    private static Tensor<double> CreateRandomImageTensor(
        int channels, int height, int width, double scale = 1.0)
    {
        var data = new double[channels * height * width];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = rng.NextDouble() * scale;
        }

        return new Tensor<double>(data, new[] { channels, height, width });
    }

    #endregion
}
