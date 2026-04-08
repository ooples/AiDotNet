#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Image;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using AiDotNet.Tensors.Helpers;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task CLIP_SmallImage_ProcessesWithoutError()
    {
        var classifier = new CLIPImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 8, 8);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task CLIP_MediumImage_ProcessesWithoutError()
    {
        var classifier = new CLIPImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task CLIP_CustomThresholds_Work()
    {
        var classifier = new CLIPImageSafetyClassifier<double>(
            nsfwThreshold: 0.5, violenceThreshold: 0.5);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region ViTImageSafetyClassifier Tests

    [Fact(Timeout = 120000)]
    public async Task ViT_StandardImage_ProcessesWithoutError()
    {
        var classifier = new ViTImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task ViT_SmallPatchSize_ProcessesWithoutError()
    {
        var classifier = new ViTImageSafetyClassifier<double>(patchSize: 8);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task ViT_CustomThreshold_Works()
    {
        var classifier = new ViTImageSafetyClassifier<double>(threshold: 0.3);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region SceneGraphSafetyClassifier Tests

    [Fact(Timeout = 120000)]
    public async Task SceneGraph_StandardImage_ProcessesWithoutError()
    {
        var classifier = new SceneGraphSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task SceneGraph_CustomGrid_Works()
    {
        var classifier = new SceneGraphSafetyClassifier<double>(threshold: 0.4, gridSize: 4);
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region EnsembleImageSafetyClassifier Tests

    [Fact(Timeout = 120000)]
    public async Task Ensemble_StandardImage_ProcessesWithoutError()
    {
        var classifier = new EnsembleImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 16, 16);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task Ensemble_LargerImage_ProcessesWithoutError()
    {
        var classifier = new EnsembleImageSafetyClassifier<double>();
        var tensor = CreateRandomImageTensor(3, 64, 64);
        var findings = classifier.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region FrequencyDeepfakeDetector Tests

    [Fact(Timeout = 120000)]
    public async Task FrequencyDeepfake_StandardImage_ProcessesWithoutError()
    {
        var detector = new FrequencyDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task FrequencyDeepfake_SmallImage_HandlesGracefully()
    {
        var detector = new FrequencyDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 8, 8, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region ConsistencyDeepfakeDetector Tests

    [Fact(Timeout = 120000)]
    public async Task ConsistencyDeepfake_StandardImage_ProcessesWithoutError()
    {
        var detector = new ConsistencyDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task ConsistencyDeepfake_UniformImage_ProcessesWithoutError()
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

    [Fact(Timeout = 120000)]
    public async Task ProvenanceDeepfake_StandardImage_ProcessesWithoutError()
    {
        var detector = new ProvenanceDeepfakeDetector<double>();
        var tensor = CreateRandomImageTensor(3, 32, 32, scale: 255.0);
        var findings = detector.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task ProvenanceDeepfake_DifferentSizes_HandleGracefully()
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

    [Fact(Timeout = 120000)]
    public async Task AllClassifiers_SameImage_ProduceResults()
    {
        var tensor = CreateRandomImageTensor(3, 32, 32);

        Assert.NotNull(new CLIPImageSafetyClassifier<double>().EvaluateImage(tensor));
        Assert.NotNull(new ViTImageSafetyClassifier<double>().EvaluateImage(tensor));
        Assert.NotNull(new SceneGraphSafetyClassifier<double>().EvaluateImage(tensor));
        Assert.NotNull(new EnsembleImageSafetyClassifier<double>().EvaluateImage(tensor));
    }

    [Fact(Timeout = 120000)]
    public async Task AllDeepfakeDetectors_SameImage_ProduceResults()
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
