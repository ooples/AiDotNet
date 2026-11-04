using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class ViTNormalizationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PredictionAndTrainingApplyEachChannelsNormalization(bool training)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 2, inputWidth: 2, inputDepth: 3, outputSize: 12,
            layers: new List<ILayer<double>> { new ActivationLayer<double>((IActivationFunction<double>)new IdentityActivation<double>()) });
        var options = new ViTOptions
        {
            ImageSize = 2, PatchSize = 1, EmbeddingDim = 3, NumHeads = 1, NumLayers = 1,
            ImageMean = new[] { 0.1, 0.2, 0.3 }, ImageStd = new[] { 0.5, 1.0, 2.0 },
        };
        using var model = new ViT<double>(architecture, options);
        var input = new Tensor<double>(Enumerable.Repeat(0.8, 12).ToArray(), new[] { 1, 3, 2, 2 });
        model.SetTrainingMode(training);
        var output = training ? model.ForwardForTraining(input) : model.Predict(input);
        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        for (int channel = 0; channel < 3; channel++)
            for (int pixel = 0; pixel < 4; pixel++)
                Assert.Equal((0.8 - options.ImageMean[channel]) / options.ImageStd[channel], output[channel * 4 + pixel], 12);
        Assert.All(input.ToArray(), value => Assert.Equal(0.8, value));
    }
}
