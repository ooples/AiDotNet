using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class VisionLanguageNativeConstructionTests
{
    [Fact]
    public void VisionMamba_SmallNativeConstructorDoesNotRequireVocabulary()
    {
        var model = new VisionMambaModel<float>(ImageArchitecture(16, 1, 3), new VisionMambaOptions
        {
            ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1,
            ModelDimension = 16, NumLayers = 1, StateDimension = 4, NumClasses = 3
        });

        Assert.Equal(16, model.NumPatches);
        Assert.Equal(1, model.NumLayers);
        Assert.NotEmpty(model.Layers);
        var prediction = model.Predict(new Tensor<float>(new[] { 1, 1, 16, 16 }));
        Assert.Equal(new[] { 1, 3 }, prediction.Shape.ToArray());
    }

    [Fact]
    public void Correspondence_SmallNativeConstructorDoesNotRequireTextOrImageOptions()
    {
        var model = new AudioVisualCorrespondenceNetwork<float>(VectorArchitecture(32, 2),
            new AudioVisualCorrespondenceOptions { EmbeddingDimension = 16, NumEncoderLayers = 1 });

        Assert.Equal(16, model.EmbeddingDimension);
        Assert.NotEmpty(model.Layers);
        var prediction = model.Predict(new Tensor<float>(new[] { 1, 32 }));
        Assert.Equal(2, prediction.Shape[prediction.Shape.Length - 1]);
    }

    [Fact]
    public void EventLocalization_SmallNativeConstructorBuildsRealDualStreamLayers()
    {
        var model = new AudioVisualEventLocalizationNetwork<float>(VectorArchitecture(16, 3),
            new AudioVisualEventLocalizationOptions
            {
                EmbeddingDimension = 16, NumEncoderLayers = 1,
                AudioEmbeddingFullyConnectedWidth = 8, AudioEmbeddingSize = 8
            });

        Assert.Equal(20, model.Layers.Count);
        Assert.Equal(10, model.Layers.OfType<MultiHeadAttentionLayer<float>>().Count());
    }

    [Fact]
    public void UnifiedMultimodal_SmallNativeConstructorDoesNotRequireImageGeometry()
    {
        var model = new UnifiedMultimodalNetwork<float>(VectorArchitecture(8, 3),
            new UnifiedMultimodalNetworkOptions
            {
                EmbeddingDimension = 8, MaxSequenceLength = 8, NumTransformerLayers = 1
            });

        Assert.Equal(16, model.Layers.Count);
        Assert.Equal(5, model.Layers.OfType<MultiHeadAttentionLayer<float>>().Count());
        var prediction = model.Predict(new Tensor<float>(new[] { 1, 8 }));
        Assert.Equal(3, prediction.Shape[prediction.Shape.Length - 1]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void EventLocalization_EncoderDepthChangesTheActualAttentionGraph(int layers)
    {
        var model = new AudioVisualEventLocalizationNetwork<float>(VectorArchitecture(16, 3),
            new AudioVisualEventLocalizationOptions
            {
                EmbeddingDimension = 16, NumEncoderLayers = layers,
                AudioEmbeddingFullyConnectedWidth = 8, AudioEmbeddingSize = 8
            });

        Assert.Equal(18 + 2 * layers, model.Layers.Count);
        Assert.Equal(8 + 2 * layers, model.Layers.OfType<MultiHeadAttentionLayer<float>>().Count());
    }

    [Fact]
    public void UnifiedMultimodal_WidthAndDepthAlterRealLayers()
    {
        var small = new UnifiedMultimodalNetwork<float>(VectorArchitecture(8, 3),
            new UnifiedMultimodalNetworkOptions { EmbeddingDimension = 8, MaxSequenceLength = 8, NumTransformerLayers = 1 });
        var wide = new UnifiedMultimodalNetwork<float>(VectorArchitecture(8, 3),
            new UnifiedMultimodalNetworkOptions { EmbeddingDimension = 16, MaxSequenceLength = 8, NumTransformerLayers = 2 });

        small.Predict(new Tensor<float>(new[] { 1, 8 }));
        wide.Predict(new Tensor<float>(new[] { 1, 8 }));
        Assert.Equal(8, small.Layers[0].GetOutputShape().Last());
        Assert.Equal(16, wide.Layers[0].GetOutputShape().Last());
        Assert.Equal(16, small.Layers.Count);
        Assert.Equal(17, wide.Layers.Count);
        Assert.True(wide.Layers[0].GetParameters().Length > small.Layers[0].GetParameters().Length);
    }

    [Fact]
    public void VisionMamba_WidthAndDepthAlterMaterializedParameters()
    {
        var small = new VisionMambaModel<float>(ImageArchitecture(16, 1, 3), new VisionMambaOptions
        {
            ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1,
            ModelDimension = 8, NumLayers = 1, StateDimension = 4, NumClasses = 3
        });
        var wide = new VisionMambaModel<float>(ImageArchitecture(16, 1, 3), new VisionMambaOptions
        {
            ImageHeight = 16, ImageWidth = 16, PatchSize = 4, Channels = 1,
            ModelDimension = 16, NumLayers = 2, StateDimension = 4, NumClasses = 3
        });

        small.Predict(new Tensor<float>(new[] { 1, 1, 16, 16 }));
        wide.Predict(new Tensor<float>(new[] { 1, 1, 16, 16 }));
        Assert.True(wide.Layers.Count > small.Layers.Count);
        Assert.True(wide.GetParameters().Length > small.GetParameters().Length);
    }

    [Theory]
    [InlineData(32, 8, 16)]
    [InlineData(31, 8, 9)]
    public void Flamingo_ConfiguredPatchesControlTheRealVisionTokenCount(int imageSize, int patchSize, int tokens)
    {
        var model = CreateSmallFlamingo(imageSize, patchSize);
        var activations = model.GetNamedLayerActivations(new Tensor<float>(new[] { 3, imageSize, imageSize }));

        Assert.Equal(new[] { tokens, 32 }, activations["vision_features"].Shape.ToArray());
        Assert.Equal(new[] { 4, 32 }, activations["perceiver_features"].Shape.ToArray());
        // The first attention block belongs to the resampler; the second is the actual LM gate.
        Assert.Equal(2, model.Layers.OfType<CrossAttentionLayer<float>>().Count());
    }

    [Fact]
    public void Flamingo_PublicGenerationActuallySuppliesImageContextToTheLanguageGate()
    {
        var model = CreateSmallFlamingo(32, 8);
        var attention = model.Layers.OfType<CrossAttentionLayer<float>>().ToArray();
        Assert.Equal(2, attention.Length);
        var languageGate = attention[1];
        var contextField = typeof(CrossAttentionLayer<float>).GetField("_lastContext", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("The actual attention context cache is required for this execution probe.");
        Assert.Null(contextField.GetValue(languageGate));

        var firstImage = new Tensor<float>(new[] { 3, 32, 32 });
        var secondImage = new Tensor<float>(new[] { 3, 32, 32 });
        for (int i = 0; i < secondImage.Length; i++)
            secondImage[i] = (i % 31 + 1) / 31.0f;

        // Greedy decoding may legitimately return EOS immediately. Its string is not the oracle:
        // observing the actual LM gate's context proves that this public path executed the gate.
        model.GenerateWithMultipleImages(new[] { firstImage }, "a", maxLength: 1);
        var firstContext = Assert.IsType<Tensor<float>>(contextField.GetValue(languageGate)).Clone();
        model.GenerateWithMultipleImages(new[] { secondImage }, "a", maxLength: 1);
        var secondContext = Assert.IsType<Tensor<float>>(contextField.GetValue(languageGate));
        Assert.Equal(4 * 32, firstContext.Length);
        Assert.Equal(firstContext.Shape.ToArray(), secondContext.Shape.ToArray());
        float largestChange = 0;
        for (int i = 0; i < firstContext.Length; i++)
            largestChange = Math.Max(largestChange, Math.Abs(firstContext[i] - secondContext[i]));
        Assert.True(largestChange > 1e-6f, $"The real LM gate received unchanged visual context (largest difference {largestChange}).");
    }

    private static FlamingoNeuralNetwork<float> CreateSmallFlamingo(int imageSize, int patchSize) => new(
        ImageArchitecture(imageSize, 3, 3),
        new FlamingoOptions
        {
            EmbeddingDimension = 32, MaxSequenceLength = 8, ImageSize = imageSize, PatchSize = patchSize,
            Channels = 3, NumPerceiverTokens = 4, MaxImagesInContext = 1,
            VisionHiddenDim = 32, LmHiddenDim = 32, NumVisionLayers = 1, NumLmLayers = 4,
            NumHeads = 2, VocabSize = 512, NumPerceiverLayers = 1
        },
        tokenizer: ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }));

    private static NeuralNetworkArchitecture<float> VectorArchitecture(int inputSize, int outputSize) => new(
        inputType: InputType.OneDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputSize: inputSize, outputSize: outputSize) { RandomSeed = 1337 };

    private static NeuralNetworkArchitecture<float> ImageArchitecture(int imageSize, int channels, int classes) => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.ImageClassification,
        inputHeight: imageSize, inputWidth: imageSize, inputDepth: channels,
        outputSize: classes) { RandomSeed = 1337 };
}
