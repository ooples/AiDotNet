using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class AudioVisualCorrespondenceExecutionTests
{
    public enum PairTask
    {
        Synchronization,
        SceneClassification,
        Separation
    }

    public AudioVisualCorrespondenceExecutionTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void DefaultPredictRetainsTheBatchedFeatureContract(int batch)
    {
        using var model = new AudioVisualCorrespondenceNetwork<float>();
        var features = new Tensor<float>(new[] { batch, 512 });
        for (int index = 0; index < features.Length; index++) features[index] = (index % 19 + 1) / 19.0f;
        var prediction = model.Predict(features);
        Assert.Equal(new[] { batch, 2 }, prediction.Shape.ToArray());
        AssertFinite(prediction);
    }

    [Fact]
    public void CustomLayersRetainPredictionWithoutInventingModalityRoles()
    {
        var architecture = CreateArchitecture();
        architecture.Layers.Add(new DenseLayer<float>(2, (AiDotNet.Interfaces.IActivationFunction<float>?)null));
        using var model = new AudioVisualCorrespondenceNetwork<float>(architecture, CreateOptions(16));
        var prediction = model.Predict(Features());
        Assert.Equal(new[] { 2, 2 }, prediction.Shape.ToArray());
        Assert.Single(model.Layers);
        Assert.Equal(model.Layers.Sum(layer => layer.ParameterCount), model.ParameterCount);
        var parameters = model.GetParameters().ToArray();
        Assert.NotEmpty(parameters);
        Assert.Equal(model.ParameterCount, parameters.Length);
        var error = Assert.Throws<NotSupportedException>(() => model.GetAudioEmbedding(Audio(), 16000));
        Assert.Contains("custom Architecture.Layers", error.Message);
        Assert.Equal(parameters, model.GetParameters().ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void InvalidEncoderDepthFailsAtTheOptionsBoundary(int depth)
    {
        var error = Assert.Throws<ArgumentException>(() =>
            new AudioVisualCorrespondenceOptions { NumEncoderLayers = depth }.Validate());
        Assert.Equal("options", error.ParamName);
        Assert.Contains(nameof(AudioVisualCorrespondenceOptions.NumEncoderLayers), error.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void UnusedChannelGeometryDoesNotBlockFeaturesButFailsAtTheVisualBoundary(int channels)
    {
        var options = CreateOptions(16);
        options.Channels = channels;
        options.Validate();
        using var model = new AudioVisualCorrespondenceNetwork<float>(CreateArchitecture(), options);
        Assert.Equal(new[] { 2, 2 }, model.Predict(Features()).Shape.ToArray());
        var error = Assert.Throws<ArgumentException>(() => model.GetVisualEmbedding(new[] { Frame() }));
        Assert.Equal("options", error.ParamName);
        Assert.Contains(nameof(AudioVisualCorrespondenceOptions.Channels), error.Message);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    public void ConfiguredChannelsDetermineTheActualVisualPatchWidth(int channels)
    {
        var options = CreateOptions(16);
        options.Channels = channels;
        using var model = new AudioVisualCorrespondenceNetwork<float>(CreateArchitecture(), options);
        var frame = new Tensor<float>(new[] { channels, 16, 32 });
        for (int index = 0; index < frame.Length; index++) frame[index] = (index % 31) / 31.0f;
        AssertUnitEmbedding(model.GetVisualEmbedding(new[] { frame }), 16);
        var wrongChannels = new Tensor<float>(new[] { channels + 1, 16, 32 });
        var error = Assert.Throws<ArgumentException>(() => model.GetVisualEmbedding(new[] { wrongChannels }));
        Assert.Equal("frame", error.ParamName);
    }

    [Theory]
    [InlineData(16, 1)]
    [InlineData(24, 3)]
    [InlineData(512, 6)]
    public void SharedFactoryHonorsTheRequestedEncoderDepthAndWidth(int width, int depth)
    {
        var layers = LayerHelper<float>.CreateAudioVisualCorrespondenceLayers(width, depth).ToList();
        try
        {
            Assert.Equal(depth * 3 + 4, layers.Count);
            for (int block = 0; block < depth; block++)
            {
                Assert.IsType<DenseLayer<float>>(layers[block * 3]);
                Assert.IsType<LayerNormalizationLayer<float>>(layers[block * 3 + 1]);
                Assert.IsType<ActivationLayer<float>>(layers[block * 3 + 2]);
            }
            var lastEncoderProjection = Assert.IsType<DenseLayer<float>>(layers[(depth - 1) * 3]);
            Assert.Equal(width, lastEncoderProjection.GetOutputShape().Last());
            Assert.Equal(2, layers.Last().GetOutputShape().Last());
        }
        finally
        {
            foreach (var layer in layers)
                if (layer is IDisposable disposable) disposable.Dispose();
        }
    }

    [Fact]
    public void ReenumeratingTheSharedFactoryCreatesIndependentLayers()
    {
        var factory = LayerHelper<float>.CreateAudioVisualCorrespondenceLayers(16, 2);
        var first = factory.ToArray();
        var second = factory.ToArray();
        try
        {
            Assert.Equal(first.Length, second.Length);
            for (int index = 0; index < first.Length; index++) Assert.NotSame(first[index], second[index]);
        }
        finally
        {
            foreach (var layer in first.Concat(second))
                if (layer is IDisposable disposable) disposable.Dispose();
        }
    }

    [Theory]
    [InlineData(16, true)]
    [InlineData(24, false)]
    public void ModalitiesAndPredictKeepIndependentInputWidthsAcrossRepeatedCalls(int width, bool audioFirst)
    {
        using var model = CreateModel(width);
        var audio = Audio();
        var frame = Frame();
        Vector<float> audioEmbedding;
        Vector<float> visualEmbedding;
        if (audioFirst)
        {
            audioEmbedding = model.GetAudioEmbedding(audio, 16000);
            visualEmbedding = model.GetVisualEmbedding(new[] { frame });
        }
        else
        {
            visualEmbedding = model.GetVisualEmbedding(new[] { frame });
            audioEmbedding = model.GetAudioEmbedding(audio, 16000);
        }
        AssertUnitEmbedding(audioEmbedding, width);
        AssertUnitEmbedding(visualEmbedding, width);
        var prediction = model.Predict(Features());
        Assert.Equal(new[] { 2, 2 }, prediction.Shape.ToArray());
        AssertFinite(prediction);
        AssertClose(audioEmbedding, model.GetAudioEmbedding(audio, 16000));
        AssertClose(visualEmbedding, model.GetVisualEmbedding(new[] { frame }));
        var differentAudio = new Tensor<float>(audio.Shape.ToArray());
        for (int index = 0; index < differentAudio.Length; index++)
            differentAudio[index] = (float)(Math.Sin(index * 0.071) + 0.2 * Math.Cos(index * 0.023));
        var differentFrame = new Tensor<float>(frame.Shape.ToArray());
        for (int index = 0; index < differentFrame.Length; index++) differentFrame[index] = 1.0f - frame[index];
        AssertDifferent(audioEmbedding, model.GetAudioEmbedding(differentAudio, 16000));
        AssertDifferent(visualEmbedding, model.GetVisualEmbedding(new[] { differentFrame }));
        Assert.InRange(model.ComputeCorrespondence(audio, new[] { frame }), -1.00001f, 1.00001f);
    }

    [Theory]
    [InlineData(16, PairTask.Synchronization)]
    [InlineData(24, PairTask.Synchronization)]
    [InlineData(16, PairTask.SceneClassification)]
    [InlineData(24, PairTask.SceneClassification)]
    [InlineData(16, PairTask.Separation)]
    [InlineData(24, PairTask.Separation)]
    public void PairTaskHeadsConsumeTheWholeFusionPath(int width, PairTask task)
    {
        using var model = CreateModel(width);
        var audio = Audio();
        var frames = new[] { Frame() };
        _ = model.Predict(Features());
        switch (task)
        {
            case PairTask.Synchronization:
                var (offset, confidence) = model.CheckSynchronization(audio, frames);
                Assert.False(double.IsNaN(offset) || double.IsInfinity(offset));
                Assert.InRange(confidence, 0.0f, 1.00001f);
                break;
            case PairTask.SceneClassification:
                var classes = model.ClassifyScene(audio, frames, new[] { "music", "speech" });
                Assert.Equal(2, classes.Count);
                Assert.InRange(classes.Values.Sum(), 0.99999f, 1.00001f);
                Assert.All(classes.Values, probability => Assert.InRange(probability, 0.0f, 1.0f));
                break;
            case PairTask.Separation:
                var separated = model.SeparateAudioByVisual(audio, frames[0]);
                Assert.NotEmpty(separated.ToArray());
                AssertFinite(separated);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(task));
        }
    }

    [Fact]
    public void SoundLocalizationPreservesTheRectangularPatchGrid()
    {
        using var model = CreateModel(16);
        var maps = model.LocalizeSoundSource(Audio(), new[] { Frame() }).ToArray();
        var map = Assert.Single(maps);
        Assert.Equal(new[] { 1, 2 }, map.Shape.ToArray());
        AssertFinite(map);
        Assert.InRange(map.ToArray().Sum(), 0.99999f, 1.00001f);
    }

    [Fact]
    public void CloneRetainsModalityParametersWithoutSharingUpdates()
    {
        using var model = CreateModel(16);
        var audio = Audio();
        var frames = new[] { Frame() };
        var expectedAudio = model.GetAudioEmbedding(audio, 16000);
        var expectedVisual = model.GetVisualEmbedding(frames);
        _ = model.CheckSynchronization(audio, frames);
        _ = model.Predict(Features());
        SetDistinctParameterValues(model);
        expectedAudio = model.GetAudioEmbedding(audio, 16000);
        expectedVisual = model.GetVisualEmbedding(frames);
        var parameters = model.GetParameters().ToArray();
        using var clone = Assert.IsType<AudioVisualCorrespondenceNetwork<float>>(model.Clone());
        AssertClose(expectedAudio, clone.GetAudioEmbedding(audio, 16000));
        AssertClose(expectedVisual, clone.GetVisualEmbedding(frames));
        Assert.Equal(parameters, clone.GetParameters().ToArray());
        var altered = clone.GetParameters();
        for (int index = 0; index < altered.Length; index++) altered[index] *= 0.75f;
        clone.UpdateParameters(altered);
        Assert.Equal(parameters, model.GetParameters().ToArray());
        AssertClose(expectedAudio, model.GetAudioEmbedding(audio, 16000));
        AssertClose(expectedVisual, model.GetVisualEmbedding(frames));
    }

    [Fact]
    public void SerializationRestoresBothModalityAdaptersAndTheSharedEncoder()
    {
        using var model = new CorrespondenceProbe(16);
        var audio = Audio();
        var frames = new[] { Frame() };
        var expectedAudio = model.GetAudioEmbedding(audio, 16000);
        var expectedVisual = model.GetVisualEmbedding(frames);
        _ = model.CheckSynchronization(audio, frames);
        SetDistinctParameterValues(model);
        expectedAudio = model.GetAudioEmbedding(audio, 16000);
        expectedVisual = model.GetVisualEmbedding(frames);
        var state = model.Serialize();
        using var restored = new CorrespondenceProbe(16);
        restored.Deserialize(state);
        Assert.Equal(model.Layers.Count, restored.Layers.Count);
        for (int index = 0; index < model.Layers.Count; index++)
            Assert.Equal(model.Layers[index].GetParameters().ToArray(), restored.Layers[index].GetParameters().ToArray());
        Assert.Equal(model.Adapters.Length, restored.Adapters.Length);
        for (int index = 0; index < model.Adapters.Length; index++)
            Assert.Equal(model.Adapters[index].GetParameters().ToArray(), restored.Adapters[index].GetParameters().ToArray());
        Assert.Equal(model.OwnedTensors.Length, restored.OwnedTensors.Length);
        for (int index = 0; index < model.OwnedTensors.Length; index++)
            Assert.True(model.OwnedTensors[index].ToArray().SequenceEqual(restored.OwnedTensors[index].ToArray()),
                $"Model-owned trainable tensor {index} must be serialized in addition to canonical and auxiliary layers.");
        AssertClose(expectedAudio, restored.GetAudioEmbedding(audio, 16000));
        AssertClose(expectedVisual, restored.GetVisualEmbedding(frames));
        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
    }

    [Fact]
    public void CorrespondenceTrainingComputesFreshNonzeroGradients()
    {
        using var model = new CorrespondenceProbe(16);
        var audio = Audio();
        var frames = new[] { Frame() };
        _ = model.GetAudioEmbedding(audio, 16000);
        _ = model.GetVisualEmbedding(frames);
        var audioAdapter = Assert.Single(model.Adapters, layer => layer.GetInputShape().Last() == 128);
        var visualAdapter = Assert.Single(model.Adapters, layer => layer.GetInputShape().Last() == 3 * 16 * 16);
        var sharedProjection = Assert.IsType<DenseLayer<float>>(model.Layers[0]);
        var audioBefore = audioAdapter.GetParameters().ToArray();
        var visualBefore = visualAdapter.GetParameters().ToArray();
        var sharedBefore = sharedProjection.GetParameters().ToArray();
        var before = model.GetParameters().ToArray();
        model.LearnCorrespondence(new[] { audio }, new[] { frames.AsEnumerable() }, epochs: 1);
        var after = model.GetParameters().ToArray();
        Assert.Equal(before.Length, after.Length);
        Assert.True(before.Zip(after, (left, right) => Math.Abs(left - right)).Any(delta => delta > 1e-9f),
            "LearnCorrespondence must backpropagate a fresh loss before updating parameters.");
        Assert.All(after, value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        AssertLayerChanged(audioBefore, audioAdapter);
        AssertLayerChanged(visualBefore, visualAdapter);
        AssertLayerChanged(sharedBefore, sharedProjection);
    }

    [Fact]
    public void CorrespondenceObjectivePreservesEqualPerFramePoolingForUnequalImageSizes()
    {
        using var model = new CorrespondenceProbe(16);
        var audio = Audio();
        var smaller = new Tensor<float>(new[] { 3, 16, 16 });
        for (int index = 0; index < smaller.Length; index++) smaller[index] = (index % 13) / 13.0f;
        var frames = new[] { Frame(), smaller };
        var audioEmbedding = model.GetAudioEmbedding(audio, 16000);
        var visualEmbedding = model.GetVisualEmbedding(frames);
        float dot = audioEmbedding.ToArray().Zip(visualEmbedding.ToArray(), (left, right) => left * right).Sum();
        float expectedLoss = (1.0f - dot) * (1.0f - dot);
        Assert.InRange(expectedLoss, 1e-8f, 4.0f);
        model.LearnCorrespondence(new[] { audio }, new[] { frames.AsEnumerable() }, epochs: 1);
        Assert.InRange(Math.Abs(expectedLoss - model.ObservedLoss), 0.0f, 1e-6f);
    }

    [Fact]
    public void FirstCorrespondenceUpdateDiscoversAndPublishesLazyAdapterGradients()
    {
        using var model = new CorrespondenceProbe(16);
        model.LearnCorrespondence(new[] { Audio() }, new[] { new[] { Frame() }.AsEnumerable() }, epochs: 1);
        var audioAdapter = Assert.Single(model.Adapters, layer => layer.GetInputShape().Last() == 128);
        var visualAdapter = Assert.Single(model.Adapters, layer => layer.GetInputShape().Last() == 3 * 16 * 16);
        foreach (var adapter in new[] { audioAdapter, visualAdapter })
        {
            var gradients = adapter.ScatteredParameterGradients;
            Assert.NotNull(gradients);
            Assert.Equal(adapter.ParameterCount, gradients.Length);
            Assert.Contains(gradients.ToArray(), value => Math.Abs(value) > 1e-9f);
            Assert.All(gradients.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        }
    }

    private sealed class CorrespondenceProbe : AudioVisualCorrespondenceNetwork<float>
    {
        public float ObservedLoss => LastLoss;
        public Tensor<float>[] OwnedTensors => GetExtraTrainableTensors().OfType<Tensor<float>>().ToArray();
        public CorrespondenceProbe(int width) : base(CreateArchitecture(), CreateOptions(width)) { }

        public DenseLayer<float>[] Adapters => GetExtraTrainableLayers().OfType<DenseLayer<float>>()
            .Where(layer => !Layers.Any(primary => ReferenceEquals(primary, layer))).Distinct().ToArray();
    }

    private static AudioVisualCorrespondenceNetwork<float> CreateModel(int width)
        => new(CreateArchitecture(), CreateOptions(width));

    private static NeuralNetworkArchitecture<float> CreateArchitecture()
        => new(InputType.OneDimensional, NeuralNetworkTaskType.BinaryClassification, inputSize: 32, outputSize: 2)
            { RandomSeed = 1337 };

    private static AudioVisualCorrespondenceOptions CreateOptions(int width)
        => new() { EmbeddingDimension = width, NumEncoderLayers = 2 };

    private static void SetDistinctParameterValues(AudioVisualCorrespondenceNetwork<float> model)
    {
        var parameters = model.GetParameters();
        for (int index = 0; index < parameters.Length; index++)
            parameters[index] += (index % 13 - 6) * 0.002f;
        model.UpdateParameters(parameters);
    }

    private static void AssertLayerChanged(float[] before, DenseLayer<float> layer)
    {
        var after = layer.GetParameters().ToArray();
        Assert.Equal(before.Length, after.Length);
        Assert.True(before.Zip(after, (left, right) => Math.Abs(left - right)).Any(delta => delta > 1e-9f),
            "The actual audio/visual adapter and shared encoder must each receive the new gradient.");
    }

    private static Tensor<float> Features()
    {
        var features = new Tensor<float>(new[] { 2, 32 });
        for (int index = 0; index < features.Length; index++) features[index] = (index % 11 + 1) / 11.0f;
        return features;
    }

    private static Tensor<float> Audio()
    {
        var audio = new Tensor<float>(new[] { 4096 });
        for (int index = 0; index < audio.Length; index++)
            audio[index] = (float)(0.5 * Math.Sin(index * 0.173) + 0.2 * Math.Cos(index * 0.047));
        return audio;
    }

    private static Tensor<float> Frame()
    {
        var frame = new Tensor<float>(new[] { 3, 16, 32 });
        for (int index = 0; index < frame.Length; index++) frame[index] = (index % 29 + 1) / 29.0f;
        return frame;
    }

    private static void AssertUnitEmbedding(Vector<float> embedding, int width)
    {
        Assert.Equal(width, embedding.Length);
        Assert.All(embedding.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        Assert.InRange(embedding.ToArray().Sum(value => value * value), 0.9999f, 1.0001f);
    }

    private static void AssertClose(Vector<float> expected, Vector<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int index = 0; index < expected.Length; index++)
            Assert.InRange(Math.Abs(expected[index] - actual[index]), 0.0f, 1e-6f);
    }

    private static void AssertDifferent(Vector<float> expected, Vector<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.True(expected.ToArray().Zip(actual.ToArray(), (left, right) => Math.Abs(left - right))
            .Any(delta => delta > 1e-6f), "An embedding must depend on the supplied modality input.");
    }

    private static void AssertFinite(Tensor<float> tensor)
        => Assert.All(tensor.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
}
