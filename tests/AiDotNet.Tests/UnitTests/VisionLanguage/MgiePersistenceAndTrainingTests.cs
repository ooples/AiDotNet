using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.VisionLanguage.Editing;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class MgiePersistenceAndTrainingTests
{
    public MgiePersistenceAndTrainingTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void ActualNoiseLoss_BackpropagatesThroughJointContextAndSourceImage()
    {
        using var model = CreateModel();
        var image = MgieConditioningContractTests.CreateImage();
        var noisy = JointVisionLanguageStateTests.Values(new[] { 1, 4, 4, 4 });
        var mapper = MgieJointMapperTests.Field<MultimodalEditMapperLayer<float>>(model, "_editMapper");
        var tokens = MgieJointMapperTests.Field<Tensor<float>>(mapper, "_editTokenEmbeddings");
        var queries = MgieJointMapperTests.Field<Tensor<float>>(mapper, "_queryEmbeddings");
        using var tape = new GradientTape<float>();
        var context = model.EncodeEditGuidance(image, new[] { 1, 2, 3 });
        var source = model.VAE.Encode(image, sampleMode: false);
        var engine = AiDotNetEngine.Current;
        var modelInput = engine.TensorConcatenate(new[] { noisy, source }, axis: 1);
        var prediction = model.NoisePredictor.PredictNoise(modelInput, 7, context);
        var weights = JointVisionLanguageStateTests.Values(prediction.Shape.ToArray());
        var loss = engine.ReduceSum(engine.TensorMultiply(prediction, weights), new[] { 0, 1, 2, 3 }, keepDims: false);
        var observed = new[] { image, noisy, tokens, queries };
        var gradients = tape.ComputeGradients(loss, observed);
        foreach (var parameter in observed)
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient));
            Assert.NotNull(gradient);
            Assert.Contains(gradient.ToArray(), value => Math.Abs(value) > 1e-8f);
            Assert.All(gradient.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        }
    }

    [Fact]
    public void SerializedModel_RestoresNestedLanguageStateAndStandaloneMapperTokens()
    {
        using var model = CreateModel();
        var image = MgieConditioningContractTests.CreateImage();
        var ids = new[] { 1, 2, 3 };
        _ = model.EncodeEditGuidance(image, ids);
        var originalParameters = RawParameters(model);
        for (int i = 0; i < originalParameters.Length; i++) originalParameters[i][i] += (i + 1) * 0.037f;
        var expected = model.EncodeEditGuidance(image, ids);
        byte[] saved = model.Serialize();
        Assert.NotEmpty(saved);
        using var restored = CreateModel();
        _ = restored.EncodeEditGuidance(image, ids);
        restored.Deserialize(saved);
        var actual = restored.EncodeEditGuidance(image, ids);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        var restoredParameters = RawParameters(restored);
        for (int i = 0; i < originalParameters.Length; i++)
        {
            Assert.NotSame(originalParameters[i], restoredParameters[i]);
            Assert.Equal(originalParameters[i].ToArray(), restoredParameters[i].ToArray());
        }
    }

    [Fact]
    public void Clone_PreservesRawOwnershipLayout_AndWritesDoNotChangeOriginal()
    {
        using var model = CreateModel();
        var image = MgieConditioningContractTests.CreateImage();
        var ids = new[] { 1, 2, 3 };
        var expected = model.EncodeEditGuidance(image, ids);
        using var cloned = Assert.IsType<MGIE<float>>(model.Clone());
        MgieJointMapperTests.Field<LLaVANeuralNetwork<float>>(cloned, "_instructionEncoder").SetTrainingMode(false);
        MgieJointMapperTests.Field<MultimodalEditMapperLayer<float>>(cloned, "_editMapper").SetTrainingMode(false);
        var beforeWrite = cloned.EncodeEditGuidance(image, ids);
        Assert.Equal(expected.ToArray(), beforeWrite.ToArray());
        var originalParameters = RawParameters(model);
        var clonedParameters = RawParameters(cloned);
        Assert.Equal(originalParameters.Length, clonedParameters.Length);
        var beforeOriginal = originalParameters.Select(parameter => parameter.ToArray()).ToArray();
        for (int i = 0; i < clonedParameters.Length; i++)
        {
            Assert.NotSame(originalParameters[i], clonedParameters[i]);
            Assert.Equal(beforeOriginal[i], clonedParameters[i].ToArray());
            clonedParameters[i][i] += (i + 1) * 0.11f;
        }
        var afterWrite = cloned.EncodeEditGuidance(image, ids);
        MgieSamplingPhysicsTests.AssertDifferent(beforeWrite, afterWrite);
        Assert.Equal(expected.ToArray(), model.EncodeEditGuidance(image, ids).ToArray());
        for (int i = 0; i < originalParameters.Length; i++) Assert.Equal(beforeOriginal[i], originalParameters[i].ToArray());
        // The clone owns its raw state the same way the original does. The edit mapper's embeddings
        // train, so the diffusion collector returns them; the MLLM's vision tensors are frozen under
        // the paper's scope (Fu et al. 2024, Sec. 3.3), so they are registered rather than collected.
        var collected = JointVisionLanguageStateTests.Collect(cloned);
        var encoder = MgieJointMapperTests.Field<LLaVANeuralNetwork<float>>(cloned, "_instructionEncoder");
        var registered = encoder.GetParameterStateChunks().ToList();
        foreach (var parameter in clonedParameters)
        {
            bool trainable = collected.Any(item => ReferenceEquals(item, parameter));
            bool frozen = registered.Any(item =>
                ReferenceEquals(item.SourceTensor, parameter)
                && item.Role == AiDotNet.Models.Parameters.ParameterSlotRole.Frozen);
            Assert.True(trainable ^ frozen,
                "Every raw parameter must be either collected for training or registered as frozen state.");
        }
    }

    internal static Tensor<float>[] RawParameters(MGIE<float> model)
    {
        var encoder = MgieJointMapperTests.Field<LLaVANeuralNetwork<float>>(model, "_instructionEncoder");
        var mapper = MgieJointMapperTests.Field<MultimodalEditMapperLayer<float>>(model, "_editMapper");
        return new[]
        {
            MgieJointMapperTests.Field<Tensor<float>>(encoder, "_visionClsToken"),
            MgieJointMapperTests.Field<Tensor<float>>(encoder, "_visionPositionalEmbeddings"),
            MgieJointMapperTests.Field<Tensor<float>>(encoder, "_textPositionalEmbeddings"),
            MgieJointMapperTests.Field<Tensor<float>>(mapper, "_editTokenEmbeddings"),
            MgieJointMapperTests.Field<Tensor<float>>(mapper, "_queryEmbeddings")
        };
    }

    private static MGIE<float> CreateModel()
    {
        var model = new MGIE<float>(options: MgieConditioningContractTests.CreateOptions(),
            unet: new UNetNoisePredictor<float>(inputChannels: 8, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1 }, numResBlocks: 1, attentionResolutions: new[] { 1 },
                contextDim: 768, numHeads: 2, inputHeight: 4, seed: 397525),
            vae: MgieConditioningContractTests.CreateVae(), seed: 397525);
        MgieJointMapperTests.Field<LLaVANeuralNetwork<float>>(model, "_instructionEncoder").SetTrainingMode(false);
        MgieJointMapperTests.Field<MultimodalEditMapperLayer<float>>(model, "_editMapper").SetTrainingMode(false);
        return model;
    }
}
