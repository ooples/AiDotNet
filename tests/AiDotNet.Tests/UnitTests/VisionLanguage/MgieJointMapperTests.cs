using System.Reflection;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Editing;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class MgieJointMapperTests
{
    public MgieJointMapperTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void Mapper_UsesRealMemoryAndQueries_WithNonzeroGradientsAndFixedOutputTokens()
    {
        using var mapper = new MultimodalEditMapperLayer<float>(8, 8, 16, 2, 3, 2, 1);
        using var engineAccess = new JointVisionLanguageStateTests.SmallLlava();
        var input = JointVisionLanguageStateTests.Values(new[] { 2, 2, 8 });
        var before = input.ToArray();
        using var tape = new GradientTape<float>();
        var output = mapper.Forward(input);
        Assert.Equal(new[] { 2, 3, 16 }, output.Shape.ToArray());
        var tokens = Field<Tensor<float>>(mapper, "_editTokenEmbeddings");
        var queries = Field<Tensor<float>>(mapper, "_queryEmbeddings");
        var gradients = tape.ComputeGradients(engineAccess.WeightedSum(output), new[] { input, tokens, queries });
        foreach (var parameter in new[] { input, tokens, queries })
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient));
            Assert.NotNull(gradient);
            Assert.Contains(gradient.ToArray(), value => Math.Abs(value) > 1e-8f);
            Assert.All(gradient.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        }
        Assert.Equal(before, input.ToArray());
        var parameters = JointVisionLanguageStateTests.Collect(mapper);
        Assert.Single(parameters.Where(parameter => ReferenceEquals(parameter, tokens)));
        Assert.Single(parameters.Where(parameter => ReferenceEquals(parameter, queries)));
    }

    [Fact]
    public void JointContext_DependsOnBothModalities_AndIncludesRegisteredRawState()
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        using var model = MgieConditioningContractTests.CreateModel(unet, vae);
        var encoder = Field<LLaVANeuralNetwork<float>>(model, "_instructionEncoder");
        encoder.SetTrainingMode(false);
        var image = MgieConditioningContractTests.CreateImage();
        var original = model.EncodeEditGuidance(image, new[] { 1, 2, 3 });
        Assert.Equal(original.ToArray(), model.EncodeEditGuidance(image, new[] { 1, 2, 3 }).ToArray());
        var changedInstruction = model.EncodeEditGuidance(image, new[] { 4, 5, 6 });
        var changedImage = image.Clone();
        changedImage[4] += 0.6f;
        var changedVisual = model.EncodeEditGuidance(changedImage, new[] { 1, 2, 3 });
        Assert.Equal(new[] { 1, 3, 768 }, original.Shape.ToArray());
        MgieSamplingPhysicsTests.AssertDifferent(original, changedInstruction);
        MgieSamplingPhysicsTests.AssertDifferent(original, changedVisual);
        var mapper = Field<MultimodalEditMapperLayer<float>>(model, "_editMapper");
        var parameters = JointVisionLanguageStateTests.Collect(model);
        foreach (var parameter in new[]
        {
            Field<Tensor<float>>(mapper, "_editTokenEmbeddings"),
            Field<Tensor<float>>(mapper, "_queryEmbeddings")
        }) Assert.Single(parameters.Where(item => ReferenceEquals(item, parameter)));

        // MGIE trains the MLLM's word embeddings and LM head only (Fu et al. 2024, Sec. 3.3), so the
        // vision tower's raw state is registered — it clones and serializes — but is deliberately not
        // part of the trainable set the diffusion collector returns.
        var registered = encoder.GetParameterStateChunks().ToList();
        foreach (var parameter in new[]
        {
            Field<Tensor<float>>(encoder, "_visionClsToken"),
            Field<Tensor<float>>(encoder, "_visionPositionalEmbeddings"),
            Field<Tensor<float>>(encoder, "_textPositionalEmbeddings")
        })
        {
            var chunk = Assert.Single(registered.Where(item => ReferenceEquals(item.SourceTensor, parameter)));
            Assert.Equal(AiDotNet.Models.Parameters.ParameterSlotRole.Frozen, chunk.Role);
            Assert.DoesNotContain(parameters, item => ReferenceEquals(item, parameter));
        }
    }

    [Fact]
    public void JointContext_GradientReachesImageAndBothMapperEmbeddingFamilies()
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        using var model = MgieConditioningContractTests.CreateModel(unet, vae);
        using var engineAccess = new JointVisionLanguageStateTests.SmallLlava();
        var image = MgieConditioningContractTests.CreateImage();
        var mapper = Field<MultimodalEditMapperLayer<float>>(model, "_editMapper");
        var tokens = Field<Tensor<float>>(mapper, "_editTokenEmbeddings");
        var queries = Field<Tensor<float>>(mapper, "_queryEmbeddings");
        using var tape = new GradientTape<float>();
        var output = model.EncodeEditGuidance(image, new[] { 1, 2, 3 });
        var gradients = tape.ComputeGradients(engineAccess.WeightedSum(output), new[] { image, tokens, queries });
        foreach (var parameter in new[] { image, tokens, queries })
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient));
            Assert.NotNull(gradient);
            Assert.Contains(gradient.ToArray(), value => Math.Abs(value) > 1e-7f);
            Assert.All(gradient.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RealEditing_PreservesImageRank_AndModes_OnSuccessAndFailure(bool training)
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        using var model = MgieConditioningContractTests.CreateModel(unet, vae);
        var encoder = Field<LLaVANeuralNetwork<float>>(model, "_instructionEncoder");
        var mapper = Field<MultimodalEditMapperLayer<float>>(model, "_editMapper");
        encoder.SetTrainingMode(training);
        mapper.SetTrainingMode(training);
        var image = new Tensor<float>(new[] { 3, 4, 4 }, new Vector<float>(MgieConditioningContractTests.CreateImage().ToArray()));
        var noise = MgieSamplingPhysicsTests.StandardNoise(64);
        var result = model.EditImage(image, "bright image", 397525, noise);
        Assert.Equal(new[] { 3, 4, 4 }, result.Shape.ToArray());
        Assert.Equal(training, encoder.IsTrainingMode);
        Assert.Equal(training, MapperTrainingMode(mapper));
        Assert.Throws<ArgumentException>(() => model.EditImage(image, "bright image", 397525, new Vector<float>(3)));
        Assert.Equal(training, encoder.IsTrainingMode);
        Assert.Equal(training, MapperTrainingMode(mapper));
    }

    [Fact]
    public void VisionResolutionIsIndependentOfDiffusionResolution()
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        var options = MgieConditioningContractTests.CreateOptions();
        options.ImageSize = 8;
        using var model = new MGIE<float>(options: options, unet: unet, vae: vae, seed: 397525);
        var image = MgieConditioningContractTests.CreateImage();
        var features = model.EncodeImage(image);
        Assert.Equal(new[] { 1, 17, 8 }, features.Shape.ToArray());
        var edited = model.EditImage(image, "bright image", 397525, MgieSamplingPhysicsTests.StandardNoise(64));
        Assert.Equal(new[] { 1, 3, 4, 4 }, edited.Shape.ToArray());
    }

    [Fact]
    public void ExpressiveInstruction_UsesActualGeneratedContinuationInJointStates()
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        var options = MgieConditioningContractTests.CreateOptions();
        options.EnableExpressiveInstructions = true;
        using var model = new MGIE<float>(options: options, unet: unet, vae: vae, seed: 397525);
        var encoder = Field<LLaVANeuralNetwork<float>>(model, "_instructionEncoder");
        encoder.SetTrainingMode(false);
        var tokenizerField = typeof(LLaVANeuralNetwork<float>).GetField("_tokenizer", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(tokenizerField);
        var tokenizer = Assert.IsAssignableFrom<ITokenizer>(tokenizerField.GetValue(encoder));
        int token = Enumerable.Range(0, options.VocabSize).First(id =>
        {
            string text = tokenizer.Decode(new List<int> { id });
            return text.Any(char.IsLetter) && text.IndexOf('<') < 0 && text.IndexOf('[') < 0;
        });
        var projection = Field<DenseLayer<float>>(encoder, "_outputProjection");
        projection.Forward(new Tensor<float>(new[] { 1, options.DecoderDim }));
        var weights = Field<Tensor<float>>(projection, "_weights");
        var biases = Field<Tensor<float>>(projection, "_biases");
        for (int i = 0; i < weights.Length; i++) weights[i] = 0;
        for (int i = 0; i < biases.Length; i++) biases[i] = i == token ? 20 : 0;

        const string instruction = "bright image";
        var image = MgieConditioningContractTests.CreateImage();
        var normalized = new Tensor<float>(new[] { 3, 4, 4 });
        for (int channel = 0; channel < 3; channel++)
        for (int offset = 0; offset < 16; offset++)
            normalized[channel * 16 + offset] = (float)((image[channel * 16 + offset] * 0.5 + 0.5 -
                options.ImageMean[channel]) / options.ImageStd[channel]);
        string continuation = encoder.Generate(normalized, instruction, options.MaxGenerationLength, temperature: 1, topP: 0);
        Assert.False(string.IsNullOrWhiteSpace(continuation));
        Assert.Equal(tokenizer.Decode(Enumerable.Repeat(token, options.MaxGenerationLength).ToList()), continuation);
        var originalTokens = encoder.EncodeInstructionTokens(instruction);
        var combinedTokens = originalTokens.Concat(encoder.EncodeInstructionTokens(continuation)).ToArray();
        Assert.True(combinedTokens.Length > originalTokens.Count);
        var expected = model.EncodeEditGuidance(image, combinedTokens);
        var withoutContinuation = model.EncodeEditGuidance(image, originalTokens);
        var method = typeof(MGIE<float>).GetMethod("EncodeStringGuidance", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        var actual = Assert.IsType<Tensor<float>>(method.Invoke(model, new object[] { image, instruction }));
        Assert.Equal(expected.ToArray(), actual.ToArray());
        MgieSamplingPhysicsTests.AssertDifferent(withoutContinuation, actual);
    }

    [Fact]
    public void OptionsAreOwned_AndMapperGeometryIsValidatedBeforeConstruction()
    {
        var options = MgieConditioningContractTests.CreateOptions();
        options.EditNumHeads = 3;
        var invalid = Assert.Throws<ArgumentOutOfRangeException>(() => new MGIE<float>(options: options));
        Assert.Equal(nameof(MGIEOptions.EditNumHeads), invalid.ParamName);
        options = MgieConditioningContractTests.CreateOptions();
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        using var model = new MGIE<float>(options: options, unet: unet, vae: vae, seed: 397525);
        options.ImageMean[0] = 999;
        options.EditQueryCount = 1;
        var first = Assert.IsType<MGIEOptions>(model.GetOptions());
        Assert.NotEqual(999, first.ImageMean[0]);
        Assert.Equal(3, first.EditQueryCount);
        first.ImageMean[0] = 555;
        Assert.NotEqual(555, Assert.IsType<MGIEOptions>(model.GetOptions()).ImageMean[0]);
    }

    internal static TValue Field<TValue>(object owner, string name)
    {
        var field = owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return Assert.IsType<TValue>(field.GetValue(owner));
    }

    private static bool MapperTrainingMode(MultimodalEditMapperLayer<float> mapper)
    {
        var field = typeof(LayerBase<float>).GetField("IsTrainingMode", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return Assert.IsType<bool>(field.GetValue(mapper));
    }
}
