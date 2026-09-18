using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.VisionLanguage.Editing;
using Xunit;
using static AiDotNet.Tests.UnitTests.VisionLanguage.MgieConditioningContractTests;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>MGIE's training objective (Fu et al. 2024, Eq. 5 and Sec. 3.3-4) on tiny real components.</summary>
public sealed class MgieEditTrainingTests
{
    private static readonly int[] Instruction = { 5, 9, 12 };
    private static readonly int[] Expressive = { 17, 3, 40 };

    public MgieEditTrainingTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void PublishedTrainingRecipe_IsTheDefault()
    {
        var options = new MGIEOptions();
        Assert.Equal(0.05, options.ConditionDropoutProbability); // 5% of data drops v, U or both.
        Assert.Equal(1.0, options.InstructionLossWeight);        // L_all = L_ins + 0.5 L_edit.
        Assert.Equal(0.5, options.EditLossWeight);
        Assert.Equal(LanguageModelTrainableScope.WordEmbeddingsAndHead, options.TrainableLanguageModelScope);
        Assert.Equal((8, 77, 4, 7.5, 1.5),
            (options.EditTokenCount, options.EditQueryCount, options.EditHeadLayers, options.GuidanceScale, options.ImageGuidanceScale));
    }

    [Fact]
    public void InstructionLoss_IsTheCrossEntropyOfEachExpressiveTokenFromThePrecedingPosition()
    {
        var options = CreateOptions();
        options.EditLossWeight = 0;
        options.ConditionDropoutProbability = 0;
        using var unet = new RecordingUnet();
        using var vae = CreateVae();
        using var model = new MGIE<float>(options: options, unet: unet, vae: vae, seed: 397525);
        var image = CreateImage();
        var encoder = Field<LLaVANeuralNetwork<float>>(model, "_instructionEncoder");
        var mapper = Field<MultimodalEditMapperLayer<float>>(model, "_editMapper");
        var visual = (Tensor<float>)Invoke(model, "PrepareVisionImage", image);

        var tokens = Instruction.Concat(Expressive).ToList();
        var joint = encoder.EncodeJointHiddenStates(visual, tokens, mapper.EditTokenEmbeddings);
        int width = joint.Shape[2];
        double expected = 0;
        for (int j = 0; j < Expressive.Length; j++)
        {
            int position = encoder.JointVisualTokenCount + Instruction.Length + j - 1;
            var row = new Tensor<float>(new[] { 1, width });
            for (int k = 0; k < width; k++) row[k] = joint[0, position, k];
            var logits = encoder.ProjectToVocabulary(row).ToArray().Select(value => (double)value).ToArray();
            double max = logits.Max();
            double logSumExp = max + Math.Log(logits.Sum(value => Math.Exp(value - max)));
            expected += logSumExp - logits[Expressive[j]];
        }
        expected /= Expressive.Length;

        double actual = model.TrainEdit(image, Instruction, image, Expressive);
        Assert.True(Math.Abs(expected - actual) <= 2e-4 * Math.Max(1, Math.Abs(expected)),
            $"Expected instruction loss {expected:R}; TrainEdit reported {actual:R}.");
    }

    [Fact]
    public void EditLossWeight_ScalesTheEditObjectiveExactly()
    {
        (double Loss, float[][] Weights) Run(double weight)
        {
            var options = CreateOptions();
            options.EditLossWeight = weight;
            options.ConditionDropoutProbability = 0;
            using var unet = new RecordingUnet();
            using var vae = CreateVae();
            using var model = new MGIE<float>(options: options, unet: unet, vae: vae, seed: 397525);
            var target = CreateImage();
            for (int i = 0; i < target.Length; i++) target[i] = -target[i];
            // Resolve every lazily sized weight FIRST: a denoiser reports no parameters until a
            // forward has built its layers, so a snapshot taken before this compares nothing.
            _ = model.EditImage(CreateImage(), "resolve the lazy shapes");
            // Snapshot the weights the objective is measured on, before the step updates them. Two
            // models built from one seed must start identical, or a difference between the two
            // losses says nothing about the weight that is supposed to scale them.
            var weights = MgiePersistenceAndTrainingTests.RawParameters(model)
                .Select(parameter => parameter.ToArray())
                .Append(model.GetParameters().ToArray())
                .Append(unet.GetParameters().ToArray())
                .ToArray();
            // Put the training draw sequence back to a known point. The objective averages ONE
            // sampled example, so the timestep, the dropout decisions and the noise must be the same
            // in both runs; otherwise the two losses differ for reasons that have nothing to do with
            // the weight under test. Resolving the lazy shapes above consumes draws of its own.
            ReseedTrainingDraws(model, 397525);
            return (model.TrainEdit(CreateImage(), Instruction, target), weights);
        }

        var half = Run(0.5);
        var full = Run(1.0);
        for (int i = 0; i < half.Weights.Length; i++)
        {
            // A zero-length snapshot would make the comparison below prove nothing: a lazily sized
            // model reports no parameters until a forward has resolved its shapes.
            Assert.NotEmpty(half.Weights[i]);
            Assert.Equal(half.Weights[i], full.Weights[i]);
        }

        Assert.True(half.Loss > 0 && !double.IsNaN(half.Loss) && !double.IsInfinity(half.Loss));
        Assert.True(Math.Abs(full.Loss - 2 * half.Loss) <= 1e-4 * Math.Abs(full.Loss),
            $"Doubling EditLossWeight must double the objective: weight 0.5 gave {half.Loss:R}, "
            + $"weight 1.0 gave {full.Loss:R}, ratio {full.Loss / half.Loss:R}.");
    }

    /// <summary>Restarts the model's training draw sequence from a known seed.</summary>
    private static void ReseedTrainingDraws(MGIE<float> model, int seed)
    {
        for (var type = model.GetType(); type is not null; type = type.BaseType)
        {
            var field = type.GetField("RandomGenerator", BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is null) continue;
            field.SetValue(model, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
            return;
        }

        throw new InvalidOperationException("The diffusion base no longer exposes RandomGenerator.");
    }

    [Fact]
    public void DefaultScope_TrainsOnlyTheWordEmbeddingsAndLmHeadOfTheMllm()
    {
        var options = CreateOptions();
        options.ConditionDropoutProbability = 0;
        using var unet = new RecordingUnet();
        using var vae = CreateVae();
        using var model = new MGIE<float>(options: options, unet: unet, vae: vae, seed: 397525);
        var encoder = Field<LLaVANeuralNetwork<float>>(model, "_instructionEncoder");
        var image = CreateImage();
        model.TrainEdit(image, Instruction, image, Expressive); // Resolves every lazily sized weight.

        var before = encoder.GetParameterStateChunks().Select(chunk => (chunk.StableId, chunk.Role, Values: chunk.Tensor.ToArray())).ToList();
        int embedding = 1 + options.NumVisionLayers + 2;
        int head = embedding + 1 + options.NumDecoderLayers;
        foreach (var chunk in before)
        {
            bool tokenInterface = chunk.StableId.StartsWith($"layers/{embedding:D8}") || chunk.StableId.StartsWith($"layers/{head:D8}");
            Assert.True(tokenInterface ? chunk.Role == ParameterSlotRole.Trainable : chunk.Role != ParameterSlotRole.Trainable,
                $"{chunk.StableId} reported {chunk.Role}.");
        }
        Assert.Contains(before, chunk => chunk.Role == ParameterSlotRole.Frozen);

        model.TrainEdit(image, Instruction, image, Expressive);
        var after = encoder.GetParameterStateChunks().ToDictionary(chunk => chunk.StableId, chunk => chunk.Tensor.ToArray());
        foreach (var chunk in before.Where(chunk => chunk.Role == ParameterSlotRole.Frozen))
            Assert.Equal(chunk.Values, after[chunk.StableId]);
        Assert.Contains(before.Where(chunk => chunk.Role == ParameterSlotRole.Trainable),
            chunk => !chunk.Values.SequenceEqual(after[chunk.StableId]));
    }

    [Fact]
    public void TrainEdit_RejectsMismatchedImagesAndEmptyInstructions()
    {
        using var unet = new RecordingUnet();
        using var vae = CreateVae();
        using var model = CreateModel(unet, vae);
        var image = CreateImage();
        Assert.Throws<ArgumentException>(() => model.TrainEdit(image, Array.Empty<int>(), image));
        Assert.Throws<ArgumentException>(() => model.TrainEdit(image, Instruction, new Tensor<float>(new[] { 1, 3, 8, 8 })));
    }

    [Theory]
    [InlineData(nameof(MGIEOptions.ConditionDropoutProbability), 1.5)]
    [InlineData(nameof(MGIEOptions.EditLossWeight), double.NaN)]
    [InlineData(nameof(MGIEOptions.InstructionLossWeight), -1.0)]
    public void InvalidTrainingOptions_AreRejectedAtConstruction(string property, double value)
    {
        var options = CreateOptions();
        var info = typeof(MGIEOptions).GetProperty(property);
        Assert.NotNull(info);
        info.SetValue(options, value);
        using var unet = new RecordingUnet();
        using var vae = CreateVae();
        Assert.Throws<ArgumentOutOfRangeException>(() => new MGIE<float>(options: options, unet: unet, vae: vae, seed: 397525));
    }

    private static TField Field<TField>(object owner, string name)
    {
        var field = owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return Assert.IsAssignableFrom<TField>(field.GetValue(owner));
    }

    private static object Invoke(object owner, string name, params object[] arguments)
    {
        var method = owner.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        var result = method.Invoke(owner, arguments);
        Assert.NotNull(result);
        return result;
    }
}
