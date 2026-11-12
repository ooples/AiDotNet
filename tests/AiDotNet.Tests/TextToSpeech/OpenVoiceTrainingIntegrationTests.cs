using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoiceTrainingIntegrationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublicTraining_RegistersLazyComponentsAndAlternatesExclusiveOptimizerGroups(bool versionTwo)
    {
        AiDotNetEngine.ResetToCpu();
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 2294 };
        var training = new OpenVoiceTrainingOptions
        {
            IpaVocabularySize = 17, IpaFilterChannels = 8, IpaHeads = 1, IpaLayers = 1, IpaDropout = 0,
            SegmentFrames = 4, MelChannels = 4, DiscriminatorWidthDivisor = 64
        };
        using TtsModelBase<double> model = versionTwo
            ? new OpenVoiceV2<double>(architecture, new OpenVoiceV2Options { HiddenDim = 4, DecoderDim = 4, SpeakerEmbeddingDim = 3, FftSize = 512, Training = training })
            : new OpenVoice<double>(architecture, new OpenVoiceOptions { HiddenDim = 4, InterChannels = 4, SpeakerEmbeddingDim = 3, FftSize = 512, Training = training });
        int previous = model.GetParameters().Length;
        var controller = versionTwo ? ((OpenVoiceV2<double>)model).TrainingForAudit : ((OpenVoice<double>)model).TrainingForAudit;
        Assert.True(training.IncludeTrainingComponents);
        Assert.True(model.GetParameters().Length > previous);
        IReadOnlyList<Tensor<double>> Parameters(IReadOnlyList<LayerBase<double>> layers)
            => TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        var generator = Parameters(controller.Graph.GeneratorLayers);
        var discriminator = Parameters(controller.Graph.DiscriminatorLayers);
        var generatorBefore = generator.Select(x => x.ToArray()).ToArray();
        var discriminatorBefore = discriminator.Select(x => x.ToArray()).ToArray();
        double[][]? discriminatorAfter = null;
        var phases = new List<string>();
        controller.AfterStepForAudit = group =>
        {
            phases.Add(group);
            if (group == "discriminator")
            {
                for (int i = 0; i < generator.Count; i++) Assert.Equal(generatorBefore[i], generator[i].ToArray());
                Assert.Contains(Enumerable.Range(0, discriminator.Count), i => !discriminatorBefore[i].SequenceEqual(discriminator[i].ToArray()));
                discriminatorAfter = discriminator.Select(x => x.ToArray()).ToArray();
            }
            else
            {
                for (int i = 0; i < discriminator.Count; i++) Assert.Equal(discriminatorAfter![i], discriminator[i].ToArray());
                Assert.Contains(Enumerable.Range(0, generator.Count), i => !generatorBefore[i].SequenceEqual(generator[i].ToArray()));
            }
        };
        var audio = new Tensor<double>(new[] { 2048 }, new Vector<double>(Enumerable.Range(0, 2048).Select(i => .2 * Math.Sin(i * .11))));
        var tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 4.0, 7.0 }));
        model.Train(tokens, audio);
        Assert.Equal(new[] { "discriminator", "generator" }, phases);
        Assert.Equal(1, training.TrainingSteps);
        model.SetTrainingMode(true);
        double evaluation = model.EvaluateTrainingObjective(new TtsTrainingSample<double> { Tokens = tokens, Audio = audio });
        Assert.True(!double.IsNaN(evaluation) && !double.IsInfinity(evaluation)); Assert.True(model.IsTrainingMode);
        Assert.Equal(1, training.TrainingSteps);
        using var copy = (TtsModelBase<double>)model.DeepCopy();
        Assert.Equal(model.GetParameters().ToArray(), copy.GetParameters().ToArray());
        double copiedEvaluation = copy.EvaluateTrainingObjective(new TtsTrainingSample<double> { Tokens = tokens, Audio = audio });
        Assert.True(Math.Abs(evaluation - copiedEvaluation) < 1e-10,
            "Cloning must preserve the auxiliary prior/discriminators and sampling step.");
        var restoredTraining = new OpenVoiceTrainingOptions(training) { IncludeTrainingComponents = false };
        using TtsModelBase<double> restored = versionTwo
            ? new OpenVoiceV2<double>(architecture, new OpenVoiceV2Options { HiddenDim = 4, DecoderDim = 4, SpeakerEmbeddingDim = 3, FftSize = 512, Training = restoredTraining })
            : new OpenVoice<double>(architecture, new OpenVoiceOptions { HiddenDim = 4, InterChannels = 4, SpeakerEmbeddingDim = 3, FftSize = 512, Training = restoredTraining });
        restored.Deserialize(model.Serialize());
        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
        double restoredEvaluation = restored.EvaluateTrainingObjective(new TtsTrainingSample<double> { Tokens = tokens, Audio = audio });
        Assert.True(Math.Abs(evaluation - restoredEvaluation) < 1e-10,
            "Restoring into an inference-only instance must reconstruct the trained auxiliary graph.");
    }
}
