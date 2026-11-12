using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class DiTToCodecTrainingTests
{
    private static string Reference(string name) => Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(item => item.GetDouble()).ToArray();
    private static double[] Parameters(IEnumerable<LayerBase<double>> layers) => layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
    private static DiTToCodecTrainingOptions Options() => new()
    {
        DiscriminatorWindows = [4, 8], ResolutionDivisors = [1, 2], DiscriminatorChannels = 2,
        SemanticModelDim = 8, SemanticFeedForwardDim = 16, SemanticKeyValueDim = 4, SemanticHeads = 2,
        SemanticDecoderLayers = 1, SemanticVocabulary = 384
    };

    [Fact]
    public void AdversarialMelObjective_MatchesPyTorchLossesAndGeneratorDiscriminatorGradients()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Reference("ditto_codec_loss_reference.json")));
        var reference = document.RootElement;
        var trainer = new DiTToCodecTrainer<double>(AiDotNetEngine.Current, Options());
        var weights = reference.GetProperty("weights");
        var parameters = new List<Tensor<double>>();
        int index = 0;
        foreach (var layer in trainer.Critics.Layers.Cast<Conv2DLayer<double>>())
        {
            var kernel = Values(weights[index++]); var bias = Values(weights[index++]);
            Assert.Equal(kernel.Length, layer.Kernel.Length); Assert.Equal(bias.Length, layer.Bias.Length);
            for (int i = 0; i < kernel.Length; i++) layer.Kernel[i] = kernel[i];
            for (int i = 0; i < bias.Length; i++) layer.Bias[i] = bias[i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(layer.Kernel);
            AiDotNetEngine.Current.InvalidatePersistentTensor(layer.Bias);
            parameters.Add(layer.Kernel); parameters.Add(layer.Bias);
        }
        Tensor<double> Matrix(string key) => new(new[] { 8, 4 }, new Vector<double>(Values(reference.GetProperty(key))));
        var real = Matrix("real"); var fake = Matrix("fake");
        var starts = reference.GetProperty("starts").EnumerateArray().Select(value => value.GetInt32()).ToArray();
        Tensor<double> Scalar(double value) { var tensor = new Tensor<double>(new[] { 1 }); tensor[0] = value; return tensor; }
        using (var tape = new GradientTape<double>())
        {
            var loss = trainer.Generator(real, fake, Scalar(.4), Scalar(.7), starts);
            Assert.Equal(reference.GetProperty("generator_loss").GetDouble(), loss[0], 9);
            var gradients = tape.ComputeGradients(loss, sources: new List<Tensor<double>> { fake });
            Close(gradients[fake].ToArray(), Values(reference.GetProperty("fake_gradient")));
        }
        using (var tape = new GradientTape<double>())
        {
            var loss = trainer.Discriminator(real, fake, starts);
            Assert.Equal(reference.GetProperty("discriminator_loss").GetDouble(), loss[0], 9);
            var gradients = tape.ComputeGradients(loss, sources: parameters);
            for (int i = 0; i < parameters.Count; i++)
                Close(gradients[parameters[i]].ToArray(), Values(reference.GetProperty("critic_gradients")[i]));
        }
    }

    [Fact]
    public void SemanticInjection_MatchesHuggingFaceByT5LossAndAllInputProjectionGradients()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Reference("ditto_semantic_reference.json")));
        var reference = document.RootElement;
        var trainer = new DiTToCodecTrainer<double>(AiDotNetEngine.Current, Options());
        trainer.InitializeSemanticDecoder(8, new Random(1));
        trainer.SemanticDecoder!.LoadHuggingFaceWeights((key, shape) =>
        {
            Assert.Equal(shape, reference.GetProperty("weight_shapes").GetProperty(key).EnumerateArray().Select(item => item.GetInt32()).ToArray());
            return Values(reference.GetProperty("weights").GetProperty(key));
        });
        AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(AiDotNetEngine.Current, trainer.Projection!, 8, 8,
            Values(reference.GetProperty("projection_weight")), Values(reference.GetProperty("projection_bias")));
        trainer.DecoderWeightsLoaded = true;
        var latents = new Tensor<double>(new[] { 4, 8 }, new Vector<double>(Values(reference.GetProperty("latents"))));
        var frozen = Parameters(trainer.DecoderLayers);
        using var tape = new GradientTape<double>();
        var loss = trainer.SemanticLoss(latents, reference.GetProperty("text").GetString()!);
        // HF forces RMS variance and attention probabilities to float32, including on a double model.
        Assert.True(Math.Abs(reference.GetProperty("loss").GetDouble() - loss[0]) < 2e-6);
        Assert.Equal(reference.GetProperty("fp64_loss").GetDouble(), loss[0], 9);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(trainer.ProjectionLayers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        var sources = parameters.Concat(new[] { latents }).ToList();
        var gradients = tape.ComputeGradients(loss, sources: sources);
        Close(gradients[latents].ToArray(), Values(reference.GetProperty("fp64_latent_gradient")));
        Close(gradients[latents].ToArray(), Values(reference.GetProperty("latent_gradient")), 2e-6);
        var torchWeight = Values(reference.GetProperty("fp64_projection_weight_gradient"));
        var nativeWeight = new double[64];
        for (int output = 0; output < 8; output++)
            for (int input = 0; input < 8; input++) nativeWeight[input * 8 + output] = torchWeight[output * 8 + input];
        Assert.Equal(2, parameters.Count);
        Close(gradients[parameters[0]].ToArray(), nativeWeight);
        Close(gradients[parameters[1]].ToArray(), Values(reference.GetProperty("fp64_projection_bias_gradient")));
        Close(gradients[parameters[1]].ToArray(), Values(reference.GetProperty("projection_bias_gradient")), 2e-6);
        Assert.Equal(frozen, Parameters(trainer.DecoderLayers));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CodecStages_UpdateEncoderQuantizerDecoderAndCritics_WhileOtherNetworksStayFixed(bool semantic)
    {
        var options = DiTToTTSPaperTests.TinyOptions(); options.CodecTraining = Options();
        options.TrainingStage = semantic ? DiTToTrainingStage.MelVaeSemanticFinetuning : DiTToTrainingStage.MelVaePretraining;
        using var model = new DiTToTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        if (semantic) model.LoadCodecSemanticDecoder(Reference("ditto_semantic_reference.safetensors"));
        var mel = new Tensor<double>(new[] { 8, 4 });
        for (int i = 0; i < mel.Length; i++) mel[i] = -2 + Math.Sin(i * .3);
        // CLaM codec pretraining needs neither text nor text tokens. Both codec stages accept prepared mels.
        var sample = new TtsTrainingSample<double> { Tokens = semantic ? model.TextToTokens("hi") : new Tensor<double>(new[] { 1 }),
            Text = semantic ? "hi" : null, Mel = mel };
        _ = model.GetParameters();
        Assert.True(TtsPaperNumerics.IsFinite(model.EvaluateTrainingObjective(sample)));
        var encoder = model.MelVae!.EncoderInput.GetParameters().ToArray();
        var decoder = model.MelVae.DecoderOutput.GetParameters().ToArray();
        var quantizer = model.MelVae.Quantizer.Codebooks[0].GetParameters().ToArray();
        var critics = Parameters(model.CodecTrainer!.Critics.Layers);
        var diffusion = model.DiffusionTransformer!.Output.GetParameters().ToArray();
        var textEncoder = model.TextEncoder!.Embedding.GetParameters().ToArray();
        var frozenDecoder = Parameters(model.CodecTrainer.DecoderLayers);
        var projection = Parameters(model.CodecTrainer.ProjectionLayers);
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Changed(encoder, model.MelVae.EncoderInput.GetParameters().ToArray());
        Changed(decoder, model.MelVae.DecoderOutput.GetParameters().ToArray());
        Changed(quantizer, model.MelVae.Quantizer.Codebooks[0].GetParameters().ToArray());
        Changed(critics, Parameters(model.CodecTrainer.Critics.Layers));
        Assert.Equal(diffusion, model.DiffusionTransformer.Output.GetParameters().ToArray());
        Assert.Equal(textEncoder, model.TextEncoder.Embedding.GetParameters().ToArray());
        Assert.Equal(frozenDecoder, Parameters(model.CodecTrainer.DecoderLayers));
        if (semantic) Changed(projection, Parameters(model.CodecTrainer.ProjectionLayers));
    }

    [Fact]
    public void SemanticStage_RejectsAnUnloadedDecoder()
    {
        var trainer = new DiTToCodecTrainer<double>(AiDotNetEngine.Current, Options());
        trainer.InitializeSemanticDecoder(8, new Random(1));
        Assert.Throws<InvalidOperationException>(() => trainer.SemanticLoss(new Tensor<double>(new[] { 4, 8 }), "hi"));
    }

    private static void Changed(double[] before, double[] after) => Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Max() > 1e-12);
    private static void Close(double[] actual, double[] expected, double tolerance = 1e-9)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < tolerance, $"Maximum error {error:R}.");
    }
}
