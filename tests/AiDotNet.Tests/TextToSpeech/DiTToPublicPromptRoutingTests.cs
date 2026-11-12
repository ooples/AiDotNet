using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class DiTToPublicPromptRoutingTests
{
    [Fact]
    public void PublicSynthesize_ActuallyPassesTheJointTranscriptToTheLengthNetwork()
    {
        using var inference = new NoGradScope<double>();
        var options = DiTToTTSPaperTests.TinyOptions();
        using var model = new DiTToTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        model.SetTrainingMode(false);
        var mel = new Tensor<double>(new[] { 6, 4 });
        int[] reference = model.TokenizeText("old");
        model.Voice = new TtsVoice<double> { Reference = mel,
            ReferenceTokens = new Tensor<double>(new[] { reference.Length }, new Vector<double>(reference.Select(x => (double)x).ToArray())) };
        var prompt = model.MelVae!.Encode(mel);
        var joint = model.TextEncoder!.Forward(model.TokenizeText("old new"));
        var target = model.TextEncoder.Forward(model.TokenizeText("new"));
        int width = options.LengthHiddenDim, classes = options.MaxLatentLength + 1;
        var predictor = model.LengthPredictor!;
        // Read the real decoder's final features through an identity projection.
        // No encoder/decoder forward or sampler is substituted.
        var projection = new double[classes * width];
        for (int i = 0; i < width; i++) projection[i * width + i] = 1;
        TorchParameters.Linear(AiDotNetEngine.Current, predictor.Classifier, width, classes, projection, new double[classes]);
        var a = predictor.Forward(joint, prompt);
        var b = predictor.Forward(target, prompt);
        int row = a.Shape[0] - 1;
        var delta = Enumerable.Range(0, width).Select(i => a[row, i] - b[row, i]).ToArray();
        double squared = delta.Sum(value => value * value);
        Assert.True(squared > 1e-18, "The real text/length networks must distinguish the conditioning inputs.");
        // Separate those two real feature vectors by a large logit margin. This
        // makes a public-path bypass observable even if RVQ collapses waveforms.
        var weights = new double[classes * width];
        var bias = Enumerable.Repeat(-100.0, classes).ToArray();
        double midpoint = 0;
        for (int i = 0; i < width; i++)
        {
            double weight = 80 * delta[i] / squared;
            weights[2 * width + i] = weight;
            weights[4 * width + i] = -weight;
            midpoint += weight * (a[row, i] + b[row, i]) / 2;
        }
        bias[2] = -midpoint;
        bias[4] = midpoint;
        TorchParameters.Linear(AiDotNetEngine.Current, predictor.Classifier, width, classes, weights, bias);
        Assert.Equal(2, predictor.ExpectedRemaining(joint, prompt));
        Assert.Equal(4, predictor.ExpectedRemaining(target, prompt));
        int samplesPerLatent = options.MelVaeDownsampleFactor * options.BigVganUpsampleRates.Aggregate(1, (x, y) => x * y);
        Assert.Equal((prompt.Shape[0] + 2) * samplesPerLatent, model.Synthesize("new").Length);
    }
}
