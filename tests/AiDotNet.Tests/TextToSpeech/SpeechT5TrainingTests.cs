using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpeechT5's training stages on a tiny model: the joint pre-training on a recording with HuBERT labels and on text alone,
/// and text-to-speech fine-tuning with the encoder freeze.
/// </summary>
public class SpeechT5TrainingTests
{
    private static SpeechT5<double> Model(SpeechT5Options options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            options);

    private static SpeechT5Options Tiny(SpeechT5Stage stage)
    {
        var o = SpeechT5HuggingFaceReferenceTests.TinyOptions();
        o.VocabSize = 81;
        o.MaskId = 79;
        o.HopSize = 256;
        o.VocoderInitialChannels = 64;
        o.VocoderUpsampleRates = new[] { 4, 4, 4, 4 };
        o.VocoderUpsampleKernels = new[] { 8, 8, 8, 8 };
        o.CurrentStage = stage;
        o.PretrainUpdates = 100;
        o.WarmupUpdates = 1;
        o.FreezeEncoderUpdates = 1;
        return o;
    }

    private static Tensor<double> Recording(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * 220 * i / 16000.0) + 0.1 * Math.Sin(2 * Math.PI * 1330 * i / 16000.0 + 0.4);
        return audio;
    }

    private static double[] Snapshot(SpeechT5<double> model, Func<SpeechT5<double>, AiDotNet.NeuralNetworks.Layers.LayerBase<double>> layer) =>
        layer(model).GetParameters().ToArray();

    [Fact(Timeout = 300000)]
    public async Task Pretraining_TrainsOnSpeechAndOnText()
    {
        await Task.Yield();
        var model = Model(Tiny(SpeechT5Stage.Pretraining));
        var audio = Recording(8000);
        // 8000 samples give 399 feature frames at the tiny pre-net's stride 20, which 25 labels at 50 Hz cover.
        var labels = new Tensor<double>(new[] { 25 });
        for (int i = 0; i < 25; i++) labels[i] = (7 * i + 3) % 16;
        var tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(model.TokenizeText("pre-training text"));
        var speech = new TtsTrainingSample<double> { Tokens = tokens, Audio = audio, CodecTokens = labels };
        var text = new TtsTrainingSample<double> { Tokens = tokens };

        var hubertBefore = Snapshot(model, m => m.Core.Hubert.LabelEmbeddings);
        double speechLoss = model.Train(speech);
        Assert.True(double.IsFinite(speechLoss) && speechLoss > 0, $"speech loss {speechLoss}");
        Assert.NotEqual(hubertBefore, Snapshot(model, m => m.Core.Hubert.LabelEmbeddings));

        var embeddingBefore = Snapshot(model, m => m.Core.Embedding);
        double textLoss = model.Train(text);
        Assert.True(double.IsFinite(textLoss) && textLoss > 0, $"text loss {textLoss}");
        Assert.NotEqual(embeddingBefore, Snapshot(model, m => m.Core.Embedding));
        Assert.Equal(2, model.Updates);
        Assert.True(model.Core.Quantizer.Temperature < 2.0);
    }

    [Fact(Timeout = 300000)]
    public async Task TextToSpeech_FreezesTheEncoderFirst()
    {
        await Task.Yield();
        var model = Model(Tiny(SpeechT5Stage.TextToSpeech));
        var sample = new TtsTrainingSample<double>
        {
            Tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(model.TokenizeText("hello")), Audio = Recording(6400),
        };
        var encoderBefore = Snapshot(model, m => m.Core.Encoder.InputNorm);
        var decoderBefore = Snapshot(model, m => m.Core.SpeechPostnet.FeatureOut);
        double first = model.Train(sample);
        Assert.True(double.IsFinite(first), $"loss {first}");
        Assert.Equal(encoderBefore, Snapshot(model, m => m.Core.Encoder.InputNorm));
        Assert.NotEqual(decoderBefore, Snapshot(model, m => m.Core.SpeechPostnet.FeatureOut));

        model.Train(sample);
        Assert.NotEqual(encoderBefore, Snapshot(model, m => m.Core.Encoder.InputNorm));
        double evaluated = model.EvaluateTrainingObjective(sample);
        Assert.Equal(evaluated, model.EvaluateTrainingObjective(sample));
    }
}
