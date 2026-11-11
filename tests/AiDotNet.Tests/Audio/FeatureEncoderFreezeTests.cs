using System.Threading.Tasks;
using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class FeatureEncoderFreezeTests
{
    [Theory(Timeout = 120000)]
    [InlineData("HuBERT", true)]
    [InlineData("HuBERT", false)]
    [InlineData("WavLM", true)]
    [InlineData("WavLM", false)]
    [InlineData("WavLMRobust", true)]
    [InlineData("WavLMRobust", false)]
    [InlineData("XLSR", true)]
    [InlineData("XLSR", false)]
    [InlineData("Omnilangual", true)]
    [InlineData("Omnilangual", false)]
    [InlineData("Wav2Vec2", true)]
    [InlineData("Wav2Vec2", false)]
    public async Task FineTuningFreezesOnlyTheConfiguredFeatureEncoder(string kind, bool freeze)
    {
        await Task.Yield();
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.SequenceToSequence, inputSize: 1600, outputSize: 8) { RandomSeed = 7 };
        TOptions Configure<TOptions>(TOptions options) where TOptions : ModelOptions
        {
            void Set(string name, object value) => options.GetType().GetProperty(name)?.SetValue(options, value);
            options.Seed = 7;
            Set("EncoderDim", 16); Set("HiddenDim", 16);
            Set("NumEncoderLayers", 1); Set("NumTransformerLayers", 1);
            Set("NumAttentionHeads", 2); Set("NumHeads", 2);
            Set("FeedForwardDim", 32); Set("FfDim", 32); Set("VocabSize", 8);
            foreach (string name in new[] { "DropoutRate", "HiddenDropout", "AttentionDropout", "ActivationDropout",
                "FeatureProjectionDropout", "FinalDropout", "LayerDrop", "MaskTimeProbability" }) Set(name, 0.0);
            Set("FreezeFeatureEncoder", freeze);
            return options;
        }
        using NeuralNetworkBase<double> model = kind switch
        {
            "HuBERT" => new AiDotNet.SpeechRecognition.Foundation.HuBERTASR<double>(architecture, Configure(new AiDotNet.SpeechRecognition.Foundation.HuBERTASROptions())),
            "WavLM" => new AiDotNet.SpeechRecognition.Foundation.WavLMASR<double>(architecture, Configure(new AiDotNet.SpeechRecognition.Foundation.WavLMASROptions())),
            "WavLMRobust" => new AiDotNet.SpeechRecognition.Robust.WavLMRobust<double>(architecture, Configure(new AiDotNet.SpeechRecognition.Robust.WavLMRobustOptions())),
            "XLSR" => new AiDotNet.SpeechRecognition.Multilingual.XLSR<double>(architecture, Configure(new AiDotNet.SpeechRecognition.Multilingual.XLSROptions())),
            "Omnilangual" => new AiDotNet.SpeechRecognition.Multilingual.OmnilangualASR<double>(architecture, Configure(new AiDotNet.SpeechRecognition.Multilingual.OmnilangualASROptions())),
            _ => new AiDotNet.Audio.SpeechRecognition.Wav2Vec2Model<double>(architecture, Configure(new Wav2Vec2ModelOptions())),
        };
        var waveform = new Tensor<double>(new[] { 1600 });
        for (int i = 0; i < waveform.Length; i++) waveform[i] = .1 * Math.Sin(.03 * i) + .05 * Math.Cos(.11 * i);
        var output = model.Predict(waveform);
        var target = new Tensor<double>(output.Shape.ToArray());
        int vocabulary = output.Shape[^1];
        for (int i = 0; i < target.Length; i++) target[i] = i % vocabulary == 2 ? 1 : 0;
        var before = model.Layers[0].GetParameters().ToArray();
        var beforeHead = model.Layers.Last().GetParameters().ToArray();
        for (int i = 0; i < 2; i++) model.Train(waveform, target);
        Assert.Equal(freeze, before.SequenceEqual(model.Layers[0].GetParameters()));
        Assert.False(beforeHead.SequenceEqual(model.Layers.Last().GetParameters()));
    }
}
