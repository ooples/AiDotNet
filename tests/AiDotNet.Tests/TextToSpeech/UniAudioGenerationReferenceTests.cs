using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class UniAudioGenerationReferenceTests
{
    [Fact]
    public void ReleasedPreset_DoesNotAddAnUnpublishedMinimumFrameFloor()
        => Assert.Equal(0, UniAudioOptions.OfficialV1().MinimumGeneratedFrames);

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void PublicSynthesis_UsesAuthorLengthGateWithoutInventingAnEmptyCandidate(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "uniaudio_generation_reference.json")));
        var reference = document.RootElement.GetProperty("cases")[index];
        var options = new UniAudioOptions
        {
            SampleRate = 320, HopSize = 32, CodecFrameRate = 10, MelChannels = 8,
            UseReleasedTextFrontend = false, NumCodebooks = 3, CodebookSize = 8,
            CodecTotalQuantizers = 3, CodecFilters = 2, CodecDimension = 6,
            CodecRatios = new[] { 4, 2, 2, 2 }, CodecKMeansIterations = 0, CodecDeadCodeThreshold = 0,
            LLMDim = 12, NumHeads = 3, NumEncoderLayers = 1, NumLLMLayers = 1,
            GlobalFeedForwardDim = 24, LocalFeedForwardDim = 24,
            JointVocabularySize = 160, VocabSize = 160, AudioTokenOffset = 32,
            PhoneTokenOffset = 57, PhoneVocabularySize = 80, MaxTextLength = 8,
            MaxContextPatches = 64, MaxCodecFrames = 64, PromptSeconds = .1,
            TopK = 1, Temperature = 1, MinimumGeneratedFrames = 0, MaximumGeneratedFrames = 4,
            MinimumLengthRatio = reference.GetProperty("minimum_ratio").GetDouble(),
            MaximumLengthRatio = 4.0 / 9, SamplingSeed = 7,
        };
        var model = new UniAudio<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 8, outputSize: 160), options);
        var languageModel = (UniAudioMultiScaleTransformer<double>)typeof(UniAudio<double>)
            .GetField("_languageModel", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        languageModel.LoadTorchWeights((name, shape) =>
        {
            var values = new double[shape.Aggregate(1, (a, b) => a * b)];
            // Constant final local hidden vector [1,0,...] yields unique EOS/code maxima,
            // matching the controlled author logits without replacing the real transformer.
            if (name == "l_ln.bias") values[0] = 1;
            if (name == "lm_head.weight")
            {
                values[options.AudioEndToken * options.LLMDim] = 20;
                for (int q = 0; q < options.NumCodebooks; q++)
                    values[(options.AudioTokenOffset + q * options.CodebookSize) * options.LLMDim] = 10;
            }
            return values;
        });
        var recording = new Tensor<double>(new[] { 32 });
        var phoneTensor = model.TokenizePhones("a");
        var phones = Enumerable.Range(0, phoneTensor.Length).Select(i => (int)phoneTensor[i]).ToArray();
        Assert.Single(phones);
        var prompt = (int[])typeof(UniAudio<double>).GetMethod("PromptTokens", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { recording })!;
        var prefix = (System.Collections.Generic.List<int>)typeof(UniAudio<double>)
            .GetMethod("BuildPrefix", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { phones, prompt })!;
        Assert.Equal(reference.GetProperty("prefix_patches").GetInt32(), prefix.Count / options.NumCodebooks);
        var waveform = model.SynthesizeWithVoice("a", recording, options.SampleRate);
        Assert.Equal(reference.GetProperty("codec_frames").GetInt32() * options.HopSize, waveform.Length);
    }
}
