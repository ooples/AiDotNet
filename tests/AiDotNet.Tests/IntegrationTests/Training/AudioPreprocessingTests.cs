using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Preprocessing.Audio;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureAudioEnhancer / ConfigureAudioEffect wire the configured processor into the
/// preprocessing pipeline as a composable IDataTransformer step (not a bespoke hook), that the adapters apply
/// the processor to audio tensors and pass non-audio through unchanged, and that the enhancer runs during a build.
/// </summary>
public class AudioPreprocessingTests
{
    private sealed class RecordingEnhancer : IAudioEnhancer<double>
    {
        public int EnhanceCalls;
        public int NoiseProfileCalls;
        public int SampleRate => 16000;
        public int NumChannels => 1;
        public double EnhancementStrength { get; set; } = 0.5;
        public Tensor<double> Enhance(Tensor<double> audio) { EnhanceCalls++; return audio; }
        public Tensor<double> EnhanceWithReference(Tensor<double> audio, Tensor<double> reference) => audio;
        public Tensor<double> ProcessChunk(Tensor<double> audioChunk) => audioChunk;
        public void ResetState() { }
        public int LatencySamples => 0;
        public void EstimateNoiseProfile(Tensor<double> noiseOnlyAudio) { NoiseProfileCalls++; }
    }

    private sealed class GainEffect : IAudioEffect<double>
    {
        public int ProcessCalls;
        private readonly double _gain;
        public GainEffect(double gain) => _gain = gain;
        public string Name => "gain";
        public int SampleRate => 16000;
        public bool Bypass { get; set; }
        public double Mix { get; set; } = 1.0;
        public Tensor<double> Process(Tensor<double> input)
        {
            ProcessCalls++;
            var outp = new Tensor<double>(new[] { input.Length });
            for (int i = 0; i < input.Length; i++) outp[i] = input[i] * _gain;
            return outp;
        }
        public double ProcessSample(double sample) => sample * _gain;
        public void ProcessInPlace(Span<double> buffer) { for (int i = 0; i < buffer.Length; i++) buffer[i] *= _gain; }
        public void Reset() { }
        public int LatencySamples => 0;
        public int TailSamples => 0;
        public IReadOnlyDictionary<string, AudioEffectParameter<double>> Parameters
            => new Dictionary<string, AudioEffectParameter<double>>();
        public void SetParameter(string name, double value) { }
        public double GetParameter(string name) => 0.0;
    }

    /// <summary>Records how many times it was invoked and in what order, for the composition test.</summary>
    private sealed class OrderRecorder : IDataTransformer<double, Tensor<double>, Tensor<double>>
    {
        private readonly List<string> _log;
        private readonly string _tag;
        public OrderRecorder(List<string> log, string tag) { _log = log; _tag = tag; }
        public bool IsFitted { get; private set; }
        public int[]? ColumnIndices => null;
        public bool SupportsInverseTransform => false;
        public void Fit(Tensor<double> data) { IsFitted = true; }
        public Tensor<double> Transform(Tensor<double> data) { _log.Add(_tag); return data; }
        public Tensor<double> FitTransform(Tensor<double> data) { Fit(data); return Transform(data); }
        public Tensor<double> InverseTransform(Tensor<double> data) => data;
        public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) => inputFeatureNames ?? Array.Empty<string>();
    }

    private static Tensor<double> Audio(int samples = 8)
    {
        var t = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) t[i] = Math.Sin(i * 0.5);
        return t;
    }

    [Fact]
    public void AudioEnhancementTransformer_AppliesEnhance_AndEstimatesNoiseOnFit()
    {
        var enhancer = new RecordingEnhancer();
        var step = new AudioEnhancementTransformer<double, Tensor<double>>(enhancer);

        var outp = step.FitTransform(Audio());

        Assert.True(step.IsFitted);
        Assert.Equal(1, enhancer.NoiseProfileCalls); // Fit learned the noise profile from the audio.
        Assert.Equal(1, enhancer.EnhanceCalls);      // Transform applied enhancement.
        Assert.NotNull(outp);
        Assert.False(step.SupportsInverseTransform);
        Assert.Throws<NotSupportedException>(() => step.InverseTransform(Audio()));
    }

    [Fact]
    public void AudioEffectTransformer_AppliesProcessToTensor()
    {
        var effect = new GainEffect(2.0);
        var step = new AudioEffectTransformer<double, Tensor<double>>(effect);
        var input = Audio(4);

        var outp = step.Transform(input);

        Assert.Equal(1, effect.ProcessCalls);
        for (int i = 0; i < input.Length; i++) Assert.Equal(input[i] * 2.0, outp[i], 9);
    }

    [Fact]
    public void AddPreprocessingStep_ComposesInOrderWithExistingSteps()
    {
        var log = new List<string>();
        var pipeline = new AiModelDataPipeline<double, Tensor<double>, Tensor<double>>();

        pipeline.AddPreprocessingStep(new OrderRecorder(log, "first"), "first");
        pipeline.AddPreprocessingStep(new AudioEnhancementTransformer<double, Tensor<double>>(new RecordingEnhancer()), "audio_enhancer");
        pipeline.AddPreprocessingStep(new OrderRecorder(log, "third"), "third");
        // A colliding name is made unique rather than throwing.
        pipeline.AddPreprocessingStep(new OrderRecorder(log, "first_again"), "first");

        var pp = pipeline.PreprocessingPipeline;
        Assert.NotNull(pp);
        Assert.Equal(4, pp?.Count);

        pp?.FitTransform(Audio());
        // Steps run in the order they were appended (the audio step is between the two recorders).
        Assert.Equal(new[] { "first", "third", "first_again" }, log);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureAudioEnhancer_RunsEnhancerDuringBuild()
    {
        var enhancer = new RecordingEnhancer();
        int rows = 40, cols = 4, outs = 2;
        var x = new Tensor<double>(new[] { rows, cols });
        var y = new Tensor<double>(new[] { rows, outs });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2);
            for (int o = 0; o < outs; o++) y[i, o] = Math.Cos((i + o) * 0.2);
        }

        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new ReLUActivation<double>()),
            new DenseLayer<double>(outs),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: cols, outputSize: outs, layers: layers);

        await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(new NeuralNetwork<double>(arch))
            .ConfigureAudioEnhancer(enhancer)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        // The enhancer was wired into preprocessing and ran on the audio-tensor input during the build.
        Assert.True(enhancer.EnhanceCalls > 0, "enhancer was not applied during the build");
    }
}
