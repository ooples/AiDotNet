using System;
using System.IO;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// Contracts for the audio front-end: every model that relies on the base preprocessing must get a
/// real spectral transform, and features without a time axis must be rejected where they are
/// produced rather than deep inside an attention layer.
/// </summary>
/// <remarks>
/// <para>
/// These exist because of a defect that was replicated across most of the audio surface: the mel
/// front-end was optional, and 132 models wrote some form of
/// <c>if (MelSpec is not null) return MelSpec.Forward(raw); return raw;</c>. When the front-end had
/// not been assigned, that silently forwarded the RAW WAVEFORM — rank 1, no time axis, no frequency
/// axis — into a transformer encoder. It did not fail there. It failed much later and much less
/// legibly, as <c>MultiHeadAttentionLayer requires rank&gt;=2 input; got rank 1</c>, which is how 29
/// CI failures across four models presented.
/// </para>
/// <para>
/// The tests target the MECHANISM rather than enumerating models. Instantiating every audio model
/// needs per-model architecture arguments, and a list of names would rot; the base default and the
/// guard are what every model inherits, so verifying those covers the ones that exist today and the
/// ones added tomorrow.
/// </para>
/// </remarks>
public sealed class AudioFrontEndContractTests
{
    /// <summary>
    /// Minimal concrete audio model: implements only what the bases require and overrides nothing
    /// about preprocessing, so it exercises exactly the inherited path.
    /// </summary>
    private sealed class BareAudioModel : AudioNeuralNetworkBase<double>
    {
        public BareAudioModel(NeuralNetworkArchitecture<double> architecture)
            : base(architecture)
        {
        }

        /// <summary>Exposes the protected inherited preprocessing for assertion.</summary>
        public Tensor<double> Preprocess(Tensor<double> rawAudio) => PreprocessAudio(rawAudio);

        /// <summary>Exposes the protected guard for assertion.</summary>
        public void Guard(Tensor<double> features) => RequireTimeAxis(features);

        protected override void InitializeLayers()
        {
        }

        protected override Tensor<double> PostprocessOutput(Tensor<double> modelOutput) => modelOutput;

        public override ModelMetadata<double> GetModelMetadata() => new();

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
        }
    }

    private static BareAudioModel CreateModel()
        => new(new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            inputSize: WaveformSamples,
            outputSize: 1));

    /// <summary>Sample count for the synthetic waveform; also the architecture's declared input size.</summary>
    private const int WaveformSamples = 16000;

    /// <summary>One second of a 440 Hz tone at 16 kHz — a rank-1 waveform, as a caller supplies it.</summary>
    private static Tensor<double> CreateWaveform(int samples = WaveformSamples)
    {
        var wave = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            wave[i] = Math.Sin(2.0 * Math.PI * 440.0 * i / 16000.0);
        }

        return wave;
    }

    [Fact]
    public void MelSpec_IsNeverNull_SoNoModelCanFallBackToTheRawWaveform()
    {
        using var model = CreateModel();

        // The property used to be nullable, which is what made `return rawAudio` look reasonable.
        Assert.NotNull(model.Preprocess(CreateWaveform()));
    }

    [Fact]
    public void PreprocessAudio_Default_ProducesFeaturesWithATimeAxis()
    {
        using var model = CreateModel();

        var features = model.Preprocess(CreateWaveform());

        Assert.True(
            features.Shape.Length >= 2,
            $"Base preprocessing returned rank {features.Shape.Length}. Audio features must carry a " +
            "time axis; a rank-1 result gives any downstream attention a sequence of length one.");
    }

    [Fact]
    public void PreprocessAudio_Default_DoesNotReturnTheWaveformItself()
    {
        using var model = CreateModel();
        var waveform = CreateWaveform();

        var features = model.Preprocess(waveform);

        // The precise regression: handing the input straight back. Distinct from the rank check,
        // because a model could return a reshaped waveform and still satisfy rank >= 2.
        Assert.False(
            ReferenceEquals(waveform, features),
            "Base preprocessing returned the input tensor. That is the raw-waveform passthrough this " +
            "contract exists to prevent.");
    }

    [Fact]
    public void RequireTimeAxis_RejectsRankOneFeatures_NamingTheModel()
    {
        using var model = CreateModel();

        var ex = Assert.Throws<InvalidOperationException>(() => model.Guard(CreateWaveform(64)));

        // Failing at the boundary is the point: the message names the model, where the old failure
        // surfaced inside MultiHeadAttentionLayer with no indication of which model produced it.
        Assert.Contains(nameof(BareAudioModel), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RequireTimeAxis_AcceptsFeaturesThatCarryTime()
    {
        using var model = CreateModel();

        model.Guard(new Tensor<double>(new[] { 8, 64 }));
    }
}
