using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class StyleTts2OptionContextTests
{
    [Fact]
    public void SlmSampleRateChangesTheActualFrozenWavLmObjective()
    {
        var options = StyleTts2Fixture.Options();
        using var control = StyleTts2Fixture.Create(options);
        using var changed = StyleTts2Fixture.Create(new StyleTTS2Options(options) { SlmSampleRate = 32000 });
        control.CurrentStage = changed.CurrentStage = StyleTts2TrainingStage.Joint;
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        var sample = StyleTts2Fixture.Sample();
        double first = control.EvaluateTrainingObjective(sample, adversarialTerms: true);
        double second = changed.EvaluateTrainingObjective(sample, adversarialTerms: true);
        Assert.True(TtsPaperNumerics.IsFinite(first));
        Assert.True(TtsPaperNumerics.IsFinite(second));
        Assert.NotEqual(first, second);
    }

    [Theory]
    [InlineData(StyleTts2DecoderType.HiFiGan, 2, 4, 20, 5)]
    [InlineData(StyleTts2DecoderType.HiFiGan, 4, 2, 20, 5)]
    [InlineData(StyleTts2DecoderType.HiFiGan, 3, 2, 20, 5)]
    [InlineData(StyleTts2DecoderType.IstftNet, 2, 4, 20, 5)]
    [InlineData(StyleTts2DecoderType.IstftNet, 2, 4, 24, 4)]
    [InlineData(StyleTts2DecoderType.IstftNet, 2, 4, 4, 1)]
    public void PairedDecoderGeometryControlsTheNativeWaveform(StyleTts2DecoderType type, int firstRate, int secondRate, int fft, int istftHop)
    {
        var options = StyleTts2Fixture.Options();
        options.DecoderType = type;
        options.UpsampleRates = new[] { firstRate, secondRate };
        options.UpsampleKernels = new[] { 2 * firstRate, 2 * secondRate };
        options.IstftFftSize = fft; options.IstftHopSize = istftHop;
        options.HopSize = firstRate * secondRate * (type == StyleTts2DecoderType.IstftNet ? istftHop : 1);
        using var model = StyleTts2Fixture.Create(options);
        var asr = new Tensor<double>(new[] { 1, options.HiddenDim, 3 });
        for (int i = 0; i < asr.Length; i++) asr[i] = 0.1 * Math.Sin(i * 0.17);
        var pitch = new Tensor<double>(new[] { 6 });
        var energy = new Tensor<double>(new[] { 6 });
        for (int i = 0; i < 6; i++) { pitch[i] = 180; energy[i] = 0.3; }
        var style = new Tensor<double>(new[] { 1, options.StyleDim });
        var wave = model.DecoderNetwork.Forward(asr, pitch, energy, style, false, new Random(7));
        Assert.Equal(6 * options.HopSize, wave.Length);
        Assert.All(wave, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.True(wave.Any(value => Math.Abs(value) > 1e-12));
    }

    [Theory]
    [InlineData("UpsampleRates")]
    [InlineData("UpsampleKernels")]
    [InlineData("HopSize")]
    [InlineData("IstftHopSize")]
    public void UnpairedDecoderGeometryIsRejectedBeforeAllocation(string option)
    {
        var options = StyleTts2Fixture.Options();
        switch (option)
        {
            case "UpsampleRates": options.UpsampleRates = new[] { 4, 4 }; break;
            case "UpsampleKernels": options.UpsampleKernels = new[] { 5, 8 }; break;
            case "HopSize": options.HopSize++; break;
            case "IstftHopSize": options.DecoderType = StyleTts2DecoderType.IstftNet; options.HopSize = 40; options.IstftHopSize = 4; break;
            default: throw new ArgumentOutOfRangeException(nameof(option));
        }
        Assert.Throws<ArgumentException>(() => StyleTts2Fixture.Create(options));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1)]
    [InlineData(1.8)]
    [InlineData(double.NaN)]
    public void InvalidBertAdamMomentumIsRejected(double beta1)
    {
        var options = StyleTts2Fixture.Options(); options.BertBeta1 = beta1;
        Assert.Throws<ArgumentOutOfRangeException>(() => StyleTts2Fixture.Create(options));
    }

    [Fact]
    public void FullDiagnosticObjectiveIncludesWaveAdversaryWithoutChangingWeights()
    {
        using var model = StyleTts2Fixture.Create(StyleTts2Fixture.Options());
        model.CurrentStage = StyleTts2TrainingStage.Joint;
        var before = model.GetParameters().ToArray();
        var sample = StyleTts2Fixture.Sample();
        double stable = model.EvaluateTrainingObjective(sample);
        double complete = model.EvaluateTrainingObjective(sample, adversarialTerms: true);
        Assert.True(TtsPaperNumerics.IsFinite(stable));
        Assert.True(TtsPaperNumerics.IsFinite(complete));
        Assert.NotEqual(stable, complete);
        Assert.Equal(before, model.GetParameters().ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void InvalidDiscriminatorWidthDivisorsAreRejectedBeforeAllocation(int divisor)
    {
        var options = StyleTts2Fixture.Options(); options.DiscriminatorWidthDivisor = divisor;
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => StyleTts2Fixture.Create(options));
        Assert.Equal(nameof(options.DiscriminatorWidthDivisor), error.ParamName);
    }

    [Theory]
    [InlineData(StyleTts2TrainingStage.Pretraining)]
    [InlineData(StyleTts2TrainingStage.Joint)]
    public void CompleteNativeStagesUpdateWeightsInsteadOfSkippingShortFixtures(StyleTts2TrainingStage stage)
    {
        using var model = StyleTts2Fixture.Create(StyleTts2Fixture.Options());
        model.CurrentStage = stage;
        var before = model.GetParameters().ToArray();
        var discriminator = model.WaveDiscriminators.Layers.SelectMany(layer => layer.GetParameters()).ToArray();
        var slmLayers = (IEnumerable<LayerBase<double>>)typeof(StyleTTS2<double>).GetField("_slmLayers",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;
        var frozen = slmLayers.SelectMany(layer => layer.GetParameters()).ToArray();
        double loss = model.Train(StyleTts2Fixture.Sample());
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.Equal(0, model.SkippedSamples);
        Assert.Equal(1, stage == StyleTts2TrainingStage.Pretraining ? model.PretrainingUpdates : model.JointUpdates);
        Assert.False(before.SequenceEqual(model.GetParameters()));
        Assert.Equal(frozen, slmLayers.SelectMany(layer => layer.GetParameters()).ToArray());
        bool discriminatorChanged = !discriminator.SequenceEqual(model.WaveDiscriminators.Layers.SelectMany(layer => layer.GetParameters()));
        Assert.Equal(stage == StyleTts2TrainingStage.Joint, discriminatorChanged);
    }
}
