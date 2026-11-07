using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class StyleTts2OptionContextTests
{
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
