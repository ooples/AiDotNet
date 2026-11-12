using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Latest;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class MegaTtsOptionTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    [Fact]
    public void PhoneVocabularySize_MustMatchTheConfiguredSymbols()
    {
        var first = MegaTtsModelReferenceTests.TinyOptions();
        first.VocabSize = int.MaxValue;
        Assert.Throws<ArgumentException>("VocabSize", () => new MegaTTS<double>(Architecture(), first));
        var second = MegaTts2ModelReferenceTests.TinyOptions();
        second.VocabSize = int.MaxValue;
        Assert.Throws<ArgumentException>("VocabSize", () => new MegaTTS2<double>(Architecture(), second));
    }

    [Fact]
    public void ContentDropout_ActsOnlyWhileTraining()
    {
        int[,] phones = { { 1, 2, 3, 4, 5, 6, 7, 8 } };
        var firstOptions = MegaTtsModelReferenceTests.TinyOptions();
        firstOptions.DropoutRate = 0.5;
        using var first = new MegaTTS<double>(Architecture(), firstOptions);
        Check(first, () => first.Network.Content.Forward(phones).ToArray());

        var secondOptions = MegaTts2ModelReferenceTests.TinyOptions();
        secondOptions.DropoutRate = 0.5;
        using var second = new MegaTTS2<double>(Architecture(), secondOptions);
        Check(second, () => second.Network.Content.Forward(phones).ToArray());

        static void Check(TtsModelBase<double> model, Func<double[]> forward)
        {
            model.SetTrainingMode(false);
            var evaluation = forward();
            model.SetTrainingMode(true);
            Assert.False(evaluation.SequenceEqual(forward()));
            model.SetTrainingMode(false);
            Assert.Equal(evaluation, forward());
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ContextLimit_RejectsAnExtraSentenceInProsodyTraining()
    {
        await Task.Yield();
        using var model = MegaTtsFixture.Create<double>();
        ((MegaTTSOptions)model.GetOptions()).ContextSentences = 1;
        var first = MegaTtsOptionFixture.Sample(model);
        var second = MegaTtsOptionFixture.Sample(model, "Good morning.");
        Assert.Throws<ArgumentException>("sentences", () => model.TrainProsodyModel(new[] { first, second }));
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(model.TrainProsodyModel(new[] { first })));
    }

    [Fact(Timeout = 120000)]
    public async Task ProsodyBudget_RejectsTheActualOversizeLanguageModelSequence()
    {
        await Task.Yield();
        using var model = MegaTts2Fixture.Create<double>();
        var options = (MegaTTS2Options)model.GetOptions();
        options.Stage = MegaTts2Stage.LanguageModels;
        var speakers = new IReadOnlyList<TtsTrainingSample<double>>[] { new[] { MegaTtsOptionFixture.Sample(model) } };
        int actualTokens = model.LanguageSequenceOf(speakers).Tokens.Length;
        Assert.True(actualTokens > 1);
        options.ProsodyTokenBudget = actualTokens - 1;
        Assert.Throws<ArgumentException>("speakers", () => model.TrainLanguageModels(speakers));
        options.ProsodyTokenBudget = actualTokens;
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(model.TrainLanguageModels(speakers)));
    }
}
