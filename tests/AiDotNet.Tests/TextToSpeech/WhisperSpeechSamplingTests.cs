using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class WhisperSpeechSamplingTests
{
    [Fact(Timeout = 120000)]
    public async Task TopK_ChangesTheActualSemanticSamplerAfterRecordingTraining()
    {
        await Task.Yield();
        var options = WhisperSpeechFixture.Options();
        using var model = new WhisperSpeech<double>(
            new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        for (int step = 0; step < 3; step++)
            Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(model.Train(WhisperSpeechFixture.Sample<double>())));
        bool changed = false;
        foreach (string text in new[] { "a tone", "tone", "a", "this is a tone", "another tone" })
        {
            options.TopK = null;
            var all = model.TextToSemanticTokens(text);
            Assert.Equal(all, model.TextToSemanticTokens(text));
            options.TopK = 1;
            var greedy = model.TextToSemanticTokens(text);
            Assert.Equal(greedy, model.TextToSemanticTokens(text));
            changed |= !all.SequenceEqual(greedy);
        }
        Assert.True(changed, "Actual semantic sampling must respond to top-k filtering with the same learned parameters.");
    }
}
