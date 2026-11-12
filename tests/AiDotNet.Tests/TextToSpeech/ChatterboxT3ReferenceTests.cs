using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class ChatterboxT3ReferenceTests
{
    [Fact]
    public void T3_OriginalForwardLogitsAndNextTokenLossMatchBeforeAndAfterSgd()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "chatterbox_t3_reference.json")));
        var reference = document.RootElement;
        double[] Values(JsonElement values) => values.EnumerateArray().Select(value => value.GetDouble()).ToArray();
        int[] Ids(string key) => Values(reference.GetProperty(key)).Select(value => (int)value).ToArray();
        var options = new ChatterboxOptions
        {
            VocabSize = 32, SpeechVocabularySize = 20, LLMDim = 16, NumLLMLayers = 1, NumHeads = 2,
            KeyValueHeads = 2, LmFeedForwardDim = 32, MaxTextLength = 16, MaxSpeechTokens = 16,
            T3SpeakerEmbeddingDim = 6, PerceiverQueries = 2, PerceiverHeads = 2, PerceiverDropout = 0,
            TextStartToken = 30, TextStopToken = 31, SpeechStartToken = 18, SpeechStopToken = 19,
            RopeTheta = 10000, RopeScalingFactor = 1
        };
        var layers = new List<LayerBase<double>>();
        var model = new ChatterboxT3<double>(AiDotNetEngine.Current, layers, options, new Random(1));
        model.LoadTorchWeights((key, shape) =>
        {
            Assert.Equal(shape, reference.GetProperty("weight_shapes").GetProperty(key).EnumerateArray().Select(value => value.GetInt32()).ToArray());
            return Values(reference.GetProperty("weights").GetProperty(key));
        });
        var condition = new ChatterboxT3Condition<double>(new Tensor<double>(new[] { 6 },
            new Vector<double>(Values(reference.GetProperty("speaker")))), Ids("prompt"), .7);
        var logits = model.TeacherForcedLogits(Ids("text"), Ids("speech"), condition);
        Close(logits.Text, Values(reference.GetProperty("text_logits")));
        Close(logits.Speech, Values(reference.GetProperty("speech_logits")));
        Tensor<double> PairedIds(string key)
        {
            var ids = Ids(key).Select(value => (double)value).ToArray();
            return new Tensor<double>(new[] { 2, ids.Length }, new Vector<double>(ids.Concat(ids).ToArray()));
        }
        Close(model.PrepareEmbeddings(condition, PairedIds("text"), PairedIds("speech"), true),
            Values(reference.GetProperty("cfg_embeddings")));
        // The released helper fails its CE-axis contract; the native training objective follows the documented AR factorization.
        Assert.Contains("Expected target size", reference.GetProperty("original_training_error").GetString());
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        IReadOnlyDictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var loss = model.Loss(Ids("text").Skip(1).Take(2).ToArray(), Ids("speech").Skip(1).Take(3).ToArray(), condition);
            Assert.True(Math.Abs(reference.GetProperty("next_token_loss").GetDouble() - loss[0]) < 2e-6,
                $"Expected next-token loss {reference.GetProperty("next_token_loss").GetDouble():R}, actual {loss[0]:R}.");
            gradients = tape.ComputeGradients(loss, sources: parameters);
        }
        foreach (var parameter in parameters)
        {
            if (!gradients.TryGetValue(parameter, out var gradient)) continue;
            for (int i = 0; i < parameter.Length; i++) parameter.SetFlat(i, parameter.GetFlat(i) - .01 * gradient.GetFlat(i));
            AiDotNetEngine.Current.InvalidatePersistentTensor(parameter);
        }
        logits = model.TeacherForcedLogits(Ids("text"), Ids("speech"), condition);
        Close(logits.Text, Values(reference.GetProperty("after_sgd_text")));
        Close(logits.Speech, Values(reference.GetProperty("after_sgd_speech")));
    }

    private static void Close(Tensor<double> actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 2e-6, $"Maximum T3 error {error:R}.");
    }
}
