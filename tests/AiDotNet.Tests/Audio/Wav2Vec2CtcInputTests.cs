using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.Audio;

[Collection("NonParallelIntegration")]
public sealed class Wav2Vec2CtcInputTests
{
    [Fact]
    public void CtcObjectiveUsesTheTrainingForwardLogits()
    {
        var previous = AiDotNetEngine.Current;
        var options = TensorCodecOptions.Current;
        var engine = new CtcObservingEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            using var model = new SuppliedLogitsModel();
            var input = new Tensor<double>(new[] { 1600 });
            model.Train(input, new Tensor<double>(new[] { 1.0 }, new[] { 1 }));
            Assert.Equal(1, model.Forwards);
            Assert.NotNull(engine.CtcLogProbabilities);
            Assert.Equal(new[] { 3, 1, 8 }, engine.CtcLogProbabilities.Shape.ToArray());
            Assert.All(engine.CtcLogProbabilities.ToArray(), value => Assert.InRange(Math.Abs(value + Math.Log(8)), 0, 1e-12));
        }
        finally
        {
            TensorCodecOptions.SetCurrent(options);
            AiDotNetEngine.Current = previous;
        }
    }

    [Fact]
    public void ShapedNativeTranscriptionKeepsSingleStreamPreparationWhilePredictionKeepsBatch()
    {
        using var model = new Wav2Vec2Model<double>(Architecture(), Options(), Vocabulary());
        var shaped = new Tensor<double>(new[] { 2, 800 });
        for (int i = 0; i < shaped.Length; i++) shaped[i] = .1 * Math.Sin(.03 * i);
        var flat = new Tensor<double>(shaped.ToArray(), new[] { shaped.Length });
        Assert.Equal(model.Transcribe(flat).Text, model.Transcribe(shaped).Text);
        var prediction = model.Predict(shaped);
        Assert.Equal(3, prediction.Rank);
        Assert.Equal(2, prediction.Shape[0]);
    }

    private static NeuralNetworkArchitecture<double> Architecture() => new(InputType.OneDimensional,
        NeuralNetworkTaskType.SequenceToSequence, inputSize: 1600, outputSize: 8) { RandomSeed = 7 };
    private static Wav2Vec2ModelOptions Options() => new()
    {
        Seed = 7, HiddenDim = 16, NumTransformerLayers = 1, NumHeads = 2, FfDim = 32,
        FinalDropout = 0, HiddenDropout = 0, AttentionDropout = 0, ActivationDropout = 0,
        LayerDrop = 0, MaskTimeProbability = 0,
    };
    private static string[] Vocabulary() => new[] { "<blank>", "a", "b", "c", "d", "e", "f", "|" };

    private sealed class SuppliedLogitsModel : Wav2Vec2Model<double>
    {
        internal int Forwards;
        internal SuppliedLogitsModel() : base(Wav2Vec2CtcInputTests.Architecture(), Wav2Vec2CtcInputTests.Options(), Vocabulary())
        {
            foreach (var parameter in ((LayerBase<double>)Layers.Last()).GetTrainableParameters()) parameter.Fill(0);
        }
        public override Tensor<double> ForwardForTraining(Tensor<double> input)
        {
            Forwards++;
            var bias = ((LayerBase<double>)Layers.Last()).GetTrainableParameters().Last();
            return Engine.TensorTile(Engine.Reshape(bias, new[] { 1, 8 }), new[] { 3, 1 });
        }
    }

    private sealed class CtcObservingEngine : CpuEngine
    {
        internal Tensor<double>? CtcLogProbabilities;
        public override Tensor<T> TensorPermute<T>(Tensor<T> input, int[] permutation)
        {
            var result = base.TensorPermute(input, permutation);
            if (CtcLogProbabilities is null && result is Tensor<double> values
                && permutation.SequenceEqual(new[] { 1, 0, 2 }) && values.Shape[2] == 8)
                CtcLogProbabilities = new Tensor<double>(values.ToArray(), values.Shape.ToArray());
            return result;
        }
    }
}
