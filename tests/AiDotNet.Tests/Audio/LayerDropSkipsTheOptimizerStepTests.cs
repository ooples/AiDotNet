using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// LayerDrop leaves a skipped layer out of the step: it gets no gradient, so the optimizer must not move it. Blending
/// the layer out with a zero multiplier hands the optimizer a zero gradient instead, and a momentum optimizer then
/// still moves a layer that earlier steps trained.
/// </summary>
public class LayerDropSkipsTheOptimizerStepTests
{
    [Theory(Timeout = 120000)]
    [InlineData("Wav2Vec2")]
    [InlineData("HuBERT")]
    [InlineData("XLSR")]
    public async Task ALayerSkippedByLayerDropIsNotSteppedOnThatStep(string kind)
    {
        await Task.Yield();
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.SequenceToSequence, inputSize: 1600, outputSize: 8) { RandomSeed = 7 };
        TOptions Configure<TOptions>(TOptions options) where TOptions : ModelOptions
        {
            void Set(string name, object value) => options.GetType().GetProperty(name)?.SetValue(options, value);
            options.Seed = 7;
            Set("EncoderDim", 16); Set("HiddenDim", 16);
            Set("NumEncoderLayers", 3); Set("NumTransformerLayers", 3);
            Set("NumAttentionHeads", 2); Set("NumHeads", 2);
            Set("FeedForwardDim", 32); Set("FfDim", 32); Set("VocabSize", 8);
            foreach (string name in new[] { "DropoutRate", "HiddenDropout", "AttentionDropout", "ActivationDropout",
                "FeatureProjectionDropout", "FinalDropout", "MaskTimeProbability" }) Set(name, 0.0);
            Set("LayerDrop", 0.5);
            return options;
        }
        using NeuralNetworkBase<double> model = kind switch
        {
            "HuBERT" => new AiDotNet.SpeechRecognition.Foundation.HuBERTASR<double>(architecture, Configure(new AiDotNet.SpeechRecognition.Foundation.HuBERTASROptions())),
            "XLSR" => new AiDotNet.SpeechRecognition.Multilingual.XLSR<double>(architecture, Configure(new AiDotNet.SpeechRecognition.Multilingual.XLSROptions())),
            _ => new AiDotNet.Audio.SpeechRecognition.Wav2Vec2Model<double>(architecture, Configure(new Wav2Vec2ModelOptions())),
        };
        var waveform = new Tensor<double>(new[] { 1600 });
        for (int i = 0; i < waveform.Length; i++) waveform[i] = .1 * Math.Sin(.03 * i) + .05 * Math.Cos(.11 * i);
        var output = model.Predict(waveform);
        var target = new Tensor<double>(output.Shape.ToArray());
        int vocabulary = output.Shape[^1];
        for (int i = 0; i < target.Length; i++) target[i] = i % vocabulary == 2 ? 1 : 0;

        const int steps = 16;
        int layers = model.Layers.Count;
        var trainedBefore = new bool[layers];
        int heldAfterTraining = 0, headSteps = 0;
        double[][] Snapshot() => model.Layers.Select(layer => layer.GetParameters().ToArray()).ToArray();
        var before = Snapshot();
        for (int step = 0; step < steps; step++)
        {
            model.Train(waveform, target);
            var after = Snapshot();
            if (!before[layers - 1].SequenceEqual(after[layers - 1])) headSteps++;
            for (int layer = 0; layer < layers; layer++)
            {
                if (before[layer].Length == 0) continue;
                bool moved = !before[layer].SequenceEqual(after[layer]);
                // A layer that has momentum from an earlier step and is bit-identical after this one was not stepped.
                if (!moved && trainedBefore[layer]) heldAfterTraining++;
                trainedBefore[layer] |= moved;
            }
            before = after;
        }

        // The head is never skipped, so it trains on every step; half of the encoder-layer steps are skips.
        Assert.Equal(steps, headSteps);
        Assert.True(heldAfterTraining > 0,
            "Every layer moved on every step after its first update, so a skipped layer was still stepped by the optimizer.");
    }
}
