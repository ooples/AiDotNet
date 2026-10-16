using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The CLAP network is LAION-CLAP's (issue #2313): HTS-AT with bn0, four hierarchical Swin stages and patch merging;
/// RoBERTa with its pooled output; Linear → ReLU → Linear projections; feature fusion; the 50 Hz – 14 kHz front end.
/// Every assertion here fails on the network this replaced.
/// </summary>
public class ClapArchitectureTests
{
    [Fact]
    public void Defaults_AreTheHtsatTinyRobertaBaseFusionModel()
    {
        var o = new CLAPModelOptions();
        Assert.Equal(48000, o.SampleRate);
        Assert.Equal(1024, o.StftWindowSize);
        Assert.Equal(480, o.HopLength);
        Assert.Equal(64, o.NumMelBands);
        Assert.Equal(64, o.SpecAugmentTimeDropWidth);
        Assert.Equal(8, o.SpecAugmentFreqDropWidth);
        Assert.Equal(50.0, o.MelMinFrequency);
        Assert.Equal(14000.0, o.MelMaxFrequency);
        Assert.Equal(480000, o.ClipSamples);
        Assert.True(o.EnableFeatureFusion);
        Assert.Equal(256, o.SpecSize);
        Assert.Equal(96, o.AudioEmbedDim);
        Assert.Equal(new[] { 2, 2, 6, 2 }, o.AudioStageDepths);
        Assert.Equal(new[] { 4, 8, 16, 32 }, o.AudioStageHeads);
        Assert.Equal(8, o.SwinWindowSize);
        Assert.Equal(768, o.AudioFeatureDim);
        Assert.Equal(50265, o.VocabSize);
        Assert.Equal(514, o.TextMaxPositions);
        Assert.Equal(768, o.TextHiddenDim);
        Assert.Equal(12, o.TextEncoderLayers);
        Assert.Equal(3072, o.TextIntermediateDim);
        Assert.Equal(1, o.TextPaddingTokenId);
        Assert.Equal(512, o.ProjectionDim);
        o.Validate();
    }

    [Fact]
    public void CopyConstructor_DoesNotAliasTheStageArrays()
    {
        var source = new CLAPModelOptions();
        var copy = new CLAPModelOptions(source);
        source.AudioStageDepths[2] = 99;
        Assert.Equal(6, copy.AudioStageDepths[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task AudioTower_IsHtsatWithHierarchicalStagesAndAnMlpProjection()
    {
        await Task.Yield();
        using var model = ClapReferenceTests.Build(fusion: true);
        var layers = model.Layers;
        var bn0 = Assert.IsType<BatchNormalizationLayer<double>>(layers[0]);
        Assert.Equal(BatchNormDataLayout.ChannelsLast, bn0.Layout);
        Assert.Equal(16, bn0.GetGamma().Length);                                  // one per mel bin
        Assert.IsType<SpectrogramStripeDropLayer<double>>(layers[1]);
        Assert.IsType<HtsatSpectrogramImageLayer<double>>(layers[2]);
        Assert.IsType<HtsatFusionPatchEmbeddingLayer<double>>(layers[3]);

        var swin = layers.OfType<SwinTransformerBlockLayer<double>>().ToList();
        Assert.Equal(6, swin.Count);
        Assert.All(swin, b => Assert.True(b.UsesExactGelu));
        Assert.Equal(2, layers.OfType<SwinPatchMergingLayer<double>>().Count());
        // Stage 0 (16 x 16 grid) alternates W-MSA / SW-MSA; stages whose grid fits one window neither shift nor pad.
        Assert.False(swin[0].UsesShiftedWindows);
        Assert.True(swin[1].UsesShiftedWindows);
        Assert.All(swin.Skip(2), b => Assert.False(b.UsesShiftedWindows));
        Assert.Equal(new[] { 8, 8, 8, 8, 4, 4 }, swin.Select(b => b.WindowSize).ToArray());

        int n = layers.Count;
        var first = Assert.IsType<DenseLayer<double>>(layers[n - 2]);
        var second = Assert.IsType<DenseLayer<double>>(layers[n - 1]);
        Assert.IsType<ReLUActivation<double>>(first.ScalarActivation);
        Assert.IsType<IdentityActivation<double>>(second.ScalarActivation);
        Assert.Equal(new[] { 32, 12 }, first.GetWeights().Shape.ToArray());     // HTS-AT width 8 * 2^2 -> joint
        Assert.IsType<GlobalPoolingLayer<double>>(layers[n - 3]);
    }

    [Fact]
    public void SpecAugmentStripes_ZeroBandsWhileTrainingOnly()
    {
        var layer = new SpectrogramStripeDropLayer<double>(64, 2, 8, 2) { RandomSeed = 3 };
        var input = new Tensor<double>(new[] { 2, 1, 100, 16 });
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, 1.0);
        layer.SetTrainingMode(false);
        var eval = layer.Forward(input);
        Assert.Equal(input.Length, Enumerable.Range(0, eval.Length).Count(i => eval.GetFlat(i) == 1.0));
        layer.SetTrainingMode(true);
        var train = layer.Forward(input);
        int zeros = Enumerable.Range(0, train.Length).Count(i => train.GetFlat(i) == 0.0);
        Assert.InRange(zeros, 1, input.Length - 1);
    }

    [Fact(Timeout = 120000)]
    public async Task NonFusionModel_UsesThePlainSwinPatchEmbedding()
    {
        await Task.Yield();
        using var model = ClapReferenceTests.Build(fusion: false);
        Assert.IsType<SwinPatchEmbeddingLayer<double>>(model.Layers[3]);
        Assert.Empty(model.Layers.OfType<HtsatFusionPatchEmbeddingLayer<double>>());
    }

    [Fact(Timeout = 120000)]
    public async Task MelBand_IsFiftyHertzToFourteenKilohertz()
    {
        await Task.Yield();
        // Content above fmax falls outside every band: a 20 kHz tone moves the mel far less than a 1 kHz one.
        using var model = ClapReferenceTests.Build(fusion: false);
        Tensor<double> Tone(double hz)
        {
            var t = new Tensor<double>(new[] { 1, 8192 });
            for (int i = 0; i < 8192; i++) t[0, i] = 0.5 * Math.Sin(2 * Math.PI * hz * i / 48000.0);
            return t;
        }
        double Energy(Tensor<double> mel)
        {
            double sum = 0;
            for (int i = 0; i < mel.Length; i++) sum += Math.Pow(10, mel.GetFlat(i) / 10);
            return sum;
        }
        double inBand = Energy(model.ComputeLogMel(Tone(1000)));
        double outOfBand = Energy(model.ComputeLogMel(Tone(20000)));
        Assert.True(outOfBand < 1e-2 * inBand, $"20 kHz carried {outOfBand:G4} of mel power against {inBand:G4} at 1 kHz.");
    }

    [Fact(Timeout = 120000)]
    public async Task TextTower_IsRobertaWithItsPoolerAndAnMlpProjection()
    {
        await Task.Yield();
        using var model = ClapReferenceTests.Build(fusion: true);
        var text = model.TextEncoderLayersForTests;
        Assert.Equal(3, text.Count);
        var roberta = Assert.IsType<RobertaEncoderLayer<double>>(text[0]);
        Assert.Equal(1, roberta.Embeddings.PaddingTokenId);
        Assert.IsType<TanhActivation<double>>(roberta.Pooler.ScalarActivation);
        Assert.IsType<ReLUActivation<double>>(Assert.IsType<DenseLayer<double>>(text[1]).ScalarActivation);
        Assert.IsType<IdentityActivation<double>>(Assert.IsType<DenseLayer<double>>(text[2]).ScalarActivation);
    }

    [Fact(Timeout = 120000)]
    public async Task ParameterCount_IsFinalBeforeTheFirstForwardAndAcrossATrainingStep()
    {
        await Task.Yield();
        // Issue #2313's measured defect: the count grew mid-step (28227 -> 28739) because a tower materialized
        // during training. Every layer is now sized at construction.
        using var model = ClapReferenceTests.Build(fusion: true);
        long before = model.ParameterCount;
        var audio = Audio(2, 6000);
        var captions = Captions();
        model.Predict(audio);
        Assert.Equal(before, model.ParameterCount);
        model.Train(audio, captions);
        Assert.Equal(before, model.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task ContrastiveTraining_ReducesTheLoss()
    {
        await Task.Yield();
        using var model = ClapReferenceTests.Build(fusion: true);
        var audio = Audio(2, 6000);
        var captions = Captions();
        model.Train(audio, captions);
        double first = model.GetLastLoss();
        double last = first;
        for (int step = 0; step < 15; step++)
        {
            model.Train(audio, captions);
            last = model.GetLastLoss();
        }
        // double.IsFinite does not exist on net471, so the check is spelled out.
        Assert.False(double.IsNaN(last) || double.IsInfinity(last), $"Contrastive loss is not finite: {last}.");
        Assert.True(last < first, $"Contrastive loss went from {first:G6} to {last:G6} over 15 steps on two pairs.");
    }

    private static Tensor<double> Audio(int batch, int samples)
    {
        var audio = new Tensor<double>(new[] { batch, samples });
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < samples; i++)
                audio[b, i] = 0.4 * Math.Sin(2 * Math.PI * (b == 0 ? 600.0 : 4200.0) * i / 48000.0);
        return audio;
    }

    private static Tensor<double> Captions() =>
        new(new[] { 2, 6 }, new Vector<double>(new double[] { 0, 5, 9, 2, 1, 1, 0, 33, 41, 17, 2, 1 }));

    [Fact(Timeout = 120000)]
    public async Task AttentionalFeatureFusion_DistinctMaps_GradientsReachEveryParameter()
    {
        await Task.Yield();
        var layer = new AttentionalFeatureFusionLayer<double>(8, 4);
        layer.SetTrainingMode(false);
        var rng = new Random(2313);
        Tensor<double> Map()
        {
            var t = new Tensor<double>(new[] { 2, 8, 4, 4 });
            var span = t.AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = rng.NextDouble() * 2 - 1;
            return t;
        }
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        IReadOnlyList<Tensor<double>>? parameters = null;
        double[]? largest = null;

        // Several input pairs: the global branch's ReLU reads only four pooled values per pair, so in evaluation mode a
        // single draw can leave it inactive by chance (that depends on the initialization). The gradient path is what
        // is under test: every parameter must receive a nonzero gradient from some pair.
        for (int draw = 0; draw < 8; draw++)
        {
            var x = Map();
            var y = Map();
            var projection = Map();
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>();
            var fused = layer.Forward(x, y);
            parameters ??= AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new[] { layer }, structureVersion: -1);
            Assert.NotEmpty(parameters);
            largest ??= new double[parameters.Count];
            var loss = engine.ReduceSum(engine.TensorMultiply(fused, projection), new[] { 0, 1, 2, 3 }, keepDims: false);
            var gradients = tape.ComputeGradients(loss, parameters);
            for (int i = 0; i < parameters.Count; i++)
            {
                Assert.True(gradients.TryGetValue(parameters[i], out var gradient) && gradient is not null,
                    $"a {string.Join("x", parameters[i].Shape.ToArray())} parameter received no gradient");
                foreach (var g in gradient!.ToArray()) largest[i] = Math.Max(largest[i], Math.Abs(g));
            }
        }

        // Both attention branches (local and global context) and their batch norms train.
        for (int i = 0; i < parameters!.Count; i++)
            Assert.True(largest![i] > 1e-12,
                $"parameter {i} ({string.Join("x", parameters[i].Shape.ToArray())}) received only zero gradients");
    }
}
