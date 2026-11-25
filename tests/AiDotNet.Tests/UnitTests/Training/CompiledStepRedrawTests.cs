using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// Shared layer code that draws per-step randomness or reads per-step statistics inside a training forward. The fused
/// training path traces the forward once and replays the plan, so a mask built with a host loop, or a statistic read
/// back to the host, is frozen into the plan as the traced step's value: every later step trains on the first step's
/// mask or normalization. Each test trains a fused and an eager copy of the same seeded model side by side on fresh
/// batches and requires the fused loss to follow the eager loss step for step; a frozen mask or statistic drops or
/// scales different elements from step 2 on and the arms diverge. The runs are in double, where the arms agree to the
/// compiled plan's own rounding (well under the 1e-6 relative tolerance) and frozen state moves the loss by percent.
/// </summary>
public class CompiledStepRedrawTests
{
    private const int Steps = 4;
    private const double Tolerance = 1e-6;

    /// <summary>
    /// Swin's stochastic depth drops whole residual branches per sample. With a rate of one half, four samples and two
    /// branches per step, the eager steps draw different masks (asserted), so a plan replaying the traced mask cannot
    /// follow them.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Swin_drop_path_is_redrawn_on_every_compiled_step()
    {
        await Task.Yield();
        const int batch = 4, tokens = 4, dim = 8;
        SwinTransformerBlockLayer<double> Build() => new(dim, numHeads: 2, windowSize: 2, mlpRatio: 2, dropPathRate: 0.5)
        {
            RandomSeed = 17,
        };
        var fused = Build();
        var eager = Build();
        var warm = new Tensor<double>(new[] { batch, tokens, dim });
        fused.Forward(warm);
        eager.Forward(warm);
        eager.SetParameters(fused.GetParameters());
        fused.SetTrainingMode(true);
        eager.SetTrainingMode(true);

        var eagerLosses = TrainSideBySide(new[] { fused }, fused.Forward, new[] { eager }, eager.Forward,
            data => (Random(data, batch, tokens, dim), Random(data, batch, tokens, dim)));
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");

        // The masks themselves: two drop-path forwards of the same layer draw different masks.
        var probe = Build();
        probe.Forward(warm);
        probe.SetParameters(fused.GetParameters());
        probe.SetTrainingMode(true);
        var sample = Random(RandomHelper.CreateSeededRandom(9), batch, tokens, dim);
        Assert.NotEqual(probe.Forward(sample).ToArray(), probe.Forward(sample).ToArray());
    }

    /// <summary>
    /// Wav2Vec2's dropouts and its SpecAugment time mask draw per step. The time mask used to be a host decision: a
    /// bool mask written into a constant. LayerDrop is not part of this plan: a dropped layer does not run and is not
    /// stepped (fairseq, transformers), which one fixed graph cannot express, so a model with LayerDrop &gt; 0 trains on
    /// the eager tape (TrainingForwardSkipsLayersPerStep; see LayerDropSkipsTheOptimizerStepTests).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Wav2Vec2_dropout_and_time_mask_are_redrawn_on_every_compiled_step()
    {
        await Task.Yield();
        const int batch = 2, samples = 400;
        var configuration = new Wav2Vec2Configuration
        {
            ConvDims = new[] { 8, 8, 8 }, ConvKernels = new[] { 10, 3, 2 }, ConvStrides = new[] { 5, 2, 2 },
            HiddenSize = 16, Layers = 2, Heads = 2, IntermediateSize = 32, PositionalConvKernel = 8, PositionalConvGroups = 4,
            FeatureProjectionDropout = 0.2, HiddenDropout = 0.2, AttentionDropout = 0.2, ActivationDropout = 0.2,
            LayerDrop = 0.0, MaskTimeProbability = 0.3, MaskTimeLength = 2, MaskTimeMinMasks = 1,
        };
        (Wav2Vec2Network<double> Network, List<LayerBase<double>> Layers) Build()
        {
            var layers = new List<LayerBase<double>>();
            var network = new Wav2Vec2Network<double>(AiDotNetEngine.Current, layers, configuration);
            network.InitializeLikeReference(RandomHelper.CreateSeededRandom(3));
            return (network, layers);
        }
        var fused = Build();
        var eager = Build();
        var fusedRandom = RandomHelper.CreateSeededRandom(29);
        var eagerRandom = RandomHelper.CreateSeededRandom(29);
        int frames = fused.Network.Frames(samples);

        var eagerLosses = TrainSideBySide(
            fused.Layers, x => fused.Network.Forward(x, training: true, fusedRandom).Hidden,
            eager.Layers, x => eager.Network.Forward(x, training: true, eagerRandom).Hidden,
            data => (Random(data, batch, samples), Random(data, batch, frames, configuration.HiddenSize)));
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");
    }

    /// <summary>
    /// The waveform standardization Wav2Vec2 language identification applies inside its training forward. Its scale,
    /// mean and spread used to be read back to the host, so a plan reused the traced batch's statistics for every
    /// batch. Each step's batch here has a different level and spread, so stale statistics misnormalize it.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Standardize_recomputes_its_statistics_on_every_compiled_step()
    {
        await Task.Yield();
        const int batch = 2, samples = 96, outputs = 3;
        DenseLayer<double> Build() => new(outputs);
        var fused = Build();
        var eager = Build();
        var warm = new Tensor<double>(new[] { batch, samples });
        fused.Forward(warm);
        eager.Forward(warm);
        eager.SetParameters(fused.GetParameters());
        var engine = AiDotNetEngine.Current;
        int step = 0;

        TrainSideBySide(
            new[] { fused }, x => fused.Forward(AudioEngineOps.Standardize(engine, x, 1e-7)),
            new[] { eager }, x => eager.Forward(AudioEngineOps.Standardize(engine, x, 1e-7)),
            data =>
            {
                double level = 0.5 * step, spread = 1.0 + step++;
                var input = Random(data, batch, samples);
                for (int i = 0; i < input.Length; i++) input[i] = level + spread * input[i];
                return (input, Random(data, batch, outputs));
            });
    }

    /// <summary>
    /// MetaVoice's GPT drops its speaker conditioning for a whole step with probability SpeakerDropout and applies
    /// dropout to its embeddings, attention and feed-forward outputs. The speaker drop used to be a host branch that
    /// left the traced step's choice in the graph, and the dropout masks host-filled constants.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task MetaVoice_dropout_and_speaker_drop_are_redrawn_on_every_compiled_step()
    {
        await Task.Yield();
        const int time = 4, dimension = 8, speakerDimension = 3;
        MetaVoiceGpt<double> Build() => new(AiDotNetEngine.Current, new MetaVoiceGptConfiguration
        {
            Dimension = dimension, Layers = 2, Heads = 2, Context = 8, InputVocabularies = new[] { 10 },
            SwiGluMultiple = 4, SpeakerDimension = speakerDimension, SpeakerDropout = 0.5, Dropout = 0.3,
        }, seed: 41);
        var fused = Build();
        var eager = Build();
        var tokens = new[] { new[] { 1, 4, 7, 2 } };

        var eagerLosses = TrainSideBySide(
            fused.Layers, x => fused.Forward(tokens, x, training: true).Hidden,
            eager.Layers, x => eager.Forward(tokens, x, training: true).Hidden,
            data => (Random(data, 1, speakerDimension), Random(data, time, dimension)));
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");
    }

    /// <summary>The inverted dropout T5's encoder and decoder apply between their sublayers.</summary>
    [Fact(Timeout = 120000)]
    public async Task T5_dropout_is_redrawn_on_every_compiled_step()
    {
        await Task.Yield();
        const int batch = 4, features = 6, outputs = 5;
        DenseLayer<double> Build() => new(outputs);
        var fused = Build();
        var eager = Build();
        var warm = new Tensor<double>(new[] { batch, features });
        fused.Forward(warm);
        eager.Forward(warm);
        eager.SetParameters(fused.GetParameters());
        var engine = AiDotNetEngine.Current;
        var fusedRandom = RandomHelper.CreateSeededRandom(13);
        var eagerRandom = RandomHelper.CreateSeededRandom(13);

        var eagerLosses = TrainSideBySide(
            new[] { fused }, x => T5Seq2Seq<double>.Dropout(engine, fused.Forward(x), 0.4, fusedRandom),
            new[] { eager }, x => T5Seq2Seq<double>.Dropout(engine, eager.Forward(x), 0.4, eagerRandom),
            data => (Random(data, batch, features), Random(data, batch, outputs)));
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");
    }

    /// <summary>
    /// LayerDrop's gate selects the layer's output or its input. A dropped layer still runs so that it stays in the
    /// traced graph, and an overflowed output (inf) must not reach the skipped step. As
    /// <c>gate * kept + (1 - gate) * skipped</c>, <c>0 * inf</c> made every element NaN.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Layer_drop_gate_returns_the_input_of_an_overflowed_dropped_layer()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var overflowed = new Tensor<double>(new[] { 2, 3 });
        var input = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < input.Length; i++)
        {
            overflowed[i] = double.PositiveInfinity;
            input[i] = i + 1;
        }
        var dropped = new Tensor<double>(new[] { 1 });
        var kept = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));

        Assert.Equal(input.ToArray(), CompiledStepRandom<double>.Gated(engine, dropped, overflowed, input).ToArray());
        Assert.Equal(input.ToArray(), CompiledStepRandom<double>.Gated(engine, kept, input, overflowed).ToArray());
    }

    /// <summary>
    /// The selection routes the gradient to the selected branch only: a kept layer trains, and a dropped layer's
    /// weights do not move (and do not turn NaN).
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Layer_drop_gate_routes_the_gradient_to_the_selected_branch_only()
    {
        await Task.Yield();
        const int batch = 4, features = 6;
        var engine = AiDotNetEngine.Current;
        double[] Moved(double gateValue)
        {
            var layer = new DenseLayer<double>(features);
            layer.Forward(new Tensor<double>(new[] { batch, features }));
            var before = layer.GetParameters().ToArray();
            var gate = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { gateValue }));
            var data = RandomHelper.CreateSeededRandom(5);
            TapeTrainingStepper<double>.EagerStep(Request(new[] { layer },
                x => CompiledStepRandom<double>.Gated(engine, gate, layer.Forward(x), x),
                Random(data, batch, features), Random(data, batch, features), Sgd()));
            return before.Zip(layer.GetParameters().ToArray(), (b, a) => Math.Abs(a - b)).ToArray();
        }

        Assert.All(Moved(0.0), moved => Assert.Equal(0.0, moved));
        Assert.Contains(Moved(1.0), moved => moved > 0.0);
    }

    /// <summary>
    /// LayerDrop is a function of the seed: the same seed draws the same keep/drop sequence, and two identically
    /// seeded Wav2Vec2 networks trained with LayerDrop reach identical weights, on the eager tape and through the
    /// compiled plan.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Layer_drop_is_reproducible_from_the_seed()
    {
        await Task.Yield();
        double[] Gates(int seed)
        {
            var random = RandomHelper.CreateSeededRandom(seed);
            return Enumerable.Range(0, 32).Select(_ => CompiledStepRandom<double>.KeepGate(random, 0.5)[0]).ToArray();
        }
        var gates = Gates(29);
        Assert.Equal(gates, Gates(29));
        Assert.NotEqual(gates, Gates(30));
        Assert.All(gates, gate => Assert.True(gate == 0.0 || gate == 1.0, $"a keep gate drew {gate:R}."));
        Assert.Contains(0.0, gates);
        Assert.Contains(1.0, gates);

        const int batch = 2, samples = 400;
        var configuration = new Wav2Vec2Configuration
        {
            ConvDims = new[] { 8, 8, 8 }, ConvKernels = new[] { 10, 3, 2 }, ConvStrides = new[] { 5, 2, 2 },
            HiddenSize = 16, Layers = 3, Heads = 2, IntermediateSize = 32, PositionalConvKernel = 8, PositionalConvGroups = 4,
            LayerDrop = 0.5,
        };
        double[] Trained(bool fused)
        {
            var layers = new List<LayerBase<double>>();
            var network = new Wav2Vec2Network<double>(AiDotNetEngine.Current, layers, configuration);
            network.InitializeLikeReference(RandomHelper.CreateSeededRandom(3));
            var random = RandomHelper.CreateSeededRandom(29);
            var data = RandomHelper.CreateSeededRandom(5);
            var stepper = new TapeTrainingStepper<double>(new object());
            var optimizer = Sgd();
            int frames = network.Frames(samples);
            for (int step = 0; step < Steps; step++)
            {
                var request = Request(layers, x => network.Forward(x, training: true, random).Hidden,
                    Random(data, batch, samples), Random(data, batch, frames, configuration.HiddenSize), optimizer);
                if (fused)
                {
                    stepper.Step(request);
                    Assert.True(stepper.LastStepFused, $"step {step}: fused path did not engage ({stepper.Session.LastMissReason}).");
                }
                else
                {
                    TapeTrainingStepper<double>.EagerStep(request);
                }
            }
            return layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
        }

        var eager = Trained(fused: false);
        Assert.Equal(eager, Trained(fused: false));
        Assert.Equal(Trained(fused: true), Trained(fused: true));
        Assert.DoesNotContain(eager, double.IsNaN);
    }

    /// <summary>
    /// The gate selects in single and half precision too: the layer's output when kept, the input when dropped, an
    /// overflowed dropped layer included, for gates set by hand and for gates drawn from a seeded generator.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Layer_drop_gate_selects_in_float_and_half()
    {
        await Task.Yield();
        AssertGateSelects(value => (float)value);
#if NET5_0_OR_GREATER
        AssertGateSelects(value => (Half)value);
#endif
    }

    private static void AssertGateSelects<T>(Func<double, T> of)
    {
        var engine = AiDotNetEngine.Current;
        var layer = new Tensor<T>(new[] { 2, 3 });
        var overflowed = new Tensor<T>(new[] { 2, 3 });
        var input = new Tensor<T>(new[] { 2, 3 });
        for (int i = 0; i < input.Length; i++)
        {
            layer[i] = of(10 + i);
            overflowed[i] = of(double.PositiveInfinity);
            input[i] = of(i + 1);
        }
        var dropped = new Tensor<T>(new[] { 1 });
        var kept = new Tensor<T>(new[] { 1 });
        kept[0] = of(1.0);

        Assert.Equal(layer.ToArray(), CompiledStepRandom<T>.Gated(engine, kept, layer, input).ToArray());
        Assert.Equal(input.ToArray(), CompiledStepRandom<T>.Gated(engine, dropped, layer, input).ToArray());
        Assert.Equal(input.ToArray(), CompiledStepRandom<T>.Gated(engine, dropped, overflowed, input).ToArray());
        Assert.Equal(input.ToArray(), CompiledStepRandom<T>.Gated(engine, kept, input, overflowed).ToArray());

        var random = RandomHelper.CreateSeededRandom(29);
        var one = of(1.0);
        int keptCount = 0;
        const int draws = 32;
        for (int draw = 0; draw < draws; draw++)
        {
            var gate = CompiledStepRandom<T>.KeepGate(random, 0.5);
            bool keep = EqualityComparer<T>.Default.Equals(gate[0], one);
            if (keep) keptCount++;
            Assert.Equal((keep ? layer : input).ToArray(), CompiledStepRandom<T>.Gated(engine, gate, layer, input).ToArray());
        }
        Assert.InRange(keptCount, 1, draws - 1);
    }

    /// <summary>
    /// Wav2Vec2 language identification with LayerDrop off: the waveform standardization, the dropouts and the
    /// SpecAugment time mask all run inside the training forward. The 300-sample clips are shorter than the feature
    /// encoder's 400-sample receptive field and are zero-padded to it, which used to be a host copy of the traced
    /// batch (#2592).
    /// </summary>
    [Theory(Timeout = 300000)]
    [InlineData(8000)]
    [InlineData(300)]
    public async Task Wav2Vec2_language_identification_follows_the_eager_tape_on_every_compiled_step(int samples)
    {
        await Task.Yield();
        var data = RandomHelper.CreateSeededRandom(5);
        var eagerLosses = TrainModelsSideBySide(
            LanguageIdentifierWithoutLayerDrop,
            model => model.FusedSession,
            step => (Random(data, 2, samples), LanguageTargets(step)));
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");
    }

    /// <summary>
    /// Two identically seeded language identifiers trained on the compiled plan end at the same weights: the plan
    /// redraws the dropouts and the time mask from the model's seeded streams, in the order the eager tape draws them
    /// (#2592; the model family's determinism check runs the default options, whose LayerDrop of 0.1 keeps the model on
    /// the eager tape).
    /// </summary>
    [Theory(Timeout = 300000)]
    [InlineData(8000)]
    [InlineData(300)]
    public async Task Wav2Vec2_language_identification_is_reproducible_on_the_compiled_plan(int samples)
    {
        await Task.Yield();
        double[] Trained()
        {
            var data = RandomHelper.CreateSeededRandom(5);
            using var model = LanguageIdentifierWithoutLayerDrop();
            for (int step = 0; step < Steps; step++)
            {
                model.Train(Random(data, 2, samples), LanguageTargets(step));
                Assert.True(model.FusedSession.IsCommitted && !model.FusedSession.LastStepRanOutsidePlan,
                    $"step {step}: fused path did not engage ({model.FusedSession.LastMissReason ?? model.LastFusedGateMissReason}).");
            }
            return model.GetParameters().ToArray();
        }

        var first = Trained();
        using var untrained = LanguageIdentifierWithoutLayerDrop();
        Assert.NotEqual(untrained.GetParameters().ToArray(), first);
        Assert.Equal(first, Trained());
    }

    private static AiDotNet.Audio.LanguageIdentification.Wav2Vec2LanguageIdentifier<double> LanguageIdentifierWithoutLayerDrop()
        => new(
            new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.TwoDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.MultiClassClassification,
                inputHeight: 64, inputWidth: 32, inputDepth: 1, outputSize: 4),
            new[] { "en", "es", "fr", "de" },
            new AiDotNet.Audio.LanguageIdentification.Wav2Vec2LidOptions
            {
                HiddenSize = 16, NumLayers = 2, NumAttentionHeads = 2, IntermediateSize = 32, FeatureEncoderDim = 16,
                HiddenDropout = 0.2, FeatureProjectionDropout = 0.2, LayerDrop = 0.0,
            });

    private static Tensor<double> LanguageTargets(int step)
    {
        var target = new Tensor<double>(new[] { 2, 4 });
        target[0, step % 4] = 1.0;
        target[1, (step + 2) % 4] = 1.0;
        return target;
    }

    /// <summary>
    /// CLAP's contrastive step (#2592). Everything its training forward does per step has to be recomputed or redrawn
    /// by the compiled plan: the log-mel front end, the random crop of a clip longer than the clip length (12,000
    /// samples against 8,192), the three feature-fusion crops of such a clip (20,000 samples), the fusion branch's
    /// random row when no clip is longer (6,000 samples with fusion), SpecAugment, drop-path, the text dropout, and
    /// the text tower's position ids and padding mask, which follow each batch's captions.
    /// The first compiled step is held to the eager step like every other: HTS-AT's patch merging used to gather
    /// through indices a trace had not computed yet, so the traced step alone merged token 0 into every patch.
    /// </summary>
    [Theory(Timeout = 300000)]
    [InlineData(false, 6000)]
    [InlineData(false, 12000)]
    [InlineData(true, 6000)]
    [InlineData(true, 20000)]
    public async Task Clap_follows_the_eager_tape_on_every_compiled_step(bool featureFusion, int samples)
    {
        await Task.Yield();
        var eagerLosses = TrainModelsSideBySide(
            () => AiDotNet.Tests.Audio.ClapReferenceTests.Build(featureFusion),
            model => TapeTrainingStepper<double>.ForOwner(model).Session,
            step =>
            {
                var audio = new Tensor<double>(new[] { 2, samples });
                var span = audio.AsWritableSpan();
                for (int b = 0; b < 2; b++)
                {
                    double frequency = b == 0 ? 600.0 + 150.0 * step : 4200.0 - 300.0 * step;
                    for (int i = 0; i < samples; i++)
                        span[b * samples + i] = (0.4 + 0.1 * step) * Math.Sin(2 * Math.PI * frequency * i / 48000.0);
                }
                // Captions of a different length each step (1 is the padding id): RoBERTa's position ids and its
                // attention mask follow the padding, so a plan that kept the traced batch's would embed and attend
                // the later captions as if they were padded like the first.
                var captions = new Tensor<double>(new[] { 2, 6 }, new Vector<double>(step % 2 == 0
                    ? new double[] { 0, 5 + step, 9, 2, 1, 1, 0, 33, 41 - step, 17, 2, 1 }
                    : new double[] { 0, 5 + step, 2, 1, 1, 1, 0, 33, 41 - step, 17, 23, 2 }));
                return (audio, captions);
            });
        Assert.True(eagerLosses.Distinct().Count() == eagerLosses.Count, "the eager steps should differ.");
    }

    /// <summary>
    /// Trains two identically seeded models through their own <c>Train</c>, one on the compiled plan and one with its
    /// session disabled, on the same fresh batch each step. Returns the eager losses, after requiring the compiled arm
    /// to engage and to follow them step for step.
    /// </summary>
    private static List<double> TrainModelsSideBySide<TModel>(Func<TModel> build,
        Func<TModel, FusedTrainingSession<double>> sessionOf,
        Func<int, (Tensor<double> Input, Tensor<double> Target)> batchOf)
        where TModel : AiDotNet.NeuralNetworks.NeuralNetworkBase<double>
    {
        using var fused = build();
        using var eager = build();
        var fusedSession = sessionOf(fused);
        var eagerSession = sessionOf(eager);
        eagerSession.IsDisabled = true;
        var eagerLosses = new List<double>();
        for (int step = 0; step < Steps; step++)
        {
            var (input, target) = batchOf(step);
            fused.Train(input, target);
            Assert.True(fusedSession.IsCommitted && !fusedSession.LastStepRanOutsidePlan,
                $"step {step}: fused path did not engage ({fusedSession.LastMissReason ?? fused.LastFusedGateMissReason}).");
            eager.Train(input, target);
            Assert.False(eagerSession.IsCommitted, $"step {step}: the eager arm ran on a compiled plan.");
            double fusedLoss = fused.GetLastLoss(), eagerLoss = eager.GetLastLoss();
            Assert.True(
                Math.Abs(fusedLoss - eagerLoss) <= Tolerance * Math.Max(1.0, Math.Abs(eagerLoss)),
                $"step {step}: fused loss {fusedLoss:R} differs from eager loss {eagerLoss:R}; the plan replays stale per-step state.");
            eagerLosses.Add(eagerLoss);
        }
        return eagerLosses;
    }

    /// <summary>The eager losses, after requiring the fused arm to engage and to follow them step for step.</summary>
    private static List<double> TrainSideBySide(
        IReadOnlyList<ITrainableLayer<double>> fusedLayers, Func<Tensor<double>, Tensor<double>> fusedForward,
        IReadOnlyList<ITrainableLayer<double>> eagerLayers, Func<Tensor<double>, Tensor<double>> eagerForward,
        Func<System.Random, (Tensor<double> Input, Tensor<double> Target)> batchOf)
    {
        var stepper = new TapeTrainingStepper<double>(new object());
        var fusedOptimizer = Sgd();
        var eagerOptimizer = Sgd();
        var data = RandomHelper.CreateSeededRandom(5);
        var eagerLosses = new List<double>();
        for (int step = 0; step < Steps; step++)
        {
            var (input, target) = batchOf(data);
            double fused = stepper.Step(Request(fusedLayers, fusedForward, input, target, fusedOptimizer));
            Assert.True(stepper.LastStepFused, $"step {step}: fused path did not engage ({stepper.Session.LastMissReason}).");
            double eager = TapeTrainingStepper<double>.EagerStep(Request(eagerLayers, eagerForward, input, target, eagerOptimizer));
            Assert.True(
                Math.Abs(fused - eager) <= Tolerance * Math.Max(1.0, Math.Abs(eager)),
                $"step {step}: fused loss {fused:R} differs from eager loss {eager:R}; the plan replays stale per-step state.");
            eagerLosses.Add(eager);
        }
        return eagerLosses;
    }

    private static FusedTrainingStepRequest<double> Request(IReadOnlyList<ITrainableLayer<double>> layers,
        Func<Tensor<double>, Tensor<double>> forward, Tensor<double> input, Tensor<double> target,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer) => new()
    {
        Layers = layers,
        Input = input,
        Target = target,
        Forward = forward,
        ComputeLoss = (predicted, expected) =>
        {
            var engine = AiDotNetEngine.Current;
            var difference = engine.TensorSubtract(predicted, expected);
            return engine.TensorMultiplyScalar(
                engine.ReduceSum(engine.TensorMultiply(difference, difference), null), 1.0 / difference.Length);
        },
        Optimizer = optimizer,
    };

    private static StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>> Sgd()
        => new(model: null, options: new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.05,
            UseAdaptiveLearningRate = false,
        });

    private static Tensor<double> Random(System.Random random, params int[] shape)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = random.NextDouble() * 2.0 - 1.0;
        return tensor;
    }
}
