using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// EnCodec encodes, quantizes, decodes and trains as its paper specifies (Défossez et al. 2022), and loads the released
/// checkpoints' layout exactly.
/// </summary>
/// <remarks>
/// Before this change EnCodec was a stack of generic layers whose "residual vector quantization" bucketed each latent
/// value into a code by its magnitude; it had no codebooks, no discriminator and trained by regression.
/// <c>ReferenceData/encodec_official_layout_reference.json</c> holds two tiny Hugging Face EncodecModel instances with
/// random weights (generator: tools/reference-data/encodec_reference.py, transformers EncodecModel), saved under
/// the released checkpoints' tensor names, with the latent, codes and decoded audio of a fixed input.
/// </remarks>
public class EnCodecPaperTests
{
    [Fact]
    public void AudioCraftTrainingCheckpoint_SelectsBestModelAndRespectsExplicitPrefix()
    {
        const string first = "encoder.model.0.conv.conv.weight_v";
        var state = new Dictionary<string, Tensor<float>>
        {
            ["model." + first] = new Tensor<float>(new[] { 1 }),
            ["best_state.model." + first] = new Tensor<float>(new[] { 1 }),
        };
        Assert.Equal("best_state.model.", EnCodec<double>.AuthorCheckpointPrefix(state, ""));
        Assert.Equal("model.", EnCodec<double>.AuthorCheckpointPrefix(state, "model."));
        state.Remove("best_state.model." + first);
        Assert.Equal("model.", EnCodec<double>.AuthorCheckpointPrefix(state, ""));
        state[first] = new Tensor<float>(new[] { 1 });
        Assert.Equal("", EnCodec<double>.AuthorCheckpointPrefix(state, ""));
    }

    private static JObject Reference()
    {
        const string fileName = "encodec_official_layout_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "Audio", "Codecs", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Audio", "Codecs", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static NeuralNetworkArchitecture<double> Architecture()
        => new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 11 };

    private static EnCodecOptions TinyOptions(int channels = 1, bool causal = true) => new()
    {
        SampleRate = 24000,
        Channels = channels,
        Filters = 4,
        Dimension = 8,
        Ratios = [4, 2],
        ResidualKernelSizes = [3, 1],
        NumQuantizers = 2,
        CodebookSize = 16,
        TargetBandwidths = [12.0, 24.0],
        TargetBandwidthKbps = 24.0,
        Causal = causal,
        SegmentSize = 96,
        KMeansIterations = 5,
        MelScales = [5, 6],
        MelBins = 8,
        DiscriminatorWindows = [32, 16],
        DiscriminatorFilters = 4,
        SamplingSeed = 3,
    };

    private static Tensor<double> TensorOf(JToken values, int[] shape)
        => new(shape, new Vector<double>(values.Select(v => (double)v).ToArray()));

    private static JToken RequireToken(JToken? parent, string property)
    {
        Assert.NotNull(parent);
        var token = parent[property];
        Assert.NotNull(token);
        return token;
    }

    [Theory(Timeout = 300000)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task OfficialLayout_LoadsAndReproducesTheReferenceModel(int index)
    {
        await Task.Yield();
        var fixtures = RequireToken(Reference(), "fixtures");
        var fixture = fixtures[index];
        Assert.NotNull(fixture);
        var config = RequireToken(fixture, "config");
        int channels = (int)RequireToken(config, "audio_channels");
        bool causal = (bool)RequireToken(config, "use_causal_conv");
        using var model = new EnCodec<double>(Architecture(), TinyOptions(channels, causal));
        string path = Path.Combine(Path.GetTempPath(), $"encodec_reference_{index}_{Guid.NewGuid():N}.safetensors");
        string? b64 = (string?)RequireToken(fixture, "safetensors_base64");
        Assert.NotNull(b64);
        File.WriteAllBytes(path, Convert.FromBase64String(b64));
        try
        {
            model.LoadPretrainedWeights(path);
        }
        finally
        {
            File.Delete(path);
        }

        var audio = TensorOf(RequireToken(fixture, "audio"), RequireToken(fixture, "audio_shape").Select(v => (int)v).ToArray());
        var latent = model.EncodeEmbeddings(audio);
        var expectedLatent = RequireToken(fixture, "latent").Select(v => (double)v).ToArray();
        Assert.Equal(expectedLatent.Length, latent.Length);
        for (int i = 0; i < latent.Length; i++)
            Assert.True(Math.Abs(latent[i] - expectedLatent[i]) < 1e-4 * (1 + Math.Abs(expectedLatent[i])),
                $"latent[{i}] = {latent[i]}, reference {expectedLatent[i]}");

        var codes = model.Encode(audio);
        var expectedCodes = RequireToken(fixture, "codes").Select(row => row.Select(v => (int)v).ToArray()).ToArray();
        Assert.Equal(expectedCodes.Length, codes.GetLength(0));
        for (int q = 0; q < expectedCodes.Length; q++)
            for (int t = 0; t < expectedCodes[q].Length; t++)
                Assert.Equal(expectedCodes[q][t], codes[q, t]);

        var decoded = model.Decode(codes);
        var expectedDecoded = RequireToken(fixture, "decoded").Select(v => (double)v).ToArray();
        Assert.Equal(expectedDecoded.Length, decoded.Length);
        for (int i = 0; i < decoded.Length; i++)
            Assert.True(Math.Abs(decoded[i] - expectedDecoded[i]) < 1e-4 * (1 + Math.Abs(expectedDecoded[i])),
                $"decoded[{i}] = {decoded[i]}, reference {expectedDecoded[i]}");
    }

    [Fact(Timeout = 300000)]
    public async Task Bandwidths_SelectTheCodebooksTheyPayFor()
    {
        await Task.Yield();
        // 24 kHz, 320x: 75 frames per second of 10-bit codes, 0.75 kbps per codebook (Sec. 3.2).
        using var model = new EnCodec<double>(Architecture(), new EnCodecOptions { DiscriminatorWindows = [128], DiscriminatorFilters = 4 });
        Assert.Equal(320, model.HopLength);
        Assert.Equal(75, model.TokenFrameRate);
        Assert.Equal(new[] { 2, 4, 8, 16, 32 }, new[] { 1.5, 3.0, 6.0, 12.0, 24.0 }.Select(model.QuantizersForBandwidth).ToArray());
        Assert.Equal(6000.0, model.GetBitrate(8), 6);
    }

    [Fact(Timeout = 60000)]
    public async Task Balancer_ReproducesTheReferenceBalancerTest()
    {
        await Task.Yield();
        // encodec/balancer.py test(): x = 0, l1 = |x - 1|, l2 = 100 |x + 1|, equal weights. Rescaled, the gradients cancel.
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var balancer = new LossBalancer<double>(engine, new Dictionary<string, double> { ["1"] = 1, ["2"] = 1 });
        var x = new Tensor<double>(new[] { 1 });
        var losses = new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
        {
            ["1"] = y => engine.TensorAbs(engine.TensorAddScalar(y, -1.0)),
            ["2"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(engine.TensorAddScalar(y, 1.0)), 100.0),
        };
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var surrogate = balancer.Surrogate(x, losses);
            gradients = tape.ComputeGradients(surrogate, new[] { x });
        }
        Assert.Equal(0.0, gradients[x][0], 12);
        Assert.Equal(1.0, balancer.AveragedNorms["1"], 12);
        Assert.Equal(100.0, balancer.AveragedNorms["2"], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task TracedBalancer_MatchesTheReferenceAndReportsTheWeightedLoss()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var balancer = new LossBalancer<double>(engine, new Dictionary<string, double> { ["1"] = 1, ["2"] = 1 });
        var x = new Tensor<double>(new[] { 1 });
        var shift = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        double weighted;
        using (var tape = new GradientTape<double>())
        {
            var (surrogate, loss) = balancer.TracedSurrogate(x, new[] { shift }, context =>
                new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
                {
                    ["1"] = y => engine.TensorAbs(engine.TensorSubtract(y, context[0])),
                    ["2"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(engine.TensorAdd(y, context[0])), 100.0),
                });
            gradients = tape.ComputeGradients(surrogate, new[] { x });
            weighted = loss.GetFlat(0);
        }
        Assert.Equal(0.0, gradients[x][0], 12);
        Assert.Equal(101.0, weighted, 12);
        Assert.Equal(1, balancer.Steps);
    }

    [Fact(Timeout = 600000)]
    public async Task BalancedTraining_UpdatesTheNormAveragesOncePerGeneratorStep()
    {
        await Task.Yield();
        using var model = new EnCodec<double>(Architecture(), TinyOptions());
        var audio = new Tensor<double>(new[] { 96 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * 1500 * i / 24000.0);
        const int steps = 6;
        for (int i = 0; i < steps; i++) model.Train(audio, audio);
        // Every balanced generator step ran a compiled plan, none the eager tape.
        Assert.True(model.CustomFusedStepCount >= steps, $"fused steps {model.CustomFusedStepCount}");
        Assert.Equal(0, model.CustomEagerStepCount);
        Assert.Equal(0, model.CustomOutsidePlanStepCount);
        var balancer = model.Balancer;
        Assert.NotNull(balancer);
        Assert.Equal(steps, balancer.Steps);
        Assert.All(balancer.AveragedNorms.Values, norm => Assert.True(norm > 0 && !double.IsInfinity(norm), $"norm {norm}"));
    }

    [Fact(Timeout = 300000)]
    public async Task Streamable_OutputDoesNotDependOnFutureInput()
    {
        await Task.Yield();
        // Sec. 3.1: all padding before the first time step; transposed convolutions keep their first s outputs.
        using var model = new EnCodec<double>(Architecture(), TinyOptions());
        var random = new Random(5);
        var audio = new Tensor<double>(new[] { 1, 1, 96 });
        for (int i = 0; i < audio.Length; i++) audio[i] = random.NextDouble() - 0.5;
        var changed = new Tensor<double>(audio._shape, audio.ToVector());
        for (int i = 80; i < 96; i++) changed[0, 0, i] += 0.7;
        var a = model.DecodeEmbeddings(model.EncodeEmbeddings(audio));
        var b = model.DecodeEmbeddings(model.EncodeEmbeddings(changed));
        // Frame f sees samples up to (f + 1) * hop - 1; samples before frame 10's start (80) are fixed.
        for (int i = 0; i < 80; i++) Assert.Equal(a[i], b[i], 12);
        Assert.True(Enumerable.Range(80, 16).Any(i => Math.Abs(a[i] - b[i]) > 1e-9), "Changing the input's end did not change the output's end.");
    }

    [Fact(Timeout = 60000)]
    public async Task MultiScaleStftDiscriminator_HasThePapersSubNetworks()
    {
        await Task.Yield();
        // Fig. 2: per scale a 3x9 conv, three 3x9 convs with stride (1, 2) and time dilations 1, 2, 4, a 3x3 conv - five
        // feature maps of 32 channels - and a 3x3 conv to one logit channel.
        var layers = new List<LayerBase<double>>();
        var d = new MultiScaleStftDiscriminator<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, [256, 128], 32, [1, 2, 4], 3, 9, 0.2, layers);
        var audio = new Tensor<double>(new[] { 1024 });
        for (int i = 0; i < audio.Length; i++) audio[i] = Math.Sin(i * 0.3);
        var outputs = d.Forward(audio);
        Assert.Equal(2, outputs.Count);
        foreach (var (logits, features) in outputs)
        {
            Assert.Equal(5, features.Count);
            Assert.All(features, f => Assert.Equal(32, f.Shape[1]));
            Assert.Equal(4, logits.Rank);
            Assert.Equal(1, logits.Shape[1]);
        }
        // Window 256, hop 64: 1 + (1024 - 256) / 64 = 13 frames, 129 bins -> frequency 129 -> 65 -> 33 -> 17 after the strides.
        Assert.Equal(13, outputs[0].Features[4].Shape[2]);
        Assert.Equal(17, outputs[0].Features[4].Shape[3]);
        Assert.All(layers.OfType<Conv2DLayer<double>>(), l => Assert.Equal("Weight", l.GetMetadata()["Normalization"]));
    }

    [Fact(Timeout = 60000)]
    public async Task Commitment_IsEq3sSumOfSquaredDistances()
    {
        await Task.Yield();
        // Eq. 3: l_w = sum over the codebooks used of ||z_c - q_c(z_c)||^2; the reference code's mean over codebooks of the
        // MSE is the option. With zero codebooks (no k-means, no dead-code replacement) every code is 0, so one codebook's
        // commitment is the input's squared norm.
        ResidualVectorQuantizerLayer<double> Quantizer(bool asMean)
        {
            var layer = new ResidualVectorQuantizerLayer<double>(2, 2, 2, kmeansIterations: 0, deadCodeThreshold: 0, commitmentAsMean: asMean);
            for (int q = 0; q < 2; q++) layer.Codebook(q).Data.Span.Clear();
            layer.SetTrainingMode(true);
            layer.ActiveQuantizers = 1;
            return layer;
        }
        var sum = Quantizer(asMean: false);
        var mean = Quantizer(asMean: true);
        var x = new Tensor<double>(new[] { 1, 2, 3 }, new Vector<double>(new[] { 1.0, 2, 3, 4, 5, 6 }));
        sum.Forward(x);
        mean.Forward(x);
        var sumLoss = sum.CommitmentLoss;
        var meanLoss = mean.CommitmentLoss;
        Assert.NotNull(sumLoss);
        Assert.NotNull(meanLoss);
        Assert.Equal(91.0, sumLoss[0], 9);
        Assert.Equal(91.0 / 6.0, meanLoss[0], 9);
    }

    [Fact(Timeout = 60000)]
    public async Task TracedBalancer_IsolatesTwoFactoriesAndSignsStateAndReplaysHostState()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var balancer = new LossBalancer<double>(engine, new Dictionary<string, double> { ["a"] = 1.0, ["b"] = 1.0 });
        var parameter = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var target = new Tensor<double>(new[] { 0.5 }, new[] { 1 });
        int factoryACalls = 0, factoryBCalls = 0;

        using (var probeA = AiDotNet.Tensors.Engines.Compilation.GraphMode.EnableProbe())
        {
            var outNode = engine.TensorMultiplyScalar(parameter, 1.0);
            var (surrogateA, _) = balancer.TracedSurrogate(outNode, new[] { target }, ctx =>
            {
                factoryACalls++;
                return new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
                {
                    ["a"] = y => engine.TensorAbs(engine.TensorSubtract(y, ctx[0])),
                    ["b"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(y), 2.0),
                };
            }, stateKey: "slot-A");

            var nodeA = Assert.Single(probeA.Nodes.Where(n => n.OpName == "EnCodecLossBalancer")
                .Cast<AiDotNet.Tensors.Engines.Compilation.ISignableLazyNode>());
            Assert.True(nodeA.ReplaysHostStateFlag);
            Assert.NotNull(nodeA.SavedStateObjects);
            Assert.NotNull(surrogateA);

            using var probeB = AiDotNet.Tensors.Engines.Compilation.GraphMode.EnableProbe();
            var outNodeB = engine.TensorMultiplyScalar(parameter, 1.0);
            var (surrogateB, _) = balancer.TracedSurrogate(outNodeB, new[] { target }, ctx =>
            {
                factoryBCalls++;
                return new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
                {
                    ["a"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(engine.TensorSubtract(y, ctx[0])), 5.0),
                    ["b"] = y => engine.TensorAbs(y),
                };
            }, stateKey: "slot-B");

            var nodeB = Assert.Single(probeB.Nodes.Where(n => n.OpName == "EnCodecLossBalancer")
                .Cast<AiDotNet.Tensors.Engines.Compilation.ISignableLazyNode>());
            Assert.True(nodeB.ReplaysHostStateFlag);
            Assert.NotNull(nodeB.SavedStateObjects);
            Assert.NotNull(surrogateB);
            Assert.NotEqual(nodeA.SavedStateObjects, nodeB.SavedStateObjects);
        }

        // Probe must never invoke either factory or advance balancer Steps/EMA.
        Assert.Equal(0, factoryACalls);
        Assert.Equal(0, factoryBCalls);
        Assert.Equal(0, balancer.Steps);

        // Two TracedSurrogate calls on the SAME balancer within one compiled plan must bind distinct slots and never clobber.
        var optimizer = new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AiDotNet.Models.Options.AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                EnableGradientClipping = false,
            });
        var stepper = AiDotNet.Training.TapeTrainingStepper<double>.ForOwner(optimizer);
        try
        {
            Tensor<double> DualObjective()
            {
                var pred = engine.TensorMultiplyScalar(parameter, 1.0);
                var (sA, _) = balancer.TracedSurrogate(pred, new[] { target }, ctx =>
                {
                    factoryACalls++;
                    return new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
                    {
                        ["a"] = y => engine.TensorAbs(engine.TensorSubtract(y, ctx[0])),
                        ["b"] = y => engine.TensorAbs(y),
                    };
                }, stateKey: "first");
                var (sB, _) = balancer.TracedSurrogate(pred, new[] { target }, ctx =>
                {
                    factoryBCalls++;
                    return new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
                    {
                        ["a"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(engine.TensorAdd(y, ctx[0])), 3.0),
                        ["b"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(y), 4.0),
                    };
                }, stateKey: "second");
                return engine.TensorAdd(sA, sB);
            }

            for (int step = 0; step < 3; step++)
            {
                stepper.Step(new AiDotNet.Training.FusedTrainingStepRequest<double>
                {
                    Layers = Array.Empty<AiDotNet.Interfaces.ITrainableLayer<double>>(),
                    Input = target,
                    Target = target,
                    Forward = static x => x,
                    ComputeLoss = (_, _) => DualObjective(),
                    ClosureObjective = DualObjective,
                    Optimizer = optimizer,
                    ExtraParameters = new[] { parameter },
                });
            }

            Assert.Equal(3, factoryACalls);
            Assert.Equal(3, factoryBCalls);
            Assert.Equal(6, balancer.Steps);
        }
        finally
        {
            stepper.Session.Reset(stickyDisable: false);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task TracedBalancer_ProbeDoesNotMutateEmaOrOverwriteCompiledSlot()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var balancer = new LossBalancer<double>(engine, new Dictionary<string, double> { ["1"] = 1.0 });
        var parameter = new Tensor<double>(new[] { 3.0 }, new[] { 1 });
        var target = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        int activeCalls = 0, probeCalls = 0;

        var optimizer = new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AiDotNet.Models.Options.AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                EnableGradientClipping = false,
            });
        var stepper = AiDotNet.Training.TapeTrainingStepper<double>.ForOwner(optimizer);
        try
        {
            Tensor<double> ActiveObjective()
            {
                var pred = engine.TensorMultiplyScalar(parameter, 1.0);
                var (surrogate, _) = balancer.TracedSurrogate(pred, new[] { target }, ctx =>
                {
                    activeCalls++;
                    return new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
                    {
                        ["1"] = y => engine.TensorAbs(engine.TensorSubtract(y, ctx[0])),
                    };
                });
                return surrogate;
            }

            AiDotNet.Training.FusedTrainingStepRequest<double> Request() => new()
            {
                Layers = Array.Empty<AiDotNet.Interfaces.ITrainableLayer<double>>(),
                Input = target,
                Target = target,
                Forward = static x => x,
                ComputeLoss = (_, _) => ActiveObjective(),
                ClosureObjective = ActiveObjective,
                Optimizer = optimizer,
                ExtraParameters = new[] { parameter },
            };

            stepper.Step(Request());
            Assert.Equal(1, balancer.Steps);
            Assert.Equal(1, activeCalls);
            double normAfterFirst = balancer.AveragedNorms["1"];

            // Unaccepted probe with a completely different factory must leave Steps, EMA, and ActiveObjective's slot untouched.
            using (AiDotNet.Tensors.Engines.Compilation.GraphMode.EnableProbe())
            {
                var pred = engine.TensorMultiplyScalar(parameter, 1.0);
                _ = balancer.TracedSurrogate(pred, new[] { target }, _ =>
                {
                    probeCalls++;
                    return new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
                    {
                        ["1"] = y => engine.TensorMultiplyScalar(y, 999.0),
                    };
                }, stateKey: "discarded-probe");
            }

            Assert.Equal(0, probeCalls);
            Assert.Equal(1, balancer.Steps);
            Assert.Equal(normAfterFirst, balancer.AveragedNorms["1"], 12);

            stepper.Step(Request());
            Assert.Equal(0, probeCalls);
            Assert.Equal(2, activeCalls);
            Assert.Equal(2, balancer.Steps);
        }
        finally
        {
            stepper.Session.Reset(stickyDisable: false);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Balancer_ThrowOrNonFiniteLossLeavesEmaAndStepsUntouched()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var balancer = new LossBalancer<double>(engine, new Dictionary<string, double> { ["1"] = 1.0, ["2"] = 1.0 });
        var x = new Tensor<double>(new[] { 2.0 }, new[] { 1 });

        using (var tape = new GradientTape<double>())
        {
            var s = balancer.Surrogate(x, new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
            {
                ["1"] = y => engine.TensorAbs(y),
                ["2"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(y), 4.0),
            });
            _ = tape.ComputeGradients(s, new[] { x });
        }

        Assert.Equal(1, balancer.Steps);
        double norm1 = balancer.AveragedNorms["1"];
        double norm2 = balancer.AveragedNorms["2"];
        double lastWeighted = balancer.LastWeightedLoss;

        // Inner loss throw must not partially mutate EMA or increment Steps.
        Assert.Throws<InvalidOperationException>(() => balancer.Surrogate(x, new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
        {
            ["1"] = y => engine.TensorMultiplyScalar(engine.TensorAbs(y), 50.0),
            ["2"] = _ => throw new InvalidOperationException("inner loss failure"),
        }));
        Assert.Equal(1, balancer.Steps);
        Assert.Equal(norm1, balancer.AveragedNorms["1"], 12);
        Assert.Equal(norm2, balancer.AveragedNorms["2"], 12);
        Assert.Equal(lastWeighted, balancer.LastWeightedLoss, 12);

        // Non-finite loss/gradient step must be skipped cleanly without poisoning EMA or incrementing Steps.
        var zero = new Tensor<double>(new[] { 0.0 }, new[] { 1 });
        _ = balancer.Surrogate(zero, new Dictionary<string, Func<Tensor<double>, Tensor<double>>>
        {
            ["1"] = y => engine.TensorSqrt(y), // gradient at 0 is +Infinity
            ["2"] = y => engine.TensorAbs(y),
        });
        Assert.Equal(1, balancer.Steps);
        Assert.Equal(norm1, balancer.AveragedNorms["1"], 12);
        Assert.Equal(norm2, balancer.AveragedNorms["2"], 12);
        Assert.Equal(lastWeighted, balancer.LastWeightedLoss, 12);
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheReconstructionLoss()
    {
        await Task.Yield();
        using var model = new EnCodec<double>(Architecture(), TinyOptions());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = new Tensor<double>(new[] { 96 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * 1500 * i / 24000.0);
        double before = provider.EvaluateTrainingObjective(audio, audio);
        for (int i = 0; i < 30; i++) model.Train(audio, audio);
        double after = provider.EvaluateTrainingObjective(audio, audio);
        Assert.True(after < before, $"The reconstruction loss did not fall ({before} -> {after}).");
    }
}
