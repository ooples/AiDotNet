using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// FEDformer reproduces the official FEDformer (<c>tools/reference-data/fedformer_reference.py</c>, MAZiqing/FEDformer,
/// version 'Fourier'): with the reference's weights loaded through <see cref="FEDformer{T}.LoadReferenceWeights"/>, a
/// tiny configuration forecasts what the reference forecasts.
/// </summary>
/// <remarks>
/// The previous network was a generic transformer encoder with no frequency-enhanced block, no decomposition and no
/// decoder: it returned one value per input step instead of the horizon, and none of these weights had a place in it.
/// </remarks>
public class FEDformerReferenceTests
{
    private const double Tolerance = 1e-9;

    private static JsonElement Fixture()
    {
        const string name = "fedformer_reference.json";
        string relative = Path.Combine("IntegrationTests", "Finance", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static FEDformerOptions<double> Options(JsonElement config) => new()
    {
        SequenceLength = config.GetProperty("seq_len").GetInt32(),
        LabelLength = config.GetProperty("label_len").GetInt32(),
        PredictionHorizon = config.GetProperty("pred_len").GetInt32(),
        NumFeatures = config.GetProperty("enc_in").GetInt32(),
        ModelDimension = config.GetProperty("d_model").GetInt32(),
        NumHeads = config.GetProperty("n_heads").GetInt32(),
        FeedForwardDimension = config.GetProperty("d_ff").GetInt32(),
        NumEncoderLayers = config.GetProperty("e_layers").GetInt32(),
        NumDecoderLayers = config.GetProperty("d_layers").GetInt32(),
        NumModes = config.GetProperty("modes").GetInt32(),
        ModeSelection = FourierModeSelection.Lowest,
        MovingAverageKernel = config.GetProperty("moving_avg").GetInt32(),
        Dropout = 0.0,
        UseInstanceNormalization = false,
    };

    private static FEDformer<double> LoadReference(JsonElement f, out IReadOnlyCollection<string> read)
    {
        var options = Options(f.GetProperty("config"));
        var architecture = new NeuralNetworkArchitecture<double>(InputType.TwoDimensional, NeuralNetworkTaskType.Regression,
            inputHeight: options.SequenceLength, inputWidth: options.NumFeatures);
        var model = new FEDformer<double>(architecture, options: options);
        var state = f.GetProperty("state_dict");
        var shapes = f.GetProperty("shapes");
        read = model.LoadReferenceWeights((name, shape) =>
        {
            Assert.True(state.TryGetProperty(name, out var values), $"The reference has no '{name}'.");
            Assert.Equal(Ints(shapes.GetProperty(name)), shape);
            return Doubles(values);
        });
        model.SetTrainingMode(false);
        return model;
    }

    [Fact(Timeout = 120000)]
    public async Task Every_reference_weight_but_the_calendar_embeddings_has_a_place()
    {
        await Task.Yield();
        var f = Fixture();
        var model = LoadReference(f, out var read);

        var unread = f.GetProperty("state_dict").EnumerateObject().Select(p => p.Name).Except(read).OrderBy(n => n).ToArray();
        // The calendar ('timeF') embeddings map the time marks, which this FEDformer does not take.
        Assert.Equal(new[] { "dec_embedding.temporal_embedding.embed.weight", "enc_embedding.temporal_embedding.embed.weight" }, unread);

        // Every distinct reference tensor is a parameter: the shared Fourier blocks are counted once, as PyTorch does.
        var state = f.GetProperty("state_dict");
        long distinct = state.EnumerateObject()
            .Where(p => !p.Name.Contains("temporal_embedding"))
            .Where(p => !(p.Name.Contains("inner_correlation") && !p.Name.Contains("attn_layers.0.") && !p.Name.Contains("layers.0.")))
            .Sum(p => (long)p.Value.GetArrayLength());
        Assert.Equal(distinct, model.GetParameters().Length);
    }

    [Fact(Timeout = 120000)]
    public async Task The_forecast_matches_the_reference()
    {
        await Task.Yield();
        var f = Fixture();
        var model = LoadReference(f, out _);
        var input = new Tensor<double>(Ints(f.GetProperty("input_shape")), new Vector<double>(Doubles(f.GetProperty("input"))));

        var forecast = model.Forecast(input);

        Assert.Equal(Ints(f.GetProperty("forecast_shape")), forecast.Shape.ToArray());
        var expected = Doubles(f.GetProperty("forecast"));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - forecast.GetFlat(i)) <= Tolerance,
                $"forecast[{i}]: {forecast.GetFlat(i):R} vs the reference's {expected[i]:R}");
    }

    [Fact(Timeout = 300000)]
    public async Task Tape_gradients_match_finite_differences()
    {
        await Task.Yield();
        var f = Fixture();
        var model = LoadReference(f, out _);
        var input = new Tensor<double>(Ints(f.GetProperty("input_shape")), new Vector<double>(Doubles(f.GetProperty("input"))));
        var target = new Tensor<double>(Ints(f.GetProperty("forecast_shape")));
        for (int i = 0; i < target.Length; i++) target[i] = Math.Cos(0.7 * i);
        var loss = new AiDotNet.LossFunctions.MeanSquaredErrorLoss<double>();

        var gradient = model.ComputeGradients(input, target, loss);
        var parameters = model.GetParameters();
        Assert.Equal(parameters.Length, gradient.Length);

        double Loss(Vector<double> p)
        {
            model.SetParameters(p);
            var output = model.Forecast(input);
            double sum = 0;
            for (int i = 0; i < output.Length; i++) sum += (output.GetFlat(i) - target[i]) * (output.GetFlat(i) - target[i]);
            return sum / output.Length;
        }

        // Every 97th parameter: every block's weights are sampled, the shared Fourier weights included.
        const double h = 1e-6;
        for (int i = 0; i < parameters.Length; i += 97)
        {
            var plus = parameters.Clone();
            var minus = parameters.Clone();
            plus[i] += h;
            minus[i] -= h;
            double numeric = (Loss(plus) - Loss(minus)) / (2 * h);
            Assert.True(Math.Abs(numeric - gradient[i]) <= 1e-6 + 1e-4 * Math.Abs(numeric),
                $"parameter {i}: tape {gradient[i]:R} vs finite difference {numeric:R}");
        }
        model.SetParameters(parameters);
    }

    [Fact(Timeout = 120000)]
    public async Task Random_mode_selection_is_fixed_by_its_seed()
    {
        await Task.Yield();
        FEDformer<double> Build(int seed)
        {
            var options = Options(Fixture().GetProperty("config"));
            options.ModeSelection = FourierModeSelection.Random;
            options.NumModes = 3;
            options.ModeSelectionSeed = seed;
            return new FEDformer<double>(new NeuralNetworkArchitecture<double>(InputType.TwoDimensional,
                NeuralNetworkTaskType.Regression, inputHeight: 16, inputWidth: 3), options: options);
        }
        var input = new Tensor<double>(new[] { 1, 16, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(i * 0.37);

        var a = Build(5);
        var b = Build(5);
        var other = Build(6);
        b.SetParameters(a.GetParameters());
        other.SetParameters(a.GetParameters());

        // The same seed reads the same frequencies; another seed (5 and 6 pick different modes of 16 steps) does not,
        // so the same weights forecast differently.
        Assert.Equal(a.Forecast(input).ToVector().ToArray(), b.Forecast(input).ToVector().ToArray());
        Assert.NotEqual(a.Forecast(input).ToVector().ToArray(), other.Forecast(input).ToVector().ToArray());
    }

    /// <summary>
    /// FEDformer's dropout masks are drawn inside its one network layer, so a compiled training plan used to replay the
    /// traced step's masks on every later step; FEDformer opted out of fused training instead. The masks now come from
    /// CompiledStepRandom, which refills them before each replay, so the fused loss follows the eager loss step for
    /// step. A plan replaying frozen masks drops different elements from step 2 on and diverges.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Fused_training_redraws_dropout_masks_like_the_eager_step()
    {
        await Task.Yield();
        const int sequence = 16, label = 8, horizon = 4, features = 3, batch = 2;
        FedformerNetwork<float> Build() => new(sequence, label, horizon, features, modelDimension: 8, heads: 2,
            feedForward: 16, encoderLayers: 1, decoderLayers: 1, modes: 4, FourierModeSelection.Lowest,
            modeSelectionSeed: 1, movingAverage: 5, dropout: 0.3) { RandomSeed = 11 };
        var fusedNetwork = Build();
        var eagerNetwork = Build();
        var warm = new Tensor<float>(new[] { batch, sequence, features });
        fusedNetwork.Forward(warm);
        eagerNetwork.Forward(warm);
        eagerNetwork.SetParameters(fusedNetwork.GetParameters());
        fusedNetwork.SetTrainingMode(true);
        eagerNetwork.SetTrainingMode(true);

        var stepper = new TapeTrainingStepper<float>(new object());
        var fusedOptimizer = Sgd();
        var eagerOptimizer = Sgd();
        var data = RandomHelper.CreateSeededRandom(5);
        for (int step = 0; step < 4; step++)
        {
            var input = Random(data, batch, sequence, features);
            var target = Random(data, batch, horizon, features);
            float fused = stepper.Step(Request(fusedNetwork, input, target, fusedOptimizer));
            Assert.True(stepper.LastStepFused, $"step {step}: fused path did not engage ({stepper.Session.LastMissReason}).");
            float eager = TapeTrainingStepper<float>.EagerStep(Request(eagerNetwork, input, target, eagerOptimizer));
            Assert.True(
                Math.Abs(fused - eager) <= 1e-4f * Math.Max(1f, Math.Abs(eager)),
                $"step {step}: fused loss {fused} differs from eager loss {eager}; the plan is not redrawing its masks.");
        }
    }

    private static FusedTrainingStepRequest<float> Request(FedformerNetwork<float> network, Tensor<float> input,
        Tensor<float> target, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer) => new()
    {
        Layers = new ITrainableLayer<float>[] { network },
        Input = input,
        Target = target,
        Forward = network.Forward,
        ComputeLoss = (predicted, expected) =>
        {
            var engine = AiDotNetEngine.Current;
            var difference = engine.TensorSubtract(predicted, expected);
            return engine.TensorMultiplyScalar(
                engine.ReduceSum(engine.TensorMultiply(difference, difference), null), 1f / difference.Length);
        },
        Optimizer = optimizer,
    };

    private static StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>> Sgd()
        => new(model: null, options: new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.05,
            UseAdaptiveLearningRate = false,
        });

    private static Tensor<float> Random(System.Random random, params int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return tensor;
    }
}