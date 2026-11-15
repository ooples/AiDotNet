using System;
using System.Collections;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// LayerBase reuses one gradient buffer per layer across training steps. A vector a caller already read must never
/// be refilled by a later step, and each step must still publish that step's own gradients.
/// </summary>
[Collection("NonParallelIntegration")]
public sealed class LayerGradientPublishReuseTests
{
    [Fact]
    public void GradientsReadAfterAStep_AreNotOverwrittenByTheNextStep()
    {
        var model = Build();
        var dense = (LayerBase<float>)model.Layers[0];

        model.Train(Input(1), Target(0));
        var first = dense.GetParameterGradients();
        Assert.True(first.Length > 0, "no gradients were published, so this test would prove nothing");
        int nonZero = 0; foreach (var g in first.ToArray()) if (g != 0f) nonZero++;
        Assert.True(nonZero > 0, "the published gradients are all zero; GetParameterGradients is not reading what the step published");
        var firstSnapshot = first.ToArray();

        model.Train(Input(2), Target(1));
        var second = dense.GetParameterGradients();

        Assert.Equal(firstSnapshot, first.ToArray());
        Assert.NotSame(first, second);
        bool differs = false;
        for (int i = 0; i < firstSnapshot.Length && !differs; i++) differs = firstSnapshot[i] != second[i];
        Assert.True(differs, "step 2 published the same gradients as step 1; the surface did not update. step 1: "
            + string.Join(",", System.Linq.Enumerable.Take(firstSnapshot, 6)) + " step 2: "
            + string.Join(",", System.Linq.Enumerable.Take(second.ToArray(), 6)));
    }

    [Fact]
    public void UnreadGradients_AreRepublishedEachStep_FromThatStepAlone()
    {
        var reused = Build();
        var fresh = Build();
        fresh.SetParameters(reused.GetParameters());

        // Step 1 is never read on `reused`, so step 2 refills its buffer; `fresh` publishes step 2 only.
        reused.Train(Input(3), Target(2));
        reused.SetParameters(fresh.GetParameters());
        reused.ResetBaseTrainOptimizerState();
        reused.Train(Input(4), Target(0));
        fresh.Train(Input(4), Target(0));

        var expected = ((LayerBase<float>)fresh.Layers[0]).GetParameterGradients().ToArray();
        var actual = ((LayerBase<float>)reused.Layers[0]).GetParameterGradients().ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-6f,
                $"gradient {i}: {actual[i]} vs {expected[i]}; a reused buffer kept step-1 values");
    }

    [Fact]
    public void CompiledPlanHostGradients_AreReadWhenTheSurfaceIsRead_AndAReadVectorStaysFixed()
    {
        // A compiled plan's gradient buffers outlive its step (it republishes them on the next one), so the layer keeps
        // them by reference and fills its flat vector only when something reads it -- copying them every step cost a
        // CPU MLP ~15% of its training time for a vector nothing read. A vector already read is a snapshot.
        var model = Build();
        var dense = (LayerBase<float>)model.Layers[0];
        var parameters = dense.GetTrainableParameters();
        Assert.NotEmpty(parameters);
        var gradients = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>();
        foreach (var p in parameters) gradients[p] = Filled(p, 1f);

        Assert.True(dense.ScatterParameterGradients(gradients, sourcesOutliveStep: true) > 0);

        // The plan's next backward writes the same buffers; a read now must see what they hold.
        foreach (var g in gradients.Values) Fill(g, 7f);
        var read = dense.GetParameterGradients();
        Assert.True(read.Length > 0);
        Assert.All(read.ToArray(), v => Assert.Equal(7f, v));

        foreach (var g in gradients.Values) Fill(g, 9f);
        Assert.All(read.ToArray(), v => Assert.Equal(7f, v));
    }

    /// <summary>
    /// A publish that fails partway must withdraw the previous step's surface, not leave the reused buffer half
    /// refilled: the slots before the failure would hold this step's values and the rest the last step's.
    /// </summary>
    [Fact]
    public void APublishThatThrowsPartway_LeavesTheSurfaceEmpty_NeverMixedWithTheLastStep()
    {
        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var model = Build();
            var dense = (LayerBase<float>)model.Layers[0];
            var parameters = dense.GetTrainableParameters();
            Assert.True(parameters.Count >= 2, "the layer needs two trainable tensors for a fill to stop between them");

            var first = new Dictionary<Tensor<float>, Tensor<float>>();
            foreach (var p in parameters) first[p] = Filled(p, 1f);
            Assert.True(dense.ScatterParameterGradients(first) > 0);

            // Step 1 is never read, so step 2 refills the same buffer. Its dictionary answers one lookup, then throws:
            // the first tensor's slots are already written when the fill stops.
            var second = new Dictionary<Tensor<float>, Tensor<float>>();
            foreach (var p in parameters) second[p] = Filled(p, 2f);
            var failing = new ThrowingAfterLookups(second, allowedLookups: 1);
            Assert.Throws<InvalidOperationException>(() => dense.ScatterParameterGradients(failing));
            Assert.True(failing.Lookups >= 1, "the fill never reached a gradient, so nothing could have been half-written");

            var surface = dense.ScatteredParameterGradients;
            Assert.True(surface is null,
                "a failed publish left a surface behind: " + (surface is null ? "" : string.Join(",", surface.ToArray())));
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }

    /// <summary>
    /// A custom objective's step on a model that opts out of compilation runs the fused request's eager fallback. Its
    /// gradients are the tape's own tensors, which the model's MaxGradNorm clip and the optimizer's clip then scale in
    /// place. Publishing them by reference, as a plan's buffers are published, made GetParameterGradients return the
    /// clipped gradient. With both clips on, the surface must equal an identical model's with both clips off.
    /// </summary>
    [Fact]
    public void FusedRequestEagerFallback_PublishesTheGradientFromBeforeTheClips()
    {
        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            const double tinyClip = 1e-6;
            var clipped = CustomObjectiveNetwork.Create(tinyClip);
            var unclipped = CustomObjectiveNetwork.Create(clip: 0.0);
            var x = DoubleBatch(5, 11);
            var y = DoubleBatch(3, 12);
            clipped.Predict(x);
            unclipped.Predict(x);
            unclipped.SetParameters(clipped.GetParameters());

            clipped.Step(x, y);
            unclipped.Step(x, y);
            Assert.Equal(1, clipped.CustomEagerStepCount);
            Assert.Equal(1, unclipped.CustomEagerStepCount);

            var expected = unclipped.GetParameterGradients().ToArray();
            var actual = clipped.GetParameterGradients().ToArray();
            Assert.Equal(expected.Length, actual.Length);
            double normSq = 0.0;
            foreach (double g in expected) normSq += g * g;
            Assert.True(Math.Sqrt(normSq) > 1000 * tinyClip,
                $"gradient norm {Math.Sqrt(normSq):G4} is too small for the clips to change it; the test proves nothing");
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-12 * Math.Max(1.0, Math.Abs(expected[i])),
                    $"gradient {i}: published {actual[i]:G6} with the clips on, {expected[i]:G6} without them");
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }

    private static Tensor<double> DoubleBatch(int width, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(new[] { 6, width });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 4 - 2;
        return t;
    }

    /// <summary>A network whose custom objective always takes the fused request's eager fallback.</summary>
    private sealed class CustomObjectiveNetwork : NeuralNetwork<double>
    {
        private readonly IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> _optimizer;

        private CustomObjectiveNetwork(NeuralNetworkArchitecture<double> architecture,
            IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer, double clip)
            : base(architecture, optimizer)
        {
            _optimizer = optimizer;
            MaxGradNorm = clip;
        }

        public static CustomObjectiveNetwork Create(double clip)
        {
            var layers = new List<ILayer<double>>
            {
                new DenseLayer<double>(8, (IActivationFunction<double>)new TanhActivation<double>()),
                new DenseLayer<double>(3, (IActivationFunction<double>?)null),
            };
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 5, outputSize: 3, layers: layers);
            var options = new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-2,
                EnableGradientClipping = clip > 0.0,
                GradientClippingMethod = GradientClippingMethod.ByNorm,
                MaxGradientNorm = clip > 0.0 ? clip : 1.0,
            };
            var model = new CustomObjectiveNetwork(architecture,
                new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null, options), clip);
            model.SetTrainingMode(true);
            return model;
        }

        protected override bool SupportsFusedCompiledTraining => false;

        public void Step(Tensor<double> input, Tensor<double> target)
            => TrainCustomTapeStep(input, target, ForwardForTraining, SquaredError, _optimizer);

        private static Tensor<double> SquaredError(Tensor<double> predicted, Tensor<double> target)
        {
            var engine = AiDotNetEngine.Current;
            var diff = engine.TensorSubtract(predicted, target);
            return engine.ReduceSum(engine.TensorMultiply(diff, diff), null);
        }
    }

    /// <summary>A gradient dictionary whose lookups throw once <c>allowedLookups</c> have been answered.</summary>
    private sealed class ThrowingAfterLookups : IReadOnlyDictionary<Tensor<float>, Tensor<float>>
    {
        private readonly Dictionary<Tensor<float>, Tensor<float>> _inner;
        private readonly int _allowedLookups;

        public ThrowingAfterLookups(Dictionary<Tensor<float>, Tensor<float>> inner, int allowedLookups)
        {
            _inner = inner;
            _allowedLookups = allowedLookups;
        }

        public int Lookups { get; private set; }

        public bool TryGetValue(Tensor<float> key,
            [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out Tensor<float> value)
        {
            if (Lookups >= _allowedLookups)
                throw new InvalidOperationException("gradient lookup failed partway through the publish");
            Lookups++;
            return _inner.TryGetValue(key, out value);
        }

        public Tensor<float> this[Tensor<float> key]
            => TryGetValue(key, out var value) ? value : throw new KeyNotFoundException();

        public IEnumerable<Tensor<float>> Keys => _inner.Keys;
        public IEnumerable<Tensor<float>> Values => _inner.Values;
        public int Count => _inner.Count;
        public bool ContainsKey(Tensor<float> key) => _inner.ContainsKey(key);
        public IEnumerator<KeyValuePair<Tensor<float>, Tensor<float>>> GetEnumerator() => _inner.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private static Tensor<float> Filled(Tensor<float> like, float value)
    {
        var t = new Tensor<float>(like.Shape.ToArray());
        Fill(t, value);
        return t;
    }

    private static void Fill(Tensor<float> t, float value)
    {
        for (int i = 0; i < t.Length; i++) t[i] = value;
    }

    private static FeedForwardNeuralNetwork<float> Build()
    {
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new DenseLayer<float>(10, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(3, activationFunction: (IActivationFunction<float>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 5,
            outputSize: 3,
            layers: layers);
        var model = new FeedForwardNeuralNetwork<float>(architecture,
            lossFunction: new CrossEntropyWithLogitsLoss<float>());
        model.SetTrainingMode(true);
        return model;
    }

    private static Tensor<float> Input(int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var input = new Tensor<float>(new[] { 6, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        return input;
    }

    private static Tensor<float> Target(int shift)
    {
        var target = new Tensor<float>(new[] { 6, 3 });
        for (int row = 0; row < 6; row++) target[row * 3 + (row + shift) % 3] = 1f;
        return target;
    }
}
