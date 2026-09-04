using System.Diagnostics;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// The builder's JIT path must never return a prediction that ignores its input.
/// </summary>
/// <remarks>
/// <para>
/// <c>AiModelBuilder.BuildCompiledPredictFunction</c> compiles the model by tracing one forward pass
/// and replaying the recorded plan. That is only sound when the trace treats the input as a graph
/// input rather than folding its values in as constants. <c>NeuralNetworkBase.PredictCore</c>
/// documents the failure and sidesteps it by never routing <c>Predict</c> through
/// <c>PredictCompiled</c>; the builder reintroduced it by hand-rolling its own cache.
/// </para>
/// <para>
/// A Transformer used to trigger it on the first op — the embedding gather read token ids out of the
/// input tensor, so the indices became constants and every replay returned the traced input's logits.
/// The Tensors input-rebinding fixes now make that graph value-stable. The builder still verifies a
/// plan against the eager forward before trusting it, and these tests pin all three outcomes: stable
/// Transformer and dense graphs stay compiled, while a typed deterministic limitation falls back
/// once per shape.
/// </para>
/// </remarks>
public class BuilderJitValueStabilityTests
{
    private const int SeqLen = 16;
    private const int Vocab = 64;

    /// <summary>Captures Trace output so the fallback warning is observable rather than inferred.</summary>
    private sealed class WarningCollector : TraceListener
    {
        public List<string> Messages { get; } = new List<string>();

        public override void Write(string? message) => Add(message);

        public override void WriteLine(string? message) => Add(message);

        private void Add(string? message)
        {
            if (!string.IsNullOrEmpty(message)) Messages.Add(message!);
        }

        public int JitDisabledCountFor(Type modelType)
        {
            string prefix = $"JIT disabled for {modelType.FullName}:";
            return Messages.Count(m => m.Contains(prefix, StringComparison.Ordinal));
        }

        public int JitFallbackCountFor(Type modelType)
        {
            string prefix = $"JIT fallback for {modelType.FullName} ";
            return Messages.Count(m => m.Contains(prefix, StringComparison.Ordinal));
        }

        /// <summary>
        /// Everything Trace emitted, for the assertion messages. A bare "the check did not fire"
        /// cannot distinguish "the plan was judged stable" from "compilation threw and the OTHER
        /// fallback warning was traced" from "the builder never built a compiled function at all",
        /// and those have three different fixes.
        /// </summary>
        public string Transcript => Messages.Count == 0
            ? "(Trace emitted nothing)"
            : string.Join(" | ", Messages.ConvertAll(m => m.Trim()));
    }

    private sealed class NonCompilableFeedForwardNetwork : FeedForwardNeuralNetwork<float>
    {
        public NonCompilableFeedForwardNetwork(NeuralNetworkArchitecture<float> architecture)
            : base(architecture)
        {
        }

        public override Tensor<float> Predict(Tensor<float> input) => input;
    }

    private static double MaxPairwise(Func<Tensor<float>, Tensor<float>> predict, int count = 12)
    {
        var outputs = new float[count][];

        for (int k = 0; k < count; k++)
        {
            var input = new Tensor<float>([1, SeqLen]);
            for (int s = 0; s < SeqLen; s++) input[0, s] = k + 1;

            var produced = predict(input);
            var flat = new float[produced.Length];
            for (int i = 0; i < produced.Length; i++) flat[i] = produced[i];
            outputs[k] = flat;
        }

        double worst = 0.0;
        for (int i = 0; i < count; i++)
        {
            for (int j = i + 1; j < count; j++)
            {
                double sum = 0.0;
                int n = Math.Min(outputs[i].Length, outputs[j].Length);
                for (int q = 0; q < n; q++)
                {
                    double d = outputs[i][q] - outputs[j][q];
                    sum += d * d;
                }

                worst = Math.Max(worst, Math.Sqrt(sum));
            }
        }

        return worst;
    }

    private static (Tensor<float> Features, Tensor<float> Labels) Corpus(int outputSize)
    {
        const int samples = 64;
        var features = new Tensor<float>([samples, SeqLen]);
        var labels = new Tensor<float>([samples, outputSize]);
        var rng = new Random(7);

        for (int n = 0; n < samples; n++)
        {
            for (int s = 0; s < SeqLen; s++) features[n, s] = rng.Next(Vocab);
            labels[n, rng.Next(outputSize)] = 1f;
        }

        return (features, labels);
    }

    private static async Task<AiDotNet.Models.Results.AiModelResult<
        float,
        Tensor<float>,
        Tensor<float>>> BuildWithJit(
        IFullModel<float, Tensor<float>, Tensor<float>> model,
        int outputSize,
        JitCompilationConfig? jitConfig = null)
    {
        var (features, labels) = Corpus(outputSize);

        return await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                { InitialLearningRate = 0.0003 }))
            .ConfigureDataLoader(DataLoaders.FromTensors(features, labels))
            .ConfigureJitCompilation(jitConfig ?? JitCompilationConfig.Default)
            .BuildAsync();
    }

    private static async Task<(
        double JitMax,
        double EagerMax,
        bool SawJitDisabled,
        bool SawJitFallback,
        string Transcript)> BuildAndMeasure(
        IFullModel<float, Tensor<float>, Tensor<float>> model,
        Func<Tensor<float>, Tensor<float>> eagerPredict,
        int outputSize)
    {
        var collector = new WarningCollector();
        Trace.Listeners.Add(collector);

        try
        {
            var (features, labels) = Corpus(outputSize);

            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                    null,
                    new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    { InitialLearningRate = 0.0003 }))
                .ConfigureDataLoader(DataLoaders.FromTensors(features, labels))
                .ConfigureJitCompilation(JitCompilationConfig.Default);

            var result = await builder.BuildAsync();

            return (
                MaxPairwise(result.Predict),
                MaxPairwise(eagerPredict),
                collector.JitDisabledCountFor(model.GetType()) > 0,
                collector.JitFallbackCountFor(model.GetType()) > 0,
                collector.Transcript);
        }
        finally
        {
            Trace.Listeners.Remove(collector);
        }
    }

    [Fact]
    public async Task ATransformerIsAccepted_AndPredictsCorrectly()
    {
        // Before the input-rebinding fix, result.Predict returned max pairwise L2 of EXACTLY 0
        // across every distinct input because the replayed plan had the traced token ids baked in.
        // The repaired plan must now remain enabled and agree with eager inference.
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 32,
            feedForwardDimension: 64,
            inputSize: SeqLen,
            outputSize: Vocab,
            maxSequenceLength: SeqLen,
            vocabularySize: Vocab);

        var model = new Transformer<float>(
            architecture,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                { InitialLearningRate = 0.0003 }));

        var (jitMax, eagerMax, sawJitDisabled, sawJitFallback, transcript) =
            await BuildAndMeasure(model, model.Predict, Vocab);

        Assert.False(sawJitDisabled,
            "the Transformer's rebound graph is value-stable, so the builder should retain the " +
            $"compiled plan. Trace said: {transcript}");
        Assert.False(sawJitFallback,
            $"the Transformer's compiled path should not fall back. Trace said: {transcript}");

        Assert.True(jitMax > 1e-4,
            $"result.Predict still ignores its input: max pairwise L2 = {jitMax:E3}");
        Assert.True(Math.Abs(jitMax - eagerMax) < 1e-4,
            $"the JIT path disagrees with the eager forward: {jitMax:E3} against {eagerMax:E3}");
    }

    [Fact]
    public async Task AFeedForwardNetworkIsAccepted_SoJitIsNotSilentlyDisabled()
    {
        // The other direction, and the one that keeps this fix from being a disguised feature
        // removal. A plain dense stack has no gather reading input values as indices, so its trace
        // IS value-stable, the plan is trusted, and no fallback warning is emitted.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: SeqLen,
            outputSize: 4);

        var model = new FeedForwardNeuralNetwork<float>(architecture);

        var (jitMax, eagerMax, sawJitDisabled, sawJitFallback, transcript) =
            await BuildAndMeasure(model, model.Predict, 4);

        Assert.False(sawJitDisabled,
            "a dense network's trace is value-stable, so the builder should keep using the " +
            $"compiled plan rather than falling back -- otherwise ConfigureJitCompilation is inert. Trace said: {transcript}");
        Assert.False(sawJitFallback,
            $"the dense network's compiled path should not fall back. Trace said: {transcript}");

        Assert.True(jitMax > 1e-6,
            $"the compiled plan ignores its input: max pairwise L2 = {jitMax:E3}");
        Assert.True(Math.Abs(jitMax - eagerMax) < 1e-3,
            $"the compiled plan disagrees with the eager forward: {jitMax:E3} against {eagerMax:E3}");
    }

    [Fact]
    public async Task DeterministicCaptureLimitation_DisablesOnlyThatShapeAfterFirstFailure()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: SeqLen,
            outputSize: SeqLen);
        var model = new NonCompilableFeedForwardNetwork(architecture);
        var result = await BuildWithJit(model, SeqLen);
        var collector = new WarningCollector();
        Trace.Listeners.Add(collector);

        try
        {
            var firstShape = new Tensor<float>([1, SeqLen]);
            firstShape[0, 3] = 7f;

            var first = result.Predict(firstShape);
            var second = result.Predict(firstShape);

            Assert.Equal(7f, first[0, 3]);
            Assert.Equal(7f, second[0, 3]);
            Assert.True(
                collector.JitDisabledCountFor(model.GetType()) == 1,
                $"expected one typed JIT disablement for the first shape. Trace said: {collector.Transcript}");

            // A different rank is a different CompiledModelCache plan. It must make its own capture
            // decision rather than inheriting the first shape's terminal state; then the first shape
            // must remain disabled without retrying when selected again.
            var secondShape = new Tensor<float>([1, 1, SeqLen]);
            secondShape[0, 0, 5] = 11f;
            var third = result.Predict(secondShape);
            Assert.Equal(11f, third[0, 0, 5]);
            Assert.Equal(2, collector.JitDisabledCountFor(model.GetType()));

            _ = result.Predict(firstShape);
            Assert.Equal(2, collector.JitDisabledCountFor(model.GetType()));
            Assert.Contains("NoCompilableOperations", collector.Transcript);
        }
        finally
        {
            Trace.Listeners.Remove(collector);
        }
    }

    [Fact]
    public async Task StrictMode_PropagatesTypedCaptureLimitation()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: SeqLen,
            outputSize: SeqLen);
        var model = new NonCompilableFeedForwardNetwork(architecture);
        var config = JitCompilationConfig.Default;
        config.ThrowOnFailure = true;
        var result = await BuildWithJit(model, SeqLen, config);

        var input = new Tensor<float>([1, SeqLen]);
        var exception = Assert.Throws<
            AiDotNet.Tensors.Engines.Compilation.GraphCaptureNotSupportedException>(
                () => result.Predict(input));

        Assert.Equal(
            AiDotNet.Tensors.Engines.Compilation.GraphCaptureLimitation.NoCompilableOperations,
            exception.Limitation);
    }

    [Fact]
    public async Task AWarmedConvolutionalFastPath_IsCapturedWithoutLeavingTheGraph()
    {
        var layers = new List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(
                outputDepth: 2,
                kernelSize: 3,
                stride: 1,
                padding: 1,
                activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(3, activationFunction: (IActivationFunction<float>?)null)
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 3,
            layers: layers);
        var model = new ConvolutionalNeuralNetwork<float>(architecture);

        // Materialize the lazy weights first so the model's raw-buffer fused stem is eligible when
        // the builder starts tracing. Without the central compilation guard, that fast path escapes
        // GraphMode and produces a zero-operation/unrooted capture.
        _ = model.Predict(new Tensor<float>([1, 1, 8, 8]));

        var features = new Tensor<float>([8, 1, 8, 8]);
        var labels = new Tensor<float>([8, 3]);
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>()
                { InitialLearningRate = 0.0003 }))
            .ConfigureDataLoader(DataLoaders.FromTensors(features, labels))
            .ConfigureJitCompilation(JitCompilationConfig.Default)
            .BuildAsync();

        var inputA = new Tensor<float>([1, 1, 8, 8]);
        var inputB = new Tensor<float>([1, 1, 8, 8]);
        for (int i = 0; i < inputB.Length; i++) inputB[i] = (i % 7) * 0.25f;

        var collector = new WarningCollector();
        Trace.Listeners.Add(collector);
        try
        {
            _ = result.Predict(inputA);
            var compiledB = result.Predict(inputB);

            Assert.Equal(0, collector.JitDisabledCountFor(model.GetType()));
            Assert.Equal(0, collector.JitFallbackCountFor(model.GetType()));

            var savedOptions = TensorCodecOptions.Current;
            Tensor<float> eagerB;
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            try
            {
                eagerB = model.Predict(inputB);
            }
            finally
            {
                TensorCodecOptions.SetCurrent(savedOptions);
            }

            Assert.Equal(eagerB.Length, compiledB.Length);
            double maxDifference = 0;
            for (int i = 0; i < eagerB.Length; i++)
                maxDifference = Math.Max(maxDifference, Math.Abs(eagerB[i] - compiledB[i]));
            Assert.True(
                maxDifference <= 1e-3,
                $"the compiled CNN plan disagrees with eager inference by {maxDifference:E3}");
        }
        finally
        {
            Trace.Listeners.Remove(collector);
        }
    }
}
