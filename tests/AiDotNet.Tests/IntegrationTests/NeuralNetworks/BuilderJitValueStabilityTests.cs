using System.Diagnostics;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
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
/// A Transformer triggers it on the first op — the embedding gather reads token ids out of the input
/// tensor, so the indices become constants and every replay returns the traced input's logits. The
/// builder now verifies a plan against the eager forward before trusting it, and these two tests pin
/// both directions of that check: it must reject the unstable model, and it must NOT reject a stable
/// one, or <c>ConfigureJitCompilation</c> would be inert.
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

        public bool SawJitDisabled => Messages.Exists(m => m.Contains("JIT disabled"));

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

    private static async Task<(double JitMax, double EagerMax, bool SawJitDisabled, string Transcript)> BuildAndMeasure(
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

            return (MaxPairwise(result.Predict), MaxPairwise(eagerPredict), collector.SawJitDisabled,
                collector.Transcript);
        }
        finally
        {
            Trace.Listeners.Remove(collector);
        }
    }

    [Fact]
    public async Task ATransformerIsRejected_AndStillPredictsCorrectly()
    {
        // The reproduction. Before the fix, result.Predict returned max pairwise L2 of EXACTLY 0
        // across every distinct input -- the replayed plan had the traced token ids baked in.
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

        var (jitMax, eagerMax, sawJitDisabled, transcript) =
            await BuildAndMeasure(model, model.Predict, Vocab);

        // The check must fire on this model...
        Assert.True(sawJitDisabled,
            "the builder should have detected that the Transformer's plan is not value-stable " +
            $"and traced the fallback. Trace said: {transcript}");

        // ...and the answers must be right regardless, which is the point of falling back.
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

        var (jitMax, eagerMax, sawJitDisabled, transcript) =
            await BuildAndMeasure(model, model.Predict, 4);

        Assert.False(sawJitDisabled,
            "a dense network's trace is value-stable, so the builder should keep using the " +
            $"compiled plan rather than falling back -- otherwise ConfigureJitCompilation is inert. Trace said: {transcript}");

        Assert.True(jitMax > 1e-6,
            $"the compiled plan ignores its input: max pairwise L2 = {jitMax:E3}");
        Assert.True(Math.Abs(jitMax - eagerMax) < 1e-3,
            $"the compiled plan disagrees with the eager forward: {jitMax:E3} against {eagerMax:E3}");
    }
}
