using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Attention-probability dropout on <see cref="MultiHeadAttentionLayer{T}"/> (GPT-2's <c>attn_pdrop</c>): the
/// softmax weights are dropped and rescaled by 1 / (1 - p) in training, the layer is unchanged at inference, and
/// the explicit training path is tape-correct.
/// </summary>
public sealed class MultiHeadAttentionDropoutTests
{
    private const int HeadCount = 2;
    private const int HeadDimension = 4;
    private const int Embedding = HeadCount * HeadDimension;

    /// <summary>
    /// In inference mode the dropout is the identity: a layer configured with rate 0.5 predicts exactly what the
    /// same weights predict without attention dropout. Fails if the dropout (or its explicit path) runs at
    /// inference.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task InferenceIsIdenticalToTheUndroppedLayer()
    {
        await Task.Yield();

        var input = CreatePattern([2, 5, Embedding], 0.019);
        var plain = NewLayer(HeadCount, causal: true, rate: 0.0);
        var dropped = NewLayer(HeadCount, causal: true, rate: 0.5);
        plain.SetTrainingMode(false);
        dropped.SetTrainingMode(false);
        Forward(plain, input);
        Forward(dropped, input);
        dropped.SetParameters(plain.GetParameters());

        var expected = Forward(plain, input);
        var actual = Forward(dropped, input);
        Assert.Equal(0.0, MaxAbsDifference(expected, actual));
        Assert.Equal(0.5, dropped.AttentionDropoutRate);
    }

    /// <summary>
    /// With a vanishing rate the explicit training composition softmax(QK^T * scale + causal bias) V must equal the
    /// fused kernel the layer uses otherwise. Fails on a wrong scale, a wrong head reshape, or a causal bias that
    /// masks the wrong keys.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task ExplicitTrainingPathMatchesTheFusedKernel()
    {
        await Task.Yield();

        foreach (bool causal in new[] { true, false })
        {
            var input = CreatePattern([2, 6, Embedding], 0.023);
            var reference = NewLayer(HeadCount, causal, rate: 0.0);
            var explicitPath = NewLayer(HeadCount, causal, rate: 1e-12);
            reference.SetTrainingMode(false);
            Forward(reference, input);
            Forward(explicitPath, input);
            explicitPath.SetParameters(reference.GetParameters());

            var expected = Forward(reference, input);
            explicitPath.SetTrainingMode(true);
            var actual = Forward(explicitPath, input);

            Assert.Equal(AttentionKernelPath.ScoreMaterializing, explicitPath.LastAttentionKernelPath);
            Assert.True(MaxAbsDifference(expected, actual) < 1e-10,
                $"causal={causal}: explicit path differs from the fused kernel by {MaxAbsDifference(expected, actual):R}.");
        }
    }

    /// <summary>
    /// A causal one-head layer gives query 0 a single key with weight exactly 1, so in training its context row is
    /// either dropped (0) or kept and scaled by 1 / (1 - 0.5) = 2. Its output is therefore one of two values whose
    /// sum is twice the inference output, and both occur. Fails if the probabilities are not dropped, if the
    /// survivors are not rescaled, or if dropout also runs at inference.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task TrainingDropsAndRescalesTheAttentionWeights()
    {
        await Task.Yield();

        var layer = NewLayer(1, causal: true, rate: 0.5, headDimension: Embedding);
        var input = CreatePattern([1, 3, Embedding], 0.031);
        layer.SetTrainingMode(false);
        var inference = Forward(layer, input);

        layer.SetTrainingMode(true);
        var outcomes = new List<double[]>();
        for (int draw = 0; draw < 40; draw++)
        {
            var output = Forward(layer, input);
            var row = new double[Embedding];
            for (int f = 0; f < Embedding; f++) row[f] = output[f];
            if (!outcomes.Any(seen => seen.Zip(row, (a, b) => Math.Abs(a - b)).Max() < 1e-12))
                outcomes.Add(row);
        }

        Assert.Equal(2, outcomes.Count);
        for (int f = 0; f < Embedding; f++)
        {
            double sum = outcomes[0][f] + outcomes[1][f];
            Assert.True(Math.Abs(sum - 2.0 * inference[f]) < 1e-10,
                $"Feature {f}: dropped + kept = {sum:R}, expected twice the inference output {2.0 * inference[f]:R}.");
        }
    }

    /// <summary>
    /// The explicit path is tape-correct: input gradients recorded through the dropout branch match central finite
    /// differences, and the causal bias keeps the future out of earlier rows' gradients. Fails if any op of the
    /// branch bypasses the tape.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ExplicitTrainingPathGradientsMatchFiniteDifferences()
    {
        await Task.Yield();

        var layer = NewLayer(HeadCount, causal: true, rate: 1e-12);
        layer.SetTrainingMode(true);
        var input = CreatePattern([2, 5, Embedding], 0.017);
        var projection = CreatePattern([2, 5, Embedding], 0.029);

        Tensor<double>? gradient;
        using (var tape = new GradientTape<double>())
        {
            var output = layer.Forward(input);
            var projected = AiDotNetEngine.Current.TensorMultiply(output, projection);
            var objective = AiDotNetEngine.Current.ReduceSum(projected, [0, 1, 2], keepDims: false);
            var gradients = tape.ComputeGradients(objective, [input]);
            Assert.True(gradients.TryGetValue(input, out gradient), "The dropout path disconnected the tape.");
        }

        Assert.NotNull(gradient);
        Assert.Equal(AttentionKernelPath.ScoreMaterializing, layer.LastAttentionKernelPath);
        const double step = 1e-6;
        int[] samples = [0, input.Length / 3, input.Length / 2, input.Length - 1];
        foreach (int index in samples)
        {
            double original = input[index];
            input[index] = original + step;
            double plus = Project(layer, input, projection);
            input[index] = original - step;
            double minus = Project(layer, input, projection);
            input[index] = original;

            double numerical = (plus - minus) / (2.0 * step);
            double analytical = gradient[index];
            double tolerance = Math.Max(1e-7, Math.Abs(numerical) * 1e-5);
            Assert.True(Math.Abs(numerical - analytical) <= tolerance,
                $"Input gradient mismatch at {index}: numerical={numerical:R}, analytical={analytical:R}.");
        }
    }

    /// <summary>Rates outside [0, 1) are rejected.</summary>
    [Fact]
    public void InvalidRatesAreRejected()
    {
        var layer = NewLayer(HeadCount, causal: false, rate: 0.0);
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.ConfigureAttentionDropout(-0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.ConfigureAttentionDropout(1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.ConfigureAttentionDropout(double.NaN));
    }

    private static MultiHeadAttentionLayer<double> NewLayer(
        int heads, bool causal, double rate, int headDimension = HeadDimension)
    {
        var layer = new MultiHeadAttentionLayer<double>(
            heads,
            headDimension,
            activationFunction: (IActivationFunction<double>)new IdentityActivation<double>());
        layer.UseCausalMask = causal;
        layer.ConfigureAttentionDropout(rate);
        return layer;
    }

    private static Tensor<double> Forward(MultiHeadAttentionLayer<double> layer, Tensor<double> input)
    {
        using var noGrad = new NoGradScope<double>();
        return layer.Forward(input);
    }

    private static double Project(
        MultiHeadAttentionLayer<double> layer, Tensor<double> input, Tensor<double> projection)
    {
        using var noGrad = new NoGradScope<double>();
        var output = layer.Forward(input);
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++)
            sum += output[i] * projection[i];
        return sum;
    }

    private static double MaxAbsDifference(Tensor<double> a, Tensor<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    private static Tensor<double> CreatePattern(int[] shape, double scale)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = (Math.Sin((i + 1) * scale) * 0.5) + (Math.Cos((i + 3) * scale * 0.5) * 0.25);
        return tensor;
    }
}
