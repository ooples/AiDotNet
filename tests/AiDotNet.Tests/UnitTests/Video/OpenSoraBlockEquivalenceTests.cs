using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Video.Generation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// Pins OpenSora's DiT building blocks to the per-element algorithms they replaced.
/// </summary>
/// <remarks>
/// LayerNorm, the local-window attention and GELU used to run element by element through the tensor
/// indexer. They were rewritten onto engine operations because that path allocated 81 GB and ran on one
/// thread in the paper-scale census fixture. The references below are those loops, ported verbatim in
/// double, so the rewrite has to compute the same function - not merely a plausible one.
/// </remarks>
public class OpenSoraBlockEquivalenceTests
{
    private const int HiddenDim = 32;   // 16 heads x headDim 2
    private const int NumHeads = 16;

    private static OpenSora<double> CreateModel() =>
        new(new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Generative,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3),
            numFrames: 1,
            hiddenDim: HiddenDim,
            numLayers: 1,
            numInferenceSteps: 2);

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 4.0 - 2.0;
        return tensor;
    }

    private static void AssertClose(Tensor<double> expected, Tensor<double> actual, double tolerance)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        double worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        Assert.True(worst <= tolerance, $"max |difference| {worst:G6} exceeds {tolerance:G3}");
    }

    [Fact]
    public void LayerNorm_MatchesPerSampleNormalizationOverAllChannelsAndPositions()
    {
        var model = CreateModel();
        var input = Random([2, 5, 3, 4], seed: 11);

        var expected = new Tensor<double>(input._shape);
        for (int b = 0; b < 2; b++)
        {
            double sum = 0, sumSq = 0;
            const int count = 5 * 3 * 4;
            for (int c = 0; c < 5; c++)
                for (int h = 0; h < 3; h++)
                    for (int w = 0; w < 4; w++)
                    {
                        sum += input[b, c, h, w];
                        sumSq += input[b, c, h, w] * input[b, c, h, w];
                    }

            double mean = sum / count;
            double std = Math.Sqrt(sumSq / count - mean * mean + 1e-5);
            for (int c = 0; c < 5; c++)
                for (int h = 0; h < 3; h++)
                    for (int w = 0; w < 4; w++)
                        expected[b, c, h, w] = (input[b, c, h, w] - mean) / std;
        }

        AssertClose(expected, model.LayerNorm(input), 1e-9);
    }

    [Theory]
    [InlineData(4, 4)]    // seqLen 16: the window (16) covers half the sequence either side
    [InlineData(10, 10)]  // seqLen 100: the window is capped at 64, and the queries span two tiles
    [InlineData(1, 3)]    // seqLen 3: odd length, halfWindow 1
    [InlineData(12, 13)]  // seqLen 156: three query tiles, the last partial, each band crossing a tile edge
    public void Attention_MatchesTheLocalWindowScalarReference(int height, int width)
    {
        var model = CreateModel();
        const int batch = 2;
        int[] inputShape = [batch, HiddenDim, height, width];
        var qkv = Random([batch, 3 * HiddenDim, height, width], seed: 7 + height * 31 + width);

        AssertClose(ReferenceAttention(qkv, inputShape), model.DiTMultiHeadAttention(qkv, inputShape), 1e-9);
    }

    [Fact]
    public void Attention_OfASinglePositionIsZero()
    {
        // A one-position sequence has a zero-width window (halfWindow 0), which the scalar loop
        // turned into an empty softmax and a zero output.
        var model = CreateModel();
        int[] inputShape = [1, HiddenDim, 1, 1];
        var result = model.DiTMultiHeadAttention(Random([1, 3 * HiddenDim, 1, 1], seed: 3), inputShape);

        Assert.All(Enumerable.Range(0, result.Length), i => Assert.Equal(0.0, result[i]));
    }

    [Fact]
    public void Gelu_IsTheTanhApproximation()
    {
        var model = CreateModel();
        var input = Random([2, 3, 4, 5], seed: 5);
        var expected = new Tensor<double>(input._shape);
        double c = Math.Sqrt(2.0 / Math.PI);
        for (int i = 0; i < input.Length; i++)
        {
            double x = input[i];
            expected[i] = 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
        }

        AssertClose(expected, model.ApplyGELU(input), 1e-12);
    }

    [Fact]
    public void Construction_RejectsAHiddenDimTheHeadsCannotTile()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OpenSora<double>(new NeuralNetworkArchitecture<double>(
                    inputType: InputType.ThreeDimensional,
                    taskType: NeuralNetworkTaskType.Generative,
                    inputHeight: 16,
                    inputWidth: 16,
                    inputDepth: 3),
                hiddenDim: 40));
    }

    /// <summary>The pre-rewrite scalar local-window attention, verbatim.</summary>
    private static Tensor<double> ReferenceAttention(Tensor<double> qkv, int[] inputShape)
    {
        int batchSize = inputShape[0], channels = inputShape[1], height = inputShape[2], width = inputShape[3];
        int seqLen = height * width;
        int headDim = channels / NumHeads;
        var output = new Tensor<double>(inputShape);
        double scale = 1.0 / Math.Sqrt(headDim);

        for (int b = 0; b < batchSize; b++)
        {
            var q = new double[channels, seqLen];
            var k = new double[channels, seqLen];
            var v = new double[channels, seqLen];
            for (int c = 0; c < channels; c++)
            {
                int pos = 0;
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        q[c, pos] = qkv[b, c, h, w];
                        k[c, pos] = qkv[b, channels + c, h, w];
                        v[c, pos] = qkv[b, channels * 2 + c, h, w];
                        pos++;
                    }
            }

            var attOutput = new double[channels, seqLen];
            int windowSize = Math.Min(seqLen, 64);
            for (int headIdx = 0; headIdx < NumHeads; headIdx++)
            {
                int headStart = headIdx * headDim;
                int headEnd = Math.Min(headStart + headDim, channels);
                for (int i = 0; i < seqLen; i++)
                {
                    int wStart = Math.Max(0, i - windowSize / 2);
                    int wEnd = Math.Min(seqLen, i + windowSize / 2);
                    var scores = new double[wEnd - wStart];
                    double maxScore = double.MinValue;
                    for (int j = wStart; j < wEnd; j++)
                    {
                        double score = 0;
                        for (int c = headStart; c < headEnd; c++) score += q[c, i] * k[c, j];
                        score *= scale;
                        scores[j - wStart] = score;
                        if (score > maxScore) maxScore = score;
                    }

                    double sumExp = 0;
                    for (int j = 0; j < scores.Length; j++)
                    {
                        scores[j] = Math.Exp(scores[j] - maxScore);
                        sumExp += scores[j];
                    }

                    for (int j = 0; j < scores.Length; j++) scores[j] /= Math.Max(sumExp, 1e-12);

                    for (int c = headStart; c < headEnd; c++)
                    {
                        double weightedSum = 0;
                        for (int j = wStart; j < wEnd; j++) weightedSum += scores[j - wStart] * v[c, j];
                        attOutput[c, i] = weightedSum;
                    }
                }
            }

            for (int c = 0; c < channels; c++)
            {
                int pos = 0;
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[b, c, h, w] = attOutput[c, pos++];
            }
        }

        return output;
    }
}
