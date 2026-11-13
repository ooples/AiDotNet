#if NET5_0_OR_GREATER
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using Xunit;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

public class Elementwise11MemoryOptimizerTests
{
    [Fact]
    public void Anp_HalfAttentionIsStableAndEmptyQueryUsesUniformWeights()
    {
        var algorithm = Uninitialized<ANPAlgorithm<Half, Matrix<Half>, Vector<Half>>>();
        typeof(ANPAlgorithm<Half, Matrix<Half>, Vector<Half>>).BaseType!
            .GetField("RepresentationDim", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(algorithm, 2);
        Field(algorithm, "_attentionParams", new Vector<Half>(new[] { (Half)1, (Half)1 }));
        var context = new List<Vector<Half>>
        {
            new(new[] { (Half)1, (Half)2 }),
            new(new[] { (Half)3, (Half)4 })
        };
        var moderate = Assert.IsType<Vector<Half>>(Call(algorithm, "CrossAttend",
            new Vector<Half>(new[] { (Half)0.5, (Half)(-0.25) }), context));
        double secondWeight = Math.Exp(0.5 / Math.Sqrt(2));
        Assert.Equal((Half)((1 + 3 * secondWeight) / (1 + secondWeight)), moderate[0]);
        Assert.Equal((Half)((2 + 4 * secondWeight) / (1 + secondWeight)), moderate[1]);
        var result = Assert.IsType<Vector<Half>>(Call(algorithm, "CrossAttend",
            new Vector<Half>(new[] { (Half)60000, (Half)60000 }), context));
        Assert.Equal(new[] { (Half)3, (Half)4 }, result.ToArray());
        var uniform = Assert.IsType<Vector<Half>>(Call(algorithm, "CrossAttend", new Vector<Half>(0), context));
        Assert.Equal(new[] { (Half)2, (Half)3 }, uniform.ToArray());
        double scale = Assert.IsType<double>(Call(algorithm, "ComputeScale",
            new Vector<Half>(new[] { (Half)1, (Half)2 }), new Vector<Half>(new[] { (Half)3, (Half)4 })));
        Assert.Equal(0.5 + 0.5 / (1 + Math.Exp(1 - Math.Sqrt(7.5))), scale, 12);
        Assert.Equal(1.0, Assert.IsType<double>(Call(algorithm, "ComputeScale",
            new Vector<Half>(new[] { (Half)60000 }), new Vector<Half>(new[] { (Half)60000 }))));
    }


    [Fact]
    public void Feat_HalfAttentionAndLayerNormalizationKeepIntermediatesWide()
    {
        var algorithm = Uninitialized<FEATAlgorithm<Half, Matrix<Half>, Vector<Half>>>();
        var options = Uninitialized<FEATOptions<Half, Matrix<Half>, Vector<Half>>>();
        options.NumTransformerLayers = 1;
        Field(algorithm, "_featOptions", options);
        Field(algorithm, "_transformerParams", new Vector<Half>(Enumerable.Repeat((Half)1, 16).ToArray()));
        var input = new Vector<Half>(new[] { (Half)100, (Half)200, (Half)300, (Half)400 });
        var result = Assert.IsType<Vector<Half>>(Call(algorithm, "AdaptPrototypes", input));
        // All scores cap at 10, so attention is uniform and the residual is input + 250.
        double standardDeviation = Math.Sqrt(12500 + 1e-5);
        for (int i = 0; i < input.Length; i++)
            Assert.Equal((Half)(((double)input[i] - 250) / standardDeviation), result[i]);
        var extremes = Assert.IsType<Vector<Half>>(Call(algorithm, "AdaptPrototypes",
            new Vector<Half>(new[] { (Half)(-60000), (Half)60000 })));
        Assert.Equal(new[] { (Half)(-1), (Half)1 }, extremes.ToArray());
    }

    [Fact]
    public void Epnet_HalfPreservesPositiveBandwidthAndFinitePropagation()
    {
        var algorithm = Uninitialized<EPNetAlgorithm<Half, Matrix<Half>, Vector<Half>>>();
        var options = Uninitialized<EPNetOptions<Half, Matrix<Half>, Vector<Half>>>();
        options.NumNeighbors = 1;
        options.PropagationAlpha = 0.5;
        options.PropagationIterations = 1;
        Field(algorithm, "_epnetOptions", options);
        var constant = Assert.IsType<Vector<Half>>(Call(algorithm, "PropagateEmbeddings", new Vector<Half>(new[] { (Half)1, (Half)1 })));
        Assert.Equal(new[] { (Half)1, (Half)1 }, constant.ToArray());
        var extremes = Assert.IsType<Vector<Half>>(Call(algorithm, "PropagateEmbeddings",
            new Vector<Half>(new[] { (Half)(-60000), (Half)60000 })));
        Assert.Equal(new[] { (Half)0, (Half)0 }, extremes.ToArray());
    }

    [Fact]
    public void DeepEmd_BlockedCostsMatchScalarDistancesAcrossAllBlockBoundaries()
    {
        var algorithm = Uninitialized<DeepEMDAlgorithm<double, Matrix<double>, Vector<double>>>();
        const int rows = 33, columns = 35, dimensions = 257;
        var first = new Tensor<double>(new[] { rows, dimensions });
        var second = new Tensor<double>(new[] { columns, dimensions });
        for (int i = 0; i < rows; i++)
        for (int d = 0; d < dimensions; d++) first[i, d] = i * 0.1 + d % 7;
        for (int j = 0; j < columns; j++)
        for (int d = 0; d < dimensions; d++) second[j, d] = j * 0.2 - d % 3;
        var costs = Assert.IsType<Tensor<double>>(Call(algorithm, "PairwiseCosts", first, second, rows, columns, dimensions));
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
        {
            double expected = 0;
            for (int d = 0; d < dimensions; d++) expected += Math.Pow(first[i, d] - second[j, d], 2);
            Assert.Equal(expected, costs[i, j], 7);
        }
    }

    [Fact]
    public void DeepEmd_LargeFeatureGridHasAnalyticTransportCost()
    {
        var algorithm = Uninitialized<DeepEMDAlgorithm<double, Matrix<double>, Vector<double>>>();
        var options = Uninitialized<DeepEMDOptions<double, Matrix<double>, Vector<double>>>();
        options.NumNodes = 257;
        options.SinkhornIterations = 3;
        options.SinkhornRegularization = 1024;
        Field(algorithm, "_deepEmdOptions", options);
        var first = new Vector<double>(257 * 257);
        var second = new Vector<double>(Enumerable.Repeat(1.0, first.Length).ToArray());
        double score = Assert.IsType<double>(Call(algorithm, "ComputeEMDScore", first, second));
        Assert.Equal(257.0, score, 8);
    }

    [Fact]
    public void DeepEmd_HalfKeepsKernelScalingAndSquaredCostsWide()
    {
        var algorithm = Uninitialized<DeepEMDAlgorithm<Half, Matrix<Half>, Vector<Half>>>();
        var options = Uninitialized<DeepEMDOptions<Half, Matrix<Half>, Vector<Half>>>();
        options.NumNodes = 1;
        options.SinkhornIterations = 3;
        options.SinkhornRegularization = 1;
        Field(algorithm, "_deepEmdOptions", options);
        var zero = new Vector<Half>(new[] { (Half)0 });
        Assert.Equal((Half)20.25, Assert.IsType<Half>(Call(algorithm, "ComputeEMDScore", zero, new Vector<Half>(new[] { (Half)4.5 }))));
        // exp(-25) underflows the plain Sinkhorn sums; one node per side still has to move its whole unit of mass,
        // so the transport cost is the squared distance 25.
        Assert.Equal((Half)25, Assert.IsType<Half>(Call(algorithm, "ComputeEMDScore", zero, new Vector<Half>(new[] { (Half)5 }))));
        options.NumNodes = 2;
        var extreme = new Vector<Half>(new[] { (Half)(-60000), (Half)60000 });
        Assert.Equal((Half)0, Assert.IsType<Half>(Call(algorithm, "ComputeEMDScore", extreme, extreme)));
        Assert.Equal((Half)0, Assert.IsType<Half>(Call(algorithm, "ComputeEMDScore", new Vector<Half>(0), extreme)));
    }

    [Fact]
    public void PublicCheckpoint_PreservesHalfEncoderParameters()
    {
        using var persistence = AiDotNet.Helpers.ModelPersistenceGuard.InternalOperation();
        var encoder = new HalfCheckpointEncoder();
        encoder.SetParameters(new Vector<Half>(new[] { (Half)0.75 }));
        var learner = new MAMLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new MAMLOptions<Half, Tensor<Half>, Tensor<Half>>(encoder));
        string path = System.IO.Path.GetTempFileName();
        try
        {
            learner.Save(path);
            var restored = new MAMLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
                new MAMLOptions<Half, Tensor<Half>, Tensor<Half>>(new HalfCheckpointEncoder()));
            restored.Load(path);
            Assert.Equal(new[] { (Half)0.75 }, restored.GetParameters().ToArray());
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }

    private static T Uninitialized<T>() => (T)FormatterServices.GetUninitializedObject(typeof(T));
    private static void Field(object target, string name, object value)
        => (target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException(name)).SetValue(target, value);
    private static object Call(object target, string name, params object[] arguments)
        => (target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException(name)).Invoke(target, arguments)
            ?? throw new InvalidOperationException(name);

    [Theory]
    [InlineData(7, 4)]
    [InlineData(131073, 20)]
    public void Bmaml_BlockedDistancesAndSvgdMatchScalarOracle(int dimensions, int count)
    {
        var algorithm = Uninitialized<BMAMLAlgorithm<double, Matrix<double>, Vector<double>>>();
        var options = Uninitialized<BMAMLOptions<double, Matrix<double>, Vector<double>>>();
        options.SVGDRepulsiveWeight = 0.7;
        Field(algorithm, "_algoOptions", options);
        Field(algorithm, "_paramDim", dimensions);
        var particles = new Vector<double>[count];
        var gradients = new Vector<double>[count];
        for (int j = 0; j < count; j++)
        {
            particles[j] = new Vector<double>(dimensions);
            gradients[j] = new Vector<double>(dimensions);
            for (int d = 0; d < dimensions; d++)
            {
                particles[j][d] = j * 0.01 + d % 5;
                gradients[j][d] = (j - d % 7) * 0.03;
            }
        }
        var distances = new List<double>();
        for (int i = 0; i < count; i++)
        for (int j = i + 1; j < count; j++)
        {
            double distance = 0;
            for (int d = 0; d < dimensions; d++)
                distance += Math.Pow(particles[j][d] - particles[i][d], 2);
            distances.Add(distance);
        }
        distances.Sort();
        double bandwidth = distances[distances.Count / 2] / Math.Max(Math.Log(count), 1);
        double actualBandwidth = Assert.IsType<double>(Call(algorithm, "ComputeMedianBandwidth", (object)particles));
        Assert.InRange(Math.Abs(actualBandwidth - bandwidth), 0, Math.Max(1, bandwidth) * 1e-10);
        var result = Assert.IsType<Vector<double>>(Call(algorithm, "ComputeSVGDUpdate", particles, gradients, 1, actualBandwidth));
        var kernels = new double[count];
        for (int j = 0; j < count; j++)
        {
            double distance = 0;
            for (int d = 0; d < dimensions; d++)
                distance += Math.Pow(particles[j][d] - particles[1][d], 2);
            kernels[j] = Math.Exp(-distance / (2 * actualBandwidth));
        }
        for (int d = 0; d < dimensions; d++)
        {
            double expected = 0;
            for (int j = 0; j < count; j++)
                expected -= kernels[j] * (gradients[j][d] + 0.7 / actualBandwidth * (particles[j][d] - particles[1][d])) / count;
            Assert.Equal(expected, result[d], 10);
        }
    }

    [Fact]
    public void Awgim_HalfAdamRetainsSmallMomentsAndFiniteZeroGradientUpdates()
    {
        var algorithm = Uninitialized<AWGIMAlgorithm<Half, Matrix<Half>, Tensor<Half>>>();
        var options = Uninitialized<AWGIMOptions<Half, Matrix<Half>, Tensor<Half>>>();
        options.WeightDecay = 0.01;
        Field(algorithm, "_awgimOptions", options);
        var parameters = new Vector<Half>(new[] { (Half)1, (Half)0 });
        var gradient = new Vector<Half>(new[] { (Half)0, (Half)0.0001 });
        var first = new Vector<double>(2);
        var second = new Vector<double>(2);
        double expected = 0;
        for (int step = 1; step <= 2; step++)
        {
            Field(algorithm, "_updates", step);
            double g = (double)gradient[1];
            double m = 0.9 * first[1] + 0.1 * g;
            double v = 0.999 * second[1] + 0.001 * g * g;
            expected = (double)(Half)(expected * 0.99 - 0.001 * (m / (1 - Math.Pow(0.9, step)))
                / (Math.Sqrt(v / (1 - Math.Pow(0.999, step))) + 1e-8));
            parameters = Assert.IsType<Vector<Half>>(Call(algorithm, "AdamW", parameters, gradient, first, second, 0.001));
            Assert.Equal((Half)expected, parameters[1]);
            Assert.True(Half.IsFinite(parameters[0]));
            Assert.InRange(second[1], 1e-12, 1e-9);
        }
        Assert.Equal((Half)((double)(Half)0.99 * 0.99), parameters[0]);
    }

    [Fact]
    public void Feat_RowBlockedAttentionMatchesTheScalarReference()
    {
        // n = 1100 exceeds one 2^20-element block, and the parameters run out partway through the second block,
        // so the block offsets and the fallback weight 0.01 are both exercised.
        const int n = 1100;
        int parameterCount = n * 600 + 37;
        var algorithm = Uninitialized<FEATAlgorithm<double, Matrix<double>, Vector<double>>>();
        var options = Uninitialized<FEATOptions<double, Matrix<double>, Vector<double>>>();
        options.NumTransformerLayers = 1;
        Field(algorithm, "_featOptions", options);
        var weights = new double[parameterCount];
        for (int k = 0; k < parameterCount; k++) weights[k] = 0.1 * Math.Cos(k);
        Field(algorithm, "_transformerParams", new Vector<double>(weights));
        var x = new double[n];
        for (int i = 0; i < n; i++) x[i] = 0.5 * Math.Sin(i);

        var expected = new double[n];
        for (int i = 0; i < n; i++)
        {
            double total = 0, sum = 0;
            for (int j = 0; j < n; j++)
            {
                long index = (long)i * n + j;
                double w = index < parameterCount ? weights[index] : 0.01;
                double weight = Math.Exp(Math.Min(x[i] * x[j] * w / Math.Sqrt(n), 10.0));
                total += weight;
                sum += x[j] * weight;
            }
            expected[i] = (total > 1e-10 ? sum / total : x[i]) + x[i];
        }
        double mean = expected.Average();
        double variance = expected.Select(v => (v - mean) * (v - mean)).Sum() / n;
        double std = Math.Sqrt(variance + 1e-5);

        var result = Assert.IsType<Vector<double>>(Call(algorithm, "AdaptPrototypes", new Vector<double>(x)));
        for (int i = 0; i < n; i++) Assert.Equal((expected[i] - mean) / std, result[i], 9);
    }

    [Fact]
    public void BayTransProto_FloatKlKeepsATinyMeanShift()
    {
        // KL(N(m, I) || N(0, I)) = 0.5 * |m|^2. With m = (1e-4, 0) the variance part is exactly zero, so the
        // squared shift 1e-8 must not be added to a running sum near D = 2, where float would drop it.
        var algorithm = Uninitialized<BayTransProtoAlgorithm<float, Matrix<float>, Vector<float>>>();
        Field(algorithm, "_logVar", new Vector<float>(2));
        var mean = new Vector<float>(new[] { 1e-4f, 0f });
        float kl = Assert.IsType<float>(Call(algorithm, "DiagonalGaussianKL", mean, new Vector<float>(2)));
        Assert.Equal(0.5f * (1e-4f * 1e-4f), kl);
    }

    [Fact]
    public void Dkt_FloatKernelOfLargeNormRowsNeverExceedsOne()
    {
        // The expanded |a|^2 + |b|^2 - 2 a.b can round below zero for float rows with large norms; clamped at
        // zero (scikit-learn euclidean_distances), every RBF entry stays in [0, 1] and the diagonal stays at 1 up
        // to the rounding of a nonnegative distance.
        var algorithm = Uninitialized<DKTAlgorithm<float, Matrix<float>, Vector<float>>>();
        Field(algorithm, "_kernelParams", new Vector<float>(2));
        const int rows = 6;
        const int width = 5;
        var values = new float[rows * width];
        for (int i = 0; i < values.Length; i++) values[i] = 1000f + 0.137f * i + 0.011f * (i % 3);
        var a = new Tensor<float>(values, new[] { rows, width });
        var mask = Tensor<float>.CreateDefault(new[] { rows, width }, 1f);
        var kernel = Assert.IsType<Tensor<float>>(Call(algorithm, "KernelMatrix", a, mask, a, mask));
        for (int i = 0; i < rows; i++)
        {
            Assert.InRange(kernel[i, i], 0.999f, 1f);
            for (int j = 0; j < rows; j++) Assert.InRange(kernel[i, j], 0f, 1f);
        }
    }

    [Fact]
    public void SetMetaModel_ParametersAndCheckpointReadTheReplacementModel()
    {
        using var persistence = AiDotNet.Helpers.ModelPersistenceGuard.InternalOperation();
        var first = new HalfCheckpointEncoder();
        first.SetParameters(new Vector<Half>(new[] { (Half)0.75 }));
        var learner = new MAMLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new MAMLOptions<Half, Tensor<Half>, Tensor<Half>>(first));
        Assert.Equal(new[] { (Half)0.75 }, learner.GetParameters().ToArray());

        var replacement = new HalfCheckpointEncoder();
        replacement.SetParameters(new Vector<Half>(new[] { (Half)0.25 }));
        learner.SetMetaModel(replacement);
        Assert.Equal(new[] { (Half)0.25 }, learner.GetParameters().ToArray());

        string path = System.IO.Path.GetTempFileName();
        try
        {
            learner.Save(path);
            var restored = new MAMLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
                new MAMLOptions<Half, Tensor<Half>, Tensor<Half>>(new HalfCheckpointEncoder()));
            restored.Load(path);
            Assert.Equal(new[] { (Half)0.25 }, restored.GetParameters().ToArray());
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }
}

internal sealed partial class HalfCheckpointEncoder : ModelBase<Half, Tensor<Half>, Tensor<Half>>
{
    [TrainableParameter]
    private Vector<Half> _weight = new(new[] { (Half)0.5 });

    public override ILossFunction<Half> DefaultLossFunction => new MeanSquaredErrorLoss<Half>();
    public override Tensor<Half> Predict(Tensor<Half> input) => Engine.TensorMultiplyScalar(input, _weight[0]);
    public override void Train(Tensor<Half> input, Tensor<Half> expectedOutput)
        => throw new NotSupportedException("Checkpoint-only fixture.");
    public override IFullModel<Half, Tensor<Half>, Tensor<Half>> WithParameters(Vector<Half> parameters)
    {
        var copy = new HalfCheckpointEncoder();
        copy.SetParameters(parameters);
        return copy;
    }
}
#endif
