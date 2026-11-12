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
        // Below the existing stability threshold, the marginal is retained rather than divided by a tiny sum.
        Assert.Equal((Half)0, Assert.IsType<Half>(Call(algorithm, "ComputeEMDScore", zero, new Vector<Half>(new[] { (Half)5 }))));
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
