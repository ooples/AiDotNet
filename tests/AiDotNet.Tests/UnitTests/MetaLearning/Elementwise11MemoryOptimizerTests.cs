#if NET5_0_OR_GREATER
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

public class Elementwise11MemoryOptimizerTests
{
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
#endif
