using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Fixtures;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

[Collection(EngineCurrentGlobalStateCollection.Name)]
public class InstantNgpHashGridApiTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void FractionalGridSchedule_PreservesForwardAndTableGradients(int levels)
    {
        var previous = AiDotNetEngine.Current;
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            using var model = new InstantNGP<double>(new InstantNGPOptions<double>
            {
                HashTableSize = 16, NumLevels = levels, FeaturesPerLevel = 2,
                CoarsestResolution = 3, FinestResolution = 10,
                MlpHiddenDim = 8, MlpNumLayers = 1, FeatureDim = 4,
                ColorHiddenDim = 4, ColorNumLayers = 1,
                OccupancyGridResolution = 4, UseOccupancyGrid = false, Seed = 3,
            });
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var tables = Assert.IsType<Dictionary<int, Tensor<double>>>(
                typeof(InstantNGP<double>).GetField("_hashTables", flags)!.GetValue(model));
            var encode = typeof(InstantNGP<double>).GetMethod("MultiresolutionHashEncoding", flags)!;
            var positions = new Tensor<double>(new[] { 2, 3 });
            double[] points = { 0.13, 0.37, 0.81, 0.69, 0.23, 0.47 };
            for (int i = 0; i < points.Length; i++) positions[i] = points[i];
            var sources = Enumerable.Range(0, levels).Select(i => tables[i]).ToArray();
            foreach (var table in sources)
                for (int i = 0; i < table.Length; i++) table[i] = Math.Sin(i + 1);
            Tensor<double> Encode() => Assert.IsType<Tensor<double>>(encode.Invoke(model, new object[] { positions }));
            var expected = engine.MultiresolutionHashEncoding(positions, sources,
                levels == 1 ? 10 : 3, levels == 1 ? 1.0 : Math.Sqrt(10.0 / 3), 2);
            var actual = Encode();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 12);

            Dictionary<Tensor<double>, Tensor<double>> gradients;
            using (var tape = new GradientTape<double>())
            {
                var loss = engine.ReduceSum(Encode(), new[] { 0, 1 }, keepDims: false);
                gradients = tape.ComputeGradients(loss, sources);
            }
            double Sum() => Encode().ToArray().Sum();
            const double epsilon = 1e-5;
            foreach (var table in sources)
            {
                Assert.True(gradients.ContainsKey(table), "Every hash level must receive a gradient.");
                Assert.Contains(gradients[table].ToArray(), value => Math.Abs(value) > 1e-8);
                for (int i = 0; i < table.Length; i++)
                {
                    double original = table[i];
                    table[i] = original + epsilon;
                    double plus = Sum();
                    table[i] = original - epsilon;
                    double minus = Sum();
                    table[i] = original;
                    Assert.True(Math.Abs(gradients[table][i] - (plus - minus) / (2 * epsilon)) < 1e-8,
                        $"Hash table entry {i} has an incorrect gradient.");
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
