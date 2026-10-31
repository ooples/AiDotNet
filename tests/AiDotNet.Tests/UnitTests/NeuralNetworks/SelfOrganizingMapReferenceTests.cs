using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The shared self-organizing map against MiniSom configured with SOM_PAK's schedules
/// (tools/reference-data/som_reference.py), and its hexagonal lattice and bubble kernel against SOM_PAK's definitions.
/// </summary>
public class SelfOrganizingMapReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "som_reference.json";
        string relative = Path.Combine("UnitTests", "NeuralNetworks", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 60000)]
    public async Task Map_MatchesMiniSomWithSomPakSchedules()
    {
        await Task.Yield();
        var k = Fixture();
        int rows = k.GetProperty("rows").GetInt32(), columns = k.GetProperty("columns").GetInt32();
        int dim = k.GetProperty("dim").GetInt32(), steps = k.GetProperty("steps").GetInt32();
        var data = Values(k.GetProperty("data"));
        var codebook = new Tensor<double>(new[] { rows * columns, dim }, new Vector<double>(Values(k.GetProperty("initial"))));
        var map = new SelfOrganizingMapCore<double>(new SelfOrganizingMapConfiguration(columns, rows, dim, steps,
            k.GetProperty("learning_rate").GetDouble(), k.GetProperty("radius").GetDouble()), codebook);
        var winners = k.GetProperty("winners").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        int samples = data.Length / dim;
        for (int t = 0; t < steps; t++)
        {
            var x = data.AsSpan((t % samples) * dim, dim).ToArray();
            Assert.Equal(winners[t], map.Step(x, t).Winner);
        }
        var final = Values(k.GetProperty("final"));
        for (int i = 0; i < final.Length; i++)
            Assert.True(Math.Abs(final[i] - map.Codebook[i]) < 1e-12, $"codebook[{i}]: {final[i]} vs {map.Codebook[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task NetworkModel_TrainsTheSharedMap()
    {
        await Task.Yield();
        var k = Fixture();
        int dim = k.GetProperty("dim").GetInt32(), steps = k.GetProperty("steps").GetInt32();
        var architecture = new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Clustering, inputSize: dim, outputSize: 20);
        var som = new SelfOrganizingMap<double>(architecture, options: new SelfOrganizingMapNNOptions
        {
            TotalEpochs = steps, InitialLearningRate = k.GetProperty("learning_rate").GetDouble(),
            InitialRadius = k.GetProperty("radius").GetDouble(),
        });
        // 20 neurons lay out as 4 rows of 5, the fixture's grid.
        som.SetParameters(new Vector<double>(Values(k.GetProperty("initial"))));
        var data = Values(k.GetProperty("data"));
        int samples = data.Length / dim;
        var winners = k.GetProperty("winners").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        for (int t = 0; t < steps; t++)
        {
            var x = new Tensor<double>(new[] { dim }, new Vector<double>(data.AsSpan((t % samples) * dim, dim).ToArray()));
            Assert.Equal(winners[t], Array.IndexOf(som.Predict(x).ToArray(), 1.0));
            som.Train(x, x);
        }
        var final = Values(k.GetProperty("final"));
        var parameters = som.GetParameters();
        for (int i = 0; i < final.Length; i++)
            Assert.True(Math.Abs(final[i] - parameters[i]) < 1e-12, $"codebook[{i}]: {final[i]} vs {parameters[i]}");
    }

    /// <summary>SOM_PAK's <c>hexa_dist</c> (som_rout.c), transcribed: odd rows sit half a unit right, rows √0.75 apart.</summary>
    private static double SomPakHexagonalDistance(int bx, int by, int tx, int ty)
    {
        double diff = bx - tx;
        if (((by - ty) % 2) != 0)
        {
            if ((by % 2) == 0) diff -= 0.5;
            else diff += 0.5;
        }
        double ret = diff * diff;
        diff = by - ty;
        ret += 0.75 * diff * diff;
        return Math.Sqrt(ret);
    }

    [Fact(Timeout = 60000)]
    public async Task HexagonalLattice_IsSomPaks_AndBubbleIsTheRadiusDisc()
    {
        await Task.Yield();
        const int columns = 5, rows = 4;
        var map = new SelfOrganizingMapCore<double>(new SelfOrganizingMapConfiguration(columns, rows, 2, 10, 0.5,
            Topology: SOMTopology.Hexagonal, Neighborhood: NeighborhoodFunction.Bubble));
        for (int a = 0; a < rows * columns; a++)
            for (int b = 0; b < rows * columns; b++)
            {
                double expected = SomPakHexagonalDistance(a % columns, a / columns, b % columns, b / columns);
                Assert.True(Math.Abs(expected - map.GridDistance(a, b)) < 1e-12, $"d({a}, {b})");
            }
        // An interior neuron has exactly six neighbours at distance 1.
        int interior = 1 * columns + 2;
        Assert.Equal(6, Enumerable.Range(0, rows * columns).Count(n => n != interior && Math.Abs(map.GridDistance(interior, n) - 1) < 1e-12));
        Assert.Equal(1.0, map.Neighborhood(1.5, 1.5));
        Assert.Equal(0.0, map.Neighborhood(1.5000001, 1.5));
        Assert.Equal(0.0, map.LearningRate(10));
        Assert.Equal(1.0, map.Radius(10));
    }
}
