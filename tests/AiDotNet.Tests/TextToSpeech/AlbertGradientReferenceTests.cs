using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class AlbertGradientReferenceTests
{
    [Fact]
    public void OriginalSharedAlbert_MatchesAll23Gradients_AndPreservesLoadedPaddingValuesWithoutTrainingThem()
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "albert_gradient_reference.json")));
        var root = document.RootElement; var entries = root.GetProperty("parameters");
        static double[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var engine = AiDotNetEngine.Current; var layers = new List<LayerBase<double>>();
        var model = new AlbertNetwork<double>(engine, layers, new AlbertConfiguration
        {
            VocabSize = 17, EmbeddingSize = 4, Hidden = 8, Layers = 3, Heads = 2,
            Intermediate = 16, MaxPositions = 16, Dropout = 0, PadTokenId = 0
        });
        model.LoadTorchWeights((name, shape) =>
        {
            var entry = entries.GetProperty(name);
            Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
            return Values(entry.GetProperty("values"));
        });
        double[] Convert(string name, string field)
        {
            var entry = entries.GetProperty(name); var values = Values(entry.GetProperty(field));
            var shape = entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            if (shape.Length != 2 || name.Contains("_embeddings.")) return values;
            var transposed = new double[values.Length];
            for (int o = 0; o < shape[0]; o++) for (int i = 0; i < shape[1]; i++) transposed[i*shape[0]+o] = values[o*shape[1]+i];
            return transposed;
        }
        var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(23, owned.Count); var remaining = entries.EnumerateObject().Select(x => x.Name).ToList();
        var mapping = new Dictionary<Tensor<double>, string>();
        foreach (var parameter in owned)
        {
            var matches = remaining.Where(name => Convert(name,"values").Length == parameter.Length &&
                Convert(name,"values").SequenceEqual(parameter.ToArray())).ToArray();
            Assert.Single(matches); mapping.Add(parameter,matches[0]); remaining.Remove(matches[0]);
        }
        Assert.Empty(remaining);
        using var tape = new GradientTape<double>();
        var result = model.Encode(root.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray(),false,new Random(1));
        void Close(double[] actual,double[] expected,string name)
        {
            Assert.Equal(expected.Length,actual.Length);
            double error = actual.Zip(expected,(a,b) => Math.Abs(a-b)).Max();
            Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error < 2e-9,name+" error "+error.ToString("R"));
        }
        Close(result.ToArray(),Values(root.GetProperty("output")),"ALBERT output");
        var probe = new Tensor<double>(result.Shape.ToArray(),new Vector<double>(Values(root.GetProperty("probe"))));
        var loss = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(result,probe),new[] {0,1},true),new[] {1});
        Close(loss.ToArray(),new[] {root.GetProperty("loss").GetDouble()},"ALBERT loss");
        var gradients = tape.ComputeGradients(loss,sources:owned);
        foreach (var parameter in owned) Close(gradients[parameter].ToArray(),Convert(mapping[parameter],"gradient"),mapping[parameter]);
        Assert.All(gradients[model.WordEmbeddings.Table].ToArray().Take(4),value => Assert.Equal(0,value));
        Assert.Contains(model.WordEmbeddings.Table.ToArray().Take(4),value => value != 0);
    }
}
