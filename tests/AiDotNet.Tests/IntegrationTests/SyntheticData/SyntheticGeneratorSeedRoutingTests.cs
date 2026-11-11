using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

public class SyntheticGeneratorSeedRoutingTests
{
    [Fact(Timeout = 120000)]
    public async Task CtabGradientPenaltyChangesTrainedCritic()
    {
        await Task.Yield();
        double[] Fit(double penalty)
        {
            var options = new CTABGANPlusOptions<double> { Seed = 7 };
            SyntheticGanOptionFixture.PrepareOptions(options, "GradientPenaltyWeight");
            options.GradientPenaltyWeight = penalty;
            using var model = new CTABGANPlusGenerator<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
                NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 3) { RandomSeed = 7 }, options);
            SyntheticGanOptionFixture.Train(model);
            var layers = (IEnumerable<AiDotNet.Interfaces.ILayer<double>>)model.GetType().GetField("_discLayers",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;
            return layers.SelectMany(layer => layer.GetParameters()).ToArray();
        }
        var control = Fit(10);
        Assert.Equal(control, Fit(10));
        Assert.False(control.SequenceEqual(Fit(20)));
    }

    [Fact]
    public void NotearsGradientMatchesFiniteDifferencesForAsymmetricGraph()
    {
        using var model = new CausalGANGenerator<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 3) { RandomSeed = 7 }, new CausalGANOptions<double> { Seed = 7 });
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var type = model.GetType();
        var adjacency = new Matrix<double>(3, 3);
        adjacency[0, 1] = .3; adjacency[1, 0] = .7;
        adjacency[1, 2] = .2; adjacency[2, 0] = .4;
        type.GetField("_adjacency", flags)!.SetValue(model, adjacency);
        type.GetField("_dataWidth", flags)!.SetValue(model, 3);
        var method = type.GetMethod("ComputeNOTEARSConstraintAndGradient", flags)!;
        (double, double[,]) Evaluate() => ((double, double[,]))method.Invoke(model, null)!;
        var (_, gradient) = Evaluate();
        const double epsilon = 1e-5;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        {
            if (i == j) continue;
            double value = adjacency[i, j];
            adjacency[i, j] = value + epsilon;
            double plus = Evaluate().Item1;
            adjacency[i, j] = value - epsilon;
            double minus = Evaluate().Item1;
            adjacency[i, j] = value;
            Assert.InRange(Math.Abs(gradient[i, j] - (plus - minus) / (2 * epsilon)), 0, 1e-7);
        }
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task CtabSynchronousAndAsynchronousFitsUpdateGeneratorAndClassifierIdentically()
    {
        CTABGANPlusGenerator<double> Create(int epochs, bool zeroLearningRate = false)
        {
            var options = new CTABGANPlusOptions<double> { Seed = 7 };
            SyntheticGanOptionFixture.PrepareOptions(options, "LearningRate");
            options.Epochs = epochs;
            if (zeroLearningRate) options.LearningRate = 0;
            return new CTABGANPlusGenerator<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
                NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 3) { RandomSeed = 7 }, options);
        }
        using var untrained = Create(1, zeroLearningRate: true);
        using var synchronous = Create(1);
        using var asynchronous = Create(1);
        SyntheticGanOptionFixture.Train(untrained);
        SyntheticGanOptionFixture.Train(synchronous);
        var (data, columns) = SyntheticGanOptionFixture.Data<double>();
        await asynchronous.FitAsync(data, columns);
        var trained = synchronous.GetParameters().ToArray();
        Assert.False(untrained.GetParameters().SequenceEqual(trained));
        Assert.Equal(trained, asynchronous.GetParameters().ToArray());
        double[] Classifier(CTABGANPlusGenerator<double> model) => ((AiDotNet.NeuralNetworks.Layers.LayerBase<double>)model.GetType()
            .GetField("_classifierHead", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(model)!).GetParameters().ToArray();
        Assert.False(Classifier(untrained).SequenceEqual(Classifier(synchronous)));
        Assert.Equal(Classifier(synchronous), Classifier(asynchronous));
    }

    [Theory]
    [InlineData("CTAB", 1)]
    [InlineData("CTAB", 99)]
    [InlineData("Copula", 1)]
    [InlineData("Copula", 99)]
    [InlineData("Causal", 1)]
    [InlineData("Causal", 99)]
    public void FittedSchemaOwnsGeneratedColumnsIndependentlyOfRiskMetadata(string kind, int metadata)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 3) { RandomSeed = 7 };
        ModelOptions options = kind switch
        {
            "CTAB" => new CTABGANPlusOptions<double>(),
            "Copula" => new CopulaGANOptions<double>(),
            _ => new CausalGANOptions<double>(),
        };
        SyntheticGanOptionFixture.PrepareOptions(options, "NumFeatures");
        var shared = (SyntheticDataGeneratorOptions<double>)options;
        shared.NumFeatures = shared.TimeHorizon = metadata;
        shared.ConfidenceLevel = metadata == 1 ? .5 : .99;
        shared.EncoderLayerCount = metadata;
        options.Seed = 7;
        options.GetType().GetProperty("Epochs")!.SetValue(options, 1);
        using NeuralNetworkBase<double> model = kind switch
        {
            "CTAB" => new CTABGANPlusGenerator<double>(architecture, (CTABGANPlusOptions<double>)options),
            "Copula" => new CopulaGANGenerator<double>(architecture, (CopulaGANOptions<double>)options),
            _ => new CausalGANGenerator<double>(architecture, (CausalGANOptions<double>)options),
        };
        SyntheticGanOptionFixture.Train(model);
        var rows = ((AiDotNet.Interfaces.ISyntheticTabularGenerator<double>)model).Generate(4);
        Assert.Equal(4, rows.Rows);
        Assert.Equal(3, rows.Columns);
    }

    [Theory]
    [InlineData("CTAB", 8, 16)]
    [InlineData("CTAB", 256, 128)]
    [InlineData("Copula", 8, 16)]
    [InlineData("Copula", 256, 128)]
    [InlineData("Causal", 8, 16)]
    [InlineData("Causal", 256, 128)]
    public void ParameterInspectionPreservesGeneratorNormalizationWidths(string kind, int firstWidth, int secondWidth)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 3) { RandomSeed = 7 };
        var widths = new[] { firstWidth, secondWidth };
        using NeuralNetworkBase<double> model = kind switch
        {
            "CTAB" => new CTABGANPlusGenerator<double>(architecture, new CTABGANPlusOptions<double>
                { EmbeddingDimension = 4, GeneratorDimensions = widths, DiscriminatorDimensions = new[] { 8, 8 }, Seed = 7 }),
            "Copula" => new CopulaGANGenerator<double>(architecture, new CopulaGANOptions<double>
                { EmbeddingDimension = 4, GeneratorDimensions = widths, DiscriminatorDimensions = new[] { 8, 8 }, Seed = 7 }),
            _ => new CausalGANGenerator<double>(architecture, new CausalGANOptions<double>
                { EmbeddingDimension = 4, HiddenDimensions = widths, Seed = 7 }),
        };
        ModelOptions held = model switch
        {
            CTABGANPlusGenerator<double> ctab => ctab.Options,
            CopulaGANGenerator<double> copula => copula.Options,
            CausalGANGenerator<double> causal => causal.Options,
            _ => throw new System.InvalidOperationException(),
        };
        Assert.Same(held, model.GetOptions());
        Assert.NotEmpty(model.GetParameters());
        var input = new Tensor<double>(new[] { 1, kind == "Causal" ? 4 : 8 });
        for (int i = 0; i < input.Length; i++) input[i] = .1 * (i + 1);
        var output = model.Predict(input);
        Assert.Equal(3, output.Length);
        Assert.Equal(output.ToArray(), model.Predict(input).ToArray());
        Assert.All(output, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SeededRebuildsRetainTheirOwnOptionsAndLayerSequence(bool pate)
    {
        var data = new Matrix<double>(20, 1);
        for (int i = 0; i < data.Rows; i++) data[i, 0] = i / 20.0;
        var columns = new List<ColumnMetadata> { new("value", ColumnDataType.Continuous, columnIndex: 0) };

        (NeuralSyntheticTabularGeneratorBase<double> Model, ModelOptions Options) Create(int seed)
        {
            var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
                NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 1);
            if (pate)
            {
                var options = new PATEGANOptions<double> { Seed = seed, NumTeachers = 2 };
                return (new PATEGANGenerator<double>(architecture, options), options);
            }
            var octOptions = new OCTGANOptions<double>
            {
                Seed = seed, EmbeddingDimension = 4, GeneratorDimensions = new[] { 8, 8 },
                DiscriminatorDimensions = new[] { 8, 8 }, VGMModes = 2,
            };
            return (new OCTGANGenerator<double>(architecture, octOptions), octOptions);
        }

        void Fit(NeuralSyntheticTabularGeneratorBase<double> model)
        {
            if (model is OCTGANGenerator<double> oct) oct.Fit(data, columns, epochs: 0);
            else ((PATEGANGenerator<double>)model).Fit(data, columns, epochs: 0);
        }

        var first = Create(7);
        var second = Create(7);
        Assert.Same(first.Options, first.Model.GetOptions());
        Assert.Same(second.Options, second.Model.GetOptions());
        Fit(first.Model);
        var distractor = Create(8);
        Fit(distractor.Model);
        Fit(second.Model);
        var parameters = first.Model.GetParameters().ToArray();
        Assert.NotEmpty(parameters);
        Assert.Equal(parameters, second.Model.GetParameters().ToArray());
    }
}
