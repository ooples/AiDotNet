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
