using System;
using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// #2147: an architecture that disagrees with the sizes a financial model was built from fails, naming both, instead of
/// being silently ignored; one that agrees, or states nothing, changes nothing.
/// </summary>
public class ArchitectureAgreementTests
{
    private static NeuralNetworkArchitecture<double> Grid(int height, int width) =>
        new(InputType.TwoDimensional, NeuralNetworkTaskType.Regression, inputHeight: height, inputWidth: width);

    private static NeuralNetworkArchitecture<double> Flat(int inputSize) =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: inputSize);

    private static void Agrees(NeuralNetworkArchitecture<double> architecture, int sequenceLength, int numFeatures) =>
        Assert.Null(Record.Exception(() => ArchitectureAgreement.RequireSequenceInput(architecture, "M", sequenceLength, numFeatures)));

    private static ArgumentException Refuses(NeuralNetworkArchitecture<double> architecture, int sequenceLength, int numFeatures) =>
        Assert.Throws<ArgumentException>(() => ArchitectureAgreement.RequireSequenceInput(architecture, "M", sequenceLength, numFeatures));

    [Fact]
    public void An_architecture_stating_no_input_size_is_never_refused()
    {
        // A three-dimensional architecture without spatial sizes leaves them lazy (-1): nothing is stated.
        Agrees(new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression), 96, 7);
    }

    [Fact]
    public void A_grid_agrees_only_when_both_axes_match()
    {
        Agrees(Grid(96, 7), 96, 7);

        var wrongFeatures = Refuses(Grid(96, 32), 96, 7);
        Assert.Contains("96 x 32", wrongFeatures.Message);
        Assert.Contains("96 x 7", wrongFeatures.Message);
        Refuses(Grid(48, 7), 96, 7);

        // As many values as the window, in a different grid: still a different input.
        var sameCount = Refuses(Grid(48, 14), 96, 7);
        Assert.Contains("48 x 14", sameCount.Message);
    }

    [Fact]
    public void A_one_by_n_grid_is_read_as_its_flat_size()
    {
        // 1 x n is the grid an architecture infers from a flat n: it agrees as the feature count, the sequence length
        // or the window, and as nothing else.
        Agrees(Grid(1, 7), 96, 7);
        Agrees(Grid(1, 96), 96, 7);
        Agrees(Grid(1, 96 * 7), 96, 7);

        var ex = Refuses(Grid(1, 32), 96, 7);
        Assert.Contains("1 x 32", ex.Message);
    }

    [Fact]
    public void A_square_grid_is_read_as_its_flat_size()
    {
        // 28 x 28 is the grid an architecture infers from 784: it agrees with a 784 x 1 or a 98 x 8 window.
        Agrees(Grid(28, 28), 784, 1);
        Agrees(Grid(28, 28), 98, 8);

        var ex = Refuses(Grid(28, 28), 96, 7);
        Assert.Contains("28 x 28", ex.Message);
    }

    [Fact]
    public void A_grid_with_more_than_one_channel_never_matches_a_sequence_window()
    {
        var channels = new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            inputHeight: 96, inputWidth: 7, inputDepth: 2);

        var ex = Refuses(channels, 96, 7);
        Assert.Contains("96 x 7 x 2", ex.Message);
    }

    [Fact]
    public void A_flat_size_agrees_as_the_feature_count_the_sequence_length_or_the_whole_window()
    {
        // A two-dimensional architecture given only a size infers its own 1 x n grid; that still agrees.
        Agrees(new NeuralNetworkArchitecture<double>(InputType.TwoDimensional, NeuralNetworkTaskType.Regression, inputSize: 96), 96, 7);
        Agrees(Flat(7), 96, 7);
        Agrees(Flat(96), 96, 7);
        Agrees(Flat(96 * 7), 96, 7);

        var ex = Refuses(Flat(32), 96, 7);
        Assert.Contains("InputSize of 32", ex.Message);
        Assert.Contains("96 x 7", ex.Message);
    }

    [Fact]
    public void The_feature_count_of_a_stated_grid_is_its_width()
    {
        // A 96 x 7 forecasting architecture records InputSize 672, the whole window; its features are its 7 columns.
        Assert.Equal(96 * 7, Grid(96, 7).InputSize);
        Assert.Equal(7, ArchitectureAgreement.SequenceFeatureCount(Grid(96, 7)));
        Assert.Equal(7, ArchitectureAgreement.SequenceFeatureCount(Flat(7)));
        // The inferred forms carry no split: their flat size stays the feature count.
        Assert.Equal(7, ArchitectureAgreement.SequenceFeatureCount(Grid(1, 7)));
        Assert.Equal(784, ArchitectureAgreement.SequenceFeatureCount(Grid(28, 28)));
    }

    [Fact]
    public void Autoformer_takes_its_features_from_the_width_of_a_stated_grid()
    {
        var options = new AutoformerOptions<double>
        {
            LookbackWindow = 8, ForecastHorizon = 2, EmbeddingDim = 8, NumEncoderLayers = 1, NumDecoderLayers = 1,
            NumAttentionHeads = 2, MovingAverageKernel = 3,
        };
        var model = new Autoformer<double>(Grid(8, 3), options);

        var output = model.Predict(new Tensor<double>(new[] { 1, 8, 3 }));

        Assert.Equal(new[] { 1, 2, 3 }, output.Shape.ToArray());
        AssertFinite(output);
    }

    private static FEDformerOptions<double> SmallFedformer() => new()
    {
        SequenceLength = 8,
        LabelLength = 4,
        PredictionHorizon = 2,
        NumFeatures = 3,
        ModelDimension = 16,
        NumHeads = 8,
        FeedForwardDimension = 16,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        NumModes = 2,
        MovingAverageKernel = 3,
    };

    private static void AssertFinite(Tensor<double> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
            Assert.True(!double.IsNaN(tensor[i]) && !double.IsInfinity(tensor[i]), $"output[{i}] is {tensor[i]}.");
    }

    [Fact]
    public void A_model_built_from_options_refuses_an_architecture_describing_another_input()
    {
        var model = new FEDformer<double>(Grid(8, 5), options: SmallFedformer());
        var input = new Tensor<double>(new[] { 1, 8, 3 });

        var ex = Assert.Throws<ArgumentException>(() => model.Predict(input));
        Assert.Contains("FEDformer", ex.Message);
        Assert.Contains("8 x 5", ex.Message);
        Assert.Contains("8 x 3", ex.Message);
    }

    [Fact]
    public void Forecast_and_Train_refuse_an_architecture_describing_another_input()
    {
        var input = new Tensor<double>(new[] { 1, 8, 3 });

        // FEDformer overrides Forecast, so this proves the override checks too, not only the base.
        var forecast = Assert.Throws<ArgumentException>(
            () => new FEDformer<double>(Grid(8, 5), options: SmallFedformer()).Forecast(input));
        Assert.Contains("8 x 5", forecast.Message);

        var model = new FEDformer<double>(Grid(8, 5), options: SmallFedformer());
        var target = new Tensor<double>(new[] { 1, 2, 3 });
        var train = Assert.Throws<ArgumentException>(() => model.Train(input, target));
        Assert.Contains("8 x 3", train.Message);
    }

    [Fact]
    public void Every_model_s_Forecast_runs_the_check_not_only_the_ones_that_called_it()
    {
        // TimesNet's Forecast never called the check; models now override ForecastCore, and Forecast checks first.
        var options = new TimesNetOptions<double>
        {
            SequenceLength = 8, PredictionHorizon = 2, NumFeatures = 3, ModelDimension = 8, FeedForwardDimension = 16,
            NumLayers = 1, TopK = 2,
        };
        var model = new TimesNet<double>(Grid(8, 5), options);

        var ex = Assert.Throws<ArgumentException>(() => model.Forecast(new Tensor<double>(new[] { 1, 8, 3 })));
        Assert.Contains("TimesNet", ex.Message);
        Assert.Contains("8 x 5", ex.Message);
        var forecast = typeof(AiDotNet.Finance.Base.FinancialModelBase<double>).GetMethod("Forecast");
        // It implements IFinancialModel.Forecast, so the runtime marks it virtual - and final: no model can override it.
        Assert.True(forecast is not null && (!forecast.IsVirtual || forecast.IsFinal),
            "Forecast is the checking template; models override ForecastCore.");
    }

    [Fact]
    public void A_model_with_its_own_input_layout_is_checked_against_its_own_sizes()
    {
        // DCRNN reads [nodes, steps, features per node], and its NumFeatures is nodes x features per node: an InputSize
        // of 2 nodes x 2 steps x 1 feature = 4 is its whole window and agrees. Reading the contract's axes as
        // [batch, steps, features] saw a 2 x 1 window and refused it.
        var options = new DCRNNOptions<double>
        {
            NumNodes = 2, NumFeatures = 1, SequenceLength = 2, ForecastHorizon = 2, HiddenDimension = 2,
            NumEncoderLayers = 1, NumDecoderLayers = 1, DiffusionSteps = 1, MaxDiffusionStep = 1, NumSamples = 1, Seed = 42,
        };
        using var agrees = new AiDotNet.Finance.Graph.DCRNN<double>(Flat(4), options);
        Assert.Null(Record.Exception(() => agrees.Predict(new Tensor<double>(new[] { 2, 2, 1 }))));

        using var disagrees = new AiDotNet.Finance.Graph.DCRNN<double>(Flat(5), options);
        var ex = Assert.Throws<ArgumentException>(() => disagrees.Predict(new Tensor<double>(new[] { 2, 2, 1 })));
        Assert.Contains("InputSize of 5", ex.Message);
    }

    [Fact]
    public void A_model_whose_architecture_agrees_predicts_as_before()
    {
        var model = new FEDformer<double>(Grid(8, 3), options: SmallFedformer());

        var output = model.Predict(new Tensor<double>(new[] { 1, 8, 3 }));

        // [batch, PredictionHorizon, NumFeatures].
        Assert.Equal(new[] { 1, 2, 3 }, output.Shape.ToArray());
        AssertFinite(output);
    }
}
