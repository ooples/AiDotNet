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

    [Fact]
    public void An_architecture_stating_no_input_size_is_never_refused()
    {
        // A three-dimensional architecture without spatial sizes leaves them lazy (-1): nothing is stated.
        ArchitectureAgreement.RequireSequenceInput(
            new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression), "M", 96, 7);
    }

    [Fact]
    public void A_grid_agrees_only_when_both_axes_match()
    {
        ArchitectureAgreement.RequireSequenceInput(Grid(96, 7), "M", 96, 7);

        var wrongFeatures = Assert.Throws<ArgumentException>(() => ArchitectureAgreement.RequireSequenceInput(Grid(96, 32), "M", 96, 7));
        Assert.Contains("96 x 32", wrongFeatures.Message);
        Assert.Contains("96 x 7", wrongFeatures.Message);
        Assert.Throws<ArgumentException>(() => ArchitectureAgreement.RequireSequenceInput(Grid(48, 7), "M", 96, 7));

        // As many values as the window, in a different grid: still a different input.
        var sameCount = Assert.Throws<ArgumentException>(() => ArchitectureAgreement.RequireSequenceInput(Grid(48, 14), "M", 96, 7));
        Assert.Contains("48 x 14", sameCount.Message);
    }

    [Fact]
    public void A_flat_size_agrees_as_the_feature_count_the_sequence_length_or_the_whole_window()
    {
        // A two-dimensional architecture given only a size infers its own 1 x n grid; that still agrees.
        ArchitectureAgreement.RequireSequenceInput(
            new NeuralNetworkArchitecture<double>(InputType.TwoDimensional, NeuralNetworkTaskType.Regression, inputSize: 96),
            "M", 96, 7);
        ArchitectureAgreement.RequireSequenceInput(Flat(7), "M", 96, 7);
        ArchitectureAgreement.RequireSequenceInput(Flat(96), "M", 96, 7);
        ArchitectureAgreement.RequireSequenceInput(Flat(96 * 7), "M", 96, 7);

        var ex = Assert.Throws<ArgumentException>(() => ArchitectureAgreement.RequireSequenceInput(Flat(32), "M", 96, 7));
        Assert.Contains("InputSize of 32", ex.Message);
        Assert.Contains("96 x 7", ex.Message);
    }

    private static FEDformerOptions<double> SmallFedformer() => new()
    {
        SequenceLength = 8,
        LabelLength = 4,
        PredictionHorizon = 2,
        NumFeatures = 3,
        ModelDimension = 8,
        NumHeads = 2,
        FeedForwardDimension = 16,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        NumModes = 2,
        MovingAverageKernel = 3,
    };

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
    public void A_model_whose_architecture_agrees_predicts_as_before()
    {
        var model = new FEDformer<double>(Grid(8, 3), options: SmallFedformer());
        var output = model.Predict(new Tensor<double>(new[] { 1, 8, 3 }));
        Assert.True(output.Length > 0);
    }
}
