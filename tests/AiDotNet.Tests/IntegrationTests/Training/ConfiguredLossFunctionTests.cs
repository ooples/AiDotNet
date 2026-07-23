using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that a configured loss function actually reaches training, and that models whose loss is
/// intrinsic to their architecture do not pretend to accept one.
/// </summary>
/// <remarks>
/// ConfigureLossFunction used to store its argument in a field nothing read, so the caller's choice
/// was silently discarded. Wiring the field alone is not enough either: the facade's loss is only
/// used to report an epoch metric, while gradients come from the model's own loss — so these tests
/// assert against the MODEL's loss, which is what training actually optimizes.
/// </remarks>
public class ConfiguredLossFunctionTests
{
    [Fact(Timeout = 60000)]
    public async Task PointForecaster_AdoptsConfiguredLoss()
    {
        var model = new NBeatsProbe(new NBEATSModelOptions<double> { LookbackWindow = 4, ForecastHorizon = 1 });

        // Default before configuration.
        Assert.IsType<MeanSquaredErrorLoss<double>>(model.DefaultLossFunction);

        ((ISupportsLossFunction<double>)model).SetLossFunction(new MeanAbsoluteErrorLoss<double>());

        // The loss the model trains against — not merely a facade-side reporting variable.
        Assert.IsType<MeanAbsoluteErrorLoss<double>>(model.DefaultLossFunction);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task ConfiguredLoss_IsMirroredOntoOptions()
    {
        var options = new NBEATSModelOptions<double> { LookbackWindow = 4, ForecastHorizon = 1 };
        var model = new NBeatsProbe(options);

        var mae = new MeanAbsoluteErrorLoss<double>();
        ((ISupportsLossFunction<double>)model).SetLossFunction(mae);

        // Options and DefaultLossFunction must not disagree about which loss is in use.
        Assert.Same(mae, options.LossFunction);
        Assert.Same(mae, model.DefaultLossFunction);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task NonTapeLoss_IsRejectedUpFront()
    {
        var model = new NBeatsProbe(new NBEATSModelOptions<double> { LookbackWindow = 4, ForecastHorizon = 1 });

        // Tape training needs ComputeTapeLoss, which lives on LossFunctionBase<T>. Rejecting here
        // turns a failure deep in the first backward pass into an immediate, legible error.
        var ex = Assert.Throws<ArgumentException>(
            () => ((ISupportsLossFunction<double>)model).SetLossFunction(new NotATapeLoss()));
        Assert.Contains("LossFunctionBase", ex.Message);
        await Task.CompletedTask;
    }

    /// <summary>
    /// The three models whose objective is dictated by their architecture must not advertise the
    /// capability at all — unsupported is a fact of the type, not a runtime surprise.
    /// </summary>
    [Theory(Timeout = 60000)]
    [InlineData(typeof(DeepARModel<double>))]
    [InlineData(typeof(TemporalFusionTransformer<double>))]
    [InlineData(typeof(DLinearModel<double>))]
    public async Task IntrinsicLossModels_DoNotClaimToSupportLossInjection(Type modelType)
    {
        Assert.False(
            typeof(ISupportsLossFunction<double>).IsAssignableFrom(modelType),
            $"{modelType.Name} trains against an objective intrinsic to its architecture and must not " +
            "implement ISupportsLossFunction<double>; a substituted loss would train the wrong objective.");
        await Task.CompletedTask;
    }

    [Theory(Timeout = 60000)]
    [InlineData(typeof(NBEATSModel<double>))]
    [InlineData(typeof(NHiTSModel<double>))]
    [InlineData(typeof(InformerModel<double>))]
    [InlineData(typeof(AutoformerModel<double>))]
    public async Task PointForecasters_DoSupportLossInjection(Type modelType)
    {
        Assert.True(
            typeof(ISupportsLossFunction<double>).IsAssignableFrom(modelType),
            $"{modelType.Name} is a point forecaster and should accept a configured loss.");
        await Task.CompletedTask;
    }

    // cols must equal the models' LookbackWindow (4): NBEATS treats each row as a lookback window.
    private static (Matrix<double> X, Vector<double> Y) BuildSeries(int rows = 48, int cols = 4)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.3) + i * 0.01;
            y[i] = Math.Sin((i + cols) * 0.3) + i * 0.01;
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task Facade_ConfigureLossFunction_ReachesTrainedModel()
    {
        var (x, y) = BuildSeries();
        var model = new NBeatsProbe(new NBEATSModelOptions<double> { LookbackWindow = 4, ForecastHorizon = 1, Epochs = 2 });

        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureLossFunction(new MeanAbsoluteErrorLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        // The loss the model actually trained against was the one configured through the facade — not a
        // facade-side reporting variable, and not the default MSE.
        Assert.IsType<MeanAbsoluteErrorLoss<double>>(model.DefaultLossFunction);
    }

    [Fact(Timeout = 120000)]
    public async Task Facade_InvalidNonTapeLoss_IsRejected()
    {
        var (x, y) = BuildSeries();
        var model = new NBeatsProbe(new NBEATSModelOptions<double> { LookbackWindow = 4, ForecastHorizon = 1, Epochs = 2 });

        // A non-tape loss must be rejected through the facade up front, not accepted and then failing deep
        // in the first backward pass.
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureLossFunction(new NotATapeLoss())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y));

        await Assert.ThrowsAsync<ArgumentException>(() => builder.BuildAsync());
    }

    /// <summary>Exposes the protected default loss for assertions.</summary>
    private sealed class NBeatsProbe : NBEATSModel<double>
    {
        public NBeatsProbe(NBEATSModelOptions<double> options) : base(options) { }
    }

    /// <summary>An ILossFunction that is deliberately NOT a LossFunctionBase.</summary>
    private sealed class NotATapeLoss : ILossFunction<double>
    {
        public double CalculateLoss(Vector<double> predicted, Vector<double> actual) => 0;

        public Vector<double> CalculateDerivative(Vector<double> predicted, Vector<double> actual) => predicted;

        public (double Loss, Tensor<double> Gradient) CalculateLossAndGradientGpu(Tensor<double> predicted, Tensor<double> actual)
            => (0, predicted);
    }
}
