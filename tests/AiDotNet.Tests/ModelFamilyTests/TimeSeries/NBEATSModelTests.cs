using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Fixtures;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

// R2_ShouldBePositive_OnTrendData gives the optimizer a fixed
// MaxTrainingTimeSeconds budget to fit a synthetic trend+seasonal
// signal. Under xUnit's default parallel execution this budget gets
// shared across 4 threads on 2-core CI runners, leaving only ~1.25 s
// of real CPU — not enough Adam steps to converge past R² = 0. Placing
// NBEATS in the ConvergenceSensitive collection ensures the full wall-
// clock budget translates to full CPU availability without being a
// timeout-bump hack (the model's compute is still the only variable).
[Collection(ConvergenceSensitiveCollection.Name)]
public class NBEATSModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new NBEATSModel<double>(new AiDotNet.Models.Options.NBEATSModelOptions<double>
        {
            NumStacks = 2,
            NumBlocksPerStack = 1,
            LookbackWindow = 10,
            ForecastHorizon = 5,
            HiddenLayerSize = 16,
            NumHiddenLayers = 1,
            MaxTrainingTimeSeconds = 5
        });
}
