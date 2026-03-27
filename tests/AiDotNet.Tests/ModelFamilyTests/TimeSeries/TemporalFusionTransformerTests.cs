using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class TemporalFusionTransformerTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new TemporalFusionTransformer<double>(new AiDotNet.Models.Options.TemporalFusionTransformerOptions<double>
        {
            LookbackWindow = 20,
            ForecastHorizon = 5,
            HiddenSize = 8,
            NumAttentionHeads = 2,
            NumLayers = 1,
            Epochs = 2,
            MaxTrainingTimeSeconds = 10
        });
}
