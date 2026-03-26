using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class InformerModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new InformerModel<double>(new AiDotNet.Models.Options.InformerOptions<double>
        {
            LookbackWindow = 40,
            ForecastHorizon = 10,
            EmbeddingDim = 16,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumAttentionHeads = 2,
            MaxTrainingTimeSeconds = 5
        });
}
