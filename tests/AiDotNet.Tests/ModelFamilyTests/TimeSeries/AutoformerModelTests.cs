using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class AutoformerModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new AutoformerModel<double>(new AiDotNet.Models.Options.AutoformerOptions<double>
        {
            LookbackWindow = 40,
            ForecastHorizon = 10,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            EmbeddingDim = 16,
            NumAttentionHeads = 2,
            Epochs = 3
        });
}
