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
            LookbackWindow = 40, LabelLength = 20,
            ForecastHorizon = 10,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            EmbeddingDim = 16,
            FeedForwardDim = 32,
            NumAttentionHeads = 2,
            MovingAverageKernel = 5,
            // 51 windows of a 100-point series: the reference's batch of 32 with the last incomplete batch dropped
            // would train one step per epoch while the rate halves every epoch.
            BatchSize = 8,
            LearningRate = 1e-3,
            Epochs = 10
        });
}
