using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class ChronosFoundationModelTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ChronosFoundationModel<double>(new ChronosOptions<double>
        {
            VocabularySize = 32,
            ContextLength = 20,
            ForecastHorizon = 5,
            EmbeddingDim = 8,
            NumLayers = 1,
            NumHeads = 2,
            Epochs = 2,
            MaxTrainingTimeSeconds = 5
        });
}
