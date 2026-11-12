using AiDotNet.Interfaces;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class ChronosFoundationModelTests : TimeSeriesModelTestBase
{
    // A small Chronos-T5 trained from scratch on the 100-point series; the forecasts are seeded sample medians. The
    // paper's rate of 1e-3 is for 200K steps: over 200 steps here it moved the loss from 5.54 to 5.34. Each training
    // takes about 10 s per 100 steps of 16 windows, and several of these tests train two models inside their 60 s
    // limit, so the fixture trains 100 steps, at 0.03: the in-sample mean residual was -7.8 at 0.01 (the later half
    // still overshooting), -1.8 at 0.02 and -0.9 at 0.03.
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ChronosFoundationModel<double>(new ChronosOptions<double>
        {
            ContextLength = 32, PredictionLength = 8, MinPast = 16, NumTokens = 256, ModelDim = 32, FeedForwardDim = 64,
            KeyValueDim = 8, NumHeads = 4, NumLayers = 2, TrainingSteps = 100, BatchSize = 16, LearningRate = 0.03,
        });
}
