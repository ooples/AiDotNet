using AiDotNet.Interfaces;
using AiDotNet.TimeSeries.AnomalyDetection;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.TimeSeries;

public class DeepANTTests : TimeSeriesModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new DeepANT<double>(new DeepANTOptions<double>
        {
            // The paper's window of 45 leaves the builder's 70-point training split 25 windows, and SGD at the default
            // rate for the default 30 epochs underfits this 100-point series (mean residual -8.0 on a range of 48). A
            // window of 12 gives it 88 windows; 150 epochs at 0.05 fit it.
            WindowSize = 12,
            Epochs = 150,
            LearningRate = 0.05,
        });
}
