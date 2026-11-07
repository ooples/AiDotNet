using AiDotNet.Interfaces;
using AiDotNet.Classification.Boosting;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class HistGradientBoostingClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new HistGradientBoostingClassifier<double>();

    /// <summary>
    /// Nodes are split in parallel and large histograms accumulated in fixed chunks added in order, so the thread
    /// count must not change the trained model: the serialized trees are compared byte for byte. 5,000 rows make the
    /// root and its children span several chunks.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void Train_ThreadCountDoesNotChangeTheModel(int classes)
    {
        const int rows = 5000, features = 6;
        var x = new Matrix<double>(rows, features);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < features; j++)
                x[i, j] = ((i * 7919 + j * 104729 + i * j * 31) % 1009) / 1009.0;
            double score = x[i, 0] + 0.6 * x[i, 1] * x[i, 2] - 0.4 * x[i, 3];
            y[i] = classes == 2 ? (score > 0.6 ? 1 : 0) : (score < 0.4 ? 0 : score < 0.8 ? 1 : 2);
            if (i % 13 == 0) y[i] = i % classes;
        }

        byte[] Fit(int jobs)
        {
            var model = new HistGradientBoostingClassifier<double>(nEstimators: 15, maxDepth: 5, seed: 3) { NJobs = jobs };
            model.Train(x, y);

            // NJobs itself is serialized; set it alike so the bytes compare only what training produced.
            model.NJobs = 1;
            return model.Serialize();
        }

        var serial = Fit(1);
        Assert.Equal(serial, Fit(4));
        Assert.Equal(serial, Fit(-1));
    }

    [Fact]
    public void Train_RejectsZeroJobs()
    {
        var model = new HistGradientBoostingClassifier<double>(nEstimators: 2) { NJobs = 0 };
        var x = new Matrix<double>(40, 1);
        var y = new Vector<double>(40);
        for (int i = 0; i < 40; i++)
        {
            x[i, 0] = i;
            y[i] = i < 20 ? 0 : 1;
        }

        Assert.Throws<InvalidOperationException>(() => model.Train(x, y));
    }
}
