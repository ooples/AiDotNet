using AiDotNet.Classification.MultiLabel;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Regression tests for the gradient average in <c>MultiLabelClassifierBase.ComputeGradients</c>.
/// The divisor rows * labels * parameters used to be computed in 32-bit integer arithmetic, so once
/// the product passed int.MaxValue it wrapped negative and flipped the sign of every gradient.
/// </summary>
public class MultiLabelGradientPrecisionTests
{
    [Fact]
    public void ComputeGradients_WhenRowsLabelsParametersProductExceedsInt32_KeepsCorrectSignAndMagnitude()
    {
        const int rows = 1000;
        const int labels = 100;
        // 1000 * 100 * 21475 = 2,147,500,000 > int.MaxValue (2,147,483,647): the old int product
        // wrapped to -2,147,467,296, turning the average gradient negative.
        var model = new ConstantProbabilityMultiLabelStub { NumLabels = labels };
        var input = new Matrix<double>(rows, 1);
        var target = new Matrix<double>(rows, labels);
        for (int i = 0; i < rows; i++)
        {
            for (int l = 0; l < labels; l++)
            {
                target[i, l] = 0.25;
            }
        }

        var gradients = model.ComputeGradients(input, target);

        double pred = ConstantProbabilityMultiLabelStub.Probability;
        double deriv = (pred - 0.25) / (pred * (1 - pred) + 1e-15);
        double expected = rows * labels * deriv
            / ((double)rows * labels * ConstantProbabilityMultiLabelStub.StubParameterCount);

        Assert.Equal(ConstantProbabilityMultiLabelStub.StubParameterCount, gradients.Length);
        Assert.True(gradients[0] > 0, $"Average gradient must stay positive, got {gradients[0]}.");
        Assert.Equal(expected, gradients[0], 12);
        Assert.Equal(expected, gradients[gradients.Length - 1], 12);
    }

    /// <summary>
    /// Minimal multi-label model: predicts a constant probability for every label and exposes a
    /// fixed-size parameter vector, so the gradient divisor can be driven past int.MaxValue with
    /// tiny inputs.
    /// </summary>
    private sealed class ConstantProbabilityMultiLabelStub : MultiLabelClassifierBase<double>
    {
        public const int StubParameterCount = 21475;
        public const double Probability = 0.75;

        protected override void TrainMultiLabelCore(Matrix<double> features, Matrix<double> labels)
        {
        }

        public override Matrix<double> PredictMultiLabelProbabilities(Matrix<double> input)
        {
            var probabilities = new Matrix<double>(input.Rows, NumLabels);
            for (int i = 0; i < input.Rows; i++)
            {
                for (int l = 0; l < NumLabels; l++)
                {
                    probabilities[i, l] = Probability;
                }
            }

            return probabilities;
        }

        public override Vector<double> GetParameters() => new Vector<double>(StubParameterCount);
    }
}
