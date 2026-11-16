using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation;

/// <summary>Uniform calibration bins with double reductions and bounded sample workspaces.</summary>
internal static class EvaluationBinOps
{
    internal const int SampleBlockSize = 65_536;

    internal static (Tensor<double> Counts, Tensor<double> MeanScores, Tensor<double> MeanLabels) Averages(
        Tensor<double> scores, Tensor<double> labels, int numBins)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<double>();
        int n = scores.Length;
        var counts = new double[numBins];
        var meanScores = new double[numBins];
        var meanLabels = new double[numBins];
        for (int bin = 0; bin < numBins; bin++)
        {
            double lower = (double)bin / numBins;
            double upper = (double)(bin + 1) / numBins;
            int count = 0;
            double scoreSum = 0, labelSum = 0;
            for (int start = 0; start < n; start += SampleBlockSize)
            {
                int length = Math.Min(SampleBlockSize, n - start);
                var sample = engine.TensorNarrow(scores, 0, start, length);
                var targets = engine.TensorNarrow(labels, 0, start, length);
                var belowUpper = bin == numBins - 1
                    ? engine.TensorLessEqual(sample, engine.TensorFullLike(sample, upper))
                    : engine.TensorLessThan(sample, upper);
                var mask = engine.TensorMultiply(engine.TensorGreaterEqual(sample, engine.TensorFullLike(sample, lower)), belowUpper);
                count += engine.TensorCountNonzero(mask);
                var zero = engine.TensorZerosLike(sample);
                scoreSum += engine.TensorSum(engine.TensorWhere(mask, sample, zero));
                labelSum += engine.TensorSum(engine.TensorWhere(mask, targets, zero));
            }
            counts[bin] = count;
            if (count > 0)
            {
                meanScores[bin] = scoreSum / count;
                meanLabels[bin] = labelSum / count;
            }
        }
        return (new Tensor<double>(counts, new[] { numBins }),
            new Tensor<double>(meanScores, new[] { numBins }), new Tensor<double>(meanLabels, new[] { numBins }));
    }
}
