using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Isolated gradient check for S6Scan to verify the backward pass
/// independently of the MambaBlock composition.
/// </summary>
public class S6ScanGradientTest
{
    [Fact]
    public void S6Scan_BackwardGradient_MatchesNumerical()
    {
        int batch = 1, seqLen = 2, innerDim = 3, stateDim = 2;
        var rng = new Random(42);

        var x = new Tensor<double>(new[] { batch, seqLen, innerDim });
        var delta = new Tensor<double>(new[] { batch, seqLen, innerDim });
        var aLog = new Tensor<double>(new[] { innerDim, stateDim });
        var b = new Tensor<double>(new[] { batch, seqLen, stateDim });
        var c = new Tensor<double>(new[] { batch, seqLen, stateDim });
        var d = new Tensor<double>(new[] { innerDim });

        // Fill with small random values
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextDouble() * 0.4 - 0.2;
        for (int i = 0; i < delta.Length; i++) delta[i] = rng.NextDouble() * 0.2 + 0.1; // positive
        for (int i = 0; i < aLog.Length; i++) aLog[i] = -1.0 + rng.NextDouble() * 0.5;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 0.4 - 0.2;
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 0.4 - 0.2;
        for (int i = 0; i < d.Length; i++) d[i] = rng.NextDouble() * 0.2;

        // Forward
        var (output, hidden) = S6Scan<double>.SequentialScanForward(
            x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);

        // Random projection loss for gradient checking
        var rngLoss = new Random(12345);
        var projWeights = new double[output.Length];
        for (int i = 0; i < projWeights.Length; i++)
            projWeights[i] = rngLoss.NextDouble() * 2.0 - 1.0;

        // Compute loss and gradient
        var dOut = new Tensor<double>(output.Shape.ToArray());
        for (int i = 0; i < dOut.Length; i++)
            dOut[i] = projWeights[i];

        var (dX, dDelta, dALog, dB, dC, dD) = S6Scan<double>.SequentialScanBackward(
            dOut, x, delta, aLog, b, c, d, hidden, batch, seqLen, innerDim, stateDim);

        // Numerical gradient check for each x element
        double eps = 1e-5;
        int failCount = 0;
        var errors = new System.Text.StringBuilder();

        for (int idx = 0; idx < Math.Min(6, x.Length); idx++)
        {
            double origVal = x[idx];

            x[idx] = origVal + eps;
            var (outPlus, _) = S6Scan<double>.SequentialScanForward(
                x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);
            double lossPlus = 0;
            for (int i = 0; i < outPlus.Length; i++) lossPlus += projWeights[i] * outPlus[i];

            x[idx] = origVal - eps;
            var (outMinus, _2) = S6Scan<double>.SequentialScanForward(
                x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);
            double lossMinus = 0;
            for (int i = 0; i < outMinus.Length; i++) lossMinus += projWeights[i] * outMinus[i];

            x[idx] = origVal;

            double numerical = (lossPlus - lossMinus) / (2 * eps);
            double analytical = dX[idx];

            double absMax = Math.Max(Math.Abs(numerical), Math.Abs(analytical));
            if (absMax < 1e-7) continue;

            double relErr = Math.Abs(numerical - analytical) / (absMax + 1e-8);
            if (relErr > 0.01)
            {
                failCount++;
                errors.Append($"x[{idx}]: analytical={analytical:G6} numerical={numerical:G6} relErr={relErr:G4} | ");
            }
        }

        Assert.True(failCount == 0,
            $"S6Scan gradient check failed for {failCount} x elements. Details: {errors}");
    }

    [Fact]
    public void S6Scan_BackwardDeltaGradient_MatchesNumerical()
    {
        int batch = 1, seqLen = 2, innerDim = 3, stateDim = 2;
        var rng = new Random(42);

        var x = new Tensor<double>(new[] { batch, seqLen, innerDim });
        var delta = new Tensor<double>(new[] { batch, seqLen, innerDim });
        var aLog = new Tensor<double>(new[] { innerDim, stateDim });
        var b = new Tensor<double>(new[] { batch, seqLen, stateDim });
        var c = new Tensor<double>(new[] { batch, seqLen, stateDim });
        var d = new Tensor<double>(new[] { innerDim });

        for (int i = 0; i < x.Length; i++) x[i] = rng.NextDouble() * 0.4 - 0.2;
        for (int i = 0; i < delta.Length; i++) delta[i] = rng.NextDouble() * 0.2 + 0.1;
        for (int i = 0; i < aLog.Length; i++) aLog[i] = -1.0 + rng.NextDouble() * 0.5;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 0.4 - 0.2;
        for (int i = 0; i < c.Length; i++) c[i] = rng.NextDouble() * 0.4 - 0.2;
        for (int i = 0; i < d.Length; i++) d[i] = rng.NextDouble() * 0.2;

        var (output, hidden) = S6Scan<double>.SequentialScanForward(
            x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);

        var rngLoss = new Random(12345);
        var projWeights = new double[output.Length];
        for (int i = 0; i < projWeights.Length; i++)
            projWeights[i] = rngLoss.NextDouble() * 2.0 - 1.0;

        var dOut = new Tensor<double>(output.Shape.ToArray());
        for (int i = 0; i < dOut.Length; i++) dOut[i] = projWeights[i];

        var (_, dDelta, dALog, dB, dC, dD) = S6Scan<double>.SequentialScanBackward(
            dOut, x, delta, aLog, b, c, d, hidden, batch, seqLen, innerDim, stateDim);

        double eps = 1e-5;
        int failCount = 0;
        var errors = new System.Text.StringBuilder();

        // Check delta gradients
        for (int idx = 0; idx < Math.Min(6, delta.Length); idx++)
        {
            double origVal = delta[idx];
            delta[idx] = origVal + eps;
            var (outPlus, _) = S6Scan<double>.SequentialScanForward(x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);
            double lossPlus = 0; for (int i = 0; i < outPlus.Length; i++) lossPlus += projWeights[i] * outPlus[i];
            delta[idx] = origVal - eps;
            var (outMinus, _2) = S6Scan<double>.SequentialScanForward(x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);
            double lossMinus = 0; for (int i = 0; i < outMinus.Length; i++) lossMinus += projWeights[i] * outMinus[i];
            delta[idx] = origVal;

            double numerical = (lossPlus - lossMinus) / (2 * eps);
            double analytical = dDelta[idx];
            double absMax = Math.Max(Math.Abs(numerical), Math.Abs(analytical));
            if (absMax < 1e-7) continue;
            double relErr = Math.Abs(numerical - analytical) / (absMax + 1e-8);
            if (relErr > 0.01) { failCount++; errors.Append($"delta[{idx}]: a={analytical:G6} n={numerical:G6} e={relErr:G4} | "); }
        }

        // Check aLog gradients
        for (int idx = 0; idx < Math.Min(4, aLog.Length); idx++)
        {
            double origVal = aLog[idx];
            aLog[idx] = origVal + eps;
            var (outPlus, _) = S6Scan<double>.SequentialScanForward(x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);
            double lossPlus = 0; for (int i = 0; i < outPlus.Length; i++) lossPlus += projWeights[i] * outPlus[i];
            aLog[idx] = origVal - eps;
            var (outMinus, _2) = S6Scan<double>.SequentialScanForward(x, delta, aLog, b, c, d, batch, seqLen, innerDim, stateDim);
            double lossMinus = 0; for (int i = 0; i < outMinus.Length; i++) lossMinus += projWeights[i] * outMinus[i];
            aLog[idx] = origVal;

            double numerical = (lossPlus - lossMinus) / (2 * eps);
            double analytical = dALog[idx];
            double absMax = Math.Max(Math.Abs(numerical), Math.Abs(analytical));
            if (absMax < 1e-7) continue;
            double relErr = Math.Abs(numerical - analytical) / (absMax + 1e-8);
            if (relErr > 0.01) { failCount++; errors.Append($"aLog[{idx}]: a={analytical:G6} n={numerical:G6} e={relErr:G4} | "); }
        }

        Assert.True(failCount == 0,
            $"S6Scan delta/aLog gradient check failed for {failCount} params. {errors}");
    }
}
