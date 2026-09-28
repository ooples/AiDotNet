using AiDotNet.Document.OCR.TextDetection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Document;

/// <summary>
/// PSENet's objective (Wang et al. 2019, Eq. 5) against values worked out here from the formula, not the code.
/// </summary>
public sealed class PSENetLossTests
{
    public PSENetLossTests() => TestModuleInitializer.EnsureInitialized();

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double Dice(double[] s, double[] g, bool[] m)
    {
        double a = 0, b = 0.001, c = 0.001;
        for (int i = 0; i < s.Length; i++)
            if (m[i]) { a += s[i] * g[i]; b += s[i] * s[i]; c += g[i] * g[i]; }
        return 2 * a / (b + c);
    }

    [Fact]
    public void Ohem_KeepsEveryPositiveAndThreeTimesAsManyHardestNegatives()
    {
        // One image, one kernel, 1 x 8 map: two positives, six negatives with distinct scores.
        double[] logits = { 2.0, 1.0, 3.0, -1.0, 0.5, -2.0, 1.5, -0.5 };
        double[] gt = { 1, 1, 0, 0, 0, 0, 0, 0 };
        var loss = new PSENetLoss<double>().ComputeTapeLoss(
            new Tensor<double>(new[] { 1, 1, 1, 8 }, new Vector<double>(logits)),
            new Tensor<double>(new[] { 1, 1, 1, 8 }, new Vector<double>(gt)));

        // OHEM keeps both positives plus the 3 * 2 = 6 hardest negatives, i.e. all six here, so every pixel.
        var s = logits.Select(Sigmoid).ToArray();
        double expected = 1 - Dice(s, gt, Enumerable.Repeat(true, 8).ToArray());
        Assert.Equal(expected, loss[0], 10);
    }

    [Fact]
    public void Ohem_DropsTheEasyNegatives()
    {
        // One positive, so only the 3 highest-scoring of the five negatives count.
        double[] logits = { 2.0, 3.0, -1.0, 0.5, -2.0, 1.5 };
        double[] gt = { 1, 0, 0, 0, 0, 0 };
        var loss = new PSENetLoss<double>().ComputeTapeLoss(
            new Tensor<double>(new[] { 1, 1, 1, 6 }, new Vector<double>(logits)),
            new Tensor<double>(new[] { 1, 1, 1, 6 }, new Vector<double>(gt)));

        var s = logits.Select(Sigmoid).ToArray();
        bool[] kept = { true, true, false, true, false, true };
        Assert.Equal(1 - Dice(s, gt, kept), loss[0], 10);
    }

    [Fact]
    public void Kernels_AreWeightedPointThreeAndScoredOnlyInsideThePredictedText()
    {
        // Two kernels: channel 0 shrunk, channel 1 complete. Complete-map logits mark pixels 0-1 as text.
        double[] complete = { 3.0, 2.0, -2.0, -3.0 };
        double[] shrunk = { 1.0, -1.0, 2.0, 2.0 };
        double[] gtComplete = { 1, 1, 0, 0 };
        double[] gtShrunk = { 1, 0, 0, 0 };
        var loss = new PSENetLoss<double>().ComputeTapeLoss(
            new Tensor<double>(new[] { 1, 2, 1, 4 }, new Vector<double>(shrunk.Concat(complete).ToArray())),
            new Tensor<double>(new[] { 1, 2, 1, 4 }, new Vector<double>(gtShrunk.Concat(gtComplete).ToArray())));

        var sc = complete.Select(Sigmoid).ToArray();
        var ss = shrunk.Select(Sigmoid).ToArray();
        // Complete map: two positives, both negatives kept. Shrunk kernel: only where S_n > 0.5 (pixels 0-1),
        // so its high scores at pixels 2-3, which lie outside the text, cost nothing.
        double lc = 1 - Dice(sc, gtComplete, new[] { true, true, true, true });
        double ls = 1 - Dice(ss, gtShrunk, new[] { true, true, false, false });
        Assert.Equal(0.7 * lc + 0.3 * ls, loss[0], 10);
    }
}