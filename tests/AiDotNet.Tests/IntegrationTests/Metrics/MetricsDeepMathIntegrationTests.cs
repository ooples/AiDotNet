using System;
using AiDotNet.Metrics;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Metrics;

/// <summary>
/// Deep math-correctness integration tests for the Metrics module.
/// Verifies PSNR, SSIM, mIoU, WER, CER, SI-SDR, SNR, Chamfer Distance,
/// F-Score, and 3D IoU against hand-computed expected values.
/// </summary>
public class MetricsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region PSNR Tests

    [Fact]
    public void PSNR_ExactFormula_UniformError()
    {
        // PSNR = 10 * log10(MAX^2 / MSE)
        // All pixels differ by 0.1: MSE = 0.01
        // MAX = 1.0: PSNR = 10 * log10(1.0 / 0.01) = 10 * log10(100) = 10 * 2 = 20 dB
        var psnr = new PeakSignalToNoiseRatio<double>();

        var data1 = new double[] { 0.5, 0.5, 0.5, 0.5 };
        var data2 = new double[] { 0.6, 0.6, 0.6, 0.6 };
        var img1 = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(data1));
        var img2 = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(data2));

        var result = psnr.Compute(img1, img2);

        Assert.Equal(20.0, result, Tolerance);
    }

    [Fact]
    public void PSNR_ExactFormula_VariedError()
    {
        // Predicted: [0.0, 0.5], GT: [0.1, 0.3]
        // Diff: [-0.1, 0.2], squared: [0.01, 0.04]
        // MSE = (0.01 + 0.04) / 2 = 0.025
        // PSNR = 10 * log10(1.0 / 0.025) = 10 * log10(40) = 10 * 1.60206 = 16.0206
        var psnr = new PeakSignalToNoiseRatio<double>();

        var pred = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.0, 0.5 }));
        var gt = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.1, 0.3 }));

        var result = psnr.Compute(pred, gt);
        double expected = 10.0 * Math.Log10(1.0 / 0.025);

        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void PSNR_CustomMaxValue_ScalesCorrectly()
    {
        // MAX = 255, pixel diff = 10 for all
        // MSE = 100, PSNR = 10 * log10(255^2 / 100) = 10 * log10(650.25) = 28.1308
        var psnr = new PeakSignalToNoiseRatio<double>(maxValue: 255.0);

        var data1 = new double[] { 100.0, 200.0, 150.0, 120.0 };
        var data2 = new double[] { 110.0, 210.0, 160.0, 130.0 };
        var img1 = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(data1));
        var img2 = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(data2));

        var result = psnr.Compute(img1, img2);
        double expected = 10.0 * Math.Log10(255.0 * 255.0 / 100.0);

        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void PSNR_IdenticalImages_Returns100()
    {
        // When MSE < epsilon (1e-10), returns 100.0
        var psnr = new PeakSignalToNoiseRatio<double>();

        var data = new double[] { 0.3, 0.7, 0.5, 0.9 };
        var img = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(data));

        var result = psnr.Compute(img, img);
        Assert.Equal(100.0, result, Tolerance);
    }

    [Fact]
    public void PSNR_BatchCompute_PerImagePSNR()
    {
        var psnr = new PeakSignalToNoiseRatio<double>();

        // Batch: 2 images, each 2x2x1
        // Image 0: all 0.5 vs all 0.5 => identical => 100 dB
        // Image 1: [0.0,0.0,0.0,0.0] vs [0.1,0.1,0.1,0.1] => MSE=0.01 => 20dB
        var pred = new double[2 * 2 * 2 * 1];
        var gt = new double[2 * 2 * 2 * 1];
        // Image 0: both 0.5
        for (int i = 0; i < 4; i++) { pred[i] = 0.5; gt[i] = 0.5; }
        // Image 1: pred=0, gt=0.1
        for (int i = 4; i < 8; i++) { pred[i] = 0.0; gt[i] = 0.1; }

        var predT = new Tensor<double>(new[] { 2, 2, 2, 1 }, new Vector<double>(pred));
        var gtT = new Tensor<double>(new[] { 2, 2, 2, 1 }, new Vector<double>(gt));

        var results = psnr.ComputeBatch(predT, gtT);

        Assert.Equal(100.0, results[0], Tolerance);
        Assert.Equal(20.0, results[1], Tolerance);
    }

    #endregion

    #region SSIM Tests

    [Fact]
    public void SSIM_IdenticalImages_ReturnsOne()
    {
        var ssim = new StructuralSimilarity<double>();
        var data = new double[] { 0.2, 0.4, 0.6, 0.8 };
        var img = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(data));

        var result = ssim.Compute(img, img);

        Assert.Equal(1.0, result, 1e-4);
    }

    [Fact]
    public void SSIM_ExactFormula_HandComputed()
    {
        // x = [0.0, 1.0], y = [0.5, 0.5]
        // μx = 0.5, μy = 0.5
        // σx² = Var(x) = ((0-0.5)²+(1-0.5)²)/1 = 0.5  (n-1 divisor, n=2)
        // σy² = 0
        // σxy = ((0-0.5)(0.5-0.5)+(1-0.5)(0.5-0.5))/1 = 0
        // C1 = (0.01*1)² = 0.0001, C2 = (0.03*1)² = 0.0009
        // numerator = (2*0.5*0.5 + 0.0001)(2*0 + 0.0009) = (0.5001)(0.0009)
        // denominator = (0.25+0.25 + 0.0001)(0.5+0 + 0.0009) = (0.5001)(0.5009)
        // SSIM = (0.5001*0.0009)/(0.5001*0.5009) = 0.0009/0.5009

        var ssim = new StructuralSimilarity<double>();
        var x = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.0, 1.0 }));
        var y = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.5, 0.5 }));

        var result = ssim.Compute(x, y);
        double expected = 0.0009 / 0.5009;

        Assert.Equal(expected, result, 1e-3);
    }

    [Fact]
    public void SSIM_ConstantShift_LowSSIM()
    {
        // x = [0,0,0,0], y = [1,1,1,1]
        // μx = 0, μy = 1, σx²=0, σy²=0, σxy=0
        // C1 = 0.0001, C2 = 0.0009
        // num = (0 + C1)(0 + C2) = C1*C2
        // den = (0+1+C1)(0+0+C2) = (1+C1)*C2
        // SSIM = C1/(1+C1)
        var ssim = new StructuralSimilarity<double>();
        var x = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 }));
        var y = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 }));

        var result = ssim.Compute(x, y);

        double c1 = Math.Pow(0.01, 2);
        double expected = c1 / (1.0 + c1);

        Assert.Equal(expected, result, 1e-4);
    }

    [Fact]
    public void SSIM_MultiChannelImage_AveragesPerChannel()
    {
        // 2x2x2 image (H=2, W=2, C=2)
        // Create both channels identical => SSIM for each channel = 1.0 => average = 1.0
        var data = new double[2 * 2 * 2];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5;

        var ssim = new StructuralSimilarity<double>();
        var img = new Tensor<double>(new[] { 2, 2, 2 }, new Vector<double>(data));

        var result = ssim.Compute(img, img);
        Assert.Equal(1.0, result, 1e-4);
    }

    #endregion

    #region mIoU Tests

    [Fact]
    public void MIoU_PerfectPrediction_ReturnsOne()
    {
        var miou = new MeanIntersectionOverUnion<double>(3);

        // pred = gt = [0, 1, 2, 0, 1, 2]
        var data = new double[] { 0, 1, 2, 0, 1, 2 };
        var pred = new Tensor<double>(new[] { 6 }, new Vector<double>(data));
        var gt = new Tensor<double>(new[] { 6 }, new Vector<double>(data));

        var result = miou.Compute(pred, gt);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MIoU_ExactPerClassIoU()
    {
        // 3 classes, 6 samples
        // pred = [0, 0, 1, 1, 2, 2]
        // gt   = [0, 1, 1, 2, 2, 0]
        // Class 0: pred={0,1}, gt={0,5} => TP=1, FP=1, FN=1 => IoU=1/3
        // Class 1: pred={2,3}, gt={1,2} => TP=1, FP=1, FN=1 => IoU=1/3
        // Class 2: pred={4,5}, gt={3,4} => TP=1, FP=1, FN=1 => IoU=1/3
        // mIoU = (1/3 + 1/3 + 1/3) / 3 = 1/3
        var miou = new MeanIntersectionOverUnion<double>(3);

        var pred = new Tensor<double>(new[] { 6 }, new Vector<double>(new double[] { 0, 0, 1, 1, 2, 2 }));
        var gt = new Tensor<double>(new[] { 6 }, new Vector<double>(new double[] { 0, 1, 1, 2, 2, 0 }));

        var result = miou.Compute(pred, gt);

        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void MIoU_PerClassIoU_MatchesHandComputed()
    {
        // 2 classes, 8 samples
        // pred = [0,0,0,0,1,1,1,1]
        // gt   = [0,0,1,1,0,0,1,1]
        // Class 0: predCount=4, gtCount=4, intersection=2 => union=4+4-2=6 => IoU=2/6=1/3
        // Class 1: predCount=4, gtCount=4, intersection=2 => union=4+4-2=6 => IoU=2/6=1/3
        var miou = new MeanIntersectionOverUnion<double>(2);

        var pred = new Tensor<double>(new[] { 8 }, new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 }));
        var gt = new Tensor<double>(new[] { 8 }, new Vector<double>(new double[] { 0, 0, 1, 1, 0, 0, 1, 1 }));

        var perClass = miou.ComputePerClass(pred, gt);

        Assert.Equal(1.0 / 3.0, perClass[0], Tolerance);
        Assert.Equal(1.0 / 3.0, perClass[1], Tolerance);
    }

    [Fact]
    public void MIoU_IgnoreBackground_SkipsClass0()
    {
        // 3 classes
        // pred = [0, 1, 2], gt = [0, 1, 2] => all perfect
        // ignoreBackground => skip class 0
        // Class 1: IoU = 1.0, Class 2: IoU = 1.0 => mIoU = 1.0
        var miou = new MeanIntersectionOverUnion<double>(3, ignoreBackground: true);

        var pred = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 0, 1, 2 }));
        var gt = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 0, 1, 2 }));

        var result = miou.Compute(pred, gt);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MIoU_AllWrong_ZeroIoU()
    {
        // 2 classes, all predictions wrong
        // pred = [0, 0, 0], gt = [1, 1, 1]
        // Class 0: predCount=3, gtCount=0, intersection=0 => union=3 => IoU=0
        // Class 1: predCount=0, gtCount=3, intersection=0 => union=3 => IoU=0
        // mIoU = 0
        var miou = new MeanIntersectionOverUnion<double>(2);

        var pred = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 0, 0, 0 }));
        var gt = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 1, 1 }));

        var result = miou.Compute(pred, gt);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region OverallAccuracy Tests

    [Fact]
    public void OverallAccuracy_ExactComputation()
    {
        // pred = [0, 1, 2, 0, 1], gt = [0, 1, 1, 0, 2]
        // Match at indices 0,1,3 => 3/5 = 0.6
        var acc = new OverallAccuracy<double>();

        var pred = new Tensor<double>(new[] { 5 }, new Vector<double>(new double[] { 0, 1, 2, 0, 1 }));
        var gt = new Tensor<double>(new[] { 5 }, new Vector<double>(new double[] { 0, 1, 1, 0, 2 }));

        var result = acc.Compute(pred, gt);
        Assert.Equal(0.6, result, Tolerance);
    }

    [Fact]
    public void OverallAccuracy_PerfectPrediction()
    {
        var acc = new OverallAccuracy<double>();
        var data = new double[] { 0, 1, 2, 3 };
        var pred = new Tensor<double>(new[] { 4 }, new Vector<double>(data));
        var gt = new Tensor<double>(new[] { 4 }, new Vector<double>(data));

        var result = acc.Compute(pred, gt);
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region WordErrorRate Tests

    [Fact]
    public void WER_PerfectTranscription_ReturnsZero()
    {
        var wer = new WordErrorRate();
        var result = wer.Compute("the cat sat on the mat", "the cat sat on the mat");
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void WER_OneSubstitution()
    {
        // ref = "the cat sat" (3 words), hyp = "the dog sat"
        // 1 substitution (cat->dog), 0 ins, 0 del => WER = 1/3
        var wer = new WordErrorRate();
        var result = wer.Compute("the dog sat", "the cat sat");
        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void WER_OneInsertion()
    {
        // ref = "the cat" (2 words), hyp = "the big cat"
        // 0 sub, 1 insertion, 0 del => WER = 1/2 = 0.5
        var wer = new WordErrorRate();
        var result = wer.Compute("the big cat", "the cat");
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void WER_OneDeletion()
    {
        // ref = "the big cat" (3 words), hyp = "the cat"
        // 0 sub, 0 ins, 1 del => WER = 1/3
        var wer = new WordErrorRate();
        var result = wer.Compute("the cat", "the big cat");
        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void WER_DetailedStats_CorrectCounts()
    {
        // ref = "a b c d" (4 words), hyp = "a x c"
        // sub: b->x, del: d => 1 sub + 0 ins + 1 del = 2/4 = 0.5
        var wer = new WordErrorRate();
        var (werVal, subs, ins, dels, refCount) = wer.ComputeDetailed("a x c", "a b c d");

        Assert.Equal(4, refCount);
        Assert.Equal(2, subs + ins + dels);
        Assert.Equal(0.5, werVal, Tolerance);
    }

    [Fact]
    public void WER_CompletelyWrong()
    {
        // ref = "hello world" (2 words), hyp = "goodbye earth"
        // 2 substitutions => WER = 2/2 = 1.0
        var wer = new WordErrorRate();
        var result = wer.Compute("goodbye earth", "hello world");
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void WER_EmptyReference_NonEmptyHyp()
    {
        var wer = new WordErrorRate();
        var result = wer.Compute("hello", "");
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void WER_BothEmpty()
    {
        var wer = new WordErrorRate();
        var result = wer.Compute("", "");
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void WER_BatchCompute_AveragesCorrectly()
    {
        var wer = new WordErrorRate();
        // Pair 1: "a b" vs "a b" => WER = 0
        // Pair 2: "a" vs "b" => WER = 1
        // Average = 0.5
        var result = wer.ComputeBatch(
            new[] { "a b", "a" },
            new[] { "a b", "b" });
        Assert.Equal(0.5, result, Tolerance);
    }

    #endregion

    #region CharacterErrorRate Tests

    [Fact]
    public void CER_PerfectMatch_ReturnsZero()
    {
        var cer = new CharacterErrorRate();
        var result = cer.Compute("hello", "hello");
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CER_SingleCharDifference()
    {
        // ref = "cat" (3 chars), hyp = "bat"
        // 1 substitution => CER = 1/3
        var cer = new CharacterErrorRate();
        var result = cer.Compute("bat", "cat");
        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void CER_IgnoreWhitespace()
    {
        // ref = "a b c" without whitespace => "abc" (3 chars)
        // hyp = "a bc" without whitespace => "abc" (3 chars) => identical
        var cer = new CharacterErrorRate();
        var result = cer.Compute("a bc", "a b c", ignoreWhitespace: true);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CER_Insertion()
    {
        // ref = "ab" (2 chars), hyp = "axb"
        // Levenshtein: 1 insertion => CER = 1/2 = 0.5
        var cer = new CharacterErrorRate();
        var result = cer.Compute("axb", "ab");
        Assert.Equal(0.5, result, Tolerance);
    }

    #endregion

    #region SNR Tests

    [Fact]
    public void SNR_ExactFormula_KnownValues()
    {
        // clean = [3, 4], noisy = [3.1, 3.9]
        // noise = [0.1, -0.1]
        // signalPower = 9 + 16 = 25
        // noisePower = 0.01 + 0.01 = 0.02
        // SNR = 10 * log10(25 / 0.02) = 10 * log10(1250) = 10 * 3.09691 = 30.9691
        var snr = new SignalToNoiseRatio<double>();

        var clean = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 3.0, 4.0 }));
        var noisy = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 3.1, 3.9 }));

        var result = snr.Compute(clean, noisy);
        double expected = 10.0 * Math.Log10(25.0 / 0.02);

        Assert.Equal(expected, result, 1e-3);
    }

    [Fact]
    public void SNR_IdenticalSignals_Returns100()
    {
        var snr = new SignalToNoiseRatio<double>();

        var clean = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));

        var result = snr.Compute(clean, clean);
        Assert.Equal(100.0, result, Tolerance);
    }

    [Fact]
    public void SNR_LargeNoise_NegativeSNR()
    {
        // clean = [1.0], noisy = [11.0] => noise = [10.0]
        // signalPower = 1, noisePower = 100
        // SNR = 10 * log10(1/100) = 10 * (-2) = -20 dB
        var snr = new SignalToNoiseRatio<double>();

        var clean = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var noisy = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 11.0 }));

        var result = snr.Compute(clean, noisy);
        double expected = 10.0 * Math.Log10(1.0 / 100.0);

        Assert.Equal(expected, result, Tolerance);
    }

    #endregion

    #region SI-SDR Tests

    [Fact]
    public void SISDR_ScaleInvariant_DifferentAmplitudes()
    {
        // SI-SDR is scale-invariant: scaling the estimate shouldn't change the result
        // target = [1, 2, 3, 4]
        // estimate = [2, 4, 6, 8] (2x target, after zero-mean same direction)
        // After centering: target=[−1.5,−0.5,0.5,1.5], est=[−2.5+5-5=−3,−1,1,3]
        // Wait, let me use simpler: target=[1,-1], est=[2,-2] (already zero-mean)
        // alpha = <est,tar>/<tar,tar> = (2+2)/(1+1) = 2
        // s_target = [2,-2]
        // e_noise = [2,-2]-[2,-2] = [0,0]
        // => perfect, SI-SDR = 100

        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();

        var target = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 1.0, -1.0 }));
        var estimate = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 2.0, -2.0 }));

        var result = sisdr.Compute(estimate, target);
        Assert.Equal(100.0, result, Tolerance);
    }

    [Fact]
    public void SISDR_ExactFormula_WithNoise()
    {
        // target = [1, 0, -1, 0] (mean=0)
        // estimate = [1.1, 0.2, -0.9, 0.1] (mean = 0.125)
        // After zero-mean:
        //   tar_c = [1, 0, -1, 0]
        //   est_c = [1.1-0.125, 0.2-0.125, -0.9-0.125, 0.1-0.125]
        //         = [0.975, 0.075, -1.025, -0.025]
        // dot(est_c, tar_c) = 0.975*1 + 0.075*0 + (-1.025)*(-1) + (-0.025)*0
        //                   = 0.975 + 0 + 1.025 + 0 = 2.0
        // dot(tar_c, tar_c) = 1 + 0 + 1 + 0 = 2.0
        // alpha = 2.0 / 2.0 = 1.0
        // s_target = 1.0 * [1, 0, -1, 0] = [1, 0, -1, 0]
        // e_noise = est_c - s_target = [0.975-1, 0.075-0, -1.025-(-1), -0.025-0]
        //         = [-0.025, 0.075, -0.025, -0.025]
        // ||s_target||² = 2.0
        // ||e_noise||² = 0.000625 + 0.005625 + 0.000625 + 0.000625 = 0.0075
        // SI-SDR = 10 * log10(2.0 / 0.0075) = 10 * log10(266.667) = 10 * 2.4260 = 24.260
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();

        var target = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 0.0, -1.0, 0.0 }));
        var estimate = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.1, 0.2, -0.9, 0.1 }));

        var result = sisdr.Compute(estimate, target);
        double expected = 10.0 * Math.Log10(2.0 / 0.0075);

        Assert.Equal(expected, result, 0.1);
    }

    [Fact]
    public void SISDR_Improvement_IsPositiveWhenBetter()
    {
        var sisdr = new ScaleInvariantSignalToDistortionRatio<double>();

        var target = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, -1.0, 1.0, -1.0 }));
        var estimate = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.9, -0.9, 0.9, -0.9 }));
        var baseline = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.5, -0.5, 0.5, 0.0 })); // worse

        var improvement = sisdr.ComputeImprovement(estimate, target, baseline);
        Assert.True(improvement > 0, "Improvement should be positive when estimate is better than baseline");
    }

    #endregion

    #region Chamfer Distance Tests

    [Fact]
    public void ChamferDistance_IdenticalPointClouds_ReturnsZero()
    {
        var cd = new ChamferDistance<double>();

        // 3 points in 2D
        var data = new double[] { 0, 0, 1, 0, 0, 1 };
        var points = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(data));

        var result = cd.Compute(points, points);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ChamferDistance_Squared_ExactComputation()
    {
        // A = {(0,0)}, B = {(3,4)}
        // d(A->B) = dist²((0,0),(3,4)) = 9+16 = 25 / 1 = 25
        // d(B->A) = dist²((3,4),(0,0)) = 25 / 1 = 25
        // CD = 25 + 25 = 50
        var cd = new ChamferDistance<double>(squared: true);

        var a = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.0, 0.0 }));
        var b = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 3.0, 4.0 }));

        var result = cd.Compute(a, b);
        Assert.Equal(50.0, result, Tolerance);
    }

    [Fact]
    public void ChamferDistance_Euclidean_ExactComputation()
    {
        // Same as above but Euclidean: each direction gives 5.0
        var cd = new ChamferDistance<double>(squared: false);

        var a = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.0, 0.0 }));
        var b = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 3.0, 4.0 }));

        var result = cd.Compute(a, b);
        Assert.Equal(10.0, result, Tolerance);
    }

    [Fact]
    public void ChamferDistance_NearestNeighborSelection()
    {
        // A = {(0,0)}, B = {(1,0), (10,0)}
        // A->B: nearest to (0,0) is (1,0) with dist²=1 => avg = 1/1 = 1
        // B->A: nearest to (1,0) from A is (0,0) dist²=1; nearest to (10,0) from A is (0,0) dist²=100
        //   => avg = (1+100)/2 = 50.5
        // CD = 1 + 50.5 = 51.5
        var cd = new ChamferDistance<double>(squared: true);

        var a = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.0, 0.0 }));
        var b = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 0.0, 10.0, 0.0 }));

        var result = cd.Compute(a, b);
        Assert.Equal(51.5, result, Tolerance);
    }

    [Fact]
    public void ChamferDistance_Asymmetric_OneWay()
    {
        // A = {(0,0), (2,0)}, B = {(1,0)}
        // A->B: (0,0)->(1,0)=1, (2,0)->(1,0)=1 => avg = 1.0
        var cd = new ChamferDistance<double>(squared: true);

        var a = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 0.0, 2.0, 0.0 }));
        var b = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 1.0, 0.0 }));

        var oneWay = cd.ComputeOneWay(a, b);
        Assert.Equal(1.0, oneWay, Tolerance);
    }

    [Fact]
    public void ChamferDistance_3D_Points()
    {
        // A = {(0,0,0)}, B = {(1,1,1)}
        // dist² = 1+1+1 = 3
        // CD = 3 + 3 = 6
        var cd = new ChamferDistance<double>(squared: true);

        var a = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 0.0, 0.0, 0.0 }));
        var b = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 1.0, 1.0, 1.0 }));

        var result = cd.Compute(a, b);
        Assert.Equal(6.0, result, Tolerance);
    }

    #endregion

    #region FScore Tests

    [Fact]
    public void FScore_PerfectOverlap_ReturnsOne()
    {
        // All predicted points are within threshold of GT points
        var fscore = new FScore<double>(threshold: 1.0);

        var pred = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 }));
        var gt = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 }));

        var result = fscore.Compute(pred, gt);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void FScore_HandComputed_PrecisionRecall()
    {
        // threshold = 1.5 (squared = 2.25)
        // pred = {(0,0), (5,5)}, gt = {(1,0), (10,10)}
        // Precision:
        //   (0,0) closest to (1,0) dist²=1 <= 2.25 => match
        //   (5,5) closest to (1,0)=16+25=41 or (10,10)=25+25=50 => no match
        //   precision = 1/2 = 0.5
        // Recall:
        //   (1,0) closest to (0,0) dist²=1 <= 2.25 => match
        //   (10,10) closest to (5,5) dist²=50 > 2.25 => no match
        //   recall = 1/2 = 0.5
        // F1 = 2*0.5*0.5/(0.5+0.5) = 0.5
        var fscore = new FScore<double>(threshold: 1.5);

        var pred = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 0.0, 5.0, 5.0 }));
        var gt = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 0.0, 10.0, 10.0 }));

        var (precision, recall) = fscore.ComputePrecisionRecall(pred, gt);
        var f1 = fscore.Compute(pred, gt);

        Assert.Equal(0.5, precision, Tolerance);
        Assert.Equal(0.5, recall, Tolerance);
        Assert.Equal(0.5, f1, Tolerance);
    }

    [Fact]
    public void FScore_AllOutOfThreshold_ReturnsZero()
    {
        // threshold = 0.1 (very small)
        // pred = {(0,0)}, gt = {(10,10)} => dist = ~14.14 >> 0.1
        var fscore = new FScore<double>(threshold: 0.1);

        var pred = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.0, 0.0 }));
        var gt = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 10.0, 10.0 }));

        var result = fscore.Compute(pred, gt);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region IoU3D Tests

    [Fact]
    public void IoU3D_VoxelGrid_ExactComputation()
    {
        // 4 voxels: A = [1,1,0,0], B = [1,0,1,0]
        // intersection = 1 (both occupied at index 0)
        // union = 3 (occupied at indices 0,1,2)
        // IoU = 1/3
        var iou = new IoU3D<double>();

        var a = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 }));
        var b = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 }));

        var result = iou.ComputeVoxelIoU(a, b);
        Assert.Equal(1.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void IoU3D_VoxelGrid_PerfectOverlap()
    {
        var iou = new IoU3D<double>();
        var data = new double[] { 1, 1, 0, 1 };
        var a = new Tensor<double>(new[] { 4 }, new Vector<double>(data));

        var result = iou.ComputeVoxelIoU(a, a);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void IoU3D_VoxelGrid_NoOverlap()
    {
        var iou = new IoU3D<double>();
        var a = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 }));
        var b = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 }));

        var result = iou.ComputeVoxelIoU(a, b);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void IoU3D_BoundingBox_ExactComputation()
    {
        // Box A: [0,0,0, 2,2,2] volume = 8
        // Box B: [1,1,1, 3,3,3] volume = 8
        // Intersection: [1,1,1, 2,2,2] = 1*1*1 = 1
        // Union = 8 + 8 - 1 = 15
        // IoU = 1/15
        var iou = new IoU3D<double>();

        var boxA = new double[] { 0, 0, 0, 2, 2, 2 };
        var boxB = new double[] { 1, 1, 1, 3, 3, 3 };

        var result = iou.ComputeBoxIoU(boxA, boxB);
        Assert.Equal(1.0 / 15.0, result, Tolerance);
    }

    [Fact]
    public void IoU3D_BoundingBox_FullOverlap()
    {
        // Identical boxes
        var iou = new IoU3D<double>();
        var box = new double[] { 0, 0, 0, 1, 1, 1 };

        var result = iou.ComputeBoxIoU(box, box);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void IoU3D_BoundingBox_NoOverlap()
    {
        // Box A: [0,0,0, 1,1,1], Box B: [2,2,2, 3,3,3]
        // No intersection => IoU = 0
        var iou = new IoU3D<double>();

        var boxA = new double[] { 0, 0, 0, 1, 1, 1 };
        var boxB = new double[] { 2, 2, 2, 3, 3, 3 };

        var result = iou.ComputeBoxIoU(boxA, boxB);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void IoU3D_BoundingBox_ContainedBox()
    {
        // Box A: [0,0,0, 4,4,4] volume = 64
        // Box B: [1,1,1, 3,3,3] volume = 8
        // Intersection: [1,1,1, 3,3,3] = 8
        // Union = 64 + 8 - 8 = 64
        // IoU = 8/64 = 1/8
        var iou = new IoU3D<double>();

        var boxA = new double[] { 0, 0, 0, 4, 4, 4 };
        var boxB = new double[] { 1, 1, 1, 3, 3, 3 };

        var result = iou.ComputeBoxIoU(boxA, boxB);
        Assert.Equal(1.0 / 8.0, result, Tolerance);
    }

    [Fact]
    public void IoU3D_BoundingBox_PartialOverlap()
    {
        // Box A: [0,0,0, 2,3,4] volume = 2*3*4 = 24
        // Box B: [1,1,1, 3,4,5] volume = 2*3*4 = 24
        // Intersection: [1,1,1, 2,3,4] = 1*2*3 = 6
        // Union = 24 + 24 - 6 = 42
        // IoU = 6/42 = 1/7
        var iou = new IoU3D<double>();

        var boxA = new double[] { 0, 0, 0, 2, 3, 4 };
        var boxB = new double[] { 1, 1, 1, 3, 4, 5 };

        var result = iou.ComputeBoxIoU(boxA, boxB);
        Assert.Equal(1.0 / 7.0, result, Tolerance);
    }

    #endregion

    #region EarthMoversDistance Tests

    [Fact]
    public void EMD_IdenticalPointClouds_ReturnsNearZero()
    {
        var emd = new EarthMoversDistance<double>(iterations: 200, epsilon: 0.01);

        var data = new double[] { 0, 0, 1, 0, 0, 1 };
        var points = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(data));

        var result = emd.Compute(points, points);

        // Should be very close to 0 for identical point clouds
        Assert.True(result < 0.1, $"EMD for identical point clouds should be near 0, got {result}");
    }

    [Fact]
    public void EMD_FarApartPoints_LargerDistance()
    {
        // Use larger epsilon to avoid exp(-dist/eps) underflow for far points
        var emd = new EarthMoversDistance<double>(iterations: 200, epsilon: 1.0);

        var a = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 0.0, 1.0, 0.0 }));
        var bClose = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.1, 0.0, 1.1, 0.0 }));
        var bFar = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 3.0, 0.0, 4.0, 0.0 }));

        var distClose = emd.Compute(a, bClose);
        var distFar = emd.Compute(a, bFar);

        Assert.True(distFar > distClose, $"EMD for farther points ({distFar}) should be > EMD for closer points ({distClose})");
    }

    #endregion

    #region Cross-Metric Consistency Tests

    [Fact]
    public void PSNR_InverseMSE_Relationship()
    {
        // PSNR = 10*log10(MAX²/MSE), so higher MSE => lower PSNR
        var psnr = new PeakSignalToNoiseRatio<double>();

        var gt = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 }));
        var smallError = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.51, 0.51, 0.51, 0.51 }));
        var largeError = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.8, 0.8, 0.8, 0.8 }));

        var psnrSmall = psnr.Compute(smallError, gt);
        var psnrLarge = psnr.Compute(largeError, gt);

        Assert.True(psnrSmall > psnrLarge, "Lower MSE should give higher PSNR");
    }

    [Fact]
    public void ChamferDistance_Symmetry()
    {
        // CD(A,B) should equal CD(B,A)
        var cd = new ChamferDistance<double>(squared: true);

        var a = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.0, 0.0, 1.0, 0.0 }));
        var b = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 2.0, 0.0, 3.0, 0.0 }));

        var cdAB = cd.Compute(a, b);
        var cdBA = cd.Compute(b, a);

        Assert.Equal(cdAB, cdBA, Tolerance);
    }

    [Fact]
    public void FScore_F1_EqualsHarmonicMean_PrecisionRecall()
    {
        var fscore = new FScore<double>(threshold: 2.0);

        var pred = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.0, 0.0, 5.0, 5.0, 1.0, 1.0 }));
        var gt = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.5, 0.5, 10.0, 10.0 }));

        var (precision, recall) = fscore.ComputePrecisionRecall(pred, gt);
        var f1 = fscore.Compute(pred, gt);

        double p = precision;
        double r = recall;
        double expectedF1 = (p + r < 1e-10) ? 0.0 : 2.0 * p * r / (p + r);

        Assert.Equal(expectedF1, f1, Tolerance);
    }

    [Fact]
    public void WER_CaseInsensitive()
    {
        // WER should be case-insensitive (TokenizeWords uses ToLowerInvariant)
        var wer = new WordErrorRate();
        var result = wer.Compute("THE CAT SAT", "the cat sat");
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void OverallAccuracy_AllWrong_ReturnsZero()
    {
        var acc = new OverallAccuracy<double>();

        var pred = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 0, 0, 0 }));
        var gt = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 1, 1 }));

        var result = acc.Compute(pred, gt);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SSIM_HigherForSimilarImages()
    {
        var ssim = new StructuralSimilarity<double>();

        var original = new Tensor<double>(new[] { 4, 4 },
            new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }));

        // Similar: small perturbation
        var similar = new Tensor<double>(new[] { 4, 4 },
            new Vector<double>(new double[] { 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 0.99, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61 }));

        // Dissimilar: large perturbation
        var dissimilar = new Tensor<double>(new[] { 4, 4 },
            new Vector<double>(new double[] { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4 }));

        var ssimSimilar = ssim.Compute(original, similar);
        var ssimDissimilar = ssim.Compute(original, dissimilar);

        Assert.True(ssimSimilar > ssimDissimilar,
            $"SSIM for similar images ({ssimSimilar}) should be > SSIM for dissimilar images ({ssimDissimilar})");
    }

    #endregion
}
