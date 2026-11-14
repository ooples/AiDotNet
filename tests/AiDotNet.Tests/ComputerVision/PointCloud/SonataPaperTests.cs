using System;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.PointCloud;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision.PointCloud;

/// <summary>
/// Sonata (Wu et al., CVPR 2025) against Pointcept's <c>sonata_v1m1_base.py</c> and
/// <c>pretrain-sonata-v1m1-0-base.py</c>: Sinkhorn-Knopp centering, the KoLeo regulariser, the EMA
/// teacher, the cosine schedules, the encoder-only backbone and a pretraining step.
/// </summary>
public class SonataPaperTests
{
    [Fact]
    public void SinkhornKnopp_MatchesPointceptIterationAndIsDoublyStochastic()
    {
        int n = 6, k = 4;
        var random = RandomHelper.CreateSeededRandom(29);
        var scores = new double[n * k];
        for (int i = 0; i < scores.Length; i++) scores[i] = random.NextDouble() * 2 - 1;

        // Pointcept sinkhorn_knopp(feat, temp=0.07, num_iter=3), transcribed in double.
        var q = new double[k, n];
        double total = 0;
        for (int p = 0; p < k; p++)
            for (int s = 0; s < n; s++) total += q[p, s] = Math.Exp(scores[s * k + p] / 0.07);
        for (int p = 0; p < k; p++)
            for (int s = 0; s < n; s++) q[p, s] /= total;
        for (int iteration = 0; iteration < 3; iteration++)
        {
            for (int p = 0; p < k; p++)
            {
                double row = 0;
                for (int s = 0; s < n; s++) row += q[p, s];
                for (int s = 0; s < n; s++) q[p, s] = q[p, s] / row / k;
            }

            for (int s = 0; s < n; s++)
            {
                double column = 0;
                for (int p = 0; p < k; p++) column += q[p, s];
                for (int p = 0; p < k; p++) q[p, s] = q[p, s] / column / n;
            }
        }

        var centering = new SinkhornKnoppCentering<double>(3);
        var targets = centering.Center(new Tensor<double>(scores, new[] { n, k }), 0.07).ToArray();
        for (int s = 0; s < n; s++)
            for (int p = 0; p < k; p++) Assert.Equal(q[p, s] * n, targets[s * k + p], 5);

        // Run to convergence: every sample row sums to 1 and every prototype column to n / k.
        var converged = SinkhornKnoppCentering<double>.Normalize(
            new Tensor<float>(scores.Select(v => (float)v).ToArray(), new[] { n, k }), 0.07, 200).ToArray();
        for (int s = 0; s < n; s++) Assert.Equal(1.0, Enumerable.Range(0, k).Sum(p => (double)converged[s * k + p]), 4);
        for (int p = 0; p < k; p++) Assert.Equal((double)n / k, Enumerable.Range(0, n).Sum(s => (double)converged[s * k + p]), 3);
    }

    [Fact]
    public void KoLeo_MatchesItsFormula()
    {
        // KoLeo = -mean_i log(||x_i - x_nn(i) + eps|| + eps) over L2-normalised rows (DINOv2 KoLeoLoss).
        int n = 7, c = 5;
        var random = RandomHelper.CreateSeededRandom(31);
        var x = new double[n * c];
        for (int i = 0; i < x.Length; i++) x[i] = random.NextDouble() * 2 - 1;
        var unit = new double[n * c];
        for (int i = 0; i < n; i++)
        {
            double norm = Math.Sqrt(Enumerable.Range(0, c).Sum(j => x[i * c + j] * x[i * c + j]));
            for (int j = 0; j < c; j++) unit[i * c + j] = x[i * c + j] / norm;
        }

        double expected = 0;
        for (int i = 0; i < n; i++)
        {
            int nearest = -1;
            double best = double.NegativeInfinity;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                double dot = Enumerable.Range(0, c).Sum(m => unit[i * c + m] * unit[j * c + m]);
                if (dot > best)
                {
                    best = dot;
                    nearest = j;
                }
            }

            double distance = Math.Sqrt(Enumerable.Range(0, c).Sum(m => Math.Pow(unit[i * c + m] - unit[nearest * c + m] + 1e-8, 2)));
            expected -= Math.Log(distance + 1e-8);
        }

        expected /= n;
        var actual = new KoLeoRegularizer<double>().Compute(new Tensor<double>(x, new[] { n, c })).ToArray();
        Assert.Single(actual);
        Assert.Equal(expected, actual[0], 9);
    }

    [Fact]
    public void EmaUpdate_IsMomentumBlendOfTeacherAndStudent()
    {
        var teacher = new Tensor<double>(new[] { 1.0, -2.0, 4.0 }, new[] { 3 });
        var student = new Tensor<double>(new[] { 3.0, 2.0, -4.0 }, new[] { 3 });
        SonataPretraining.UpdateTeacher(new[] { teacher }, new[] { student }, 0.994);
        var expected = new[] { 0.994 * 1 + 0.006 * 3, 0.994 * -2 + 0.006 * 2, 0.994 * 4 + 0.006 * -4 };
        var actual = teacher.ToArray();
        for (int i = 0; i < 3; i++) Assert.Equal(expected[i], actual[i], 12);
        Assert.Equal(new[] { 3.0, 2.0, -4.0 }, student.ToArray());
    }

    [Fact]
    public void CosineScheduler_MatchesPointcept()
    {
        // CosineScheduler(base=0.07, final=0.07, total=20, start=0.04, warmup=4): linspace then constant.
        var temperature = new SonataCosineSchedule(0.07, 0.07, 20, 0.04, 4);
        Assert.Equal(0.04, temperature.Get(0), 12);
        Assert.Equal(0.05, temperature.Get(1), 12);
        Assert.Equal(0.07, temperature.Get(3), 12);
        Assert.Equal(0.07, temperature.Get(10), 12);

        // Momentum: base 0.994, final 1, no warm-up: final + 0.5 (base - final)(1 + cos(pi i / total)).
        var momentum = new SonataCosineSchedule(0.994, 1.0, 10);
        for (int i = 0; i < 10; i++)
            Assert.Equal(1.0 + 0.5 * (0.994 - 1.0) * (1 + Math.Cos(Math.PI * i / 10)), momentum.Get(i), 12);
        Assert.Equal(1.0, momentum.Get(10), 12);
    }

    [Fact]
    public void Defaults_AreTheOfficialConfiguration()
    {
        var o = new SonataOptions();
        Assert.Equal(9, o.InputChannels);
        Assert.Equal(new[] { 3, 3, 3, 12, 3 }, o.EncoderDepths);
        Assert.Equal(new[] { 48, 96, 192, 384, 512 }, o.EncoderChannels);
        Assert.Equal(new[] { 3, 6, 12, 24, 32 }, o.EncoderNumHeads);
        Assert.Equal(2, o.GlobalViewCount);
        Assert.Equal(4, o.LocalViewCount);
        Assert.Equal(4096, o.PrototypeCount);
        Assert.Equal(256, o.HeadEmbedChannels);
        Assert.Equal(0.994, o.MomentumBase);
        Assert.Equal(1.0, o.MomentumFinal);
        Assert.Equal(3, o.SinkhornIterations);
        Assert.Equal(0.0, o.KoLeoWeight);
        Assert.Equal(0.25 + 0.25 + 0.5, o.MaskLossWeight + o.RollMaskLossWeight + o.UnmaskLossWeight, 12);

        // Head input is the features up-cast two levels: 512 + 384 + 192 = 1088 (head_in_channels).
        Assert.Equal(1088, o.EncoderChannels.Skip(o.EncoderChannels.Length - 1 - o.UpCastLevel).Sum());
    }

    [Fact]
    public void Encoder_IsEncoderOnlyAndUpCastsEveryStage()
    {
        var model = new Sonata<float>(Arch(), options: PointCloudTestOptions.SmallSonata(5));
        var encoder = model.Encoder ?? throw new InvalidOperationException("Encoder expected.");
        Assert.True(encoder.IsEncoderOnly);
        Assert.Equal(8 + 16, encoder.OutputChannels);
        var logits = model.Predict(PointCloudTestOptions.Cloud(48, 9, seed: 37));
        Assert.Equal(new[] { 48, 5 }, logits.Shape.ToArray());
    }

    [Fact]
    public void PretrainStep_DecreasesTheDistillationLossOnAToyScene()
    {
        var options = PointCloudTestOptions.SmallSonata();
        options.DropPathRate = 0.0;
        options.MomentumBase = 1.0;          // freeze the teacher so the student chases a fixed target
        options.PretrainingEpochs = 15;
        options.LearningRate = 0.003;
        options.WeightDecay = 0.0;
        options.MaxViewSize = 64;
        var model = new Sonata<float>(Arch(), options: options);
        var scene = PointCloudTestOptions.Cloud(64, 9, seed: 41);

        double before = model.EvaluatePretrainingLoss(scene, seed: 3);
        model.Pretrain(new[] { scene });
        double after = model.EvaluatePretrainingLoss(scene, seed: 3);

        Assert.False(double.IsNaN(before) || double.IsInfinity(before) || double.IsNaN(after) || double.IsInfinity(after));
        Assert.True(after < before, $"loss {before} -> {after}");
    }

    private static NeuralNetworkArchitecture<float> Arch()
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Deep, 0, 32, 32, 3, 0);
}
