using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The RNN transducer loss (Graves 2012) against brute-force enumeration of the lattice's alignments, and its gradient
/// against central finite differences.
/// </summary>
public class TransducerLossTests
{
    private static Tensor<double> LogSoftmaxLattice(int frames, int width, int vocabulary, int seed)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var lattice = new Tensor<double>(new[] { frames, width, vocabulary });
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                var logits = new double[vocabulary];
                double max = double.NegativeInfinity, sum = 0;
                for (int k = 0; k < vocabulary; k++) { logits[k] = 2 * random.NextDouble() - 1; max = Math.Max(max, logits[k]); }
                for (int k = 0; k < vocabulary; k++) sum += Math.Exp(logits[k] - max);
                for (int k = 0; k < vocabulary; k++) lattice[t, u, k] = logits[k] - max - Math.Log(sum);
            }
        return lattice;
    }

    // Every alignment: at node (t, u) either emit blank (move to t + 1) or the next label (move to u + 1); the path ends by
    // emitting blank at (T - 1, U).
    private static double BruteForce(Tensor<double> lattice, int[] labels, int blank, int t, int u)
    {
        int frames = lattice.Shape[0];
        if (t == frames - 1 && u == labels.Length) return Math.Exp(lattice[t, u, blank]);
        double total = 0;
        if (t < frames - 1) total += Math.Exp(lattice[t, u, blank]) * BruteForce(lattice, labels, blank, t + 1, u);
        if (u < labels.Length) total += Math.Exp(lattice[t, u, labels[u]]) * BruteForce(lattice, labels, blank, t, u + 1);
        return total;
    }

    [Fact(Timeout = 60000)]
    public async Task Loss_IsTheNegativeLogOfEveryAlignmentsProbability()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var labels = new[] { 2, 1, 3 };
        var lattice = LogSoftmaxLattice(4, labels.Length + 1, 5, seed: 7);
        double loss = TransducerLoss.Compute(engine, lattice, labels, blank: 0)[0];
        Assert.Equal(-Math.Log(BruteForce(lattice, labels, 0, 0, 0)), loss, 10);
    }

    [Fact(Timeout = 60000)]
    public async Task Gradient_MatchesFiniteDifferences()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var labels = new[] { 1, 2 };
        var lattice = LogSoftmaxLattice(3, labels.Length + 1, 4, seed: 11);
        Tensor<double> grad;
        using (var tape = new GradientTape<double>())
        {
            var loss = TransducerLoss.Compute(engine, lattice, labels, blank: 0);
            grad = tape.ComputeGradients(loss, new[] { lattice })[lattice];
        }
        const double h = 1e-6;
        for (int i = 0; i < lattice.Length; i++)
        {
            var plus = lattice.Clone(); plus.Data.Span[i] += h;
            var minus = lattice.Clone(); minus.Data.Span[i] -= h;
            double numeric = (TransducerLoss.Compute(engine, plus, labels, 0)[0] - TransducerLoss.Compute(engine, minus, labels, 0)[0]) / (2 * h);
            Assert.Equal(numeric, grad.Data.Span[i], 6);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FastEmit_ScalesOnlyTheLabelEmissionGradients()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var labels = new[] { 1, 2 };
        var lattice = LogSoftmaxLattice(3, labels.Length + 1, 4, seed: 13);
        Tensor<double> Grad(double lambda)
        {
            using var tape = new GradientTape<double>();
            var loss = TransducerLoss.Compute(engine, lattice, labels, 0, lambda);
            return tape.ComputeGradients(loss, new[] { lattice })[lattice];
        }
        var plain = Grad(0.0);
        var fastEmit = Grad(0.01);
        for (int t = 0; t < 3; t++)
            for (int u = 0; u < 3; u++)
            {
                Assert.Equal(plain[t, u, 0], fastEmit[t, u, 0], 12);
                if (u < 2) Assert.Equal(1.01 * plain[t, u, labels[u]], fastEmit[t, u, labels[u]], 12);
            }
    }

    // The engine form (anti-diagonal log-adds, autodiff gradient) against the host alpha-beta recursions with Graves'
    // explicit gradient, on lattices with one frame, no labels, more labels than frames, and FastEmit.
    [Theory(Timeout = 120000)]
    [InlineData(1, 0, 5, 0.0)]
    [InlineData(1, 3, 5, 0.0)]
    [InlineData(6, 0, 4, 0.0)]
    [InlineData(3, 7, 9, 0.0)]
    [InlineData(20, 6, 7, 0.0)]
    [InlineData(20, 6, 7, 0.25)]
    public async Task EngineForm_MatchesTheHostRecursions_LossAndGradient(int frames, int length, int vocabulary, double lambda)
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var random = new Random(frames * 31 + length);
        var labels = new int[length];
        for (int i = 0; i < length; i++) labels[i] = 1 + random.Next(vocabulary - 1);
        var lattice = LogSoftmaxLattice(frames, length + 1, vocabulary, seed: 23 + frames + length);
        (double Loss, Tensor<double> Grad) Run(bool host)
        {
            using var tape = new GradientTape<double>();
            var loss = host ? TransducerLoss.ComputeOnHost(engine, lattice, labels, 0, lambda) : TransducerLoss.Compute(engine, lattice, labels, 0, lambda);
            return (loss[0], tape.ComputeGradients(loss, new[] { lattice })[lattice]);
        }
        var reference = Run(host: true);
        var device = Run(host: false);
        Assert.Equal(reference.Loss, device.Loss, 9);
        for (int i = 0; i < lattice.Length; i++)
            Assert.Equal(reference.Grad.Data.Span[i], device.Grad.Data.Span[i], 9);

        // Single precision: the same within float rounding.
        var single = new Tensor<float>(lattice.Shape.ToArray());
        for (int i = 0; i < lattice.Length; i++) single.Data.Span[i] = (float)lattice.Data.Span[i];
        using var floatTape = new GradientTape<float>();
        var floatLoss = TransducerLoss.Compute(engine, single, labels, 0, lambda);
        var floatGrad = floatTape.ComputeGradients(floatLoss, new[] { single })[single];
        Assert.True(Math.Abs(floatLoss[0] - reference.Loss) <= 1e-4 * Math.Max(1, Math.Abs(reference.Loss)),
            $"float loss {floatLoss[0]} vs {reference.Loss}");
        for (int i = 0; i < lattice.Length; i++)
            Assert.True(Math.Abs(floatGrad.Data.Span[i] - reference.Grad.Data.Span[i]) <= 1e-4,
                $"float gradient {i}: {floatGrad.Data.Span[i]} vs {reference.Grad.Data.Span[i]}");
    }

    // The engine form of the token-and-duration loss against the host recursions and their explicit gradient: one frame,
    // no labels, more labels than frames, durations with and without 0, and a duration longer than the utterance.
    [Theory(Timeout = 120000)]
    [InlineData(1, 0, 5, new[] { 0, 1 })]
    [InlineData(1, 2, 5, new[] { 0, 1, 2 })]
    [InlineData(6, 0, 4, new[] { 1, 2 })]
    [InlineData(3, 6, 7, new[] { 0, 1, 2, 3, 4 })]
    [InlineData(12, 4, 6, new[] { 0, 1, 2, 3, 4 })]
    [InlineData(9, 3, 5, new[] { 1, 2, 4, 20 })]
    public async Task TokenAndDuration_EngineForm_MatchesTheHostRecursions_LossAndGradient(int frames, int length, int vocabulary, int[] durations)
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var random = new Random(frames * 17 + length);
        var labels = new int[length];
        for (int i = 0; i < length; i++) labels[i] = 1 + random.Next(vocabulary - 1);
        var lattice = TdtLattice(frames, length + 1, vocabulary, durations.Length, seed: 41 + frames + length);
        (double Loss, Tensor<double> Grad) Run(bool host)
        {
            using var tape = new GradientTape<double>();
            var loss = host
                ? TransducerLoss.ComputeTokenAndDurationOnHost(engine, lattice, vocabulary, durations, labels, 0)
                : TransducerLoss.ComputeTokenAndDuration(engine, lattice, vocabulary, durations, labels, 0);
            return (loss[0], tape.ComputeGradients(loss, new[] { lattice })[lattice]);
        }
        var reference = Run(host: true);
        var device = Run(host: false);
        Assert.Equal(reference.Loss, device.Loss, 9);
        for (int i = 0; i < lattice.Length; i++)
            Assert.Equal(reference.Grad.Data.Span[i], device.Grad.Data.Span[i], 9);
    }

    // TDT paths: at (t, u) a blank with duration d >= 1 moves to (t + d, u), a label with any duration to (t + d, u + 1);
    // reaching frame T by a blank with every label emitted completes the sequence.
    private static double TdtBruteForce(Tensor<double> lattice, int vocabulary, int[] durations, int[] labels, int t, int u)
    {
        int frames = lattice.Shape[0];
        double total = 0;
        for (int i = 0; i < durations.Length; i++)
        {
            int d = durations[i], to = t + d;
            double pd = Math.Exp(lattice[t, u, vocabulary + i]);
            if (d > 0 && to == frames && u == labels.Length) total += Math.Exp(lattice[t, u, 0]) * pd;
            if (d > 0 && to < frames) total += Math.Exp(lattice[t, u, 0]) * pd * TdtBruteForce(lattice, vocabulary, durations, labels, to, u);
            if (u < labels.Length && to < frames) total += Math.Exp(lattice[t, u, labels[u]]) * pd * TdtBruteForce(lattice, vocabulary, durations, labels, to, u + 1);
        }
        return total;
    }

    private static Tensor<double> TdtLattice(int frames, int width, int vocabulary, int durations, int seed)
    {
        var tokens = LogSoftmaxLattice(frames, width, vocabulary, seed);
        var lengths = LogSoftmaxLattice(frames, width, durations, seed + 1);
        var lattice = new Tensor<double>(new[] { frames, width, vocabulary + durations });
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                for (int k = 0; k < vocabulary; k++) lattice[t, u, k] = tokens[t, u, k];
                for (int k = 0; k < durations; k++) lattice[t, u, vocabulary + k] = lengths[t, u, k];
            }
        return lattice;
    }

    [Fact(Timeout = 60000)]
    public async Task TokenAndDuration_IsTheNegativeLogOfEveryPath_AndItsGradientMatchesFiniteDifferences()
    {
        await Task.Yield();
        var engine = new CpuEngine();
        var labels = new[] { 2, 1 };
        var durations = new[] { 0, 1, 2 };
        var lattice = TdtLattice(4, labels.Length + 1, 4, durations.Length, seed: 17);
        Tensor<double> grad;
        double loss;
        using (var tape = new GradientTape<double>())
        {
            var value = TransducerLoss.ComputeTokenAndDuration(engine, lattice, 4, durations, labels, 0);
            loss = value[0];
            grad = tape.ComputeGradients(value, new[] { lattice })[lattice];
        }
        Assert.Equal(-Math.Log(TdtBruteForce(lattice, 4, durations, labels, 0, 0)), loss, 10);
        const double h = 1e-6;
        for (int i = 0; i < lattice.Length; i++)
        {
            var plus = lattice.Clone(); plus.Data.Span[i] += h;
            var minus = lattice.Clone(); minus.Data.Span[i] -= h;
            double numeric = (TransducerLoss.ComputeTokenAndDuration(engine, plus, 4, durations, labels, 0)[0]
                - TransducerLoss.ComputeTokenAndDuration(engine, minus, 4, durations, labels, 0)[0]) / (2 * h);
            Assert.Equal(numeric, grad.Data.Span[i], 6);
        }
    }
}
