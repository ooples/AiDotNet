using System;
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
