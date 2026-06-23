using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

namespace AiDotNetTestConsole;

/// <summary>
/// #1670 diagnostic: is NTM's run-to-run variance order-dependent in CONSTRUCTION
/// (a process-global RNG) or in TRAINING (process-global compute scratch)?
/// Mirrors the test harness's seed override, then fingerprints a freshly-built
/// NTM's parameters (a) cold and (b) after another NTM has been built+trained.
/// </summary>
internal static class NtmDeterminismProbe
{
    public static void Run()
    {
        NeuralNetworkArchitecture<float>.DefaultRandomSeedOverride = 1234;

        string cold = Fingerprint(BuildAndProbe());

        // Perturb process-global state the way earlier NTM tests would: build + train.
        var throwaway = new NeuralTuringMachine<float>();
        var x = Rand(new[] { 128 });
        var probe = throwaway.Predict(x);            // triggers lazy init via eval (like MoreData's probe)
        var y = Rand(probe.Shape.ToArray());
        for (int i = 0; i < 30; i++) throwaway.Train(x, y);

        string afterTrain = Fingerprint(BuildAndProbe());

        Console.WriteLine($"cold        param-fingerprint: {cold}");
        Console.WriteLine($"after-train param-fingerprint: {afterTrain}");
        Console.WriteLine(cold == afterTrain
            ? "=> CONSTRUCTION is order-INDEPENDENT (init deterministic). Variance is in TRAINING/forward global state."
            : "=> CONSTRUCTION is order-DEPENDENT. A process-global RNG is consumed during build — found it.");

        // Now the decisive check: are TRAINED weights order-dependent? Train a fresh
        // NTM on a fixed (seeded) task cold, vs after a throwaway build+train has
        // dirtied process-global state. If they differ, TRAINING consumes
        // non-isolated process-global scratch (the verify-gate scratch class).
        // Mirror MoreData_ShouldNotDegrade EXACTLY (build → probe Predict → CLONE →
        // train network1) and fingerprint network1's trained params across repeats.
        // If they vary, Clone() shares mutable state with network1 (NTM's stateful
        // memory), corrupting training run-to-run — the real MoreData flake cause.
        Console.WriteLine("--- MoreData-mirror (clone path), with perturbing ops between repeats ---");
        Console.WriteLine($"  baseline           : {Fingerprint(MoreDataMirror())}");
        Perturb();
        Console.WriteLine($"  after perturb ops  : {Fingerprint(MoreDataMirror())}");

        string trainedCold = Fingerprint(BuildTrainProbe());
        var t2 = new NeuralTuringMachine<float>();
        var tx = Rand(new[] { 128 });
        var tp = t2.Predict(tx);
        var ty = Rand(tp.Shape.ToArray());
        for (int i = 0; i < 30; i++) t2.Train(tx, ty);
        string trainedAfter = Fingerprint(BuildTrainProbe());
        Console.WriteLine($"trained cold        : {trainedCold}");
        Console.WriteLine($"trained after-perturb: {trainedAfter}");
        Console.WriteLine(trainedCold == trainedAfter
            ? "=> TRAINING is order-INDEPENDENT too. Variance must be elsewhere."
            : "=> TRAINING is order-DEPENDENT: a fresh NTM trained on identical seeded data lands on DIFFERENT weights depending on prior process activity => non-isolated Tensors process-global training scratch (package-level).");
    }

    // Build a fresh NTM, force lazy init via a probe forward, return its parameter vector.
    private static Vector<float> BuildAndProbe()
    {
        var ntm = new NeuralTuringMachine<float>();
        ntm.Predict(Rand(new[] { 128 }));   // force lazy weight init
        return ntm.GetParameters();
    }

    // Build a fresh NTM, train it on a fixed seeded task, return its parameter vector.
    private static Vector<float> BuildTrainProbe()
    {
        var ntm = new NeuralTuringMachine<float>();
        var x = Rand(new[] { 128 });
        var probe = ntm.Predict(x);
        var y = Rand(probe.Shape.ToArray());
        for (int i = 0; i < 50; i++) ntm.Train(x, y);
        return ntm.GetParameters();
    }

    // Mirror MoreData: build network1, probe-Predict, clone to network2, train
    // network1 50 iters, return network1's params. (network2 is built like the test
    // does, to reproduce any clone side-effect on network1.)
    private static Vector<float> MoreDataMirror()
    {
        var rng1 = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var network1 = new NeuralTuringMachine<float>();
        var input = RandFrom(rng1, new[] { 128 });
        var probe = network1.Predict(input);
        var target = RandFrom(rng1, probe.Shape.ToArray());
        var network2 = (NeuralTuringMachine<float>)network1.Clone();
        for (int i = 0; i < 50; i++) network1.Train(input, target);
        GC.KeepAlive(network2);
        return network1.GetParameters();
    }

    // Operations the OTHER NeuralTuringMachineTests methods perform, which run before
    // MoreData in the full class. If any dirties process-global state, the post-perturb
    // MoreData-mirror fingerprint will differ from baseline.
    private static void Perturb()
    {
        var net = new NeuralTuringMachine<float>();
        var x = Rand(new[] { 128 });
        net.Predict(x);
        try { net.GetNamedLayerActivations(x); } catch { }
        try { var bytes = net.Serialize(); var n2 = new NeuralTuringMachine<float>(); n2.Deserialize(bytes); n2.Predict(x); } catch { }
        net.SetTrainingMode(true);
        var y = Rand(net.Predict(x).Shape.ToArray());
        for (int i = 0; i < 10; i++) net.Train(x, y);
        net.SetTrainingMode(false);
        var clone = (NeuralTuringMachine<float>)net.Clone();
        clone.Predict(x);
    }

    private static Tensor<float> RandFrom(Random rng, int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static string Fingerprint(Vector<float> p)
    {
        double sum = 0, sumSq = 0;
        for (int i = 0; i < p.Length; i++) { double v = p[i]; sum += v; sumSq += v * v; }
        return $"len={p.Length} sum={sum:F8} sumSq={sumSq:F8} first3=[{(p.Length > 0 ? p[0] : 0):F8},{(p.Length > 1 ? p[1] : 0):F8},{(p.Length > 2 ? p[2] : 0):F8}]";
    }

    private static Tensor<float> Rand(int[] shape)
    {
        var rng = new Random(7);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
