using System.Diagnostics;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

// Measures what a training STEP costs, which is the number the shard deaths and the
// 700 MB/step optimizer figure are both really about. Deliberately a tiny model: at this
// size neither runtime is compute-bound, so any per-step gap is structural, not kernels.
int InDim = args.Length > 0 ? int.Parse(args[0]) : 4;
int Hidden = args.Length > 1 ? int.Parse(args[1]) : 8;
int Batch  = args.Length > 2 ? int.Parse(args[2]) : 32;
int Steps  = args.Length > 3 ? int.Parse(args[3]) : 100;

var layers = new List<ILayer<float>>
{
    new InputLayer<float>(InDim),
    new DenseLayer<float>(Hidden, activationFunction: new ReLUActivation<float>()),
    new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
};
var arch = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
    inputSize: InDim, outputSize: 1, layers: layers);
var model = new FeedForwardNeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());

var x = new Tensor<float>(new[] { Batch, InDim });
var y = new Tensor<float>(new[] { Batch, 1 });
var rng = RandomHelper.CreateSeededRandom(1234);
for (int i = 0; i < Batch; i++)
{
    for (int j = 0; j < InDim; j++) x[i, j] = (float)(rng.NextDouble() * 2 - 1);
    y[i, 0] = (float)(rng.NextDouble());
}

// Warm up so JIT, first-forward materialization and one-time caches are not billed to the steps.
model.Train(x, y);

// TRACE MODE: print the PID and wait, so dotnet-trace attaches AFTER warm-up and captures only
// steady-state steps. Otherwise the profile is dominated by JIT and kernel compilation.
if (Environment.GetEnvironmentVariable("ALLOCPROBE_WAIT") == "1")
{
    Console.WriteLine($"PID {Environment.ProcessId} waiting for tracer...");
    Console.Out.Flush();
    Thread.Sleep(TimeSpan.FromSeconds(12));
}

GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
long heapBefore = GC.GetTotalMemory(forceFullCollection: true);
int[] gcBefore = { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) };
var sw = Stopwatch.StartNew();

for (int s = 0; s < Steps; s++) model.Train(x, y);
sw.Stop();

// PHASE DECOMPOSITION. Train = forward + backward + optimizer update. Measuring each surface
// separately attributes the per-step bytes without needing allocation-tick symbolication:
//   Predict          -> forward only
//   ComputeGradients -> forward + backward
//   Train            -> all three
// so (Train - ComputeGradients) is the optimizer's own share, which is the number the 700 MB/step
// finding is about.
static long Measure(int n, Action f)
{
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    long b = GC.GetTotalAllocatedBytes(precise: true);
    for (int i = 0; i < n; i++) f();
    return (GC.GetTotalAllocatedBytes(precise: true) - b) / n;
}

long fwd  = Measure(Steps, () => model.Predict(x));
long grad = Measure(Steps, () => model.ComputeGradients(x, y));
long full = Measure(Steps, () => model.Train(x, y));
Console.WriteLine();
Console.WriteLine($"PHASE forward            : {fwd / 1024.0:F1} KB/step");
Console.WriteLine($"PHASE forward+backward   : {grad / 1024.0:F1} KB/step");
Console.WriteLine($"PHASE full train         : {full / 1024.0:F1} KB/step");
Console.WriteLine($"PHASE backward share     : {(grad - fwd) / 1024.0:F1} KB/step");
Console.WriteLine($"PHASE optimizer share    : {(full - grad) / 1024.0:F1} KB/step   (UNRELIABLE: Train and ComputeGradients are separate paths, not nested)");

// OPTIMIZER IN ISOLATION. UpdateParameters(Vector, Vector) is the real per-step surface, and it
// RETURNS a Vector -- so the signature alone guarantees at least one parameter-sized allocation
// every step. PyTorch's Adam writes in place and allocates nothing at steady state. Measuring it
// away from forward/backward gives a number that is actually attributable.
{
    int n = model.GetParameters().Length;
    var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(model);
    var pvec = model.GetParameters();
    var gvec = new Vector<float>(n);
    for (int i = 0; i < n; i++) gvec[i] = 0.001f;

    adam.UpdateParameters(pvec, gvec);   // warm up lazy moment buffers

    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    long ob = GC.GetTotalAllocatedBytes(precise: true);
    for (int i = 0; i < Steps; i++) pvec = adam.UpdateParameters(pvec, gvec);
    long oa = GC.GetTotalAllocatedBytes(precise: true);

    double perStep = (oa - ob) / (double)Steps;
    double oneVec = n * sizeof(float);
    Console.WriteLine($"OPTIMIZER alloc/step     : {perStep / 1024.0:F1} KB   ({perStep / oneVec:F1}x one parameter vector)");
    Console.WriteLine($"OPTIMIZER param vector   : {oneVec / 1024.0:F1} KB  ({n} params)");
}
sw.Start();

sw.Stop();
long allocAfter = GC.GetTotalAllocatedBytes(precise: true);
long heapAfter = GC.GetTotalMemory(forceFullCollection: true);

double perStepMb = (allocAfter - allocBefore) / (double)Steps / (1024 * 1024);
double retainedMb = (heapAfter - heapBefore) / (double)(1024 * 1024);
Console.WriteLine($"CFG in={InDim} hid={Hidden} batch={Batch} steps={Steps}");
Console.WriteLine($"alloc/step       : {perStepMb:F3} MB");
Console.WriteLine($"total allocated  : {(allocAfter - allocBefore) / (1024.0 * 1024):F1} MB");
Console.WriteLine($"retained delta   : {retainedMb:F3} MB   <- climbs => leak, flat => churn only");
Console.WriteLine($"gc gen0/1/2      : {GC.CollectionCount(0) - gcBefore[0]}/{GC.CollectionCount(1) - gcBefore[1]}/{GC.CollectionCount(2) - gcBefore[2]}");
Console.WriteLine($"ms/step          : {sw.Elapsed.TotalMilliseconds / Steps:F3}");
Console.WriteLine($"params           : {model.GetParameters().Length}");

// ── CORRECTNESS: does ComputeGradients still return the right numbers in the right order? ──
// The one-allocation rewrite changed how the flat gradient vector is ASSEMBLED (sizing pass +
// span fill instead of List.Add), so the risk is a wrong length or a wrong offset, either of
// which scrambles which gradient lands on which parameter. Finite differences catch both:
// a misaligned vector decorrelates from the numerical gradient immediately.
if (Environment.GetEnvironmentVariable("ALLOCPROBE_GRADCHECK") == "1")
{
    var loss = new MeanSquaredErrorLoss<float>();
    var analytic = model.ComputeGradients(x, y, loss);
    var p0 = model.GetParameters();
    Console.WriteLine($"\nGRADCHECK params={p0.Length} analytic={analytic.Length} " +
                      (p0.Length == analytic.Length ? "LENGTH-OK" : "LENGTH-MISMATCH"));

    float LossAt(Vector<float> p)
    {
        model.SetParameters(p);
        var pred = model.Predict(x);
        float sum = 0f;
        for (int i = 0; i < pred.Length; i++) { float d = pred[i] - y[i]; sum += d * d; }
        return sum / pred.Length;
    }

    const float Eps = 1e-2f;
    var rngIdx = RandomHelper.CreateSeededRandom(7);
    double worst = 0; int worstAt = -1; int checkedCount = 0;
    for (int trial = 0; trial < 40; trial++)
    {
        int k = rngIdx.Next(Math.Min(p0.Length, analytic.Length));
        var pPlus = p0.Clone(); pPlus[k] = p0[k] + Eps;
        var pMinus = p0.Clone(); pMinus[k] = p0[k] - Eps;
        double numeric = (LossAt(pPlus) - LossAt(pMinus)) / (2.0 * Eps);
        model.SetParameters(p0);
        double a = analytic[k];
        double scale = Math.Max(1e-3, Math.Abs(a) + Math.Abs(numeric));
        double rel = Math.Abs(a - numeric) / scale;
        checkedCount++;
        if (rel > worst) { worst = rel; worstAt = k; }
    }
    Console.WriteLine($"GRADCHECK checked={checkedCount} worst-rel-err={worst:F4} at index {worstAt}");
    Console.WriteLine(worst < 0.05 ? "GRADCHECK PASS" : "GRADCHECK FAIL");
}
