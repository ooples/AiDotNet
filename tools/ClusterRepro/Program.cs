using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// DBM Clone bug: cloned predict differs from trained predict by ~3%.
// Want to know which layer's parameters differ between original and clone.

var network = new DeepBoltzmannMachine<double>();  // matches the failing test's config (128 input)

var trainInput = new Tensor<double>(new[] { 128 });
var trainTarget = new Tensor<double>(new[] { 128 });
var rng = new System.Random(42);
for (int i = 0; i < trainInput.Length; i++) trainInput[i] = rng.NextDouble() * 2 - 1;
for (int i = 0; i < trainTarget.Length; i++) trainTarget[i] = rng.NextDouble() * 2 - 1;

for (int i = 0; i < 10; i++) network.Train(trainInput, trainTarget);

network.SetTrainingMode(false);

var probe = new Tensor<double>(new[] { 128 });
for (int i = 0; i < probe.Length; i++) probe[i] = rng.NextDouble() * 2 - 1;

var trainedOut = network.Predict(probe);

// Clone-twice probe: clone the trained network, then clone the
// clone. If output drifts further with each clone, the bug is
// memory-layout-dependent; if the second clone matches the first,
// then once SetParameters has put the tensor in a canonical layout,
// further round-trips are stable.
var cloned = (DeepBoltzmannMachine<double>)network.Clone();
cloned.SetTrainingMode(false);
var clonedOut = cloned.Predict(probe);

var cloned2 = (DeepBoltzmannMachine<double>)cloned.Clone();
cloned2.SetTrainingMode(false);
var cloned2Out = cloned2.Predict(probe);

double maxC1C2 = 0;
for (int i = 0; i < clonedOut.Length; i++)
{
    double d = System.Math.Abs(clonedOut[i] - cloned2Out[i]);
    if (d > maxC1C2) maxC1C2 = d;
}
System.Console.WriteLine($"Clone1 vs Clone2 maxDiff: {maxC1C2:E3}");

double maxOC1 = 0;
for (int i = 0; i < trainedOut.Length; i++)
{
    double d = System.Math.Abs(trainedOut[i] - clonedOut[i]);
    if (d > maxOC1) maxOC1 = d;
}
System.Console.WriteLine($"Original vs Clone1 maxDiff: {maxOC1:E3}");

System.Console.WriteLine($"Original layer count: {network.Layers.Count}");
System.Console.WriteLine($"Cloned layer count:   {cloned.Layers.Count}");
// Iterate up to the minimum layer count so a clone with FEWER layers
// (the exact failure mode this repro tool exists to diagnose) doesn't
// throw IndexOutOfRangeException at line 58 and lose the per-layer
// diagnostics that come after. Surface any count mismatch as its own
// labelled log line for the missing range so we can still see WHICH
// layers got dropped on the clone side.
int sharedLayerCount = System.Math.Min(network.Layers.Count, cloned.Layers.Count);
for (int i = 0; i < sharedLayerCount; i++)
{
    var origParams = network.Layers[i].GetParameters();
    var clonedParams = cloned.Layers[i].GetParameters();
    int origLen = origParams.Length;
    int clonedLen = clonedParams.Length;
    float maxDelta = 0f;
    int comparisons = System.Math.Min(origLen, clonedLen);
    for (int j = 0; j < comparisons; j++)
    {
        double d = System.Math.Abs(origParams[j] - clonedParams[j]);
        if (d > maxDelta) maxDelta = (float)d;
    }
    var status = origLen == clonedLen ? "OK" : "LEN-MISMATCH";
    System.Console.WriteLine($"Layer {i} {network.Layers[i].GetType().Name}: orig={origLen} cloned={clonedLen} maxDelta={maxDelta:E3} {status}");
}
// Report the layers that exist on only ONE side so the repro output
// shows exactly which layers got dropped (vs the prior throw that hid
// the entire diagnostic).
for (int i = sharedLayerCount; i < network.Layers.Count; i++)
{
    System.Console.WriteLine($"Layer {i} {network.Layers[i].GetType().Name}: orig=present cloned=MISSING (clone has only {cloned.Layers.Count} layers)");
}
for (int i = sharedLayerCount; i < cloned.Layers.Count; i++)
{
    System.Console.WriteLine($"Layer {i} {cloned.Layers[i].GetType().Name}: orig=MISSING cloned=present (orig has only {network.Layers.Count} layers)");
}

System.Console.WriteLine();
System.Console.WriteLine($"Trained output: [{string.Join(",", System.Linq.Enumerable.Range(0, trainedOut.Length).Select(i => $"{trainedOut[i]:F4}"))}]");
System.Console.WriteLine($"Cloned output:  [{string.Join(",", System.Linq.Enumerable.Range(0, clonedOut.Length).Select(i => $"{clonedOut[i]:F4}"))}]");
