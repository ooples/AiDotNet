using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>Result of one local-objective gradient computation.</summary>
internal sealed class LocalObjectiveGradients<T>
{
    public LocalObjectiveGradients(Dictionary<Tensor<T>, Tensor<T>> gradients, Tensor<T> outputLoss, Tensor<T> prediction, int hiddenLayers, int localObjectives)
    {
        Gradients = gradients; OutputLoss = outputLoss; Prediction = prediction; HiddenLayers = hiddenLayers; LocalObjectives = localObjectives;
    }

    /// <summary>Gradient of every network parameter that received one (keyed by parameter tensor reference).</summary>
    public Dictionary<Tensor<T>, Tensor<T>> Gradients { get; }

    /// <summary>The scalar output objective (what the step reports as its loss).</summary>
    public Tensor<T> OutputLoss { get; }

    /// <summary>The readout's prediction for the step.</summary>
    public Tensor<T> Prediction { get; }

    /// <summary>Hidden trainable layers offered a local objective.</summary>
    public int HiddenLayers { get; }

    /// <summary>Hidden layers whose rule actually returned a local objective.</summary>
    public int LocalObjectives { get; }
}

/// <summary>
/// Computes one training step's gradients for an <see cref="ILocalObjectiveCreditRule{T}"/>: a single forward in which
/// the input of every trainable pre-readout layer is detached, a combined objective
/// <c>output_objective(readout(h_last)) + Σ_k local_objective_k(h_k)</c>, and one backward. Because each hidden
/// layer's output depends only on its own parameters, the gradient a hidden layer receives is exactly the gradient of
/// its own local objective; the readout layers receive the output objective plus every local objective that was
/// routed through them.
/// </summary>
internal static class LocalObjectiveGradientComputer<T>
{
    public static LocalObjectiveGradients<T> Compute(
        NeuralNetworkBase<T> network,
        Tensor<T> input,
        Tensor<T> target,
        ILocalObjectiveCreditRule<T> rule,
        LossFunctionBase<T> loss,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> alignTarget)
    {
        var layers = network.Layers;
        int readoutCount = rule.ReadoutLayerCount;
        if (readoutCount < 1 || readoutCount >= layers.Count)
        {
            throw new InvalidOperationException(
                $"Credit rule '{rule.Name}' asked for {readoutCount} readout layers, but the network has {layers.Count} " +
                "layers; the readout must be at least one layer and leave at least one layer before it.");
        }
        int readoutStart = layers.Count - readoutCount;

        var paramsOf = new List<Tensor<T>>[layers.Count];
        for (int i = 0; i < layers.Count; i++)
            paramsOf[i] = CollectParameters(layers[i]);

        var hiddenIdx = new List<int>();
        for (int i = 0; i < readoutStart; i++)
            if (paramsOf[i].Count > 0) hiddenIdx.Add(i);
        if (hiddenIdx.Count == 0)
            throw new InvalidOperationException($"Credit rule '{rule.Name}' found no trainable layer before the readout.");

        // Everything that outlives the tape (detached activations) must not come from the step arena; see the
        // matching note in CreditAssignmentGradientComputer.
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        bool prevTraining = network.IsTrainingMode;
        network.SetTrainingMode(true);
        try
        {
            using var tape = new GradientTape<T>();
            IEngine engine = AiDotNetEngine.Current;

            var hiddenOut = new Tensor<T>?[layers.Count];
            var current = input;
            for (int i = 0; i < readoutStart; i++)
            {
                if (paramsOf[i].Count > 0 && i > 0)
                    // stop-gradient: layer i's params see nothing from below. StopGradient is a device copy with no
                    // backward (torch's detach); the host round-trip it replaces downloaded and re-uploaded every
                    // hidden activation each step on a GPU engine.
                    current = engine.StopGradient(current);
                current = layers[i].Forward(current);
                if (paramsOf[i].Count > 0) hiddenOut[i] = current;
            }

            Tensor<T> Readout(Tensor<T> h)
            {
                var r = h;
                for (int i = readoutStart; i < layers.Count; i++) r = layers[i].Forward(r);
                return r;
            }

            // The last hidden layer feeds the readout un-detached, so it (and the readout) are trained exactly by the
            // output objective, as in back-propagation through the readout alone.
            var prediction = Readout(current);
            var alignedTarget = alignTarget(prediction, target);
            Tensor<T> LossFn(Tensor<T> p, Tensor<T> t) => ReduceToScalar(engine, loss.ComputeTapeLoss(p, t));
            // A single-layer readout that fuses itself with the loss (an LM head + cross-entropy) serves every hidden
            // layer's readout loss without putting a [rows, vocabulary] tensor on the tape per layer.
            var fusedReadout = readoutCount == 1 ? layers[readoutStart] as IFusedReadoutLossLayer<T> : null;
            Tensor<T> ReadoutLossFn(Tensor<T> h, Tensor<T> t) =>
                fusedReadout?.TryComputeFusedLoss(h, t, loss) is { } fused ? ReduceToScalar(engine, fused) : LossFn(Readout(h), t);
            var step = new LocalObjectiveStep<T>(input, alignedTarget, engine, Readout, LossFn, hiddenIdx.Count, ReadoutLossFn);

            var defaultLoss = LossFn(prediction, alignedTarget);
            var outputObjective = rule.BuildOutputObjective(step, prediction, defaultLoss) ?? defaultLoss;
            var combined = outputObjective;

            int locals = 0;
            for (int k = 0; k < hiddenIdx.Count; k++)
            {
                int i = hiddenIdx[k];
                var local = rule.BuildLocalObjective(new LocalObjectiveContext<T>(step, k, i, layers[i],
                    hiddenOut[i] ?? throw new InvalidOperationException($"Hidden layer {i} produced no output.")));
                if (local is null) continue;
                combined = engine.TensorAdd(combined, ReduceToScalar(engine, local));
                locals++;
            }

            var sources = new List<Tensor<T>>();
            var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var ps in paramsOf)
                foreach (var p in ps)
                    if (seen.Add(p)) sources.Add(p);
            var aux = rule.AuxiliaryParameters;
            foreach (var p in aux)
                if (seen.Add(p)) sources.Add(p);

            var all = tape.ComputeGradients(combined, sources: sources);

            var grads = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            var auxGrads = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            var auxSet = new HashSet<Tensor<T>>(aux, TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var kv in all)
            {
                if (auxSet.Contains(kv.Key)) auxGrads[kv.Key] = kv.Value;
                else if (seen.Contains(kv.Key)) grads[kv.Key] = kv.Value;
            }
            rule.ApplyAuxiliaryGradients(auxGrads);

            // The output objective and prediction outlive this tape (the caller reads the loss value and the
            // prediction's shape). Only the loss handed to ComputeGradients is kept automatically; on a GPU engine the
            // rest are released with the tape, so reading the loss threw "GPU intermediate of a finished GradientTape".
            tape.Retain(outputObjective);
            tape.Retain(prediction);
            return new LocalObjectiveGradients<T>(grads, outputObjective, prediction, hiddenIdx.Count, locals);
        }
        finally
        {
            network.SetTrainingMode(prevTraining);
        }
    }

    private static Tensor<T> ReduceToScalar(IEngine engine, Tensor<T> t)
    {
        if (t.Shape.Length == 0) return t;
        var axes = Enumerable.Range(0, t.Shape.Length).ToArray();
        return engine.ReduceSum(t, axes, keepDims: false);
    }


    internal static List<Tensor<T>> CollectParameters(ILayer<T> layer)
    {
        var result = new List<Tensor<T>>();
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        Collect(layer, result, seen);
        return result;

        static void Collect(ILayer<T> l, List<Tensor<T>> acc, HashSet<Tensor<T>> seen)
        {
            if (l is ITrainableLayer<T> t)
                foreach (var p in t.GetTrainableParameters())
                    if (p is not null && p.Length > 0 && seen.Add(p)) acc.Add(p);
            if (l is LayerBase<T> lb)
                foreach (var sub in lb.GetSubLayers())
                    Collect(sub, acc, seen);
        }
    }
}
