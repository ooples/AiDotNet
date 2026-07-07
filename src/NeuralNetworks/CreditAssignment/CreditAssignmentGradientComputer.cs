using System.Linq;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Produces the flat per-parameter gradient vector for a neural network using a pluggable
/// <see cref="ICreditRule{T}"/> instead of end-to-end back-propagation.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>Mechanism (tape vector-Jacobian products).</b> The network's forward pass is run under a gradient tape,
/// capturing each trainable layer's output node. The rule supplies a <i>teaching signal</i> at each hidden
/// layer's output (for Direct Feedback Alignment, a fixed random projection of the network output error). For each
/// hidden layer, the engine forms the scalar <c>sum(output ⊙ teachingSignal)</c> and backpropagates it through
/// <i>only that layer</i> to its own parameters — a local VJP that automatically supplies the layer's activation
/// Jacobian. The output layer is trained with the exact loss gradient. Because the local VJP works for any layer
/// type, this supports dense, multi-head attention, feed-forward, LayerNorm and embedding layers — so DFA trains
/// Transformers, not just dense stacks.
/// </para>
/// <para>
/// The returned vector is laid out exactly like the network's flat gradient (via <c>GetParameterChunks</c>), so
/// the optimizer, batching and scheduler consume it unchanged.
/// </para>
/// </remarks>
internal static class CreditAssignmentGradientComputer<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    public static Vector<T> ComputeGradients(
        NeuralNetworkBase<T> network,
        Tensor<T> input,
        Tensor<T> target,
        ICreditRule<T> rule,
        Random random,
        ILossFunction<T> lossFunction)
    {
        if (lossFunction is not LossFunctionBase<T> tapeLoss)
        {
            throw new NotSupportedException(
                "Credit-rule (non-backprop) training requires a tape-capable loss (a LossFunctionBase<T> such as " +
                "CategoricalCrossEntropyLoss or MeanSquaredErrorLoss).");
        }

        IEngine engine = AiDotNetEngine.Current;
        var layers = network.Layers;
        bool prevTrainingMode = network.IsTrainingMode;
        network.SetTrainingMode(true);

        try
        {
            using var tape = new GradientTape<T>();

            // Forward pass under the tape, capturing each layer's INPUT and output node.
            // layerInputs[i] is the tensor fed INTO layer i (needed to re-run a single hidden
            // layer in isolation for its local teaching-signal VJP — see the hidden loop below).
            var outputs = new Tensor<T>[layers.Count];
            var layerInputs = new Tensor<T>[layers.Count];
            var current = input;
            for (int i = 0; i < layers.Count; i++)
            {
                layerInputs[i] = current;
                current = layers[i].Forward(current);
                outputs[i] = current;
            }
            var predictionNode = current; // final (post-softmax) network output, tracked

            // Constant output error e = (prediction - target) / batch (detached from the tape).
            // The 1/batch scaling matches the MEAN-over-batch reduction the tape loss uses for the
            // exact output-layer gradient: without it the hidden-layer teaching VJP (which SUMS the
            // local Jacobian over the batch) would be ~batchSize larger than the output-layer step,
            // so DFA would take enormous, destabilizing hidden-layer steps and diverge.
            var predConst = Detach(predictionNode);
            var targetTensor = BuildTargetTensor(target, predConst.Shape.ToArray());
            var outputError = SubtractConst(predConst, targetTensor);
            int batchSize = predConst.Shape.Length > 0 ? predConst.Shape[0] : 1;
            if (batchSize > 1)
                outputError = ScaleConst(outputError, Ops.FromDouble(1.0 / batchSize));

            // Collect the trainable layers (those with at least one trainable parameter tensor), in order.
            var trainableLayers = new List<ILayer<T>>();
            var trainableParams = new List<IReadOnlyList<Tensor<T>>>();
            var trainableOutputs = new List<Tensor<T>>();
            var trainableInputs = new List<Tensor<T>>(); // detached input fed into each trainable layer
            for (int i = 0; i < layers.Count; i++)
            {
                var ps = CollectParameters(layers[i]);
                if (ps.Count == 0) continue;
                trainableLayers.Add(layers[i]);
                trainableParams.Add(ps);
                trainableOutputs.Add(outputs[i]);
                trainableInputs.Add(Detach(layerInputs[i]));
            }

            if (trainableLayers.Count == 0)
            {
                throw new InvalidOperationException(
                    "Credit-rule training found no trainable layers with parameters.");
            }

            // Build the credit-layer views for the rule.
            var creditLayers = new List<CreditLayer<T>>(trainableLayers.Count);
            for (int k = 0; k < trainableLayers.Count; k++)
            {
                bool isOutput = k == trainableLayers.Count - 1;
                creditLayers.Add(new CreditLayer<T>(
                    index: k,
                    isOutputLayer: isOutput,
                    output: trainableOutputs[k],
                    weights: TryGetWeightMatrix(trainableLayers[k])));
            }

            var context = new CreditAssignmentContext<T>(creditLayers, outputError, Ops, random);
            rule.Initialize(context);
            rule.ComputeTeachingSignals(context);

            var grads = new Dictionary<Tensor<T>, Tensor<T>>();

            // Output layer: exact loss gradient (handles the softmax head + matched loss correctly).
            // This is the sole ComputeGradients call on the main tape, so no graph-retention flag
            // is needed — the tape is consumed once and never re-walked.
            var lossTensor = tapeLoss.ComputeTapeLoss(predictionNode, targetTensor);
            Merge(grads, tape.ComputeGradients(lossTensor, sources: trainableParams[trainableLayers.Count - 1]));

            // Hidden layers: local teaching-signal VJP, each on its OWN fresh single-shot tape.
            //
            // DFA credit assignment is LOCAL by construction — layer i's parameter gradient depends
            // only on layer i's own Jacobian (input_i -> output_i) contracted with the teaching signal
            // B_i·e, NOT on any cross-layer backward chain. Re-running just that one layer under a
            // fresh tape (with its input detached to a constant) computes exactly that local Jacobian
            // and sidesteps the fragility of repeatedly walking one shared tape: the persistent/compiled
            // backward fast path prunes the graph to the FIRST call's sources and severs every later
            // layer's nodes (all hidden grads came back zero), and forcing createGraph to retain the
            // graph still silently dropped the EmbeddingLayer's scatter-add backward. A per-layer
            // single-shot tape is the same well-tested path normal training uses, so every layer type
            // — embeddings included — produces a correct gradient.
            for (int k = 0; k < trainableLayers.Count - 1; k++)
            {
                var teaching = creditLayers[k].TeachingSignal
                    ?? throw new InvalidOperationException(
                        $"Credit rule '{rule.Name}' did not set a teaching signal for hidden layer {k}.");

                using var localTape = new GradientTape<T>();
                var localOutput = trainableLayers[k].Forward(trainableInputs[k]);
                var prod = engine.TensorMultiply(localOutput, teaching);
                var allAxes = Enumerable.Range(0, prod.Shape.Length).ToArray();
                var scalar = engine.ReduceSum(prod, allAxes, keepDims: false);
                Merge(grads, localTape.ComputeGradients(scalar, sources: trainableParams[k]));
            }

            return Flatten(network, grads);
        }
        finally
        {
            network.SetTrainingMode(prevTrainingMode);
        }
    }

    // ---- helpers -------------------------------------------------------------------------------

    private static void Merge(Dictionary<Tensor<T>, Tensor<T>> into, Dictionary<Tensor<T>, Tensor<T>> from)
    {
        foreach (var kv in from)
            into[kv.Key] = kv.Value;
    }

    /// <summary>Recursively collects a layer's trainable parameter tensors (including composite sub-layers), deduped by reference.</summary>
    private static List<Tensor<T>> CollectParameters(ILayer<T> layer)
    {
        var result = new List<Tensor<T>>();
        var seen = new HashSet<Tensor<T>>();
        Collect(layer, result, seen);
        return result;

        static void Collect(ILayer<T> l, List<Tensor<T>> acc, HashSet<Tensor<T>> seen)
        {
            if (l is ITrainableLayer<T> t)
            {
                foreach (var p in t.GetTrainableParameters())
                    if (p is not null && p.Length > 0 && seen.Add(p))
                        acc.Add(p);
            }
            if (l is LayerBase<T> lb)
            {
                foreach (var sub in lb.GetSubLayers())
                    Collect(sub, acc, seen);
            }
        }
    }

    private static Matrix<T>? TryGetWeightMatrix(ILayer<T> layer)
    {
        var w = (layer as LayerBase<T>)?.GetWeights();
        if (w is null || w.Shape.Length != 2) return null;
        var m = new Matrix<T>(w.Shape[0], w.Shape[1]);
        for (int i = 0; i < w.Shape[0]; i++)
            for (int j = 0; j < w.Shape[1]; j++)
                m[i, j] = w[i, j];
        return m;
    }

    private static Tensor<T> Detach(Tensor<T> t) => new Tensor<T>(t.Shape.ToArray(), t.ToVector());

    private static Tensor<T> SubtractConst(Tensor<T> a, Tensor<T> b)
    {
        var va = a.ToVector();
        var vb = b.ToVector();
        var r = new Vector<T>(va.Length);
        for (int i = 0; i < va.Length; i++)
            r[i] = Ops.Subtract(va[i], vb[i]);
        return new Tensor<T>(a.Shape.ToArray(), r);
    }

    private static Tensor<T> ScaleConst(Tensor<T> a, T scale)
    {
        var va = a.ToVector();
        var r = new Vector<T>(va.Length);
        for (int i = 0; i < va.Length; i++)
            r[i] = Ops.Multiply(va[i], scale);
        return new Tensor<T>(a.Shape.ToArray(), r);
    }

    /// <summary>Aligns the target to the prediction shape (one-hot expanding class-index targets), as a constant tensor.</summary>
    private static Tensor<T> BuildTargetTensor(Tensor<T> target, int[] predShape)
    {
        if (target.Shape.Length == predShape.Length && target.Shape.ToArray().SequenceEqual(predShape))
            return new Tensor<T>(predShape, target.ToVector());

        int batch = predShape[0];
        int outFeatures = predShape[predShape.Length - 1];

        bool isIndexShape =
            (target.Shape.Length == 1 && target.Shape[0] == batch) ||
            (target.Shape.Length == 2 && target.Shape[0] == batch && target.Shape[1] == 1);

        if (predShape.Length == 2 && isIndexShape && outFeatures > 1)
        {
            var t = new Tensor<T>(predShape); // zeros
            for (int b = 0; b < batch; b++)
            {
                T raw = target.Shape.Length == 1 ? target[b] : target[b, 0];
                int cls = (int)Math.Round(Ops.ToDouble(raw));
                if (cls >= 0 && cls < outFeatures)
                    t[b, cls] = Ops.One;
            }
            return t;
        }

        if (predShape.Length == 2 && isIndexShape && outFeatures == 1)
        {
            var t = new Tensor<T>(predShape);
            for (int b = 0; b < batch; b++)
                t[b, 0] = target.Shape.Length == 1 ? target[b] : target[b, 0];
            return t;
        }

        throw new NotSupportedException(
            $"Credit-rule training could not align a target of shape [{string.Join(",", target.Shape.ToArray())}] to a " +
            $"prediction of shape [{string.Join(",", predShape)}].");
    }

    /// <summary>Flattens the reference-keyed gradient map into a vector matching the network's parameter layout.</summary>
    private static Vector<T> Flatten(NeuralNetworkBase<T> network, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var flat = new List<T>();
        foreach (var paramTensor in network.GetParameterChunks())
        {
            if (paramTensor is null || paramTensor.Length == 0) continue;
            if (grads.TryGetValue(paramTensor, out var grad))
            {
                for (int i = 0; i < grad.Length; i++)
                    flat.Add(grad[i]);
            }
            else
            {
                for (int i = 0; i < paramTensor.Length; i++)
                    flat.Add(Ops.Zero);
            }
        }
        return new Vector<T>(flat.ToArray());
    }
}
