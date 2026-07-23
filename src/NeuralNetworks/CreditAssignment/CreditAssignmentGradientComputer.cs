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

        var layers = network.Layers;
        bool prevTrainingMode = network.IsTrainingMode;
        network.SetTrainingMode(true);

        try
        {
            // ---- Capture pass: run the full forward once to identify the trainable layers, each layer's input
            // activation, and the network's output error. Values are detached to constants; graph is discarded.
            var trainableLayers = new List<ILayer<T>>();
            var trainableParams = new List<IReadOnlyList<Tensor<T>>>();
            var trainableInputs = new List<Tensor<T>>();   // detached (constant) input to each trainable layer
            var trainableOutputs = new List<Tensor<T>>();  // detached (constant) output of each trainable layer
            var trainableLayerIndex = new List<int>();      // index into `layers` of each trainable layer
            Tensor<T> targetTensor;
            Tensor<T> outputError;

            using (var captureTape = new GradientTape<T>())
            {
                var inputs = new Tensor<T>[layers.Count];
                var outputs = new Tensor<T>[layers.Count];
                var current = input;
                for (int i = 0; i < layers.Count; i++)
                {
                    inputs[i] = current;
                    current = layers[i].Forward(current);
                    outputs[i] = current;
                }

                var predConst = Detach(current);
                targetTensor = BuildTargetTensor(target, predConst.Shape.ToArray());
                outputError = SubtractConst(predConst, targetTensor);

                for (int i = 0; i < layers.Count; i++)
                {
                    var ps = CollectParameters(layers[i]);
                    if (ps.Count == 0) continue;
                    trainableLayers.Add(layers[i]);
                    trainableParams.Add(ps);
                    trainableInputs.Add(Detach(inputs[i]));
                    trainableOutputs.Add(Detach(outputs[i]));
                    trainableLayerIndex.Add(i);
                }
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
                    input: trainableInputs[k],
                    output: trainableOutputs[k],
                    weights: TryGetWeightMatrix(trainableLayers[k])));
            }

            var context = new CreditAssignmentContext<T>(creditLayers, outputError, targetTensor, Ops, random);
            rule.Initialize(context);
            rule.ComputeTeachingSignals(context);

            int last = trainableLayers.Count - 1;

            // ---- Gradient pass: a SINGLE combined objective backpropagated once. The teaching-driven design needs
            // each trainable layer's gradient to be a LOCAL VJP (its output treated as a function of only its own
            // parameters). We achieve that by re-running the forward and DETACHING the input immediately before each
            // trainable layer, so that layer's output depends on nothing below it. The objective is
            //     loss(prediction)  +  Σ_hidden  sum(output_k ⊙ teachingSignal_k)
            // whose gradient w.r.t. each layer's parameters is exactly that layer's intended update: the exact loss
            // gradient for the output layer, the teaching-driven VJP for every hidden layer, with no cross-layer
            // leakage. A single backward pass is required because GradientTape is single-use — one backward frees
            // the graph, so calling it once per layer silently zeroed every layer after the first.
            var isTrainable = new bool[layers.Count];
            var trainOrderOf = new int[layers.Count];
            for (int t = 0; t < trainableLayerIndex.Count; t++)
            {
                isTrainable[trainableLayerIndex[t]] = true;
                trainOrderOf[trainableLayerIndex[t]] = t;
            }

            using var tape = new GradientTape<T>();
            IEngine engine = AiDotNetEngine.Current; // the tape's recording engine (installed for this scope)

            var trainOutputNode = new Tensor<T>[trainableLayers.Count];
            var current2 = input;
            for (int i = 0; i < layers.Count; i++)
            {
                if (isTrainable[i])
                    current2 = Detach(current2); // isolate this trainable layer's params from everything below
                current2 = layers[i].Forward(current2);
                if (isTrainable[i])
                    trainOutputNode[trainOrderOf[i]] = current2;
            }
            var predictionNode = current2;

            // Output layer: exact loss gradient (handles the softmax head + matched loss correctly).
            Tensor<T> combined = ReduceToScalar(engine, tapeLoss.ComputeTapeLoss(predictionNode, targetTensor));
            var allSources = new List<Tensor<T>>(trainableParams[last]);

            // Hidden layers: teaching-driven local VJP, seeded by the rule's teaching signal.
            for (int k = 0; k < last; k++)
            {
                var teaching = creditLayers[k].TeachingSignal
                    ?? throw new InvalidOperationException(
                        $"Credit rule '{rule.Name}' did not set a teaching signal for hidden layer {k}.");
                var prod = engine.TensorMultiply(trainOutputNode[k], teaching);
                combined = engine.TensorAdd(combined, ReduceToScalar(engine, prod));
                allSources.AddRange(trainableParams[k]);
            }

            var grads = new Dictionary<Tensor<T>, Tensor<T>>();
            Merge(grads, tape.ComputeGradients(combined, sources: allSources));

            // Feedback-learning rules (Kolen-Pollack / Direct Kolen-Pollack) update their learned feedback state
            // once per step, using the same activations, output error and teaching signals just consumed above.
            if (rule is IFeedbackLearningRule<T> feedbackLearner)
                feedbackLearner.OnParametersUpdated(context);

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

    /// <summary>Reduces a tensor to a single scalar (sum over every axis) so heterogeneous terms can be added into one objective.</summary>
    private static Tensor<T> ReduceToScalar(IEngine engine, Tensor<T> t)
    {
        if (t.Shape.Length == 0) return t;
        var axes = Enumerable.Range(0, t.Shape.Length).ToArray();
        return engine.ReduceSum(t, axes, keepDims: false);
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
