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

            // Forward pass under the tape, capturing each layer's output node.
            var outputs = new Tensor<T>[layers.Count];
            var current = input;
            for (int i = 0; i < layers.Count; i++)
            {
                current = layers[i].Forward(current);
                outputs[i] = current;
            }
            var predictionNode = current; // final (post-softmax) network output, tracked

            // Constant output error e = prediction - target (detached from the tape).
            var predConst = Detach(predictionNode);
            var targetTensor = BuildTargetTensor(target, predConst.Shape.ToArray());
            var outputError = SubtractConst(predConst, targetTensor);

            // Collect the trainable layers (those with at least one trainable parameter tensor), in order.
            var trainableLayers = new List<ILayer<T>>();
            var trainableParams = new List<IReadOnlyList<Tensor<T>>>();
            var trainableOutputs = new List<Tensor<T>>();
            for (int i = 0; i < layers.Count; i++)
            {
                var ps = CollectParameters(layers[i]);
                if (ps.Count == 0) continue;
                trainableLayers.Add(layers[i]);
                trainableParams.Add(ps);
                trainableOutputs.Add(outputs[i]);
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
            var lossTensor = tapeLoss.ComputeTapeLoss(predictionNode, targetTensor);
            Merge(grads, tape.ComputeGradients(lossTensor, sources: trainableParams[trainableLayers.Count - 1]));

            // Hidden layers: local VJP seeded by the rule's teaching signal.
            for (int k = 0; k < trainableLayers.Count - 1; k++)
            {
                var teaching = creditLayers[k].TeachingSignal
                    ?? throw new InvalidOperationException(
                        $"Credit rule '{rule.Name}' did not set a teaching signal for hidden layer {k}.");

                var prod = engine.TensorMultiply(trainableOutputs[k], teaching);
                var allAxes = Enumerable.Range(0, prod.Shape.Length).ToArray();
                var scalar = engine.ReduceSum(prod, allAxes, keepDims: false);
                Merge(grads, tape.ComputeGradients(scalar, sources: trainableParams[k]));
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
