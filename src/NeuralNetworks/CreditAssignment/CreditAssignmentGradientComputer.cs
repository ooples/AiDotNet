using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Produces the flat per-parameter gradient vector for a dense feed-forward network using a pluggable
/// <see cref="ICreditRule{T}"/> instead of reverse-mode back-propagation. The result is laid out exactly like
/// <c>NeuralNetworkBase&lt;T&gt;.GetParameters()</c> (per layer: weights row-major, then biases), so the optimizer,
/// batching and scheduler consume it unchanged.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>Supported networks.</b> The credit-rule path targets a stack of <see cref="FullyConnectedLayer{T}"/>
/// layers (each carrying its own element-wise activation) with a matched output loss (MSE+linear or
/// cross-entropy+softmax). This is the domain in which the feedback-alignment literature is defined. Any other
/// layer type causes a clear <see cref="NotSupportedException"/> so the limitation is explicit rather than
/// silently wrong.
/// </para>
/// </remarks>
internal static class CreditAssignmentGradientComputer<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Runs a forward pass, builds the credit-assignment context, invokes <paramref name="rule"/>, and returns
    /// the flattened gradient vector.
    /// </summary>
    public static Vector<T> ComputeGradients(
        IReadOnlyList<ILayer<T>> layers,
        Tensor<T> input,
        Tensor<T> target,
        ICreditRule<T> rule,
        Random random)
    {
        var denseLayers = ValidateAndCollect(layers);

        var x = ToMatrix(input);                       // [B, inputFeatures]
        int batch = x.Rows;

        // Forward pass: capture per-layer input, pre-activation z, activated output a.
        var creditLayers = new List<ICreditLayer<T>>(denseLayers.Count);
        var currentInput = x;
        for (int i = 0; i < denseLayers.Count; i++)
        {
            var layer = denseLayers[i];
            bool isOutput = i == denseLayers.Count - 1;

            var weights = TensorToMatrix(GetWeights(layer));   // [out, in]
            var bias = TensorToVector(GetBiases(layer));       // [out]

            if (weights.Columns != currentInput.Columns)
            {
                throw new NotSupportedException(
                    $"Credit-rule training: layer {i} expects input width {weights.Columns} but received " +
                    $"{currentInput.Columns}. The credit-rule path supports a plain dense feed-forward stack.");
            }

            var z = Linear(currentInput, weights, bias);        // [B, out]
            var a = ApplyActivation(layer, z, isHidden: !isOutput);

            creditLayers.Add(new CreditLayer<T>(
                index: i,
                isOutputLayer: isOutput,
                weights: weights,
                input: currentInput,
                preActivation: z,
                output: a,
                scalarActivation: layer.ScalarActivation,
                numOps: Ops));

            currentInput = a;
        }

        var prediction = currentInput;                          // [B, outputFeatures]
        var targetMatrix = ToTargetMatrix(target, prediction);  // [B, outputFeatures]
        var outputError = Subtract(prediction, targetMatrix);   // prediction − target

        var context = new CreditAssignmentContext<T>(
            creditLayers, x, prediction, targetMatrix, outputError, Ops, random);

        rule.Initialize(context);
        rule.ComputeUpdates(context);

        return Flatten(creditLayers);
    }

    private static List<FullyConnectedLayer<T>> ValidateAndCollect(IReadOnlyList<ILayer<T>> layers)
    {
        var dense = new List<FullyConnectedLayer<T>>();
        foreach (var layer in layers)
        {
            if (layer is FullyConnectedLayer<T> fc)
            {
                dense.Add(fc);
            }
            else
            {
                throw new NotSupportedException(
                    $"Credit-rule (non-backprop) training currently supports a stack of " +
                    $"{nameof(FullyConnectedLayer<T>)} layers only; found '{layer.GetType().Name}'. " +
                    "Use the default back-propagation path for this architecture, or express the model as a " +
                    "dense feed-forward network.");
            }
        }

        if (dense.Count == 0)
        {
            throw new NotSupportedException("Credit-rule training requires at least one FullyConnectedLayer.");
        }

        return dense;
    }

    private static Matrix<T> Linear(Matrix<T> input, Matrix<T> weights, Vector<T> bias)
    {
        // z = input · Wᵀ + b   (weights are [out, in]; output = input · Wᵀ)
        var z = input.Multiply(weights.Transpose());  // [B, out]
        for (int r = 0; r < z.Rows; r++)
            for (int c = 0; c < z.Columns; c++)
                z[r, c] = Ops.Add(z[r, c], bias[c]);
        return z;
    }

    private static Matrix<T> ApplyActivation(FullyConnectedLayer<T> layer, Matrix<T> z, bool isHidden)
    {
        if (layer.ScalarActivation is not null)
        {
            // Apply the activation ROW-WISE via the vector overload, matching how the real layer applies it.
            // For genuinely element-wise activations (ReLU/Tanh/Sigmoid) this is the same as per-element;
            // for a row-coupled activation exposed through IActivationFunction (e.g. Softmax, whose
            // Activate(Vector) does a proper row soft-max) this is essential — applying Activate(scalar)
            // element-wise would be wrong.
            var a = new Matrix<T>(z.Rows, z.Columns);
            for (int r = 0; r < z.Rows; r++)
            {
                var activated = layer.ScalarActivation.Activate(z.GetRow(r));
                for (int c = 0; c < z.Columns; c++)
                    a[r, c] = activated[c];
            }
            return a;
        }

        if (layer.VectorActivation is not null)
        {
            if (isHidden)
            {
                throw new NotSupportedException(
                    "Credit-rule training: hidden FullyConnectedLayer layers must use an element-wise " +
                    "(scalar) activation so f'(z) is well-defined. Vector activations (e.g. softmax) are only " +
                    "supported on the output layer.");
            }

            var a = new Matrix<T>(z.Rows, z.Columns);
            for (int r = 0; r < z.Rows; r++)
            {
                var activated = layer.VectorActivation.Activate(z.GetRow(r));
                for (int c = 0; c < z.Columns; c++)
                    a[r, c] = activated[c];
            }
            return a;
        }

        // No activation → linear/identity.
        return z;
    }

    private static Vector<T> Flatten(List<ICreditLayer<T>> layers)
    {
        int total = 0;
        foreach (var layer in layers)
            total += layer.OutputDim * layer.InputDim + layer.OutputDim;

        var flat = new Vector<T>(total);
        int idx = 0;
        foreach (var layer in layers)
        {
            var wg = layer.WeightGradient;
            for (int o = 0; o < wg.Rows; o++)
                for (int i = 0; i < wg.Columns; i++)
                    flat[idx++] = wg[o, i];
            var bg = layer.BiasGradient;
            for (int o = 0; o < bg.Length; o++)
                flat[idx++] = bg[o];
        }
        return flat;
    }

    // --- Tensor / Matrix conversion helpers -------------------------------------------------

    private static Matrix<T> ToMatrix(Tensor<T> t)
    {
        if (t.Shape.Length == 2)
        {
            var m = new Matrix<T>(t.Shape[0], t.Shape[1]);
            for (int i = 0; i < t.Shape[0]; i++)
                for (int j = 0; j < t.Shape[1]; j++)
                    m[i, j] = t[i, j];
            return m;
        }
        if (t.Shape.Length == 1)
        {
            var m = new Matrix<T>(1, t.Shape[0]);
            for (int j = 0; j < t.Shape[0]; j++)
                m[0, j] = t[j];
            return m;
        }
        throw new NotSupportedException(
            $"Credit-rule training expects rank-1 or rank-2 input tensors; got rank {t.Shape.Length}.");
    }

    private static Matrix<T> TensorToMatrix(Tensor<T> t)
    {
        var m = new Matrix<T>(t.Shape[0], t.Shape[1]);
        for (int i = 0; i < t.Shape[0]; i++)
            for (int j = 0; j < t.Shape[1]; j++)
                m[i, j] = t[i, j];
        return m;
    }

    private static Vector<T> TensorToVector(Tensor<T> t)
    {
        var v = new Vector<T>(t.Shape[0]);
        for (int i = 0; i < t.Shape[0]; i++)
            v[i] = t[i];
        return v;
    }

    private static Matrix<T> ToTargetMatrix(Tensor<T> target, Matrix<T> prediction)
    {
        int batch = prediction.Rows;
        int outFeatures = prediction.Columns;
        var m = new Matrix<T>(batch, outFeatures);

        // Case 1: target already matches [B, outFeatures].
        if (target.Shape.Length == 2 && target.Shape[0] == batch && target.Shape[1] == outFeatures)
        {
            for (int i = 0; i < batch; i++)
                for (int j = 0; j < outFeatures; j++)
                    m[i, j] = target[i, j];
            return m;
        }

        // Case 2: class-index targets → one-hot (outFeatures > 1). Accept [B], [B,1].
        bool isIndexShape =
            (target.Shape.Length == 1 && target.Shape[0] == batch) ||
            (target.Shape.Length == 2 && target.Shape[0] == batch && target.Shape[1] == 1);

        if (isIndexShape && outFeatures > 1)
        {
            for (int i = 0; i < batch; i++)
            {
                T raw = target.Shape.Length == 1 ? target[i] : target[i, 0];
                int cls = (int)Math.Round(Ops.ToDouble(raw));
                if (cls >= 0 && cls < outFeatures)
                    m[i, cls] = Ops.One;
            }
            return m;
        }

        // Case 3: scalar regression [B] / [B,1] with outFeatures == 1.
        if (isIndexShape && outFeatures == 1)
        {
            for (int i = 0; i < batch; i++)
                m[i, 0] = target.Shape.Length == 1 ? target[i] : target[i, 0];
            return m;
        }

        throw new NotSupportedException(
            $"Credit-rule training could not align a target of shape [{string.Join(",", target.Shape)}] " +
            $"to a prediction of shape [{batch},{outFeatures}].");
    }

    private static Matrix<T> Subtract(Matrix<T> a, Matrix<T> b)
    {
        var result = new Matrix<T>(a.Rows, a.Columns);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < a.Columns; j++)
                result[i, j] = Ops.Subtract(a[i, j], b[i, j]);
        return result;
    }

    private static Tensor<T> GetWeights(FullyConnectedLayer<T> layer)
        => layer.GetWeights() ?? throw new NotSupportedException("FullyConnectedLayer has no weights (unresolved shape).");

    private static Tensor<T> GetBiases(FullyConnectedLayer<T> layer)
        => layer.GetBiases() ?? throw new NotSupportedException("FullyConnectedLayer has no biases (unresolved shape).");
}
