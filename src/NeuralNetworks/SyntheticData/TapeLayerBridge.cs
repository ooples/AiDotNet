using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Bridges the existing layer system with GradientTape for proper automatic differentiation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TapeLayerBridge eliminates the need for manual gradient computation
/// (ManualLinearBackward) in GAN generators. It uses the GradientTape autodiff system
/// to automatically compute gradients of a network output with respect to its input,
/// which is essential for WGAN-GP gradient penalty computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> In WGAN-GP training, we need to know how the discriminator's output
/// changes when we slightly change its input. Previously, this was computed manually by
/// extracting weights and multiplying backwards through each layer. This utility does
/// the same thing automatically using the GradientTape system, which:
/// - Is less error-prone (no manual weight index calculations)
/// - Handles all activation functions automatically
/// - Produces correct gradients via the chain rule
/// - Supports any layer configuration without custom backward code
/// </para>
/// <para>
/// <b>Supported Layer Types:</b>
/// <list type="bullet">
/// <item><description>FullyConnectedLayer — linear transform via TensorOperations.MatrixMultiply</description></item>
/// <item><description>DropoutLayer — skipped (gradient penalty uses eval mode)</description></item>
/// <item><description>BatchNormalizationLayer — affine transform using running statistics</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Usage Example (WGAN-GP gradient penalty):</b>
/// <code>
/// var inputGrad = TapeLayerBridge&lt;double&gt;.ComputeInputGradient(
///     interpolatedInput,
///     discriminatorLayers,
///     HiddenActivation.LeakyReLU,
///     applyActivationOnLast: false);
/// double gradNorm = ComputeL2Norm(inputGrad);
/// double penalty = (gradNorm - 1.0) * (gradNorm - 1.0);
/// </code>
/// </para>
/// </remarks>
public static class TapeLayerBridge<T>
{
    /// <summary>
    /// Specifies the activation function applied between hidden layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// GAN discriminators typically use IdentityActivation on their FullyConnectedLayers
    /// and apply activations manually. This enum tells the bridge which activation to apply
    /// between layers during the TensorOperations-based forward pass.
    /// </para>
    /// </remarks>
    public enum HiddenActivation
    {
        /// <summary>No activation (identity).</summary>
        None,
        /// <summary>LeakyReLU with alpha=0.2 (standard for GAN discriminators).</summary>
        LeakyReLU,
        /// <summary>Standard ReLU activation.</summary>
        ReLU,
        /// <summary>Sigmoid activation.</summary>
        Sigmoid,
        /// <summary>Tanh activation.</summary>
        Tanh,
        /// <summary>SiLU/Swish activation.</summary>
        SiLU,
        /// <summary>GELU activation.</summary>
        GELU
    }

    /// <summary>
    /// Computes the gradient of a layer sequence's output with respect to its input
    /// using GradientTape automatic differentiation.
    /// </summary>
    /// <param name="input">The input tensor to compute gradients for.</param>
    /// <param name="layers">The discriminator layers (FullyConnectedLayer + optional DropoutLayer).</param>
    /// <param name="activation">The activation function applied between hidden layers.</param>
    /// <param name="applyActivationOnLast">Whether to apply activation on the final layer output.</param>
    /// <param name="leakyAlpha">The alpha parameter for LeakyReLU (default 0.2).</param>
    /// <returns>The gradient tensor of the same shape as input, representing dOutput/dInput.</returns>
    /// <remarks>
    /// <para>
    /// This method replaces the manual ManualLinearBackward pattern used in GAN generators.
    /// It creates a temporary GradientTape, performs a forward pass through the layers
    /// using TensorOperations (which records to the tape), then computes the gradient
    /// automatically via reverse-mode automatic differentiation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Instead of manually calculating how each layer affects the gradient,
    /// we let the GradientTape do it automatically. This is exactly how PyTorch and TensorFlow
    /// compute gradients — by recording operations and playing them backwards.
    /// </para>
    /// <para>
    /// The layers' internal state is NOT modified by this method — weights are extracted
    /// and used as constants in the computation graph.
    /// </para>
    /// </remarks>
    public static Tensor<T> ComputeInputGradient(
        Tensor<T> input,
        IReadOnlyList<ILayer<T>> layers,
        HiddenActivation activation = HiddenActivation.LeakyReLU,
        bool applyActivationOnLast = false,
        double leakyAlpha = 0.2)
    {
        // Ensure input is 2D for matrix operations: [1, features]
        bool was1D = input.Shape.Length == 1;
        var input2D = was1D ? input.Reshape([1, input.Length]) : input;

        // Use the tensor-based GradientTape API with direct Engine calls
        using var tape = new GradientTape<T>(persistent: true);
        tape.Watch(input2D);

        // Forward through layers using Engine (auto-records to tape)
        var output = ForwardWithEngine(input2D, layers, activation, applyActivationOnLast, leakyAlpha);

        // Compute gradient: dOutput/dInput via reverse-mode AD
        var gradients = tape.Gradient(output, createGraph: true);

        if (gradients.TryGetValue(input2D, out var inputGrad))
        {
            return was1D ? inputGrad.Reshape([inputGrad.Length]) : inputGrad;
        }

        // Fallback: return zero gradient
        return new Tensor<T>(input.Shape.ToArray());
    }

    /// <summary>
    /// Computes the WGAN-GP gradient penalty for a discriminator given interpolated samples.
    /// </summary>
    /// <param name="interpolated">Interpolated samples between real and fake data.</param>
    /// <param name="layers">The discriminator layers.</param>
    /// <param name="activation">The activation function between hidden layers.</param>
    /// <param name="leakyAlpha">The alpha parameter for LeakyReLU.</param>
    /// <returns>The gradient penalty value (||grad||_2 - 1)^2.</returns>
    /// <remarks>
    /// <para>
    /// Computes the complete gradient penalty: GP = (||∇_x D(x)||_2 - 1)^2
    /// where x is an interpolation between real and fake samples.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The gradient penalty ensures the discriminator is a smooth function.
    /// It checks that the gradient norm is close to 1 everywhere between real and fake data.
    /// If the gradient is too large or too small, a penalty is applied to encourage smoothness.
    /// This stabilizes GAN training and prevents mode collapse.
    /// </para>
    /// </remarks>
    public static double ComputeGradientPenalty(
        Tensor<T> interpolated,
        IReadOnlyList<ILayer<T>> layers,
        HiddenActivation activation = HiddenActivation.LeakyReLU,
        double leakyAlpha = 0.2)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        var inputGrad = ComputeInputGradient(
            interpolated, layers, activation,
            applyActivationOnLast: false, leakyAlpha: leakyAlpha);

        // Compute L2 norm of the gradient
        double gradNormSq = 0;
        for (int i = 0; i < inputGrad.Length; i++)
        {
            double g = numOps.ToDouble(inputGrad[i]);
            gradNormSq += g * g;
        }
        double gradNorm = Math.Sqrt(gradNormSq + 1e-12);

        // Gradient penalty: (||grad||_2 - 1)^2
        double deviation = gradNorm - 1.0;
        return deviation * deviation;
    }

    /// <summary>
    /// Forwards a tensor through layers using direct Engine calls (auto-records to active tape).
    /// </summary>
    private static Tensor<T> ForwardWithEngine(
        Tensor<T> input,
        IReadOnlyList<ILayer<T>> layers,
        HiddenActivation activation,
        bool applyActivationOnLast,
        double leakyAlpha)
    {
        var engine = AiDotNetEngine.Current;
        var current = input;

        // Filter out dropout layers (gradient penalty uses eval mode)
        var activeLayers = new List<ILayer<T>>();
        foreach (var layer in layers)
        {
            if (layer is not DropoutLayer<T>)
                activeLayers.Add(layer);
        }

        for (int i = 0; i < activeLayers.Count; i++)
        {
            bool isLast = (i == activeLayers.Count - 1);
            var layer = activeLayers[i];

            if (layer is FullyConnectedLayer<T>)
            {
                var weights = layer.GetWeights();
                if (weights == null)
                    throw new InvalidOperationException(
                        "FullyConnectedLayer has null weights. Ensure initialization before gradient computation.");

                var weightsT = engine.TensorTranspose(weights);
                current = engine.TensorMatMul(current, weightsT);

                var biases = layer.GetBiases();
                if (biases != null)
                {
                    var biasReshaped = biases.Reshape([1, biases.Length]);
                    current = engine.TensorAdd(current, biasReshaped);
                }
            }
            else
            {
                // Opaque forward: use layer.Forward + record as single op for backward
                var layerInput = current;
                var output = layer.Forward(current);
                var tape = GradientTape<T>.Current;
                if (tape is not null)
                {
                    tape.RecordOp($"opaque_{layer.GetType().Name}", [layerInput], output,
                        grad => [layer.Backward(grad)]);
                }
                current = output;
            }

            // Apply activation
            if (!isLast || applyActivationOnLast)
            {
                current = activation switch
                {
                    HiddenActivation.ReLU => engine.TensorReLU(current),
                    HiddenActivation.Sigmoid => engine.TensorSigmoid(current),
                    HiddenActivation.Tanh => engine.TensorTanh(current),
                    HiddenActivation.SiLU => engine.TensorSiLU(current),
                    HiddenActivation.GELU => engine.TensorGELU(current),
                    HiddenActivation.LeakyReLU => engine.TensorLeakyReLU(current, MathHelper.GetNumericOperations<T>().FromDouble(leakyAlpha)),
                    _ => current,
                };
            }
        }

        return current;
    }

    /// <summary>
    /// Exports an MLP-based generator network as a JIT-compilable computation graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This helper constructs a computation graph by chaining fully connected layers,
    /// optional batch normalization layers, and activation functions. It supports the
    /// CTGAN-style residual architecture where the original input is concatenated back
    /// at each hidden layer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Most GAN generators use the same basic MLP structure for
    /// their generator network. This method converts that structure into a computation
    /// graph that the JIT compiler can optimize for 2-10x faster generation.
    /// The graph excludes column-specific output activations (Tanh per continuous column,
    /// Softmax per categorical group), which are applied separately after the MLP forward.
    /// </para>
    /// </remarks>
    /// <param name="inputNodes">List to populate with the input (noise) variable node.</param>
    /// <param name="inputSize">The size of the noise input vector.</param>
    /// <param name="hiddenLayers">The hidden fully connected layers (excluding output).</param>
    /// <param name="bnLayers">Batch normalization layers (one per hidden layer), or null.</param>
    /// <param name="outputLayer">The final output fully connected layer.</param>
    /// <param name="hiddenAct">Activation to apply between hidden layers.</param>
    /// <param name="outputAct">Activation to apply on the output layer (use None for identity).</param>
    /// <param name="useResidualConcat">If true, concatenates the original input at each hidden layer.</param>
    /// <returns>The output computation node.</returns>
    public static ComputationNode<T> ExportMLPGeneratorGraph(
        List<ComputationNode<T>> inputNodes,
        int inputSize,
        IReadOnlyList<ILayer<T>> hiddenLayers,
        IReadOnlyList<ILayer<T>>? bnLayers,
        ILayer<T> outputLayer,
        HiddenActivation hiddenAct = HiddenActivation.ReLU,
        HiddenActivation outputAct = HiddenActivation.None,
        bool useResidualConcat = false)
    {
        var inputNode = TensorOperations<T>.Variable(
            new Tensor<T>([1, inputSize]), "generator_input", requiresGradient: false);
        inputNodes.Add(inputNode);

        var current = inputNode;

        for (int i = 0; i < hiddenLayers.Count; i++)
        {
            // Residual concatenation (CTGAN-style skip connections)
            if (useResidualConcat && i > 0)
            {
                current = TensorOperations<T>.Concat(
                    new List<ComputationNode<T>> { current, inputNode }, axis: 1);
            }

            // Forward through FC layer
            var fcInputs = new List<ComputationNode<T>> { current };
            current = hiddenLayers[i].ExportComputationGraph(fcInputs);

            // Forward through BN layer if present
            if (bnLayers is not null && i < bnLayers.Count)
            {
                var bnInputs = new List<ComputationNode<T>> { current };
                current = bnLayers[i].ExportComputationGraph(bnInputs);
            }

            // Apply hidden activation
            current = ApplyComputationNodeActivation(current, hiddenAct, 0.2);
        }

        // Output layer with optional residual concat
        if (useResidualConcat)
        {
            current = TensorOperations<T>.Concat(
                new List<ComputationNode<T>> { current, inputNode }, axis: 1);
        }

        var outInputs = new List<ComputationNode<T>> { current };
        current = outputLayer.ExportComputationGraph(outInputs);

        // Apply output activation
        current = ApplyComputationNodeActivation(current, outputAct, 0.2);

        return current;
    }

    /// <summary>
    /// Applies activation using ComputationNode API (legacy, for ExportMLPGeneratorGraph).
    /// </summary>
    private static ComputationNode<T> ApplyComputationNodeActivation(
        ComputationNode<T> node,
        HiddenActivation activation,
        double leakyAlpha)
    {
        return activation switch
        {
            HiddenActivation.LeakyReLU => TensorOperations<T>.LeakyReLU(node, leakyAlpha),
            HiddenActivation.ReLU => TensorOperations<T>.ReLU(node),
            HiddenActivation.Sigmoid => TensorOperations<T>.Sigmoid(node),
            HiddenActivation.Tanh => TensorOperations<T>.Tanh(node),
            HiddenActivation.SiLU => TensorOperations<T>.Swish(node),
            HiddenActivation.GELU => TensorOperations<T>.GELU(node),
            HiddenActivation.None => node,
            _ => node,
        };
    }
}
