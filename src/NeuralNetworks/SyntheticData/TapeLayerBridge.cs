using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
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
            current = ((Layers.LayerBase<T>)hiddenLayers[i]).ExportComputationGraph(fcInputs);

            // Forward through BN layer if present
            if (bnLayers is not null && i < bnLayers.Count)
            {
                var bnInputs = new List<ComputationNode<T>> { current };
                current = ((Layers.LayerBase<T>)bnLayers[i]).ExportComputationGraph(bnInputs);
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
        current = ((Layers.LayerBase<T>)outputLayer).ExportComputationGraph(outInputs);

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
