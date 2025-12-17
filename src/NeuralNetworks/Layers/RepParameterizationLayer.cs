namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a reparameterization layer used in variational autoencoders (VAEs) to enable backpropagation through random sampling.
/// </summary>
/// <remarks>
/// <para>
/// The RepParameterizationLayer implements the reparameterization trick commonly used in variational autoencoders.
/// It takes an input tensor that contains means and log variances of a latent distribution, samples from this
/// distribution using the reparameterization trick, and outputs the sampled values. This approach allows
/// gradients to flow through the random sampling process, which is essential for training VAEs.
/// </para>
/// <para><b>For Beginners:</b> This layer is a special component used in variational autoencoders (VAEs).
/// 
/// Think of the RepParameterizationLayer as a clever randomizer with memory:
/// - It takes information about a range of possible values (represented by mean and variance)
/// - It generates random samples from this range
/// - It remembers how it generated these samples so it can learn during training
/// 
/// For example, in a VAE generating faces:
/// - Input might represent "average nose size is 5 with variation of ±2"
/// - This layer randomly picks a specific nose size (like 6.3) based on those statistics
/// - But it does this in a way that allows the network to learn better statistics
/// 
/// The "reparameterization trick" is what makes this possible - it separates the random sampling
/// (which can't be directly learned from) from the statistical parameters (which can be learned).
/// 
/// This layer is crucial for variational autoencoders to learn meaningful latent representations
/// while still incorporating randomness, which helps with generating diverse outputs.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RepParameterizationLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Stores the mean values extracted from the input tensor during the forward pass.
    /// </summary>
    /// <remarks>
    /// This tensor holds the mean values for each dimension of the latent space for each item in the batch.
    /// It represents the center of the distribution from which samples are drawn. The tensor is null
    /// before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastMean;

    /// <summary>
    /// Stores the log variance values extracted from the input tensor during the forward pass.
    /// </summary>
    /// <remarks>
    /// This tensor holds the log variance values for each dimension of the latent space for each item in the batch.
    /// Log variance is used instead of variance for numerical stability. It represents the spread of the
    /// distribution from which samples are drawn. The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastLogVar;

    /// <summary>
    /// Stores the random noise values used during the sampling process in the forward pass.
    /// </summary>
    /// <remarks>
    /// This tensor holds the random noise values (epsilon) drawn from a standard normal distribution
    /// during the forward pass. These values are used to generate samples from the parameterized
    /// distribution. Saving these values is necessary for the backward pass. The tensor is null
    /// before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastEpsilon;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for RepParameterizationLayer, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the RepParameterizationLayer can propagate gradients during backpropagation.
    /// Although this layer does not have trainable parameters itself, it needs to participate in the training process
    /// by correctly propagating gradients to previous layers.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can participate in the learning process.
    /// 
    /// A value of true means:
    /// - The layer can pass learning signals (gradients) backward through it
    /// - It contributes to the training of the entire network
    /// 
    /// While this layer doesn't have any internal values that it learns directly,
    /// it's designed to let learning signals flow through it to previous layers.
    /// This is critical for training a variational autoencoder.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="RepParameterizationLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor. The first dimension is the batch size, and the second dimension must be even (half for means, half for log variances).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RepParameterizationLayer with the specified input shape. The output shape
    /// is set to match the input shape except for the second dimension, which is halved since the output
    /// contains only the sampled values, not both means and log variances.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new reparameterization layer for your variational autoencoder.
    /// 
    /// When you create this layer, you specify:
    /// - inputShape: The shape of the data coming into the layer
    /// 
    /// The input is expected to contain two parts:
    /// - The first half contains the mean values for each latent dimension
    /// - The second half contains the log variance values for each latent dimension
    /// 
    /// For example, if inputShape[1] is 100, then:
    /// - The first 50 values represent means
    /// - The last 50 values represent log variances
    /// - The output will have 50 values (the sampled points)
    /// 
    /// This layer doesn't have any trainable parameters - it just performs the reparameterization operation.
    /// </para>
    /// </remarks>
    public RepParameterizationLayer(int[] inputShape)
        : base(inputShape, inputShape)
    {
    }

    /// <summary>
    /// Performs the forward pass of the reparameterization layer.
    /// </summary>
    /// <param name="input">The input tensor containing concatenated mean and log variance values.</param>
    /// <returns>The output tensor containing sampled points from the latent distribution.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the reparameterization layer. It splits the input tensor
    /// into mean and log variance parts, generates random noise (epsilon) from a standard normal distribution,
    /// and uses the reparameterization trick (z = mean + std_dev * epsilon) to sample from the latent distribution.
    /// The input, means, log variances, and epsilon values are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method samples random points from your specified distribution.
    /// 
    /// During the forward pass:
    /// 1. The layer separates the input into mean values and log variance values
    /// 2. It generates random noise values (epsilon) from a standard normal distribution
    /// 3. It calculates standard deviation values from the log variances
    /// 4. It produces samples using the formula: sample = mean + (std_dev * epsilon)
    /// 
    /// This reparameterization trick is clever because:
    /// - The randomness comes from epsilon, which is independent of what the network is learning
    /// - The mean and standard deviation can be learned and improved through backpropagation
    /// - During inference, you can either use random samples or just use the mean values
    /// 
    /// The layer saves all intermediate values for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int latentSize = input.Shape[1] / 2;

        // Use Engine.TensorSlice to split mean and logvar
        _lastMean = Engine.TensorSlice(input, [0, 0], [batchSize, latentSize]);
        _lastLogVar = Engine.TensorSlice(input, [0, latentSize], [batchSize, latentSize]);

        // Generate random epsilon using Tensor<T>.CreateRandom
        _lastEpsilon = Tensor<T>.CreateRandom(batchSize, latentSize);

        // Compute stdDev = exp(logvar * 0.5) using Engine operations
        var halfTensor = new Tensor<T>([batchSize, latentSize]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var scaledLogVar = Engine.TensorMultiply(_lastLogVar, halfTensor);
        var stdDev = Engine.TensorExp(scaledLogVar);

        // Compute output = mean + stdDev * epsilon using Engine operations
        var scaledEpsilon = Engine.TensorMultiply(stdDev, _lastEpsilon);
        var output = Engine.TensorAdd(_lastMean, scaledEpsilon);

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the reparameterization layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input (means and log variances).</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the reparameterization layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients with respect to
    /// the means and log variances based on the gradients of the output. The gradient flow through the
    /// random sampling process is what makes the reparameterization trick valuable for training.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in the means and variances would affect the loss.
    /// 
    /// During the backward pass:
    /// 1. The layer receives gradients indicating how the network's output should change
    /// 2. It calculates how changes in the mean values would affect the output
    /// 3. It calculates how changes in the log variance values would affect the output
    /// 4. It combines these into gradients for the original input (means and log variances)
    /// 
    /// The gradient for means is straightforward - changes in the mean directly affect the output.
    /// The gradient for log variances is more complex because it controls the scale of the random noise.
    /// 
    /// This backward flow of information is what allows a VAE to learn good latent representations
    /// even though it involves random sampling.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastMean == null || _lastLogVar == null || _lastEpsilon == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = outputGradient.Shape[0];
        int latentSize = outputGradient.Shape[1];

        // Compute stdDev = exp(logvar * 0.5) using Engine operations
        var halfTensor = new Tensor<T>([batchSize, latentSize]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var scaledLogVar = Engine.TensorMultiply(_lastLogVar, halfTensor);
        var stdDev = Engine.TensorExp(scaledLogVar);

        // Gradient for mean = outputGradient (unchanged)
        var gradMean = outputGradient;

        // Gradient for log variance = outputGradient * epsilon * stdDev * 0.5
        var gradLogVar = Engine.TensorMultiply(
            Engine.TensorMultiply(
                Engine.TensorMultiply(outputGradient, _lastEpsilon),
                stdDev),
            halfTensor);

        // Concatenate gradients: [gradMean, gradLogVar]
        var inputGradient = new Tensor<T>([batchSize, latentSize * 2]);
        inputGradient = Engine.TensorSetSlice(inputGradient, gradMean, [0, 0]);
        inputGradient = Engine.TensorSetSlice(inputGradient, gradLogVar, [0, latentSize]);

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the reparameterization trick and uses autodiff
    /// to compute gradients. The forward computation is: z = mean + exp(logvar * 0.5) * epsilon
    ///
    /// The gradients are:
    /// - dL/d_mean = dL/dz (gradient passes through unchanged)
    /// - dL/d_logvar = dL/dz * epsilon * exp(logvar * 0.5) * 0.5
    /// </para>
    /// <para>
    /// <b>Production-Ready Features:</b>
    /// <list type="bullet">
    /// <item>Builds proper computation graph using TensorOperations</item>
    /// <item>Uses inline topological sort for backward pass</item>
    /// <item>Fully vectorized - no nested loops</item>
    /// <item>GPU-accelerated via IEngine</item>
    /// </list>
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastMean == null || _lastLogVar == null || _lastEpsilon == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = outputGradient.Shape[0];
        int latentSize = outputGradient.Shape[1];

        // Build computation graph for the reparameterization trick
        // Create variable nodes for mean and logvar (these require gradients)
        var meanNode = TensorOperations<T>.Variable(_lastMean, "mean", requiresGradient: true);
        var logvarNode = TensorOperations<T>.Variable(_lastLogVar, "logvar", requiresGradient: true);

        // Create constant node for epsilon (random noise - no gradients)
        var epsilonNode = TensorOperations<T>.Constant(_lastEpsilon, "epsilon");

        // Create constant for 0.5 scalar multiplication
        var halfTensor = new Tensor<T>([batchSize, latentSize]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var halfNode = TensorOperations<T>.Constant(halfTensor, "half");

        // Build forward graph: z = mean + exp(logvar * 0.5) * epsilon
        // Step 1: scaledLogVar = logvar * 0.5
        var scaledLogVarNode = TensorOperations<T>.ElementwiseMultiply(logvarNode, halfNode);

        // Step 2: stdDev = exp(scaledLogVar)
        var stdDevNode = TensorOperations<T>.Exp(scaledLogVarNode);

        // Step 3: scaledEpsilon = stdDev * epsilon
        var scaledEpsilonNode = TensorOperations<T>.ElementwiseMultiply(stdDevNode, epsilonNode);

        // Step 4: z = mean + scaledEpsilon
        var zNode = TensorOperations<T>.Add(meanNode, scaledEpsilonNode);

        // Set the output gradient as the seed for backpropagation
        zNode.Gradient = outputGradient;

        // Inline topological sort (matching FullyConnectedLayer pattern)
        var visited = new HashSet<ComputationNode<T>>();
        var topoOrder = new List<ComputationNode<T>>();
        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
        stack.Push((zNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients for mean and logvar
        if (meanNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed for mean.");
        if (logvarNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed for logvar.");

        var gradMean = meanNode.Gradient;
        var gradLogVar = logvarNode.Gradient;

        // Concatenate gradients: [gradMean, gradLogVar] using Engine operations
        var inputGradient = new Tensor<T>([batchSize, latentSize * 2]);
        inputGradient = Engine.TensorSetSlice(inputGradient, gradMean, [0, 0]);
        inputGradient = Engine.TensorSetSlice(inputGradient, gradLogVar, [0, latentSize]);

        return inputGradient;
    }


    /// <summary>
    /// Updates the parameters of the reparameterization layer.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is required by the LayerBase class but does nothing in the RepParameterizationLayer
    /// because this layer has no trainable parameters to update. The learning happens in the encoder
    /// network that produces the means and log variances.
    /// </para>
    /// <para><b>For Beginners:</b> This method is empty because the layer has no internal values to update.
    /// 
    /// Unlike most layers in a neural network, the reparameterization layer doesn't have any
    /// weights or biases that need to be adjusted during training. It's more like a mathematical
    /// operation that passes gradients through.
    /// 
    /// The actual learning happens in:
    /// - The encoder network that produces the means and log variances
    /// - The decoder network that processes the samples this layer produces
    /// 
    /// This method exists only because all layers in the network must implement it.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all trainable parameters of the reparameterization layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since this layer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector because the RepParameterizationLayer has no trainable parameters.
    /// The method is required by the LayerBase class but is essentially a no-op for this layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because the layer has no learnable values.
    /// 
    /// As mentioned earlier, the reparameterization layer doesn't have any weights or biases
    /// that it learns during training. It just performs the sampling operation and passes
    /// gradients through.
    /// 
    /// This method returns an empty vector to indicate that there are no parameters to retrieve.
    /// It exists only because all layers in the network must implement it.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // This layer has no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the reparameterization layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the reparameterization layer, including the cached means,
    /// log variances, and epsilon values from the forward pass. This is useful when starting to process
    /// a new batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored means, log variances, and random noise values are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Since this layer has no learned parameters, resetting just clears the temporary
    /// values used during the forward and backward passes.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastMean = null;
        _lastLogVar = null;
        _lastEpsilon = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Input contains [batch, latentSize * 2] where first half is mean, second half is logvar
        int latentSize = InputShape[0] / 2;
        var symbolicInput = new Tensor<T>(new int[] { 1, InputShape[0] });
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Split input into mean and logvar along axis 1
        var splitOutputs = TensorOperations<T>.Split(inputNode, numSplits: 2, axis: 1);

        // splitOutputs will contain [meanNode, logvarNode]
        // For deterministic VAE inference (standard practice), return only the mean
        // This avoids randomness and gives the expected value of the latent distribution
        var meanNode = splitOutputs[0];  // Get the first split (mean)

        return meanNode;
    }

    public override bool SupportsJitCompilation => true;

}
