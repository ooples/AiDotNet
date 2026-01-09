namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Capsule Network, a type of neural network that preserves spatial relationships between features.
/// </summary>
/// <remarks>
/// <para>
/// A Capsule Network is a neural network architecture designed to address limitations of traditional convolutional
/// neural networks. Instead of using scalar-output feature detectors (neurons), Capsule Networks use vector-output
/// capsules. Each capsule's output vector represents the presence of an entity and its instantiation parameters
/// (like position, orientation, and scale). This architecture helps to preserve hierarchical relationships
/// between features, making it particularly effective for tasks requiring understanding of spatial relationships.
/// </para>
/// <para><b>For Beginners:</b> A Capsule Network is like a more advanced version of traditional neural networks.
/// 
/// Think of it this way:
/// - Traditional networks detect features like edges or textures, but lose information about how these features relate to each other
/// - Capsule Networks not only detect features, but also understand their relationships, orientations, and positions
/// - This is like the difference between recognizing individual puzzle pieces versus understanding how they fit together
/// 
/// For example, a traditional network might recognize an eye, a nose, and a mouth separately, but a Capsule Network
/// can better understand that these features need to be in a specific arrangement to make a face. This makes
/// Capsule Networks particularly good at recognizing objects from different angles or when parts are arranged differently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class CapsuleNetwork<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    private ILossFunction<T> _lossFunction { get; set; }

    /// <summary>
    /// Stores the last capsule outputs for reconstruction loss computation.
    /// </summary>
    private Tensor<T>? _lastCapsuleOutputs;

    /// <summary>
    /// Stores the last input for reconstruction loss computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the last computed reconstruction loss for diagnostics.
    /// </summary>
    private T _lastReconstructionLoss;

    /// <summary>
    /// Stores the last computed margin loss for diagnostics.
    /// </summary>
    private T _lastMarginLoss;

    /// <summary>
    /// Gets or sets whether to use auxiliary loss (reconstruction regularization) during training.
    /// Default is true as per Sabour et al. (2017) - required for proper CapsNet functionality.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets the weight for reconstruction loss.
    /// Default is 0.0005 (standard value from original CapsNet paper).
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CapsuleNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Capsule Network with the specified architecture. The architecture
    /// defines the structure of the network, including the input dimensions, number and types of layers,
    /// and output dimensions. The initialization process sets up the layers based on the provided architecture
    /// or creates default capsule network layers if none are specified.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Capsule Network with your chosen settings.
    /// 
    /// When you create a Capsule Network:
    /// - You provide an "architecture" that defines how the network is structured
    /// - This includes information like how large the input is and what kinds of layers to use
    /// - The constructor sets up the basic structure, but doesn't actually train the network yet
    /// 
    /// Think of it like setting up a blank canvas and easel before you start painting -
    /// you're just getting everything ready to use.
    /// </para>
    /// </remarks>
    public CapsuleNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? new MarginLoss<T>())
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.0005);
        _lastReconstructionLoss = NumOps.Zero;
        _lastMarginLoss = NumOps.Zero;

        _lossFunction = lossFunction ?? new MarginLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Capsule Network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Capsule Network. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default capsule network layers are created based on the architecture's
    /// specifications. After adding the layers, the method validates that the custom layers are properly configured.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the actual structure of the network.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers that work well for most cases
    /// - The method also checks that all layers are compatible with each other
    /// 
    /// This is like assembling the different sections of a factory production line -
    /// each layer processes the data and passes it to the next layer.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultCapsuleNetworkLayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the Capsule Network.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter vector among all the layers in the network.
    /// Each layer receives a portion of the parameter vector corresponding to its number of parameters.
    /// The method keeps track of the starting index for each layer's parameters in the input vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the network's internal values during training.
    /// 
    /// When updating parameters:
    /// - The input is a long list of numbers representing all values in the entire network
    /// - The method divides this list into smaller chunks
    /// - Each layer gets its own chunk of values
    /// - The layers use these values to adjust their internal settings
    /// 
    /// Think of it like giving each department in a company their specific budget allocations
    /// from the overall company budget.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Performs a forward pass through the Capsule Network to make a prediction.
    /// </summary>
    /// <param name="input">The input tensor to the network.</param>
    /// <returns>The output tensor (prediction) from the network.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input tensor through each layer of the network in sequence.
    /// Each layer processes the output from the previous layer (or the input for the first layer)
    /// and produces an output that becomes the input for the next layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like passing a piece of information through a series of processing stations.
    /// 
    /// Imagine an assembly line:
    /// - The input is the raw material
    /// - Each layer is a workstation that modifies or processes the material
    /// - The output is the final product after it has passed through all stations
    /// 
    /// In a Capsule Network, this process preserves and processes spatial relationships,
    /// allowing the network to understand complex structures in the input data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Computes the auxiliary loss for the CapsuleNetwork, which is the reconstruction regularization.
    /// </summary>
    /// <returns>The reconstruction loss value.</returns>
    /// <remarks>
    /// <para>
    /// The reconstruction loss encourages the digit capsules to encode instantiation parameters of the input.
    /// A decoder network uses the activity vector of the correct DigitCaps to reconstruct the input image.
    /// The reconstruction loss is the sum of squared differences between the input and reconstruction.
    /// This is scaled down by a factor (typically 0.0005) so it doesn't dominate the margin loss during training.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how well the network can reconstruct the original input from its capsule representation.
    ///
    /// Reconstruction regularization:
    /// - Takes the capsule outputs (compressed representation)
    /// - Tries to recreate the original input from them
    /// - Measures how different the reconstruction is from the original
    /// - Encourages capsules to preserve important information about the input
    ///
    /// Why this is important:
    /// - Ensures capsules learn meaningful representations
    /// - Prevents the network from learning arbitrary encodings
    /// - Acts as a regularizer to improve generalization
    /// - Helps capsules encode pose/instantiation parameters
    ///
    /// This is similar to how an autoencoder works, but specifically designed for capsule networks.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastCapsuleOutputs == null || _lastInput == null)
        {
            return NumOps.Zero;
        }

        // Compute reconstruction loss
        // In a full implementation, this would use a decoder network to reconstruct the input
        // from the capsule outputs, then calculate the MSE between input and reconstruction
        _lastReconstructionLoss = ComputeReconstructionLoss(_lastCapsuleOutputs, _lastInput);

        return _lastReconstructionLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary losses.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about CapsuleNetwork training.</returns>
    /// <remarks>
    /// <para>
    /// This method provides insights into CapsuleNetwork training dynamics, including:
    /// - Margin loss (primary classification loss)
    /// - Reconstruction loss (auxiliary regularization)
    /// - Total loss
    /// - Reconstruction weight
    /// </para>
    /// <para><b>For Beginners:</b> This gives you information to monitor CapsuleNetwork training health.
    ///
    /// The diagnostics include:
    /// - Margin Loss: The main classification loss from the capsule network
    /// - Reconstruction Loss: How well the network can recreate inputs from capsules
    /// - Total Loss: Combined loss used for training
    /// - Reconstruction Weight: How much reconstruction influences training (usually small)
    ///
    /// These values help you:
    /// - Monitor training convergence
    /// - Balance classification and reconstruction objectives
    /// - Detect overfitting or underfitting
    /// - Tune the reconstruction weight hyperparameter
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "MarginLoss", _lastMarginLoss?.ToString() ?? "0" },
            { "ReconstructionLoss", _lastReconstructionLoss?.ToString() ?? "0" },
            { "TotalLoss", LastLoss?.ToString() ?? "0" },
            { "ReconstructionWeight", AuxiliaryLossWeight?.ToString() ?? "0.0005" },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
        };

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Computes the reconstruction loss by measuring how well the input can be reconstructed from capsule outputs.
    /// </summary>
    /// <param name="capsuleOutputs">The output activations from the capsule layers.</param>
    /// <param name="originalInput">The original input to the network.</param>
    /// <returns>The reconstruction loss value.</returns>
    /// <remarks>
    /// This is a simplified implementation. A full implementation would:
    /// 1. Use a decoder network (typically 3 fully connected layers: 512, 1024, input_size)
    /// 2. Mask the capsule outputs to use only the correct class capsule during training
    /// 3. Feed masked capsules through decoder to reconstruct input
    /// 4. Calculate MSE between original input and reconstruction
    ///
    /// For now, this returns a placeholder that can be enhanced when a full decoder is added.
    /// </remarks>
    private T ComputeReconstructionLoss(Tensor<T> capsuleOutputs, Tensor<T> originalInput)
    {
        // Simplified reconstruction loss computation
        // A full implementation would require a decoder network
        // For now, compute a simple L2 loss between capsule outputs and a projection of the input

        // Calculate mean squared error as a proxy for reconstruction quality
        T sumSquaredError = NumOps.Zero;
        int minLength = Math.Min(capsuleOutputs.Length, originalInput.Length);

        for (int i = 0; i < minLength; i++)
        {
            T diff = NumOps.Subtract(capsuleOutputs[i], originalInput[i]);
            sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(diff, diff));
        }

        // Average over all elements
        T reconstructionLoss = NumOps.Divide(sumSquaredError, NumOps.FromDouble(minLength));

        return reconstructionLoss;
    }

    /// <summary>
    /// Trains the Capsule Network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training iteration:
    /// 1. It makes a prediction using the current network parameters.
    /// 2. Calculates the loss between the prediction and the expected output.
    /// 3. Computes the gradient of the loss with respect to the network parameters.
    /// 4. Updates the network parameters based on the computed gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a practice session where the network learns from its mistakes.
    /// 
    /// The process is similar to learning a new skill:
    /// 1. You try to perform the task (make a prediction)
    /// 2. You see how far off you were (calculate the loss)
    /// 3. You figure out what you need to change to do better (compute the gradient)
    /// 4. You adjust your approach based on what you learned (update parameters)
    /// 
    /// This process is repeated many times with different inputs to improve the network's performance.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Store input for reconstruction loss
        _lastInput = input;

        // Forward pass
        var prediction = Predict(input);
        _lastCapsuleOutputs = prediction;

        // Calculate margin loss (primary loss)
        var marginLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        _lastMarginLoss = marginLoss;

        // Calculate auxiliary loss (reconstruction) if enabled
        T auxiliaryLoss = NumOps.Zero;
        if (UseAuxiliaryLoss)
        {
            var reconstructionLoss = ComputeAuxiliaryLoss();
            auxiliaryLoss = NumOps.Multiply(reconstructionLoss, AuxiliaryLossWeight);
        }

        // Total loss combines margin loss and reconstruction loss
        var totalLoss = NumOps.Add(marginLoss, auxiliaryLoss);
        LastLoss = totalLoss;

        // Backward pass
        var gradient = CalculateGradient(totalLoss);

        // Update parameters
        UpdateParameters(gradient);
    }

    /// <summary>
    /// Retrieves metadata about the Capsule Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the network's structure and configuration.
    /// It includes details such as the input and output dimensions, the number of layers, and the types of layers used.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a summary or overview of the network's structure.
    /// 
    /// Think of it as a quick reference guide that tells you:
    /// - What kind of network it is (a Capsule Network)
    /// - How big the input and output are
    /// - How many layers the network has
    /// - What types of layers are used
    /// 
    /// This information is useful for understanding the network's capabilities and for saving/loading the network.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.CapsuleNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", Layers[0].GetInputShape()[0] },
                { "OutputDimension", Layers[Layers.Count - 1].GetOutputShape()[0] },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Capsule Network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// This method saves the loss function used by the network, allowing it to be reconstructed when the network is deserialized.
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
    }

    /// <summary>
    /// Deserializes Capsule Network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// This method loads the loss function used by the network. If deserialization fails, it defaults to using a MarginLoss.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _lossFunction = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader) ?? new MarginLoss<T>();
    }

    /// <summary>
    /// Calculates the gradient of the loss with respect to the network parameters.
    /// </summary>
    /// <param name="loss">The scalar loss value.</param>
    /// <returns>A vector containing the gradients for all network parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a backward pass through the network, computing gradients for each layer.
    /// It starts from the output layer and moves backwards, accumulating gradients along the way.
    /// </para>
    /// <para><b>For Beginners:</b> This is like tracing back through the network to see how each part contributed to the final result.
    /// 
    /// Imagine you're trying to improve a recipe:
    /// - You start with how the final dish turned out (the loss)
    /// - You work backwards through each step of the recipe
    /// - At each step, you figure out how changing that step would affect the final result
    /// - You collect all these potential changes (gradients) to know how to improve the recipe
    /// 
    /// In a neural network, this process helps determine how to adjust each parameter to reduce the loss.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateGradient(T loss)
    {
        List<Tensor<T>> gradients = new List<Tensor<T>>();

        // Backward pass through all layers
        Tensor<T> currentGradient = new Tensor<T>([1], new Vector<T>(Enumerable.Repeat(loss, 1)));
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
            gradients.Insert(0, currentGradient);
        }

        // Flatten all gradients into a single vector
        return new Vector<T>([.. gradients.SelectMany(g => g.ToVector())]);
    }

    /// <summary>
    /// Computes the reconstruction loss for capsule network regularization.
    /// </summary>
    /// <param name="input">The original input tensor.</param>
    /// <param name="trueLabel">The true class label for masking. If null, uses argmax of capsule outputs.</param>
    /// <returns>The reconstruction loss (MSE between reconstruction and input).</returns>
    /// <remarks>
    /// <para>
    /// This method implements the reconstruction loss from the original Capsule Networks paper
    /// (Sabour et al., 2017). During training, only the capsule corresponding to the true class
    /// is used for reconstruction (others are masked to zero). During inference, the capsule with
    /// the highest activation is used. The reconstruction helps regularize the network and ensures
    /// that capsule vectors encode meaningful instantiation parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps the network learn better by making it reconstruct the input.
    ///
    /// How it works:
    /// 1. Forward pass through the network to get capsule outputs
    /// 2. Mask: Zero out all capsules except the correct one (or most active one)
    /// 3. Reconstruction: Pass masked capsules through decoder layers
    /// 4. Loss: Measure how different the reconstruction is from the original input
    ///
    /// Why this helps:
    /// - Forces capsules to encode useful information (position, rotation, etc.)
    /// - Acts as regularization to prevent overfitting
    /// - Improves interpretability of what capsules represent
    ///
    /// The reconstruction loss is typically weighted much lower than the main classification loss
    /// (e.g., 0.0005 * reconstruction_loss) to avoid overwhelming the primary objective.
    /// </para>
    /// </remarks>
    public T ComputeReconstructionLoss(Tensor<T> input, int? trueLabel = null)
    {
        // Find the reconstruction layer (should be last layer in default architecture)
        var reconstructionLayer = Layers.OfType<ReconstructionLayer<T>>().FirstOrDefault();
        if (reconstructionLayer == null)
        {
            // No reconstruction layer in architecture - return zero loss
            return NumOps.Zero;
        }

        // Perform forward pass with memory to capture intermediate outputs
        if (SupportsTraining)
        {
            ForwardWithMemory(input);
        }
        else
        {
            // If not in training mode, do regular forward pass
            Predict(input);
        }

        // Find the digit capsule layer (second-to-last layer, before reconstruction)
        // Use type-safe checking instead of string matching to avoid fragility
        int digitCapsLayerIndex = -1;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            if (Layers[i] is DigitCapsuleLayer<T> ||
                Layers[i] is CapsuleLayer<T> ||
                Layers[i] is PrimaryCapsuleLayer<T>)
            {
                digitCapsLayerIndex = i;
                break;
            }
        }

        if (digitCapsLayerIndex == -1 || !_layerOutputs.TryGetValue(digitCapsLayerIndex, out var capsuleOutputs))
        {
            // Could not find capsule layer output
            return NumOps.Zero;
        }

        // Apply masking: zero out all capsules except the target capsule
        var maskedCapsules = ApplyCapsuleMask(capsuleOutputs, trueLabel);

        // Pass masked capsules through reconstruction layer
        var reconstruction = reconstructionLayer.Forward(maskedCapsules);

        // Flatten original input for comparison
        // Validate the input can be flattened before attempting reshape
        int expectedLength = 1;
        foreach (int dim in input.Shape)
        {
            expectedLength *= dim;
        }

        if (expectedLength != input.Length)
        {
            throw new InvalidOperationException(
                $"Input tensor product of dimensions ({expectedLength}) " +
                $"does not match actual length ({input.Length}).");
        }

        var flattenedInput = input.Reshape([input.Length]);

        // Validate shapes match before computing MSE
        if (flattenedInput.Length != reconstruction.Length)
        {
            throw new InvalidOperationException(
                $"Shape mismatch: input length ({flattenedInput.Length}) " +
                $"does not match reconstruction length ({reconstruction.Length}). " +
                $"This indicates a misconfiguration in the reconstruction layer architecture.");
        }

        // Compute MSE loss manually since Tensor doesn't have ToEnumerable
        T sumSquaredError = NumOps.Zero;

        for (int i = 0; i < flattenedInput.Length; i++)
        {
            T diff = NumOps.Subtract(flattenedInput[i], reconstruction[i]);
            sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(sumSquaredError, NumOps.FromDouble(flattenedInput.Length));
    }

    /// <summary>
    /// Applies masking to capsule outputs, zeroing out all capsules except the target.
    /// </summary>
    /// <param name="capsuleOutputs">The capsule output tensor.</param>
    /// <param name="targetClass">The target class index. If null, uses argmax.</param>
    /// <returns>Masked capsule tensor.</returns>
    /// <remarks>
    /// <para>
    /// During training, we mask out the activity vectors of all but the target capsule.
    /// During inference (targetClass = null), we use the capsule with the highest norm.
    /// </para>
    /// <para>
    /// Note: Current implementation applies the same target capsule mask to all batch elements.
    /// This is appropriate for reconstruction loss with batch size 1 or when all samples
    /// in the batch have the same target class. For per-sample masking with different targets,
    /// this method would need to accept an array of target indices.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyCapsuleMask(Tensor<T> capsuleOutputs, int? targetClass)
    {
        // Use provided label (training mode) or argmax of capsule norms (inference mode)
        int targetCapsuleIndex = targetClass.HasValue
            ? targetClass.Value
            : GetPredictedClass(capsuleOutputs);

        // Create masked copy
        var masked = capsuleOutputs.Clone();

        // Zero out all capsules except target
        // Shape is expected to be [batch, numCapsules, capsuleDim]
        // Note: Same target capsule is masked for all batch elements
        if (capsuleOutputs.Shape.Length >= 2)
        {
            int batchSize = capsuleOutputs.Shape[0];
            int numCapsules = capsuleOutputs.Shape[1];
            int capsuleDim = capsuleOutputs.Shape.Length > 2 ? capsuleOutputs.Shape[2] : 1;

            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < numCapsules; c++)
                {
                    if (c != targetCapsuleIndex)
                    {
                        // Zero out this capsule
                        for (int d = 0; d < capsuleDim; d++)
                        {
                            int index = b * numCapsules * capsuleDim + c * capsuleDim + d;
                            if (index < masked.Length)
                            {
                                masked[index] = NumOps.Zero;
                            }
                        }
                    }
                }
            }
        }

        return masked;
    }

    /// <summary>
    /// Gets the predicted class from capsule outputs based on capsule norms.
    /// </summary>
    /// <param name="capsuleOutputs">The capsule output tensor.</param>
    /// <returns>The index of the capsule with the highest norm.</returns>
    /// <remarks>
    /// Note: For batched inputs, this currently only examines the first batch element (batch index 0).
    /// For proper batched inference, this would need to return predictions for all batch elements.
    /// Current usage is primarily for reconstruction masking during training with batch size 1.
    /// </remarks>
    private int GetPredictedClass(Tensor<T> capsuleOutputs)
    {
        // Compute norm of each capsule and return argmax
        // Assuming shape is [batch, numCapsules, capsuleDim]
        // Currently only processes batch index 0 for simplicity
        int batchOffset = 0; // Could be parameterized for specific batch element
        int numCapsules = capsuleOutputs.Shape.Length > 1 ? capsuleOutputs.Shape[1] : 1;
        int capsuleDim = capsuleOutputs.Shape.Length > 2 ? capsuleOutputs.Shape[2] : 1;

        T maxNorm = NumOps.Zero;
        int maxIndex = 0;

        for (int c = 0; c < numCapsules; c++)
        {
            T normSquared = NumOps.Zero;
            for (int d = 0; d < capsuleDim; d++)
            {
                int index = batchOffset + c * capsuleDim + d;
                if (index < capsuleOutputs.Length)
                {
                    T val = capsuleOutputs[index];
                    normSquared = NumOps.Add(normSquared, NumOps.Multiply(val, val));
                }
            }

            if (NumOps.GreaterThan(normSquared, maxNorm))
            {
                maxNorm = normSquared;
                maxIndex = c;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Creates a new instance of the capsule network model.
    /// </summary>
    /// <returns>A new instance of the capsule network model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the capsule network model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization processes to create a fresh instance that can be populated
    /// with the serialized data. The new instance will have the same architecture and loss function as the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the network structure without copying the learned data.
    ///
    /// Think of it like creating a blueprint of the capsule network:
    /// - It copies the same overall design (architecture)
    /// - It uses the same loss function to measure performance
    /// - But it doesn't copy any of the learned values or weights
    ///
    /// This is primarily used when saving or loading models, creating a framework that the saved parameters
    /// can be loaded into later. It's like creating an empty duplicate of the network's structure
    /// that can later be filled with the knowledge from the original network.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CapsuleNetwork<T>(
            Architecture,
            _lossFunction
        );
    }
}
