using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Extreme Learning Machine (ELM), a type of feedforward neural network with a unique training approach.
/// </summary>
/// <remarks>
/// <para>
/// An Extreme Learning Machine is a special type of single-hidden-layer feedforward neural network that uses a
/// non-iterative training approach. Unlike traditional neural networks that use backpropagation to adjust all weights,
/// ELMs randomly assign the weights between the input and hidden layer and only train the weights between the hidden
/// and output layer. This is done analytically using a pseudo-inverse operation rather than through iterative
/// optimization, resulting in extremely fast training times while maintaining good generalization performance.
/// </para>
/// <para><b>For Beginners:</b> An Extreme Learning Machine is like a neural network on fast-forward.
/// 
/// Think of it like building a bridge:
/// - Traditional neural networks carefully adjust every piece of the bridge (slow but thorough)
/// - ELMs randomly set up most of the bridge, then only carefully adjust the final section
/// - This approach is much faster but can still create a surprisingly strong bridge
/// 
/// The "extreme" part refers to its extremely fast training time. While traditional networks
/// might take hours or days to train, ELMs can often be trained in seconds or minutes, making
/// them useful for applications where training speed is critical.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ExtremeLearningMachine<T> : NeuralNetworkBase<T>
{
    private readonly ExtremeLearningMachineOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the size of the hidden layer (number of neurons).
    /// </summary>
    /// <value>An integer representing the number of neurons in the hidden layer.</value>
    /// <remarks>
    /// <para>
    /// The hidden layer size determines the dimensionality of the feature space that the ELM projects the input data into.
    /// A larger hidden layer can capture more complex patterns but may lead to overfitting with small datasets.
    /// This is a key hyperparameter that significantly affects the ELM's performance and capacity.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many "pattern detectors" the network has.
    /// 
    /// Think of HiddenLayerSize as:
    /// - The number of different patterns the network can recognize
    /// - Like having a team of people each looking for specific features
    /// - More neurons (larger size) means more patterns can be detected
    /// - But too many neurons might make the network "memorize" rather than "learn"
    /// 
    /// For example, a hidden layer size of 100 means the network has 100 different
    /// pattern detectors that work together to analyze the input data.
    /// </para>
    /// </remarks>
    private readonly int _hiddenLayerSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExtremeLearningMachine{T}"/> class with the specified architecture and hidden layer size.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerSize">The number of neurons in the hidden layer.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Extreme Learning Machine with the specified architecture and hidden layer size.
    /// The hidden layer size is a key parameter that determines the capacity and learning ability of the ELM.
    /// After setting this parameter, the constructor initializes the network layers.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new ELM with a specific number of pattern detectors.
    /// 
    /// When creating a new ELM:
    /// - The architecture defines the overall structure (input and output sizes)
    /// - The hiddenLayerSize determines how many pattern detectors the network will have
    /// - The constructor sets up the initial structure, but doesn't train the network yet
    /// 
    /// Think of it like assembling a team of a specific size to look for patterns,
    /// where each team member will be randomly assigned what to look for.
    /// </para>
    /// </remarks>
    public ExtremeLearningMachine(NeuralNetworkArchitecture<T> architecture, int hiddenLayerSize, ILossFunction<T>? lossFunction = null, ExtremeLearningMachineOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new ExtremeLearningMachineOptions();
        Options = _options;
        _hiddenLayerSize = hiddenLayerSize;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Extreme Learning Machine based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Extreme Learning Machine. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default layers are created based on the architecture's specifications and
    /// the specified hidden layer size. A typical ELM consists of an input layer, a hidden layer with random weights,
    /// and an output layer.
    /// </para>
    /// <para><b>For Beginners:</b> This builds the structure of the neural network.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers suitable for an ELM:
    ///   1. A random input-to-hidden layer with fixed weights
    ///   2. A non-linear activation function (typically sigmoid or tanh)
    ///   3. A trainable hidden-to-output layer
    /// 
    /// The most important part is that only the final layer (hidden-to-output)
    /// will be trained - the other layers will keep their random weights.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultELMLayers(Architecture, _hiddenLayerSize));
        }
    }

    /// <summary>
    /// Updates the parameters of all layers in the Extreme Learning Machine.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <exception cref="NotImplementedException">
    /// Always thrown because ELM does not support traditional parameter updates.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method is not implemented for Extreme Learning Machines because they do not use traditional parameter updates.
    /// In an ELM, the input-to-hidden weights are randomly generated and remain fixed, while the hidden-to-output weights
    /// are calculated analytically in a single step rather than through iterative updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method always throws an error because ELMs don't train like regular neural networks.
    /// 
    /// Extreme Learning Machines are different from standard neural networks:
    /// - They don't use backpropagation or gradient descent
    /// - Most of their weights stay fixed (unchangeable) after random initialization
    /// - The output weights are calculated in one step, not iteratively updated
    /// 
    /// If you try to update parameters like in a regular neural network,
    /// you'll get an error because this isn't how ELMs work.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        throw new InvalidOperationException("Extreme Learning Machines do not support direct parameter updates via this method. Input-to-hidden weights are randomly initialized and remain fixed. Only output layer weights are computed analytically via the Train method.");
    }

    /// <summary>
    /// Makes a prediction using the Extreme Learning Machine.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input data through the Extreme Learning Machine network to make a prediction.
    /// It sequentially processes the input through all layers of the network, which typically includes a fixed
    /// random projection from the input layer to the hidden layer, followed by a nonlinear activation function,
    /// and finally the analytically trained weights from the hidden layer to the output layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the ELM processes new data to make predictions.
    /// 
    /// The prediction process works in three stages:
    /// 1. Input data is projected through random weights to the hidden layer
    /// 2. A nonlinear activation function (like sigmoid) is applied to these projections
    /// 3. The resulting hidden layer activations are multiplied by the trained output weights
    /// 
    /// Unlike traditional neural networks, the first step uses fixed random weights that were never trained,
    /// which is part of what makes ELMs so unique and fast.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Process the input through each layer sequentially
        Tensor<T> current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Trains the Extreme Learning Machine on a single batch of data.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method implements the ELM training algorithm, which is fundamentally different from traditional
    /// neural network training. Instead of using iterative optimization with backpropagation, ELM fixes the
    /// weights from input to hidden layer with random values and analytically calculates the optimal weights
    /// from hidden to output layer using a Moore-Penrose pseudo-inverse operation. This allows for extremely
    /// fast training compared to traditional neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This trains the ELM in one single step, which is much faster than traditional neural networks.
    /// 
    /// The ELM training process is unique:
    /// 1. The weights from input to hidden layer stay fixed at their random initial values
    /// 2. The input data is projected through these fixed random weights to get hidden layer activations
    /// 3. The optimal output weights are calculated using linear algebra (pseudo-inverse) in one step
    /// 
    /// This is like solving a system of equations directly rather than making many small adjustments,
    /// which is why ELMs can train thousands of times faster than traditional neural networks.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Check if the network structure matches the ELM requirements
        if (Layers.Count < 3)
        {
            throw new InvalidOperationException("ELM requires at least 3 layers: input projection, activation, and output.");
        }

        // STEP 1: Get the hidden layer activations by projecting the input through the fixed random weights
        Tensor<T> hiddenActivations = input;

        // Process through all layers except the last one (which is the output layer)
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            hiddenActivations = Layers[i].Forward(hiddenActivations);
        }

        // STEP 2: Calculate the optimal output weights using pseudo-inverse
        // We'll use the Moore-Penrose pseudoinverse: OutputWeights = (H+ × T)
        // where H+ is the pseudoinverse of H (hidden activations) and T is the target output

        // Convert hidden activations and expected output to matrices for the calculation
        Matrix<T> H = hiddenActivations.ConvertToMatrix();
        Matrix<T> T = expectedOutput.ConvertToMatrix();

        // Calculate pseudoinverse of H 
        Matrix<T> HPseudoInverse = CalculatePseudoInverse(H);

        // Calculate output weights
        Matrix<T> outputWeights = HPseudoInverse.Multiply(T);

        // STEP 3: Update only the last layer (output layer) with the calculated weights
        // In an ELM, only the output layer is trained
        UpdateOutputLayerWeights(outputWeights);

        // STEP 4: Calculate and store the loss after training
        Tensor<T> prediction = Predict(input);
        LastLoss = LossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
    }

    /// <summary>
    /// Calculates the Moore-Penrose pseudoinverse of a matrix.
    /// </summary>
    /// <param name="matrix">The matrix to calculate the pseudoinverse for.</param>
    /// <returns>The pseudoinverse of the input matrix.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the Moore-Penrose pseudoinverse of a matrix, which is a generalization of the matrix inverse
    /// for non-square matrices. The pseudoinverse is used in the ELM training algorithm to analytically solve
    /// for the optimal output layer weights. For computational efficiency, this implementation uses the formula:
    /// A+ = (A^T × A)^(-1) × A^T for full column rank matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates a special type of matrix inverse used in ELM training.
    /// 
    /// The pseudoinverse is a way to "divide" by a matrix even when traditional division isn't possible.
    /// It's a key part of what makes ELM training so fast, allowing us to directly solve for the optimal
    /// output weights in one step, rather than iteratively adjusting them as in traditional neural networks.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculatePseudoInverse(Matrix<T> matrix)
    {
        // Calculate the pseudoinverse using the formula: A+ = (A^T × A)^(-1) × A^T
        // This works well for matrices with full column rank, which is common in ELMs with
        // more data samples than hidden neurons

        // Step 1: Calculate A^T (transpose)
        Matrix<T> transposeA = matrix.Transpose();

        // Step 2: Calculate A^T × A
        Matrix<T> aTa = transposeA.Multiply(matrix);

        // Step 3: Calculate (A^T × A)^(-1)
        Matrix<T> aTaInverse = aTa.Inverse();

        // Step 4: Calculate (A^T × A)^(-1) × A^T
        Matrix<T> pseudoInverse = aTaInverse.Multiply(transposeA);

        return pseudoInverse;

        // Note: In a production implementation, you might want to use singular value decomposition (SVD)
        // for better numerical stability, or use a regularized version like:
        // A+ = (A^T × A + λI)^(-1) × A^T where λ is a small regularization parameter
    }

    /// <summary>
    /// Updates the weights of the output layer with the calculated weights.
    /// </summary>
    /// <param name="outputWeights">The matrix of new weights for the output layer.</param>
    /// <remarks>
    /// <para>
    /// This method updates only the output layer of the ELM with the weights calculated during training.
    /// In an ELM, this is the only layer that gets trained - the earlier layers maintain their random weights.
    /// </para>
    /// <para><b>For Beginners:</b> This applies the calculated weights to the output layer of the network.
    /// 
    /// In an ELM, only the final layer (connecting the hidden layer to the output) gets its weights updated.
    /// All other layers keep their randomly initialized weights. This method handles the process of
    /// taking the mathematically optimal weights we calculated and applying them to the output layer.
    /// </para>
    /// </remarks>
    private void UpdateOutputLayerWeights(Matrix<T> outputWeights)
    {
        // Get the last layer (output layer)
        var outputLayer = Layers[Layers.Count - 1];

        // Convert the output weights to the format expected by the layer
        Vector<T> flattenedWeights = new Vector<T>(outputWeights.Rows * outputWeights.Columns);
        int index = 0;
        for (int i = 0; i < outputWeights.Rows; i++)
        {
            for (int j = 0; j < outputWeights.Columns; j++)
            {
                flattenedWeights[index++] = outputWeights[i, j];
            }
        }

        // Update the output layer weights
        outputLayer.UpdateParameters(flattenedWeights);
    }

    /// <summary>
    /// Gets metadata about the Extreme Learning Machine model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the ELM, including its model type, hidden layer size,
    /// and additional configuration information. This metadata is useful for model management
    /// and for generating reports about the model's structure and configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your ELM's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (Extreme Learning Machine)
    /// - The size of the hidden layer
    /// - Information about the layers and their configurations
    /// - Serialized data that can be used to save and reload the model
    /// 
    /// Think of it like a label that describes the specific type and characteristics
    /// of your neural network. This is useful for organizing and managing multiple models.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ExtremeLearningMachine,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HiddenLayerSize", _hiddenLayerSize },
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Extreme Learning Machine-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes ELM-specific configuration data to a binary stream. It includes
    /// properties such as the hidden layer size and the weights of all layers. This data is needed
    /// to reconstruct the ELM when deserializing.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the special configuration of your ELM.
    /// 
    /// It's like writing down the recipe for how your specific ELM was built:
    /// - How many hidden neurons it has
    /// - The random weights used in the input-to-hidden connections
    /// - The trained weights used in the hidden-to-output connections
    /// 
    /// This allows you to save the model and reload it later, without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write hidden layer size
        writer.Write(_hiddenLayerSize);

        // Write whether we're in training mode
        writer.Write(IsTrainingMode);
    }

    /// <summary>
    /// Deserializes Extreme Learning Machine-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads ELM-specific configuration data from a binary stream. It retrieves
    /// properties such as the hidden layer size and the weights of all layers. After reading this data,
    /// the ELM's state is fully restored to what it was when saved.
    /// </para>
    /// <para><b>For Beginners:</b> This restores the special configuration of your ELM from saved data.
    /// 
    /// It's like following the recipe to rebuild your ELM exactly as it was:
    /// - Setting the hidden layer to the right size
    /// - Restoring the random weights for the input-to-hidden connections
    /// - Restoring the trained weights for the hidden-to-output connections
    /// 
    /// By reading these details, the ELM can be reconstructed exactly as it was
    /// when it was saved, preserving all its behavior and learned patterns.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read hidden layer size
        int hiddenLayerSize = reader.ReadInt32();

        // Check if the hiddenLayerSize matches the current instance
        if (hiddenLayerSize != _hiddenLayerSize)
        {
            Console.WriteLine($"Warning: Loaded ELM has hidden layer size {hiddenLayerSize}, " +
                             $"but current instance has size {_hiddenLayerSize}");
        }

        // Read training mode
        IsTrainingMode = reader.ReadBoolean();
    }

    /// <summary>
    /// Trains the ELM using regularized least squares for improved generalization.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <param name="regularizationFactor">The regularization factor (lambda) to use. Default is 0.01.</param>
    /// <remarks>
    /// <para>
    /// This method implements a regularized version of the ELM training algorithm. It adds a regularization term
    /// to the pseudoinverse calculation, which helps prevent overfitting. The formula becomes:
    /// OutputWeights = (H^T * H + λI)^(-1) * H^T * T, where λ is the regularization factor, I is the identity matrix,
    /// H is the hidden layer activations, and T is the target output.
    /// </para>
    /// <para><b>For Beginners:</b> This is a more robust training method that helps prevent overfitting.
    /// 
    /// Regularization is like adding a penalty for overly complex solutions:
    /// - It keeps the weights from becoming too large or specialized to the training data
    /// - It helps the model generalize better to new, unseen data
    /// - The regularization factor controls how strong this penalty is
    /// 
    /// This method still trains in one fast step (unlike traditional neural networks),
    /// but produces weights that typically work better on new data.
    /// </para>
    /// </remarks>
    public void TrainWithRegularization(Tensor<T> input, Tensor<T> expectedOutput, double regularizationFactor = 0.01)
    {
        // Check if the network structure matches the ELM requirements
        if (Layers.Count < 3)
        {
            throw new InvalidOperationException("ELM requires at least 3 layers: input projection, activation, and output.");
        }

        // STEP 1: Get the hidden layer activations by projecting the input through the fixed random weights
        Tensor<T> hiddenActivations = input;

        // Process through all layers except the last one (which is the output layer)
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            hiddenActivations = Layers[i].Forward(hiddenActivations);
        }

        // STEP 2: Calculate the optimal output weights using regularized pseudoinverse

        // Convert hidden activations and expected output to matrices for the calculation
        Matrix<T> H = hiddenActivations.ConvertToMatrix();
        Matrix<T> T = expectedOutput.ConvertToMatrix();

        // Calculate regularized pseudoinverse: (H^T * H + λI)^(-1) * H^T
        Matrix<T> transposeH = H.Transpose();
        Matrix<T> hTh = transposeH.Multiply(H);

        // Create identity matrix for regularization
        Matrix<T> identity = Matrix<T>.CreateIdentity(hTh.Rows);

        // Apply regularization: hTh + λI
        T regFactor = NumOps.FromDouble(regularizationFactor);
        for (int i = 0; i < identity.Rows; i++)
        {
            for (int j = 0; j < identity.Columns; j++)
            {
                if (i == j)
                {
                    hTh[i, j] = NumOps.Add(hTh[i, j], NumOps.Multiply(identity[i, j], regFactor));
                }
            }
        }

        // Calculate inverse of regularized matrix
        Matrix<T> regularizedInverse = hTh.Inverse();

        // Calculate final pseudoinverse
        Matrix<T> regularizedPseudoInverse = regularizedInverse.Multiply(transposeH);

        // Calculate output weights
        Matrix<T> outputWeights = regularizedPseudoInverse.Multiply(T);

        // STEP 3: Update only the last layer (output layer) with the calculated weights
        UpdateOutputLayerWeights(outputWeights);
    }

    /// <summary>
    /// Creates a new instance of the ExtremeLearningMachine with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new ExtremeLearningMachine instance with the same architecture and hidden layer size as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the ExtremeLearningMachine with the same architecture and hidden layer size
    /// as the current instance. This is useful for model cloning, ensemble methods, or cross-validation scenarios where
    /// multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the ELM's blueprint.
    /// 
    /// When you need multiple versions of the same type of ELM with identical settings:
    /// - This method creates a new, empty ELM with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new ELM has the same structure but no trained data
    /// - This is useful for techniques that need multiple models, like ensemble methods
    /// 
    /// For example, when training multiple ELMs on different subsets of data,
    /// you'd want each one to have the same architecture and hidden layer size.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ExtremeLearningMachine<T>(Architecture, _hiddenLayerSize);
    }
}
