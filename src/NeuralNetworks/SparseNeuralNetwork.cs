using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Sparse Neural Network with efficient sparse weight matrices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Sparse Neural Network uses sparse weight matrices where most values are zero.
/// This provides significant memory and computational savings for large networks,
/// especially when combined with network pruning techniques.
/// </para>
/// <para>
/// <b>For Beginners:</b> In a regular neural network, every neuron in one layer is connected
/// to every neuron in the next layer. In a sparse network, many of these connections are
/// removed (set to zero), keeping only the most important ones. This has several benefits:
/// - Uses less memory (only stores non-zero values)
/// - Runs faster (skips multiplications with zero)
/// - Can prevent overfitting (acts as regularization)
/// - Enables very large networks to fit in limited memory
///
/// Common use cases include:
/// - Network compression for mobile/edge deployment
/// - Recommender systems with sparse user-item matrices
/// - Graph neural networks with sparse adjacency matrices
/// - Pruned networks from neural architecture search
/// </para>
/// </remarks>
public class SparseNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The loss function used to calculate the error between predicted and expected outputs.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The sparsity level (fraction of weights that are zero).
    /// </summary>
    private double _sparsity;

    /// <summary>
    /// Initializes a new instance of the SparseNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="sparsity">The fraction of weights that should be zero (0.0 to 1.0). Default is 0.9 (90% sparse).</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, MSE is used.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping during training.</param>
    /// <remarks>
    /// <para>
    /// Higher sparsity values mean fewer connections and faster computation, but may reduce
    /// the network's capacity to learn complex patterns. A sparsity of 0.9 (90% zeros) is
    /// a good starting point for most applications.
    /// </para>
    /// </remarks>
    public SparseNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        double sparsity = 0.9,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0) : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
        if (sparsity < 0 || sparsity >= 1.0)
        {
            throw new ArgumentException("Sparsity must be in [0, 1).", nameof(sparsity));
        }

        _sparsity = sparsity;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the sparse neural network based on the provided architecture.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            var inputShape = Architecture.GetInputShape();
            var outputShape = Architecture.GetOutputShape();
            var hiddenSizes = Architecture.GetHiddenLayerSizes();

            int inputFeatures = inputShape[0];
            int outputFeatures = outputShape[0];

            if (hiddenSizes.Length == 0)
            {
                Layers.Add(new SparseLinearLayer<T>(inputFeatures, outputFeatures, _sparsity));
            }
            else
            {
                Layers.Add(new SparseLinearLayer<T>(inputFeatures, hiddenSizes[0], _sparsity));

                for (int i = 0; i < hiddenSizes.Length - 1; i++)
                {
                    Layers.Add(new SparseLinearLayer<T>(hiddenSizes[i], hiddenSizes[i + 1], _sparsity));
                }

                Layers.Add(new SparseLinearLayer<T>(hiddenSizes[^1], outputFeatures, _sparsity));
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the sparse neural network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        IsTrainingMode = false;

        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(SparseNeuralNetwork<T>), "prediction");

        var predictions = Forward(input);

        IsTrainingMode = true;

        return predictions;
    }

    /// <summary>
    /// Performs a forward pass through the network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass uses sparse matrix-vector multiplication (SpMV) for efficiency.
    /// Only non-zero weights are used in computation, significantly reducing the number
    /// of operations for highly sparse networks.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Validate input shape before any processing (including GPU path)
        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(SparseNeuralNetwork<T>), "forward pass");

        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Performs a backward pass through the network to calculate gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the network's output.</param>
    /// <returns>The gradient of the loss with respect to the network's input.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients are only computed for non-zero weights,
    /// maintaining the sparsity pattern throughout training.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        Tensor<T> gradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        return gradient;
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Trains the sparse neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// Training maintains the sparsity pattern - only non-zero weights are updated.
    /// This means the network structure is fixed after initialization; use dynamic
    /// sparsity techniques if you need the sparsity pattern to evolve during training.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        IsTrainingMode = true;

        var prediction = Forward(input);

        var primaryLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        T auxiliaryLoss = NumOps.Zero;
        foreach (var auxLayer in Layers.OfType<IAuxiliaryLossLayer<T>>().Where(l => l.UseAuxiliaryLoss))
        {
            var layerAuxLoss = auxLayer.ComputeAuxiliaryLoss();
            var weightedAuxLoss = NumOps.Multiply(layerAuxLoss, auxLayer.AuxiliaryLossWeight);
            auxiliaryLoss = NumOps.Add(auxiliaryLoss, weightedAuxLoss);
        }

        LastLoss = NumOps.Add(primaryLoss, auxiliaryLoss);

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = Tensor<T>.FromVector(outputGradient);

        Backward(outputGradientTensor);

        _optimizer.UpdateParameters(Layers);

        IsTrainingMode = false;
    }

    /// <summary>
    /// Retrieves metadata about the sparse neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.FeedForwardNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "SparseNeuralNetwork" },
                { "Sparsity", _sparsity },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
                { "HiddenLayerSizes", Architecture.GetHiddenLayerSizes() },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes sparse neural network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sparsity);
        writer.Write(_optimizer.GetType().FullName ?? "AdamOptimizer");
        writer.Write(_lossFunction.GetType().FullName ?? "MeanSquaredErrorLoss");
    }

    /// <summary>
    /// Deserializes sparse neural network-specific data from a binary reader.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sparsity = reader.ReadDouble();

        // Read type names for forward compatibility and validation
        string optimizerType = reader.ReadString();
        string lossFunctionType = reader.ReadString();

        // Note: Optimizer and loss function instances should be provided during construction.
        // The type names are read for data integrity verification but new instances
        // need to be created via the constructor or a dedicated factory method.
        _ = optimizerType;
        _ = lossFunctionType;
    }

    /// <summary>
    /// Creates a new instance of the SparseNeuralNetwork with the same configuration.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SparseNeuralNetwork<T>(
            Architecture,
            _sparsity,
            _optimizer,
            _lossFunction,
            Convert.ToDouble(MaxGradNorm));
    }

    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Determines if a layer can serve as a valid input layer for this network.
    /// </summary>
    protected override bool IsValidInputLayer(ILayer<T> layer)
    {
        // Sparse layers are valid input layers for this network
        if (layer is SparseLinearLayer<T>)
            return true;

        return base.IsValidInputLayer(layer);
    }

    /// <summary>
    /// Determines if a layer can serve as a valid output layer for this network.
    /// </summary>
    protected override bool IsValidOutputLayer(ILayer<T> layer)
    {
        // Sparse layers are valid output layers for this network
        if (layer is SparseLinearLayer<T>)
            return true;

        return base.IsValidOutputLayer(layer);
    }
}
