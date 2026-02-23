using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Hyperbolic Neural Network for learning hierarchical representations in Poincare ball space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// A Hyperbolic Neural Network operates in hyperbolic space (specifically the Poincare ball model)
/// rather than Euclidean space. This allows it to naturally capture hierarchical and tree-like
/// structures in data with lower distortion than traditional networks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Hyperbolic neural networks are designed for data that has a natural
/// hierarchy or tree-like structure. Examples include:
/// - Taxonomies (e.g., animal kingdom classification)
/// - Organizational hierarchies
/// - Social networks with community structures
/// - Knowledge graphs
///
/// In hyperbolic space, the "distance" near the center is smaller than near the edges,
/// allowing hierarchies to be represented more efficiently than in regular flat space.
/// Points near the center represent "root" concepts, while points near the edge represent
/// more specific "leaf" concepts.
/// </para>
/// </remarks>
public class HyperbolicNeuralNetwork<T> : NeuralNetworkBase<T>
{
    private readonly HyperbolicNeuralNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The optimization algorithm used to update the network's parameters during training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The curvature of the hyperbolic space (must be negative).
    /// </summary>
    private double _curvature;

    /// <summary>
    /// Initializes a new instance of the HyperbolicNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="curvature">The curvature of hyperbolic space (default -1.0, must be negative).</param>
    /// <param name="optimizer">The optimization algorithm to use for training. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function to use for training. If null, MSE is used.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping during training.</param>
    /// <remarks>
    /// <para>
    /// The curvature parameter controls how "curved" the hyperbolic space is. More negative values
    /// mean stronger curvature, which can better capture deep hierarchies but may be harder to optimize.
    /// A curvature of -1.0 is a good default for most applications.
    /// </para>
    /// </remarks>
    public HyperbolicNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        double curvature = -1.0,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        HyperbolicNeuralNetworkOptions? options = null) : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
        _options = options ?? new HyperbolicNeuralNetworkOptions();
        Options = _options;
        if (curvature >= 0)
        {
            throw new ArgumentException("Curvature must be negative for hyperbolic space.", nameof(curvature));
        }

        _curvature = curvature;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Note: LossFunction is inherited from NeuralNetworkBase and set in base constructor call
        // No need to duplicate storage here

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the hyperbolic neural network based on the provided architecture.
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
                Layers.Add(new HyperbolicLinearLayer<T>(inputFeatures, outputFeatures, _curvature));
            }
            else
            {
                Layers.Add(new HyperbolicLinearLayer<T>(inputFeatures, hiddenSizes[0], _curvature));

                for (int i = 0; i < hiddenSizes.Length - 1; i++)
                {
                    Layers.Add(new HyperbolicLinearLayer<T>(hiddenSizes[i], hiddenSizes[i + 1], _curvature));
                }

                Layers.Add(new HyperbolicLinearLayer<T>(hiddenSizes[^1], outputFeatures, _curvature));
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the hyperbolic neural network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// Input points should be inside the Poincare ball (norm less than 1/sqrt(-curvature)).
    /// Points near the origin represent high-level concepts; points near the boundary
    /// represent more specific concepts.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var previousTrainingMode = IsTrainingMode;
        IsTrainingMode = false;

        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(HyperbolicNeuralNetwork<T>), "prediction");

        var predictions = Forward(input);

        IsTrainingMode = previousTrainingMode;

        return predictions;
    }

    /// <summary>
    /// Performs a forward pass through the network with the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;


        TensorValidator.ValidateShape(input, Architecture.GetInputShape(),
            nameof(HyperbolicNeuralNetwork<T>), "forward pass");

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
    /// The backward pass in hyperbolic space uses Riemannian gradients, which are
    /// automatically handled by the HyperbolicLinearLayer. This ensures that
    /// gradient updates respect the geometry of hyperbolic space.
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
    /// Trains the hyperbolic neural network using the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor for the given input.</param>
    /// <remarks>
    /// <para>
    /// Training uses Riemannian gradient descent, where parameter updates follow
    /// geodesics (shortest paths) in hyperbolic space rather than straight lines.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        IsTrainingMode = true;

        try
        {
            var prediction = Forward(input);

            var primaryLoss = LossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            T auxiliaryLoss = NumOps.Zero;
            foreach (var auxLayer in Layers.OfType<IAuxiliaryLossLayer<T>>().Where(l => l.UseAuxiliaryLoss))
            {
                var layerAuxLoss = auxLayer.ComputeAuxiliaryLoss();
                var weightedAuxLoss = NumOps.Multiply(layerAuxLoss, auxLayer.AuxiliaryLossWeight);
                auxiliaryLoss = NumOps.Add(auxiliaryLoss, weightedAuxLoss);
            }

            LastLoss = NumOps.Add(primaryLoss, auxiliaryLoss);

            var outputGradient = LossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = Tensor<T>.FromVector(outputGradient);

            Backward(outputGradientTensor);

            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            IsTrainingMode = false;
        }
    }

    /// <summary>
    /// Retrieves metadata about the hyperbolic neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.FeedForwardNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "HyperbolicNeuralNetwork" },
                { "Curvature", _curvature },
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
    /// Serializes hyperbolic neural network-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_curvature);
        writer.Write(_optimizer.GetType().FullName ?? "AdamOptimizer");
        writer.Write(LossFunction.GetType().FullName ?? "MeanSquaredErrorLoss");
    }

    /// <summary>
    /// Deserializes hyperbolic neural network-specific data from a binary reader.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when deserialized curvature is not negative.</exception>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        double deserializedCurvature = reader.ReadDouble();

        // Validate the deserialized curvature - must be negative for hyperbolic space
        if (deserializedCurvature >= 0)
        {
            throw new InvalidOperationException(
                $"Invalid curvature value {deserializedCurvature} in serialized data. " +
                "Curvature must be negative for hyperbolic space. The serialized data may be corrupted.");
        }

        _curvature = deserializedCurvature;

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
    /// Creates a new instance of the HyperbolicNeuralNetwork with the same configuration.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new HyperbolicNeuralNetwork<T>(
            Architecture,
            _curvature,
            _optimizer,
            LossFunction,
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
        // Hyperbolic layers are valid input layers for this network
        if (layer is HyperbolicLinearLayer<T>)
            return true;

        return base.IsValidInputLayer(layer);
    }

    /// <summary>
    /// Determines if a layer can serve as a valid output layer for this network.
    /// </summary>
    protected override bool IsValidOutputLayer(ILayer<T> layer)
    {
        // Hyperbolic layers are valid output layers for this network
        if (layer is HyperbolicLinearLayer<T>)
            return true;

        return base.IsValidOutputLayer(layer);
    }
}
