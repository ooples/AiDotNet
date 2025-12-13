using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Builds a computation graph from a neural network or sequence of layers.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class GraphBuilder<T> where T : struct
{
    private readonly ComputationGraph<T> _graph;
    private readonly Dictionary<object, ComputationNode<T>> _layerToNode;

    public GraphBuilder()
    {
        _graph = new ComputationGraph<T>();
        _layerToNode = new Dictionary<object, ComputationNode<T>>();
    }

    /// <summary>
    /// Creates a graph from a list of layers.
    /// </summary>
    public IComputationGraph<T> BuildFromLayers(IEnumerable<ILayer<T>> layers)
    {
        ComputationNode<T>? previousNode = null;

        // Create input node
        var inputNode = new ComputationNode<T>
        {
            OperationType = OperationType.Input,
            Name = "input"
        };
        _graph.AddNode(inputNode);
        previousNode = inputNode;

        // Convert each layer to a node
        foreach (var layer in layers)
        {
            var node = LayerToNode(layer);

            if (previousNode != null)
            {
                node.AddInput(previousNode);
            }

            _graph.AddNode(node);
            _layerToNode[layer] = node;

            previousNode = node;
        }

        // Create output node
        var outputNode = new ComputationNode<T>
        {
            OperationType = OperationType.Output,
            Name = "output"
        };

        if (previousNode != null)
        {
            outputNode.AddInput(previousNode);
        }

        _graph.AddNode(outputNode);

        return _graph;
    }

    /// <summary>
    /// Converts a layer to a computation node.
    /// </summary>
    private ComputationNode<T> LayerToNode(ILayer<T> layer)
    {
        var layerType = layer.GetType();
        var operationType = InferOperationType(layerType.Name);

        var node = new ComputationNode<T>
        {
            OperationType = operationType,
            Name = layerType.Name,
            OriginalLayer = layer,
            Parameters = ExtractParameters(layer)
        };

        return node;
    }

    /// <summary>
    /// Infers the operation type from the layer type name.
    /// </summary>
    private OperationType InferOperationType(string layerTypeName)
    {
        // Remove generic type suffixes and "Layer" suffix
        var cleanName = layerTypeName
            .Replace("`1", "")
            .Replace("Layer", "")
            .Replace("<T>", "");

        return cleanName switch
        {
            "Convolutional" => OperationType.Convolution,
            "Convolution2D" => OperationType.Convolution2D,
            "BatchNormalization" => OperationType.BatchNormalization,
            "LayerNormalization" => OperationType.LayerNormalization,
            "ReLU" => OperationType.ReLU,
            "LeakyReLU" => OperationType.LeakyReLU,
            "Sigmoid" => OperationType.Sigmoid,
            "Tanh" => OperationType.Tanh,
            "Softmax" => OperationType.Softmax,
            "MaxPooling" => OperationType.MaxPooling,
            "AveragePooling" => OperationType.AveragePooling,
            "FullyConnected" => OperationType.FullyConnected,
            "Dense" => OperationType.Dense,
            "LSTM" => OperationType.LSTM,
            "GRU" => OperationType.GRU,
            "Attention" => OperationType.Attention,
            "MultiHeadAttention" => OperationType.MultiHeadAttention,
            "Dropout" => OperationType.Dropout,
            "Flatten" => OperationType.Flatten,
            "Embedding" => OperationType.Embedding,
            _ => OperationType.Custom
        };
    }

    /// <summary>
    /// Extracts parameters from a layer.
    /// </summary>
    private Dictionary<string, object> ExtractParameters(ILayer<T> layer)
    {
        var parameters = new Dictionary<string, object>();

        // Use reflection to get layer properties
        var props = layer.GetType().GetProperties();

        foreach (var prop in props)
        {
            try
            {
                var value = prop.GetValue(layer);
                if (value != null)
                {
                    parameters[prop.Name] = value;
                }
            }
            catch (System.Reflection.TargetInvocationException)
            {
                // Skip properties whose getter throws
            }
            catch (InvalidOperationException)
            {
                // Skip properties that can't be read
            }
        }

        return parameters;
    }

    /// <summary>
    /// Gets the computation graph.
    /// </summary>
    public IComputationGraph<T> GetGraph()
    {
        return _graph;
    }
}
