using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Deployment.Export.Onnx;

/// <summary>
/// Exports AiDotNet models to ONNX format for cross-platform deployment.
/// Properly integrates with IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class OnnxModelExporter<T, TInput, TOutput> : ModelExporterBase<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public override string ExportFormat => "ONNX";

    /// <inheritdoc/>
    public override string FileExtension => ".onnx";

    /// <inheritdoc/>
    public override byte[] ExportToBytes(IFullModel<T, TInput, TOutput> model, ExportConfiguration config)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        // Build ONNX graph based on model type
        // IFullModel can be cast to more specific types for specialized export logic
        var onnxGraph = model switch
        {
            INeuralNetworkModel<T> neuralNetwork => BuildNeuralNetworkGraph(neuralNetwork, config),
            IModel<T[], T[], object> linearModel => BuildLinearModelGraph(linearModel, config),
            _ => throw new NotSupportedException($"Model type {model.GetType().Name} is not supported for ONNX export")
        };

        // Convert to ONNX proto format
        return SerializeOnnxGraph(onnxGraph, config);
    }

    /// <inheritdoc/>
    public override IReadOnlyList<string> GetValidationErrors(IFullModel<T, TInput, TOutput> model)
    {
        var errors = new List<string>();

        if (model == null)
        {
            errors.Add("Model is null");
            return errors;
        }

        // Check supported model types
        if (model is not INeuralNetworkModel<T> && model is not IModel<T[], T[], object>)
        {
            errors.Add($"Model type {model.GetType().Name} is not supported for ONNX export");
        }

        // Additional validation for neural networks
        if (model is INeuralNetworkModel<T> nn)
        {
            var unsupportedLayers = GetUnsupportedLayers(nn);
            if (unsupportedLayers.Any())
            {
                errors.Add($"Model contains unsupported layers: {string.Join(", ", unsupportedLayers)}");
            }
        }

        return errors;
    }

    private OnnxGraph BuildNeuralNetworkGraph(INeuralNetworkModel<T> model, ExportConfiguration config)
    {
        var graph = new OnnxGraph
        {
            Name = config.ModelName ?? "AiDotNet_NeuralNetwork",
            OpsetVersion = config.OpsetVersion
        };

        // Get layers from the neural network
        var layers = GetLayersFromModel(model);

        // Build input node
        var inputShape = GetInputShapeWithBatch(model, config);
        graph.Inputs.Add(new OnnxNode
        {
            Name = "input",
            Shape = inputShape,
            DataType = GetOnnxDataType<T>()
        });

        // Convert each layer to ONNX operations
        string currentOutput = "input";
        int layerIndex = 0;

        foreach (var layer in layers)
        {
            var layerOutput = $"layer_{layerIndex}_output";
            var onnxOps = ConvertLayerToOnnxOperations(layer, currentOutput, layerOutput, layerIndex);

            foreach (var op in onnxOps)
            {
                graph.Operations.Add(op);
            }

            currentOutput = layerOutput;
            layerIndex++;
        }

        // Build output node
        graph.Outputs.Add(new OnnxNode
        {
            Name = "output",
            Shape = null, // Will be inferred
            DataType = GetOnnxDataType<T>()
        });

        // Rename last layer output to match graph output
        if (graph.Operations.Count > 0)
        {
            var lastOp = graph.Operations[graph.Operations.Count - 1];
            lastOp.Outputs[0] = "output";
        }

        return graph;
    }

    private OnnxGraph BuildLinearModelGraph(IModel<T[], T[], object> model, ExportConfiguration config)
    {
        var graph = new OnnxGraph
        {
            Name = config.ModelName ?? "AiDotNet_LinearModel",
            OpsetVersion = config.OpsetVersion
        };

        // IModel extends IFullModel which extends IParameterizable
        // We need to cast to access GetParameters - IModel doesn't directly expose this
        if (model is not IFullModel<T, T[], T[]> fullModel)
        {
            throw new InvalidOperationException("Model must implement IFullModel to be exported");
        }

        var inputShape = GetInputShapeWithBatch(fullModel, config);

        graph.Inputs.Add(new OnnxNode
        {
            Name = "input",
            Shape = inputShape,
            DataType = GetOnnxDataType<T>()
        });

        // Add MatMul operation for linear model
        var parameters = fullModel.GetParameters();

        graph.Operations.Add(new OnnxOperation
        {
            Type = "MatMul",
            Inputs = new List<string> { "input", "weights" },
            Outputs = new List<string> { "matmul_output" },
            Attributes = new Dictionary<string, object>()
        });

        // Add bias if available
        graph.Operations.Add(new OnnxOperation
        {
            Type = "Add",
            Inputs = new List<string> { "matmul_output", "bias" },
            Outputs = new List<string> { "output" },
            Attributes = new Dictionary<string, object>()
        });

        // Store weights as initializers
        graph.Initializers.Add("weights", parameters);

        graph.Outputs.Add(new OnnxNode
        {
            Name = "output",
            Shape = null,
            DataType = GetOnnxDataType<T>()
        });

        return graph;
    }

    private List<ILayer<T>> GetLayersFromModel(INeuralNetworkModel<T> model)
    {
        var layers = new List<ILayer<T>>();

        // Use reflection to get layers from the model
        var modelType = model.GetType();
        var layersProperty = modelType.GetProperty("Layers");

        if (layersProperty != null)
        {
            var layersValue = layersProperty.GetValue(model);
            if (layersValue is IEnumerable<ILayer<T>> layerList)
            {
                layers.AddRange(layerList);
            }
        }

        return layers;
    }

    private List<OnnxOperation> ConvertLayerToOnnxOperations(ILayer<T> layer, string input, string output, int layerIndex)
    {
        var operations = new List<OnnxOperation>();
        var layerTypeName = layer.GetType().Name;

        // Map common layer types to ONNX operations
        switch (layerTypeName)
        {
            case "DenseLayer":
            case "FullyConnectedLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "Gemm", // General Matrix Multiplication
                    Inputs = new List<string> { input, $"layer_{layerIndex}_weights", $"layer_{layerIndex}_bias" },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>
                    {
                        ["alpha"] = 1.0f,
                        ["beta"] = 1.0f,
                        ["transB"] = 1
                    }
                });
                break;

            case "ConvolutionLayer":
            case "Conv2DLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "Conv",
                    Inputs = new List<string> { input, $"layer_{layerIndex}_weights", $"layer_{layerIndex}_bias" },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>
                    {
                        ["kernel_shape"] = new[] { 3, 3 }, // Default, should be extracted from layer
                        ["pads"] = new[] { 1, 1, 1, 1 },
                        ["strides"] = new[] { 1, 1 }
                    }
                });
                break;

            case "MaxPoolingLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "MaxPool",
                    Inputs = new List<string> { input },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>
                    {
                        ["kernel_shape"] = new[] { 2, 2 },
                        ["strides"] = new[] { 2, 2 }
                    }
                });
                break;

            case "ReLULayer":
                operations.Add(new OnnxOperation
                {
                    Type = "Relu",
                    Inputs = new List<string> { input },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>()
                });
                break;

            case "SigmoidLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "Sigmoid",
                    Inputs = new List<string> { input },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>()
                });
                break;

            case "TanhLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "Tanh",
                    Inputs = new List<string> { input },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>()
                });
                break;

            case "SoftmaxLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "Softmax",
                    Inputs = new List<string> { input },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>
                    {
                        ["axis"] = -1
                    }
                });
                break;

            case "BatchNormalizationLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "BatchNormalization",
                    Inputs = new List<string>
                    {
                        input,
                        $"layer_{layerIndex}_scale",
                        $"layer_{layerIndex}_bias",
                        $"layer_{layerIndex}_mean",
                        $"layer_{layerIndex}_var"
                    },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>
                    {
                        ["epsilon"] = 1e-5f
                    }
                });
                break;

            case "DropoutLayer":
                // Dropout is typically not needed during inference
                operations.Add(new OnnxOperation
                {
                    Type = "Identity",
                    Inputs = new List<string> { input },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>()
                });
                break;

            case "LSTMLayer":
                operations.Add(new OnnxOperation
                {
                    Type = "LSTM",
                    Inputs = new List<string>
                    {
                        input,
                        $"layer_{layerIndex}_W",
                        $"layer_{layerIndex}_R",
                        $"layer_{layerIndex}_B"
                    },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>
                    {
                        ["hidden_size"] = 128 // Default, should be extracted from layer
                    }
                });
                break;

            case "GRULayer":
                operations.Add(new OnnxOperation
                {
                    Type = "GRU",
                    Inputs = new List<string>
                    {
                        input,
                        $"layer_{layerIndex}_W",
                        $"layer_{layerIndex}_R",
                        $"layer_{layerIndex}_B"
                    },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>
                    {
                        ["hidden_size"] = 128
                    }
                });
                break;

            default:
                // For unsupported layers, use Identity as placeholder
                operations.Add(new OnnxOperation
                {
                    Type = "Identity",
                    Inputs = new List<string> { input },
                    Outputs = new List<string> { output },
                    Attributes = new Dictionary<string, object>()
                });
                break;
        }

        return operations;
    }

    private List<string> GetUnsupportedLayers(INeuralNetworkModel<T> model)
    {
        var unsupported = new List<string>();
        var layers = GetLayersFromModel(model);

        var supportedLayerTypes = new HashSet<string>
        {
            "DenseLayer", "FullyConnectedLayer", "ConvolutionLayer", "Conv2DLayer",
            "MaxPoolingLayer", "ReLULayer", "SigmoidLayer", "TanhLayer",
            "SoftmaxLayer", "BatchNormalizationLayer", "DropoutLayer",
            "LSTMLayer", "GRULayer"
        };

        unsupported.AddRange(layers
            .Select(layer => layer.GetType().Name)
            .Where(layerType => !supportedLayerTypes.Contains(layerType)));

        return unsupported;
    }

    private int[] GetInputShapeWithBatch<TModelInput, TModelOutput>(IFullModel<T, TModelInput, TModelOutput> model, ExportConfiguration config)
    {
        // Get input shape from config or infer from model
        if (config.InputShape != null && config.InputShape.Length > 0)
        {
            var shape = config.InputShape;

            // Use -1 for dynamic batch dimension if enabled, otherwise use batch size
            return (config.UseDynamicShapes
                ? new[] { -1 }.Concat(shape)
                : new[] { config.BatchSize }.Concat(shape)
            ).ToArray();
        }

        // Try to infer from model parameters
        var parameters = model.GetParameters();
        if (parameters != null && parameters.Length > 0)
        {
            var shape = new[] { parameters.Length };

            return (config.UseDynamicShapes
                ? new[] { -1 }.Concat(shape)
                : new[] { config.BatchSize }.Concat(shape)
            ).ToArray();
        }

        throw new InvalidOperationException(
            "Could not determine input shape. Please specify InputShape in ExportConfiguration.");
    }

    private string GetOnnxDataType<TData>()
    {
        return typeof(TData).Name switch
        {
            "Single" => "float",
            "Double" => "double",
            "Int32" => "int32",
            "Int64" => "int64",
            "Int16" => "int16",
            "Byte" => "uint8",
            _ => "float"
        };
    }

    private byte[] SerializeOnnxGraph(OnnxGraph graph, ExportConfiguration config)
    {
        // Serialize to proper ONNX protobuf format
        return OnnxProto.CreateModelProto(graph, config);
    }
}
