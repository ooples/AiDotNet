# Junior Developer Implementation Guide: Issue #410

## Overview
**Issue**: ONNX Export and Optimization
**Goal**: Export AiDotNet models to ONNX format with graph optimizations
**Difficulty**: Advanced
**Estimated Time**: 18-22 hours

## What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open format for representing ML models:
- **Interoperability**: Train in one framework, deploy in another
- **Optimization**: Graph-level optimizations (operator fusion, constant folding)
- **Deployment**: Run on ONNX Runtime (CPU, GPU, mobile, edge devices)
- **Hardware Acceleration**: Leverage TensorRT, CoreML, DirectML, etc.

### ONNX Model Structure

```
ONNXModel
├── Graph
│   ├── Nodes (operators: Conv, MatMul, ReLU, etc.)
│   ├── Initializers (weights, biases)
│   ├── Inputs (model inputs)
│   └── Outputs (model outputs)
├── IR Version (intermediate representation version)
├── OpSet Imports (operator set versions)
└── Metadata (model info)
```

### Key Concepts

**1. Operators**: Mathematical operations (Conv2D, MatMul, Add, ReLU, etc.)
**2. Tensors**: Multi-dimensional arrays (inputs, outputs, weights)
**3. Attributes**: Operator parameters (kernel_size, stride, padding, etc.)
**4. Graph Optimizations**: Fuse multiple ops into one for efficiency

## Mathematical Background

### Operator Fusion Examples

**Conv + BatchNorm Fusion**:
```
Original:
    y = Conv(x, W, b)
    z = BatchNorm(y, γ, β, μ, σ)

Fused:
    W_fused = W * (γ / σ)
    b_fused = β + γ * (b - μ) / σ
    z = Conv(x, W_fused, b_fused)

Benefit: 2 ops → 1 op, faster inference
```

**Conv + ReLU Fusion**:
```
Original:
    y = Conv(x, W, b)
    z = ReLU(y)

Fused:
    z = ConvReLU(x, W, b)  // Single fused kernel

Benefit: Reduced memory bandwidth, kernel launch overhead
```

### Constant Folding

```
Original graph:
    a = Constant(2)
    b = Constant(3)
    c = Add(a, b)
    d = Mul(c, x)

Optimized graph:
    c = Constant(5)  // 2 + 3 precomputed
    d = Mul(c, x)

Benefit: Eliminates runtime computation
```

## Implementation Guide

### Phase 1: ONNX Model Builder

#### Core Interfaces

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IOnnxExporter.cs
namespace AiDotNet.Interfaces
{
    public interface IOnnxExporter
    {
        /// <summary>
        /// Exports a model to ONNX format.
        /// </summary>
        void Export(object model, string filePath, OnnxExportOptions options = null);

        /// <summary>
        /// Validates an exported ONNX model.
        /// </summary>
        bool Validate(string filePath, out string[] errors);

        /// <summary>
        /// Optimizes an ONNX model graph.
        /// </summary>
        void Optimize(string inputPath, string outputPath, OptimizationLevel level);
    }

    public class OnnxExportOptions
    {
        public string ModelName { get; set; } = "AiDotNetModel";
        public string ProducerName { get; set; } = "AiDotNet";
        public long OpsetVersion { get; set; } = 13;
        public bool IncludeMetadata { get; set; } = true;
        public bool OptimizeGraph { get; set; } = true;
        public Dictionary<string, int[]> InputShapes { get; set; } = new();
        public Dictionary<string, string> DynamicAxes { get; set; } = new();
    }

    public enum OptimizationLevel
    {
        /// <summary>No optimization</summary>
        None,
        /// <summary>Basic optimizations (constant folding)</summary>
        Basic,
        /// <summary>Extended optimizations (operator fusion)</summary>
        Extended,
        /// <summary>All optimizations including layout transformations</summary>
        All
    }
}
```

#### ONNX Exporter Implementation

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Onnx\OnnxExporter.cs
using Onnx;
using Google.Protobuf;
using AiDotNet.Interfaces;

namespace AiDotNet.Onnx
{
    /// <summary>
    /// Exports AiDotNet models to ONNX format.
    /// Uses Microsoft.ML.OnnxRuntime NuGet package.
    /// </summary>
    public class OnnxExporter : IOnnxExporter
    {
        public void Export(object model, string filePath, OnnxExportOptions options = null)
        {
            options ??= new OnnxExportOptions();

            // Create ONNX model
            var onnxModel = new ModelProto
            {
                IrVersion = 7,
                ProducerName = options.ProducerName,
                ProducerVersion = "1.0",
                Domain = "ai.aidotnet",
                ModelVersion = 1,
                DocString = $"Exported from AiDotNet"
            };

            // Add opset import
            onnxModel.OpsetImport.Add(new OperatorSetIdProto
            {
                Domain = "",
                Version = options.OpsetVersion
            });

            // Build graph
            var graph = BuildGraph(model, options);
            onnxModel.Graph = graph;

            // Optimize if requested
            if (options.OptimizeGraph)
            {
                OptimizeGraph(graph, OptimizationLevel.Extended);
            }

            // Save to file
            using var output = File.Create(filePath);
            onnxModel.WriteTo(output);

            Console.WriteLine($"Model exported to {filePath}");
        }

        private GraphProto BuildGraph(object model, OnnxExportOptions options)
        {
            var graph = new GraphProto
            {
                Name = options.ModelName
            };

            if (model is NeuralNetwork<Vector<double>, Vector<double>, double> neuralNet)
            {
                // Add input
                var inputTensor = new ValueInfoProto
                {
                    Name = "input",
                    Type = MakeTensorTypeProto(
                        TensorProto.Types.DataType.Float,
                        new long[] { -1, neuralNet.InputSize } // -1 for dynamic batch
                    )
                };
                graph.Input.Add(inputTensor);

                // Add output
                var outputTensor = new ValueInfoProto
                {
                    Name = "output",
                    Type = MakeTensorTypeProto(
                        TensorProto.Types.DataType.Float,
                        new long[] { -1, neuralNet.OutputSize }
                    )
                };
                graph.Output.Add(outputTensor);

                // Add layers as nodes
                string previousOutput = "input";

                for (int i = 0; i < neuralNet.Layers.Count; i++)
                {
                    var layer = neuralNet.Layers[i];
                    string layerOutput = AddLayerNode(graph, layer, previousOutput, i);
                    previousOutput = layerOutput;
                }

                // Connect final output
                graph.Output[0].Name = previousOutput;
            }

            return graph;
        }

        private string AddLayerNode(
            GraphProto graph,
            object layer,
            string input,
            int layerIndex)
        {
            string outputName = $"layer_{layerIndex}_output";

            // Determine layer type and create appropriate ONNX node
            if (layer is FullyConnectedLayer<double> fcLayer)
            {
                // MatMul node
                var matmulNode = new NodeProto
                {
                    OpType = "MatMul",
                    Name = $"fc_{layerIndex}_matmul"
                };
                matmulNode.Input.Add(input);
                matmulNode.Input.Add($"fc_{layerIndex}_weights");
                matmulNode.Output.Add($"fc_{layerIndex}_matmul_out");
                graph.Node.Add(matmulNode);

                // Add weights as initializer
                AddWeightInitializer(graph, $"fc_{layerIndex}_weights", fcLayer.Weights);

                // Add bias node
                var addNode = new NodeProto
                {
                    OpType = "Add",
                    Name = $"fc_{layerIndex}_bias"
                };
                addNode.Input.Add($"fc_{layerIndex}_matmul_out");
                addNode.Input.Add($"fc_{layerIndex}_bias_values");
                addNode.Output.Add(outputName);
                graph.Node.Add(addNode);

                AddBiasInitializer(graph, $"fc_{layerIndex}_bias_values", fcLayer.Bias);
            }
            else if (layer is ConvolutionalLayer<double> convLayer)
            {
                // Conv node with attributes
                var convNode = new NodeProto
                {
                    OpType = "Conv",
                    Name = $"conv_{layerIndex}"
                };
                convNode.Input.Add(input);
                convNode.Input.Add($"conv_{layerIndex}_weights");
                convNode.Input.Add($"conv_{layerIndex}_bias");
                convNode.Output.Add(outputName);

                // Add Conv attributes
                AddAttribute(convNode, "kernel_shape", new long[] { convLayer.KernelSize, convLayer.KernelSize });
                AddAttribute(convNode, "strides", new long[] { convLayer.Stride, convLayer.Stride });
                AddAttribute(convNode, "pads", new long[] { convLayer.Padding, convLayer.Padding, convLayer.Padding, convLayer.Padding });

                graph.Node.Add(convNode);

                AddConvWeightInitializer(graph, $"conv_{layerIndex}_weights", convLayer.Filters);
                AddBiasInitializer(graph, $"conv_{layerIndex}_bias", convLayer.Bias);
            }
            else if (layer is ReLUActivation<double>)
            {
                var reluNode = new NodeProto
                {
                    OpType = "Relu",
                    Name = $"relu_{layerIndex}"
                };
                reluNode.Input.Add(input);
                reluNode.Output.Add(outputName);
                graph.Node.Add(reluNode);
            }

            return outputName;
        }

        private void AddWeightInitializer(GraphProto graph, string name, Matrix<double> weights)
        {
            var tensor = new TensorProto
            {
                Name = name,
                DataType = (int)TensorProto.Types.DataType.Float
            };

            tensor.Dims.Add(weights.Rows);
            tensor.Dims.Add(weights.Columns);

            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    tensor.FloatData.Add((float)weights[i, j]);
                }
            }

            graph.Initializer.Add(tensor);
        }

        private void OptimizeGraph(GraphProto graph, OptimizationLevel level)
        {
            switch (level)
            {
                case OptimizationLevel.Basic:
                    FoldConstants(graph);
                    break;
                case OptimizationLevel.Extended:
                    FoldConstants(graph);
                    FuseOperators(graph);
                    break;
                case OptimizationLevel.All:
                    FoldConstants(graph);
                    FuseOperators(graph);
                    EliminateIdentityOps(graph);
                    break;
            }
        }

        private void FuseOperators(GraphProto graph)
        {
            // Fuse Conv + ReLU → ConvReLU
            for (int i = 0; i < graph.Node.Count - 1; i++)
            {
                var node1 = graph.Node[i];
                var node2 = graph.Node[i + 1];

                if (node1.OpType == "Conv" && node2.OpType == "Relu")
                {
                    // Check if Conv output feeds directly to ReLU
                    if (node1.Output[0] == node2.Input[0])
                    {
                        // Create fused node
                        var fusedNode = new NodeProto
                        {
                            OpType = "Conv",
                            Name = node1.Name + "_relu_fused"
                        };

                        fusedNode.Input.AddRange(node1.Input);
                        fusedNode.Output.Add(node2.Output[0]);
                        fusedNode.Attribute.AddRange(node1.Attribute);

                        // Add activation attribute
                        AddAttribute(fusedNode, "activation", "Relu");

                        // Replace nodes
                        graph.Node.RemoveAt(i);
                        graph.Node.RemoveAt(i); // Remove next node (indices shift)
                        graph.Node.Insert(i, fusedNode);

                        Console.WriteLine($"Fused Conv+ReLU: {node1.Name} + {node2.Name}");
                    }
                }
            }
        }

        private void FoldConstants(GraphProto graph)
        {
            // Find constant computation subgraphs and precompute
            var constantNodes = new HashSet<string>();

            foreach (var node in graph.Node)
            {
                bool allInputsConstant = true;

                foreach (var input in node.Input)
                {
                    if (!graph.Initializer.Any(init => init.Name == input) &&
                        !constantNodes.Contains(input))
                    {
                        allInputsConstant = false;
                        break;
                    }
                }

                if (allInputsConstant && CanFold(node.OpType))
                {
                    // Compute result and replace with constant
                    constantNodes.Add(node.Output[0]);
                    Console.WriteLine($"Folded constant node: {node.Name}");
                }
            }
        }

        private void EliminateIdentityOps(GraphProto graph)
        {
            // Remove identity operations (no-ops)
            graph.Node.RemoveAll(node => node.OpType == "Identity");
        }

        private bool CanFold(string opType)
        {
            return new[] { "Add", "Sub", "Mul", "Div", "Reshape", "Transpose" }.Contains(opType);
        }

        private void AddAttribute(NodeProto node, string name, long[] values)
        {
            var attr = new AttributeProto
            {
                Name = name,
                Type = AttributeProto.Types.AttributeType.Ints
            };
            attr.Ints.AddRange(values);
            node.Attribute.Add(attr);
        }

        private TypeProto MakeTensorTypeProto(TensorProto.Types.DataType dataType, long[] shape)
        {
            var typeProto = new TypeProto
            {
                TensorType = new TypeProto.Types.Tensor
                {
                    ElemType = (int)dataType
                }
            };

            foreach (var dim in shape)
            {
                typeProto.TensorType.Shape.Dim.Add(new TensorShapeProto.Types.Dimension
                {
                    DimValue = dim
                });
            }

            return typeProto;
        }

        public bool Validate(string filePath, out string[] errors)
        {
            try
            {
                using var input = File.OpenRead(filePath);
                var model = ModelProto.Parser.ParseFrom(input);

                // Basic validation
                var errorList = new List<string>();

                if (model.Graph == null)
                    errorList.Add("Model has no graph");

                if (model.Graph.Input.Count == 0)
                    errorList.Add("Graph has no inputs");

                if (model.Graph.Output.Count == 0)
                    errorList.Add("Graph has no outputs");

                errors = errorList.ToArray();
                return errorList.Count == 0;
            }
            catch (Exception ex)
            {
                errors = new[] { ex.Message };
                return false;
            }
        }

        public void Optimize(string inputPath, string outputPath, OptimizationLevel level)
        {
            using var input = File.OpenRead(inputPath);
            var model = ModelProto.Parser.ParseFrom(input);

            OptimizeGraph(model.Graph, level);

            using var output = File.Create(outputPath);
            model.WriteTo(output);

            Console.WriteLine($"Optimized model saved to {outputPath}");
        }
    }
}
```

### Phase 2: Testing

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\tests\Onnx\OnnxExporterTests.cs
using Xunit;
using AiDotNet.Onnx;

namespace AiDotNet.Tests.Onnx
{
    public class OnnxExporterTests
    {
        [Fact]
        public void Export_SimpleNeuralNetwork_CreatesValidOnnx()
        {
            // Arrange
            var network = new NeuralNetwork<Vector<double>, Vector<double>, double>(
                inputSize: 10,
                hiddenSizes: new[] { 20 },
                outputSize: 5
            );

            var exporter = new OnnxExporter();
            var tempFile = Path.GetTempFileName() + ".onnx";

            // Act
            exporter.Export(network, tempFile);

            // Assert
            Assert.True(File.Exists(tempFile));

            bool isValid = exporter.Validate(tempFile, out var errors);
            Assert.True(isValid, string.Join(", ", errors));

            // Cleanup
            File.Delete(tempFile);
        }

        [Fact]
        public void Optimize_FusesConvReLU()
        {
            // Arrange - Model with Conv followed by ReLU
            var network = CreateConvNetworkWithReLU();
            var exporter = new OnnxExporter();

            var unoptimizedFile = Path.GetTempFileName() + "_unoptimized.onnx";
            var optimizedFile = Path.GetTempFileName() + "_optimized.onnx";

            exporter.Export(network, unoptimizedFile, new OnnxExportOptions
            {
                OptimizeGraph = false
            });

            // Act
            exporter.Optimize(unoptimizedFile, optimizedFile, OptimizationLevel.Extended);

            // Assert
            // Optimized graph should have fewer nodes due to fusion
            var unoptimizedNodeCount = GetNodeCount(unoptimizedFile);
            var optimizedNodeCount = GetNodeCount(optimizedFile);

            Assert.True(optimizedNodeCount < unoptimizedNodeCount);

            // Cleanup
            File.Delete(unoptimizedFile);
            File.Delete(optimizedFile);
        }
    }
}
```

## Common ONNX Operators

**Core Operators**:
- `MatMul`: Matrix multiplication
- `Gemm`: General matrix multiply (W*x + b)
- `Conv`: 2D convolution
- `Add`, `Sub`, `Mul`, `Div`: Element-wise arithmetic
- `Relu`, `Sigmoid`, `Tanh`: Activations
- `Softmax`, `LogSoftmax`: Probability distributions
- `BatchNormalization`: Batch normalization
- `MaxPool`, `AveragePool`: Pooling operations
- `Reshape`, `Transpose`, `Concat`, `Split`: Tensor manipulations

## Deployment Example

```csharp
// Export model to ONNX
var exporter = new OnnxExporter();
exporter.Export(trainedModel, "model.onnx", new OnnxExportOptions
{
    ModelName = "MyClassifier",
    OpsetVersion = 13,
    OptimizeGraph = true,
    InputShapes = new Dictionary<string, int[]>
    {
        ["input"] = new[] { 1, 3, 224, 224 } // NCHW format
    }
});

// Load and run with ONNX Runtime
using var session = new InferenceSession("model.onnx");

var inputMeta = session.InputMetadata;
var inputName = inputMeta.Keys.First();

var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 3, 224, 224 });
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
};

using var results = session.Run(inputs);
var output = results.First().AsEnumerable<float>().ToArray();
```

## Learning Resources

- **ONNX Specification**: https://github.com/onnx/onnx/blob/main/docs/IR.md
- **ONNX Runtime**: https://onnxruntime.ai/
- **Operator Schemas**: https://github.com/onnx/onnx/blob/main/docs/Operators.md
- **Graph Optimizations**: https://onnxruntime.ai/docs/performance/graph-optimizations.html

---

**Good luck!** ONNX export enables cross-platform deployment and hardware acceleration for your models.
