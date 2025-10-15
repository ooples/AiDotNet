using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.LossFunctions;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Vision Transformer (ViT) for image classification and feature extraction
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class VisionTransformer<T> : NeuralNetworkBase<T>
    {
        private readonly int imageSize = default!;
        private readonly int patchSize = default!;
        private readonly int numPatches = default!;
        private readonly int embedDim = default!;
        private readonly int depth = default!;
        private readonly int numHeads = default!;
        private readonly int mlpDim = default!;
        private readonly int numClasses = default!;
        private readonly double dropoutRate = default!;

        // Layers
        private ILayer<T> patchEmbedding = null!;
        private Tensor<T> positionEmbedding = null!;
        private Tensor<T> classToken = null!;
        private List<TransformerBlock<T>> transformerBlocks = null!;
        private ILayer<T> mlpHead = null!;
        private LayerNormalizationLayer<T> layerNorm = null!;
        
        // Optimizer for training
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        
        public VisionTransformer(
            NeuralNetworkArchitecture<double> architecture,
            int imageSize = 224,
            int patchSize = 16,
            int embedDim = 768,
            int depth = 12,
            int numHeads = 12,
            int mlpDim = 3072,
            int numClasses = 1000,
            double dropoutRate = 0.1,
            string modelName = "VisionTransformer") : base(
                new NeuralNetworkArchitecture<T>(
                    NetworkComplexity.Deep,
                    NeuralNetworkTaskType.ImageClassification
                ),
                new CrossEntropyLoss<T>(),
                1.0) // maxGradNorm as double
        {
            this.imageSize = imageSize;
            this.patchSize = patchSize;
            this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
            this.embedDim = embedDim;
            this.depth = depth;
            this.numHeads = numHeads;
            this.mlpDim = mlpDim;
            this.numClasses = numClasses;
            this.dropoutRate = dropoutRate;
            
            _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            
            InitializeLayers();
        }
        
        public override bool SupportsTraining => true;
        
        protected override void InitializeLayers()
        {
            // Patch embedding layer
            patchEmbedding = new PatchEmbeddingLayer<T>(patchSize, embedDim);
            
            // Position embeddings
            positionEmbedding = new Tensor<T>(new[] { 1, numPatches + 1, embedDim });
            InitializePositionEmbeddings();

            // Class token
            classToken = new Tensor<T>(new[] { 1, 1, embedDim });
            InitializeXavier(classToken);

            // Transformer blocks
            transformerBlocks = new List<TransformerBlock<T>>();
            for (int i = 0; i < depth; i++)
            {
                transformerBlocks.Add(new TransformerBlock<T>(embedDim, numHeads, mlpDim, dropoutRate));
            }

            // Layer normalization
            layerNorm = new LayerNormalizationLayer<T>(embedDim);
            
            // MLP head for classification
            mlpHead = new DenseLayer<T>(embedDim, numClasses, (IActivationFunction<T>?)null);
            
            // Add layers to the Layers collection for base class functionality
            Layers.Clear();
            Layers.Add(patchEmbedding);
            Layers.AddRange(transformerBlocks);
            Layers.Add(layerNorm);
            Layers.Add(mlpHead);
        }
        
        public Tensor<T> Forward(Tensor<T> input)
        {
            // Expected input shape: [batch, channels, height, width]
            var batchSize = input.Shape[0];

            // Convert image to patches and embed them
            var patches = patchEmbedding.Forward(input);
            
            // Add class token
            // Create batch of class tokens
            var clsTokens = new Tensor<T>(new[] { batchSize, 1, embedDim });
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < embedDim; i++)
                {
                    clsTokens[b, 0, i] = classToken[0, 0, i];
                }
            }
            patches = ConcatenateClassToken(clsTokens, patches);

            // Add position embeddings
            patches = patches.Add(positionEmbedding);

            // Apply transformer blocks
            var x = patches;
            foreach (var block in transformerBlocks)
            {
                x = block.Forward(x);
            }

            // Apply final layer norm
            x = layerNorm.Forward(x);

            // Extract class token (first token)
            var classOutput = ExtractClassToken(x);

            // Apply MLP head for classification
            var output = mlpHead.Forward(classOutput);

            return output;
        }
        
        public Tensor<T> Backward(Tensor<T> gradOutput)
        {
            // Backward through MLP head
            var gradClass = mlpHead.Backward(gradOutput);
            
            // Expand gradient back to include all tokens
            var gradX = ExpandClassTokenGradient(gradClass);
            
            // Backward through layer norm
            gradX = layerNorm.Backward(gradX);
            
            // Backward through transformer blocks
            for (int i = transformerBlocks.Count - 1; i >= 0; i--)
            {
                gradX = transformerBlocks[i].Backward(gradX);
            }
            
            // Remove position embedding gradient
            // Position embeddings are learned parameters
            
            // Remove class token gradient
            var gradPatches = RemoveClassTokenGradient(gradX);
            
            // Backward through patch embedding
            return patchEmbedding.Backward(gradPatches);
        }
        
        public override Tensor<T> Predict(Tensor<T> input)
        {
            IsTrainingMode = false;
            var output = Forward(input);
            IsTrainingMode = true;
            return output;
        }
        
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            IsTrainingMode = true;
            
            // Forward pass
            var output = Forward(input);
            
            // Calculate loss - convert tensors to vectors for loss calculation
            var outputVector = new Vector<T>(output.Length);
            var expectedVector = new Vector<T>(expectedOutput.Length);
            
            // Copy data from tensors to vectors
            for (int i = 0; i < output.Length; i++)
            {
                var indices = GetMultiDimensionalIndices(i, output.Shape);
                outputVector[i] = output[indices];
            }
            
            for (int i = 0; i < expectedOutput.Length; i++)
            {
                var indices = GetMultiDimensionalIndices(i, expectedOutput.Shape);
                expectedVector[i] = expectedOutput[indices];
            }
            
            var loss = LossFunction.CalculateLoss(outputVector, expectedVector);
            LastLoss = loss;
            
            // Backward pass - simplified
            var lossGradient = output.Subtract(expectedOutput);
            Backward(lossGradient);
            
            // Update parameters using the optimizer
            // This is simplified - actual implementation would use optimizer properly
        }
        
        public override void UpdateParameters(Vector<T> parameters)
        {
            SetParameters(parameters);
        }
        
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.VisionTransformer
            };
        }
        
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new VisionTransformer<T>(
                imageSize, patchSize, embedDim, depth, numHeads, 
                mlpDim, numClasses, dropoutRate, "VisionTransformer"
            );
        }
        
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(imageSize);
            writer.Write(patchSize);
            writer.Write(embedDim);
            writer.Write(depth);
            writer.Write(numHeads);
            writer.Write(mlpDim);
            writer.Write(numClasses);
            writer.Write(dropoutRate);
            
            // Serialize position embeddings
            writer.Write(positionEmbedding.Length);
            for (int i = 0; i < positionEmbedding.Length; i++)
            {
                var indices = GetMultiDimensionalIndices(i, positionEmbedding.Shape);
                writer.Write(Convert.ToDouble((object)positionEmbedding[indices]!));
            }
            
            // Serialize class token
            writer.Write(classToken.Length);
            for (int i = 0; i < classToken.Length; i++)
            {
                var indices = GetMultiDimensionalIndices(i, classToken.Shape);
                writer.Write(Convert.ToDouble((object)classToken[indices]!));
            }
        }
        
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            // Note: The constructor parameters are already set when CreateNewInstance is called
            // We just need to read the serialized data to match the writer
            
            // Read configuration (to consume the data, values already set by constructor)
            reader.ReadInt32(); // imageSize
            reader.ReadInt32(); // patchSize
            reader.ReadInt32(); // embedDim
            reader.ReadInt32(); // depth
            reader.ReadInt32(); // numHeads
            reader.ReadInt32(); // mlpDim
            reader.ReadInt32(); // numClasses
            reader.ReadDouble(); // dropoutRate
            
            // Read position embeddings
            int posEmbedLength = reader.ReadInt32();
            for (int i = 0; i < posEmbedLength && i < positionEmbedding.Length; i++)
            {
                var indices = GetMultiDimensionalIndices(i, positionEmbedding.Shape);
                positionEmbedding[indices] = this.NumOps.FromDouble(reader.ReadDouble());
            }
            
            // Read class token
            int clsTokenLength = reader.ReadInt32();
            for (int i = 0; i < clsTokenLength && i < classToken.Length; i++)
            {
                var indices = GetMultiDimensionalIndices(i, classToken.Shape);
                classToken[indices] = this.NumOps.FromDouble(reader.ReadDouble());
            }
        }
        
        /// <summary>
        /// Extract features from intermediate layers
        /// </summary>
        public Tensor<T> ExtractFeatures(Tensor<T> input, int layerIndex = -1)
        {
            var batchSize = input.Shape[0];

            // Convert image to patches and embed them
            var patches = patchEmbedding.Forward(input);
            
            // Add class token
            // Create batch of class tokens
            var clsTokens = new Tensor<T>(new[] { batchSize, 1, embedDim });
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < embedDim; i++)
                {
                    clsTokens[b, 0, i] = classToken[0, 0, i];
                }
            }
            patches = ConcatenateClassToken(clsTokens, patches);

            // Add position embeddings
            patches = patches.Add(positionEmbedding);

            // Apply transformer blocks up to specified layer
            var x = patches;
            var maxLayer = layerIndex < 0 ? transformerBlocks.Count : Math.Min(layerIndex, transformerBlocks.Count);

            for (int i = 0; i < maxLayer; i++)
            {
                x = transformerBlocks[i].Forward(x);
            }

            return x;
        }
        
        private Tensor<double> RepeatTensor(Tensor<double> tensor, int batchSize)
        {
            // Replicate tensor along batch dimension
            // For a tensor with shape [1, seqLen, dim], create [batchSize, seqLen, dim]
            var originalShape = tensor.Shape;
            var newShape = new int[originalShape.Length];
            newShape[0] = batchSize;
            for (int i = 1; i < originalShape.Length; i++)
            {
                newShape[i] = originalShape[i];
            }

            var originalData = tensor.Data;
            var elementCount = originalData.Length;
            var newData = new double[batchSize * elementCount];

            // Replicate data for each batch
            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(originalData, 0, newData, b * elementCount, elementCount);
            }

            return new Tensor<double>(newShape, new Vector<double>(newData));
        }

        private void InitializePositionEmbeddings()
        {
            // Initialize position embeddings with sine/cosine positional encoding
            var positions = numPatches + 1; // +1 for class token

            for (int pos = 0; pos < positions; pos++)
            {
                for (int i = 0; i < embedDim; i++)
                {
                    if (i % 2 == 0)
                    {
                        positionEmbedding[0, pos, i] = this.NumOps.FromDouble(Math.Sin(pos / Math.Pow(10000, 2.0 * i / embedDim)));
                    }
                    else
                    {
                        positionEmbedding[0, pos, i] = this.NumOps.FromDouble(Math.Cos(pos / Math.Pow(10000, 2.0 * (i - 1) / embedDim)));
                    }
                }
            }
        }
        
        private void InitializeXavier(Tensor<T> tensor)
        {
            var fan_in = tensor.Shape[tensor.Shape.Length - 1];
            var fan_out = tensor.Shape.Length > 1 ? tensor.Shape[tensor.Shape.Length - 2] : 1;
            var scale = Math.Sqrt(2.0 / (fan_in + fan_out));
            
            var random = new Random();
            
            for (int i = 0; i < tensor.Length; i++)
            {
                tensor[i] = this.NumOps.FromDouble((random.NextDouble() * 2 - 1) * scale);
            }
        }
        
        private Tensor<T> ConcatenateClassToken(Tensor<T> classTokens, Tensor<T> patches)
        {
            // Concatenate class token to the beginning of patch sequence
            // patches shape: [batch, numPatches, embedDim]
            // classTokens shape: [batch, 1, embedDim]
            // result shape: [batch, numPatches + 1, embedDim]
            
            var batchSize = patches.Shape[0];
            var numPatches = patches.Shape[1];
            var embedDim = patches.Shape[2];
            
            var result = new Tensor<T>(new[] { batchSize, numPatches + 1, embedDim });
            
            // Copy class tokens
            for (int b = 0; b < batchSize; b++)
            {
                for (int e = 0; e < embedDim; e++)
                {
                    result[b, 0, e] = classTokens[b, 0, e];
                }
            }
            
            // Copy patches
            for (int b = 0; b < batchSize; b++)
            {
                for (int p = 0; p < numPatches; p++)
                {
                    for (int e = 0; e < embedDim; e++)
                    {
                        result[b, p + 1, e] = patches[b, p, e];
                    }
                }
            }
            
            return result;
        }
        
        private Tensor<T> ExtractClassToken(Tensor<T> x)
        {
            // Extract the first token (class token)
            // x shape: [batch, numPatches + 1, embedDim]
            // result shape: [batch, embedDim]
            
            var batchSize = x.Shape[0];
            var embedDim = x.Shape[2];
            
            var result = new Tensor<T>(new[] { batchSize, embedDim });
            
            for (int b = 0; b < batchSize; b++)
            {
                for (int e = 0; e < embedDim; e++)
                {
                    result[b, e] = x[b, 0, e];
                }
            }
            
            return result;
        }
        
        private Tensor<T> ExpandClassTokenGradient(Tensor<T> gradClass)
        {
            // Expand gradient from class token to all tokens
            // gradClass shape: [batch, embedDim]
            // result shape: [batch, numPatches + 1, embedDim]
            
            var batchSize = gradClass.Shape[0];
            var embedDim = gradClass.Shape[1];
            
            var result = new Tensor<T>(new[] { batchSize, numPatches + 1, embedDim });
            
            // Only the class token (first token) receives gradient
            for (int b = 0; b < batchSize; b++)
            {
                for (int e = 0; e < embedDim; e++)
                {
                    result[b, 0, e] = gradClass[b, e];
                    // All other tokens get zero gradient
                    for (int p = 1; p < numPatches + 1; p++)
                    {
                        result[b, p, e] = this.NumOps.Zero;
                    }
                }
            }
            
            return result;
        }
        
        private Tensor<T> RemoveClassTokenGradient(Tensor<T> gradX)
        {
            // Remove class token from gradient
            // gradX shape: [batch, numPatches + 1, embedDim]
            // result shape: [batch, numPatches, embedDim]
            
            var batchSize = gradX.Shape[0];
            var embedDim = gradX.Shape[2];
            
            var result = new Tensor<T>(new[] { batchSize, numPatches, embedDim });
            
            // Skip the first token (class token)
            for (int b = 0; b < batchSize; b++)
            {
                for (int p = 0; p < numPatches; p++)
                {
                    for (int e = 0; e < embedDim; e++)
                    {
                        result[b, p, e] = gradX[b, p + 1, e];
                    }
                }
            }
            
            return result;
        }
        
        private int[] GetMultiDimensionalIndices(int flatIndex, int[] shape)
        {
            var indices = new int[shape.Length];
            var remaining = flatIndex;
            
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                indices[i] = remaining % shape[i];
                remaining /= shape[i];
            }
            
            return indices;
        }
    }
}