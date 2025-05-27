namespace AiDotNet.Enums;

/// <summary>
/// Defines the available model compression techniques.
/// </summary>
public enum CompressionTechnique
{
    /// <summary>
    /// No compression is applied.
    /// </summary>
    None = 0,
    
    /// <summary>
    /// Reduces the numerical precision of model parameters.
    /// </summary>
    Quantization = 1,
    
    /// <summary>
    /// Removes less important connections from the neural network.
    /// </summary>
    Pruning = 2,
    
    /// <summary>
    /// Trains a smaller model to mimic a larger one.
    /// </summary>
    KnowledgeDistillation = 3,
    
    /// <summary>
    /// Decomposes weight matrices into lower-rank approximations.
    /// </summary>
    LowRankFactorization = 4,
    
    /// <summary>
    /// Uses Huffman coding for efficient parameter storage.
    /// </summary>
    HuffmanCoding = 5,
    
    /// <summary>
    /// Combines quantization and pruning.
    /// </summary>
    QuantizedPruning = 6,
    
    /// <summary>
    /// Applies clustering to group similar weights together.
    /// </summary>
    WeightClustering = 7,
    
    /// <summary>
    /// Uses tensor decomposition to compress higher-dimensional weight tensors.
    /// </summary>
    TensorDecomposition = 8
}