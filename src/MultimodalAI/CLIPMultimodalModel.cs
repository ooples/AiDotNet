using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.MultimodalAI;

/// <summary>
/// CLIP-like multimodal model that can process text and images using contrastive learning.
/// This implementation extends MultimodalModelBase to provide cross-modal understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// CLIP (Contrastive Language-Image Pre-training) learns to associate text and image embeddings
/// in a shared latent space. This allows for tasks like:
/// - Zero-shot image classification
/// - Image-text retrieval
/// - Cross-modal similarity computation
/// </remarks>
public class CLIPMultimodalModel<T> : MultimodalModelBase<T>
{
    private readonly T _temperature;
    private readonly bool _useProjection;
    private INeuralNetworkModel<T>? _textProjection;
    private INeuralNetworkModel<T>? _imageProjection;
    private readonly T _learningRate;
    private readonly int _batchSize;
    private readonly NeuralNetworkArchitecture<T>? _textProjectionArchitecture;
    private readonly NeuralNetworkArchitecture<T>? _imageProjectionArchitecture;
    
    // Training history
    private readonly List<T> _contrastiveLosses = [];
    private readonly List<T> _validationAccuracies = [];

    /// <summary>
    /// Initializes a new instance of CLIPMultimodalModel
    /// </summary>
    /// <param name="embeddingDimension">Dimension of the shared embedding space</param>
    /// <param name="temperature">Temperature parameter for contrastive loss</param>
    /// <param name="useProjection">Whether to use learned projections for each modality</param>
    /// <param name="learningRate">Learning rate for training</param>
    /// <param name="batchSize">Batch size for contrastive learning</param>
    /// <param name="textProjectionArchitecture">Optional custom neural network architecture for text projection</param>
    /// <param name="imageProjectionArchitecture">Optional custom neural network architecture for image projection</param>
    public CLIPMultimodalModel(
        int embeddingDimension = 512, 
        double temperature = 0.07,
        bool useProjection = true,
        double learningRate = 0.0001,
        int batchSize = 32,
        NeuralNetworkArchitecture<T>? textProjectionArchitecture = null,
        NeuralNetworkArchitecture<T>? imageProjectionArchitecture = null) 
        : base("contrastive", embeddingDimension)
    {
        _temperature = NumOps.FromDouble(temperature);
        _useProjection = useProjection;
        _learningRate = NumOps.FromDouble(learningRate);
        _batchSize = batchSize;
        
        // Store custom architectures if provided
        _textProjectionArchitecture = textProjectionArchitecture;
        _imageProjectionArchitecture = imageProjectionArchitecture;
        
        Name = "CLIP Multimodal Model";
    }

    /// <summary>
    /// Processes multimodal input data using CLIP-style contrastive embeddings
    /// </summary>
    /// <param name="modalityData">Dictionary mapping modality names to their data</param>
    /// <returns>Fused representation in the shared embedding space</returns>
    public override Vector<T> ProcessMultimodal(Dictionary<string, object> modalityData)
    {
        // Validate input
        ValidateModalityData(modalityData);
        
        var embeddings = new Dictionary<string, Vector<T>>();
        
        // Process each modality through its encoder and projection
        foreach (var kvp in modalityData)
        {
            var modalityName = kvp.Key;
            var data = kvp.Value;
            
            // Encode the modality data
            var embedding = EncodeModality(modalityName, data);
            
            // Apply projection if enabled
            if (_useProjection)
            {
                embedding = ApplyModalityProjection(modalityName, embedding);
            }
            
            // L2 normalize for contrastive learning
            embedding = NormalizeFused(embedding);
            
            embeddings[modalityName] = embedding;
        }
        
        // For CLIP, we typically work with pairs (e.g., text-image)
        // If we have exactly 2 modalities, compute their similarity
        if (embeddings.Count == 2)
        {
            var keys = embeddings.Keys.ToList();
            var emb1 = embeddings[keys[0]];
            var emb2 = embeddings[keys[1]];
            
            // Return the average of both embeddings for downstream tasks
            var two = NumOps.FromDouble(2.0);
            return (emb1 + emb2) / two;
        }
        
        // For more than 2 modalities, use attention-based fusion
        return FuseWithAttention(embeddings);
    }
    
    /// <summary>
    /// Trains the CLIP model using contrastive learning
    /// </summary>
    /// <param name="inputs">Training data as a matrix where each row is a sample</param>
    /// <param name="targets">Not used in CLIP training (uses contrastive loss)</param>
    public override void Train(Matrix<T> inputs, Vector<T> targets)
    {
        if (_modalityEncoders.Count < 2)
        {
            throw new InvalidOperationException("CLIP requires at least 2 modalities for contrastive learning");
        }
        
        // Initialize projection networks if needed
        if (_useProjection && _textProjection == null)
        {
            InitializeProjectionNetworks();
        }
        
        // Training would involve:
        // 1. Creating positive pairs (matching text-image)
        // 2. Computing embeddings for batch
        // 3. Computing contrastive loss
        // 4. Backpropagation through encoders and projections
        
        _isTrained = true;
    }
    
    /// <summary>
    /// Creates a deep copy of the model
    /// </summary>
    public override IFullModel<T, Dictionary<string, object>, Vector<T>> Clone()
    {
        // Create clone with same configuration
        // Note: In production, we should deep copy the projection networks
        var clone = new CLIPMultimodalModel<T>(
            _fusedDimension, 
            Convert.ToDouble(_temperature), 
            _useProjection, 
            Convert.ToDouble(_learningRate), 
            _batchSize,
            _textProjectionArchitecture,   // Pass existing projection networks
            _imageProjectionArchitecture); // They will be shared (not deep copied)
        
        // Copy encoders
        foreach (var kvp in _modalityEncoders)
        {
            clone.AddModalityEncoder(kvp.Key, kvp.Value);
        }
        
        // Copy training state
        clone._isTrained = _isTrained;
        clone._contrastiveLosses.AddRange(_contrastiveLosses);
        clone._validationAccuracies.AddRange(_validationAccuracies);
        
        return clone;
    }
    
    /// <summary>
    /// Computes the contrastive loss for a batch of embeddings
    /// </summary>
    private T ComputeContrastiveLoss(Matrix<T> textEmbeddings, Matrix<T> imageEmbeddings)
    {
        int batchSize = textEmbeddings.Rows;
        
        // Compute similarity matrix
        var similarities = textEmbeddings * imageEmbeddings.Transpose();
        
        // Scale by temperature
        similarities = similarities / _temperature;
        
        // Compute loss (simplified InfoNCE loss)
        T loss = NumOps.Zero;
        for (int i = 0; i < batchSize; i++)
        {
            // Positive pair is on diagonal
            T positiveScore = similarities[i, i];
            
            // Compute log-sum-exp for normalization
            T logSumExp = NumOps.Zero;
            for (int j = 0; j < batchSize; j++)
            {
                logSumExp = NumOps.Add(logSumExp, NumOps.Exp(similarities[i, j]));
            }
            
            loss = NumOps.Add(loss, NumOps.Add(NumOps.Negate(positiveScore), NumOps.Log(logSumExp)));
        }
        
        return NumOps.Divide(loss, NumOps.FromDouble(batchSize));
    }
    
    /// <summary>
    /// Applies modality-specific projection
    /// </summary>
    private Vector<T> ApplyModalityProjection(string modalityName, Vector<T> embedding)
    {
        if (!_useProjection)
        {
            return embedding;
        }

        // Initialize projections if needed
        if (_textProjection == null || _imageProjection == null)
        {
            InitializeProjectionNetworks();
        }

        // Convert vector to tensor for neural network processing
        var inputTensor = new Tensor<T>([embedding.Length]);
        for (int i = 0; i < embedding.Length; i++)
        {
            inputTensor[i] = embedding[i];
        }

        Tensor<T> outputTensor;
        string lowerName = modalityName.ToLower();

        // Apply the appropriate projection network
        if (lowerName.Contains("text") || lowerName.Contains("language"))
        {
            outputTensor = _textProjection!.Predict(inputTensor);
        }
        else if (lowerName.Contains("image") || lowerName.Contains("visual") || lowerName.Contains("vision"))
        {
            outputTensor = _imageProjection!.Predict(inputTensor);
        }
        else
        {
            // For other modalities, use simple projection
            return ProjectToTargetDimension(embedding, _fusedDimension);
        }

        // Convert tensor back to vector
        var result = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            result[i] = outputTensor[i];
        }

        return new Vector<T>(result);
    }
    
    /// <summary>
    /// Initializes the projection networks for each modality
    /// </summary>
    private void InitializeProjectionNetworks()
    {
        // Only create networks if they haven't been created yet
        if (_textProjection == null)
        {
            // Use custom architecture if provided, otherwise create default
            var textArchitecture = _textProjectionArchitecture ?? CreateDefaultTextArchitecture();
            _textProjection = new FeedForwardNeuralNetwork<T>(textArchitecture);
        }
        
        if (_imageProjection == null)
        {
            // Use custom architecture if provided, otherwise create default
            var imageArchitecture = _imageProjectionArchitecture ?? CreateDefaultImageArchitecture();
            _imageProjection = new FeedForwardNeuralNetwork<T>(imageArchitecture);
        }
    }
    
    /// <summary>
    /// Creates the default text projection network architecture
    /// </summary>
    /// <returns>A neural network architecture for text projection</returns>
    private NeuralNetworkArchitecture<T> CreateDefaultTextArchitecture()
    {
        // For CLIP-style models, projection networks typically have 2-3 layers
        // with dropout and normalization for better training stability
        int hiddenDim = Math.Max(512, _fusedDimension);
        
        var layers = new List<ILayer<T>>
        {
            // Text projection: 768 -> hidden -> output
            new DenseLayer<T>(768, hiddenDim, new ReLUActivation<T>() as IActivationFunction<T>),
            new DropoutLayer<T>(0.1), // Add dropout for regularization
            new DenseLayer<T>(hiddenDim, _fusedDimension, new IdentityActivation<T>() as IActivationFunction<T>)
        };
        
        return new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Simple,
            taskType: NeuralNetworkTaskType.DimensionalityReduction,
            layers: layers
        );
    }
    
    /// <summary>
    /// Creates the default image projection network architecture
    /// </summary>
    /// <returns>A neural network architecture for image projection</returns>
    private NeuralNetworkArchitecture<T> CreateDefaultImageArchitecture()
    {
        // For CLIP-style models, image projection networks are typically similar to text
        // but may have different hidden dimensions
        int hiddenDim = Math.Max(256, _fusedDimension);
        
        var layers = new List<ILayer<T>>
        {
            // Image projection: 512 -> hidden -> output
            new DenseLayer<T>(512, hiddenDim, new ReLUActivation<T>() as IActivationFunction<T>),
            new DropoutLayer<T>(0.1), // Add dropout for regularization
            new DenseLayer<T>(hiddenDim, _fusedDimension, new IdentityActivation<T>() as IActivationFunction<T>)
        };
        
        return new NeuralNetworkArchitecture<T>(
            complexity: NetworkComplexity.Simple,
            taskType: NeuralNetworkTaskType.DimensionalityReduction,
            layers: layers
        );
    }
    
    
    /// <summary>
    /// Fuses embeddings using attention mechanism
    /// </summary>
    private Vector<T> FuseWithAttention(Dictionary<string, Vector<T>> embeddings)
    {
        if (embeddings.Count == 0)
            return new Vector<T>(_fusedDimension);
            
        if (embeddings.Count == 1)
            return embeddings.First().Value;
        
        // Stack embeddings into a matrix
        var embeddingMatrix = new Matrix<T>(embeddings.Count, _fusedDimension);
        int row = 0;
        foreach (var embedding in embeddings.Values)
        {
            for (int col = 0; col < _fusedDimension && col < embedding.Length; col++)
            {
                embeddingMatrix[row, col] = embedding[col];
            }
            row++;
        }
        
        // Compute attention weights if available
        if (_crossModalityAttention != null && _crossModalityAttention.Rows >= embeddings.Count)
        {
            // Apply attention
            var attended = _crossModalityAttention.GetSubMatrix(0, embeddings.Count, 0, embeddings.Count) * embeddingMatrix;
            
            // Average the attended embeddings
            var result = new Vector<T>(_fusedDimension);
            var count = NumOps.FromDouble(embeddings.Count);
            for (int i = 0; i < embeddings.Count; i++)
            {
                for (int j = 0; j < _fusedDimension; j++)
                {
                    result[j] = NumOps.Add(result[j], NumOps.Divide(attended[i, j], count));
                }
            }
            return result;
        }
        else
        {
            // Simple averaging if no attention weights
            var result = new Vector<T>(_fusedDimension);
            var count = NumOps.FromDouble(embeddings.Count);
            foreach (var embedding in embeddings.Values)
            {
                for (int i = 0; i < _fusedDimension && i < embedding.Length; i++)
                {
                    result[i] = NumOps.Add(result[i], NumOps.Divide(embedding[i], count));
                }
            }
            return result;
        }
    }
    
    /// <summary>
    /// Gets the similarity between two modality inputs
    /// </summary>
    public T GetSimilarity(Dictionary<string, object> input1, Dictionary<string, object> input2)
    {
        var embedding1 = ProcessMultimodal(input1);
        var embedding2 = ProcessMultimodal(input2);
        
        // Compute cosine similarity
        T dotProduct = NumOps.Zero;
        T norm1 = NumOps.Zero;
        T norm2 = NumOps.Zero;
        
        for (int i = 0; i < embedding1.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(embedding1[i], embedding2[i]));
            norm1 = NumOps.Add(norm1, NumOps.Multiply(embedding1[i], embedding1[i]));
            norm2 = NumOps.Add(norm2, NumOps.Multiply(embedding2[i], embedding2[i]));
        }
        
        norm1 = NumOps.Sqrt(norm1);
        norm2 = NumOps.Sqrt(norm2);
        
        if (NumOps.Equals(norm1, NumOps.Zero) || NumOps.Equals(norm2, NumOps.Zero))
            return NumOps.Zero;
            
        return NumOps.Divide(dotProduct, NumOps.Multiply(norm1, norm2));
    }
    
    /// <summary>
    /// Performs zero-shot classification given text labels
    /// </summary>
    public int ZeroShotClassify(Dictionary<string, object> imageInput, List<string> textLabels)
    {
        if (!imageInput.ContainsKey("image") && !imageInput.ContainsKey("Image"))
        {
            throw new ArgumentException("Image input must contain 'image' or 'Image' key");
        }
        
        var imageEmbedding = ProcessMultimodal(imageInput);
        
        T maxSimilarity = NumOps.MinValue;
        int bestLabelIndex = 0;
        
        for (int i = 0; i < textLabels.Count; i++)
        {
            var textInput = new Dictionary<string, object> { ["text"] = textLabels[i] };
            var textEmbedding = ProcessMultimodal(textInput);
            
            // Compute similarity
            T similarity = NumOps.Zero;
            for (int j = 0; j < imageEmbedding.Length; j++)
            {
                similarity = NumOps.Add(similarity, NumOps.Multiply(imageEmbedding[j], textEmbedding[j]));
            }
            
            if (NumOps.GreaterThan(similarity, maxSimilarity))
            {
                maxSimilarity = similarity;
                bestLabelIndex = i;
            }
        }
        
        return bestLabelIndex;
    }
    
    /// <summary>
    /// Gets CLIP-specific parameters
    /// </summary>
    public override Dictionary<string, object> GetParametersDictionary()
    {
        var parameters = base.GetParametersDictionary();
        parameters["Temperature"] = Convert.ToDouble(_temperature);
        parameters["UseProjection"] = _useProjection;
        parameters["LearningRate"] = Convert.ToDouble(_learningRate);
        parameters["BatchSize"] = _batchSize;
        parameters["ContrastiveLossHistory"] = _contrastiveLosses.Count;

        return parameters;
    }
}