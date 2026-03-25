namespace AiDotNet.Enums;

/// <summary>
/// Tasks or architectural roles that a neural network layer performs.
/// </summary>
public enum LayerTask
{
    /// <summary>Modeling sequential/temporal dependencies (RNN, LSTM, SSM, Transformer).</summary>
    SequenceModeling,
    /// <summary>Extracting features from input data (Conv, attention, embedding).</summary>
    FeatureExtraction,
    /// <summary>Preventing overfitting (Dropout, BatchNorm, weight decay).</summary>
    Regularization,
    /// <summary>Processing 2D spatial data (Conv2D, pooling, spatial attention).</summary>
    SpatialProcessing,
    /// <summary>Processing temporal/time-series data (RNN, 1D conv, causal attention).</summary>
    TemporalProcessing,
    /// <summary>Processing graph-structured data (GCN, GAT, message passing).</summary>
    GraphProcessing,
    /// <summary>Computing attention scores and weighted sums.</summary>
    AttentionComputation,
    /// <summary>Reducing spatial/temporal resolution (pooling, strided convolution).</summary>
    DownSampling,
    /// <summary>Increasing spatial/temporal resolution (upsample, transpose convolution).</summary>
    UpSampling,
    /// <summary>Normalizing activations for training stability.</summary>
    ActivationNormalization,
    /// <summary>Encoding position information into representations.</summary>
    PositionalEncoding,
    /// <summary>Combining multiple input streams (concat, add, multiply).</summary>
    FeatureFusion,
    /// <summary>Transforming between representation spaces (linear projection).</summary>
    Projection,
    /// <summary>Routing information between capsules or experts.</summary>
    Routing,
    /// <summary>Processing 3D volumetric data (Conv3D, 3D pooling).</summary>
    VolumetricProcessing,
    /// <summary>Cross-modal attention between different input types.</summary>
    CrossModalAttention
}
