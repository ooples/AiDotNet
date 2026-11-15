namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabR, a retrieval-augmented model for tabular data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabR combines neural networks with instance-based learning by retrieving similar
/// training examples and using their information to help make predictions. It encodes
/// both the query sample and retrieved neighbors, then aggregates the information
/// using attention.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabR is like a student who looks at similar past problems
/// to help solve a new one.
///
/// How it works:
/// 1. **Encode**: Convert features to a learned representation
/// 2. **Retrieve**: Find the K most similar training samples
/// 3. **Attend**: Use attention to aggregate information from neighbors
/// 4. **Predict**: Make a prediction using both query and neighbor information
///
/// Why this works well for tabular data:
/// - Tabular data often has local structure (similar inputs → similar outputs)
/// - Retrieval adds "memory" without needing to memorize in network weights
/// - Combines strengths of neural networks (feature learning) and k-NN (locality)
/// - Naturally handles rare patterns by finding similar examples
///
/// Example usage:
/// <code>
/// var options = new TabROptions&lt;double&gt;
/// {
///     NumNeighbors = 96,
///     EmbeddingDimension = 265,
///     EncoderNumBlocks = 0,
///     PredictorNumBlocks = 1
/// };
/// var model = new TabRClassifier&lt;double&gt;(numFeatures: 10, numClasses: 3, options);
/// </code>
/// </para>
/// <para>
/// Reference: "TabR: Tabular Deep Learning Meets Nearest Neighbors" (2023)
/// </para>
/// </remarks>
public class TabROptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the number of nearest neighbors to retrieve.
    /// </summary>
    /// <value>The number of neighbors, defaulting to 96.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many similar training examples to look at:
    /// - Fewer neighbors (32-64): Faster, but might miss relevant examples
    /// - More neighbors (96-128): More context, but slower and might include less relevant samples
    ///
    /// The default of 96 works well for most datasets. Increase for larger datasets.
    /// </para>
    /// </remarks>
    public int NumNeighbors { get; set; } = 96;

    /// <summary>
    /// Gets or sets the width of every hidden representation: the reference implementation's <c>d_main</c>.
    /// </summary>
    /// <value>265 by default, the paper's default TabR-S configuration.</value>
    /// <remarks>
    /// <para>
    /// The encoder's first linear map, every block's input and output, the keys, the values and the prediction
    /// head all have this width. Gorishniy et al. (2023) list <c>d = 265</c> among the default TabR-S
    /// hyperparameters. Must be positive.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls the size of the learned representation:
    /// - Smaller (128-192): Faster, good for smaller datasets
    /// - Larger (256-384): More capacity, better for complex relationships
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 265;

    /// <summary>
    /// Gets or sets the hidden width of every block and of the value MLP <c>T</c> as a multiple of
    /// <see cref="EmbeddingDimension"/>: the reference implementation's <c>d_multiplier</c>.
    /// </summary>
    /// <value>2.0 by default: the paper's first linear layer of the FFN and of T has output size 2d.</value>
    /// <remarks>
    /// The reference sets <c>d_block = int(d_main * d_multiplier)</c>, truncating, and each block maps
    /// <c>d_main -&gt; d_block -&gt; d_main</c>. Must be positive and give a block width of at least 1.
    /// </remarks>
    public double BlockWidthMultiplier { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the number of residual blocks after the encoder's first linear map: the reference
    /// implementation's <c>encoder_n_blocks</c>.
    /// </summary>
    /// <value>0 by default, the paper's TabR-S configuration, whose encoder is the single linear map.</value>
    /// <remarks>
    /// Each block is <c>x + Dropout(dropout1)(Linear(Dropout(dropout0)(Activation(Linear(Norm(x))))))</c>, the norm
    /// skipped on the first encoder block. Must not be negative.
    /// </remarks>
    public int EncoderNumBlocks { get; set; } = 0;

    /// <summary>
    /// Gets or sets the number of residual blocks between the retrieval module and the prediction head: the
    /// reference implementation's <c>predictor_n_blocks</c>.
    /// </summary>
    /// <value>1 by default, the paper's TabR-S configuration.</value>
    /// <remarks>
    /// Every predictor block normalizes its input first. The head after them is
    /// <c>Linear(Activation(Norm(x)))</c>. Must not be negative.
    /// </remarks>
    public int PredictorNumBlocks { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether the encoder output is normalized before the key projection <c>K</c>: the reference
    /// implementation's <c>mixer_normalization</c>.
    /// </summary>
    /// <value>Null by default, the reference's <c>'auto'</c>: on exactly when <see cref="EncoderNumBlocks"/> is positive.</value>
    /// <remarks>
    /// Only the key sees the normalized encoding; the representation the retrieved context is added to stays
    /// unnormalized. The reference asserts that an encoder without blocks has no mixer normalization, so true with
    /// <see cref="EncoderNumBlocks"/> = 0 is rejected.
    /// </remarks>
    public bool? MixerNormalization { get; set; } = null;

    /// <summary>
    /// Gets or sets the normalization layer of the blocks, the mixer and the head: the reference implementation's
    /// <c>normalization</c>.
    /// </summary>
    /// <value><see cref="TabRNormalization.LayerNorm"/> by default, the paper's choice.</value>
    public TabRNormalization Normalization { get; set; } = TabRNormalization.LayerNorm;

    /// <summary>
    /// Gets or sets the activation inside every block, inside the value MLP <c>T</c> and in the head: the reference
    /// implementation's <c>activation</c>.
    /// </summary>
    /// <value><see cref="ActivationFunction.ReLU"/> by default, the paper's choice.</value>
    /// <remarks>
    /// Any element-wise activation; <see cref="ActivationFunction.Softmax"/> is rejected because it is not one.
    /// </remarks>
    public ActivationFunction Activation { get; set; } = ActivationFunction.ReLU;

    /// <summary>
    /// Gets or sets how the numeric features are embedded before the encoder: the reference implementation's
    /// <c>num_embeddings</c>.
    /// </summary>
    /// <value><see cref="TabRNumericEmbedding.None"/> by default, the paper's TabR-S configuration.</value>
    /// <remarks>
    /// With <see cref="TabRNumericEmbedding.PeriodicLinearReLU"/> every feature becomes a
    /// <see cref="NumericEmbeddingDimension"/>-wide vector and the encoder's first linear map reads
    /// <c>features * d_embedding</c> inputs. All features are treated as numeric: one-hot encode categorical features
    /// before training, as the reference's preprocessing does. Needs the feature count at construction.
    /// </remarks>
    public TabRNumericEmbedding NumericEmbedding { get; set; } = TabRNumericEmbedding.None;

    /// <summary>
    /// Gets or sets the number of learned frequencies per feature of the periodic embedding: the reference
    /// <c>n_frequencies</c>.
    /// </summary>
    /// <value>48 by default, the <c>rtdl_num_embeddings</c> default.</value>
    /// <remarks>Read only with <see cref="TabRNumericEmbedding.PeriodicLinearReLU"/>. Must be positive.</remarks>
    public int NumFrequencies { get; set; } = 48;

    /// <summary>
    /// Gets or sets the standard deviation of the initial frequencies of the periodic embedding: the reference
    /// <c>frequency_scale</c> (<c>frequency_init_scale</c> in <c>rtdl_num_embeddings</c>).
    /// </summary>
    /// <value>0.01 by default, the <c>rtdl_num_embeddings</c> default.</value>
    /// <remarks>
    /// The frequencies start as <c>N(0, scale^2)</c> draws. The scale matters: TabR tunes it log-uniformly in
    /// [0.01, 100]. Read only with <see cref="TabRNumericEmbedding.PeriodicLinearReLU"/>. Must be positive.
    /// </remarks>
    public double FrequencyInitScale { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the width of each feature's embedding: the reference <c>d_embedding</c>.
    /// </summary>
    /// <value>24 by default, the <c>rtdl_num_embeddings</c> default.</value>
    /// <remarks>Read only with <see cref="TabRNumericEmbedding.PeriodicLinearReLU"/>. Must be positive.</remarks>
    public int NumericEmbeddingDimension { get; set; } = 24;

    /// <summary>
    /// Gets or sets whether the periodic embedding's linear map is shared by every feature: the reference <c>lite</c>.
    /// </summary>
    /// <value>True by default: every TabR configuration with PLR embeddings uses <c>lite = true</c>.</value>
    /// <remarks>
    /// False gives the original PLR embedding, one <c>Linear(2 n_frequencies, d_embedding)</c> per feature. Read only
    /// with <see cref="TabRNumericEmbedding.PeriodicLinearReLU"/>.
    /// </remarks>
    public bool LiteNumericEmbedding { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of hidden layers of <see cref="AiDotNet.NeuralNetworks.Tabular.TabRNetwork{T}"/>'s
    /// retrieval-free layer stack.
    /// </summary>
    /// <value>The number of layers, defaulting to 4.</value>
    /// <remarks>
    /// Only the layer-stack network reads this. The retrieval models built on
    /// <see cref="AiDotNet.NeuralNetworks.Tabular.TabRBase{T}"/> follow the reference architecture and take their depth
    /// from <see cref="EncoderNumBlocks"/> and <see cref="PredictorNumBlocks"/>.
    /// </remarks>
    public int NumLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the dropout rate after every hidden activation: the reference implementation's <c>dropout0</c>.
    /// </summary>
    /// <value>0.38852797479169876 by default, the paper's default TabR-S configuration.</value>
    /// <remarks>
    /// The reference <c>make_block</c> is <c>Linear -&gt; ReLU -&gt; Dropout(dropout0) -&gt; Linear -&gt; Dropout(dropout1)</c>,
    /// and the value MLP is <c>T = LinearWithoutBias(Dropout(dropout0)(ReLU(Linear(.))))</c>. This rate follows the
    /// activation of every encoder and predictor block and of T; <see cref="BlockOutputDropoutRate"/> is
    /// <c>dropout1</c>. It drops only inside training steps; prediction never drops. Must lie in [0, 1).
    /// </remarks>
    public double DropoutRate { get; set; } = 0.38852797479169876;

    /// <summary>
    /// Gets or sets the dropout rate on the output of every encoder and predictor block, before the residual add: the
    /// reference implementation's <c>dropout1</c>.
    /// </summary>
    /// <value>0 by default, the paper's TabR-S configuration.</value>
    /// <remarks>
    /// Drawn through the same training-step masks as <see cref="DropoutRate"/>: it drops only inside training steps,
    /// and prediction never drops. The reference's <c>dropout1 = 'dropout0'</c> is this set equal to
    /// <see cref="DropoutRate"/>. Must lie in [0, 1).
    /// </remarks>
    public double BlockOutputDropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the dropout rate on the retrieval weights: the reference implementation's <c>context_dropout</c>.
    /// </summary>
    /// <value>0.38920071545944357 by default, the paper's default TabR-S configuration.</value>
    /// <remarks>
    /// The reference drops the softmax weights over the retrieved neighbours (<c>probs = self.dropout(probs)</c>) before
    /// they weight the values. It drops only inside training steps. Must lie in [0, 1).
    /// </remarks>
    public double ContextDropoutRate { get; set; } = 0.38920071545944357;

    /// <summary>
    /// Gets or sets whether the last linear layer of the value MLP <c>T</c> carries a bias.
    /// </summary>
    /// <value><see cref="BiasMode.Never"/> by default, the paper's bias-free output layer.</value>
    /// <remarks>
    /// Gorishniy et al. (2023) define <c>T(.) = LinearWithoutBias(Dropout(ReLU(Linear(.))))</c>, and the reference
    /// <c>yandex-research/tabular-dl-tabr</c> builds it with <c>nn.Linear(d_block, d_main, bias=False)</c>: a bias
    /// there adds one constant to every neighbour's value, which the attention weights (summing to 1) pass straight
    /// into the context. Set <see cref="BiasMode.Always"/> to keep one anyway.
    /// </remarks>
    public BiasMode ValueOutputBiasMode { get; set; } = BiasMode.Never;

    /// <summary>
    /// Gets or sets whether to include target values of retrieved neighbors.
    /// </summary>
    /// <value>True to include neighbor targets; false otherwise. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When true, the model can directly see what the outputs
    /// were for similar training samples. This helps especially when:
    /// - The relationship between features and target is complex
    /// - There are rare patterns that benefit from direct memory
    ///
    /// During training, leave-one-out ensures we don't use the sample's own target.
    /// </para>
    /// </remarks>
    public bool IncludeNeighborTargets { get; set; } = true;

    /// <summary>
    /// Gets or sets the temperature for retrieval softmax.
    /// </summary>
    /// <value>The temperature, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls how "sharp" the attention to neighbors is:
    /// - Lower temperature (&lt;1): Focus more on the most similar neighbors
    /// - Higher temperature (&gt;1): Spread attention more evenly across neighbors
    /// </para>
    /// </remarks>
    public double RetrievalTemperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to normalize embeddings for retrieval.
    /// </summary>
    /// <value>True to normalize embeddings; false otherwise. Defaults to false.</value>
    /// <remarks>
    /// <para>
    /// Gorishniy et al. (2023) retrieve by the squared L2 distance between unnormalized keys, so the default is
    /// off. Turning it on scales every encoding to unit length before retrieval.
    /// </para>
    /// </remarks>
    public bool NormalizeEmbeddings { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight decay coefficient.
    /// </summary>
    /// <value>The weight decay, defaulting to 1e-5.</value>
    public double WeightDecay { get; set; } = 1e-5;

    /// <summary>
    /// Creates a copy of the options.
    /// </summary>
    public TabROptions<T> Clone()
    {
        return new TabROptions<T>
        {
            NumNeighbors = NumNeighbors,
            EmbeddingDimension = EmbeddingDimension,
            BlockWidthMultiplier = BlockWidthMultiplier,
            EncoderNumBlocks = EncoderNumBlocks,
            PredictorNumBlocks = PredictorNumBlocks,
            MixerNormalization = MixerNormalization,
            Normalization = Normalization,
            Activation = Activation,
            NumericEmbedding = NumericEmbedding,
            NumFrequencies = NumFrequencies,
            FrequencyInitScale = FrequencyInitScale,
            NumericEmbeddingDimension = NumericEmbeddingDimension,
            LiteNumericEmbedding = LiteNumericEmbedding,
            NumLayers = NumLayers,
            DropoutRate = DropoutRate,
            BlockOutputDropoutRate = BlockOutputDropoutRate,
            ContextDropoutRate = ContextDropoutRate,
            ValueOutputBiasMode = ValueOutputBiasMode,
            IncludeNeighborTargets = IncludeNeighborTargets,
            RetrievalTemperature = RetrievalTemperature,
            NormalizeEmbeddings = NormalizeEmbeddings,
            WeightDecay = WeightDecay,
            MaxGradNorm = MaxGradNorm
        };
    }
}
