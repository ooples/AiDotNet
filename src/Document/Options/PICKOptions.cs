using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the PICK document model (Yu et al., ICPR 2020). The defaults are the reference
/// implementation's configuration (wenwenyu/PICK-pytorch, config.json).
/// </summary>
public class PICKOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PICKOptions"/> class with the reference configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They reproduce the published model:
    /// <list type="bullet">
    /// <item>A 512-wide, 4-head, 3-layer transformer text encoder.</item>
    /// <item>A ResNet-50 image branch pooled 7x7 per text box.</item>
    /// <item>A 2-layer graph learning-convolution network.</item>
    /// <item>A 2-layer bidirectional LSTM with a CRF tagger.</item>
    /// </list>
    /// </para>
    /// </remarks>
    public PICKOptions()
    {
        NumEntityTypes = 14;
        ImageSize = 512;
        MaxSequenceLength = 512;
        HiddenDim = 512;
        NumGcnLayers = 2;
        NumHeads = 4;
        VocabSize = 30522;
        NumEncoderLayers = 3;
        FeedForwardDim = 1024;
        ImageFeatureDim = 512;
        ImageEncoderDepths = new[] { 3, 4, 6, 3 };
        GraphLearningDim = 128;
        GraphEta = 1.0;
        GraphGamma = 1.0;
        LstmHiddenDim = 512;
        LstmLayers = 2;
        GraphLossWeight = 0.01;
    }

    /// <summary>Gets or sets the number of IOB tags the CRF predicts. Default: 14.</summary>
    public int NumEntityTypes { get; set; }

    /// <summary>Gets or sets the graph convolution depth. Reference: 2.</summary>
    public int NumGcnLayers { get; set; }

    /// <summary>Gets or sets the text encoder's feed-forward width. Reference: 1024.</summary>
    public int FeedForwardDim { get; set; }

    /// <summary>Gets or sets the image backbone's output channels before the ROI head. Reference: 512.</summary>
    public int ImageFeatureDim { get; set; }

    /// <summary>
    /// Gets or sets the bottleneck blocks per ResNet stage of the image branch. Reference: ResNet-50, [3, 4, 6, 3];
    /// the reference also offers ResNet-101 [3, 4, 23, 3] and ResNet-152 [3, 8, 36, 3].
    /// </summary>
    public int[] ImageEncoderDepths { get; set; }

    /// <summary>Gets or sets the graph-learning projection width. Reference: 128.</summary>
    public int GraphLearningDim { get; set; }

    /// <summary>Gets or sets eta, the weight of node distance in the graph-learning loss. Reference: 1.</summary>
    public double GraphEta { get; set; }

    /// <summary>Gets or sets gamma, the weight of the adjacency's Frobenius norm in the graph-learning loss. Reference: 1.</summary>
    public double GraphGamma { get; set; }

    /// <summary>Gets or sets the BiLSTM hidden size per direction. Reference: 512.</summary>
    public int LstmHiddenDim { get; set; }

    /// <summary>Gets or sets the BiLSTM depth. Reference: 2.</summary>
    public int LstmLayers { get; set; }

    /// <summary>Gets or sets the graph-learning loss weight added to the CRF loss (gl_loss_lambda). Reference: 0.01.</summary>
    public double GraphLossWeight { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        // This model renders the page as an image, so it needs a size. The family base
        // cannot require this: 15 of the 29 document models work from text and layout
        // coordinates and have no image at all.
        Require(ImageSize, nameof(ImageSize));
        Require(NumEncoderLayers, nameof(NumEncoderLayers));
        Require(FeedForwardDim, nameof(FeedForwardDim));
        Require(ImageFeatureDim, nameof(ImageFeatureDim));
        Require(GraphLearningDim, nameof(GraphLearningDim));
        Require(LstmHiddenDim, nameof(LstmHiddenDim));
        Require(LstmLayers, nameof(LstmLayers));
        Require(NumGcnLayers, nameof(NumGcnLayers));
        if (ImageEncoderDepths is null || ImageEncoderDepths.Length != 4 || ImageEncoderDepths.Any(d => d <= 0))
            throw new ArgumentException("ImageEncoderDepths must list four positive ResNet stage depths.", nameof(ImageEncoderDepths));
        ValidateCore();
    }
}
