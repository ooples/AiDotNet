using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the TableTransformer document model.
/// </summary>
public class TableTransformerOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageSize, HiddenDim, NumEncoderLayers, NumDecoderLayers and NumHeads are declared by
    /// DocumentNeuralNetworkOptions and were previously left unset, with the values living in
    /// constructor parameters instead. Table Transformer (Smock et al. 2022) is DETR applied to
    /// tables: an 800-pixel page through a 256-wide 6+6 encoder-decoder with 8 attention heads.
    /// </para>
    /// </remarks>
    public TableTransformerOptions()
    {
        ImageSize = 800;
        HiddenDim = 256;
        NumEncoderLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        NumQueries = 100;
        NumTableClasses = 2;
        NumStructureClasses = 7;
    }

    /// <summary>
    /// Gets or sets how many object queries the decoder carries. Default: 100.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DETR-style detectors do not scan the page; they start with a
    /// fixed number of "slots" and each one either claims an object or reports empty. This is how
    /// many slots there are, so it caps how many tables can be found in one page.</para>
    /// </remarks>
    public int NumQueries { get; set; }

    /// <summary>
    /// Gets or sets the number of classes the detection head predicts. Default: 2 — background
    /// and table.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Previously the literal 2 in both constructors, with the meaning only in a trailing
    /// comment. It sizes the detection classifier, so it is part of the model's shape.
    /// </para>
    /// </remarks>
    public int NumTableClasses { get; set; }

    /// <summary>
    /// Gets or sets the number of classes the structure head predicts. Default: 7 — background,
    /// table, column, row, column header, projected row header and spanning cell.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Previously the literal 7 in both constructors. It sizes the structure classifier.
    /// </para>
    /// </remarks>
    public int NumStructureClasses { get; set; }

    /// <summary>
    /// Gets or sets the learning rate used when the model creates its own optimizer.
    /// Default: 1e-4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Table Transformer is a DETR-based detector and DETR fine-tunes at 1e-4 with gradient-norm
    /// clipping, which is well below Adam's generic 1e-3. That value was written directly into
    /// both constructors; it lives here now so a caller can see and change it.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;
}
