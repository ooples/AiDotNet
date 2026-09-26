using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the DocGCN document model.
/// </summary>
public class DocGCNOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DocGCNOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public DocGCNOptions()
    {
        NodeDim = 256;
        EdgeDim = 64;
        GcnLayers = 3;
        NumClasses = 9;
        MaxNodes = 512;
    }


    /// <summary>
    /// Gets or sets the node dim.
    /// </summary>
    public int NodeDim { get; set; }

    /// <summary>
    /// Gets or sets the edge dim.
    /// </summary>
    public int EdgeDim { get; set; }

    /// <summary>
    /// Gets or sets the gcn layers.
    /// </summary>
    public int GcnLayers { get; set; }

    /// <summary>
    /// Gets or sets the max nodes.
    /// </summary>
    public int MaxNodes { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
