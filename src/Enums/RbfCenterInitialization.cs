namespace AiDotNet.Enums;

/// <summary>
/// Where <see cref="AiDotNet.NeuralNetworks.Layers.RBFLayer{T}"/> puts its centres before training.
/// </summary>
/// <remarks>
/// <para>
/// An RBF network's centres are points in the input space (Broomhead and Lowe 1988; Moody and Darken 1989). Placed
/// at random near the origin they can sit far from every input, and with Gaussian units that leaves every unit at
/// exp(-large) = 0 for every input. The papers place them on the training data, which is the default here.
/// </para>
/// <para><b>For Beginners:</b> Each RBF unit reacts to inputs close to its centre. Putting the centres on real
/// examples guarantees every unit starts out reacting to something.</para>
/// </remarks>
public enum RbfCenterInitialization
{
    /// <summary>
    /// The first batch the layer sees supplies the centres: evenly spaced rows of that batch, or one centre per row
    /// when it has fewer rows than centres (the others keep their random initialization), after which the widths
    /// are set from the spread of the centres. Deterministic, so a
    /// clone placed from the same batch gets the same centres. The layer then switches to
    /// <see cref="AsInitialized"/>, which is what a clone or a saved model carries.
    /// </summary>
    FromFirstBatch,

    /// <summary>
    /// Use the centres as they are: the random initialization, values set through SetParameters or a loaded model,
    /// or centres already placed from data.
    /// </summary>
    AsInitialized,
}