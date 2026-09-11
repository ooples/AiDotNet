namespace AiDotNet.Enums;

/// <summary>
/// Identifies the optimizer a research paper specifies for a model.
/// </summary>
/// <remarks>
/// <para>
/// Used by <see cref="AiDotNet.Attributes.PaperOptimizerAttribute"/> so a model can declare which
/// optimizer its paper trains with, alongside the hyperparameters that paper states.
/// </para>
/// <para><b>For Beginners:</b> Research papers do not just pick a learning rate — they pick an
/// optimizer too, and the two go together. A learning rate that is right for SGD is usually wrong
/// for Adam. Recording which optimizer the paper used is what makes its other numbers meaningful.
/// </para>
/// <para>
/// An enum rather than a string so a typo cannot silently produce a model that matches no
/// optimizer and quietly keeps the library default.
/// </para>
/// </remarks>
public enum OptimizerKind
{
    /// <summary>The paper does not state an optimizer, or it has not been recorded yet.</summary>
    /// <remarks>
    /// The default. A declaration left at <see cref="Unspecified"/> contributes no hyperparameters,
    /// so the model keeps the library defaults rather than inheriting a guess.
    /// </remarks>
    Unspecified = 0,

    /// <summary>Stochastic gradient descent.</summary>
    Sgd,

    /// <summary>SGD with momentum.</summary>
    SgdMomentum,

    /// <summary>Adam (Kingma and Ba, 2015).</summary>
    Adam,

    /// <summary>AdamW, Adam with decoupled weight decay (Loshchilov and Hutter, 2019).</summary>
    AdamW,

    /// <summary>Adam with 8-bit optimizer state.</summary>
    Adam8Bit,

    /// <summary>RMSProp.</summary>
    RmsProp,

    /// <summary>Adagrad.</summary>
    Adagrad,

    /// <summary>Adadelta.</summary>
    Adadelta,

    /// <summary>Adamax.</summary>
    Adamax,

    /// <summary>Nadam, Adam with Nesterov momentum.</summary>
    Nadam,

    /// <summary>LAMB, layer-wise adaptive moments for large-batch training.</summary>
    Lamb,

    /// <summary>Lion (Chen et al., 2023).</summary>
    Lion,

    /// <summary>L-BFGS, limited-memory quasi-Newton.</summary>
    LBfgs,
}
