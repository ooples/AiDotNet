namespace AiDotNet.Enums;

/// <summary>
/// The functions an <see cref="AiDotNet.NeuralNetworks.Layers.ObliviousDecisionTreeLayer{T}"/> uses to
/// choose a feature per level and to make each soft split decision.
/// </summary>
/// <remarks>
/// <para>
/// NODE (Popov et al. 2019) chooses features with 1.5-entmax and splits with 1.5-entmoid. Both are
/// sparse: a level can put all its weight on one feature, and a sample far enough from a threshold
/// goes entirely to one side. Softmax and sigmoid never reach exactly zero or one.
/// </para>
/// <para><b>For Beginners:</b> keep the default to get the tree described in the NODE paper. Choose
/// <see cref="SoftmaxSigmoid"/> only to reproduce models trained with the earlier smooth variant.
/// </para>
/// </remarks>
public enum ObliviousTreeChoiceFunction
{
    /// <summary>1.5-entmax feature selection and 1.5-entmoid splits, as in NODE (Popov et al. 2019).</summary>
    Entmax15 = 0,

    /// <summary>Softmax feature selection and sigmoid splits, the layer's earlier behaviour.</summary>
    SoftmaxSigmoid = 1,
}
