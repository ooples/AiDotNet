using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Modules;

/// <summary>
/// Relation module that computes similarity between feature pairs for Relation Networks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The relation module is the core component that makes Relation Networks unique.
/// It takes concatenated features from two examples and outputs a scalar relation score
/// indicating how similar/related they are.
/// </para>
/// <para><b>For Beginners:</b> Instead of using a fixed formula to measure similarity
/// (like Euclidean distance), the relation module is a small neural network that LEARNS
/// how to compare examples. It takes two feature vectors as input (concatenated together)
/// and outputs a number between 0 and 1 indicating how related they are.
/// </para>
/// </remarks>
public class RelationModule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _hiddenDimension;
    private Vector<T> _weights;
    private bool _isTraining;

    /// <summary>
    /// Initializes a new instance of RelationModule.
    /// </summary>
    /// <param name="hiddenDimension">The hidden layer dimension.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The hidden dimension controls how complex the
    /// relation function can be. Larger values allow more complex comparisons
    /// but require more data to train effectively.
    /// </para>
    /// </remarks>
    public RelationModule(int hiddenDimension)
    {
        _hiddenDimension = hiddenDimension;
        _weights = new Vector<T>(hiddenDimension);
        _isTraining = false;
    }

    /// <summary>
    /// Performs forward pass through the relation module.
    /// </summary>
    /// <param name="combinedFeatures">Combined feature tensor of two examples.</param>
    /// <returns>Relation score tensor (scalar value between 0 and 1).</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes a relation score by:
    /// 1. Computing a weighted sum of the combined features
    /// 2. Applying a sigmoid activation to produce a score in [0, 1]
    /// </para>
    /// <para><b>For Beginners:</b> This takes the concatenated features of two examples
    /// and produces a number indicating how related they are. A value close to 1 means
    /// very related (likely same class), while close to 0 means unrelated.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> combinedFeatures)
    {
        // Simplified: compute dot product of features with weights
        int inputSize = 1;
        for (int i = 0; i < combinedFeatures.Shape.Length; i++)
        {
            inputSize *= combinedFeatures.Shape[i];
        }

        T score = NumOps.Zero;
        int weightSize = Math.Min(inputSize, _weights.Length);

        for (int i = 0; i < weightSize; i++)
        {
            score = NumOps.Add(score, NumOps.Multiply(combinedFeatures.GetFlat(i), _weights[i]));
        }

        // Apply sigmoid activation
        double scoreValue = NumOps.ToDouble(score);
        double sigmoidScore = 1.0 / (1.0 + Math.Exp(-scoreValue));

        var output = new Tensor<T>(new int[] { 1 });
        output[0] = NumOps.FromDouble(sigmoidScore);

        return output;
    }

    /// <summary>
    /// Gets the learnable parameters of the relation module.
    /// </summary>
    /// <returns>Vector of learnable weights.</returns>
    public Vector<T> GetParameters()
    {
        return _weights;
    }

    /// <summary>
    /// Sets the training mode.
    /// </summary>
    /// <param name="isTraining">True for training mode, false for inference mode.</param>
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    /// <summary>
    /// Creates a deep copy of the relation module.
    /// </summary>
    /// <returns>A new RelationModule with copied weights.</returns>
    public RelationModule<T> Clone()
    {
        var cloned = new RelationModule<T>(_hiddenDimension);
        for (int i = 0; i < _weights.Length; i++)
        {
            cloned._weights[i] = _weights[i];
        }
        return cloned;
    }
}
