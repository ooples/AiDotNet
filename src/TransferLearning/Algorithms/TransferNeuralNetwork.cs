using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Helpers;
using AiDotNet.TransferLearning.FeatureMapping;

namespace AiDotNet.TransferLearning.Algorithms;

/// <summary>
/// Implements transfer learning for Neural Network models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class enables neural networks to transfer knowledge from one
/// domain to another, even when the feature spaces are different. It uses techniques like
/// adapter layers and knowledge distillation to make this possible.
/// </para>
/// </remarks>
public class TransferNeuralNetwork<T> : TransferLearningBase<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Transfers a Neural Network model to a target domain with the same feature space.
    /// </summary>
    /// <remarks>
    /// NOTE: Domain adaptation without source data is not meaningful. This method skips
    /// domain adaptation and only performs fine-tuning. For proper domain adaptation,
    /// use the public Transfer() method that accepts source data.
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> TransferSameDomain(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        // Clone the source model to create a target model
        var targetModel = sourceModel.DeepCopy();

        // Fine-tune on target domain using batch training
        // Note: Domain adaptation is skipped here due to lack of source data
        // In a full implementation, this would use a reduced learning rate
        // and possibly freeze early layers
        targetModel.Train(targetData, targetLabels);

        return targetModel;
    }

    /// <summary>
    /// Transfers a Neural Network model to a target domain with a different feature space.
    /// </summary>
    /// <remarks>
    /// NOTE: This implementation requires source domain data to properly train the feature mapper.
    /// The current API limitations prevent passing source data, so this method will throw
    /// InvalidOperationException. Users should use the public Transfer() method that accepts source data.
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> TransferCrossDomain(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        throw new InvalidOperationException(
            "Cross-domain transfer cannot be performed directly through this protected method due to the need for source domain data. " +
            "Please use the public 'Transfer(sourceModel, sourceData, targetData, targetLabels)' method which accepts both source and target domain data for feature mapping and transfer.");
    }

    /// <summary>
    /// Transfers a Neural Network model to a target domain with proper source data.
    /// </summary>
    /// <param name="sourceModel">The model trained on the source domain.</param>
    /// <param name="sourceData">Training data from the source domain (required for cross-domain transfer).</param>
    /// <param name="targetData">Training data from the target domain.</param>
    /// <param name="targetLabels">Labels for the target domain data.</param>
    /// <returns>A new model adapted to the target domain.</returns>
    public IFullModel<T, Matrix<T>, Vector<T>> Transfer(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> sourceData,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        // Determine if cross-domain transfer is needed
        bool needsCrossDomain = RequiresCrossDomainTransfer(sourceModel, targetData);

        if (!needsCrossDomain)
        {
            return TransferSameDomain(sourceModel, targetData, targetLabels);
        }

        // Cross-domain transfer with proper source data
        if (FeatureMapper == null)
        {
            throw new InvalidOperationException(
                "Cross-domain transfer requires a feature mapper. Use SetFeatureMapper() before transfer.");
        }

        // Step 1: Train feature mapper with actual source and target data
        if (!FeatureMapper.IsTrained)
        {
            FeatureMapper.Train(sourceData, targetData);
        }

        // Step 2: Get dimensions
        int sourceFeatures = sourceModel.GetActiveFeatureIndices().Count();

        // Step 3: Map target data to source feature space
        Matrix<T> mappedTargetData = FeatureMapper.MapToSource(targetData, sourceFeatures);

        // Step 4: Use source model for predictions (knowledge distillation)
        Vector<T> softLabels = sourceModel.Predict(mappedTargetData);

        // Step 5: Combine soft labels with true labels
        // trueWeight of 0.7 means combinedLabels = 0.7 * trueLabels + 0.3 * softLabels (favors true labels for more accurate target domain learning)
        Vector<T> combinedLabels = CombineLabels(softLabels, targetLabels, 0.7);

        // Step 6: Create and train a new model on the target domain
        // Use original targetData (not mapped) since the model should learn in target feature space
        // The mapping was only needed to get predictions from source model for knowledge distillation
        var targetModel = sourceModel.DeepCopy();
        targetModel.Train(targetData, combinedLabels);

        return targetModel;
    }

    /// <summary>
    /// Combines soft labels from source model with true target labels.
    /// </summary>
    private Vector<T> CombineLabels(Vector<T> softLabels, Vector<T> trueLabels, double trueWeight)
    {
        var combined = new Vector<T>(softLabels.Length);
        T trueW = NumOps.FromDouble(trueWeight);
        T softW = NumOps.FromDouble(1.0 - trueWeight);

        for (int i = 0; i < combined.Length; i++)
        {
            combined[i] = NumOps.Add(
                NumOps.Multiply(trueW, trueLabels[i]),
                NumOps.Multiply(softW, softLabels[i]));
        }

        return combined;
    }
}
