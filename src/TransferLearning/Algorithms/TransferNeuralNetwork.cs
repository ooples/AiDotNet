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
    protected override IFullModel<T, Matrix<T>, Vector<T>> TransferSameDomain(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        // Apply domain adaptation if available
        Matrix<T> adaptedData = targetData;
        if (DomainAdapter != null)
        {
            // Train adapter if needed
            if (DomainAdapter.RequiresTraining)
            {
                DomainAdapter.Train(targetData, targetData);
            }
            adaptedData = DomainAdapter.AdaptSource(targetData, targetData);
        }

        // Clone the source model to create a target model
        var targetModel = sourceModel.DeepCopy();

        // Fine-tune on target domain using batch training
        // In a full implementation, this would use a reduced learning rate
        // and possibly freeze early layers
        targetModel.Train(adaptedData, targetLabels);

        return targetModel;
    }

    /// <summary>
    /// Transfers a Neural Network model to a target domain with a different feature space.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> TransferCrossDomain(
        IFullModel<T, Matrix<T>, Vector<T>> sourceModel,
        Matrix<T> targetData,
        Vector<T> targetLabels)
    {
        if (FeatureMapper == null)
        {
            throw new InvalidOperationException(
                "Cross-domain transfer requires a feature mapper. Use SetFeatureMapper() before transfer.");
        }

        // Step 1: Train feature mapper if not already trained
        if (!FeatureMapper.IsTrained)
        {
            FeatureMapper.Train(targetData, targetData);
        }

        // Step 2: Get dimensions
        int sourceFeatures = sourceModel.GetActiveFeatureIndices().Count();
        int targetFeatures = targetData.Columns;

        // Step 3: Map target data to source feature space
        Matrix<T> mappedTargetData = FeatureMapper.MapToSource(targetData, sourceFeatures);

        // Step 4: Use source model for predictions (knowledge distillation)
        Vector<T> softLabels = sourceModel.Predict(mappedTargetData);

        // Step 5: Combine soft labels with true labels
        Vector<T> combinedLabels = CombineLabels(softLabels, targetLabels, 0.7);

        // Step 6: Create and train a new model on the target domain
        // Note: This is simplified - in practice, you'd create a compatible model type
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
