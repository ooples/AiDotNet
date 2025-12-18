namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality for tree-based machine learning models.
/// </summary>
/// <remarks>
/// Tree-based models make predictions by following a series of decision rules organized in a tree-like structure.
/// These models can be used for both classification (predicting categories) and regression (predicting numeric values).
/// 
/// <b>For Beginners:</b> Tree-based models work like a flowchart of yes/no questions to make predictions.
/// Imagine you're trying to predict if someone will like a movie:
/// 
/// 1. Is it an action movie? If yes, go to question 2. If no, go to question 3.
/// 2. Does it have their favorite actor? If yes, predict "Like". If no, predict "Dislike".
/// 3. Is it less than 2 hours long? If yes, predict "Like". If no, predict "Dislike".
/// 
/// This is a simple decision tree. More advanced tree-based models like Random Forests or 
/// Gradient Boosted Trees use multiple trees together to make better predictions.
/// 
/// This interface inherits from IFullModel&lt;T&gt;, which provides the basic methods for training,
/// predicting, and evaluating machine learning models.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface ITreeBasedRegression<T> : INonLinearRegression<T>
{
    /// <summary>
    /// Gets the number of decision trees used in the model.
    /// </summary>
    /// <remarks>
    /// For single decision tree models, this value is 1. For ensemble methods like Random Forests
    /// or Gradient Boosted Trees, this represents the number of trees in the ensemble.
    /// 
    /// <b>For Beginners:</b> Think of this as "how many different flowcharts is the model using to make its decision?"
    /// 
    /// - If NumberOfTrees = 1: The model is using a single decision tree (like the movie example above)
    /// - If NumberOfTrees > 1: The model is using multiple trees and combining their predictions
    /// 
    /// More trees often lead to better predictions but make the model slower and more complex.
    /// Common values range from 10 to 1000 trees, depending on the specific algorithm and dataset.
    /// </remarks>
    int NumberOfTrees { get; }

    /// <summary>
    /// Gets the maximum depth (number of sequential decisions) allowed in each decision tree.
    /// </summary>
    /// <remarks>
    /// The depth of a tree is the maximum number of decisions that must be made to reach a prediction.
    /// 
    /// <b>For Beginners:</b> This tells you how many questions the model can ask before making a prediction.
    /// 
    /// For example, if MaxDepth = 3:
    /// - The model can ask at most 3 questions before making a prediction
    /// - This creates a simpler model that might be easier to understand
    /// - But it might miss complex patterns in your data
    /// 
    /// If MaxDepth = 20:
    /// - The model can ask up to 20 questions before deciding
    /// - This creates a more complex model that can capture detailed patterns
    /// - But it might "memorize" your training data instead of learning general rules
    /// 
    /// Setting the right MaxDepth helps balance between a model that's too simple (underfitting)
    /// and one that's too complex (overfitting).
    /// </remarks>
    int MaxDepth { get; }

    /// <summary>
    /// Gets the relative importance of each feature in making predictions.
    /// </summary>
    /// <remarks>
    /// Feature importance indicates how much each input variable contributes to the model's predictions.
    /// Higher values indicate more important features.
    /// 
    /// <b>For Beginners:</b> This tells you which of your input variables are most helpful for making predictions.
    /// 
    /// For example, if you're predicting house prices:
    /// - FeatureImportances[0] = 0.7 for "square footage"
    /// - FeatureImportances[1] = 0.2 for "number of bedrooms"
    /// - FeatureImportances[2] = 0.1 for "year built"
    /// 
    /// This would tell you that square footage is the most important factor in your model's predictions,
    /// followed by number of bedrooms, with year built having the least impact.
    /// 
    /// You can use this information to:
    /// - Focus on collecting better data for important features
    /// - Possibly remove unimportant features to simplify your model
    /// - Better understand what drives the predictions in your specific problem
    /// </remarks>
    Vector<T> FeatureImportances { get; }
}
