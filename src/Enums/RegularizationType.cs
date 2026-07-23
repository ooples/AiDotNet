namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of regularization to apply to a machine learning model.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Regularization is like adding training wheels to your AI model.
/// 
/// When models learn too much from their training data, they might become too specialized
/// (this is called "overfitting"). Regularization helps prevent this by encouraging the model
/// to keep things simple.
/// 
/// Think of it like this:
/// - Without regularization: The model might create very complex rules that work perfectly for
///   training data but fail on new data.
/// - With regularization: The model is encouraged to create simpler rules that work well enough
///   for training data and are more likely to work on new data too.
/// 
/// Different regularization types use different approaches to encourage simplicity.
/// </remarks>
public enum RegularizationType
{
    /// <summary>
    /// No regularization is applied to the model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This option turns off regularization completely.
    /// 
    /// Use this when:
    /// - You have lots of training data compared to model complexity
    /// - Your model is already simple and unlikely to overfit
    /// - You want to see how the model performs without any restrictions
    /// 
    /// It's like removing the training wheels - sometimes it works fine,
    /// but there's a higher risk the model might become too specialized to your training data.
    /// </remarks>
    None,

    /// <summary>
    /// L1 regularization (also known as Lasso regularization) that encourages sparsity in the model parameters.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> L1 regularization encourages the model to completely ignore less important features.
    /// 
    /// It works by penalizing the absolute size of the model's parameters, which often results in
    /// many parameters becoming exactly zero.
    /// 
    /// Think of it like a strict teacher who says: "If a feature isn't clearly helpful, don't use it at all."
    /// 
    /// Benefits:
    /// - Automatically selects the most important features
    /// - Creates simpler models that are easier to interpret
    /// - Works well when you suspect many features aren't relevant
    /// 
    /// Example: If you're predicting house prices with 100 features, L1 might decide that only 20 features
    /// (like size, location, and age) actually matter and ignore the rest.
    /// </remarks>
    L1,

    /// <summary>
    /// L2 regularization (also known as Ridge regularization) that discourages large parameter values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> L2 regularization encourages the model to use all features, but keep their influence small.
    /// 
    /// It works by penalizing the squared size of the model's parameters, which results in
    /// all parameters becoming smaller but rarely exactly zero.
    /// 
    /// Think of it like a balanced teacher who says: "Use all the information available, but don't rely too much on any single piece."
    /// 
    /// Benefits:
    /// - Handles correlated features well
    /// - Generally prevents overfitting without eliminating features
    /// - Usually the safest default choice for regularization
    /// 
    /// Example: For house price prediction, L2 might keep all 100 features but ensure that no single feature
    /// (like having a swimming pool) has an excessively large impact on the prediction.
    /// </remarks>
    L2,

    /// <summary>
    /// A combination of L1 and L2 regularization that balances their properties.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> ElasticNet combines the best of both L1 and L2 regularization.
    /// 
    /// It works by applying both types of penalties at the same time, with adjustable weights
    /// to control how much of each to use.
    /// 
    /// Think of it like a flexible teacher who says: "Let's mostly keep all features but with limited influence,
    /// while still completely removing the least useful ones."
    /// 
    /// Benefits:
    /// - Can eliminate irrelevant features (like L1)
    /// - Handles groups of correlated features well (like L2)
    /// - Provides more flexibility through adjustable balance between L1 and L2
    /// 
    /// Example: For house price prediction, ElasticNet might eliminate 30 truly irrelevant features
    /// while keeping the remaining 70 with appropriately controlled influence.
    /// 
    /// This is often the best choice when you're not sure which regularization to use.
    /// </remarks>
    ElasticNet
}
