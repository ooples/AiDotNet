namespace AiDotNet.Enums;

/// <summary>
/// Defines methods for interpreting and explaining model predictions.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These methods help understand why a model made a particular prediction. 
/// It's like asking the model to explain its reasoning, which is crucial for trust and debugging.
/// </para>
/// </remarks>
public enum InterpretationMethod
{
    /// <summary>
    /// No interpretation method applied.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The model just gives predictions without any explanation - like 
    /// getting an answer without showing the work.
    /// </remarks>
    None,

    /// <summary>
    /// Feature importance scores.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Shows which input features mattered most for the prediction - like 
    /// highlighting the most important factors in a decision.
    /// </remarks>
    FeatureImportance,

    /// <summary>
    /// SHAP (SHapley Additive exPlanations) values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Assigns each feature a contribution score showing how it pushed the 
    /// prediction higher or lower - based on game theory mathematics.
    /// </remarks>
    SHAP,

    /// <summary>
    /// LIME (Local Interpretable Model-agnostic Explanations).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Explains individual predictions by training a simple model to mimic 
    /// the complex model's behavior around that specific prediction.
    /// </remarks>
    LIME,

    /// <summary>
    /// Gradient-based attribution.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses calculus to measure how sensitive the prediction is to each 
    /// input feature - larger gradients mean more influence.
    /// </remarks>
    GradientAttribution,

    /// <summary>
    /// Integrated gradients method.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An improved gradient method that accumulates attributions along a 
    /// path from a baseline to the input - more reliable than simple gradients.
    /// </remarks>
    IntegratedGradients,

    /// <summary>
    /// Attention visualization for neural networks.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Shows what parts of the input the model is "looking at" most - 
    /// commonly used in image and text models.
    /// </remarks>
    AttentionVisualization,

    /// <summary>
    /// Counterfactual explanations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Shows what would need to change in the input to get a different 
    /// prediction - "if your income was $5000 higher, you would be approved."
    /// </remarks>
    Counterfactual,

    /// <summary>
    /// Permutation importance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Randomly shuffles each feature and measures how much the predictions 
    /// get worse - bigger drops mean more important features.
    /// </remarks>
    PermutationImportance,

    /// <summary>
    /// Partial dependence plots.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Shows how predictions change as one feature varies while keeping 
    /// others constant - like a graph showing how price affects sales.
    /// </remarks>
    PartialDependence,

    /// <summary>
    /// Individual conditional expectation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Like partial dependence but for individual predictions - shows how 
    /// changing one feature affects a specific instance's prediction.
    /// </remarks>
    ICE,

    /// <summary>
    /// Anchors (rule-based explanations).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Finds simple if-then rules that explain when the model will make 
    /// the same prediction - "if age > 25 AND income > 50k, then approved."
    /// </remarks>
    Anchors,

    /// <summary>
    /// DeepLIFT (Deep Learning Important FeaTures).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Compares neuron activations to a reference to determine feature 
    /// contributions - designed specifically for deep neural networks.
    /// </remarks>
    DeepLIFT,

    /// <summary>
    /// Layer-wise relevance propagation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Traces the prediction backwards through the neural network to see 
    /// which inputs contributed most at each layer.
    /// </remarks>
    LRP,

    /// <summary>
    /// Concept activation vectors.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tests whether the model has learned human-understandable concepts - 
    /// like checking if an image model understands "stripes" or "wheels."
    /// </remarks>
    CAV,

    /// <summary>
    /// Prototype-based explanations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Explains predictions by showing similar examples from the training 
    /// data - "this looks like these 5 dogs I've seen before."
    /// </remarks>
    Prototype,

    /// <summary>
    /// Custom interpretation method.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to implement your own method for explaining model predictions.
    /// </remarks>
    Custom
}