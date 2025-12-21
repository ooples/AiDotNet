namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Radial Basis Function (RBF) models, a type of artificial neural network
/// that uses radial basis functions as activation functions for approximating complex non-linear relationships.
/// </summary>
/// <remarks>
/// <para>
/// Radial Basis Function networks are a specialized type of neural network that utilize radially symmetric
/// functions (typically Gaussian) centered at specific points in the feature space. These networks excel at
/// function approximation, interpolation, and classification tasks. RBF networks consist of an input layer,
/// a hidden layer with RBF activation functions, and an output layer. Each neuron in the hidden layer represents
/// a radial basis function centered at a particular point. The output of the network is typically a linear
/// combination of these basis functions. RBF networks are known for their ability to model complex non-linear
/// relationships while often requiring less training time than traditional multilayer perceptrons. They are
/// particularly effective for problems where the data exhibits localized patterns or when smooth interpolation
/// between data points is desired.
/// </para>
/// <para><b>For Beginners:</b> Radial Basis Function networks are a special kind of AI model that's good at finding patterns in data.
/// 
/// Think about weather prediction:
/// - Traditional models might try to find one formula that works for the whole world
/// - But an RBF network places "experts" at different locations
/// - Each "expert" (or center) specializes in predicting weather in their local area
/// - The final prediction combines opinions from nearby experts, with closer ones having more influence
/// 
/// What this technique does:
/// - It places a number of "centers" throughout your data
/// - Each center is like a spotlight that illuminates the nearby data points
/// - The model learns how strong each spotlight should be
/// - Predictions are made by seeing how much light falls on new data points
/// 
/// This is especially useful when:
/// - Your data has clusters or regions with different patterns
/// - You need a model that can adapt to different "neighborhoods" in your data
/// - You want smooth transitions between these different regions
/// - The relationship between inputs and outputs changes across your data space
/// 
/// For example, in image recognition, different RBF centers might specialize in detecting different
/// shapes or textures, and the combined output helps identify the complete image.
///
/// This class lets you configure how the RBF network is structured and initialized.
/// </para>
/// </remarks>
public class RadialBasisFunctionOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the number of RBF centers (hidden neurons) in the network.
    /// </summary>
    /// <value>The number of centers, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many radial basis functions (centers) will be used in the hidden layer
    /// of the network. Each center represents a specific point in the feature space around which a radial
    /// basis function (typically Gaussian) is placed. The centers are usually determined through clustering
    /// algorithms such as k-means, or they may be randomly sampled from the training data. More centers
    /// allow the network to capture more complex patterns and local variations in the data, but increase
    /// computational cost and may lead to overfitting with limited training data. Fewer centers result in
    /// a simpler model that generalizes more broadly but might miss important local patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many "experts" or "centers" the model uses to analyze your data.
    /// 
    /// The default value of 10 means:
    /// - The model will place 10 different centers throughout your data space
    /// - Each center specializes in making predictions for data points near it
    /// - The final prediction combines information from multiple centers
    /// 
    /// Think of it like placing weather stations across a country:
    /// - Each weather station (center) is good at predicting local conditions
    /// - More stations give you more detailed coverage but require more maintenance
    /// - Too few stations might miss important local weather patterns
    /// - Too many stations might be wasteful and could start reporting noise
    /// 
    /// You might want more centers (like 50 or 100):
    /// - When you have a lot of training data
    /// - When your data has many distinct regions or clusters
    /// - When you need to capture very detailed local patterns
    /// - When your problem is complex and requires fine-grained analysis
    /// 
    /// You might want fewer centers (like 5 or 3):
    /// - When you have limited training data
    /// - When you want to avoid overfitting
    /// - When your data has simple patterns
    /// - When you need faster prediction times
    /// - When you want a more interpretable model
    /// 
    /// Finding the right number of centers often requires experimentation to balance
    /// model complexity against generalization ability.
    /// </para>
    /// </remarks>
    public int NumberOfCenters { get; set; } = 10;
}
