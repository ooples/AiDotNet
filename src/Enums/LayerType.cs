namespace AiDotNet.Enums;

/// <summary>
/// Specifies different types of layers used in neural networks, particularly in deep learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Neural networks are composed of layers of artificial neurons that process information.
/// Think of a neural network as an assembly line in a factory, where each layer is a workstation that 
/// performs a specific task on the data before passing it to the next layer.
/// 
/// Different layer types serve different purposes:
/// - Some extract features from data (like identifying edges in images)
/// - Some reduce the amount of data to process (making computation faster)
/// - Some make final decisions based on processed information
/// 
/// The combination of different layer types allows neural networks to learn complex patterns
/// and solve difficult problems in areas like image recognition, language processing, and more.
/// </para>
/// </remarks>
public enum LayerType
{
    /// <summary>
    /// A layer that reduces the spatial dimensions of data by combining nearby values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pooling layers work like summarizers - they take a group of nearby values 
    /// and combine them into a single value, making the data smaller and more manageable.
    /// 
    /// Think of it as:
    /// - Looking at a detailed image through a window that only shows the brightest pixel in each area
    /// - Zooming out on a photo to see the general shapes rather than every detail
    /// - Summarizing a paragraph of text with a single sentence
    /// 
    /// Common pooling operations include:
    /// - Max pooling: keeping only the maximum value in each region
    /// - Average pooling: taking the average of all values in each region
    /// 
    /// Benefits of pooling:
    /// - Reduces computation by making data smaller
    /// - Makes the network less sensitive to exact positions of features
    /// - Helps the network focus on what's important rather than minor details
    /// - Reduces the risk of overfitting (memorizing training data too precisely)
    /// </para>
    /// </remarks>
    Pooling,

    /// <summary>
    /// A layer that applies filters to detect patterns in input data, commonly used for image processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convolutional layers are like pattern detectors that scan across data 
    /// (especially images) looking for specific features.
    /// 
    /// Think of it as:
    /// - A flashlight moving across an image, illuminating small regions at a time
    /// - A detective using a magnifying glass to search for clues
    /// - A set of templates that the network learns to match against parts of the input
    /// 
    /// How it works:
    /// - Small "filters" (also called kernels) slide across the input data
    /// - Each filter learns to detect a specific pattern (like edges, textures, or shapes)
    /// - Early layers detect simple patterns, while deeper layers combine these to detect complex features
    /// 
    /// Convolutional layers are especially powerful for:
    /// - Image recognition and computer vision
    /// - Any data with spatial or sequential patterns
    /// - Problems where the same pattern might appear in different locations
    /// </para>
    /// </remarks>
    Convolutional,

    /// <summary>
    /// A layer where each neuron is connected to every neuron in the previous layer, used for complex pattern recognition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fully Connected layers (also called Dense layers) connect every input to every output,
    /// allowing the network to combine all available information to make decisions.
    ///
    /// Think of it as:
    /// - A voting system where every piece of evidence gets to influence the final decision
    /// - A committee where everyone listens to all information before making a judgment
    /// - The "thinking" part of the network that combines all the features detected by earlier layers
    ///
    /// How it works:
    /// - Each neuron receives input from all neurons in the previous layer
    /// - Each connection has a weight that strengthens or weakens that particular influence
    /// - The network learns which connections are important by adjusting these weights
    ///
    /// Fully Connected layers are typically used:
    /// - Near the end of a neural network
    /// - To combine features extracted by earlier layers
    /// - For final classification or regression tasks
    /// - When all input features might be relevant to all outputs
    /// </para>
    /// </remarks>
    FullyConnected,

    /// <summary>
    /// A densely connected layer where each neuron is connected to every neuron in the previous layer (alias for FullyConnected).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dense is another name for FullyConnected layers. They are exactly the same thing.
    ///
    /// This naming convention is commonly used in popular frameworks like Keras/TensorFlow, where "Dense"
    /// is used to create fully connected layers. Having this alias makes the code more familiar to developers
    /// coming from those frameworks.
    ///
    /// Use this when:
    /// - You're more familiar with Keras/TensorFlow terminology
    /// - You want your code to read like those frameworks
    /// - You need a fully connected layer
    /// </para>
    /// </remarks>
    Dense,

    /// <summary>
    /// A layer implementing Low-Rank Adaptation for parameter-efficient fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LoRA (Low-Rank Adaptation) layers enable efficient fine-tuning of neural networks
    /// by learning small adaptations instead of updating all weights.
    ///
    /// Think of it as:
    /// - Adding "correction notes" to an existing layer instead of rewriting it entirely
    /// - Using a few master controls to adjust many parameters at once
    /// - Learning what changes are needed rather than learning everything from scratch
    ///
    /// How it works:
    /// - Decomposes weight updates into two small matrices (A and B)
    /// - Dramatically reduces trainable parameters (often by 98% or more)
    /// - Can be merged back into the original weights after training
    ///
    /// LoRA layers are especially useful for:
    /// - Fine-tuning large pre-trained models with limited resources
    /// - Adapting models to multiple tasks efficiently
    /// - Reducing memory requirements during training
    /// - Faster experimentation with model adaptations
    /// </para>
    /// </remarks>
    LoRA
}
