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
    /// Input layer that receives the initial data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Input layer is the entry point for data into a neural network.
    /// It defines the shape and format of the data that the network will process.
    /// </para>
    /// </remarks>
    Input,

    /// <summary>
    /// Output layer that produces the final predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Output layer is the final layer that produces the network's predictions.
    /// It uses an appropriate activation function based on the task (e.g., softmax for classification,
    /// linear for regression).
    /// </para>
    /// </remarks>
    Output,

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
    /// A densely connected layer where each neuron connects to all neurons in the previous layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dense layers (also known as Fully Connected layers) are the basic building blocks
    /// of neural networks where every input connects to every output neuron.
    /// </para>
    /// </remarks>
    Dense,

    /// <summary>
    /// Long Short-Term Memory layer for processing sequential data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LSTM layers are specialized for learning patterns in sequential data like
    /// text or time series, with memory cells that can remember important information over long sequences.
    /// </para>
    /// </remarks>
    LSTM,

    /// <summary>
    /// Gated Recurrent Unit layer for processing sequential data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GRU layers are similar to LSTM but simpler, used for sequential data
    /// processing with fewer parameters while maintaining good performance.
    /// </para>
    /// </remarks>
    GRU,

    /// <summary>
    /// Dropout layer that randomly drops neurons during training to prevent overfitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout layers randomly ignore some neurons during training, which helps
    /// prevent the model from memorizing the training data and improves generalization.
    /// </para>
    /// </remarks>
    Dropout,

    /// <summary>
    /// Batch Normalization layer that normalizes inputs to improve training stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Batch Normalization layers normalize the inputs to each layer, which
    /// speeds up training and makes the network more stable and less sensitive to initialization.
    /// </para>
    /// </remarks>
    BatchNormalization,

    /// <summary>
    /// Max Pooling layer that takes the maximum value from each region.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Max Pooling layers reduce data size by keeping only the maximum value
    /// from each region, preserving the strongest features while reducing computation.
    /// </para>
    /// </remarks>
    MaxPooling,

    /// <summary>
    /// Average Pooling layer that takes the average value from each region.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Average Pooling layers reduce data size by computing the average of each region,
    /// providing a smoother down-sampling compared to max pooling.
    /// </para>
    /// </remarks>
    AveragePooling,

    /// <summary>
    /// Custom layer type for user-defined or specialized layer implementations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Custom layers allow you to implement specialized functionality
    /// that isn't covered by the standard layer types.
    /// </para>
    /// </remarks>
    Custom
}