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
    /// A Long Short-Term Memory layer that processes sequential data while maintaining long-term dependencies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LSTM layers are like smart memory systems that can remember important information 
    /// for a long time while forgetting irrelevant details.
    /// 
    /// Think of it as:
    /// - A notepad with special features that knows what to remember and what to forget
    /// - A selective memory that keeps track of important patterns in sequences
    /// - A gatekeeper that controls the flow of information through time
    /// 
    /// LSTMs are particularly good at:
    /// - Processing text and natural language
    /// - Time series prediction (like stock prices or weather)
    /// - Speech recognition and generation
    /// - Any task where the order and context of information matters
    /// </para>
    /// </remarks>
    LSTM,

    /// <summary>
    /// A Gated Recurrent Unit layer that efficiently processes sequential data with fewer parameters than LSTM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GRU layers are a simpler version of LSTM that still handle sequences well 
    /// but with less complexity.
    /// 
    /// Think of it as:
    /// - A streamlined memory system with fewer moving parts
    /// - A more efficient way to process sequences when you don't need all LSTM features
    /// - A balance between simple RNNs and complex LSTMs
    /// 
    /// GRUs are often used for:
    /// - Similar tasks as LSTMs but when computational efficiency is important
    /// - Smaller datasets where simpler models work better
    /// - Real-time applications that need faster processing
    /// </para>
    /// </remarks>
    GRU,

    /// <summary>
    /// A layer that randomly drops connections during training to prevent overfitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout layers help prevent the network from memorizing training data too precisely 
    /// by randomly "turning off" some neurons during training.
    /// 
    /// Think of it as:
    /// - Training with different team members absent each day, so no one becomes indispensable
    /// - Learning to solve problems even when some information is missing
    /// - Building redundancy so the network doesn't rely too heavily on specific connections
    /// 
    /// Benefits of dropout:
    /// - Reduces overfitting (memorization of training data)
    /// - Makes the network more robust and generalizable
    /// - Acts like training multiple networks at once
    /// - Only active during training, not during actual use
    /// </para>
    /// </remarks>
    Dropout,

    /// <summary>
    /// A layer that normalizes inputs across the batch dimension to stabilize training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Batch Normalization layers help keep the data flowing through the network 
    /// in a consistent range, making training faster and more stable.
    /// 
    /// Think of it as:
    /// - Adjusting the volume of different instruments in a band so they're balanced
    /// - Standardizing measurements so they're all on the same scale
    /// - Keeping data centered and spread consistently through the network
    /// 
    /// Benefits include:
    /// - Faster training convergence
    /// - Ability to use higher learning rates
    /// - Reduces sensitivity to weight initialization
    /// - Acts as a form of regularization
    /// </para>
    /// </remarks>
    BatchNormalization,

    /// <summary>
    /// A pooling layer that selects the maximum value in each pooling window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Max Pooling layers keep only the strongest signal from each region, 
    /// like picking the brightest pixel in each area of an image.
    /// 
    /// Think of it as:
    /// - Finding the loudest voice in each group
    /// - Keeping only the most prominent feature in each region
    /// - Highlighting the strongest patterns while reducing data size
    /// 
    /// Max pooling is useful for:
    /// - Detecting if a feature exists anywhere in a region
    /// - Making the network invariant to small translations
    /// - Reducing computational load while preserving important features
    /// </para>
    /// </remarks>
    MaxPooling,

    /// <summary>
    /// A pooling layer that computes the average value in each pooling window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Average Pooling layers smooth out the data by taking the average 
    /// of all values in each region.
    /// 
    /// Think of it as:
    /// - Taking the average temperature of a room instead of the hottest spot
    /// - Smoothing out noise by considering all values equally
    /// - Getting a general sense of what's happening in each region
    /// 
    /// Average pooling is preferred when:
    /// - You want to preserve more information than max pooling
    /// - The overall pattern matters more than individual strong features
    /// - Dealing with data where extremes might be noise
    /// </para>
    /// </remarks>
    AveragePooling,

    /// <summary>
    /// A layer that applies an activation function element-wise to its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Activation layers add non-linearity to the network, allowing it 
    /// to learn complex patterns beyond simple linear relationships.
    /// 
    /// Think of it as:
    /// - Adding personality to neurons - they don't just pass information linearly
    /// - Deciding whether a neuron should "fire" based on its input
    /// - Transforming signals to make them more useful for learning
    /// 
    /// Common activation functions include:
    /// - ReLU: Keeps positive values, zeros out negatives
    /// - Sigmoid: Squashes values between 0 and 1
    /// - Tanh: Squashes values between -1 and 1
    /// </para>
    /// </remarks>
    Activation,

    /// <summary>
    /// A layer that adds multiple input tensors element-wise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Add layers combine multiple inputs by adding them together element by element, 
    /// useful for creating shortcuts or residual connections.
    /// 
    /// Think of it as:
    /// - Combining multiple streams of information
    /// - Creating shortcuts that help gradients flow during training
    /// - Allowing the network to learn both transformations and identity mappings
    /// 
    /// Add layers are crucial in:
    /// - Residual networks (ResNets) for very deep architectures
    /// - Skip connections that preserve information
    /// - Combining features from different processing paths
    /// </para>
    /// </remarks>
    Add,

    /// <summary>
    /// A layer specialized for detecting anomalies or unusual patterns in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Anomaly Detector layers are designed to identify data points 
    /// that don't fit the normal pattern.
    /// 
    /// Think of it as:
    /// - A security system that notices unusual behavior
    /// - Quality control that spots defective products
    /// - A pattern recognizer that flags anything out of the ordinary
    /// 
    /// Used in applications like:
    /// - Fraud detection in financial transactions
    /// - Fault detection in manufacturing
    /// - Network intrusion detection
    /// </para>
    /// </remarks>
    AnomalyDetector,

    /// <summary>
    /// An attention mechanism layer that allows the network to focus on relevant parts of the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention layers help the network focus on the most important parts 
    /// of the input, like how you pay attention to key words when reading.
    /// 
    /// Think of it as:
    /// - A spotlight that highlights important information
    /// - The ability to look back at relevant parts of the input when making decisions
    /// - Weighted importance scores for different parts of the data
    /// 
    /// Attention is fundamental in:
    /// - Machine translation (focusing on relevant source words)
    /// - Image captioning (attending to relevant image regions)
    /// - Question answering (focusing on relevant context)
    /// </para>
    /// </remarks>
    Attention,

    /// <summary>
    /// A wrapper layer that processes sequences in both forward and backward directions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bidirectional layers process sequences from both directions, 
    /// getting context from both past and future.
    /// 
    /// Think of it as:
    /// - Reading a sentence both forwards and backwards to understand it better
    /// - Having hindsight when processing sequential data
    /// - Combining information from both directions for better understanding
    /// 
    /// Particularly useful for:
    /// - Natural language processing where context matters
    /// - Speech recognition
    /// - Any sequence where future context helps understand the present
    /// </para>
    /// </remarks>
    Bidirectional,

    /// <summary>
    /// A capsule layer that preserves spatial relationships between features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Capsule layers are designed to preserve not just what features exist, 
    /// but also how they're arranged relative to each other.
    /// 
    /// Think of it as:
    /// - Recognizing not just that a face has eyes, nose, and mouth, but their arrangement
    /// - Understanding objects as collections of parts with specific relationships
    /// - Preserving pose and position information that CNNs might lose
    /// 
    /// Capsule networks excel at:
    /// - Recognizing objects from different viewpoints
    /// - Understanding spatial hierarchies
    /// - Generalizing to new poses and positions
    /// </para>
    /// </remarks>
    Capsule,

    /// <summary>
    /// A layer that concatenates multiple inputs along a specified axis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Concatenate layers combine multiple inputs by joining them together, 
    /// like connecting train cars.
    /// 
    /// Think of it as:
    /// - Combining different types of features into one representation
    /// - Merging outputs from parallel processing paths
    /// - Stacking information from different sources
    /// 
    /// Common uses include:
    /// - Combining features from different network branches
    /// - Merging multi-scale features in U-Net architectures
    /// - Combining different types of inputs (e.g., image and text)
    /// </para>
    /// </remarks>
    Concatenate,

    /// <summary>
    /// A layer implementing conditional random fields for structured prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Conditional Random Field layers model dependencies between outputs, 
    /// ensuring predictions make sense together.
    /// 
    /// Think of it as:
    /// - Ensuring word labels in a sentence are grammatically consistent
    /// - Making sure neighboring pixels in segmentation are coherent
    /// - Adding structure to predictions that should follow rules
    /// 
    /// Often used in:
    /// - Named entity recognition
    /// - Part-of-speech tagging
    /// - Image segmentation
    /// </para>
    /// </remarks>
    ConditionalRandomField,

    /// <summary>
    /// A convolutional LSTM layer that processes spatial-temporal data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ConvLSTM layers combine the spatial pattern detection of CNNs 
    /// with the temporal memory of LSTMs.
    /// 
    /// Think of it as:
    /// - Watching a movie and understanding both what's in each frame and how it changes
    /// - Processing weather data that has both spatial patterns and temporal evolution
    /// - Tracking objects that move and change shape over time
    /// 
    /// Ideal for:
    /// - Video prediction and analysis
    /// - Weather forecasting
    /// - Any data with both spatial and temporal patterns
    /// </para>
    /// </remarks>
    ConvLSTM,

    /// <summary>
    /// A layer that removes parts of the input tensor, typically used for data augmentation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cropping layers remove edges from inputs, like cropping a photo 
    /// to focus on the center.
    /// 
    /// Think of it as:
    /// - Trimming unnecessary borders from images
    /// - Focusing on the central region of interest
    /// - Removing padding that was added earlier
    /// 
    /// Useful for:
    /// - Data augmentation during training
    /// - Removing artifacts from convolution borders
    /// - Matching tensor sizes in skip connections
    /// </para>
    /// </remarks>
    Cropping,

    /// <summary>
    /// A decoder layer used in sequence-to-sequence models and autoencoders.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decoder layers reconstruct or generate output from encoded representations, 
    /// like translating a compressed message back to full form.
    /// 
    /// Think of it as:
    /// - Unpacking a compressed file
    /// - Translating encoded thoughts into words
    /// - Generating detailed output from abstract representations
    /// 
    /// Common in:
    /// - Machine translation (generating target language)
    /// - Text generation
    /// - Image reconstruction in autoencoders
    /// </para>
    /// </remarks>
    Decoder,

    /// <summary>
    /// A transposed convolution layer used for upsampling in generative models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deconvolutional layers (transposed convolutions) do the opposite of regular convolutions, 
    /// increasing spatial dimensions instead of reducing them.
    /// 
    /// Think of it as:
    /// - Enlarging a small image to a bigger one
    /// - Going from compressed features back to full resolution
    /// - Reversing the convolution process
    /// 
    /// Essential for:
    /// - Image generation and super-resolution
    /// - Semantic segmentation (pixel-wise classification)
    /// - Any task requiring upsampling
    /// </para>
    /// </remarks>
    Deconvolutional,

    /// <summary>
    /// A fully connected layer, identical to FullyConnected but often used in specific frameworks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dense layers are another name for Fully Connected layers, 
    /// where every input connects to every output.
    /// 
    /// The name "Dense" emphasizes that connections are densely packed, 
    /// as opposed to sparse connections in other layer types.
    /// </para>
    /// </remarks>
    Dense,

    /// <summary>
    /// An efficient convolutional layer that separates spatial and channel-wise operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Depthwise Separable Convolutions break down standard convolutions 
    /// into two simpler operations, making them much more efficient.
    /// 
    /// Think of it as:
    /// - First looking at patterns within each color channel separately
    /// - Then combining information across channels
    /// - Doing the same work with far fewer calculations
    /// 
    /// Benefits:
    /// - Much faster and uses less memory
    /// - Often performs just as well as regular convolutions
    /// - Key component in mobile-friendly architectures like MobileNet
    /// </para>
    /// </remarks>
    DepthwiseSeparableConvolutional,

    /// <summary>
    /// A specialized capsule layer for digit recognition tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Digit Capsule layers are specifically designed for recognizing 
    /// handwritten or printed digits.
    /// 
    /// Think of it as:
    /// - A specialist that's really good at recognizing numbers
    /// - Understanding not just what digit it is, but how it's written
    /// - Preserving style and orientation information
    /// 
    /// Used in:
    /// - Handwritten digit recognition (MNIST)
    /// - OCR systems for numerical data
    /// - Check processing and form reading
    /// </para>
    /// </remarks>
    DigitCapsule,

    /// <summary>
    /// A convolutional layer with dilated filters for increased receptive field.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dilated Convolutions have gaps in their filters, allowing them 
    /// to see a wider area without increasing computation.
    /// 
    /// Think of it as:
    /// - Looking at every other pixel to see a bigger picture
    /// - A filter with holes that covers more area
    /// - Seeing context without losing resolution
    /// 
    /// Particularly useful for:
    /// - Semantic segmentation where context matters
    /// - Audio processing (WaveNet)
    /// - Any task needing large receptive fields
    /// </para>
    /// </remarks>
    DilatedConvolutional,

    /// <summary>
    /// A layer that outputs probability distributions instead of single values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Distributional layers output entire probability distributions 
    /// rather than just point estimates.
    /// 
    /// Think of it as:
    /// - Predicting not just "it will rain" but "30% chance of light rain, 20% heavy rain"
    /// - Capturing uncertainty in predictions
    /// - Providing richer information about possible outcomes
    /// 
    /// Used in:
    /// - Reinforcement learning (distributional RL)
    /// - Uncertainty quantification
    /// - Risk-sensitive decision making
    /// </para>
    /// </remarks>
    Distributional,

    /// <summary>
    /// A layer that converts discrete tokens (like words) into continuous vector representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Embedding layers convert discrete items (like words or categories) 
    /// into numerical vectors that neural networks can process.
    /// 
    /// Think of it as:
    /// - Translating words into a language that neural networks understand
    /// - Mapping similar words to nearby points in space
    /// - Creating meaningful numerical representations of categories
    /// 
    /// Essential for:
    /// - Natural language processing
    /// - Recommendation systems (user/item embeddings)
    /// - Any task with categorical inputs
    /// </para>
    /// </remarks>
    Embedding,

    /// <summary>
    /// A standard feed-forward layer used in transformer architectures.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feed-forward layers in transformers process each position 
    /// independently with the same dense network.
    /// 
    /// Think of it as:
    /// - Applying the same transformation to each word in a sentence
    /// - Adding non-linearity after attention operations
    /// - The "thinking" step after gathering information
    /// 
    /// In transformers, these layers:
    /// - Follow attention layers
    /// - Process positions independently
    /// - Often have a hidden dimension larger than the model dimension
    /// </para>
    /// </remarks>
    FeedForward,

    /// <summary>
    /// A layer that converts multi-dimensional inputs into a 1D vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Flatten layers reshape multi-dimensional data into a single long vector, 
    /// like unrolling a matrix into a list.
    /// 
    /// Think of it as:
    /// - Arranging a 2D grid of pixels into a single line
    /// - Preparing spatial data for fully connected layers
    /// - Converting any shape into a format dense layers can process
    /// 
    /// Commonly used:
    /// - Between convolutional and dense layers
    /// - When transitioning from spatial to non-spatial processing
    /// - To prepare data for final classification layers
    /// </para>
    /// </remarks>
    Flatten,

    /// <summary>
    /// A layer implementing gated linear units for selective information flow.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gated Linear Units control information flow by learning 
    /// which parts of the input to let through.
    /// 
    /// Think of it as:
    /// - A smart filter that learns what information is important
    /// - Gates that open and close based on the input
    /// - Selective attention at the feature level
    /// 
    /// Benefits:
    /// - Helps with gradient flow in deep networks
    /// - Provides a learnable way to select features
    /// - Often improves performance in language models
    /// </para>
    /// </remarks>
    GatedLinearUnit,

    /// <summary>
    /// A regularization layer that adds random Gaussian noise during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gaussian Noise layers add random variations to data during training, 
    /// making the model more robust.
    /// 
    /// Think of it as:
    /// - Training with slightly fuzzy data to handle real-world imperfections
    /// - Making the model less sensitive to exact input values
    /// - Simulating measurement errors or natural variations
    /// 
    /// Helps with:
    /// - Preventing overfitting
    /// - Making models robust to noisy inputs
    /// - Simulating real-world data variations
    /// </para>
    /// </remarks>
    GaussianNoise,

    /// <summary>
    /// A pooling layer that pools over the entire spatial dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Global Pooling reduces entire feature maps to single values, 
    /// summarizing spatial information completely.
    /// 
    /// Think of it as:
    /// - Getting one number to represent an entire image channel
    /// - The ultimate summary of spatial features
    /// - Reducing any size input to a fixed size output
    /// 
    /// Advantages:
    /// - Makes the network accept any input size
    /// - Reduces parameters compared to flatten + dense
    /// - Often used before final classification
    /// </para>
    /// </remarks>
    GlobalPooling,

    /// <summary>
    /// A layer designed for processing graph-structured data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Graph Convolutional layers process data that's organized as networks 
    /// or graphs, like social networks or molecular structures.
    /// 
    /// Think of it as:
    /// - Understanding relationships between connected items
    /// - Processing data where connections matter as much as values
    /// - Learning from network structures
    /// 
    /// Used for:
    /// - Social network analysis
    /// - Molecular property prediction
    /// - Knowledge graphs and recommendation systems
    /// </para>
    /// </remarks>
    GraphConvolutional,

    /// <summary>
    /// A layer with gated connections that can bypass transformations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Highway layers have learnable gates that decide whether to transform 
    /// the input or let it pass through unchanged.
    /// 
    /// Think of it as:
    /// - A highway with express lanes that bypass processing
    /// - Smart shortcuts that the network learns to use
    /// - Adaptive depth - using transformations only when needed
    /// 
    /// Benefits:
    /// - Enables training of very deep networks
    /// - Information can flow directly when needed
    /// - Precursor to modern residual connections
    /// </para>
    /// </remarks>
    Highway,

    /// <summary>
    /// The entry point layer that receives raw input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Input layers are where data enters the neural network, 
    /// defining the shape and type of data the network expects.
    /// 
    /// Think of it as:
    /// - The front door of the neural network
    /// - A specification of what kind of data the network accepts
    /// - The starting point of all computations
    /// 
    /// Input layers:
    /// - Don't perform computations themselves
    /// - Define the expected data format
    /// - Are required in every neural network
    /// </para>
    /// </remarks>
    Input,

    /// <summary>
    /// A layer that applies custom user-defined functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Lambda layers let you insert custom operations that aren't 
    /// available as standard layers.
    /// 
    /// Think of it as:
    /// - A customizable tool for special operations
    /// - A way to add your own computations to the network
    /// - A flexible layer for experimental features
    /// 
    /// Use cases:
    /// - Implementing novel operations
    /// - Quick prototyping of new ideas
    /// - Adding simple custom transformations
    /// </para>
    /// </remarks>
    Lambda,

    /// <summary>
    /// A normalization layer that normalizes inputs across features instead of batch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Layer Normalization normalizes across features for each sample independently, 
    /// unlike batch normalization which normalizes across the batch.
    /// 
    /// Think of it as:
    /// - Normalizing each example independently
    /// - Standardizing features within each data point
    /// - Batch-size independent normalization
    /// 
    /// Preferred for:
    /// - Recurrent networks where batch norm is tricky
    /// - Transformer architectures
    /// - Small batch sizes or online learning
    /// </para>
    /// </remarks>
    LayerNormalization,

    /// <summary>
    /// A layer where each neuron connects to a local region of the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Locally Connected layers are like convolutions but without 
    /// weight sharing - each position has its own filters.
    /// 
    /// Think of it as:
    /// - Specialized detectors for each region of an image
    /// - Convolution where each location learns different patterns
    /// - Location-specific feature extraction
    /// 
    /// Useful when:
    /// - Different regions need different processing
    /// - Spatial invariance is not desired
    /// - Processing structured data with position-dependent features
    /// </para>
    /// </remarks>
    LocallyConnected,

    /// <summary>
    /// A layer that computes the logarithm of variance, used in variational models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Log Variance layers compute logarithmic variance values, 
    /// often used in probabilistic models to ensure numerical stability.
    /// 
    /// Think of it as:
    /// - Measuring uncertainty in log scale
    /// - Avoiding numerical issues with very small or large variances
    /// - A component in variational autoencoders
    /// 
    /// Used in:
    /// - Variational Autoencoders (VAEs)
    /// - Probabilistic neural networks
    /// - Uncertainty quantification
    /// </para>
    /// </remarks>
    LogVariance,

    /// <summary>
    /// A layer that masks (ignores) certain parts of the input, like padding in sequences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Masking layers tell the network to ignore certain parts of the input, 
    /// like padding added to make sequences the same length.
    /// 
    /// Think of it as:
    /// - Putting a cover over parts you want to ignore
    /// - Telling the network "don't look at these values"
    /// - Handling variable-length inputs properly
    /// 
    /// Essential for:
    /// - Processing padded sequences
    /// - Attention mechanisms (preventing attention to padding)
    /// - Any task with variable-length inputs
    /// </para>
    /// </remarks>
    Masking,

    /// <summary>
    /// A layer that computes the mean of inputs along specified axes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mean layers calculate averages across specified dimensions, 
    /// useful for aggregating information.
    /// 
    /// Think of it as:
    /// - Taking the average of multiple values
    /// - Summarizing information by averaging
    /// - Reducing dimensions through averaging
    /// 
    /// Common uses:
    /// - Global average pooling
    /// - Aggregating sequence outputs
    /// - Combining multiple representations
    /// </para>
    /// </remarks>
    Mean,

    /// <summary>
    /// A layer used in quantum-inspired neural networks for measurement operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Measurement layers in quantum-inspired networks simulate 
    /// the measurement process in quantum systems.
    /// 
    /// Think of it as:
    /// - Collapsing quantum-like states to classical values
    /// - Extracting observable information from complex states
    /// - The interface between quantum and classical processing
    /// 
    /// Used in:
    /// - Quantum machine learning models
    /// - Hybrid quantum-classical networks
    /// - Quantum-inspired algorithms
    /// </para>
    /// </remarks>
    Measurement,

    /// <summary>
    /// A layer that reads from external memory in memory-augmented networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory Read layers access external memory banks, allowing 
    /// networks to store and retrieve information.
    /// 
    /// Think of it as:
    /// - Looking up information in a notebook
    /// - Retrieving stored knowledge when needed
    /// - Giving the network long-term memory
    /// 
    /// Key component in:
    /// - Neural Turing Machines
    /// - Differentiable Neural Computers
    /// - Memory-augmented networks
    /// </para>
    /// </remarks>
    MemoryRead,

    /// <summary>
    /// A layer that writes to external memory in memory-augmented networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory Write layers store information in external memory banks 
    /// for later retrieval.
    /// 
    /// Think of it as:
    /// - Taking notes for future reference
    /// - Storing important information outside the main network
    /// - Building a knowledge base while processing
    /// 
    /// Paired with:
    /// - Memory Read layers
    /// - Attention mechanisms for addressing
    /// - Controller networks that decide what to store
    /// </para>
    /// </remarks>
    MemoryWrite,

    /// <summary>
    /// An attention layer with multiple parallel attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-Head Attention runs multiple attention operations in parallel, 
    /// each focusing on different types of relationships.
    /// 
    /// Think of it as:
    /// - Multiple experts each looking for different patterns
    /// - Parallel spotlights highlighting different aspects
    /// - Combining multiple perspectives for richer understanding
    /// 
    /// Core component of:
    /// - Transformer models
    /// - BERT, GPT, and other language models
    /// - Modern attention-based architectures
    /// </para>
    /// </remarks>
    MultiHeadAttention,

    /// <summary>
    /// A layer that multiplies multiple inputs element-wise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiply layers combine inputs through element-wise multiplication, 
    /// useful for gating and modulation.
    /// 
    /// Think of it as:
    /// - Applying masks or weights to inputs
    /// - Modulating one signal with another
    /// - Creating interactions between features
    /// 
    /// Used in:
    /// - Attention mechanisms (applying attention weights)
    /// - Gating mechanisms
    /// - Feature modulation
    /// </para>
    /// </remarks>
    Multiply,

    /// <summary>
    /// A linear layer with added noise for exploration in reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Noisy Linear layers have weights with learnable noise, 
    /// encouraging exploration in reinforcement learning.
    /// 
    /// Think of it as:
    /// - Weights that naturally explore different values
    /// - Built-in randomness for trying new strategies
    /// - Exploration that adapts during training
    /// 
    /// Benefits:
    /// - Better exploration than epsilon-greedy
    /// - Noise parameters are learned
    /// - State-dependent exploration
    /// </para>
    /// </remarks>
    NoisyLinear,

    /// <summary>
    /// A layer that adds padding around inputs, typically for maintaining dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Padding layers add extra values (usually zeros) around the edges 
    /// of inputs to control output dimensions.
    /// 
    /// Think of it as:
    /// - Adding a border around an image
    /// - Extending sequences to a fixed length
    /// - Preserving spatial dimensions through convolutions
    /// 
    /// Types of padding:
    /// - Zero padding: adds zeros
    /// - Reflection padding: mirrors edge values
    /// - Replication padding: repeats edge values
    /// </para>
    /// </remarks>
    Padding,

    /// <summary>
    /// A layer that adds positional information to embeddings in transformer models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Positional Encoding layers add information about position 
    /// in a sequence, since transformers don't inherently understand order.
    /// 
    /// Think of it as:
    /// - Adding line numbers to text so the model knows word order
    /// - GPS coordinates for each element in a sequence
    /// - Teaching the network about "before" and "after"
    /// 
    /// Essential because:
    /// - Transformers process all positions in parallel
    /// - Without it, "cat eats fish" = "fish eats cat"
    /// - Different encodings offer different benefits
    /// </para>
    /// </remarks>
    PositionalEncoding,

    /// <summary>
    /// The first capsule layer that converts regular features to capsules.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Primary Capsule layers are the entry point to capsule networks, 
    /// converting standard features into capsule format.
    /// 
    /// Think of it as:
    /// - Organizing features into groups that represent parts
    /// - Creating the initial capsules from convolutional features
    /// - The bridge between CNNs and capsule networks
    /// 
    /// Primary capsules:
    /// - Group related features together
    /// - Output vectors instead of scalars
    /// - Prepare data for routing to higher capsules
    /// </para>
    /// </remarks>
    PrimaryCapsule,

    /// <summary>
    /// A layer implementing quantum computing principles in neural networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantum layers simulate quantum computing operations, 
    /// potentially offering computational advantages for certain problems.
    /// 
    /// Think of it as:
    /// - Processing information in superposition (multiple states at once)
    /// - Leveraging quantum-like interference patterns
    /// - Exploring multiple solutions simultaneously
    /// 
    /// Potential advantages:
    /// - Exponential representation power
    /// - Natural handling of uncertainty
    /// - Novel optimization landscapes
    /// </para>
    /// </remarks>
    Quantum,

    /// <summary>
    /// A radial basis function layer for local pattern matching.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RBF layers respond strongly to inputs near specific centers, 
    /// like having specialized detectors for specific patterns.
    /// 
    /// Think of it as:
    /// - Sensors that activate when input is close to what they're tuned for
    /// - Pattern matchers with adjustable sensitivity
    /// - Local experts for different regions of input space
    /// 
    /// Useful for:
    /// - Function approximation
    /// - Pattern recognition
    /// - Creating locally responsive features
    /// </para>
    /// </remarks>
    RBF,

    /// <summary>
    /// A restricted Boltzmann machine layer for unsupervised feature learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RBM layers learn to reconstruct inputs, discovering hidden 
    /// patterns without labeled data.
    /// 
    /// Think of it as:
    /// - Learning to compress and reconstruct data
    /// - Finding hidden patterns without being told what to look for
    /// - Unsupervised feature discovery
    /// 
    /// Applications:
    /// - Pretraining deep networks
    /// - Dimensionality reduction
    /// - Generative modeling
    /// </para>
    /// </remarks>
    RBM,

    /// <summary>
    /// A layer that extracts final predictions from graph neural networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Readout layers aggregate node-level information in graphs 
    /// to make graph-level predictions.
    /// 
    /// Think of it as:
    /// - Summarizing all nodes to describe the entire graph
    /// - Creating a final verdict from distributed information
    /// - The graph equivalent of global pooling
    /// 
    /// Common readout operations:
    /// - Sum of all node features
    /// - Average of node features
    /// - More sophisticated aggregations
    /// </para>
    /// </remarks>
    Readout,

    /// <summary>
    /// A layer that reconstructs inputs from compressed representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reconstruction layers rebuild original inputs from compressed 
    /// representations, used in autoencoders.
    /// 
    /// Think of it as:
    /// - Rebuilding a picture from a sketch
    /// - Decompressing data back to original form
    /// - Testing if important information was preserved
    /// 
    /// Key component in:
    /// - Autoencoders
    /// - Denoising networks
    /// - Compression algorithms
    /// </para>
    /// </remarks>
    Reconstruction,

    /// <summary>
    /// A general recurrent layer for processing sequential data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Recurrent layers process sequences by maintaining a hidden state 
    /// that gets updated with each new input.
    /// 
    /// Think of it as:
    /// - A network with memory of previous inputs
    /// - Processing a story word by word while remembering context
    /// - Maintaining running state through time
    /// 
    /// Basic building block for:
    /// - Language processing
    /// - Time series analysis
    /// - Any sequential data processing
    /// </para>
    /// </remarks>
    Recurrent,

    /// <summary>
    /// A layer that can switch between training and inference parameterizations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RepParameterization layers use different parameter structures 
    /// during training and inference for efficiency.
    /// 
    /// Think of it as:
    /// - Training wheels that come off after learning
    /// - Using a complex structure for learning, simple for inference
    /// - Optimizing differently for training vs deployment
    /// 
    /// Benefits:
    /// - Better training dynamics
    /// - Faster inference
    /// - Used in RepVGG and similar architectures
    /// </para>
    /// </remarks>
    RepParameterization,

    /// <summary>
    /// A layer implementing reservoir computing for temporal processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reservoir layers contain a fixed random recurrent network 
    /// that creates rich representations of temporal inputs.
    /// 
    /// Think of it as:
    /// - Throwing inputs into a pond and watching the ripples
    /// - A complex but fixed transformation that captures temporal patterns
    /// - Random features that happen to be useful
    /// 
    /// Advantages:
    /// - Very fast training (only output weights are learned)
    /// - Good at capturing temporal dynamics
    /// - Biologically inspired
    /// </para>
    /// </remarks>
    Reservoir,

    /// <summary>
    /// A layer that changes the shape of tensors without altering data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reshape layers reorganize data into different dimensions 
    /// without changing the actual values.
    /// 
    /// Think of it as:
    /// - Rearranging a 2x3 grid into a 3x2 grid
    /// - Changing the shape of data to match what the next layer expects
    /// - Reorganizing without losing information
    /// 
    /// Common uses:
    /// - Preparing data for different layer types
    /// - Combining or splitting dimensions
    /// - Adapting between layer architectures
    /// </para>
    /// </remarks>
    Reshape,

    /// <summary>
    /// A layer implementing skip connections for deep networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Residual layers add skip connections that allow information 
    /// to bypass transformations, enabling very deep networks.
    /// 
    /// Think of it as:
    /// - Shortcuts that preserve original information
    /// - Learning changes rather than entirely new representations
    /// - Highway lanes for gradient flow
    /// 
    /// Revolutionary because:
    /// - Enabled training networks with 100+ layers
    /// - Solves vanishing gradient problem
    /// - Foundation of ResNet architecture
    /// </para>
    /// </remarks>
    Residual,

    /// <summary>
    /// An attention layer where positions attend to themselves.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Self-Attention layers let each position in a sequence look at 
    /// all other positions to gather relevant information.
    /// 
    /// Think of it as:
    /// - Each word in a sentence looking at all other words
    /// - Finding relationships between different parts of the input
    /// - Building context by connecting related elements
    /// 
    /// Key innovation in:
    /// - Transformer models
    /// - Replacing recurrence with attention
    /// - Parallel processing of sequences
    /// </para>
    /// </remarks>
    SelfAttention,

    /// <summary>
    /// A separable convolution layer splitting spatial and channel operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Separable Convolutions break down complex operations into 
    /// simpler, more efficient steps.
    /// 
    /// Think of it as:
    /// - Divide and conquer for convolutions
    /// - Processing space and channels separately
    /// - Getting similar results with less computation
    /// 
    /// Benefits:
    /// - Fewer parameters
    /// - Faster computation
    /// - Often similar accuracy to regular convolutions
    /// </para>
    /// </remarks>
    SeparableConvolutional,

    /// <summary>
    /// A layer implementing spatial pooling in Hierarchical Temporal Memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spatial Pooler layers are part of HTM systems, learning 
    /// sparse distributed representations of inputs.
    /// 
    /// Think of it as:
    /// - Creating a sparse code for inputs
    /// - Learning which features commonly occur together
    /// - Building efficient representations
    /// 
    /// HTM principles:
    /// - Biologically inspired
    /// - Sparse distributed representations
    /// - Online learning
    /// </para>
    /// </remarks>
    SpatialPooler,

    /// <summary>
    /// A layer that applies spatial transformations to inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spatial Transformer layers can rotate, scale, and deform inputs, 
    /// learning to focus on relevant regions.
    /// 
    /// Think of it as:
    /// - A smart camera that learns where to look
    /// - Automatically cropping and aligning inputs
    /// - Attention through geometric transformation
    /// 
    /// Enables:
    /// - Invariance to rotation and scale
    /// - Focusing on regions of interest
    /// - End-to-end spatial attention
    /// </para>
    /// </remarks>
    SpatialTransformer,

    /// <summary>
    /// A layer implementing spiking neural network dynamics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spiking layers simulate biological neurons that communicate 
    /// through discrete spikes rather than continuous values.
    /// 
    /// Think of it as:
    /// - Neurons that fire like real brain cells
    /// - Communication through timing of spikes
    /// - More biologically realistic processing
    /// 
    /// Potential advantages:
    /// - Energy efficiency
    /// - Temporal processing
    /// - Neuromorphic hardware compatibility
    /// </para>
    /// </remarks>
    Spiking,

    /// <summary>
    /// A layer that splits inputs into multiple outputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Split layers divide inputs into multiple parts for parallel 
    /// processing paths.
    /// 
    /// Think of it as:
    /// - Dividing a stream into multiple channels
    /// - Sending different parts to different processors
    /// - Creating branches in the network
    /// 
    /// Used for:
    /// - Multi-task learning
    /// - Parallel processing paths
    /// - Inception-style architectures
    /// </para>
    /// </remarks>
    Split,

    /// <summary>
    /// A layer that recalibrates channel-wise features through squeeze and excitation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Squeeze-and-Excitation layers learn to emphasize important 
    /// channels and suppress less useful ones.
    /// 
    /// Think of it as:
    /// - An automatic volume control for each feature channel
    /// - Learning which features deserve more attention
    /// - Channel-wise attention mechanism
    /// 
    /// Process:
    /// - Squeeze: global pooling to get channel statistics
    /// - Excitation: learn channel importance weights
    /// - Scale: multiply features by importance weights
    /// </para>
    /// </remarks>
    SqueezeAndExcitation,

    /// <summary>
    /// A layer for efficient upsampling using pixel shuffle operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Subpixel Convolutional layers efficiently increase image resolution 
    /// by rearranging channels into spatial dimensions.
    /// 
    /// Think of it as:
    /// - Unfolding channels into a larger image
    /// - Smart upsampling that learns how to fill in details
    /// - Efficient alternative to transposed convolutions
    /// 
    /// Used in:
    /// - Super-resolution networks
    /// - Efficient upsampling
    /// - Real-time image generation
    /// </para>
    /// </remarks>
    SubpixelConvolutional,

    /// <summary>
    /// A layer implementing synaptic plasticity for continual learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Synaptic Plasticity layers can strengthen or weaken connections 
    /// based on activity, mimicking brain learning.
    /// 
    /// Think of it as:
    /// - Connections that adapt based on use
    /// - "Neurons that fire together, wire together"
    /// - Dynamic learning rules
    /// 
    /// Enables:
    /// - Continual learning without forgetting
    /// - Adaptation to new tasks
    /// - More biological learning dynamics
    /// </para>
    /// </remarks>
    SynapticPlasticity,

    /// <summary>
    /// A layer implementing temporal memory in Hierarchical Temporal Memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Temporal Memory layers learn sequences and make predictions 
    /// based on temporal context in HTM systems.
    /// 
    /// Think of it as:
    /// - Learning patterns that unfold over time
    /// - Predicting what comes next in sequences
    /// - Memory that understands temporal relationships
    /// 
    /// Key features:
    /// - Online sequence learning
    /// - Anomaly detection
    /// - Predictive capabilities
    /// </para>
    /// </remarks>
    TemporalMemory,

    /// <summary>
    /// A wrapper that applies the same layer to each time step of a sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimeDistributed layers apply the same operation independently 
    /// to each time step in a sequence.
    /// 
    /// Think of it as:
    /// - Applying the same filter to each frame of a video
    /// - Processing each word in a sentence with the same network
    /// - Reusing layers across time
    /// 
    /// Useful for:
    /// - Video processing (same CNN for each frame)
    /// - Sequence-to-sequence models
    /// - Any time-distributed operation
    /// </para>
    /// </remarks>
    TimeDistributed,

    /// <summary>
    /// A decoder layer in transformer architectures with masked self-attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transformer Decoder layers generate outputs using attention 
    /// to both the input and previously generated outputs.
    /// 
    /// Think of it as:
    /// - A translator that looks at source text and what it's written so far
    /// - Autoregressive generation with attention
    /// - The output-generating part of transformers
    /// 
    /// Features:
    /// - Masked self-attention (can't see future)
    /// - Cross-attention to encoder output
    /// - Used in GPT, machine translation
    /// </para>
    /// </remarks>
    TransformerDecoder,

    /// <summary>
    /// An encoder layer in transformer architectures with full self-attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transformer Encoder layers process inputs using self-attention 
    /// to build rich representations.
    /// 
    /// Think of it as:
    /// - Understanding text by having each word look at all other words
    /// - Building context-aware representations
    /// - The understanding part of transformers
    /// 
    /// Components:
    /// - Multi-head self-attention
    /// - Feed-forward network
    /// - Layer normalization and residual connections
    /// </para>
    /// </remarks>
    TransformerEncoder,

    /// <summary>
    /// A layer that increases spatial dimensions through interpolation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Upsampling layers increase the size of inputs, like enlarging 
    /// a small image to a bigger one.
    /// 
    /// Think of it as:
    /// - Zooming in on an image
    /// - Filling in values between existing ones
    /// - Reversing the effect of pooling
    /// 
    /// Methods include:
    /// - Nearest neighbor (repeat values)
    /// - Bilinear interpolation (smooth transitions)
    /// - Learned upsampling
    /// </para>
    /// </remarks>
    Upsampling
}