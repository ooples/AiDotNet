namespace AiDotNet.Enums;

/// <summary>
/// Defines the different types of tasks that a neural network can be designed to perform.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Neural networks can solve many different types of problems. This enum lists 
/// the common problem types that neural networks can tackle. Think of it as a menu of what 
/// your AI can do - from sorting things into categories (classification), to predicting numbers 
/// (regression), to understanding images or text. Choosing the right task type helps the library 
/// set up the appropriate neural network structure for your specific problem.
/// </para>
/// </remarks>
public enum NeuralNetworkTaskType
{
    /// <summary>
    /// Binary classification task (two classes)
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Binary classification is about deciding between two options - yes/no, 
    /// spam/not spam, fraud/legitimate, etc. The neural network learns to make this two-way 
    /// decision based on the input data. For example, determining if an email is spam or not spam.
    /// </para>
    /// </remarks>
    BinaryClassification,

    /// <summary>
    /// Multi-class classification task (more than two classes)
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-class classification sorts items into one of several categories. 
    /// Unlike binary classification which has only two options, this can have many possible 
    /// categories. For example, classifying a photo as containing a dog, cat, bird, or horse 
    /// (where each image belongs to exactly one category).
    /// </para>
    /// </remarks>
    MultiClassClassification,

    /// <summary>
    /// Multi-label classification task (multiple labels can be assigned)
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-label classification allows assigning multiple categories to a single 
    /// item. Unlike multi-class classification where an item belongs to exactly one category, 
    /// here an item can belong to several categories simultaneously. For example, a photo might 
    /// contain both a dog AND a cat, so it would get both labels.
    /// </para>
    /// </remarks>
    MultiLabelClassification,

    /// <summary>
    /// Regression task (predicting continuous values)
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Regression predicts a number value rather than a category. While classification 
    /// is like sorting items into labeled boxes, regression is like placing items on a number line. 
    /// Examples include predicting house prices, temperature forecasts, or a person's age from a photo.
    /// </para>
    /// </remarks>
    Regression,

    /// <summary>
    /// Sequence-to-sequence task (e.g., machine translation)
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sequence-to-sequence tasks convert one sequence of items into another sequence. 
    /// Think of it as transforming a list of things into a different list. The most common example 
    /// is language translation, where a sequence of words in English becomes a sequence of words in 
    /// Spanish. Other examples include converting speech to text or summarizing text.
    /// </para>
    /// </remarks>
    SequenceToSequence,

    /// <summary>
    /// Sequence-to-sequence classification
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This task analyzes a sequence of data (like words in a sentence or time-ordered 
    /// events) and assigns a category to the entire sequence. For example, determining the sentiment 
    /// (positive/negative) of a movie review, or classifying a series of user actions as normal or suspicious.
    /// </para>
    /// </remarks>
    SequenceClassification,

    /// <summary>
    /// Time series forecasting task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Time series forecasting predicts future values based on past observations that 
    /// occur in a specific time order. It's like looking at historical patterns to predict what comes 
    /// next. Examples include stock price prediction, weather forecasting, or predicting website traffic 
    /// for the coming week based on previous weeks' data.
    /// </para>
    /// </remarks>
    TimeSeriesForecasting,

    /// <summary>
    /// Image classification task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Image classification identifies what's in an image by assigning it to one or more 
    /// categories. The neural network learns to recognize visual patterns that distinguish different objects. 
    /// Examples include identifying whether a photo contains a dog, cat, or bird, or determining if a 
    /// medical image shows signs of a disease.
    /// </para>
    /// </remarks>
    ImageClassification,

    /// <summary>
    /// Object detection task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Object detection not only identifies what objects are in an image but also locates 
    /// where they are by drawing boxes around them. Unlike simple image classification, it can find multiple 
    /// objects in a single image. For example, identifying and locating all people, cars, and traffic signs 
    /// in a street photo.
    /// </para>
    /// </remarks>
    ObjectDetection,

    /// <summary>
    /// Image segmentation task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Image segmentation goes beyond object detection by identifying exactly which pixels 
    /// in an image belong to each object. Instead of just drawing boxes around objects, it creates a 
    /// detailed outline of each object - like tracing their exact shapes. This is useful for applications 
    /// like medical imaging where precise boundaries matter.
    /// </para>
    /// </remarks>
    ImageSegmentation,

    /// <summary>
    /// Natural language processing task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Natural Language Processing (NLP) helps computers understand, interpret, and generate 
    /// human language. This is a broad category covering many text-related tasks. Examples include understanding 
    /// the meaning of sentences, answering questions, or determining the relationships between words.
    /// </para>
    /// </remarks>
    NaturalLanguageProcessing,

    /// <summary>
    /// Text generation task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Text generation creates new text content that resembles human writing. The neural 
    /// network learns patterns from existing text and then produces new text with similar characteristics. 
    /// Examples include writing stories, completing sentences, generating product descriptions, or creating 
    /// chatbot responses.
    /// </para>
    /// </remarks>
    TextGeneration,

    /// <summary>
    /// Reinforcement learning task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reinforcement learning is about teaching an AI to make sequences of decisions by 
    /// rewarding good outcomes. Unlike other approaches that learn from examples, reinforcement learning 
    /// learns through trial and error - like training a pet with treats. Examples include teaching AI to 
    /// play games, control robots, or optimize resource allocation.
    /// </para>
    /// </remarks>
    ReinforcementLearning,

    /// <summary>
    /// Anomaly detection task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Anomaly detection identifies unusual patterns that don't conform to expected behavior. 
    /// It's like finding the odd one out or spotting something suspicious. Examples include detecting fraud 
    /// in credit card transactions, identifying manufacturing defects, or finding unusual network traffic 
    /// that might indicate a security breach.
    /// </para>
    /// </remarks>
    AnomalyDetection,

    /// <summary>
    /// Recommendation system task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Recommendation systems suggest items to users based on their preferences or behavior. 
    /// They learn patterns from past interactions to predict what you might like. Examples include movie 
    /// recommendations on Netflix, product suggestions on Amazon, or "People You May Know" on social media.
    /// </para>
    /// </remarks>
    Recommendation,

    /// <summary>
    /// Clustering task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Clustering groups similar items together without being told in advance what the groups 
    /// should be. Unlike classification where categories are predefined, clustering discovers natural groupings 
    /// in data. Examples include grouping customers with similar buying habits, organizing news articles by topic, 
    /// or identifying different usage patterns in an app.
    /// </para>
    /// </remarks>
    Clustering,

    /// <summary>
    /// Dimensionality reduction task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dimensionality reduction simplifies complex data while preserving its important characteristics. 
    /// It's like creating a simplified map that still shows the main features of a landscape. This helps with 
    /// visualization, speeds up learning, and reduces storage needs. For example, compressing a large set of 
    /// customer attributes down to the few most meaningful factors.
    /// </para>
    /// </remarks>
    DimensionalityReduction,

    /// <summary>
    /// Generative task (e.g., GANs, VAEs, image synthesis)
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generative tasks create new data that resembles existing data. The neural network learns
    /// the patterns and structure of the training data, then generates new examples that look like they could
    /// have been part of the original dataset. Examples include creating realistic images of faces that don't
    /// exist, generating music, or synthesizing speech. For text generation specifically, use the
    /// <see cref="TextGeneration"/> task type instead.
    /// </para>
    /// </remarks>
    Generative,

    /// <summary>
    /// Speech recognition task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Speech recognition converts spoken language into written text. The neural network learns 
    /// to identify the words and phrases in audio recordings. Examples include voice assistants like Siri or 
    /// Alexa, transcription services, or voice-controlled interfaces.
    /// </para>
    /// </remarks>
    SpeechRecognition,

    /// <summary>
    /// Audio processing task
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Audio processing analyzes or modifies sound data beyond just recognizing speech. This 
    /// includes identifying sounds (like detecting a baby crying), separating mixed audio sources (like isolating 
    /// a voice from background noise), or enhancing audio quality. Examples include noise cancellation, music 
    /// analysis, or environmental sound monitoring.
    /// </para>
    /// </remarks>
    AudioProcessing,

    /// <summary>
    /// Language translation
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Language translation converts text from one language to another while preserving the meaning. 
    /// The neural network learns the patterns and relationships between languages to make accurate translations. 
    /// Examples include translating websites, documents, or conversations between different languages like English 
    /// to Spanish or Japanese to French.
    /// </para>
    /// </remarks>
    Translation,

    /// <summary>
    /// Custom task type
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Custom option allows you to define your own specialized task that doesn't fit neatly 
    /// into the predefined categories. This gives you flexibility to create neural networks for unique or 
    /// experimental purposes that combine aspects of different task types or implement entirely new approaches.
    /// </para>
    /// </remarks>
    Custom
}
