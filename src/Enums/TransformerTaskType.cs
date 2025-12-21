namespace AiDotNet.Enums;

/// <summary>
/// Defines the different types of tasks that transformer-based AI models can perform.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Transformers are a type of neural network architecture that has revolutionized AI, 
/// especially in natural language processing. They're the technology behind models like GPT, BERT, and T5.
/// 
/// This enum lists the common tasks these models can perform. Think of these as different "jobs" 
/// you can assign to an AI model based on what you want it to accomplish.
/// </remarks>
public enum TransformerTaskType
{
    /// <summary>
    /// Task of categorizing input data into predefined classes or categories.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Classification is about sorting things into categories.
    /// 
    /// Imagine you have a bunch of emails and you want the AI to sort them into "Spam" or "Not Spam" - 
    /// that's classification. The AI looks at the content and decides which category it belongs to.
    /// 
    /// Other examples include:
    /// - Sentiment analysis (Is this review positive, negative, or neutral?)
    /// - Topic categorization (Is this news article about sports, politics, or entertainment?)
    /// - Intent detection (Is the user asking a question, making a request, or giving feedback?)
    /// 
    /// Classification models output the probability of the input belonging to each possible category.
    /// </remarks>
    Classification,

    /// <summary>
    /// Task of predicting continuous numerical values based on input features.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Regression is about predicting numbers rather than categories.
    /// 
    /// While classification predicts categories (like "Spam" or "Not Spam"), regression predicts 
    /// numerical values. For example:
    /// - Predicting the price of a house based on its features
    /// - Estimating how many views a video will get
    /// - Forecasting temperature for tomorrow
    /// 
    /// Regression models look at the input and output a specific number (or set of numbers) as a prediction.
    /// 
    /// In transformer models, regression tasks might involve analyzing text to predict associated 
    /// numerical values, like predicting a movie's box office earnings based on its synopsis.
    /// </remarks>
    Regression,

    /// <summary>
    /// Task of generating coherent and contextually relevant text based on a prompt or input.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Text Generation is about creating new text that continues or responds to some input.
    /// 
    /// This is what models like ChatGPT do - you provide some text (a prompt), and the model generates 
    /// new text that follows naturally from what you provided. The model predicts what words should 
    /// come next, one token at a time.
    /// 
    /// Examples include:
    /// - Chatbots that respond to user messages
    /// - Story generators that continue a narrative
    /// - Content creation tools that help write articles or emails
    /// - Code completion systems that suggest the next lines of code
    /// 
    /// Text generation models are trained to understand context and produce coherent, relevant content 
    /// that matches the style and intent of the prompt.
    /// </remarks>
    TextGeneration,

    /// <summary>
    /// Task of labeling each element in a sequence with a specific tag or category.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Sequence Tagging is about labeling each word or piece in a text with a specific tag.
    /// 
    /// Unlike classification which assigns one label to an entire input, sequence tagging assigns 
    /// a label to each element in the sequence. It's like going through a sentence with a highlighter 
    /// and marking each word according to its role.
    /// 
    /// Common examples include:
    /// - Named Entity Recognition: Identifying names of people, organizations, locations, etc. in text
    /// - Part-of-Speech Tagging: Labeling words as nouns, verbs, adjectives, etc.
    /// - Chunking: Identifying phrases like noun phrases or verb phrases
    /// 
    /// For instance, in the sentence "Apple is launching a new iPhone in September":
    /// - "Apple" might be tagged as ORGANIZATION
    /// - "iPhone" as PRODUCT
    /// - "September" as DATE
    /// </remarks>
    SequenceTagging,

    /// <summary>
    /// Task of converting text from one language to another while preserving meaning.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Translation is about converting text from one language to another.
    /// 
    /// This task involves understanding the meaning of text in a source language and expressing 
    /// that same meaning in a target language. Modern transformer-based translation systems don't 
    /// just replace words one-by-one; they understand context and cultural nuances.
    /// 
    /// Translation models need to:
    /// - Understand the source language's grammar, idioms, and context
    /// - Generate fluent, natural-sounding text in the target language
    /// - Preserve the original meaning, tone, and intent
    /// 
    /// Examples include translating:
    /// - Website content for international audiences
    /// - Documents and books into different languages
    /// - Real-time conversation in multilingual settings
    /// 
    /// Modern translation models can often handle dozens or even hundreds of language pairs.
    /// </remarks>
    Translation
}
