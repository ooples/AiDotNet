namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// The Florence-2 tasks. Each is a task prompt the language encoder reads after the image tokens.
/// </summary>
/// <remarks>
/// The prompts are the ones the Florence-2 processor substitutes for its task tokens (HF
/// <c>Florence2Processor.task_prompts_without_inputs</c> and <c>task_prompts_with_input</c>). Tasks marked
/// "with input" take a phrase or region description; region inputs are written as location tokens.
/// </remarks>
public enum Florence2Task
{
    /// <summary>&lt;OCR&gt;: "What is the text in the image?"</summary>
    Ocr,

    /// <summary>&lt;OCR_WITH_REGION&gt;: "What is the text in the image, with regions?"</summary>
    OcrWithRegion,

    /// <summary>&lt;CAPTION&gt;: "What does the image describe?"</summary>
    Caption,

    /// <summary>&lt;DETAILED_CAPTION&gt;: "Describe in detail what is shown in the image."</summary>
    DetailedCaption,

    /// <summary>&lt;MORE_DETAILED_CAPTION&gt;: "Describe with a paragraph what is shown in the image."</summary>
    MoreDetailedCaption,

    /// <summary>&lt;OD&gt;: "Locate the objects with category name in the image."</summary>
    ObjectDetection,

    /// <summary>&lt;DENSE_REGION_CAPTION&gt;: "Locate the objects in the image, with their descriptions."</summary>
    DenseRegionCaption,

    /// <summary>&lt;REGION_PROPOSAL&gt;: "Locate the region proposals in the image."</summary>
    RegionProposal,

    /// <summary>&lt;CAPTION_TO_PHRASE_GROUNDING&gt; (with input): "Locate the phrases in the caption: {input}"</summary>
    CaptionToPhraseGrounding,

    /// <summary>&lt;OPEN_VOCABULARY_DETECTION&gt; (with input): "Locate {input} in the image."</summary>
    OpenVocabularyDetection,

    /// <summary>&lt;REGION_TO_CATEGORY&gt; (with input): "What is the region {input}?"</summary>
    RegionToCategory,

    /// <summary>&lt;REGION_TO_DESCRIPTION&gt; (with input): "What does the region {input} describe?"</summary>
    RegionToDescription,

    /// <summary>&lt;REGION_TO_OCR&gt; (with input): "What text is in the region {input}?"</summary>
    RegionToOcr
}
