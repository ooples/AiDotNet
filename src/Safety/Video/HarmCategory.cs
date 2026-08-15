namespace AiDotNet.Safety.Video;

/// <summary>
/// The six-category taxonomy of online harm for video platforms.
/// </summary>
/// <remarks>
/// <para>
/// From "Harmful YouTube Video Detection: A Taxonomy of Online Harm and MLLMs as Alternative
/// Annotators" (arXiv:2411.05854). The taxonomy was built with a grounded-theory approach, reviewing
/// existing harm taxonomies and the community guidelines of YouTube, Meta and TikTok, then
/// "synthesizing, converging, and reorganizing the subcategories in existing taxonomies and platform
/// policies".
/// </para>
/// <para>
/// It rests on three stated principles: every category must be discernible from the multimodal
/// content itself (text, audio, visual); a single piece of content may fall under SEVERAL
/// categories, so the set is NON-MUTUALLY EXCLUSIVE; and categories are chosen to maximise objective
/// assessment, which is why deliberately vague labels like "problematic", "extreme" or "radical"
/// are excluded.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the list of ways a video can be harmful, agreed by reading what the
/// big platforms actually forbid and what researchers have catalogued. A single video can be in more
/// than one of these at once — a video that jokes hatefully about a group while showing people being
/// hurt is both hate and physical harm.
/// </para>
/// </remarks>
public enum HarmCategory
{
    /// <summary>
    /// "the dissemination of false information that misleads and deceives people. This includes fake
    /// news, misinformation, disinformation, conspiracy theories, unverified medical treatments, and
    /// unproven scientific myths."
    /// </summary>
    Information = 0,

    /// <summary>
    /// "the promotion of hatred towards specific groups. This includes insults and obscenities,
    /// identity attacks, and hate speech based on gender, race, ethnicity, age, religion, political
    /// ideology, disability, or sexual orientation."
    /// </summary>
    HateAndHarassment = 1,

    /// <summary>
    /// "content that promotes or glorifies behaviors associated with addiction, including excessive
    /// gaming, gambling, or substance use (drugs, smoking, or alcohol)."
    /// </summary>
    Addictive = 2,

    /// <summary>
    /// "sensationalized content designed to attract clicks without delivering valuable information.
    /// This includes exaggerated headlines intended to boost click rates, unverified financial
    /// schemes, and sensational gossip or defamatory videos."
    /// </summary>
    Clickbait = 3,

    /// <summary>
    /// "explicit sexual content, including erotic scenes, depictions of sexual acts and nudity, and
    /// videos of sexual abuse, inappropriate for general audience due to their sensitive or
    /// non-consensual nature."
    /// </summary>
    Sexual = 4,

    /// <summary>
    /// "content that portrays dangerous behaviors and graphic violence, including self-injury,
    /// suicide, eating disorders, and dangerous challenges."
    /// </summary>
    Physical = 5,
}
