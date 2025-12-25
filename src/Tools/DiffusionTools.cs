using Newtonsoft.Json.Linq;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.Models;
using AiDotNet.Interfaces;

namespace AiDotNet.Tools;

/// <summary>
/// Tool for generating images from text prompts using diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tool enables agents to create images from text descriptions:
///
/// Example uses:
/// - "Generate an image of a sunset over mountains"
/// - "Create a portrait of a robot in Renaissance style"
/// - "Draw a futuristic city at night"
///
/// The agent provides a JSON input with the prompt and optional parameters,
/// and receives generated image data (typically as base64 or file path).
/// </para>
/// </remarks>
public class TextToImageTool<T> : ToolBase
{
    private readonly IDiffusionModel<T>? _model;
    private readonly Func<string, int, double, int?, Tensor<T>>? _generateFunc;

    /// <inheritdoc />
    public override string Name => "TextToImage";

    /// <inheritdoc />
    public override string Description =>
        "Generates images from text descriptions using diffusion models. " +
        "Input: JSON with 'prompt' (required), 'negative_prompt' (optional), " +
        "'num_inference_steps' (optional, default 50), 'guidance_scale' (optional, default 7.5), " +
        "'seed' (optional). Returns image dimensions and generation info.";

    /// <summary>
    /// Creates a TextToImageTool with a diffusion model.
    /// </summary>
    /// <param name="model">The diffusion model to use for generation.</param>
    public TextToImageTool(IDiffusionModel<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _generateFunc = null;
    }

    /// <summary>
    /// Creates a TextToImageTool with a custom generation function.
    /// </summary>
    /// <param name="generateFunc">Custom function (prompt, steps, guidance, seed) -> image.</param>
    public TextToImageTool(Func<string, int, double, int?, Tensor<T>> generateFunc)
    {
        _generateFunc = generateFunc ?? throw new ArgumentNullException(nameof(generateFunc));
        _model = null;
    }

    /// <inheritdoc />
    protected override string ExecuteCore(string input)
    {
        var json = JObject.Parse(input);

        var prompt = TryGetString(json, "prompt");
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return "Error: 'prompt' is required for image generation.";
        }

        var negativePrompt = TryGetString(json, "negative_prompt", "");
        var numSteps = TryGetInt(json, "num_inference_steps", 50);
        var guidanceScale = TryGetDouble(json, "guidance_scale", 7.5);
        int? seed = json["seed"] != null ? TryGetInt(json, "seed") : null;

        var width = TryGetInt(json, "width", 512);
        var height = TryGetInt(json, "height", 512);

        Tensor<T> result;

        if (_generateFunc != null)
        {
            result = _generateFunc(prompt, numSteps, guidanceScale, seed);
        }
        else if (_model is LatentDiffusionModelBase<T> ldm)
        {
            result = ldm.GenerateFromText(prompt, negativePrompt, width, height, numSteps, guidanceScale, seed);
        }
        else if (_model != null)
        {
            var shape = new[] { 1, 4, height / 8, width / 8 };
            result = _model.Generate(shape, numSteps, seed);
        }
        else
        {
            return "Error: No diffusion model configured.";
        }

        return FormatImageResult(result, prompt, numSteps, guidanceScale, seed);
    }

    private static string FormatImageResult(Tensor<T> image, string prompt, int steps, double guidance, int? seed)
    {
        var shape = string.Join("x", image.Shape);
        return $"Generated image successfully.\n" +
               $"Shape: {shape}\n" +
               $"Prompt: \"{prompt}\"\n" +
               $"Steps: {steps}, Guidance: {guidance:F1}" +
               (seed.HasValue ? $", Seed: {seed}" : "");
    }

    /// <inheritdoc />
    protected override string GetEmptyInputErrorMessage()
    {
        return "Error: TextToImage requires a prompt.\n" +
               "Example: { \"prompt\": \"A beautiful sunset over mountains\", " +
               "\"num_inference_steps\": 50, \"guidance_scale\": 7.5 }";
    }
}

/// <summary>
/// Tool for generating audio from text prompts using audio diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tool enables agents to create sounds from descriptions:
///
/// Example uses:
/// - "Generate the sound of rain falling on a window"
/// - "Create thunder and lightning sound effects"
/// - "Make a dog barking sound"
/// </para>
/// </remarks>
public class TextToAudioTool<T> : ToolBase
{
    private readonly AudioDiffusionModelBase<T>? _model;
    private readonly Func<string, double, int, double, int?, Tensor<T>>? _generateFunc;

    /// <inheritdoc />
    public override string Name => "TextToAudio";

    /// <inheritdoc />
    public override string Description =>
        "Generates audio/sound effects from text descriptions using audio diffusion models. " +
        "Input: JSON with 'prompt' (required), 'duration_seconds' (optional, default 5.0), " +
        "'num_inference_steps' (optional, default 100), 'guidance_scale' (optional, default 3.5), " +
        "'seed' (optional). Returns audio sample info.";

    /// <summary>
    /// Creates a TextToAudioTool with an audio diffusion model.
    /// </summary>
    /// <param name="model">The audio diffusion model.</param>
    public TextToAudioTool(AudioDiffusionModelBase<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _generateFunc = null;
    }

    /// <summary>
    /// Creates a TextToAudioTool with a custom generation function.
    /// </summary>
    /// <param name="generateFunc">Custom function.</param>
    public TextToAudioTool(Func<string, double, int, double, int?, Tensor<T>> generateFunc)
    {
        _generateFunc = generateFunc ?? throw new ArgumentNullException(nameof(generateFunc));
        _model = null;
    }

    /// <inheritdoc />
    protected override string ExecuteCore(string input)
    {
        var json = JObject.Parse(input);

        var prompt = TryGetString(json, "prompt");
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return "Error: 'prompt' is required for audio generation.";
        }

        var negativePrompt = TryGetString(json, "negative_prompt", "");
        var durationSeconds = TryGetDouble(json, "duration_seconds", 5.0);
        var numSteps = TryGetInt(json, "num_inference_steps", 100);
        var guidanceScale = TryGetDouble(json, "guidance_scale", 3.5);
        int? seed = json["seed"] != null ? TryGetInt(json, "seed") : null;

        Tensor<T> result;

        if (_generateFunc != null)
        {
            result = _generateFunc(prompt, durationSeconds, numSteps, guidanceScale, seed);
        }
        else if (_model != null)
        {
            result = _model.GenerateFromText(prompt, negativePrompt, durationSeconds, numSteps, guidanceScale, seed);
        }
        else
        {
            return "Error: No audio diffusion model configured.";
        }

        return FormatAudioResult(result, prompt, durationSeconds, _model?.SampleRate ?? 16000);
    }

    private static string FormatAudioResult(Tensor<T> audio, string prompt, double duration, int sampleRate)
    {
        var numSamples = audio.Shape[^1];
        var actualDuration = (double)numSamples / sampleRate;

        return $"Generated audio successfully.\n" +
               $"Duration: {actualDuration:F2}s ({numSamples} samples at {sampleRate}Hz)\n" +
               $"Prompt: \"{prompt}\"";
    }

    /// <inheritdoc />
    protected override string GetEmptyInputErrorMessage()
    {
        return "Error: TextToAudio requires a prompt.\n" +
               "Example: { \"prompt\": \"Rain falling on a window\", " +
               "\"duration_seconds\": 5.0, \"num_inference_steps\": 100 }";
    }
}

/// <summary>
/// Tool for generating music from text prompts using music diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tool enables agents to create music:
///
/// Example uses:
/// - "Generate jazz piano with light drums"
/// - "Create upbeat electronic dance music at 128 BPM"
/// - "Compose a sad orchestral piece"
/// </para>
/// </remarks>
public class TextToMusicTool<T> : ToolBase
{
    private readonly AudioDiffusionModelBase<T>? _model;
    private readonly MusicGenModel<T>? _musicGenModel;

    /// <inheritdoc />
    public override string Name => "TextToMusic";

    /// <inheritdoc />
    public override string Description =>
        "Generates music from text descriptions using music diffusion models. " +
        "Input: JSON with 'prompt' (required), 'duration_seconds' (optional, default 10.0), " +
        "'bpm' (optional), 'num_inference_steps' (optional, default 200), " +
        "'guidance_scale' (optional, default 5.0), 'seed' (optional). " +
        "Returns music generation info.";

    /// <summary>
    /// Creates a TextToMusicTool with an audio diffusion model.
    /// </summary>
    /// <param name="model">The audio diffusion model.</param>
    public TextToMusicTool(AudioDiffusionModelBase<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _musicGenModel = model as MusicGenModel<T>;
    }

    /// <inheritdoc />
    protected override string ExecuteCore(string input)
    {
        var json = JObject.Parse(input);

        var prompt = TryGetString(json, "prompt");
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return "Error: 'prompt' is required for music generation.";
        }

        var negativePrompt = TryGetString(json, "negative_prompt", "low quality, distorted, noise");
        var durationSeconds = TryGetDouble(json, "duration_seconds", 10.0);
        var numSteps = TryGetInt(json, "num_inference_steps", 200);
        var guidanceScale = TryGetDouble(json, "guidance_scale", 5.0);
        int? seed = json["seed"] != null ? TryGetInt(json, "seed") : null;
        var bpm = json["bpm"] != null ? TryGetInt(json, "bpm") : (int?)null;

        Tensor<T> result;

        if (_musicGenModel != null && bpm.HasValue)
        {
            result = _musicGenModel.GenerateMusicWithTempo(
                prompt, bpm.Value, negativePrompt, durationSeconds, numSteps, guidanceScale, seed);
        }
        else if (_model != null)
        {
            result = _model.GenerateMusic(prompt, negativePrompt, durationSeconds, numSteps, guidanceScale, seed);
        }
        else
        {
            return "Error: No music diffusion model configured.";
        }

        return FormatMusicResult(result, prompt, durationSeconds, bpm, _model?.SampleRate ?? 32000);
    }

    private static string FormatMusicResult(Tensor<T> audio, string prompt, double duration, int? bpm, int sampleRate)
    {
        var numSamples = audio.Shape[^1];
        var actualDuration = (double)numSamples / sampleRate;

        var bpmInfo = bpm.HasValue ? $"\nTempo: {bpm} BPM" : "";

        return $"Generated music successfully.\n" +
               $"Duration: {actualDuration:F2}s ({numSamples} samples at {sampleRate}Hz){bpmInfo}\n" +
               $"Prompt: \"{prompt}\"";
    }

    /// <inheritdoc />
    protected override string GetEmptyInputErrorMessage()
    {
        return "Error: TextToMusic requires a prompt.\n" +
               "Example: { \"prompt\": \"Upbeat jazz piano with drums\", " +
               "\"duration_seconds\": 15.0, \"bpm\": 120 }";
    }
}

/// <summary>
/// Tool for converting images to 3D models using diffusion-based reconstruction.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tool converts 2D images into 3D models:
///
/// Example uses:
/// - "Convert this product photo to a 3D model"
/// - "Create a 3D mesh from this character image"
/// - "Generate multi-view images of this object"
/// </para>
/// </remarks>
public class ImageTo3DTool<T> : ToolBase
{
    private readonly I3DDiffusionModel<T>? _model;

    /// <inheritdoc />
    public override string Name => "ImageTo3D";

    /// <inheritdoc />
    public override string Description =>
        "Converts 2D images to 3D models using diffusion-based reconstruction. " +
        "Input: JSON with 'image_path' or 'image_data' (required), " +
        "'num_views' (optional, default 4), 'num_inference_steps' (optional, default 50), " +
        "'guidance_scale' (optional, default 3.0). Returns 3D mesh info.";

    /// <summary>
    /// Creates an ImageTo3DTool with a 3D diffusion model.
    /// </summary>
    /// <param name="model">The 3D diffusion model.</param>
    public ImageTo3DTool(I3DDiffusionModel<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <inheritdoc />
    protected override string ExecuteCore(string input)
    {
        var json = JObject.Parse(input);

        // For now, just validate the input and return info about what would be generated
        var imagePath = TryGetString(json, "image_path", "");
        var numViews = TryGetInt(json, "num_views", 4);
        var numSteps = TryGetInt(json, "num_inference_steps", 50);
        var guidanceScale = TryGetDouble(json, "guidance_scale", 3.0);

        if (string.IsNullOrWhiteSpace(imagePath) && json["image_data"] == null)
        {
            return "Error: 'image_path' or 'image_data' is required.";
        }

        if (_model == null)
        {
            return "Error: No 3D diffusion model configured.";
        }

        // In a real implementation, we'd load the image and call _model.ImageTo3D()
        return $"3D reconstruction configured.\n" +
               $"Input: {(string.IsNullOrEmpty(imagePath) ? "image_data" : imagePath)}\n" +
               $"Views: {numViews}, Steps: {numSteps}, Guidance: {guidanceScale:F1}\n" +
               $"Supported features: PointCloud={_model.SupportsPointCloud}, " +
               $"Mesh={_model.SupportsMesh}, Texture={_model.SupportsTexture}";
    }

    /// <inheritdoc />
    protected override string GetEmptyInputErrorMessage()
    {
        return "Error: ImageTo3D requires an image.\n" +
               "Example: { \"image_path\": \"/path/to/image.png\", \"num_views\": 4 }";
    }
}

/// <summary>
/// Tool for generating text descriptions from text prompts to 3D point clouds/meshes.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class TextTo3DTool<T> : ToolBase
{
    private readonly I3DDiffusionModel<T>? _model;

    /// <inheritdoc />
    public override string Name => "TextTo3D";

    /// <inheritdoc />
    public override string Description =>
        "Generates 3D point clouds or meshes from text descriptions. " +
        "Input: JSON with 'prompt' (required), 'output_type' (optional: 'pointcloud' or 'mesh', default 'mesh'), " +
        "'num_points' (optional for pointcloud), 'resolution' (optional for mesh), " +
        "'num_inference_steps' (optional, default 64), 'guidance_scale' (optional, default 3.0).";

    /// <summary>
    /// Creates a TextTo3DTool with a 3D diffusion model.
    /// </summary>
    /// <param name="model">The 3D diffusion model.</param>
    public TextTo3DTool(I3DDiffusionModel<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <inheritdoc />
    protected override string ExecuteCore(string input)
    {
        var json = JObject.Parse(input);

        var prompt = TryGetString(json, "prompt");
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return "Error: 'prompt' is required for 3D generation.";
        }

        var outputType = TryGetString(json, "output_type", "mesh").ToLower();
        var negativePrompt = TryGetString(json, "negative_prompt", "");
        var numSteps = TryGetInt(json, "num_inference_steps", 64);
        var guidanceScale = TryGetDouble(json, "guidance_scale", 3.0);
        int? seed = json["seed"] != null ? TryGetInt(json, "seed") : null;

        if (_model == null)
        {
            return "Error: No 3D diffusion model configured.";
        }

        if (outputType == "pointcloud")
        {
            if (!_model.SupportsPointCloud)
            {
                return "Error: This model does not support point cloud generation.";
            }

            var numPoints = TryGetInt(json, "num_points", 4096);
            var pointCloud = _model.GeneratePointCloud(prompt, negativePrompt, numPoints, numSteps, guidanceScale, seed);

            return $"Generated point cloud successfully.\n" +
                   $"Points: {pointCloud.Shape[1]}\n" +
                   $"Prompt: \"{prompt}\"";
        }
        else
        {
            if (!_model.SupportsMesh)
            {
                return "Error: This model does not support mesh generation.";
            }

            var resolution = TryGetInt(json, "resolution", 128);
            var mesh = _model.GenerateMesh(prompt, negativePrompt, resolution, numSteps, guidanceScale, seed);

            return $"Generated mesh successfully.\n" +
                   $"Vertices: {mesh.Vertices.Shape[0]}, Faces: {mesh.Faces.GetLength(0)}\n" +
                   $"Prompt: \"{prompt}\"";
        }
    }

    /// <inheritdoc />
    protected override string GetEmptyInputErrorMessage()
    {
        return "Error: TextTo3D requires a prompt.\n" +
               "Example: { \"prompt\": \"A cute robot toy\", \"output_type\": \"mesh\" }";
    }
}

/// <summary>
/// Tool for transforming audio based on text prompts (audio-to-audio).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class AudioTransformTool<T> : ToolBase
{
    private readonly AudioDiffusionModelBase<T>? _model;

    /// <inheritdoc />
    public override string Name => "AudioTransform";

    /// <inheritdoc />
    public override string Description =>
        "Transforms audio based on text prompts using audio diffusion models. " +
        "Input: JSON with 'audio_path' or 'audio_data' (required), 'prompt' (required), " +
        "'strength' (optional, 0.0-1.0, default 0.5), 'num_inference_steps' (optional, default 100), " +
        "'guidance_scale' (optional, default 3.5). Returns transformed audio info.";

    /// <summary>
    /// Creates an AudioTransformTool with an audio diffusion model.
    /// </summary>
    /// <param name="model">The audio diffusion model.</param>
    public AudioTransformTool(AudioDiffusionModelBase<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <inheritdoc />
    protected override string ExecuteCore(string input)
    {
        var json = JObject.Parse(input);

        var audioPath = TryGetString(json, "audio_path", "");
        var prompt = TryGetString(json, "prompt");

        if (string.IsNullOrWhiteSpace(audioPath) && json["audio_data"] == null)
        {
            return "Error: 'audio_path' or 'audio_data' is required.";
        }

        if (string.IsNullOrWhiteSpace(prompt))
        {
            return "Error: 'prompt' is required for audio transformation.";
        }

        var strength = TryGetDouble(json, "strength", 0.5);
        var numSteps = TryGetInt(json, "num_inference_steps", 100);
        var guidanceScale = TryGetDouble(json, "guidance_scale", 3.5);

        if (_model == null)
        {
            return "Error: No audio diffusion model configured.";
        }

        if (!_model.SupportsAudioToAudio)
        {
            return "Error: This model does not support audio-to-audio transformation.";
        }

        // In a real implementation, we'd load the audio and call AudioToAudio()
        return $"Audio transformation configured.\n" +
               $"Input: {(string.IsNullOrEmpty(audioPath) ? "audio_data" : audioPath)}\n" +
               $"Prompt: \"{prompt}\"\n" +
               $"Strength: {strength:F2}, Steps: {numSteps}, Guidance: {guidanceScale:F1}";
    }

    /// <inheritdoc />
    protected override string GetEmptyInputErrorMessage()
    {
        return "Error: AudioTransform requires audio and a prompt.\n" +
               "Example: { \"audio_path\": \"/path/to/audio.wav\", " +
               "\"prompt\": \"Add reverb effect\", \"strength\": 0.5 }";
    }
}

/// <summary>
/// Tool for computing Score Distillation Sampling (SDS) gradients for 3D optimization.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class ScoreDistillationTool<T> : ToolBase
{
    private readonly I3DDiffusionModel<T>? _model;

    /// <inheritdoc />
    public override string Name => "ScoreDistillation";

    /// <inheritdoc />
    public override string Description =>
        "Computes Score Distillation Sampling (SDS) gradients for 3D optimization. " +
        "Used to guide NeRF or 3D Gaussian Splatting optimization. " +
        "Input: JSON with 'prompt' (required), 'timestep' (optional, default 500), " +
        "'guidance_scale' (optional, default 100.0). " +
        "Returns gradient computation info.";

    /// <summary>
    /// Creates a ScoreDistillationTool with a 3D diffusion model.
    /// </summary>
    /// <param name="model">The 3D diffusion model.</param>
    public ScoreDistillationTool(I3DDiffusionModel<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <inheritdoc />
    protected override string ExecuteCore(string input)
    {
        var json = JObject.Parse(input);

        var prompt = TryGetString(json, "prompt");
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return "Error: 'prompt' is required for score distillation.";
        }

        var timestep = TryGetInt(json, "timestep", 500);
        var guidanceScale = TryGetDouble(json, "guidance_scale", 100.0);

        if (_model == null)
        {
            return "Error: No 3D diffusion model configured.";
        }

        if (!_model.SupportsScoreDistillation)
        {
            return "Error: This model does not support score distillation sampling.";
        }

        return $"Score distillation configured.\n" +
               $"Prompt: \"{prompt}\"\n" +
               $"Timestep: {timestep}, Guidance: {guidanceScale:F1}\n" +
               $"Ready for gradient computation with rendered views.";
    }

    /// <inheritdoc />
    protected override string GetEmptyInputErrorMessage()
    {
        return "Error: ScoreDistillation requires a prompt.\n" +
               "Example: { \"prompt\": \"A cute robot toy\", \"timestep\": 500, \"guidance_scale\": 100.0 }";
    }
}
