# Issue #271: Audio Generation via Codec Language Models Implementation Guide

## For Junior Developers: Complete Implementation Tutorial

### Table of Contents
1. [Understanding Audio Generation](#understanding-audio-generation)
2. [Neural Audio Codecs](#neural-audio-codecs)
3. [Codec Language Models](#codec-language-models)
4. [AudioGen/MusicGen Architecture](#audiogen-musicgen-architecture)
5. [Implementation Guide](#implementation-guide)
6. [Testing Strategy](#testing-strategy)
7. [Common Pitfalls](#common-pitfalls)

---

## Understanding Audio Generation

### What is AI Audio Generation?

**For Beginners:** AI audio generation creates new sounds and music from text descriptions. Instead of converting text to speech (TTS), it generates creative audio content like:
- Sound effects: "thunder rumbling", "dog barking", "car engine"
- Music: "upbeat jazz piano", "calm acoustic guitar"
- Ambient sounds: "rain on a window", "ocean waves"

**Real-world applications:**
- Game audio: Generate sound effects on-demand
- Content creation: Create background music for videos
- Sound design: Prototype audio for films/ads
- Accessibility: Generate audio descriptions

**How it differs from TTS:**

| Feature | TTS | Audio Generation |
|---------|-----|------------------|
| Input | Exact text to speak | Creative text prompt |
| Output | Speech (words) | Any sound/music |
| Goal | Clarity, intelligibility | Creativity, realism |
| Example | "Hello world" → voice | "dog barking" → woof sound |

### The Challenge

**Why is audio generation hard?**

Raw audio at 24kHz has 24,000 samples per second. For 10 seconds:
- **240,000 samples** to generate
- Each sample can be any value from -32768 to 32767
- Autoregressive generation would be painfully slow

**The solution:** Neural audio codecs

---

## Neural Audio Codecs

### What is an Audio Codec?

**For Beginners:** A codec (coder-decoder) compresses audio into a compact representation and reconstructs it.

**Traditional codecs:** MP3, AAC, Opus
- Designed by humans using signal processing
- Good compression but lossy (removes "unimportant" frequencies)

**Neural codecs:** EnCodec, SoundStream, DAC
- Learned by neural networks from data
- Better quality at same compression rate
- Designed for AI generation tasks

### EnCodec Architecture

**EnCodec** (Enhanced Neural Codec) by Meta AI (2022) is the most popular neural audio codec for generation.

**Key innovation:** Compresses 24kHz audio by 240x with minimal quality loss
- Input: 24,000 samples/second (raw audio)
- Output: 75 codes/second (compressed representation)
- Compression: 320x smaller!

**Architecture overview:**

```
Encoder: Audio → Latent Codes
Raw Audio (24kHz)
  ↓
1D Convolution Layers (downsampling)
  - Stride-2 convs reduce time resolution
  - 24kHz → 12kHz → 6kHz → 3kHz → 1.5kHz → 750Hz → 75Hz
  ↓
Residual Vector Quantization (RVQ)
  - Converts continuous values to discrete codes
  - 8 quantization levels (8 codebooks)
  ↓
Latent Codes: (75 codes/sec, 8 levels)

Decoder: Latent Codes → Audio
Latent Codes
  ↓
1D Transposed Convolution Layers (upsampling)
  - Stride-2 transposed convs increase time resolution
  - 75Hz → 750Hz → 1.5kHz → 3kHz → 6kHz → 12kHz → 24kHz
  ↓
Residual Blocks
  ↓
Raw Audio (24kHz)
```

### Residual Vector Quantization (RVQ)

**For Beginners:** RVQ is like describing a number using multiple levels of precision:

**Example: Describing 3.14159**
- Level 1: "About 3" (integer part)
- Level 2: "Plus 0.1" (first decimal)
- Level 3: "Plus 0.04" (second decimal)
- Level 4: "Plus 0.001" (third decimal)
- ...and so on

**In audio:**
- Level 1: Captures main frequency/amplitude (coarse)
- Level 2: Adds more detail (finer)
- Level 3-8: Progressively finer details

**How it works:**

1. **Quantize first level:**
   ```csharp
   Vector<T> input = encoderOutput;  // Continuous values
   int code1 = FindNearestCodebookEntry(input, codebook1);
   Vector<T> approx1 = codebook1[code1];
   Vector<T> residual1 = input - approx1;  // Error/remainder
   ```

2. **Quantize residual:**
   ```csharp
   int code2 = FindNearestCodebookEntry(residual1, codebook2);
   Vector<T> approx2 = codebook2[code2];
   Vector<T> residual2 = residual1 - approx2;
   ```

3. **Repeat for all levels** (typically 8)

**Final representation:**
```
Audio frame → [code1, code2, code3, code4, code5, code6, code7, code8]
              [325,   142,   89,    234,   56,    178,   91,    203]
```

**Benefits:**
- Each code is an integer (0-1023, depends on codebook size)
- 8 integers per frame vs thousands of floating-point samples
- Can be processed by language models (like text tokens!)

### Codebook Size

**Typical EnCodec settings:**
- **Codebook size**: 1024 entries per level
- **Levels**: 8
- **Frame rate**: 75 Hz
- **Total vocabulary**: 1024^8 (but we use 1024 × 8 = 8192 unique codes)

**Think of it like:**
- Each level has 1024 possible values (like having 1024 words in a mini-dictionary)
- We use 8 levels to describe each audio frame (8-word sentence)
- Total codes per second: 75 frames/sec × 8 levels = 600 codes/sec

---

## Codec Language Models

### The Big Idea

**For Beginners:** If audio can be represented as discrete codes (like words), we can use a language model to generate those codes!

**Text generation:**
```
Language Model: [The, cat, sat, on, the, mat] → [.]
                Previous tokens → Next token
```

**Audio generation:**
```
Language Model: [325, 142, 89, 234, ...] → [56]
                Previous audio codes → Next audio code
```

**Key insight:** Replace "word tokens" with "audio codes" and use the same transformer architecture.

### Challenge: Multi-Stream Prediction

**Problem:** EnCodec produces 8 codes per frame (8 quantization levels). How do we predict all 8?

**Solutions:**

#### 1. Flattening (AudioGen approach)
Treat 8 codes as a sequence:
```
Frame 1: [c1_1, c1_2, c1_3, c1_4, c1_5, c1_6, c1_7, c1_8]
Frame 2: [c2_1, c2_2, c2_3, ...]
```

Generate: `c1_1 → c1_2 → c1_3 → ... → c1_8 → c2_1 → c2_2 → ...`

**Pros:** Simple, uses standard language model
**Cons:** 8x slower generation (600 codes/sec instead of 75 frames/sec)

#### 2. Parallel Prediction (Advanced)
Predict all 8 levels simultaneously with special architecture.

**For this implementation, we'll use flattening** (simpler, works well).

### Language Model Architecture

**For Beginners:** The language model is a transformer (like GPT) trained to predict audio codes.

```
Text Conditioning (optional)
"a dog barking"
  ↓
Text Encoder (T5, CLAP)
  ↓
Text Embeddings

Audio Codes (previous)
[325, 142, 89, ...]
  ↓
Code Embedding Layer
  ↓
Positional Encoding
  ↓
Transformer Decoder Blocks (12-24 layers)
  - Causal Self-Attention
  - Cross-Attention to Text (conditioning)
  - Feed-Forward Network
  ↓
Output Projection
  ↓
Logits over 1024 possible codes
  ↓
Sample/Argmax → Next Code
```

**Training:** Model learns patterns in audio codes
- "Dog bark" sounds have specific code patterns
- "Piano music" has different patterns
- Text descriptions guide which patterns to generate

---

## AudioGen/MusicGen Architecture

### AudioGen (Meta AI, 2023)

**Purpose:** Generate sound effects and ambient audio from text

**Architecture:**
1. **Text Encoder**: T5 or CLAP (Contrastive Language-Audio Pre-training)
2. **Language Model**: Transformer with ~1.5B parameters
3. **Audio Codec**: EnCodec at 16kHz or 24kHz

**Training data:**
- Sound effects databases (free sound, etc.)
- Environmental sounds (AudioSet)
- ~10,000 hours of audio with text descriptions

### MusicGen (Meta AI, 2023)

**Purpose:** Generate music from text descriptions

**Similar to AudioGen but:**
- Trained on music datasets (20,000+ hours)
- Better at rhythm, harmony, melody
- Can condition on melody (hum a tune, generate full song)

### Example Prompts

**AudioGen:**
- "thunder and rain"
- "dog barking loudly"
- "crowd cheering at a stadium"
- "car engine starting"
- "footsteps on gravel"

**MusicGen:**
- "upbeat jazz piano solo"
- "calm acoustic guitar"
- "80s synthwave with drums"
- "orchestral epic trailer music"
- "lo-fi hip hop beats"

---

## Implementation Guide

### Phase 1: AudioGenModel Wrapper

**File:** `src/Models/Audio/AudioGenModel.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using System.Collections.Generic;

namespace AiDotNet.Models.Audio
{
    /// <summary>
    /// Wrapper for AudioGen/MusicGen language model + EnCodec to generate audio from text.
    /// </summary>
    /// <typeparam name="T">Numeric type for computations.</typeparam>
    /// <remarks>
    /// <b>For Beginners:</b> This class orchestrates two-stage audio generation:
    /// 1. Language Model: Text → Audio Codes (like GPT generating tokens)
    /// 2. EnCodec Decoder: Audio Codes → Raw Audio (decompression)
    ///
    /// Think of it like:
    /// - Stage 1: AI "imagines" what the sound should be (as compressed codes)
    /// - Stage 2: Decoder "renders" those codes into playable audio
    /// </remarks>
    public class AudioGenModel<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        private readonly OnnxModel<T> _languageModel;
        private readonly OnnxModel<T> _audioCodec;
        private readonly int _frameRate;
        private readonly int _numQuantizers;
        private readonly int _codebookSize;
        private readonly int _sampleRate;

        /// <summary>
        /// Creates a new AudioGen model instance.
        /// </summary>
        /// <param name="languageModelPath">Path to language model ONNX file.</param>
        /// <param name="audioCodecPath">Path to EnCodec ONNX file.</param>
        /// <param name="sampleRate">Audio sample rate (default: 24000 Hz).</param>
        /// <param name="frameRate">EnCodec frame rate (default: 75 Hz).</param>
        /// <param name="numQuantizers">Number of RVQ levels (default: 8).</param>
        /// <param name="codebookSize">Size of each codebook (default: 1024).</param>
        /// <remarks>
        /// <b>For Beginners:</b> You need TWO model files:
        /// 1. Language Model ONNX: Generates audio codes from text
        /// 2. EnCodec ONNX: Decodes codes to audio waveform
        ///
        /// <b>Default parameters explained:</b>
        /// - 24000 Hz: High-quality audio (standard for EnCodec)
        /// - 75 Hz: EnCodec generates 75 code frames per second
        /// - 8 quantizers: 8 levels of RVQ (detail levels)
        /// - 1024 codebook: Each level has 1024 possible values
        ///
        /// <b>Why these defaults:</b>
        /// These match Meta's AudioGen/MusicGen published models.
        /// 75 Hz frame rate means 320x compression (24000/75 = 320).
        /// </remarks>
        public AudioGenModel(
            string languageModelPath,
            string audioCodecPath,
            int sampleRate = 24000,
            int frameRate = 75,
            int numQuantizers = 8,
            int codebookSize = 1024)
        {
            if (string.IsNullOrWhiteSpace(languageModelPath))
                throw new ArgumentException("Language model path cannot be empty", nameof(languageModelPath));
            if (string.IsNullOrWhiteSpace(audioCodecPath))
                throw new ArgumentException("Audio codec path cannot be empty", nameof(audioCodecPath));
            if (!File.Exists(languageModelPath))
                throw new FileNotFoundException($"Language model not found: {languageModelPath}");
            if (!File.Exists(audioCodecPath))
                throw new FileNotFoundException($"Audio codec not found: {audioCodecPath}");
            if (sampleRate <= 0)
                throw new ArgumentException("Sample rate must be positive", nameof(sampleRate));
            if (frameRate <= 0)
                throw new ArgumentException("Frame rate must be positive", nameof(frameRate));
            if (numQuantizers <= 0)
                throw new ArgumentException("Number of quantizers must be positive", nameof(numQuantizers));
            if (codebookSize <= 0)
                throw new ArgumentException("Codebook size must be positive", nameof(codebookSize));

            _languageModel = new OnnxModel<T>(languageModelPath);
            _audioCodec = new OnnxModel<T>(audioCodecPath);
            _sampleRate = sampleRate;
            _frameRate = frameRate;
            _numQuantizers = numQuantizers;
            _codebookSize = codebookSize;
        }

        /// <summary>
        /// Generates audio from text prompt.
        /// </summary>
        /// <param name="textPrompt">Description of audio to generate (e.g., "dog barking").</param>
        /// <param name="durationInSeconds">Length of audio to generate (default: 5 seconds).</param>
        /// <param name="temperature">Sampling temperature for creativity (default: 1.0).</param>
        /// <param name="topK">Top-K sampling (default: 250).</param>
        /// <returns>Generated audio waveform tensor.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is the main generation method. Pass in a text description
        /// of the sound you want, and get back raw audio.
        ///
        /// <b>Example:</b>
        /// <code>
        /// var audioGen = new AudioGenModel&lt;float&gt;("lm.onnx", "codec.onnx");
        /// Tensor&lt;float&gt; bark = audioGen.Generate("a dog barking loudly", durationInSeconds: 3);
        /// AudioFileHelper.SaveWav(bark, "bark.wav", sampleRate: 24000);
        /// </code>
        ///
        /// <b>Parameters explained:</b>
        /// - <b>temperature</b>: Controls randomness
        ///   - 0.1 = Very predictable, less creative
        ///   - 1.0 = Balanced (recommended)
        ///   - 2.0 = Very random, more creative but less coherent
        ///
        /// - <b>topK</b>: Only sample from top K most likely codes
        ///   - Lower = More conservative (safer, less diverse)
        ///   - Higher = More diverse (riskier, more creative)
        ///   - 250 is a good default for audio generation
        /// </remarks>
        public Tensor<T> Generate(
            string textPrompt,
            int durationInSeconds = 5,
            double temperature = 1.0,
            int topK = 250)
        {
            if (string.IsNullOrWhiteSpace(textPrompt))
                throw new ArgumentException("Text prompt cannot be empty", nameof(textPrompt));
            if (durationInSeconds <= 0)
                throw new ArgumentException("Duration must be positive", nameof(durationInSeconds));
            if (temperature <= 0.0)
                throw new ArgumentException("Temperature must be positive", nameof(temperature));
            if (topK <= 0)
                throw new ArgumentException("Top-K must be positive", nameof(topK));

            // Step 1: Encode text prompt
            Tensor<T> textEmbedding = EncodeText(textPrompt);

            // Step 2: Calculate number of codes to generate
            int numFrames = durationInSeconds * _frameRate;
            int numCodes = numFrames * _numQuantizers;

            // Step 3: Generate audio codes autoregressively
            int[] audioCodes = GenerateAudioCodes(textEmbedding, numCodes, temperature, topK);

            // Step 4: Decode codes to audio waveform
            Tensor<T> audioWaveform = DecodeAudioCodes(audioCodes, numFrames);

            return audioWaveform;
        }

        /// <summary>
        /// Encodes text prompt into embeddings for conditioning.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> The text prompt guides what audio to generate.
        /// We convert the text to a numerical representation (embedding) that
        /// the language model can understand and use as context.
        ///
        /// For example:
        /// - "dog barking" → embedding that captures "dog" + "bark" concepts
        /// - "piano music" → embedding for musical instrument + melody
        ///
        /// The language model will generate codes that match this description.
        /// </remarks>
        private Tensor<T> EncodeText(string textPrompt)
        {
            // In production, use T5 or CLAP text encoder
            // For now, simplified tokenization

            // Convert text to token IDs (using simple character-level for demo)
            var tokens = new List<int>();
            foreach (char c in textPrompt.ToLowerInvariant())
            {
                if (char.IsLetterOrDigit(c) || c == ' ')
                {
                    tokens.Add(c);
                }
            }

            // Convert to tensor
            var tokenTensor = new Tensor<T>(new int[] { 1, tokens.Count });
            for (int i = 0; i < tokens.Count; i++)
            {
                tokenTensor[0, i] = NumOps.FromDouble(tokens[i]);
            }

            // In real implementation, this would call text encoder ONNX model
            // For now, return placeholder (would be replaced with actual encoder)
            return tokenTensor;
        }

        /// <summary>
        /// Generates audio codes autoregressively using language model.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is like GPT generating text tokens, but for audio codes.
        ///
        /// The process:
        /// 1. Start with empty sequence
        /// 2. Generate next code based on:
        ///    - Previous codes (context)
        ///    - Text embedding (what to generate)
        /// 3. Add code to sequence
        /// 4. Repeat until we have enough codes for desired duration
        ///
        /// <b>Key differences from text generation:</b>
        /// - Much longer sequences (5 seconds = 3000 codes vs ~50 word tokens)
        /// - Codes must form coherent audio (temporal consistency)
        /// - Temperature tuning is critical (too high = noise, too low = repetitive)
        /// </remarks>
        private int[] GenerateAudioCodes(
            Tensor<T> textEmbedding,
            int numCodes,
            double temperature,
            int topK)
        {
            var generatedCodes = new List<int>();

            // Autoregressive generation loop
            for (int step = 0; step < numCodes; step++)
            {
                // Prepare inputs for language model
                Tensor<T> codesTensor = CodesToTensor(generatedCodes);

                var inputs = new Dictionary<string, Tensor<T>>
                {
                    { "codes", codesTensor },                // Previous audio codes
                    { "text_embedding", textEmbedding }      // Text conditioning
                };

                // Run language model forward pass
                var outputs = _languageModel.Forward(inputs);
                Tensor<T> logits = outputs["logits"];

                // Sample next code with temperature and top-K
                int nextCode = SampleNextCode(logits, temperature, topK);

                // Add to sequence
                generatedCodes.Add(nextCode);

                // Optional: Print progress every second
                if (step % (_frameRate * _numQuantizers) == 0)
                {
                    int secondsGenerated = step / (_frameRate * _numQuantizers);
                    Console.WriteLine($"Generated {secondsGenerated} seconds of audio...");
                }
            }

            return generatedCodes.ToArray();
        }

        /// <summary>
        /// Samples next code from logits with temperature and top-K filtering.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> Instead of always picking the most likely code (greedy),
        /// we sample randomly with some constraints to add creativity while staying coherent.
        ///
        /// <b>Temperature:</b>
        /// - Divides logits before softmax
        /// - High temp (2.0) → flatter distribution → more random
        /// - Low temp (0.1) → sharper distribution → more deterministic
        ///
        /// <b>Top-K:</b>
        /// - Only consider top K most likely codes
        /// - Prevents sampling very unlikely codes (which would be noise)
        /// - K=250 means we only sample from top 250 out of 1024 codes
        /// </remarks>
        private int SampleNextCode(Tensor<T> logits, double temperature, int topK)
        {
            // Logits shape: (batch=1, sequence_length, vocab_size=1024)
            int vocabSize = _codebookSize;
            int lastPos = logits.Shape[1] - 1;

            // Extract logits for last position
            var finalLogits = new T[vocabSize];
            for (int i = 0; i < vocabSize; i++)
            {
                finalLogits[i] = logits[0, lastPos, i];
            }

            // Apply temperature
            for (int i = 0; i < vocabSize; i++)
            {
                double logitValue = Convert.ToDouble(finalLogits[i]);
                finalLogits[i] = NumOps.FromDouble(logitValue / temperature);
            }

            // Get top-K indices
            var logitPairs = finalLogits
                .Select((value, idx) => (value, idx))
                .OrderByDescending(pair => Convert.ToDouble(pair.value))
                .Take(topK)
                .ToList();

            // Softmax over top-K
            var probabilities = new double[topK];
            double maxLogit = Convert.ToDouble(logitPairs[0].value);
            double sumExp = 0.0;

            for (int i = 0; i < topK; i++)
            {
                double expValue = Math.Exp(Convert.ToDouble(logitPairs[i].value) - maxLogit);
                probabilities[i] = expValue;
                sumExp += expValue;
            }

            for (int i = 0; i < topK; i++)
            {
                probabilities[i] /= sumExp;
            }

            // Sample from distribution
            double randomValue = new Random().NextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < topK; i++)
            {
                cumulative += probabilities[i];
                if (randomValue <= cumulative)
                {
                    return logitPairs[i].idx;
                }
            }

            // Fallback: Return top-1
            return logitPairs[0].idx;
        }

        /// <summary>
        /// Decodes audio codes to waveform using EnCodec decoder.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This takes the compressed codes and "decompresses"
        /// them back into raw audio samples you can play.
        ///
        /// The decoder is a neural network that learned how to reconstruct audio
        /// from the quantized codes. It's like JPEG decompression, but for audio.
        ///
        /// <b>Input:</b> Array of 3000 codes (for 5 seconds at 75 Hz × 8 quantizers)
        /// <b>Output:</b> 120,000 audio samples (5 seconds at 24 kHz)
        /// <b>Compression ratio:</b> 40x smaller (3000 vs 120,000)
        /// </remarks>
        private Tensor<T> DecodeAudioCodes(int[] codes, int numFrames)
        {
            // Reshape codes into (frames, quantizers)
            // e.g., [c1_q1, c1_q2, ..., c1_q8, c2_q1, c2_q2, ...]
            //    → [[c1_q1, c1_q2, ..., c1_q8],
            //       [c2_q1, c2_q2, ..., c2_q8],
            //       ...]

            int[] shape = new int[] { 1, numFrames, _numQuantizers };
            var codesTensor = new Tensor<T>(shape);

            for (int frame = 0; frame < numFrames; frame++)
            {
                for (int q = 0; q < _numQuantizers; q++)
                {
                    int codeIdx = frame * _numQuantizers + q;
                    codesTensor[0, frame, q] = NumOps.FromDouble(codes[codeIdx]);
                }
            }

            // Prepare inputs for EnCodec decoder
            var inputs = new Dictionary<string, Tensor<T>>
            {
                { "codes", codesTensor }
            };

            // Run decoder forward pass
            var outputs = _audioCodec.Forward(inputs);
            Tensor<T> audioWaveform = outputs["audio"];

            return audioWaveform;
        }

        /// <summary>
        /// Converts list of codes to tensor for ONNX input.
        /// </summary>
        private Tensor<T> CodesToTensor(List<int> codes)
        {
            if (codes.Count == 0)
            {
                // Start with special "begin" token or zeros
                return new Tensor<T>(new int[] { 1, 1 });
            }

            int[] shape = new int[] { 1, codes.Count };
            var data = new Vector<T>(codes.Count);

            for (int i = 0; i < codes.Count; i++)
            {
                data[i] = NumOps.FromDouble(codes[i]);
            }

            return new Tensor<T>(shape, data);
        }

        /// <summary>
        /// Gets the audio sample rate.
        /// </summary>
        public int SampleRate => _sampleRate;
    }
}
```

### Phase 2: Text Encoder Helper (CLAP/T5)

**For Beginners:** In production, you'd use a pre-trained text encoder like CLAP or T5. Here's a simplified interface:

**File:** `src/Audio/TextEncoder.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Audio
{
    /// <summary>
    /// Encodes text descriptions into embeddings for audio generation conditioning.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <remarks>
    /// <b>For Beginners:</b> This converts text like "dog barking" into a numerical
    /// representation (embedding) that captures the meaning of the text.
    ///
    /// <b>Why we need this:</b>
    /// The language model needs to know WHAT to generate. The text embedding
    /// provides that information as numbers the model can process.
    ///
    /// <b>Popular text encoders for audio:</b>
    /// - CLAP: Contrastive Language-Audio Pre-training (learns text-audio pairs)
    /// - T5: Text-to-Text Transfer Transformer (general text understanding)
    /// </remarks>
    public class TextEncoder<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        private readonly OnnxModel<T> _encoderModel;
        private readonly int _embeddingDim;

        /// <summary>
        /// Creates a text encoder.
        /// </summary>
        /// <param name="modelPath">Path to CLAP or T5 ONNX model.</param>
        /// <param name="embeddingDim">Dimension of output embeddings (default: 512 for CLAP).</param>
        public TextEncoder(string modelPath, int embeddingDim = 512)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Text encoder model not found: {modelPath}");
            if (embeddingDim <= 0)
                throw new ArgumentException("Embedding dimension must be positive", nameof(embeddingDim));

            _encoderModel = new OnnxModel<T>(modelPath);
            _embeddingDim = embeddingDim;
        }

        /// <summary>
        /// Encodes text to embedding.
        /// </summary>
        /// <param name="text">Text description.</param>
        /// <returns>Embedding tensor of shape (1, embedding_dim).</returns>
        public Tensor<T> Encode(string text)
        {
            // Tokenize text (simplified - use actual tokenizer in production)
            int[] tokens = SimpleTokenize(text);

            // Convert to tensor
            var tokenTensor = new Tensor<T>(new int[] { 1, tokens.Length });
            for (int i = 0; i < tokens.Length; i++)
            {
                tokenTensor[0, i] = NumOps.FromDouble(tokens[i]);
            }

            // Run encoder
            var inputs = new Dictionary<string, Tensor<T>>
            {
                { "input_ids", tokenTensor }
            };

            var outputs = _encoderModel.Forward(inputs);
            return outputs["embedding"];
        }

        private int[] SimpleTokenize(string text)
        {
            // Simplified character-level tokenization
            // In production, use SentencePiece or similar
            return text.ToLowerInvariant()
                .Where(c => char.IsLetterOrDigit(c) || c == ' ')
                .Select(c => (int)c)
                .ToArray();
        }
    }
}
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/UnitTests/Models/AudioGenModelTests.cs`

```csharp
using Xunit;
using Moq;
using AiDotNet.Models.Audio;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Models
{
    public class AudioGenModelTests
    {
        [Fact]
        public void Generate_CallsLanguageModelThenCodec()
        {
            // Arrange
            var mockLM = new Mock<OnnxModel<float>>();
            var mockCodec = new Mock<OnnxModel<float>>();

            int numCalls = 0;

            // Mock language model (generates codes)
            mockLM.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns(() =>
                {
                    numCalls++;
                    var logits = new Tensor<float>(new int[] { 1, numCalls, 1024 });
                    // Set highest probability for code 500
                    logits[0, numCalls - 1, 500] = 100.0f;
                    return new Dictionary<string, Tensor<float>> { { "logits", logits } };
                });

            // Mock codec decoder
            int expectedSamples = 1 * 24000; // 1 second at 24kHz
            var audioWave = new Tensor<float>(new int[] { 1, expectedSamples });
            mockCodec.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns(new Dictionary<string, Tensor<float>> { { "audio", audioWave } });

            var model = new AudioGenModel<float>(mockLM.Object, mockCodec.Object);

            // Act
            var result = model.Generate("dog barking", durationInSeconds: 1, temperature: 1.0, topK: 250);

            // Assert
            // 1 second × 75 frames/sec × 8 quantizers = 600 language model calls
            Assert.Equal(600, numCalls);
            mockCodec.Verify(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()), Times.Once);
            Assert.NotNull(result);
        }

        [Fact]
        public void Generate_EmptyPrompt_ThrowsException()
        {
            // Arrange
            var model = new AudioGenModel<float>("dummy_lm.onnx", "dummy_codec.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Generate("", durationInSeconds: 5));
            Assert.Throws<ArgumentException>(() => model.Generate("   ", durationInSeconds: 5));
        }

        [Theory]
        [InlineData(1)]
        [InlineData(5)]
        [InlineData(10)]
        public void Generate_DifferentDurations_GeneratesCorrectLength(int duration)
        {
            // Arrange
            var mockLM = new Mock<OnnxModel<float>>();
            var mockCodec = new Mock<OnnxModel<float>>();

            // Mock LM returns random codes
            mockLM.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns((Dictionary<string, Tensor<float>> inputs) =>
                {
                    int seqLen = inputs["codes"].Shape[1];
                    var logits = new Tensor<float>(new int[] { 1, seqLen, 1024 });
                    logits[0, seqLen - 1, 100] = 50.0f; // Dummy high logit
                    return new Dictionary<string, Tensor<float>> { { "logits", logits } };
                });

            // Mock codec returns audio
            int expectedSamples = duration * 24000;
            var audioWave = new Tensor<float>(new int[] { 1, expectedSamples });
            mockCodec.Setup(m => m.Forward(It.IsAny<Dictionary<string, Tensor<float>>>()))
                .Returns(new Dictionary<string, Tensor<float>> { { "audio", audioWave } });

            var model = new AudioGenModel<float>(mockLM.Object, mockCodec.Object);

            // Act
            var result = model.Generate("test", durationInSeconds: duration, temperature: 1.0, topK: 10);

            // Assert
            Assert.Equal(expectedSamples, result.Shape[result.Shape.Length - 1]);
        }
    }
}
```

### Integration Tests

**File:** `tests/IntegrationTests/Models/AudioGenModelIntegrationTests.cs`

```csharp
using Xunit;
using AiDotNet.Models.Audio;
using AiDotNet.Audio;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.Models
{
    public class AudioGenModelIntegrationTests
    {
        [Fact(Skip = "Requires AudioGen and EnCodec ONNX models")]
        public void Generate_RealModels_GeneratesPlausibleAudio()
        {
            // Arrange
            string lmPath = "audiogen_lm.onnx";
            string codecPath = "encodec_24khz.onnx";

            if (!File.Exists(lmPath) || !File.Exists(codecPath))
            {
                Assert.True(false, "Download AudioGen models from Hugging Face");
            }

            var audioGen = new AudioGenModel<float>(lmPath, codecPath, sampleRate: 24000);

            // Act
            string prompt = "a dog barking loudly";
            Tensor<float> audio = audioGen.Generate(prompt, durationInSeconds: 3, temperature: 1.0, topK: 250);

            // Assert
            Assert.NotNull(audio);
            int expectedSamples = 3 * 24000; // 3 seconds at 24kHz
            Assert.True(Math.Abs(audio.Shape[audio.Shape.Length - 1] - expectedSamples) < 1000);

            // Save for manual listening
            string outputPath = "audiogen_dog_bark.wav";
            AudioFileHelper.SaveWav(audio, outputPath, sampleRate: 24000);

            Assert.True(File.Exists(outputPath));
            Console.WriteLine($"Audio saved to: {outputPath}");
            Console.WriteLine("Listen to verify it sounds like a dog barking.");
        }

        [Fact(Skip = "Requires models")]
        public void Generate_DifferentPrompts_GeneratesDifferentSounds()
        {
            // Arrange
            var audioGen = new AudioGenModel<float>("audiogen_lm.onnx", "encodec_24khz.onnx");

            var prompts = new[]
            {
                "thunder and rain",
                "car engine starting",
                "crowd cheering"
            };

            // Act & Assert
            foreach (var prompt in prompts)
            {
                var audio = audioGen.Generate(prompt, durationInSeconds: 3, temperature: 1.0, topK: 250);

                Assert.NotNull(audio);

                string filename = $"audiogen_{prompt.Replace(" ", "_")}.wav";
                AudioFileHelper.SaveWav(audio, filename, sampleRate: 24000);

                Console.WriteLine($"Generated: {filename}");
            }

            Console.WriteLine("Compare the audio files - they should sound distinctly different.");
        }

        [Fact(Skip = "Requires models")]
        public void Generate_DifferentTemperatures_AffectsCreativity()
        {
            // Arrange
            var audioGen = new AudioGenModel<float>("audiogen_lm.onnx", "encodec_24khz.onnx");
            string prompt = "piano music";

            // Act
            var conservative = audioGen.Generate(prompt, durationInSeconds: 5, temperature: 0.5, topK: 100);
            var balanced = audioGen.Generate(prompt, durationInSeconds: 5, temperature: 1.0, topK: 250);
            var creative = audioGen.Generate(prompt, durationInSeconds: 5, temperature: 1.5, topK: 500);

            // Assert
            AudioFileHelper.SaveWav(conservative, "piano_conservative.wav", 24000);
            AudioFileHelper.SaveWav(balanced, "piano_balanced.wav", 24000);
            AudioFileHelper.SaveWav(creative, "piano_creative.wav", 24000);

            Console.WriteLine("Compare files:");
            Console.WriteLine("- Conservative: More predictable, repetitive");
            Console.WriteLine("- Balanced: Good coherence and variety");
            Console.WriteLine("- Creative: More diverse but possibly less coherent");
        }
    }
}
```

---

## Common Pitfalls

### 1. Code Sequence Length Mismatch

**Problem:** Generating wrong number of codes for desired duration

**Solution:** Calculate correctly
```csharp
// For 5 seconds at 75 Hz frame rate with 8 quantizers:
int numFrames = 5 * 75 = 375 frames
int numCodes = 375 * 8 = 3000 codes

// NOT: 5 * 75 = 375 codes (this is only frames, missing quantizers!)
```

### 2. Temperature Too High/Low

**Problem:** Generated audio is noisy or repetitive

**Solution:** Tune temperature
```csharp
// Too low (0.1): Repetitive, boring
// Too high (5.0): Random noise
// Sweet spot: 0.8 - 1.2
```

### 3. Codec Input Shape Mismatch

**Problem:** EnCodec expects `(batch, quantizers, time)` but you provide `(batch, time, quantizers)`

**Solution:** Check model signature and transpose if needed
```csharp
// Check with Netron or ONNX model inspector
// Reshape codes to match expected input
```

### 4. Running Out of Memory

**Problem:** Generating long audio (60+ seconds) causes OOM

**Solution:** Generate in chunks
```csharp
// Generate 10 seconds at a time
var chunks = new List<Tensor<float>>();
for (int i = 0; i < 6; i++)
{
    var chunk = audioGen.Generate(prompt, durationInSeconds: 10);
    chunks.Add(chunk);
}
// Concatenate chunks
```

### 5. Poor Quality with High Top-K

**Problem:** topK=1000 (nearly all vocab) produces incoherent audio

**Solution:** Use conservative top-K
```csharp
// Recommended: topK = 100-250 for codebook size 1024
// This is 10-25% of vocabulary
```

### 6. Text Embedding Dimension Mismatch

**Problem:** Text encoder outputs 512-dim embeddings but LM expects 768-dim

**Solution:** Use matching models or add projection layer
```csharp
// Ensure text encoder and language model are compatible
// Check ONNX model signatures before loading
```

---

## Resources

### Papers
- **EnCodec**: "High Fidelity Neural Audio Compression" (Meta AI, 2022)
- **AudioGen**: "AudioGen: Textually Guided Audio Generation" (Meta AI, 2023)
- **MusicGen**: "Simple and Controllable Music Generation" (Meta AI, 2023)
- **SoundStream**: "SoundStream: An End-to-End Neural Audio Codec" (Google, 2021)

### Code Examples
- AudioCraft (Meta's official): https://github.com/facebookresearch/audiocraft
- EnCodec implementation: https://github.com/facebookresearch/encodec
- Hugging Face Transformers (AudioGen/MusicGen): https://huggingface.co/docs/transformers/model_doc/musicgen

### ONNX Models
- Hugging Face audio models: https://huggingface.co/models?pipeline_tag=audio-classification&library=onnx
- Convert AudioCraft to ONNX: Use `torch.onnx.export()` on loaded models

### Audio Processing
- Understanding audio codecs: https://en.wikipedia.org/wiki/Audio_codec
- Vector quantization: https://en.wikipedia.org/wiki/Vector_quantization
- Residual VQ explained: https://arxiv.org/abs/2107.03312

---

## Advanced Topics

### Multi-Band Audio Generation

Some models generate different frequency bands separately:
- Low frequencies: Bass, rhythm
- Mid frequencies: Melody, vocals
- High frequencies: Cymbals, sibilants

Then combine all bands for final audio.

### Classifier-Free Guidance

Technique to strengthen text conditioning:
```csharp
// Generate with and without text conditioning
Tensor<T> conditioned = GenerateWithText(prompt);
Tensor<T> unconditioned = GenerateWithoutText();

// Blend with guidance scale
Tensor<T> guided = conditioned * guidanceScale - unconditioned * (guidanceScale - 1);
```

**Effect:** Stronger adherence to text prompt (but can reduce diversity)

### Melody Conditioning (MusicGen)

Generate music that follows a hummed melody:
1. Extract melody features (pitch contour) from reference audio
2. Condition language model on both text + melody
3. Generate codes that match melody but with full instrumentation

---

## Next Steps

After implementing Issue #271, you'll have:
1. Complete audio generation pipeline
2. Understanding of neural audio codecs
3. Experience with long-sequence generation
4. Foundation for music generation and sound design tools

You can extend this to:
- Music generation with style transfer
- Interactive sound design tools
- Game audio generation (dynamic soundtracks)
- Audio editing (extend, remix, transform)

Good luck with your implementation!
