---
title: "Text To Speech"
description: "All 254 public types in the AiDotNet.texttospeech namespace, organized by kind."
section: "API Reference"
---

**254** public types in this namespace, organized by kind.

## Models & Types (119)

| Type | Summary |
|:-----|:--------|
| [`APNet2<T>`](/docs/reference/wiki/texttospeech/apnet2/) | APNet2: improved amplitude-phase network with ResNet backbone and multi-resolution STFT loss for higher-quality waveform reconstruction. |
| [`APNet<T>`](/docs/reference/wiki/texttospeech/apnet/) | APNet: amplitude-phase network that predicts amplitude and phase spectra separately then reconstructs waveform via iSTFT. |
| [`AdaSpeech2<T>`](/docs/reference/wiki/texttospeech/adaspeech2/) | AdaSpeech 2: adaptive TTS that leverages untranscribed speech via mel-to-phoneme pipeline. |
| [`AdaSpeech<T>`](/docs/reference/wiki/texttospeech/adaspeech/) | AdaSpeech: adaptive TTS for custom voice with acoustic condition modeling and conditional layer normalization. |
| [`AlignTTS<T>`](/docs/reference/wiki/texttospeech/aligntts/) | AlignTTS: alignment-free non-autoregressive TTS using feed-forward transformer with mix density network alignment. |
| [`AmazonPolly<T>`](/docs/reference/wiki/texttospeech/amazonpolly/) | Amazon Polly: AWS neural TTS service with neural and standard engines. |
| [`Amphion<T>`](/docs/reference/wiki/texttospeech/amphion/) | Amphion: open-source audio toolkit supporting multiple TTS architectures. |
| [`AudioLM<T>`](/docs/reference/wiki/texttospeech/audiolm/) | AudioLM: language modeling approach to audio generation with semantic and acoustic tokens. |
| [`AudioPaLM<T>`](/docs/reference/wiki/texttospeech/audiopalm/) | AudioPaLM: AudioPaLM: A Large Language Model That Can Speak and Listen. |
| [`AzureNeuralTTS<T>`](/docs/reference/wiki/texttospeech/azureneuraltts/) | Azure Neural TTS: local neural synthesis model inspired by Microsoft's architecture. |
| [`Bark<T>`](/docs/reference/wiki/texttospeech/bark/) | Bark: GPT-based text-to-audio model generating speech, music, and sound effects from text prompts. |
| [`BigVGAN<T>`](/docs/reference/wiki/texttospeech/bigvgan/) | BigVGAN: large-scale universal vocoder with anti-aliased multi-periodicity composition (AMP) and Snake activation for high-fidelity synthesis. |
| [`CSM<T>`](/docs/reference/wiki/texttospeech/csm/) | CSM: Conversational Speech Model for context-sensitive multi-turn dialogue synthesis. |
| [`ChatTTS<T>`](/docs/reference/wiki/texttospeech/chattts/) | ChatTTS: conversational TTS with prosody control tokens for natural dialogue synthesis. |
| [`Chatterbox<T>`](/docs/reference/wiki/texttospeech/chatterbox/) | Chatterbox: emotion-controllable zero-shot TTS with watermarking and exaggeration control. |
| [`CoMoSpeech<T>`](/docs/reference/wiki/texttospeech/comospeech/) | CoMoSpeech: CoMoSpeech: One-Step Speech Synthesis via Consistency Model. |
| [`CosyVoice2<T>`](/docs/reference/wiki/texttospeech/cosyvoice2/) | CosyVoice 2: improved streaming TTS with chunk-aware causal flow matching and FSQ. |
| [`CosyVoice3<T>`](/docs/reference/wiki/texttospeech/cosyvoice3/) | CosyVoice 3: Fun-CosyVoice 3 zero-shot multilingual TTS with enhanced semantic tokens. |
| [`CosyVoiceClone<T>`](/docs/reference/wiki/texttospeech/cosyvoiceclone/) | CosyVoice: scalable zero-shot voice cloning using supervised semantic tokens and OT-CFM decoder. |
| [`CosyVoice<T>`](/docs/reference/wiki/texttospeech/cosyvoice/) | CosyVoice: multilingual TTS with supervised semantic tokens and conditional flow matching. |
| [`DeepVoice3<T>`](/docs/reference/wiki/texttospeech/deepvoice3/) | Deep Voice 3: fully convolutional attention-based TTS with monotonic attention. |
| [`DiTToTTS<T>`](/docs/reference/wiki/texttospeech/dittotts/) | DiTToTTS: DiTTo-TTS: Efficient and Scalable Zero-Shot TTS with Diffusion Transformer. |
| [`Dia<T>`](/docs/reference/wiki/texttospeech/dia/) | Dia: dialogue-oriented TTS generating multi-speaker conversations with expressive prosody. |
| [`DiffWave<T>`](/docs/reference/wiki/texttospeech/diffwave/) | DiffWave: diffusion probabilistic model for conditional and unconditional waveform generation. |
| [`E2TTS<T>`](/docs/reference/wiki/texttospeech/e2tts/) | E2 TTS: fully non-autoregressive flow-matching TTS with character-level text input. |
| [`E3TTS<T>`](/docs/reference/wiki/texttospeech/e3tts/) | E3 TTS: non-autoregressive end-to-end diffusion TTS without explicit duration modeling. |
| [`ElevenLabsTTS<T>`](/docs/reference/wiki/texttospeech/elevenlabstts/) | ElevenLabs: AI voice synthesis platform with voice cloning, emotion control, and multilingual support. |
| [`EmotiVoice<T>`](/docs/reference/wiki/texttospeech/emotivoice/) | EmotiVoice: multi-voice and prompt-controlled TTS with emotion and style control. |
| [`F5TTS<T>`](/docs/reference/wiki/texttospeech/f5tts/) | F5-TTS: non-autoregressive flow-matching TTS that generates speech from text using DiT backbone. |
| [`FastSpeech2<T>`](/docs/reference/wiki/texttospeech/fastspeech2/) | FastSpeech 2: non-autoregressive TTS with variance adaptor for pitch, energy, and duration. |
| [`FastSpeech<T>`](/docs/reference/wiki/texttospeech/fastspeech/) | FastSpeech: non-autoregressive TTS with knowledge-distilled duration predictor for parallel generation. |
| [`FireRedTTS<T>`](/docs/reference/wiki/texttospeech/fireredtts/) | FireRedTTS: foundation TTS model with large-scale training and multi-codebook generation. |
| [`FishSpeechV15<T>`](/docs/reference/wiki/texttospeech/fishspeechv15/) | FishSpeechV15: Fish Speech V1.5: Improved Multilingual TTS. |
| [`FishSpeech<T>`](/docs/reference/wiki/texttospeech/fishspeech/) | Fish Speech: dual-AR codec language model with grouped-finite-scalar-quantization for fast TTS. |
| [`ForwardTacotron<T>`](/docs/reference/wiki/texttospeech/forwardtacotron/) | Forward Tacotron: non-autoregressive Tacotron variant using duration predictor instead of attention. |
| [`FreGrad<T>`](/docs/reference/wiki/texttospeech/fregrad/) | FreGrad: lightweight diffusion vocoder that operates in the frequency domain via DWT (discrete wavelet transform) for faster synthesis. |
| [`GLM4Voice<T>`](/docs/reference/wiki/texttospeech/glm4voice/) | GLM-4-Voice: intelligent and human-like end-to-end spoken chatbot with emotion and prosody control. |
| [`GPTSoVITS<T>`](/docs/reference/wiki/texttospeech/gptsovits/) | GPT-SoVITS: few-shot TTS combining GPT-style autoregressive with SoVITS decoder. |
| [`GlowTTS<T>`](/docs/reference/wiki/texttospeech/glowtts/) | Glow-TTS: flow-based generative model for non-autoregressive TTS with monotonic alignment search (MAS). |
| [`GoogleCloudTTS<T>`](/docs/reference/wiki/texttospeech/googlecloudtts/) | Google Cloud Text-to-Speech: local WaveNet-style neural synthesis model inspired by Google's architecture. |
| [`GradTTS<T>`](/docs/reference/wiki/texttospeech/gradtts/) | Grad-TTS: diffusion-based acoustic model using score matching and stochastic differential equations. |
| [`HiFiGAN<T>`](/docs/reference/wiki/texttospeech/hifigan/) | HiFi-GAN: high-fidelity neural vocoder with multi-receptive field fusion for parallel waveform generation. |
| [`ISTFTNet<T>`](/docs/reference/wiki/texttospeech/istftnet/) | iSTFTNet: vocoder that predicts STFT magnitude and phase, then uses inverse STFT for waveform reconstruction. |
| [`IndexTTS2<T>`](/docs/reference/wiki/texttospeech/indextts2/) | IndexTTS2: IndexTTS-2: Duration and Emotion Control for Zero-Shot TTS. |
| [`IndexTTS<T>`](/docs/reference/wiki/texttospeech/indextts/) | IndexTTS: LLM-based zero-shot TTS with reference audio indexing for voice cloning. |
| [`KaniTTS2<T>`](/docs/reference/wiki/texttospeech/kanitts2/) | KaniTTS2: Kani-TTS-2: Improved Lightweight Codec TTS. |
| [`KaniTTS<T>`](/docs/reference/wiki/texttospeech/kanitts/) | KaniTTS: Kani-TTS: Efficient Codec-Based TTS. |
| [`Kokoro<T>`](/docs/reference/wiki/texttospeech/kokoro/) | Kokoro: lightweight end-to-end TTS with a StyleTTS2-inspired architecture using style tokens and ISTFTNet decoder. |
| [`LlamaOmni<T>`](/docs/reference/wiki/texttospeech/llamaomni/) | LLaMA-Omni: seamless speech interaction with LLM, enabling low-latency speech-to-speech conversation. |
| [`Llasa<T>`](/docs/reference/wiki/texttospeech/llasa/) | Llasa: LLaMA-based speech synthesis using XCodec2 for multi-level codec representation. |
| [`MARS5TTS<T>`](/docs/reference/wiki/texttospeech/mars5tts/) | MARS5-TTS: two-stage TTS with shallow AR for coarse prosody then deep NAR for fine acoustics. |
| [`MaskGCT<T>`](/docs/reference/wiki/texttospeech/maskgct/) | MaskGCT: non-autoregressive masked generative codec transformer for zero-shot TTS. |
| [`MatchaTTS<T>`](/docs/reference/wiki/texttospeech/matchatts/) | Matcha-TTS: optimal-transport conditional flow matching for fast non-autoregressive TTS. |
| [`MegaTTS2<T>`](/docs/reference/wiki/texttospeech/megatts2/) | MegaTTS2: Mega-TTS 2: Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis. |
| [`MegaTTS3<T>`](/docs/reference/wiki/texttospeech/megatts3/) | MegaTTS 3: sparse diffusion transformer TTS using DiT backbone for efficient generation. |
| [`MegaTTS<T>`](/docs/reference/wiki/texttospeech/megatts/) | MegaTTS: Mega-TTS: Zero-Shot TTS with Prosody Decomposition. |
| [`MelGAN<T>`](/docs/reference/wiki/texttospeech/melgan/) | MelGAN: lightweight non-autoregressive GAN vocoder with feature matching loss for fast inference. |
| [`MeloTTS<T>`](/docs/reference/wiki/texttospeech/melotts/) | MeloTTS: high-quality multilingual TTS with VITS backbone, language-specific text processing, and mixed-language support. |
| [`MetaVoice1B<T>`](/docs/reference/wiki/texttospeech/metavoice1b/) | MetaVoice1B: MetaVoice-1B: 1.2B Parameter Voice Cloning Model. |
| [`MinMo<T>`](/docs/reference/wiki/texttospeech/minmo/) | MinMo: multimodal LLM with speech understanding and generation for seamless voice interaction. |
| [`Moshi<T>`](/docs/reference/wiki/texttospeech/moshi/) | Moshi: full-duplex spoken dialogue framework enabling real-time voice interaction. |
| [`MultiBandMelGAN<T>`](/docs/reference/wiki/texttospeech/multibandmelgan/) | Multi-band MelGAN: decomposes target into sub-bands, generates each in parallel, then synthesizes full-band. |
| [`Murf<T>`](/docs/reference/wiki/texttospeech/murf/) | Murf: enterprise AI voice platform with studio-quality text-to-speech generation. |
| [`NVIDIARivaTTS<T>`](/docs/reference/wiki/texttospeech/nvidiarivatts/) | NVIDIARivaTTS: NVIDIA Riva TTS. |
| [`NaturalSpeech2<T>`](/docs/reference/wiki/texttospeech/naturalspeech2/) | NaturalSpeech 2: latent diffusion model with continuous latent vectors for zero-shot speech synthesis. |
| [`NaturalSpeech3<T>`](/docs/reference/wiki/texttospeech/naturalspeech3/) | NaturalSpeech 3: factorized codec + diffusion for disentangled speech attribute control. |
| [`NaturalSpeech<T>`](/docs/reference/wiki/texttospeech/naturalspeech/) | NaturalSpeech: fully end-to-end TTS with VAE, normalizing flow, and bidirectional prior/posterior for human-level quality. |
| [`OpenVoiceV2<T>`](/docs/reference/wiki/texttospeech/openvoicev2/) | OpenVoiceV2: OpenVoice V2: Improved Instant Voice Cloning. |
| [`OpenVoice<T>`](/docs/reference/wiki/texttospeech/openvoice/) | OpenVoice: versatile instant voice cloning with decoupled tone color conversion. |
| [`OrpheusTTS<T>`](/docs/reference/wiki/texttospeech/orpheustts/) | Orpheus: real-time emotion-controllable TTS with SNAC codec and LLaMA backbone. |
| [`OuteTTS<T>`](/docs/reference/wiki/texttospeech/outetts/) | OuteTTS: text-to-speech using pure language modeling on audio tokens. |
| [`ParallelWaveGAN<T>`](/docs/reference/wiki/texttospeech/parallelwavegan/) | Parallel WaveGAN: non-autoregressive GAN vocoder with multi-resolution STFT loss for stable training. |
| [`ParlerTTS<T>`](/docs/reference/wiki/texttospeech/parlertts/) | Parler-TTS: text-described TTS that generates speech matching a natural language voice description. |
| [`Pheme<T>`](/docs/reference/wiki/texttospeech/pheme/) | Pheme: Pheme: Efficient and Conversational Speech Generation. |
| [`Piper<T>`](/docs/reference/wiki/texttospeech/piper/) | Piper: lightweight local TTS system based on VITS optimized for edge/embedded deployment with fast inference. |
| [`PlayHT<T>`](/docs/reference/wiki/texttospeech/playht/) | PlayHT: AI voice generation with 2.0 turbo model for ultra-realistic speech synthesis. |
| [`PortaSpeech<T>`](/docs/reference/wiki/texttospeech/portaspeech/) | PortaSpeech: portable TTS with word-level prosody modeling and normalizing flow-based post-net for expressiveness. |
| [`PriorGrad<T>`](/docs/reference/wiki/texttospeech/priorgrad/) | PriorGrad: adaptive diffusion vocoder that uses data-dependent prior (mel-conditioned noise) instead of isotropic Gaussian. |
| [`ProDiff<T>`](/docs/reference/wiki/texttospeech/prodiff/) | ProDiff: progressive fast diffusion model for high-quality TTS with knowledge distillation. |
| [`PromptTTS<T>`](/docs/reference/wiki/texttospeech/prompttts/) | PromptTTS: description-based TTS that controls speaker attributes via natural language prompts (e.g., "a young female with a cheerful tone"). |
| [`SPEARTTS<T>`](/docs/reference/wiki/texttospeech/speartts/) | SPEAR-TTS: high-fidelity text-to-speech with minimal supervision using a speak-read-prompt pipeline. |
| [`SeedTTSClone<T>`](/docs/reference/wiki/texttospeech/seedttsclone/) | Seed-TTS: diffusion-based zero-shot voice cloning with speaker factorization for high-fidelity synthesis. |
| [`SeedTTS<T>`](/docs/reference/wiki/texttospeech/seedtts/) | Seed-TTS: large-scale autoregressive TTS with self-distillation for in-context learning. |
| [`SoundStorm<T>`](/docs/reference/wiki/texttospeech/soundstorm/) | SoundStorm: parallel audio generation via MaskGIT-style iterative decoding of SoundStream tokens. |
| [`SparkTTS<T>`](/docs/reference/wiki/texttospeech/sparktts/) | Spark-TTS: LLM-based zero-shot TTS with BiCodec for controllable synthesis. |
| [`SpeechGPT<T>`](/docs/reference/wiki/texttospeech/speechgpt/) | SpeechGPT: empowering LLMs with intrinsic cross-modal conversational abilities via discrete speech tokens. |
| [`SpeechT5<T>`](/docs/reference/wiki/texttospeech/speecht5/) | SpeechT5: SpeechT5: Unified-Modal Encoder-Decoder Pre-Training. |
| [`SpeedySpeech<T>`](/docs/reference/wiki/texttospeech/speedyspeech/) | SpeedySpeech: lightweight non-autoregressive TTS with convolutional residual blocks and teacher-student duration distillation. |
| [`SpiritLM<T>`](/docs/reference/wiki/texttospeech/spiritlm/) | Spirit-LM: interleaved text-speech language model bridging written and spoken communication. |
| [`StepAudio<T>`](/docs/reference/wiki/texttospeech/stepaudio/) | Step-Audio: unified understanding and generation speech language model for intelligent voice interaction. |
| [`StyleTTS2<T>`](/docs/reference/wiki/texttospeech/styletts2/) | StyleTTS 2: style diffusion and adversarial training with large SLMs for human-level expressive TTS. |
| [`StyleTTSZS<T>`](/docs/reference/wiki/texttospeech/stylettszs/) | StyleTTSZS: StyleTTS-ZS: Zero-Shot Voice Cloning with Style and Duration Control. |
| [`StyleTTS<T>`](/docs/reference/wiki/texttospeech/styletts/) | StyleTTS: style-based generative model for expressive TTS with style diffusion and adversarial training. |
| [`Tacotron2<T>`](/docs/reference/wiki/texttospeech/tacotron2/) | Tacotron 2: improved attention-based TTS with location-sensitive attention and simplified decoder. |
| [`Tacotron<T>`](/docs/reference/wiki/texttospeech/tacotron/) | Tacotron: sequence-to-sequence attention-based TTS with CBHG encoder and autoregressive decoder. |
| [`TortoiseTTS<T>`](/docs/reference/wiki/texttospeech/tortoisetts/) | TortoiseTTS: TorToise: Better Speech Synthesis Through Scaling. |
| [`TransformerTTS<T>`](/docs/reference/wiki/texttospeech/transformertts/) | Transformer TTS: multi-head self-attention based acoustic model replacing RNNs with transformers. |
| [`UniAudio<T>`](/docs/reference/wiki/texttospeech/uniaudio/) | UniAudio: unified multi-task audio tokenizer and language model for TTS, music, and sound effects. |
| [`UnivNet<T>`](/docs/reference/wiki/texttospeech/univnet/) | UnivNet: universal neural vocoder with location-variable convolution (LVC) for adaptive kernel generation. |
| [`VALLE2<T>`](/docs/reference/wiki/texttospeech/valle2/) | VALLE2: VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot TTS. |
| [`VALLEXClone<T>`](/docs/reference/wiki/texttospeech/vallexclone/) | VALL-E X: cross-lingual zero-shot voice cloning with language ID conditioning for multi-language synthesis. |
| [`VALLEX<T>`](/docs/reference/wiki/texttospeech/vallex/) | VALL-E X: cross-lingual zero-shot text-to-speech extending VALL-E with language ID conditioning. |
| [`VALLE<T>`](/docs/reference/wiki/texttospeech/valle/) | VALL-E: neural codec language model for zero-shot text-to-speech using autoregressive and non-autoregressive transformers. |
| [`VITS2<T>`](/docs/reference/wiki/texttospeech/vits2/) | VITS2: improved VITS with duration discriminator, transformed prior, and speaker-conditional normalizing flow. |
| [`VITS<T>`](/docs/reference/wiki/texttospeech/vits/) | VITS: end-to-end TTS with conditional VAE, normalizing flows, and adversarial training for parallel high-quality synthesis. |
| [`Vocos<T>`](/docs/reference/wiki/texttospeech/vocos/) | Vocos: ConvNeXt-based vocoder that reconstructs waveform from Fourier coefficients (STFT magnitude + phase via ISTFT) instead of time-domain upsampling. |
| [`VoiceCraft<T>`](/docs/reference/wiki/texttospeech/voicecraft/) | VoiceCraft: VoiceCraft: Zero-Shot Speech Editing and TTS in the Wild. |
| [`VoiceFlow<T>`](/docs/reference/wiki/texttospeech/voiceflow/) | VoiceFlow: rectified flow matching for non-autoregressive TTS with straight ODE paths. |
| [`Voicebox<T>`](/docs/reference/wiki/texttospeech/voicebox/) | Voicebox: text-guided multilingual universal speech generation at scale using non-autoregressive flow matching. |
| [`WaveGlow<T>`](/docs/reference/wiki/texttospeech/waveglow/) | WaveGlow: flow-based generative vocoder combining Glow invertible 1x1 convolutions with WaveNet affine coupling layers. |
| [`WaveGrad<T>`](/docs/reference/wiki/texttospeech/wavegrad/) | WaveGrad: gradient-based conditional waveform generation using continuous noise level conditioning. |
| [`WaveNet<T>`](/docs/reference/wiki/texttospeech/wavenet/) | WaveNet: autoregressive generative model using dilated causal convolutions for raw audio generation. |
| [`WaveRNN<T>`](/docs/reference/wiki/texttospeech/wavernn/) | WaveRNN: efficient autoregressive vocoder with dual softmax and subscale sample generation. |
| [`WellSaidLabs<T>`](/docs/reference/wiki/texttospeech/wellsaidlabs/) | WellSaid Labs: enterprise neural TTS with custom voice avatars for brand consistency. |
| [`WhisperSpeech<T>`](/docs/reference/wiki/texttospeech/whisperspeech/) | WhisperSpeech: text-to-speech using Whisper encoder features as semantic tokens plus acoustic generation. |
| [`XTTSv2Clone<T>`](/docs/reference/wiki/texttospeech/xttsv2clone/) | XTTS v2: multilingual voice cloning using GPT-2 backbone with 6-second reference audio. |
| [`XTTSv2<T>`](/docs/reference/wiki/texttospeech/xttsv2/) | XTTS v2: multilingual zero-shot TTS using GPT-2 backbone to autoregressively predict VQ-VAE audio tokens. |
| [`YourTTS<T>`](/docs/reference/wiki/texttospeech/yourtts/) | YourTTS: multilingual zero-shot multi-speaker TTS built on VITS with speaker and language conditioning. |
| [`Zonos<T>`](/docs/reference/wiki/texttospeech/zonos/) | Zonos: zero-shot TTS with conditioning for speaker, emotion, and quality. |

## Base Classes (3)

| Type | Summary |
|:-----|:--------|
| [`AcousticModelBase<T>`](/docs/reference/wiki/texttospeech/acousticmodelbase/) | Base class for acoustic TTS models that generate mel-spectrograms from text. |
| [`TtsModelBase<T>`](/docs/reference/wiki/texttospeech/ttsmodelbase/) | Base class for text-to-speech neural networks that can operate in both ONNX inference and native training modes. |
| [`VocoderBase<T>`](/docs/reference/wiki/texttospeech/vocoderbase/) | Base class for neural vocoder models that convert mel-spectrograms to audio waveforms. |

## Interfaces (7)

| Type | Summary |
|:-----|:--------|
| [`IAcousticModel<T>`](/docs/reference/wiki/texttospeech/iacousticmodel/) | Interface for acoustic models that generate mel-spectrograms from text. |
| [`ICodecTts<T>`](/docs/reference/wiki/texttospeech/icodectts/) | Interface for codec-based TTS models that use neural audio codecs with language model decoding. |
| [`IEndToEndTts<T>`](/docs/reference/wiki/texttospeech/iendtoendtts/) | Interface for end-to-end TTS models that generate waveforms directly from text without a separate vocoder stage. |
| [`IStreamingTts<T>`](/docs/reference/wiki/texttospeech/istreamingtts/) | Interface for TTS models that support streaming/chunked synthesis with low latency. |
| [`ITtsModel<T>`](/docs/reference/wiki/texttospeech/ittsmodel/) | Base interface for all text-to-speech models. |
| [`IVocoder<T>`](/docs/reference/wiki/texttospeech/ivocoder/) | Interface for neural vocoders that convert mel-spectrograms to audio waveforms. |
| [`IVoiceCloner<T>`](/docs/reference/wiki/texttospeech/ivoicecloner/) | Interface for TTS models that support zero-shot or few-shot voice cloning from reference audio. |

## Options & Configuration (125)

| Type | Summary |
|:-----|:--------|
| [`APNet2Options`](/docs/reference/wiki/texttospeech/apnet2options/) | Options for APNet2 (improved amplitude-phase network with ResNet backbone and multi-resolution STFT loss). |
| [`APNetOptions`](/docs/reference/wiki/texttospeech/apnetoptions/) | Options for APNet (amplitude-phase network with dual-stream spectrum prediction and anti-wrapping phase loss). |
| [`AcousticModelOptions`](/docs/reference/wiki/texttospeech/acousticmodeloptions/) | Base configuration options for classic acoustic TTS models. |
| [`AdaSpeech2Options`](/docs/reference/wiki/texttospeech/adaspeech2options/) | Options for AdaSpeech 2 (adaptive TTS with untranscribed speech data). |
| [`AdaSpeechOptions`](/docs/reference/wiki/texttospeech/adaspeechoptions/) | Options for AdaSpeech (adaptive TTS with acoustic condition modeling). |
| [`AlignTTSOptions`](/docs/reference/wiki/texttospeech/alignttsoptions/) | Options for AlignTTS (alignment-free non-autoregressive TTS with mix density network). |
| [`AmazonPollyOptions`](/docs/reference/wiki/texttospeech/amazonpollyoptions/) | Options for AmazonPolly TTS API wrapper. |
| [`AmphionOptions`](/docs/reference/wiki/texttospeech/amphionoptions/) | Options for Amphion. |
| [`AudioLMOptions`](/docs/reference/wiki/texttospeech/audiolmoptions/) | Options for AudioLM TTS model. |
| [`AudioPaLMOptions`](/docs/reference/wiki/texttospeech/audiopalmoptions/) | Options for AudioPaLM TTS model. |
| [`AzureNeuralTTSOptions`](/docs/reference/wiki/texttospeech/azureneuralttsoptions/) | Options for Azure Neural TTS API wrapper. |
| [`BarkOptions`](/docs/reference/wiki/texttospeech/barkoptions/) | Options for Bark. |
| [`BigVGANOptions`](/docs/reference/wiki/texttospeech/bigvganoptions/) | Options for BigVGAN (universal vocoder with anti-aliased multi-periodicity composition and Snake activation). |
| [`CSMOptions`](/docs/reference/wiki/texttospeech/csmoptions/) | Options for CSM (Conversational Speech Model). |
| [`ChatTTSOptions`](/docs/reference/wiki/texttospeech/chatttsoptions/) | Options for ChatTTS. |
| [`ChatterboxOptions`](/docs/reference/wiki/texttospeech/chatterboxoptions/) | Options for Chatterbox. |
| [`CoMoSpeechOptions`](/docs/reference/wiki/texttospeech/comospeechoptions/) | Options for CoMoSpeech TTS model. |
| [`CodecTtsOptions`](/docs/reference/wiki/texttospeech/codecttsoptions/) | Base configuration options for codec-based TTS models. |
| [`CosyVoice2Options`](/docs/reference/wiki/texttospeech/cosyvoice2options/) | Options for CosyVoice2. |
| [`CosyVoice3Options`](/docs/reference/wiki/texttospeech/cosyvoice3options/) | Options for CosyVoice3 TTS model. |
| [`CosyVoiceCloneOptions`](/docs/reference/wiki/texttospeech/cosyvoicecloneoptions/) | Options for CosyVoiceClone voice cloning model. |
| [`CosyVoiceOptions`](/docs/reference/wiki/texttospeech/cosyvoiceoptions/) | Options for CosyVoice. |
| [`DeepVoice3Options`](/docs/reference/wiki/texttospeech/deepvoice3options/) | Options for Deep Voice 3 (fully convolutional attention-based TTS). |
| [`DiTToTTSOptions`](/docs/reference/wiki/texttospeech/dittottsoptions/) | Options for DiTToTTS TTS model. |
| [`DiaOptions`](/docs/reference/wiki/texttospeech/diaoptions/) | Options for Dia. |
| [`DiffWaveOptions`](/docs/reference/wiki/texttospeech/diffwaveoptions/) | Options for DiffWave (diffusion-based vocoder using denoising score matching). |
| [`E2TTSOptions`](/docs/reference/wiki/texttospeech/e2ttsoptions/) | Options for E2TTS. |
| [`E3TTSOptions`](/docs/reference/wiki/texttospeech/e3ttsoptions/) | Options for E3TTS TTS model. |
| [`ElevenLabsTTSOptions`](/docs/reference/wiki/texttospeech/elevenlabsttsoptions/) | Options for ElevenLabsTTS API wrapper. |
| [`EmotiVoiceOptions`](/docs/reference/wiki/texttospeech/emotivoiceoptions/) | Options for EmotiVoice TTS model. |
| [`EndToEndTtsOptions`](/docs/reference/wiki/texttospeech/endtoendttsoptions/) | Base options for end-to-end TTS models that generate waveforms directly from text. |
| [`F5TTSOptions`](/docs/reference/wiki/texttospeech/f5ttsoptions/) | Options for F5TTS. |
| [`FastSpeech2Options`](/docs/reference/wiki/texttospeech/fastspeech2options/) | Options for FastSpeech 2 (variance adaptor with pitch, energy, and duration predictors). |
| [`FastSpeechOptions`](/docs/reference/wiki/texttospeech/fastspeechoptions/) | Options for FastSpeech (non-autoregressive TTS with duration predictor). |
| [`FireRedTTSOptions`](/docs/reference/wiki/texttospeech/fireredttsoptions/) | Options for FireRedTTS. |
| [`FishSpeechOptions`](/docs/reference/wiki/texttospeech/fishspeechoptions/) | Options for FishSpeech (Fish Audio, 2024) dual-AR architecture with GFSQ. |
| [`FishSpeechV15Options`](/docs/reference/wiki/texttospeech/fishspeechv15options/) | Options for FishSpeechV15 TTS model. |
| [`ForwardTacotronOptions`](/docs/reference/wiki/texttospeech/forwardtacotronoptions/) | Options for Forward Tacotron (non-autoregressive Tacotron with duration predictor). |
| [`FreGradOptions`](/docs/reference/wiki/texttospeech/fregradoptions/) | Options for FreGrad (frequency-domain diffusion vocoder with DWT sub-band processing). |
| [`GLM4VoiceOptions`](/docs/reference/wiki/texttospeech/glm4voiceoptions/) | Options for GLM4Voice TTS model. |
| [`GPTSoVITSOptions`](/docs/reference/wiki/texttospeech/gptsovitsoptions/) | Options for GPTSoVITS. |
| [`GlowTTSOptions`](/docs/reference/wiki/texttospeech/glowttsoptions/) | Options for Glow-TTS (flow-based non-autoregressive TTS with monotonic alignment search). |
| [`GoogleCloudTTSOptions`](/docs/reference/wiki/texttospeech/googlecloudttsoptions/) | Options for GoogleCloudTTS TTS API wrapper. |
| [`GradTTSOptions`](/docs/reference/wiki/texttospeech/gradttsoptions/) | Options for Grad-TTS (diffusion-based acoustic model with score matching). |
| [`HiFiGANOptions`](/docs/reference/wiki/texttospeech/hifiganoptions/) | Options for HiFi-GAN (high-fidelity GAN-based vocoder with multi-receptive field fusion). |
| [`ISTFTNetOptions`](/docs/reference/wiki/texttospeech/istftnetoptions/) | Options for iSTFTNet (inverse STFT-based vocoder that outputs STFT coefficients then iSTFT). |
| [`IndexTTS2Options`](/docs/reference/wiki/texttospeech/indextts2options/) | Options for IndexTTS2 TTS model. |
| [`IndexTTSOptions`](/docs/reference/wiki/texttospeech/indexttsoptions/) | Options for IndexTTS. |
| [`KaniTTS2Options`](/docs/reference/wiki/texttospeech/kanitts2options/) | Options for KaniTTS2 TTS model. |
| [`KaniTTSOptions`](/docs/reference/wiki/texttospeech/kanittsoptions/) | Options for KaniTTS TTS model. |
| [`KokoroOptions`](/docs/reference/wiki/texttospeech/kokorooptions/) | Options for Kokoro (lightweight StyleTTS2-inspired TTS with style tokens and ISTFTNet decoder). |
| [`LlamaOmniOptions`](/docs/reference/wiki/texttospeech/llamaomnioptions/) | Options for LlamaOmni TTS model. |
| [`LlasaOptions`](/docs/reference/wiki/texttospeech/llasaoptions/) | Options for Llasa. |
| [`MARS5TTSOptions`](/docs/reference/wiki/texttospeech/mars5ttsoptions/) | Options for MARS5TTS. |
| [`MaskGCTOptions`](/docs/reference/wiki/texttospeech/maskgctoptions/) | Options for MaskGCT. |
| [`MatchaTTSOptions`](/docs/reference/wiki/texttospeech/matchattsoptions/) | Options for MatchaTTS TTS model. |
| [`MegaTTS2Options`](/docs/reference/wiki/texttospeech/megatts2options/) | Options for MegaTTS2 TTS model. |
| [`MegaTTS3Options`](/docs/reference/wiki/texttospeech/megatts3options/) | Options for MegaTTS3. |
| [`MegaTTSOptions`](/docs/reference/wiki/texttospeech/megattsoptions/) | Options for MegaTTS TTS model. |
| [`MelGANOptions`](/docs/reference/wiki/texttospeech/melganoptions/) | Options for MelGAN (lightweight GAN vocoder with no need for paired training data). |
| [`MeloTTSOptions`](/docs/reference/wiki/texttospeech/melottsoptions/) | Options for MeloTTS (multilingual VITS-based TTS with BERT-enhanced text processing and mixed-language support). |
| [`MetaVoice1BOptions`](/docs/reference/wiki/texttospeech/metavoice1boptions/) | Options for MetaVoice1B TTS model. |
| [`MinMoOptions`](/docs/reference/wiki/texttospeech/minmooptions/) | Options for MinMo TTS model. |
| [`MoshiOptions`](/docs/reference/wiki/texttospeech/moshioptions/) | Options for Moshi TTS model. |
| [`MultiBandMelGANOptions`](/docs/reference/wiki/texttospeech/multibandmelganoptions/) | Options for Multi-band MelGAN (multi-band signal decomposition for faster vocoding). |
| [`MurfOptions`](/docs/reference/wiki/texttospeech/murfoptions/) | Options for Murf TTS API wrapper. |
| [`NVIDIARivaTTSOptions`](/docs/reference/wiki/texttospeech/nvidiarivattsoptions/) | Options for NVIDIARivaTTS TTS model. |
| [`NaturalSpeech2Options`](/docs/reference/wiki/texttospeech/naturalspeech2options/) | Options for NaturalSpeech2 TTS model. |
| [`NaturalSpeech3Options`](/docs/reference/wiki/texttospeech/naturalspeech3options/) | Options for NaturalSpeech3 TTS model. |
| [`NaturalSpeechOptions`](/docs/reference/wiki/texttospeech/naturalspeechoptions/) | Options for NaturalSpeech TTS model. |
| [`OpenVoiceOptions`](/docs/reference/wiki/texttospeech/openvoiceoptions/) | Options for OpenVoice TTS model. |
| [`OpenVoiceV2Options`](/docs/reference/wiki/texttospeech/openvoicev2options/) | Options for OpenVoiceV2 TTS model. |
| [`OrpheusTTSOptions`](/docs/reference/wiki/texttospeech/orpheusttsoptions/) | Options for OrpheusTTS. |
| [`OuteTTSOptions`](/docs/reference/wiki/texttospeech/outettsoptions/) | Options for OuteTTS. |
| [`ParallelWaveGANOptions`](/docs/reference/wiki/texttospeech/parallelwaveganoptions/) | Options for Parallel WaveGAN (non-autoregressive GAN vocoder with multi-resolution STFT loss). |
| [`ParlerTTSOptions`](/docs/reference/wiki/texttospeech/parlerttsoptions/) | Options for ParlerTTS. |
| [`PhemeOptions`](/docs/reference/wiki/texttospeech/phemeoptions/) | Options for Pheme TTS model. |
| [`PiperOptions`](/docs/reference/wiki/texttospeech/piperoptions/) | Options for Piper (lightweight VITS-based TTS optimized for edge/embedded deployment). |
| [`PlayHTOptions`](/docs/reference/wiki/texttospeech/playhtoptions/) | Options for PlayHT TTS API wrapper. |
| [`PortaSpeechOptions`](/docs/reference/wiki/texttospeech/portaspeechoptions/) | Options for PortaSpeech (portable TTS with word-level prosody modeling and normalizing flow post-net). |
| [`PriorGradOptions`](/docs/reference/wiki/texttospeech/priorgradoptions/) | Options for PriorGrad (diffusion vocoder with data-dependent prior for adaptive noise). |
| [`ProDiffOptions`](/docs/reference/wiki/texttospeech/prodiffoptions/) | Options for ProDiff (progressive fast diffusion model for high-quality TTS). |
| [`PromptTTSOptions`](/docs/reference/wiki/texttospeech/promptttsoptions/) | Options for PromptTTS description-based TTS model. |
| [`SPEARTTSOptions`](/docs/reference/wiki/texttospeech/spearttsoptions/) | Options for SPEARTTS TTS model. |
| [`SeedTTSCloneOptions`](/docs/reference/wiki/texttospeech/seedttscloneoptions/) | Options for SeedTTSClone voice cloning model. |
| [`SeedTTSOptions`](/docs/reference/wiki/texttospeech/seedttsoptions/) | Options for SeedTTS. |
| [`SoundStormOptions`](/docs/reference/wiki/texttospeech/soundstormoptions/) | Options for SoundStorm (parallel MaskGIT-style audio generation with SoundStream tokens). |
| [`SparkTTSOptions`](/docs/reference/wiki/texttospeech/sparkttsoptions/) | Options for SparkTTS. |
| [`SpeechGPTOptions`](/docs/reference/wiki/texttospeech/speechgptoptions/) | Options for SpeechGPT TTS model. |
| [`SpeechT5Options`](/docs/reference/wiki/texttospeech/speecht5options/) | Options for SpeechT5 TTS model. |
| [`SpeedySpeechOptions`](/docs/reference/wiki/texttospeech/speedyspeechoptions/) | Options for SpeedySpeech (teacher-student distilled non-autoregressive TTS). |
| [`SpiritLMOptions`](/docs/reference/wiki/texttospeech/spiritlmoptions/) | Options for SpiritLM TTS model. |
| [`StepAudioOptions`](/docs/reference/wiki/texttospeech/stepaudiooptions/) | Options for StepAudio TTS model. |
| [`StyleTTS2Options`](/docs/reference/wiki/texttospeech/styletts2options/) | Options for StyleTTS2 TTS model. |
| [`StyleTTSOptions`](/docs/reference/wiki/texttospeech/stylettsoptions/) | Options for StyleTTS TTS model. |
| [`StyleTTSZSOptions`](/docs/reference/wiki/texttospeech/stylettszsoptions/) | Options for StyleTTSZS TTS model. |
| [`Tacotron2Options`](/docs/reference/wiki/texttospeech/tacotron2options/) | Options for Tacotron 2 (location-sensitive attention with WaveNet vocoder). |
| [`TacotronOptions`](/docs/reference/wiki/texttospeech/tacotronoptions/) | Options for Tacotron (attention-based seq2seq TTS). |
| [`TortoiseTTSOptions`](/docs/reference/wiki/texttospeech/tortoisettsoptions/) | Options for TortoiseTTS TTS model. |
| [`TransformerTTSOptions`](/docs/reference/wiki/texttospeech/transformerttsoptions/) | Options for Transformer TTS (multi-head self-attention acoustic model). |
| [`TtsModelOptions`](/docs/reference/wiki/texttospeech/ttsmodeloptions/) | Base configuration options for text-to-speech models. |
| [`UniAudioOptions`](/docs/reference/wiki/texttospeech/uniaudiooptions/) | Options for UniAudio TTS model. |
| [`UnivNetOptions`](/docs/reference/wiki/texttospeech/univnetoptions/) | Options for UnivNet (universal neural vocoder with multi-resolution spectrogram discriminator). |
| [`VALLE2Options`](/docs/reference/wiki/texttospeech/valle2options/) | Options for VALLE2 TTS model. |
| [`VALLEOptions`](/docs/reference/wiki/texttospeech/valleoptions/) | Options for VALL-E (neural codec language model with AR + NAR transformers for zero-shot TTS). |
| [`VALLEXCloneOptions`](/docs/reference/wiki/texttospeech/vallexcloneoptions/) | Options for VALLEXClone voice cloning model. |
| [`VALLEXOptions`](/docs/reference/wiki/texttospeech/vallexoptions/) | Options for VALLEX. |
| [`VITS2Options`](/docs/reference/wiki/texttospeech/vits2options/) | Options for VITS2 (improved VITS with duration discriminator, Gaussian mixture prior, and speaker-conditional flow). |
| [`VITSOptions`](/docs/reference/wiki/texttospeech/vitsoptions/) | Options for VITS (end-to-end TTS with conditional VAE, normalizing flows, and adversarial training). |
| [`VocoderOptions`](/docs/reference/wiki/texttospeech/vocoderoptions/) | Base configuration options for neural vocoder models. |
| [`VocosOptions`](/docs/reference/wiki/texttospeech/vocosoptions/) | Options for Vocos (ConvNeXt-based Fourier vocoder predicting STFT magnitude and instantaneous frequency). |
| [`VoiceCloningOptions`](/docs/reference/wiki/texttospeech/voicecloningoptions/) | Base options for voice cloning TTS models. |
| [`VoiceCraftOptions`](/docs/reference/wiki/texttospeech/voicecraftoptions/) | Options for VoiceCraft TTS model. |
| [`VoiceFlowOptions`](/docs/reference/wiki/texttospeech/voiceflowoptions/) | Options for VoiceFlow TTS model. |
| [`VoiceboxOptions`](/docs/reference/wiki/texttospeech/voiceboxoptions/) | Options for Voicebox TTS model. |
| [`WaveGlowOptions`](/docs/reference/wiki/texttospeech/waveglowoptions/) | Options for WaveGlow (flow-based vocoder combining Glow and WaveNet). |
| [`WaveGradOptions`](/docs/reference/wiki/texttospeech/wavegradoptions/) | Options for WaveGrad (gradient-based conditional waveform diffusion). |
| [`WaveNetOptions`](/docs/reference/wiki/texttospeech/wavenetoptions/) | Options for WaveNet (autoregressive dilated causal convolution vocoder). |
| [`WaveRNNOptions`](/docs/reference/wiki/texttospeech/wavernnoptions/) | Options for WaveRNN (efficient autoregressive vocoder with subscale generation). |
| [`WellSaidLabsOptions`](/docs/reference/wiki/texttospeech/wellsaidlabsoptions/) | Options for WellSaidLabs TTS API wrapper. |
| [`WhisperSpeechOptions`](/docs/reference/wiki/texttospeech/whisperspeechoptions/) | Options for WhisperSpeech. |
| [`XTTSv2CloneOptions`](/docs/reference/wiki/texttospeech/xttsv2cloneoptions/) | Options for XTTSv2Clone voice cloning model. |
| [`XTTSv2Options`](/docs/reference/wiki/texttospeech/xttsv2options/) | Options for XTTS v2 (GPT-2 based multilingual zero-shot TTS with VQ-VAE audio tokens). |
| [`YourTTSOptions`](/docs/reference/wiki/texttospeech/yourttsoptions/) | Options for YourTTS (multilingual zero-shot multi-speaker VITS variant with speaker and language conditioning). |
| [`ZonosOptions`](/docs/reference/wiki/texttospeech/zonosoptions/) | Options for Zonos. |

