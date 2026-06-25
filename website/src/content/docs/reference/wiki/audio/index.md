---
title: "Audio"
description: "All 279 public types in the AiDotNet.audio namespace, organized by kind."
section: "API Reference"
---

**279** public types in this namespace, organized by kind.

## Models & Types (148)

| Type | Summary |
|:-----|:--------|
| [`ACEStep<T>`](/docs/reference/wiki/audio/acestep/) | ACE-Step accelerated consistency-enhanced music generation model. |
| [`ASTModel<T>`](/docs/reference/wiki/audio/astmodel/) | AST (Audio Spectrogram Transformer) — a single-stream Vision-Transformer applied to log-mel spectrograms, trained for audio event classification and fingerprinting (Gong et al. |
| [`AST<T>`](/docs/reference/wiki/audio/ast/) | AST (Audio Spectrogram Transformer) model for audio event detection and classification. |
| [`AudioEvent`](/docs/reference/wiki/audio/audioevent/) | Represents a detected audio event. |
| [`AudioEventDetector<T>`](/docs/reference/wiki/audio/audioeventdetector/) | Audio event detection model for identifying sounds in audio (AudioSet-style). |
| [`AudioFlamingo2<T>`](/docs/reference/wiki/audio/audioflamingo2/) | Audio Flamingo 2 multimodal audio-language model for audio understanding with interleaved inputs. |
| [`AudioGenModel<T>`](/docs/reference/wiki/audio/audiogenmodel/) | AudioGen model for generating audio from text descriptions using neural audio codecs. |
| [`AudioGenResult<T>`](/docs/reference/wiki/audio/audiogenresult/) | Result of audio generation. |
| [`AudioLDMClassifier<T>`](/docs/reference/wiki/audio/audioldmclassifier/) | AudioLDM Classifier that repurposes AudioLDM's latent representations for audio event detection. |
| [`AudioLDMModel<T>`](/docs/reference/wiki/audio/audioldmmodel/) | AudioLDM (Audio Latent Diffusion Model) for generating audio from text descriptions. |
| [`AudioLM<T>`](/docs/reference/wiki/audio/audiolm/) | AudioLM hierarchical audio language model for high-quality audio generation. |
| [`AudioMAE<T>`](/docs/reference/wiki/audio/audiomae/) | Audio-MAE (Masked Autoencoders for Audio) model for audio classification. |
| [`AudioSep<T>`](/docs/reference/wiki/audio/audiosep/) | AudioSep - foundation model for open-vocabulary audio separation and sound event detection (Liu et al., ICML 2024). |
| [`AudioSuperResolution<T>`](/docs/reference/wiki/audio/audiosuperresolution/) | Audio Super-Resolution model for upsampling low-resolution audio to high-resolution (Kuleshov et al., 2017; Li et al., 2021). |
| [`BEATs<T>`](/docs/reference/wiki/audio/beats/) | BEATs (Audio Pre-Training with Acoustic Tokenizers) model for state-of-the-art audio event detection and classification. |
| [`BSRoFormer<T>`](/docs/reference/wiki/audio/bsroformer/) | BS-RoFormer (Band-Split Rotary Transformer) for music source separation. |
| [`BandSplitRNNEnhancer<T>`](/docs/reference/wiki/audio/bandsplitrnnenhancer/) | Band-Split RNN speech enhancement model (Luo and Yu, 2023). |
| [`BandSplitRNN<T>`](/docs/reference/wiki/audio/bandsplitrnn/) | BandSplitRNN for music source separation (Luo and Yu, 2023). |
| [`BasicPitch<T>`](/docs/reference/wiki/audio/basicpitch/) | Basic Pitch polyphonic music transcription model from Spotify. |
| [`BeatTracker<T>`](/docs/reference/wiki/audio/beattracker/) | Extracts beat and tempo information from audio. |
| [`BeatTrackingResult`](/docs/reference/wiki/audio/beattrackingresult/) | Result of beat tracking. |
| [`CAMPlusPlus<T>`](/docs/reference/wiki/audio/camplusplus/) | CAM++ (Context-Aware Masking Plus Plus) speaker verification model (Wang et al., 2023). |
| [`CLAPModel<T>`](/docs/reference/wiki/audio/clapmodel/) | CLAP (Contrastive Language-Audio Pretraining) — a dual-encoder neural network that learns to align audio and text representations in a shared embedding space via a contrastive objective. |
| [`CLAP<T>`](/docs/reference/wiki/audio/clap/) | CLAP (Contrastive Language-Audio Pre-training) model for zero-shot and fine-tuned audio classification. |
| [`CMGAN<T>`](/docs/reference/wiki/audio/cmgan/) | CMGAN (Conformer-based Metric GAN) for speech enhancement. |
| [`CREPE<T>`](/docs/reference/wiki/audio/crepe/) | CREPE (Convolutional Representation for Pitch Estimation) neural pitch detector. |
| [`CRNNEventDetector<T>`](/docs/reference/wiki/audio/crnneventdetector/) | CRNN (Convolutional Recurrent Neural Network) model for Sound Event Detection. |
| [`CTCDecoder<T>`](/docs/reference/wiki/audio/ctcdecoder/) | CTC (Connectionist Temporal Classification) decoder-based speech recognition model. |
| [`Canary<T>`](/docs/reference/wiki/audio/canary/) | Canary multilingual speech recognition and translation model from NVIDIA. |
| [`ChordRecognizer<T>`](/docs/reference/wiki/audio/chordrecognizer/) | Recognizes chords from audio using chromagram analysis. |
| [`ChordSegment`](/docs/reference/wiki/audio/chordsegment/) | Represents a chord segment in audio. |
| [`ChromaExtractor<T>`](/docs/reference/wiki/audio/chromaextractor/) | Extracts chromagram (pitch class profile) features from audio signals. |
| [`ChromaprintFingerprinter<T>`](/docs/reference/wiki/audio/chromaprintfingerprinter/) | Chromaprint-style audio fingerprinter based on chroma features. |
| [`Compressor<T>`](/docs/reference/wiki/audio/compressor/) | Dynamic range compressor effect. |
| [`ConformerFP<T>`](/docs/reference/wiki/audio/conformerfp/) | Conformer-based audio fingerprinting model combining self-attention with convolutions. |
| [`Conformer<T>`](/docs/reference/wiki/audio/conformer/) | Conformer speech recognition model (Gulati et al., 2020, Google). |
| [`ConstantQTransform<T>`](/docs/reference/wiki/audio/constantqtransform/) | Constant-Q Transform (CQT) for music analysis with logarithmic frequency resolution. |
| [`ConvTasNet<T>`](/docs/reference/wiki/audio/convtasnet/) | Conv-TasNet: A fully-convolutional time-domain audio separation network. |
| [`CosyVoice2<T>`](/docs/reference/wiki/audio/cosyvoice2/) | CosyVoice2 scalable streaming TTS model from Alibaba. |
| [`DAC<T>`](/docs/reference/wiki/audio/dac/) | Descript Audio Codec (DAC) - high-fidelity universal neural audio codec (Kumar et al., 2024, Descript). |
| [`DCCRN<T>`](/docs/reference/wiki/audio/dccrn/) | DCCRN - Deep Complex Convolution Recurrent Network for speech enhancement. |
| [`DannaSep<T>`](/docs/reference/wiki/audio/dannasep/) | Danna-Sep music source separation model using dual-path attention networks. |
| [`Data2Vec2<T>`](/docs/reference/wiki/audio/data2vec2/) | data2vec 2.0 self-supervised audio representation model. |
| [`DeepFilterNet<T>`](/docs/reference/wiki/audio/deepfilternet/) | DeepFilterNet - State-of-the-art deep filtering network for speech enhancement. |
| [`DemucsNoise<T>`](/docs/reference/wiki/audio/demucsnoise/) | Demucs for Noise - real-time noise suppression using the Demucs architecture (Defossez et al., 2020, Meta). |
| [`DiarizationResult`](/docs/reference/wiki/audio/diarizationresult/) | Result of speaker diarization. |
| [`EAT<T>`](/docs/reference/wiki/audio/eat/) | EAT (Efficient Audio Transformer) model for efficient audio event detection and classification. |
| [`ECAPATDNNLanguageIdentifier<T>`](/docs/reference/wiki/audio/ecapatdnnlanguageidentifier/) | ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network) for spoken language identification. |
| [`ECAPATDNNSpeaker<T>`](/docs/reference/wiki/audio/ecapatdnnspeaker/) | ECAPA-TDNN speaker verification and embedding extraction model. |
| [`Emotion2Vec<T>`](/docs/reference/wiki/audio/emotion2vec/) | emotion2vec universal speech emotion recognition model. |
| [`EnCodec<T>`](/docs/reference/wiki/audio/encodec/) | EnCodec neural audio codec from Meta for high-fidelity audio compression. |
| [`EnergyBasedVad<T>`](/docs/reference/wiki/audio/energybasedvad/) | Simple energy-based voice activity detector (algorithmic, no neural network). |
| [`EqBand<T>`](/docs/reference/wiki/audio/eqband/) | Represents a single EQ band with biquad filter. |
| [`FDYSED<T>`](/docs/reference/wiki/audio/fdysed/) | FDY-SED (Frequency Dynamic Sound Event Detection) model for DCASE-winning SED. |
| [`FRCRN<T>`](/docs/reference/wiki/audio/frcrn/) | FRCRN (Frequency Recurrence CRN) speech enhancement model (Zhao et al., ICASSP 2022). |
| [`FastConformer<T>`](/docs/reference/wiki/audio/fastconformer/) | Fast Conformer speech recognition model (Rekesh et al., 2023, NVIDIA NeMo). |
| [`FishSpeech<T>`](/docs/reference/wiki/audio/fishspeech/) | Fish Speech open-source multilingual TTS with zero-shot voice cloning. |
| [`FullSubNetPlus<T>`](/docs/reference/wiki/audio/fullsubnetplus/) | FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) for speech enhancement. |
| [`GenreClassificationResult`](/docs/reference/wiki/audio/genreclassificationresult/) | Result of genre classification. |
| [`GenreClassifier<T>`](/docs/reference/wiki/audio/genreclassifier/) | Music genre classification model. |
| [`GenreFeatures`](/docs/reference/wiki/audio/genrefeatures/) | Features extracted for genre classification. |
| [`HTDemucs<T>`](/docs/reference/wiki/audio/htdemucs/) | HTDemucs (Hybrid Transformer Demucs) for music source separation. |
| [`HTSAT<T>`](/docs/reference/wiki/audio/htsat/) | HTS-AT (Hierarchical Token-Semantic Audio Transformer) model for efficient audio classification. |
| [`HuBERTSER<T>`](/docs/reference/wiki/audio/hubertser/) | HuBERT-SER (HuBERT for Speech Emotion Recognition) model. |
| [`HuBERT<T>`](/docs/reference/wiki/audio/hubert/) | HuBERT (Hidden-Unit BERT) self-supervised speech representation model. |
| [`IdentificationResult`](/docs/reference/wiki/audio/identificationresult/) | Result of speaker identification. |
| [`KeyDetectionResult`](/docs/reference/wiki/audio/keydetectionresult/) | Result of key detection. |
| [`KeyDetector<T>`](/docs/reference/wiki/audio/keydetector/) | Detects the musical key of audio using chromagram analysis. |
| [`LocalizationResult`](/docs/reference/wiki/audio/localizationresult/) | Result of sound source localization. |
| [`MERT<T>`](/docs/reference/wiki/audio/mert/) | MERT self-supervised music understanding foundation model. |
| [`MPSENet<T>`](/docs/reference/wiki/audio/mpsenet/) | MP-SENet (Multi-Path Speech Enhancement Network) model (Lu et al., 2023). |
| [`MT3<T>`](/docs/reference/wiki/audio/mt3/) | MT3 (Multi-Track Music Transcription) model using T5-style encoder-decoder architecture. |
| [`MadmomBeatTracker<T>`](/docs/reference/wiki/audio/madmombeattracker/) | Madmom-style neural beat tracker using bidirectional RNNs. |
| [`MarbleNet<T>`](/docs/reference/wiki/audio/marblenet/) | MarbleNet lightweight 1D separable convolutional VAD model (NVIDIA NeMo). |
| [`MatchaTTS<T>`](/docs/reference/wiki/audio/matchatts/) | Matcha-TTS fast text-to-speech model using conditional flow matching (Mehta et al., 2024). |
| [`MelBandRoFormer<T>`](/docs/reference/wiki/audio/melbandroformer/) | MelBand-RoFormer for mel-band music source separation. |
| [`MelodyExtractor<T>`](/docs/reference/wiki/audio/melodyextractor/) | Neural Melody Extractor that identifies the primary melodic line from polyphonic audio. |
| [`MfccExtractor<T>`](/docs/reference/wiki/audio/mfccextractor/) | Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio signals. |
| [`MusicFlamingo<T>`](/docs/reference/wiki/audio/musicflamingo/) | Music Flamingo multimodal music-language model for music understanding and reasoning. |
| [`MusicGenModel<T>`](/docs/reference/wiki/audio/musicgenmodel/) | Meta's MusicGen model for generating music from text descriptions. |
| [`MusicSegment<T>`](/docs/reference/wiki/audio/musicsegment/) | Represents a labeled segment of music structure. |
| [`MusicSourceSeparator<T>`](/docs/reference/wiki/audio/musicsourceseparator/) | Music source separation model for separating audio into stems (vocals, drums, bass, other). |
| [`MusicStructureAnalyzer<T>`](/docs/reference/wiki/audio/musicstructureanalyzer/) | Music Structure Analyzer that segments songs into structural sections (intro, verse, chorus, etc.). |
| [`MusicTaggingTransformer<T>`](/docs/reference/wiki/audio/musictaggingtransformer/) | Music Tagging Transformer for multi-label music tag prediction (genre, mood, instrument, era). |
| [`NeuralNoiseReducer<T>`](/docs/reference/wiki/audio/neuralnoisereducer/) | Neural network-based noise reducer for high-quality audio enhancement. |
| [`NeuralParametricEQ<T>`](/docs/reference/wiki/audio/neuralparametriceq/) | Neural Parametric EQ model for automatic equalization (Steinmetz et al., 2022). |
| [`OnsetsAndFrames<T>`](/docs/reference/wiki/audio/onsetsandframes/) | Onsets and Frames piano transcription model from Google Magenta. |
| [`PANNsModel<T>`](/docs/reference/wiki/audio/pannsmodel/) | PANNs (Pretrained Audio Neural Networks) audio classifier — a CNN14-style convolutional backbone over log-mel spectrograms, trained for AudioSet tagging (Kong et al. |
| [`PANNs<T>`](/docs/reference/wiki/audio/panns/) | PANNs (Pre-trained Audio Neural Networks) CNN14 model for audio classification. |
| [`ParametricEqualizer<T>`](/docs/reference/wiki/audio/parametricequalizer/) | Multi-band parametric equalizer effect. |
| [`PeakNetFP<T>`](/docs/reference/wiki/audio/peaknetfp/) | PeakNetFP spectral peak-based neural audio fingerprinting model. |
| [`Pengi<T>`](/docs/reference/wiki/audio/pengi/) | Pengi audio language model that frames all audio tasks as text-generation. |
| [`PyAnnote<T>`](/docs/reference/wiki/audio/pyannote/) | pyannote 3.x end-to-end speaker diarization model. |
| [`QuailVad<T>`](/docs/reference/wiki/audio/quailvad/) | Quail VAD - lightweight voice activity detection optimized for on-device deployment. |
| [`Qwen2Audio<T>`](/docs/reference/wiki/audio/qwen2audio/) | Qwen2-Audio multimodal audio-language model for audio understanding and reasoning. |
| [`RNNTransducer<T>`](/docs/reference/wiki/audio/rnntransducer/) | RNN-Transducer (RNN-T) streaming speech recognition model. |
| [`Reverb<T>`](/docs/reference/wiki/audio/reverb/) | Algorithmic reverb effect using Schroeder-Moorer structure. |
| [`RoomImpulseResponse<T>`](/docs/reference/wiki/audio/roomimpulseresponse/) | Neural Room Impulse Response estimation model for acoustic analysis and dereverberation. |
| [`SALMONN<T>`](/docs/reference/wiki/audio/salmonn/) | SALMONN dual-encoder audio-language model for speech and audio understanding. |
| [`SCNet<T>`](/docs/reference/wiki/audio/scnet/) | SCNet (Sparse Compression Network) for music source separation (Tong et al., 2024). |
| [`SceneClassificationResult`](/docs/reference/wiki/audio/sceneclassificationresult/) | Result of acoustic scene classification. |
| [`SceneClassifier<T>`](/docs/reference/wiki/audio/sceneclassifier/) | Acoustic scene classification model for identifying recording environments. |
| [`SceneFeatures`](/docs/reference/wiki/audio/scenefeatures/) | Features extracted for scene classification. |
| [`SeparationResult<T>`](/docs/reference/wiki/audio/separationresult/) | Result of music source separation containing individual stems. |
| [`SileroVad<T>`](/docs/reference/wiki/audio/silerovad/) | Silero Voice Activity Detection model - high accuracy neural network VAD. |
| [`SoundLocalizer<T>`](/docs/reference/wiki/audio/soundlocalizer/) | Sound source localization using microphone arrays. |
| [`SoundStream<T>`](/docs/reference/wiki/audio/soundstream/) | SoundStream neural audio codec from Google for efficient audio compression. |
| [`SpeakerDiarizer<T>`](/docs/reference/wiki/audio/speakerdiarizer/) | Performs speaker diarization (who spoke when) on audio recordings. |
| [`SpeakerEmbeddingExtractor<T>`](/docs/reference/wiki/audio/speakerembeddingextractor/) | Extracts speaker embeddings (d-vectors) from audio for speaker recognition. |
| [`SpeakerEmbedding<T>`](/docs/reference/wiki/audio/speakerembedding/) | Represents a speaker embedding vector. |
| [`SpeakerLM<T>`](/docs/reference/wiki/audio/speakerlm/) | SpeakerLM language-model-based speaker diarization and verification model. |
| [`SpeakerMatch`](/docs/reference/wiki/audio/speakermatch/) | A speaker match with score. |
| [`SpeakerTurn`](/docs/reference/wiki/audio/speakerturn/) | Represents a speaker turn in diarization output. |
| [`SpeakerVerifier<T>`](/docs/reference/wiki/audio/speakerverifier/) | Verifies speaker identity by comparing embeddings against enrolled speakers. |
| [`SpectralFeatureExtractor<T>`](/docs/reference/wiki/audio/spectralfeatureextractor/) | Extracts spectral features from audio signals including centroid, bandwidth, rolloff, and flux. |
| [`SpectralSubtractionEnhancer<T>`](/docs/reference/wiki/audio/spectralsubtractionenhancer/) | Audio enhancer using spectral subtraction for noise reduction. |
| [`SpectrogramFingerprinter<T>`](/docs/reference/wiki/audio/spectrogramfingerprinter/) | Spectrogram peak-based audio fingerprinter (Shazam-style). |
| [`SpeechEmotionRecognizer<T>`](/docs/reference/wiki/audio/speechemotionrecognizer/) | Neural network-based speech emotion recognition model that classifies emotional states from audio. |
| [`SpikingFullSubNet<T>`](/docs/reference/wiki/audio/spikingfullsubnet/) | Spiking-FullSubNet speech enhancement model using spiking neural networks. |
| [`StableAudioModel<T>`](/docs/reference/wiki/audio/stableaudiomodel/) | Stable Audio model for generating high-quality audio from text descriptions. |
| [`StyleTTS2<T>`](/docs/reference/wiki/audio/styletts2/) | StyleTTS 2 text-to-speech model (Li et al., 2023). |
| [`TFGridNet<T>`](/docs/reference/wiki/audio/tfgridnet/) | TF-GridNet (Time-Frequency GridNet) for speech enhancement and separation. |
| [`Tacotron2Model<T>`](/docs/reference/wiki/audio/tacotron2model/) | Tacotron2 attention-based text-to-speech model. |
| [`Tempogram<T>`](/docs/reference/wiki/audio/tempogram/) | Neural Tempogram model for tempo estimation over time. |
| [`TitaNet<T>`](/docs/reference/wiki/audio/titanet/) | TitaNet speaker verification and embedding extraction model. |
| [`TtsModel<T>`](/docs/reference/wiki/audio/ttsmodel/) | Text-to-speech model for synthesizing speech from text. |
| [`TtsPreprocessor`](/docs/reference/wiki/audio/ttspreprocessor/) | Preprocesses text for text-to-speech synthesis. |
| [`TtsResult<T>`](/docs/reference/wiki/audio/ttsresult/) | Result of text-to-speech synthesis. |
| [`VALLE<T>`](/docs/reference/wiki/audio/valle/) | VALL-E zero-shot text-to-speech via neural codec language modeling. |
| [`VITSModel<T>`](/docs/reference/wiki/audio/vitsmodel/) | VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) model. |
| [`VerificationResult`](/docs/reference/wiki/audio/verificationresult/) | Result of speaker verification. |
| [`VoiceCraft<T>`](/docs/reference/wiki/audio/voicecraft/) | VoiceCraft neural codec language model for speech editing and zero-shot TTS. |
| [`VoxLingua107Identifier<T>`](/docs/reference/wiki/audio/voxlingua107identifier/) | VoxLingua107 language identifier supporting 107 languages. |
| [`Wav2Small<T>`](/docs/reference/wiki/audio/wav2small/) | Wav2Small lightweight speech emotion recognition model (Gomez-Alanis et al., 2024). |
| [`Wav2Vec2LanguageIdentifier<T>`](/docs/reference/wiki/audio/wav2vec2languageidentifier/) | Wav2Vec2 model fine-tuned for spoken language identification. |
| [`Wav2Vec2Model<T>`](/docs/reference/wiki/audio/wav2vec2model/) | Wav2Vec2 self-supervised speech recognition model. |
| [`Wav2Vec2<T>`](/docs/reference/wiki/audio/wav2vec2/) | wav2vec 2.0 self-supervised speech representation model. |
| [`WavLMSpeaker<T>`](/docs/reference/wiki/audio/wavlmspeaker/) | WavLM-based speaker verification and embedding extraction model (Chen et al., 2022). |
| [`WavLM<T>`](/docs/reference/wiki/audio/wavlm/) | WavLM self-supervised speech representation model from Microsoft. |
| [`WebRTCVad<T>`](/docs/reference/wiki/audio/webrtcvad/) | Neural WebRTC VAD model for low-latency voice activity detection. |
| [`WhisperModel<T>`](/docs/reference/wiki/audio/whispermodel/) | Whisper automatic speech recognition model for transcribing audio to text. |
| [`WhisperResult`](/docs/reference/wiki/audio/whisperresult/) | Result of Whisper transcription. |
| [`WhisperSegment`](/docs/reference/wiki/audio/whispersegment/) | A segment of transcribed speech with timing. |
| [`WhisperTokenizer`](/docs/reference/wiki/audio/whispertokenizer/) | Tokenizer for Whisper speech recognition model. |
| [`WhisperWord`](/docs/reference/wiki/audio/whisperword/) | A transcribed word with timing information. |
| [`YinPitchDetector<T>`](/docs/reference/wiki/audio/yinpitchdetector/) | YIN pitch detection algorithm implementation. |
| [`YuE<T>`](/docs/reference/wiki/audio/yue/) | YuE full-song music generation model with vocals and accompaniment. |
| [`Zipformer<T>`](/docs/reference/wiki/audio/zipformer/) | Zipformer speech recognition model (Yao et al., 2023, Next-gen Kaldi). |

## Base Classes (10)

| Type | Summary |
|:-----|:--------|
| [`AudioClassifierBase<T>`](/docs/reference/wiki/audio/audioclassifierbase/) | Base class for audio classification models (genre, event detection, scene classification). |
| [`AudioEffectBase<T>`](/docs/reference/wiki/audio/audioeffectbase/) | Base class for audio effects processors. |
| [`AudioEnhancerBase<T>`](/docs/reference/wiki/audio/audioenhancerbase/) | Base class for algorithmic audio enhancement (non-neural network based). |
| [`AudioFeatureExtractorBase<T>`](/docs/reference/wiki/audio/audiofeatureextractorbase/) | Base class for audio feature extractors providing common functionality. |
| [`AudioFingerprinterBase<T>`](/docs/reference/wiki/audio/audiofingerprinterbase/) | Base class for audio fingerprinting algorithms. |
| [`AudioNeuralNetworkBase<T>`](/docs/reference/wiki/audio/audioneuralnetworkbase/) | Base class for audio-focused neural networks that can operate in both ONNX inference and native training modes. |
| [`MusicAnalysisBase<T>`](/docs/reference/wiki/audio/musicanalysisbase/) | Base class for music analysis algorithms (beat tracking, chord recognition, key detection). |
| [`PitchDetectorBase<T>`](/docs/reference/wiki/audio/pitchdetectorbase/) | Base class for pitch detection implementations. |
| [`SpeakerRecognitionBase<T>`](/docs/reference/wiki/audio/speakerrecognitionbase/) | Base class for speaker recognition models (embedding extraction, verification, diarization). |
| [`VoiceActivityDetectorBase<T>`](/docs/reference/wiki/audio/voiceactivitydetectorbase/) | Base class for algorithmic voice activity detection implementations (non-neural network). |

## Enums (13)

| Type | Summary |
|:-----|:--------|
| [`AudioGenModelSize`](/docs/reference/wiki/audio/audiogenmodelsize/) | Available AudioGen model sizes. |
| [`AudioLDMModelSize`](/docs/reference/wiki/audio/audioldmmodelsize/) | Specifies the size variant of the AudioLDM model. |
| [`EqFilterType`](/docs/reference/wiki/audio/eqfiltertype/) | Types of EQ filters. |
| [`KeyMode`](/docs/reference/wiki/audio/keymode/) | Musical key mode. |
| [`LocalizationAlgorithm`](/docs/reference/wiki/audio/localizationalgorithm/) | Sound source localization algorithms. |
| [`MusicGenModelSize`](/docs/reference/wiki/audio/musicgenmodelsize/) | Specifies the size variant of the MusicGen model. |
| [`PANNsArchitecture`](/docs/reference/wiki/audio/pannsarchitecture/) | PANNs architecture variants (legacy enum — the new code uses `PANNsModelOptions` with explicit dim / depth fields). |
| [`SpectralFeatureType`](/docs/reference/wiki/audio/spectralfeaturetype/) | Types of spectral features that can be extracted. |
| [`StableAudioModelSize`](/docs/reference/wiki/audio/stableaudiomodelsize/) | Specifies the size variant of the Stable Audio model. |
| [`TtsModelType`](/docs/reference/wiki/audio/ttsmodeltype/) | Available TTS model types. |
| [`VocoderType`](/docs/reference/wiki/audio/vocodertype/) | Available vocoder types. |
| [`WhisperModelSize`](/docs/reference/wiki/audio/whispermodelsize/) | Available Whisper model sizes. |
| [`WindowType`](/docs/reference/wiki/audio/windowtype/) | Window types for spectral analysis. |

## Options & Configuration (108)

| Type | Summary |
|:-----|:--------|
| [`ACEStepOptions`](/docs/reference/wiki/audio/acestepoptions/) | Configuration options for the ACE-Step model. |
| [`ASTOptions`](/docs/reference/wiki/audio/astoptions/) | Configuration options for the AST (Audio Spectrogram Transformer) model. |
| [`AudioEventDetectorOptions`](/docs/reference/wiki/audio/audioeventdetectoroptions/) | Options for audio event detection. |
| [`AudioFlamingo2Options`](/docs/reference/wiki/audio/audioflamingo2options/) | Configuration options for the Audio Flamingo 2 model. |
| [`AudioGenOptions`](/docs/reference/wiki/audio/audiogenoptions/) | Configuration options for audio generation models. |
| [`AudioLDMClassifierOptions`](/docs/reference/wiki/audio/audioldmclassifieroptions/) | Configuration options for the AudioLDM Classifier model. |
| [`AudioLDMOptions`](/docs/reference/wiki/audio/audioldmoptions/) | Configuration options for AudioLDM text-to-audio generation. |
| [`AudioLMOptions`](/docs/reference/wiki/audio/audiolmoptions/) | Configuration options for the AudioLM audio generation model. |
| [`AudioMAEOptions`](/docs/reference/wiki/audio/audiomaeoptions/) | Configuration options for the Audio-MAE (Masked Autoencoders for Audio) model. |
| [`AudioSepOptions`](/docs/reference/wiki/audio/audiosepoptions/) | Configuration options for the AudioSep (Audio Separation with Natural Language Queries) model. |
| [`AudioSuperResolutionOptions`](/docs/reference/wiki/audio/audiosuperresolutionoptions/) | Configuration options for the Audio Super-Resolution model. |
| [`BEATsOptions`](/docs/reference/wiki/audio/beatsoptions/) | Configuration options for the BEATs (Audio Pre-Training with Acoustic Tokenizers) model. |
| [`BSRoFormerOptions`](/docs/reference/wiki/audio/bsroformeroptions/) | Configuration options for the BS-RoFormer (Band-Split Rotary Transformer) model. |
| [`BandSplitRNNEnhancerOptions`](/docs/reference/wiki/audio/bandsplitrnnenhanceroptions/) | Configuration options for the Band-Split RNN enhancement model. |
| [`BandSplitRNNOptions`](/docs/reference/wiki/audio/bandsplitrnnoptions/) | Configuration options for the BandSplitRNN source separation model. |
| [`BasicPitchOptions`](/docs/reference/wiki/audio/basicpitchoptions/) | Configuration options for the Basic Pitch multi-pitch detection model. |
| [`BeatTrackerOptions`](/docs/reference/wiki/audio/beattrackeroptions/) | Configuration options for beat tracking. |
| [`CAMPlusPlusOptions`](/docs/reference/wiki/audio/camplusplusoptions/) | Configuration options for the CAM++ (Context-Aware Masking Plus Plus) speaker model. |
| [`CLAPOptions`](/docs/reference/wiki/audio/clapoptions/) | Configuration options for the CLAP (Contrastive Language-Audio Pre-training) model. |
| [`CMGANOptions`](/docs/reference/wiki/audio/cmganoptions/) | Configuration options for the CMGAN (Conformer-based Metric GAN) speech enhancement model. |
| [`CREPEOptions`](/docs/reference/wiki/audio/crepeoptions/) | Configuration options for the CREPE (Convolutional Representation for Pitch Estimation) model. |
| [`CRNNEventDetectorOptions`](/docs/reference/wiki/audio/crnneventdetectoroptions/) | Configuration options for the CRNN (Convolutional Recurrent Neural Network) Sound Event Detector. |
| [`CTCDecoderOptions`](/docs/reference/wiki/audio/ctcdecoderoptions/) | Configuration options for the CTC Decoder speech recognition model. |
| [`CanaryOptions`](/docs/reference/wiki/audio/canaryoptions/) | Configuration options for the Canary model. |
| [`ChordRecognizerOptions`](/docs/reference/wiki/audio/chordrecognizeroptions/) | Configuration options for chord recognition. |
| [`ChromaOptions`](/docs/reference/wiki/audio/chromaoptions/) | Options for chroma feature extraction. |
| [`ChromaprintOptions`](/docs/reference/wiki/audio/chromaprintoptions/) | Configuration options for Chromaprint fingerprinting. |
| [`ConformerFPOptions`](/docs/reference/wiki/audio/conformerfpoptions/) | Configuration options for the Conformer-based audio fingerprinting model. |
| [`ConformerOptions`](/docs/reference/wiki/audio/conformeroptions/) | Configuration options for the Conformer speech recognition model. |
| [`CosyVoice2Options`](/docs/reference/wiki/audio/cosyvoice2options/) | Configuration options for the CosyVoice2 model. |
| [`DACOptions`](/docs/reference/wiki/audio/dacoptions/) | Configuration options for the Descript Audio Codec (DAC) model. |
| [`DannaSepOptions`](/docs/reference/wiki/audio/dannasepoptions/) | Configuration options for the Danna-Sep (Dual-path Attention Neural Network Audio Separator) model. |
| [`Data2Vec2Options`](/docs/reference/wiki/audio/data2vec2options/) | Configuration options for the data2vec 2.0 self-supervised audio foundation model. |
| [`DemucsNoiseOptions`](/docs/reference/wiki/audio/demucsnoiseoptions/) | Configuration options for the Demucs for Noise model. |
| [`EATOptions`](/docs/reference/wiki/audio/eatoptions/) | Configuration options for the EAT (Efficient Audio Transformer) model. |
| [`ECAPATDNNOptions`](/docs/reference/wiki/audio/ecapatdnnoptions/) | Configuration options specific to ECAPA-TDNN language identification. |
| [`ECAPATDNNSpeakerOptions`](/docs/reference/wiki/audio/ecapatdnnspeakeroptions/) | Configuration options for the ECAPA-TDNN speaker verification and embedding model. |
| [`Emotion2VecOptions`](/docs/reference/wiki/audio/emotion2vecoptions/) | Configuration options for the emotion2vec speech emotion recognition model. |
| [`EnCodecOptions`](/docs/reference/wiki/audio/encodecoptions/) | Configuration options for the EnCodec neural audio codec model. |
| [`FDYSEDOptions`](/docs/reference/wiki/audio/fdysedoptions/) | Configuration options for the FDY-SED (Frequency Dynamic Sound Event Detection) model. |
| [`FRCRNOptions`](/docs/reference/wiki/audio/frcrnoptions/) | Configuration options for the FRCRN (Frequency Recurrence CRN) model. |
| [`FastConformerOptions`](/docs/reference/wiki/audio/fastconformeroptions/) | Configuration options for the Fast Conformer speech recognition model. |
| [`FishSpeechOptions`](/docs/reference/wiki/audio/fishspeechoptions/) | Configuration options for the Fish Speech TTS model. |
| [`FullSubNetPlusOptions`](/docs/reference/wiki/audio/fullsubnetplusoptions/) | Configuration options for the FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) model. |
| [`GenreClassifierOptions`](/docs/reference/wiki/audio/genreclassifieroptions/) | Options for genre classification. |
| [`GraFPrintOptions`](/docs/reference/wiki/audio/grafprintoptions/) | Configuration options for the GraFPrint graph-based audio fingerprinting model. |
| [`HTDemucsOptions`](/docs/reference/wiki/audio/htdemucsoptions/) | Configuration options for the HTDemucs (Hybrid Transformer Demucs) model. |
| [`HTSATOptions`](/docs/reference/wiki/audio/htsatoptions/) | Configuration options for the HTS-AT (Hierarchical Token-Semantic Audio Transformer) model. |
| [`HuBERTOptions`](/docs/reference/wiki/audio/hubertoptions/) | Configuration options for the HuBERT (Hidden-Unit BERT) self-supervised speech model. |
| [`HuBERTSEROptions`](/docs/reference/wiki/audio/hubertseroptions/) | Configuration options for the HuBERT-based Speech Emotion Recognition model. |
| [`KeyDetectorOptions`](/docs/reference/wiki/audio/keydetectoroptions/) | Configuration options for key detection. |
| [`LanguageIdentifierOptions`](/docs/reference/wiki/audio/languageidentifieroptions/) | Configuration options for language identification models. |
| [`MERTOptions`](/docs/reference/wiki/audio/mertoptions/) | Configuration options for the MERT music understanding foundation model. |
| [`MPSENetOptions`](/docs/reference/wiki/audio/mpsenetoptions/) | Configuration options for the MP-SENet (Multi-Path Speech Enhancement Network) model. |
| [`MT3Options`](/docs/reference/wiki/audio/mt3options/) | Configuration options for the MT3 (Multi-Track Music Transcription) model. |
| [`MadmomBeatTrackerOptions`](/docs/reference/wiki/audio/madmombeattrackeroptions/) | Configuration options for the Madmom-style neural beat tracker. |
| [`MarbleNetOptions`](/docs/reference/wiki/audio/marblenetoptions/) | Configuration options for the MarbleNet voice activity detection model. |
| [`MatchaTTSOptions`](/docs/reference/wiki/audio/matchattsoptions/) | Configuration options for the Matcha-TTS model. |
| [`MelBandRoFormerOptions`](/docs/reference/wiki/audio/melbandroformeroptions/) | Configuration options for the MelBand-RoFormer model. |
| [`MelodyExtractorOptions`](/docs/reference/wiki/audio/melodyextractoroptions/) | Configuration options for the neural Melody Extraction model. |
| [`MfccOptions`](/docs/reference/wiki/audio/mfccoptions/) | Options for MFCC extraction. |
| [`MusicFlamingoOptions`](/docs/reference/wiki/audio/musicflamingooptions/) | Configuration options for the Music Flamingo model. |
| [`MusicGenOptions`](/docs/reference/wiki/audio/musicgenoptions/) | Configuration options for MusicGen text-to-music generation. |
| [`MusicStructureAnalyzerOptions`](/docs/reference/wiki/audio/musicstructureanalyzeroptions/) | Configuration options for the Music Structure Analyzer model. |
| [`MusicTaggingTransformerOptions`](/docs/reference/wiki/audio/musictaggingtransformeroptions/) | Configuration options for the Music Tagging Transformer model. |
| [`NeuralFPOptions`](/docs/reference/wiki/audio/neuralfpoptions/) | Configuration options for the Neural Audio Fingerprint (NeuralFP) model. |
| [`NeuralParametricEQOptions`](/docs/reference/wiki/audio/neuralparametriceqoptions/) | Configuration options for the Neural Parametric EQ model. |
| [`OnsetsAndFramesOptions`](/docs/reference/wiki/audio/onsetsandframesoptions/) | Configuration options for the Onsets and Frames piano transcription model. |
| [`PANNsOptions`](/docs/reference/wiki/audio/pannsoptions/) | Configuration options for the PANNs (Pre-trained Audio Neural Networks) model. |
| [`PeakNetFPOptions`](/docs/reference/wiki/audio/peaknetfpoptions/) | Configuration options for the PeakNetFP spectral peak-based fingerprinting model. |
| [`PengiOptions`](/docs/reference/wiki/audio/pengioptions/) | Configuration options for the Pengi model. |
| [`PyAnnoteOptions`](/docs/reference/wiki/audio/pyannoteoptions/) | Configuration options for the pyannote 3.x speaker diarization model. |
| [`QuailVadOptions`](/docs/reference/wiki/audio/quailvadoptions/) | Configuration options for the Quail VAD model. |
| [`Qwen2AudioOptions`](/docs/reference/wiki/audio/qwen2audiooptions/) | Configuration options for the Qwen2-Audio multimodal audio-language model. |
| [`RNNTransducerOptions`](/docs/reference/wiki/audio/rnntransduceroptions/) | Configuration options for the RNN-Transducer (RNN-T) speech recognition model. |
| [`RoomImpulseResponseOptions`](/docs/reference/wiki/audio/roomimpulseresponseoptions/) | Configuration options for the Room Impulse Response (RIR) estimation model. |
| [`SALMONNOptions`](/docs/reference/wiki/audio/salmonnoptions/) | Configuration options for the SALMONN multimodal audio-language model. |
| [`SCNetOptions`](/docs/reference/wiki/audio/scnetoptions/) | Configuration options for the SCNet (Sparse Compression Network) source separation model. |
| [`SceneClassifierOptions`](/docs/reference/wiki/audio/sceneclassifieroptions/) | Options for acoustic scene classification. |
| [`SoundLocalizerOptions`](/docs/reference/wiki/audio/soundlocalizeroptions/) | Options for sound source localization. |
| [`SoundStreamOptions`](/docs/reference/wiki/audio/soundstreamoptions/) | Configuration options for the SoundStream neural audio codec model. |
| [`SourceSeparationOptions`](/docs/reference/wiki/audio/sourceseparationoptions/) | Options for music source separation. |
| [`SpeakerDiarizerOptions`](/docs/reference/wiki/audio/speakerdiarizeroptions/) | Configuration options for speaker diarization. |
| [`SpeakerEmbeddingOptions`](/docs/reference/wiki/audio/speakerembeddingoptions/) | Configuration options for speaker embedding extraction. |
| [`SpeakerLMOptions`](/docs/reference/wiki/audio/speakerlmoptions/) | Configuration options for the SpeakerLM model. |
| [`SpeakerVerifierOptions`](/docs/reference/wiki/audio/speakerverifieroptions/) | Configuration options for speaker verification. |
| [`SpectralFeatureOptions`](/docs/reference/wiki/audio/spectralfeatureoptions/) | Options for spectral feature extraction. |
| [`SpectrogramFingerprintOptions`](/docs/reference/wiki/audio/spectrogramfingerprintoptions/) | Configuration options for spectrogram fingerprinting. |
| [`SpikingFullSubNetOptions`](/docs/reference/wiki/audio/spikingfullsubnetoptions/) | Configuration options for the Spiking-FullSubNet model. |
| [`StableAudioOptions`](/docs/reference/wiki/audio/stableaudiooptions/) | Configuration options for Stable Audio generation. |
| [`StyleTTS2Options`](/docs/reference/wiki/audio/styletts2options/) | Configuration options for the StyleTTS 2 text-to-speech model. |
| [`TFGridNetOptions`](/docs/reference/wiki/audio/tfgridnetoptions/) | Configuration options for the TF-GridNet (Time-Frequency GridNet) speech enhancement model. |
| [`TempogramOptions`](/docs/reference/wiki/audio/tempogramoptions/) | Configuration options for the neural Tempogram tempo estimation model. |
| [`TitaNetOptions`](/docs/reference/wiki/audio/titanetoptions/) | Configuration options for the TitaNet speaker verification and embedding model. |
| [`TtsOptions`](/docs/reference/wiki/audio/ttsoptions/) | Configuration options for text-to-speech models. |
| [`VALLEOptions`](/docs/reference/wiki/audio/valleoptions/) | Configuration options for the VALL-E zero-shot TTS model. |
| [`VoiceCraftOptions`](/docs/reference/wiki/audio/voicecraftoptions/) | Configuration options for the VoiceCraft speech editing and generation model. |
| [`VoxLingua107Options`](/docs/reference/wiki/audio/voxlingua107options/) | Configuration options specific to VoxLingua107 language identification. |
| [`Wav2SmallOptions`](/docs/reference/wiki/audio/wav2smalloptions/) | Configuration options for the Wav2Small speech emotion recognition model. |
| [`Wav2Vec2LidOptions`](/docs/reference/wiki/audio/wav2vec2lidoptions/) | Configuration options specific to Wav2Vec2 language identification. |
| [`Wav2Vec2Options`](/docs/reference/wiki/audio/wav2vec2options/) | Configuration options for the wav2vec 2.0 self-supervised speech model. |
| [`WavLMOptions`](/docs/reference/wiki/audio/wavlmoptions/) | Configuration options for the WavLM self-supervised speech model. |
| [`WavLMSEROptions`](/docs/reference/wiki/audio/wavlmseroptions/) | Configuration options for the WavLM-SER speech emotion recognition model. |
| [`WavLMSpeakerOptions`](/docs/reference/wiki/audio/wavlmspeakeroptions/) | Configuration options for the WavLM Speaker verification and embedding model. |
| [`WebRTCVadOptions`](/docs/reference/wiki/audio/webrtcvadoptions/) | Configuration options for the WebRTC VAD neural model. |
| [`WhisperOptions`](/docs/reference/wiki/audio/whisperoptions/) | Configuration options for the Whisper speech recognition model. |
| [`YuEOptions`](/docs/reference/wiki/audio/yueoptions/) | Configuration options for the YuE music generation model. |
| [`ZipformerOptions`](/docs/reference/wiki/audio/zipformeroptions/) | Configuration options for the Zipformer speech recognition model. |

