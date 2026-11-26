Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
# AI Audiobook Studio

AI Audiobook Studio runs locally on Windows. The core generator uses Qwen3-TTS 0.6B CustomVoice with Ryan, the narrator selected in Phase 1.

## Start the studio

Double-click `start-studio.cmd` or run `.\start-studio.ps1` in PowerShell. The browser opens at `http://127.0.0.1:7860`. If that port is in use, run `.\start-studio.ps1 --port 7861`.

1. In **Library**, upload a local TXT edition or paste text. Enter title and author if desired. Review the chapter list and extracted text before generating. The importer recognizes common Chapter, Part, Book, Prologue, Epilogue, and standalone uppercase numbered headings. It skips a front-matter contents list when its entries repeat as headings in the body. It removes the Gutenberg wrapper only when both explicit markers are present.
2. In **Narration**, select **Start or resume**. The worker saves one raw WAV per segment and records its progress in SQLite. The stats show recorded generation time, saved audio length, average speed, completed words and chapters, and estimated finished audiobook length. The remaining-time estimate uses recent prose segments after enough have finished. **Load latest book audio** plays the selected book's latest saved segments at their original synthesis pace. **Pause** and **Cancel** finish the in-flight segment. Resume continues at the next pending segment, including after restarting Windows. A failed segment is tried up to three times.
3. In **Export**, select the full audiobook or a completed chapter and choose 192, 256, or 320 kbps. The progress bar counts encoded segments and shows when FFmpeg is finalizing the file. The default target pace is 145 WPM. Export uses pitch-preserving FFmpeg tempo adjustment and inserts 350 ms between paragraphs and one second between chapters. The pace is approximate because source clips include model pauses. Raw WAVs remain in the library. Confirm rights before publishing.

The local app binds to loopback. Book data, WAV checkpoints, MP3s, and logs are stored under `library`. Back up that directory to keep your projects. The current app imports TXT. Automatic Gutenberg downloads, EPUB support, review and regeneration tools, mastering, and video production are later phases.

The launcher uses the portable Qwen Python 3.12 environment and model from Phase 1. The app finds the existing portable FFmpeg under `.runtime/tempo-tools` or an FFmpeg with `libmp3lame` on PATH. If restoring a missing model, run `.runtime\qwen\Scripts\python.exe fetch_models.py qwen` from the project directory after approving the checkpoint download. To run the automated tests, use `.runtime\qwen\Scripts\python.exe -m unittest discover -s tests -v`.

When the local CUDA Graphs adapter is present, the studio uses it with the same BF16 Qwen checkpoint and Ryan voice. A graph runtime error switches subsequent attempts to standard CUDA. To start directly in standard mode, run `.\start-studio.ps1 --standard-gpu`. GPU benchmark clips and measurements are in `comparison/gpu-efficiency`.

## Phase 1 listening samples

Open `comparison/chatterbox.wav` and `comparison/qwen.wav`. They use the same 310-word opening from Arthur Conan Doyle's [“The Red-Headed League”](https://www.gutenberg.org/files/1661/1661-0.txt), with the original wording and punctuation as input and 350 ms of added silence at each paragraph boundary. Chatterbox performs its own internal punctuation normalization. The main preview applies one gain per narration to target active-sample RMS at −22 dBFS and cap the sample peak at −1 dBFS.

The `*-regenerated.wav` previews substitute segment 6, which was synthesized with a fresh random seed. Listen for changes in voice or timing at that join. The engine subdirectories hold the raw floating-point WAVs and the longer stress clip. `passage.txt` and `long-paragraph.txt` provide the corresponding text.

## Results on the RTX 3070

| Narrator | Listening WAV | Warmed synthesis | Peak PyTorch reserved | Peak whole-GPU used |
| --- | ---: | ---: | ---: | ---: |
| Chatterbox English 500M, built-in | 1:24 | 2:02 | 4.36 GiB | 5.79 GiB |
| Qwen3-TTS 0.6B CustomVoice, Ryan | 2:20 | 8:58 | 3.38 GiB | 5.47 GiB |

Both engines completed the eleven audition segments, the replacement segment and the 102-word stress clip. Qwen reached the frame limit on the separate four-word warmup. This run measured short-passage behavior.

The JSON files report synthesis time, WAV duration, CUDA allocator peaks and sampled whole-GPU memory. Cold load time includes the engine import and model loading with files already on disk. Warmed synthesis time excludes warmup, WAV writes, regeneration and the stress clip. Memory peaks cover model load and all synthesis. Whole-GPU polling includes other Windows applications and may miss brief spikes. Some raw Chatterbox float WAVs exceed full scale. The listening previews attenuate them before conversion to 24-bit PCM.

## Rerun offline

From this directory in PowerShell, run:

```powershell
.\run-comparison.ps1
```

Use `-Engine chatterbox` or `-Engine qwen` to rerun one model. Rerunning replaces that engine's comparison files. The launcher runs one GPU worker at a time using the checkpoints and source text already under `.runtime`.

## Benchmark environment

- `.runtime/chatterbox` uses Python 3.11, Chatterbox TTS 0.1.7, PyTorch and torchaudio 2.6.0 with CUDA 12.4, and setuptools 80.9.0 for PerTh's `pkg_resources` import.
- `.runtime/qwen` uses Python 3.12, Qwen TTS 0.1.1, and PyTorch and torchaudio 2.8.0 with CUDA 12.8. The model uses BF16 and PyTorch SDPA. The script applies a text-length-based codec frame limit to bound generation when Qwen misses its stop signal.
- Both use SoundFile for WAV output. `fetch_models.py` downloads the two checkpoints and records their revisions in `.runtime/models`. Run it with the matching virtual environment's Python if restoring the model files.

Chatterbox code and weights are [MIT licensed](https://huggingface.co/ResembleAI/chatterbox) and its audio carries the PerTh watermark. Qwen code and weights are [Apache 2.0 licensed](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice). The first comparison uses the engines' supplied voices.

