# omni-tokenizer

Native VQWhisper and VQSigLIP encoding with a C++17 CPU/CUDA library,
Python API, CLI, and WebUI.

## Install

Requires Python 3.10+ and a C++17 compiler. On Windows, use a Visual Studio
developer terminal.

```sh
python -m pip install .
```

For CUDA, use Toolkit 11.8+ and an Ampere or newer GPU:

```sh
cmake -S . -B build-cuda -DOMNI_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release --parallel
```

Set `OMNI_LIBRARY` to the resulting shared library. CUDA runtime libraries must
be on the loader's search path (`CUDA_PATH/bin` on Windows).

## Models

```sh
python -m pip install '.[convert]'
python tools/fetch_checkpoints.py --model all --output checkpoints
python tools/convert.py audio --checkpoint checkpoints/vqwhisper --output models/audio
python tools/convert.py image --checkpoint checkpoints/vqsiglip --output models/image
```

## Use

```python
from omni_tokenizer import Tokenizer

with Tokenizer(audio_model="models/audio", image_model="models/image",
               device="cuda") as tokenizer:
    print(tokenizer.encode_image("image.png")["codes"])
    for segment in tokenizer.iter_audio("speech.wav"):
        print(segment["codes"])
```

Use `device="cpu"` for CPU inference. Audio formats other than PCM WAV require
`ffmpeg` and `ffprobe` on PATH.

```sh
omni-tokenizer --image-model models/image --device cuda encode image image.png
omni-tokenizer --audio-model models/audio --device cuda encode audio speech.wav --format stream -o audio.ovs
```

For the HTTP service and WebUI:

```sh
python -m pip install '.[server]'
omni-tokenizer --audio-model models/audio --image-model models/image --device cuda serve
```

Open http://127.0.0.1:8000/. See `omni-tokenizer --help` and each command's
`--help` for options.
