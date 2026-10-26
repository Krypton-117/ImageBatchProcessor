# Third-party notices — optional Photo Agent

Original Agent code is distributed under the repository MIT license. This is an independent community preview, not an upstream endorsed release. Preserve this file and all bundled licenses when redistributing.

- ImageBatchProcessor v1.1.0: MIT. Original Core/runtime binaries plus modified Studio source and their existing `app/LICENSE`, `app/THIRD_PARTY_NOTICES.md`, `app/licenses` are retained. Source revision 3edbc74ef35a43edd88cc3684fccde9c0d0857ef. Exact unmodified IBP LibRaw source is included under sources/external/LibRaw, under the upstream distribution's CDDL election.
- Qwen3-1.7B-GGUF: Apache-2.0; official Q8_0 revision 90862c4b9d2787eaed51d12237eafdfe7c5f6077. Bundled Q4_K_M is a locally requantized derivative using llama.cpp b11889. Exact original and derived hashes, conversion and provenance are in models/manifest.json. License and official model card are retained. https://huggingface.co/Qwen/Qwen3-1.7B-GGUF
- Qwen3-0.6B trainable base: Apache-2.0, official Qwen ModelScope revision 09b42cad3d112e832108974449ccb5e8e0f5b5d1. Complete license, model card and per-file SHA256 manifest are retained under training/base/Qwen3-0.6B. https://modelscope.cn/models/Qwen/Qwen3-0.6B
- llama.cpp b11889 CPU runtime: MIT; see licenses/llama.cpp-LICENSE and llama/LICENSE-LLVM-OpenMP. https://github.com/ggml-org/llama.cpp/tree/b11889
- rawpy 0.27.1: MIT wrapper plus its bundled LibRaw under LGPL-2.1; retain licenses/rawpy, and the exact source distribution licenses/rawpy-0.27.1.tar.gz, including external/LibRaw. Python extension and shared libraries are replaceable in _internal; no restriction on reverse engineering for debugging library modifications is imposed. https://github.com/letmaik/rawpy
- NumPy, Pillow, and bundled numerical/image dependencies: preserve complete package license collections under licenses/numpy and licenses/pillow. These include notices for bundled libraries, which are not relicensed as MIT by this project.
- Python 3.12: PSF license and historical notices in licenses/Python-LICENSE.txt. Auxiliary OpenSSL, libffi and SQLite notices and source locations are retained in licenses/runtime/.
- PyInstaller: GPL with bootloader distribution exception; complete notices in licenses/pyinstaller. This exception allows distributing the bundled application under its own license.
- Scientific papers: metadata and source URLs only in the installer; research PDFs have not been relicensed or redistributed. Locally downloaded RAW test photos are CC0 and omitted from installer.

The application is unsigned. Its local build and test status does not represent certification, upstream approval, or a guarantee of photographic fidelity. For a public release publish the source branch, dependency/source artifacts and these notices alongside the installer.

- nlohmann/json 3.11.3: MIT; vendored source and license in sources/external/json.
- CPU training runtime: PyTorch, Transformers, PEFT, Accelerate, Hugging Face Hub, tokenizers, safetensors and their dependencies. Original package notices and third-party licenses remain under training/python/Lib/site-packages (including each dist-info/licenses directory and torch/LICENSE). Exact installed versions are in training/requirements-lock.txt.
- pypdf: BSD-3-Clause; notices under licenses/pypdf. No PDF input documents are redistributed.
