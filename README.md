# smart-home-llm

The ML engineering parts of a private voice-assistant project, made standalone. The repo covers:

- fine-tuning a small open model to run a house through tools;
- serving and measuring that model;
- getting a Russian wake word to work on a consumer GPU.

The assistant itself (voice loop, UI, home integration, product logic) is not part of this repository.

Everything here ran on one desktop: an RTX 2060 with 6 GB, a Ryzen 5 3500X and 16 GB of RAM.

```
finetune/   QLoRA training, adapter merge, MLflow tracking and model registry
llm/        provider-neutral LLM interface, Ollama client, the exact Qwen chat template used in training
serving/    FastAPI model service: /v1/generate, /v1/command, Prometheus /metrics, load test, Dockerfile
wakeword/   wake-word experiments: own detector, Whisper-based name scoring, synthetic chatter
data/       20 sample training dialogues (the full set stays private)
tests/      run without a GPU or a model
```

## 1. Fine-tuning a 1.5B model for tool use

**Base model.** Qwen2.5-1.5B-Instruct, out of the box, picks the right tool calls for a spoken home command 34% of the time. That is too low to run a house.

**Training data.** Synthetic multi-turn dialogues, 4,144 in the latest version, in this shape:

- a spoken phrase;
- the model's tool calls;
- the tool results;
- a short spoken reply.

They are generated against a simulated house with several layouts, so the model learns to call the tools and doesn't memorise one floor plan. The loss covers only the assistant's turns (`finetune/train_lora.py`).

**Training setup.**

- **Method:** QLoRA with an nf4 base, double quantisation and fp16 compute.
- **LoRA:** rank 16, alpha 32, applied to every attention and MLP projection.
- **Optimiser:** paged AdamW 8-bit, cosine schedule.
- **Length and batching:** max length 3,072; effective batch of 8 via gradient accumulation.
- **Fit:** the whole run fits in 6 GB of VRAM.

**Deployment.** The adapter is merged, converted to GGUF q8_0 and served by Ollama.

**Prompt format.** Prompts are built with the training template (`llm/qwen_template.py`) and sent raw. Ollama's own chat template renders the tools differently, and that alone costs 12 points (see the table).

**Exams.** Held-out cases are scored on the tool calls the model makes, not on its wording. There are three suites:

| Suite | Cases | What it covers |
|---|---|---|
| main | 103 | everyday commands and questions |
| brightness | 30 | relative and absolute dimming |
| house2 | 56 | a floor plan never seen in training |

| version | main | brightness | house2 | s / answer |
|---|---|---|---|---|
| Qwen2.5-1.5B, untrained | 34.0% | – | – | 0.45 |
| v1 | 98.1% | – | – | 0.77 |
| v1, through Ollama's template | 86.4% | – | – | 0.57 |
| v3 | 100% | 3.3% | – | 0.65 |
| v4 (brightness data added) | 99.0% | 100% | – | 0.65 |
| v5 (second layout added) | 100% | 100% | 94.6% | 0.66 |
| **v6** (in production) | 98.1% | 100% | 100% | 0.69 |

v3 shows why the exam needs more than one suite: it scored 100% on main and still failed almost every dimming command.

**Tracking.** Every version is an MLflow run (`finetune/tracking.py`) holding:

- the parameters: base model, data commit, LoRA settings;
- the loss curve;
- every exam score as `exam.<suite>.accuracy`.

The registered model `home-llm` holds versions 1–6, with the alias `production` pointing at v6.

## 2. The model as a service

`serving/` puts one HTTP API in front of the model:

- `POST /v1/generate` – one turn: system prompt, history and tools in; the reply and tool calls out.
- `POST /v1/command` – a phrase in; the model's first step out, using the prompt and tools that ship with the service.
- `GET /v1/model` – the Ollama model, its digest and quantisation, and its registry version.
- `GET /v1/stats` and `GET /metrics` – latency histograms, tokens per second, malformed tool calls and errors, as JSON and for Prometheus.

The service also handles model loading:

- it warms the model at startup;
- it can keep it resident (`MODEL_SERVICE_KEEP_ALIVE`). Without that, the first command after a pause took about 6 s instead of 0.6 s.

**Load test.** `python -m serving.loadtest` replays real phrases. Results on the desktop above:

- p50 of 351 ms;
- about 2.85 requests per second;
- one model on one 6 GB GPU.

The Docker image is 243 MB. The model stays in Ollama on the host, which owns the GPU.

```bash
pip install -r serving/requirements.txt
python -m serving                    # http://localhost:8090/docs
docker build -f serving/Dockerfile -t home-llm-serving .
python -m pytest -q
```

## 3. A Russian wake word on a consumer GPU

The name is "Джарвис". The baseline is a small general Russian recognizer (Vosk), which hears the name in its own transcript. On recordings from the household it catches 91 of 139 calls. Three approaches were tried to recover the rest:

| Approach | Names heard (of 139) | False wakes | Verdict |
|---|---|---|---|
| Vosk alone | 91 | baseline | misses a third |
| Own detector: openWakeWord embeddings + MLP, trained on real and synthetic voices (`wakeword/train.py`) | 29% recall at 1.8 false wakes/h | too many | dropped |
| Whisper transcribes with "Джарвис." as a hint | 137 | ~14 per hour live: Whisper echoed the hint over TV and talk | dropped |
| **Whisper *scores* the name** (`wakeword/name_score.py`) | **109** | **1 in 237 phrases of nonstop chatter salted with near-names; 0 of 51 room phrases** | **in use** |

**How the name scorer works.** It does not ask Whisper to write anything. It force-decodes " Джарвис" right after the start-of-transcript tokens and takes the mean log-probability as the score. With no hint, there is nothing to echo. The cascade (`wakeword/cascade.py`):

- runs the scorer only when Vosk heard no name;
- runs it only on phrases between 0.3 and 8 s long;
- logs only the time, the score and the decision – never the words – so false wakes per hour can be counted from real days.

**Test audio.** The chatter set comes from `wakeword/chatter.py`: synthetic Russian speech played into the room while the microphone listens, with near-names such as "Джарвин" and "Дарвис" mixed in.

**The same Whisper doubles as the local speech recogniser.** large-v3-turbo in half precision:

- **Building it:** the model is built on the CPU, then moved to the GPU. LayerNorm stays in float32, because a plain `.half()` fails.
- **Memory:** 1.7 GB of VRAM instead of 4.9 GB.
- **Speed and accuracy:** 0.27 s per command, and the same 25 of 28 on a command set.

The fine-tuned model fits next to it.

## License

All rights reserved. The code is published to be read. Using, copying or modifying it requires written permission from the author.
