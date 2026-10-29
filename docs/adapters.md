# Model adapters

The package ships no provider SDKs. Model-backed components talk to one of two small
interfaces, and an experiment config names a factory that builds the client.

## Interfaces

**Text models** (JSON interpreter, end-to-end decider), from
[`interpreters/llm.py`](../src/cdd/interpreters/llm.py):

```python
class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str: ...
```

`schema` is a JSON Schema the reply must satisfy. The JSON interpreter sends one when a
condition sets `constrained: true` (condition (c)); the end-to-end decider does the same for
its orders. Clients without constrained decoding may ignore it.

**Typed decision models** (atomic-question interpreter, typed deciders), from
[`interpreters/typed.py`](../src/cdd/interpreters/typed.py):

```python
@dataclass(frozen=True)
class TypedQuestion:
    key: str
    prompt: str
    options: tuple[str, ...]          # 1 to 255 options
    definitions: tuple[str, ...] = () # one written definition per option, or none

@dataclass(frozen=True)
class TypedAnswer:
    key: str
    choice: str                       # must be one of the question's options
    probability: float

class TypedModelClient(Protocol):
    def ask(self, context: str, questions: Sequence[TypedQuestion]) -> Sequence[TypedAnswer]: ...
```

When `definitions` is set, show each option with its definition; typed models can follow an
option's name over its definition, so the definitions carry the meaning. Questions passed
together in one `ask` call are independent. Option sets larger than 255 are
split into a two-stage question by `choose`, so an adapter never sees more than 255 options.

## Built-in: Ollama

[`cdd.adapters.ollama`](../src/cdd/adapters/ollama.py) talks to a local Ollama server over
HTTP with the standard library and implements both interfaces:

| `client_args` | Default | Meaning |
| --- | --- | --- |
| `model` | (required) | an Ollama tag, e.g. `qwen2.5:7b-instruct` |
| `host` | `http://127.0.0.1:11434` | server address |
| `temperature`, `seed` | `0`, `0` | sent with every request |
| `num_ctx` | `8192` | context window |
| `max_tokens` | `1024` | cap on each reply (Ollama's `num_predict`); a reply cut off by it fails to parse and counts as that message's failure, where a runaway generation would otherwise hit the 300 s timeout and stop the run |
| `cache` | `false` | reuse replies to identical requests within a run; keep it off for repeat runs that measure consistency |
| `cache_path` | none | a SQLite file of replies kept across runs, keyed by the request body and the model digest; the runner sets it from the experiment's `reply_cache` |
| `log_path` | none | append every request and reply as JSON lines |

The client checks that the server is reachable and the model is pulled when it is created, and
records the model's digest and the server version; `provenance()` returns them, and the runner
writes them to `models.json` in the run folder.
Typed questions become one JSON property per question, constrained to that question's options.
Ollama does report log-probabilities, but under constrained decoding they are not masked
probabilities, so the adapter returns 1.0 for every answer. See
[`configs/experiments/local-ollama.yaml`](../configs/experiments/local-ollama.yaml).

## Built-in: Laya

[`cdd.adapters.laya`](../src/cdd/adapters/laya.py) runs Laya typed decision models (ConvAI
Innovations, Apache-2.0) locally. Install the optional extra, which pulls in PyTorch and
transformers (about 0.7 GB):

```bash
pip install -e ".[laya]"
```

The extra pins `laya==0.4.1`, the newest release at least two weeks old when it was added (the
project releases several times a week). The client loads the `typed-decisions` checkpoint
(about 0.9 GB, cached by the Hugging Face hub) at the revision that release pins. Each typed
question becomes a Laya `choice` question whose criteria map option labels to definitions, and
Laya's calibrated confidence becomes the answer's probability, so κ is meaningful.

```yaml
- name: laya-atomic
  interpreter: typed_atomic
  client: cdd.adapters.laya:make_client            # pinned typed-decisions checkpoint
- name: laya-finetuned
  interpreter: typed_atomic
  client: cdd.adapters.laya:make_client
  client_args: {model: models/laya-ft, subfolder: null}
```

**Fine-tuning** on training streams, never on the development or reported streams:

```bash
cdd generate --config configs/benchmark/train.yaml --out data/train
cdd generate --config configs/benchmark/dev.yaml --out data/dev
cdd laya-export --streams data/train --out data/laya/train.jsonl
cdd laya-export --streams data/dev --out data/laya/dev.jsonl
cdd laya-finetune --data data/laya/train.jsonl --eval data/laya/dev.jsonl --out models/laya-ft --epochs 2
```

Export rows use exactly the question groups and wording the interpreter asks at inference.
`laya-finetune` fine-tunes and recalibrates the pinned checkpoint and writes a report to
`train_report.json`; `--eval` makes its accuracy and calibration figures held-out.

On a 24 GB Apple-silicon laptop, one epoch over 300 rows took about 5 minutes and pushed the
machine about 13 GB into swap; a full run over the 2,574 pilot-sized training rows takes hours.
Laya's model card links a notebook for Kaggle's free 2×T4 GPUs as an alternative. A fine-tuned
checkpoint is about 1.6 GB, half of it a `checkpoint_latest/` copy you can delete afterwards.

## Writing an adapter

Keep adapters outside `src/cdd`, for example in an `adapters/` folder on your `PYTHONPATH`:

```python
# adapters/my_provider.py
class MyLLM:
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        ...  # call the provider (with structured output if `schema` is set); return the text


def make_llm(model: str, temperature: float = 0.0) -> MyLLM:
    return MyLLM(model, temperature)
```

Then reference the factory in the experiment config:

```yaml
conditions:
  - name: llm-json
    interpreter: llm_json
    decider: milp
    client: my_provider:make_llm
    client_args: {model: "<pinned model id>", temperature: 0}

  - name: typed-over-compiled-state
    interpreter: llm_json
    client: my_provider:make_llm
    decider: typed_state
    decider_client: typed_provider:make_client
    decider_client_args: {model: "<pinned model id>"}
```

`client` and `client_args` serve the interpreter, and the decider too unless `decider_client`
is set. Each distinct factory and argument set is built once per run and reused across
streams.

## Recommendations for reported runs

- Pin exact model versions and record them in `client_args`; the runner copies the resolved
  config into the run folder and each client's digest or revision into `models.json`.
- Model deciders get the same objective text (`cdd.prompts.render_objective`: costs and
  physical rules); stream deciders also get the master data as of the first week
  (`render_master`). A decider that is not told what the solver optimises is not a fair rival.
- Log raw requests and replies inside the adapter. The audit log records what the pipeline did
  with them, not the provider traffic.
- Interleave conditions and repeat runs to guard against endpoint drift, as planned in the
  paper's statistics section.
- Check each provider's terms before publishing benchmark results for its models.
