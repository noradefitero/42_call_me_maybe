*This project has been created as part of the 42 curriculum by noradefitero.*

# call_me_maybe

Function calling in LLMs: constrained decoding with `Qwen/Qwen3-0.6B`.

## Description

`call_me_maybe` is a function-calling tool. It reads a set of function
definitions and a list of natural-language requests, and turns each request into
a structured function call — a function name plus an object of arguments.

The naive way to do this is to prompt a model with the definitions, ask for JSON
in the reply, and then try to parse whatever came back. That approach is
unreliable: a small model wraps the object in prose, forgets a required key, or
hallucinates a function that does not exist, and the program has to cope with a
broken string every single time.

This project does something different. The model never gets to emit free-form
text. Instead, generation is **constrained**: at every decoding step the model is
only allowed to pick tokens that keep the answer on a path towards a valid,
schema-compliant function call. The output is therefore valid JSON *by
construction*, not by luck, which is what makes a 0.6B-parameter model usable
here at all.

The goal is to demonstrate that **structural guidance beats raw model size**: a
model with 500 million parameters, steered by a grammar, produces machine-readable
output as reliably as models many times larger.

### What the program does

1. Loads `functions_definition.json` — the available functions, with their
   parameter names and types.
2. Loads `function_calling_tests.json` — the natural-language prompts.
3. Builds a prompt for each request: the system instructions, the function
   definitions compacted with TOON, and the user's request.
4. Generates the answer **one token at a time**, rejecting at each step every
   token that would break the JSON structure or the schema.
5. Validates the finished answer with pydantic and writes
   `data/output/function_calling_results.json`.

The run happens in a `rich` terminal UI: the prompt and the answer are typed
into panels as they are produced, a sidebar shows the progress of every input,
and each answer turns green when it validates or red when it does not.

---

## Instructions

### Prerequisites

- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/)
- Enough disk space for the model weights (~1.2 GB). They are downloaded from
  the Hugging Face Hub on the first run and cached afterwards.

### Installation

```bash
uv sync
```

`make install` does the same and additionally installs the pre-commit hooks.

### Run

```bash
uv run python -m src
```

`make run` is a shortcut for the same command. With no arguments the program
reads from `data/input/` and writes to `data/output/` (see
[Example usage](#example-usage)).

### Debug

```bash
uv run python -m pdb -m src
```

`make debug` is a shortcut for the same command. Breakpoints can also be set in
the source; `debugger()` on a running line drops into `pdb`.

### Lint

```bash
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

`make lint` runs both. `mypy` runs in strict mode: no untyped function
definitions are allowed, and any `Any` that leaks into a return type is an error.

### Tests

```bash
uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml
```

`make test` runs the same. Coverage is configured in `pyproject.toml`
(`branch = true`, `source = ["src"]`).

### Clean

```bash
make clean
```

Removes the Python build and cache artifacts left in the tree
(`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, …).

---

## Example usage

The real invocation, with the three optional file flags written out:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Every flag has a default, so this is exactly equivalent to a bare
`uv run python -m src` from the project root.

### Flag reference

| Flag | Short | Default | Meaning |
|---|---|---|---|
| `--functions-definition` / `--functions_definition` | `-f` | `data/input/functions_definition.json` | The functions the model may call |
| `--input` | `-i` | `data/input/function_calling_tests.json` | The prompts to answer |
| `--output` | `-o` | `data/output/function_calling_results.json` | Where the results are written |
| `--model` | `-m` | `Qwen/Qwen3-0.6B` | Any model the `llm_sdk` can load |
| `--system-prompt` | | *(none)* | Extra instructions for this run |
| `--system-prompt-file` | | *(none)* | A file replacing the bundled system prompt |
| `--version` | | | Print the version and exit |

**Both spellings of the first flag are accepted.** The subject writes
`--functions_definition` with an underscore, while the option is declared in
hyphenated style; the CLI accepts `--functions_definition`,
`--functions-definition` and `-f` interchangeably, so either works:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json
uv run python -m src --functions-definition  data/input/functions_definition.json
```

### Defaults

With no flags at all, the program:

- reads both inputs from `data/input/`;
- creates `data/output/` if it does not exist;
- writes every result to `data/output/function_calling_results.json`.

### Input files

`data/input/functions_definition.json` — an array of function definitions:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

`data/input/function_calling_tests.json` — an array of objects, each with a single
`prompt` key:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Reverse the string 'hello'" }
]
```

Both files are validated by pydantic on load; a missing file, malformed JSON, or
a wrong shape is reported as a clear error instead of a traceback.

### Output file

`data/output/function_calling_results.json` — one object per prompt, in the same
order, with exactly three keys: `prompt`, `name` and `parameters`.

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {
            "s": "hello"
        }
    }
]
```

The numbers are written as `2.0` and `3.0`, not `2` and `3`. That is deliberate:
the subject's own example output shows floats for parameters declared as
`number`, so `pipeline/runner.py` widens an integer that a `number`/`float`
parameter received. Every other value is written exactly as the model produced
it, since the grammar has already fixed its type.

---

## Algorithm explanation

### The model and the SDK

The model is **`Qwen/Qwen3-0.6B`**, loaded through the provided wrapper
`llm_sdk.Small_LLM_Model`. The wrapper auto-selects the device: **MPS** on Apple
Silicon, **CUDA** when a GPU is available, **CPU** otherwise; it runs in
`float16` on an accelerator and `float32` on CPU.

Only the public SDK API is used:

| Method | Use here |
|---|---|
| `encode(text)` | Encode the full prompt into input ids |
| `get_logits_from_input_ids(ids)` | Get the raw logits for the next token |
| `decode([id])` | Get the text of one token |

The SDK's vocabulary-file helpers (`get_path_to_vocab_file` and friends) are
**not** used. Token-id ↔ text conversion goes through `decode` instead, so the
code does not depend on the layout of any particular vocabulary file.

The generation loop lives in `Generator.__generate_tokens_loop`:

```python
input_ids = self.llm.encode(str(prompt))[0].tolist()
answer = ""
for _ in range(self.MAX_TOKENS):
    logits = self.llm.get_logits_from_input_ids(input_ids)
    token_id = self.__get_valid_token(logits, answer, functions, attempt)
    if token_id is None:
        raise RuntimeError("No token fits the JSON structure")
    input_ids.append(token_id)
    token_decoded = self.llm.decode([token_id])
    answer += token_decoded
    if JSONGrammar.is_closed(answer, functions):
        break
```

Generation is **autoregressive, one token at a time**. The chosen token's id is
appended to `input_ids` and the sequence is fed back for the next step. The
implementation re-feeds the **whole token sequence** at every step and does not
use a KV cache; it also calls `decode` on each candidate id it considers.

### Constrained argmax

The interesting part is `__get_valid_token`. Instead of taking the argmax of the
logits and then checking whether the result is acceptable, it walks the
candidates **in descending logit order** and returns the first one that keeps the
answer legal:

```python
for token_id in np.argsort(logits)[::-1]:
    token_decoded = self.llm.decode([int(token_id)])
    if not token_decoded or not token_decoded.strip():
        continue  # special / blank tokens
    if "\n" in token_decoded:
        continue  # no line breaks in a call
    if JSONGrammar.valid_prefix(answer + token_decoded, functions):
        ...
        return int(token_id)
```

`np.argsort(logits)[::-1]` is the full vocabulary ordered from the highest logit
to the lowest. Each candidate is decoded and offered to the grammar as
`answer + token_decoded`.

- The **first candidate that passes is chosen**, so the emitted token is the
  *constrained argmax*: the highest-logit token among the valid ones. Every
  invalid token is effectively excluded from the choice — not penalised
  afterwards, never considered in the first place.
- Candidates that decode to nothing (special tokens) or to only whitespace are
  skipped: they carry no text and are not part of a function call.
- Candidates containing a newline are skipped, because a line break has no place
  in a single-line JSON object and the tokenizer would otherwise hide it.

`valid_prefix` is *stateless*: it re-tokenizes the whole answer and re-walks the
grammar from the first state on every call. That is simple to reason about and
fast enough, because the answer is a single short JSON line.

### The JSON grammar

`src/call_me_maybe/json_grammar.py` is a hand-written state machine over the
steps of a function call:

```text
Start → Name → NameColon → NameValue → NameComma → Parameters → ParametersColon
      → ParametersOpen → ParameterKey → ParameterColon → ParameterValue
      → ParameterComma → Close → Done
```

`JSONGrammar.valid_prefix(text, functions)` walks the tokenized answer through
those states and returns `True` as long as nothing has contradicted the call
still being completable. `JSONGrammar.is_closed(text, functions)` returns `True`
only when the walk reached `Done`.

The states that need to *decide* something hold that decision in a small `_Call`
dataclass:

| Step | What it enforces |
|---|---|
| `Start` | The object opens with `{` |
| `Name` | The first key is exactly `name` |
| `NameColon` | `:` follows |
| `NameValue` | The value is a string, and it **is one of the provided function names**. This is what selects the function: the matching definition is stored and its parameter names are copied into `call.remaining` |
| `NameComma` | `,` follows |
| `Parameters` | The second key is exactly `parameters` |
| `ParametersColon` | `:` follows |
| `ParametersOpen` | The parameters object opens with `{` |
| `ParameterKey` | The key is a **declared parameter of the selected function that has not been used yet**. Using a key removes it from `remaining`, so it cannot be repeated, and an undeclared key is refused — that is how extra keys are rejected |
| `ParameterColon` | `:` follows |
| `ParameterValue` | The value's token type matches the **declared type** of that parameter |
| `ParameterComma` | `,` while `remaining` is non-empty; `}` only once every declared parameter is present |
| `Close` | The object closes with the matching `}` |
| `Done` | Nothing can follow a finished call |

Two consequences fall out of this design directly:

- **Every declared parameter must appear exactly once.** A missing parameter
  keeps `remaining` non-empty, so the grammar refuses to accept `}`; a repeated
  one is no longer in `remaining`, so it is refused too.
- **No extra keys, no prose.** Any token that is not part of the call — a word,
  a sentence, a second object, anything after `Done` — takes the walk off the
  grammar and `valid_prefix` returns `False`.

Value typing is driven by the same table (`_VALUE_TYPES`): a `"string"` parameter
accepts a JSON string (or `null`), a `"number"` parameter accepts a JSON number
(or `null`). Any other declared type accepts any JSON value, so the code does not
need to change when a definition file uses a type this grammar has never seen.

### The tokenizer and `Incomplete`

`src/call_me_maybe/json_parser.py` turns a partial text into JSON tokens. It
emits punctuation, strings, numbers, the literals `true` / `false` / `null`, and
one extra token type that makes partial generation possible:

> **`Incomplete`** — a trailing word that could still grow into a valid token.

`Incomplete` is what separates *"this is not a token yet"* from *"this could never
be a token"*. It is reported for:

- an unterminated string (`{"name": "fn_ad` → `Incomplete("fn_ad")`);
- a keyword prefix (`n`, `nu`, `nul` → could still become `null`);
- a number fragment (`-`, `12.`, `1e+` — anything that becomes a number by
  appending one digit).

The grammar accepts `Incomplete` **only as the last token**, and only if it could
still grow into something legal in the current state:

- in `Name`, the fragment must be a prefix of `"name"`;
- in `NameValue`, a prefix of one of the provided function names;
- in `ParameterKey`, a prefix of one of the remaining parameter names;
- in `ParameterValue`, it must still be able to become a value of the declared
  type — quoted for a string, a growing number for a number.

A fragment also only counts if the answer really ends with `"` followed by it.
Without its opening quote it could never be completed, so a bare word in the
answer is never mistaken for a half-written string.

String scanning is **escape-aware**: `\"` is content, not the end of the string.
`__find_closing_quote` looks for the next quote and, if a backslash appears
before it inside the same string, skips the escaped character and keeps looking.
The escape search is bounded to the string's own span, because a backslash past
the closing quote cannot escape it.

### Stopping, budget and retries

- **Stopping.** After each token the grammar is asked whether the call is closed.
  Once the matching `}` is in, generation stops immediately — nothing valid could
  follow anyway, and `Done` refuses any further token.
- **Budget.** `MAX_TOKENS = 300` bounds each attempt. If the loop ends without
  the call closing, it is not an error: the loop finishes, a warning is logged,
  and the truncated answer goes to validation, which will reject it and trigger a
  retry.
- **Retries.** `pipeline/runner.py` allows `MAX_RETRIES = 5` attempts per
  prompt. An attempt fails when either
  1. the grammar **deadlocks** — no candidate token fits, so `__get_valid_token`
     returns `None` and the generator raises; or
  2. the finished answer **fails pydantic validation**
     (`LLMFunctionCall.model_validate_json(answer)`).

  Each attempt is *not* a rerun of the same thing. The attempt number is passed
  down into `__get_valid_token`, which skips the first `attempt` valid candidates
  and takes the next one. Attempt 0 takes the best valid token; attempt 1 takes
  the second best valid token, and so on. If fewer candidates survive than the
  attempt asks for, it falls back to the best of them. The effect is that a
  failed answer is not repeated identically — the retry explores the next-best
  legal continuation instead of dead-ending in the same place.
- **When every attempt fails.** If all `MAX_RETRIES` attempts are used up, the run
  stops with a `NoValidCallError`: the CLI prints the offending prompt and exits
  with status 1, and no output file is written. Nothing is invented to fill the
  gap. The obvious alternative — writing a placeholder entry — was rejected on
  purpose: the subject requires that keys and types match the schema *exactly* and
  that all required arguments be present, so a placeholder would have to either
  invent a function name or emit `null` for a parameter declared as a string or a
  number, breaking the "100% valid JSON, schema-compliant" requirement. Failing
  loudly keeps the guarantee absolute instead of trading it for a longer run.

### This is not prompting and hoping

The system prompt *asks* for a single JSON object and nothing else. But the
prompt is not what makes the output valid — it is only there to help the model
pick the right function and the right argument values. The structure is enforced
by the decoder:

- the model is never given the chance to emit free-form text that has to be
  parsed and repaired afterwards;
- there is no "extract the JSON substring" step, no regex fallback, no repair of
  truncated output;
- any token that would take the answer off the grammar is not selected at all.

So when `json.loads` runs on the output, it succeeds, because the tokenizer was
never allowed to produce anything else.

---

## Design decisions

### A hand-written grammar instead of a library

There are mature libraries for grammar-constrained decoding — `outlines`,
`lm-format-enforcer`, `xgrammar` — and using one would have been less code. They
were not used, for reasons specific to this project:

- **They want to own the model.** `outlines` and `lm-format-enforcer` integrate
  with `transformers` generation loops and logits processors, and expect to build
  the prompt and run `generate()` themselves. Here the model is reached only
  through `llm_sdk`, which exposes raw logits for a given id sequence. A library
  that bypasses that API would break the constraint the subject sets.
- **The grammar is data-dependent.** Which keys are legal depends on the
  function the model already chose, which in turn depends on the runtime input.
  A static grammar — a JSON Schema, a regex, a fixed CFG — cannot express "the
  second key must be `parameters`, and the keys inside must be exactly the
  parameters of whichever function was named in the first key". It needs
  runtime state, which is exactly what a state machine gives.
- **The constraint is "valid prefix", not "valid string".** These libraries mostly
  answer *"is this complete output valid?"*. Constrained decoding needs the
  stronger and cheaper question *"could this partial text still become valid?"*,
  asked thousands of times per answer, on prefixes that are not JSON at all.
- **Nothing to install, nothing to fight.** A state machine over 14 states is a
  few hundred lines of pure Python with no dependency and no version coupling.

The cost is that the grammar is only as good as the tests on it — which is why
`json_grammar.py` and `json_parser.py` are deliberately dependency-free, pure
functions that can be exercised without loading a model.

### Pydantic for every boundary

Pydantic is used at four boundaries, and the same models cover the whole flow:

- **Input files.** `FunctionDefinitionList`, `FunctionDefinition` and
  `InputList` validate the JSON that arrives. A malformed file produces a
  `PromptLoadError` with a readable message, not a `KeyError` deep in the
  grammar.
- **The model answer.** `LLMFunctionCall` (`name: str`, `parameters: dict`) is
  the last line of defence. The grammar already guarantees the shape; pydantic
  confirms it and converts the JSON text into real Python values.
- **The result.** `FunctionCall` adds the `prompt` back and `FunctionCallList`
  serializes the whole file with `model_dump_json(indent=4)`.

Models are the single source of truth for the data shape. The grammar reads the
same typed `FunctionDefinition` objects that came from the validated input file,
and the output is written from validated models, so there is no place where a
dict-shaped assumption can drift from reality.

### Retry by taking the next-best token

A plain "retry until it works" loop would recompute the same answer and fail the
same way, because decoding is deterministic: same prompt, same model, same
argmax. The fix is to make each attempt *different at the source*, which is why
the attempt index is threaded down into token selection and shifts which valid
candidate is taken. It is a cheap diversification — no temperature, no
sampling, no change to the output distribution — that reuses the ordering the
decoder already computed.

Retrying on pydantic failure rather than on the grammar alone also matters: the
grammar guarantees structure, but not that the structure is *complete* if the
budget cut generation short, and pydantic is what catches that case.

### TOON for the prompt

`functions_definition.json` is verbose: every parameter repeats a `"type"` object.
`toon_format.encode` rewrites it as TOON, a compact tabular notation, before it
goes into the prompt. Fewer tokens per definition means more of the context
budget left for the request, and a shorter prompt is a shorter forward pass on
every one of the `N × steps` forward passes a run makes. The definitions are also
rendered once per prompt rather than per attempt, so the saving is repeated.

The prompt itself is a ChatML-style template built with `string.Template`
(`templates/prompt.txt`), with the definitions inside a `<tools>` block and
`/no_think` to keep Qwen3 out of its reasoning mode — anything the reasoning mode
emits would be rejected by the grammar anyway, so there is no point paying for
it.

### Dependencies

| Package | Why |
|---|---|
| `numpy` | Sorting the logits. `np.argsort(logits)[::-1]` is the whole token-ordering mechanism; doing it in Python over a 150k-entry vocabulary would dominate the runtime |
| `pydantic` | Validation and serialization at every I/O boundary (above) |
| `llm_sdk` | The provided model wrapper — the only way to reach the model |
| `toon-format` | Compact function definitions in the prompt |
| `typer` | The CLI, including the dual `--functions-definition` / `--functions_definition` spelling |
| `rich` | The live terminal UI: panels, the typing animation, spinners, the log panel |

There is no `transformers` usage in this codebase — it comes in through
`llm_sdk` — and no `torch` usage either. The constrained decoding itself needs
nothing more than a logits vector and a `decode`.

---

## Performance analysis

All figures below were measured on the shipped dataset, on an Apple Silicon Mac
with the model running on MPS.

### Measured results

| Metric | Subject's target | Measured |
|---|---|---|
| Validity of the output | 100% valid JSON | **100%** — valid by construction of the grammar |
| Correct function selection | 90%+ | **11 / 11** prompts select the right function, with correct argument names |
| Schema compliance | All required arguments, no extra keys | **11 / 11** |
| Wall-clock time | under 5 minutes | **~55 seconds** end to end for the whole run, *including* loading the model |

The shipped test set is **11 prompts** over 5 functions. That is a small set, so
these numbers should be read as "the approach works on every case tried", not as
a statistical claim. The peer review may run different prompts and a different
function set; the accuracy figure in particular is a property of the sample, not
a guarantee of the method.

### Reliability: where the guarantee comes from

The 100%-valid-JSON figure is not an empirical observation, it is a structural
property. Every emitted token was accepted by the grammar, so the string is
always a complete, schema-compliant function call. There is no error class left
there to measure: "invalid JSON" cannot be produced, "missing required argument"
cannot be produced, "undeclared key" cannot be produced.

That is also why it does not degrade with prompt difficulty. Accuracy, on the
other hand, depends entirely on the model: the grammar guarantees *where* the
model may go, not that it picks the right function or copies the right number
out of the prompt. A prompt for which Qwen3 has no good guess will produce a
well-formed call to the wrong function — valid output, wrong answer. That is the
line between the two metrics.

### Where the time goes

The cost per prompt is roughly `steps × full_forward_pass`, and there is no KV
cache: the whole token sequence — prompt plus everything generated so far — is
fed back through the model at every step. The prompt is the long part of that
sequence, which is exactly why compacting the definitions with TOON pays off.

On top of that, each step sorts the full logits vector and tests candidates
against the grammar until one is accepted. Two things were done about it:

- candidates are examined in descending logit order and the **first** valid one
  is returned, so in the common case only a handful are tested;
- the tokenizer runs on this hot path once per candidate, so it is built out of
  `str.find` scans over bounded spans rather than a character-by-character loop.

The model load is paid once per run, not once per prompt: a single `Generator`
holds the `Small_LLM_Model` for every input in the file.

### Bounds

Two constants keep a bad case from running away:

- `MAX_TOKENS = 300` per attempt — a call that needs more than 300 tokens is cut
  off, logged, and handed to validation, which rejects it and triggers a retry.
- `MAX_RETRIES = 5` per prompt — worst case 5 × 300 tokens for one input, and
  the prompt is skipped in the output rather than the run hanging.

With a 55-second run and a 5-minute budget, there is comfortable headroom on the
shipped dataset. The margin narrows on CPU, where every forward pass is
substantially slower than on MPS.

---

## Challenges faced

### Validating a *partial* string

The obvious grammar answers "is this complete string valid?". Constrained decoding
needs "can this **incomplete** text still become valid?" — asked on fragments like
`{"name": "fn_ad` that are not parseable JSON at all.

The `Incomplete` token type is the answer. The tokenizer distinguishes "not a
token yet" from "never a token" and reports the first case; the grammar accepts
it only as the last token, only if it could still grow into something legal in
the current state, and only if the answer really ends with an opening quote
followed by it. Everything else — `{"nam`, a number where a key belongs, a
closing brace with parameters still missing — is rejected outright.

### Escape sequences, and a tokenizer that lied

The shipped test set contains a prompt with a quoted string inside it, and the
model duly emitted `"...\"Hello 34 I'm 233 years old\""`. The first version of the
tokenizer scanned for the next `"` and stopped there, so it read the string as
ending mid-value. Everything after it was then garbage, the grammar saw a
truncated answer that could never be completed, and the run deadlocked.

The fix is `__find_closing_quote`: find the next quote, and if a backslash occurs
before it *within the same string*, skip the escaped character and keep looking.
The escape search is deliberately bounded to the string's own span — a backslash
past the closing quote cannot escape it, and searching to the end of the text on
every string is what makes this cost.

### Making `null` reachable

The system prompt asks for `null` when a required value is missing, but `null`
only became reachable once the grammar stopped thinking of it as an ordinary
string. Two things were needed:

- `Null` has to be in the set of token types a value may be, so a `"number"`
  parameter accepts a number *or* `null` (`_VALUE_TYPES`);
- and the *fragment* `n`, `nu`, `nul` has to be accepted while it is being
  written, otherwise the very first character is refused and the grammar
  deadlocks. `__could_be_literal` checks the fragment against the literals the
  declared type allows, which is what makes `null` reachable character by
  character.

### Deadlocking when no token fits

When every candidate is rejected, `__get_valid_token` returns `None` and the
generator raises. Retrying that case naively would loop forever on the same
prompt. The attempt-index mechanism solves it: the retry takes the next-best
valid token, so a state that was reached by an unlucky token is not reached the
same way twice. If fewer candidates survive than the attempt asks for, it falls
back to the best of them, and if even that fails the prompt is dropped from the
output instead of hanging the run.

### Keeping the tokenizer fast

The tokenizer is the unexpected hot path. It is re-run for **every candidate
token at every generation step**, on a string that grows with each accepted
token. A straightforward character-by-character scanner was measurably visible
in the total runtime.

Three things fixed it: the string scan is done with `str.find` on bounded spans
rather than per-character work; the backslash search is bounded to the string
rather than to the end of the text; and the grammar walks *tokens*, not
characters, so its own cost is proportional to the token count.

### Numbers, booleans and the `bool` trap

`isinstance(True, int)` is `True` in Python, so a naive "is this a number?"
check classifies the literal `true` as a number — and would let a boolean through
a `"number"` parameter. Three places could have tripped on that, and each one
avoids it differently:

- **`json_parser.__tokenize_word`** resolves a word against a literal table
  (`true`, `false`, `null`) *before* it ever considers numbers, so a boolean can
  never be classified as a number in the first place. Behind that sits a second,
  redundant guard: `__is_number` decides by parsing the word and checking
  `type(parsed) in (int, float)` rather than `isinstance(parsed, (int, float))`,
  so that even if the literal table were bypassed, `type(True) is bool` and not
  `int`. The two are deliberately not collapsed into one — the table is the
  mechanism, the `type()` check is the belt to its braces, and the tests pin the
  behaviour rather than the redundancy.
- **`json_grammar.__value_allowed`** never asks Python what a value is at all. It
  compares the *token type* against the types the parameter declared, and `true`
  tokenizes as `JSONTokenType.BoolTrue` rather than `JSONTokenType.Number`, so the
  two cannot be confused whatever Python thinks about `bool`. An `integer`
  parameter adds one more rule: a number token carrying a `.` or an exponent is
  refused even though it is a perfectly well-formed number.
- **`runner.__typed_parameters`** does look at real Python values, to decide
  whether an integer should be widened to a float. There the guard is
  `isinstance(value, int) and not isinstance(value, bool)`, so the literal `true`
  is never widened into `1.0`.

Numbers also needed three states rather than two: complete (`-12.5e3`), still
growing (`12.`, `1e+`) and not a number at all (`hello`, `6.e`). A word counts as
still growing when it ends on a number character *and* would parse if a single
`0` were appended.

---

## Testing strategy

The validation strategy is built around one observation: the parts of this
project most likely to be wrong are the tokenizer and the grammar, and neither
needs a model. Both are pure functions, so they are tested directly, with the
model involved only at the very end.

### Unit tests — the tokenizer (`json_parser`)

The tokenizer is tested as "text in, expected token list out":

- **Punctuation and structure** — `{`, `}`, `[`, `]`, `:`, `,`, whitespace
  dropped, tokens in order.
- **Complete strings** — plain content, empty string `""`, content with spaces,
  content with Unicode.
- **Escape sequences** — `\"` stays inside the string, `\\` is consumed as a
  pair, a trailing backslash before the closing quote, and a backslash *after*
  the closing quote (which must not affect that string).
- **Numbers** — integers, negatives, floats, exponents (`1e3`, `1.5e-3`), the
  growable fragments (`-`, `12.`, `1e+`) reported as `Incomplete`, and
  non-numbers that must raise.
- **Literals** — `true`, `false`, `null`, and their prefixes as `Incomplete`.
- **The `Incomplete` distinction** — the central property: a trailing word that
  could still grow is `Incomplete`, a trailing word that never could is an
  error. `tru` is accepted, `hell` is not.
- **`Incomplete` position** — `Incomplete` only ever appears last.

### Unit tests — the grammar (`json_grammar`)

The grammar is tested as `valid_prefix(text, functions)` / `is_closed(text,
functions)` over hand-written prefixes, at every state of the state machine.

- **Valid prefixes** — a correct answer checked after every single token, which
  proves the answer stays a valid prefix the whole way and that nothing valid is
  rejected on the way. Also: whitespace anywhere it is allowed, and the `null`
  value for a missing required value.
- **Invalid prefixes** — a wrong or missing opening brace, a first key other
  than `name`, an undeclared function name, a second key other than
  `parameters`, a missing colon or comma, trailing text after the closing brace.
- **Parameter keys** — a declared parameter accepted; an **undeclared** key
  rejected (extra keys); a **repeated** key rejected; a **missing** parameter
  rejected, both as a missing key and as an early `}`.
- **Parameter ordering** — parameters in declaration order and in any other
  order are both accepted, since the grammar tracks a set, not a sequence.
- **Types** — a number where a string is declared is rejected and vice versa; a
  number fragment for a `"number"` parameter is accepted while it grows; a
  quoted value for a `"number"` parameter is rejected.
- **Completion** — `is_closed` is `True` exactly at the matching `}` and not
  before, including for a function with no parameters.
- **Functions with no parameters, and functions with several** — the empty
  object closes on the spot; a three-parameter function must not close after two.

### End-to-end

The full pipeline is validated by running it on the shipped dataset and checking
the output file: it exists, parses as JSON, has one entry per input in order,
each entry has exactly `prompt` / `name` / `parameters`, every `name` is a
function from the definitions file, and every entry carries exactly the declared
parameters with types matching the declaration.

### Coverage

`make test` reports coverage with branch coverage on, and the distribution is
the point: the two modules where a bug would be invisible and most damaging are
the two best covered.

| Module | Coverage | |
|---|---|---|
| `json_parser.py` | **100%** | every branch, including all escapes and number forms |
| `json_grammar.py` | **95%** | the whole state machine |
| `pipeline/parse.py`, `models/*`, `config.py`, `exceptions.py` | **100%** | |
| `prompt.py` | 90% | |
| `generator.py` | 21% | needs the model |
| `pipeline/runner.py` | 25% | needs the model |
| `ui/countdown.py` | 31% | TUI animation |

The low numbers are not neglected, they are untestable in isolation: loading
Qwen3-0.6B and driving the terminal UI are precisely the parts that cannot run
without weights or a real terminal. `tests/conftest.py` makes that explicit by
monkeypatching `Small_LLM_Model` to raise, so a test that accidentally reaches
for the model fails loudly instead of quietly downloading a gigabyte. Those
paths are exercised by running the program, not by mocking it.

The one honest gap: the output-file invariants above are asserted against the
pipeline's real inputs with a fake answer source, because asserting them for
real means running the model. The full run is what closes it.

### Edge cases

The cases the subject calls out, and where each is covered:

| Case | Covered by |
|---|---|
| Empty strings | Tokenizer and grammar unit tests — `""` is a complete token, and the answer still validates |
| Large numbers | Tokenizer number tests; the grammar's number rule is on token type, not magnitude, so size is irrelevant to validity |
| Special characters | Escape-sequence tests — quotes, backslashes, quotes inside values |
| Wrong types | Grammar type tests — a string where a number is declared is refused by the decoder, so it cannot appear in the output |
| Ambiguous prompts | Behavioural: the output stays valid whichever function the model picks. The end-to-end check verifies validity, not correctness, and the run logs the choice |
| Functions with multiple parameters | Grammar tests for two- and three-parameter functions, including ordering, duplication and early closing |

---

## Resources

### The subject and the SDK

- **42 subject — "Introduction to function calling in LLMs"** (`subject.pdf`, at the
  root of the repository). The specification this project implements: input and
  output formats, the constrained decoding requirement, and the README
  requirements.
- **`docs/llm_sdk.md`** — the documentation for the provided `llm_sdk` wrapper:
  `Small_LLM_Model`, `encode` / `decode` / `get_logits_from_input_ids`, and the
  device and precision defaults.

### Constrained decoding and logits processing

- [Hugging Face Transformers — Logits processing](https://huggingface.co/docs/transformers/main/en/logits_process)
  and [Custom constrained generation with `generate()`](https://huggingface.co/docs/transformers/main/en/constrained-decoding).
  The standard reference for what this project does by hand: processors that
  mask logits before the token is chosen, which is the same mechanism, expressed
  as logits processors instead of a hand-written loop over candidates.
- [`outlines`](https://github.com/dottxt-ai/outlines) — grammar-constrained
  decoding as a library, including regex- and JSON-Schema-based grammars. The
  closest prior art to this project, and the main alternative considered.
- [`lm-format-enforcer`](https://github.com/no-gamr/lm-format-enforcer) — the same
  idea built on character-level parsing engines (llguidance, XGrammar), showing
  the other end of the design space: a general parser engine rather than a
  grammar written for one schema.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) and the
  [Qwen3 documentation](https://qwen.readthedocs.io/) — the model used here,
  including the reasoning ("thinking") mode that `/no_think` disables in the
  prompt template.

### Grammars, schemas and parsing languages

- [JSON Schema](https://json-schema.org/) — the declarative vocabulary for
  describing exactly which keys and value types a JSON object may have, and the
  starting point for thinking about "what is a valid function call".
- Bryan Ford, *Parsing Expression Grammars: a Recognition-Based Syntactic
  Foundation*, POPL 2004 — the reference for recognition-based grammars: instead
  of a grammar that generates text, a set of predicates that decide whether a
  string is acceptable. That is exactly the shape of `valid_prefix` here, and
  the paper is a good explanation of why recognizing is the right direction for
  constrained generation.
- The **parsing languages** line of work, which separates the vocabulary of a
  language (its tokens) from the grammar over that vocabulary — the split
  between `json_parser.py` and `json_grammar.py` in this project.

### Prompt compaction

- [`toon-format`](https://github.com/toon-format/toon-python) — TOON, a compact
  token-efficient serialization for tabular and nested structured data, used here
  to shrink the function definitions before they go into the prompt.

---

## How AI was used

AI tools were used as an assistant throughout the project, in the following ways:

- **Exploring the approach.** Comparing the viable strategies for keeping a small
  model's output valid — a stricter prompt, a grammar-driven decoder, or a
  parser over the output — and reasoning about which one actually satisfies the
  requirement of never emitting text that has to be repaired afterwards.
- **Reviewing and reasoning about the code.** Working through the design of the
  state machine and the tokenizer, and catching problems in them — the escape
  handling in string tokenization, the reachability of the `null` literal, the
  `bool`-is-an-`int` trap in the number check, and the deadlock case where no
  candidate token is valid.
- **Drafting and refining documentation and tests.** Structuring the comments and
  docstrings, writing the unit tests for the tokenizer and the grammar, and
  drafting and revising this README.

The decoding logic, the grammar, the tokenizer and the pipeline structure were
written and decided by the author, and every piece of behaviour described above
corresponds to code in this repository. The model's final answers were always
produced by running the program, not by an AI tool.
