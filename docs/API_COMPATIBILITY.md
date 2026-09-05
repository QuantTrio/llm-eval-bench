# API compatibility

The core evaluates text answers to text or still-image inputs over HTTP. API format is independent of provider and
model architecture. `--api` selects a wire format; `--base-url` selects where to send it.
Each new evaluation requires `--model MODEL_ID` (or a configured model ID).

| API | Generation path relative to API root | Authentication | Streaming |
|---|---|---|---|
| `chat` | `/chat/completions` | Bearer token | OpenAI chat SSE |
| `responses` | `/responses` | Bearer token | Responses typed SSE |
| `messages` | `/messages` | `x-api-key`, `anthropic-version` | Anthropic message SSE |
| `generate-content` | `/models/MODEL:generateContent` | `x-goog-api-key` | `:streamGenerateContent?alt=sse` |

Custom roots preserve their path prefix. Provider presets choose their official root and
credential environment variable. Local services may use no authentication. Model IDs are
passed explicitly and are not checked against a mandatory `/models` call.

## Evaluation contract

- Text and still-image input, text answer output, streaming/non-streaming, native local scoring.
- Canonical image inputs become Chat `image_url`, Responses `input_image`, Messages
  `image.source`, or Gemini `inlineData` blocks. Image hashes enter the run fingerprint.
- Common output budget mapping and optional sampling settings, with saved resolved config.
- Text is extracted from answer blocks; reasoning/thought events do not become answers.
- Input/output usage is provider-reported. Reasoning/cache counts are retained where supplied.
- Unknown usage is flagged; missing throughput is not displayed as measured zero.
- TTFT measures first answer text, not an initial status or thinking event. TPOT is only
  calculated when the available text-token accounting permits it; it is not token-level ITL.
- Midstream errors, missing terminal events, malformed bodies, truncation, and HTTP failures
  remain visible. Transient failures can retry the same API; protocols never auto-fallback.
- Request IDs and a redacted final provider payload are retained. Full SSE transcripts are
  not buffered. Dataset hashes, prompt hashes, and settings accompany every evaluation.

Supporting a route does not mean reproducing every provider feature. Stateful conversations,
server tools, computer use, file uploads, background jobs, and live audio/video are outside
the core evaluation contract. Animated images and image embeddings are not included.
The legacy suite is separate from this adapter contract.

## Scope of verification

DeepSeek `deepseek-v4-flash` has additionally passed real Chat/Responses/Messages JSON and
SSE tests, low-effort thinking, and actual process interruption/resume checks. See the
[live acceptance record](releases/2026-09-05-thinking-resume.md) for counts and limits.
These results do not substitute for testing OpenAI, Anthropic, xAI, or Google directly.

Local tests cover all four formats with both streaming and JSON responses, custom URL paths,
authentication, scoring, retries, usage, failures, model ID enforcement, and resume.
No cloud-provider or GPU-server result is claimed by those tests. A real deployment must be
tested with its installed version, model template, credentials, and frontend configuration.
vLLM/SGLang deployments may expose Chat, Responses, or Messages; Gemini wire compatibility
is only available where the chosen server actually implements that protocol.

Official references consulted for the adapter contract (2026-09-05):

- [OpenAI Responses](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
- [Anthropic Messages](https://platform.claude.com/docs/en/api/http/messages/create)
- [Gemini GenerateContent](https://ai.google.dev/api/generate-content)
- [xAI inference endpoints](https://docs.x.ai/developers/rest-api-reference/inference/chat)
