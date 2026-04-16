package server

// Inference webhooks — an optional mechanism to hand off each inference
// request and response to an external HTTP endpoint for inspection,
// modification, or blocking. Designed as a vendor-neutral extension point so
// third-party guardrail/observability/policy systems can plug in without
// Ollama depending on any of them.
//
// Enabled by setting OLLAMA_HOOK_URL_PRE_INFERENCE and/or
// OLLAMA_HOOK_URL_POST_INFERENCE. Both unset → zero overhead, no middleware
// is added to the handler chain.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// inferenceHook carries the HTTP configuration used by both pre- and
// post-inference webhook calls.
type inferenceHook struct {
	preURL  string
	postURL string
	timeout time.Duration
	onError string // "deny" (fail-closed) or "allow" (fail-open)
	headers http.Header
	client  *http.Client
}

// newInferenceHook reads the relevant OLLAMA_HOOK_* environment variables and
// returns a configured hook if any URLs are set, nil otherwise.
func newInferenceHook() *inferenceHook {
	pre := envconfig.HookURLPreInference()
	post := envconfig.HookURLPostInference()
	if pre == "" && post == "" {
		return nil
	}

	timeout := envconfig.HookTimeout()
	if timeout == 0 {
		timeout = 5 * time.Second
	}
	onErr := strings.ToLower(envconfig.HookOnError())
	if onErr != "allow" {
		onErr = "deny"
	}

	headers := http.Header{}
	for _, hv := range envconfig.HookHeaders() {
		if k, v, ok := strings.Cut(hv, ":"); ok {
			headers.Set(strings.TrimSpace(k), strings.TrimSpace(v))
		}
	}

	return &inferenceHook{
		preURL:  pre,
		postURL: post,
		timeout: timeout,
		onError: onErr,
		headers: headers,
		client:  &http.Client{Timeout: timeout},
	}
}

// Server integration ---------------------------------------------------------

func (s *Server) initInferenceHook() {
	s.inferenceHook = newInferenceHook()
	if s.inferenceHook != nil {
		slog.Info("inference webhooks enabled",
			"pre_url", s.inferenceHook.preURL,
			"post_url", s.inferenceHook.postURL,
			"on_error", s.inferenceHook.onError,
			"timeout", s.inferenceHook.timeout)
	}
}

// withInferenceHook wraps the given handler chain with the pre-inference
// middleware when a pre URL is configured. When no pre URL is set, the
// handlers pass through unchanged — zero overhead.
//
// Post-inference is NOT installed here — it must be invoked from within the
// handler because it needs access to the response assembly point (the
// channel in ChatHandler/GenerateHandler). See PostInference() below.
func (s *Server) withInferenceHook(route string, handlers ...gin.HandlerFunc) []gin.HandlerFunc {
	if s.inferenceHook == nil || s.inferenceHook.preURL == "" {
		return handlers
	}
	return append([]gin.HandlerFunc{s.inferenceHook.preMiddleware(route)}, handlers...)
}

// hookedChain composes request-logging + pre-inference-hook + the given
// protocol-conversion middlewares + the final handler into a single slice for
// r.POST(...). It exists because the /v1/* routes otherwise devolve into
// three-level-nested append() calls that are unreadable.
//
// Order: [requestLogging, [protocolConvert...], [preHook?], handler].
func (s *Server) hookedChain(route string, convert []gin.HandlerFunc, handler gin.HandlerFunc) []gin.HandlerFunc {
	chain := append(convert, s.withInferenceHook(route, handler)...)
	return s.withInferenceRequestLogging(route, chain...)
}

// Wire protocol types --------------------------------------------------------

// HookRequest is the JSON payload POSTed to a hook URL.
type HookRequest struct {
	Event      string         `json:"event"`
	RequestID  string         `json:"request_id"`
	Route      string         `json:"route"`
	Model      string         `json:"model,omitempty"`
	Messages   []HookMessage  `json:"messages,omitempty"`
	Tools      []HookTool     `json:"tools,omitempty"`
	Options    map[string]any `json:"options,omitempty"`
	OutputText string         `json:"output_text,omitempty"`
	ToolCalls  []HookToolCall `json:"tool_calls,omitempty"`
}

// HookMessage is an OpenAI-format chat message. Ollama's native shape is
// normalized to this before sending to hooks so the wire contract is stable
// across the /api/chat, /v1/chat/completions, and /v1/messages routes.
type HookMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCalls  []HookToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

// HookTool is the OpenAI-format tool (function) definition.
type HookTool struct {
	Type     string      `json:"type"`
	Function HookToolDef `json:"function"`
}

type HookToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type HookToolCall struct {
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type"`
	Function HookToolCallFn `json:"function"`
}

type HookToolCallFn struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// HookResponse is the JSON body expected from a hook URL.
//
// Action verbs are standardized across pre- and post-inference calls:
//   - "allow":  proceed unchanged
//   - "deny":   reject the request/response (returns HTTP 400)
//   - "ask":    human-in-the-loop signal — reject the request/response with
//              HTTP 403 and a body callers can surface as a confirmation
//              prompt. Content is NOT revealed to the caller.
//   - "modify": overwrite fields per the channel (messages pre, output/
//              tool_calls post)
type HookResponse struct {
	Action     string         `json:"action"` // "allow" | "deny" | "ask" | "modify"
	Reason     string         `json:"reason,omitempty"`
	Messages   []HookMessage  `json:"messages,omitempty"`    // pre: modify
	OutputText string         `json:"output_text,omitempty"` // post: modify
	ToolCalls  []HookToolCall `json:"tool_calls,omitempty"`  // post: modify
}

// Pre-inference --------------------------------------------------------------

// preMiddleware reads the inbound body, converts it to a normalized hook
// payload, calls the configured URL, and either aborts the request (block),
// rewrites the body (modify), or continues (allow).
//
// The middleware runs AFTER format-conversion middleware for /v1/* routes so
// the body it reads is already in api.ChatRequest / api.GenerateRequest
// shape.
func (h *inferenceHook) preMiddleware(route string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request == nil || c.Request.Body == nil {
			c.Next()
			return
		}
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			slog.Warn("inference hook: read body", "route", route, "err", err)
			c.Request.Body = io.NopCloser(bytes.NewReader(body))
			c.Next()
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))

		reqID := uuid.NewString()
		c.Set("inference_hook_request_id", reqID)

		payload, buildErr := buildPrePayload(route, reqID, body)
		if buildErr != nil {
			// Not an error we block on — the route may not carry inference
			// payloads (e.g., someone mis-wired this middleware). Pass
			// through.
			slog.Debug("inference hook: non-inference body", "route", route, "err", buildErr)
			c.Next()
			return
		}

		res, err := h.call(c.Request.Context(), h.preURL, "pre_inference", payload)
		if err != nil {
			if h.onError == "allow" {
				slog.Warn("inference hook: pre call failed, fail-open", "route", route, "err", err)
				c.Next()
				return
			}
			slog.Warn("inference hook: pre call failed, fail-closed", "route", route, "err", err)
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
				"error": "inference hook unavailable",
			})
			return
		}

		switch res.Action {
		case "deny":
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
				"action": "deny",
				"error":  "denied by inference hook: " + res.Reason,
				"reason": res.Reason,
			})
			return
		case "ask":
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"action": "ask",
				"error":  "approval required by inference hook: " + res.Reason,
				"reason": res.Reason,
			})
			return
		case "modify":
			newBody, err := applyPreModify(body, res.Messages)
			if err != nil {
				slog.Warn("inference hook: apply modify", "route", route, "err", err)
				if h.onError == "allow" {
					c.Next()
					return
				}
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{
					"error": "inference hook modify failed",
				})
				return
			}
			c.Request.Body = io.NopCloser(bytes.NewReader(newBody))
			c.Request.ContentLength = int64(len(newBody))
			c.Next()
			return
		case "", "allow":
			c.Next()
			return
		default:
			slog.Warn("inference hook: unknown action", "route", route, "action", res.Action)
			c.Next()
			return
		}
	}
}

// buildPrePayload parses the request body as a chat or generate request and
// builds a normalized HookRequest. Returns an error if the body doesn't
// parse as either.
func buildPrePayload(route, requestID string, body []byte) (HookRequest, error) {
	hp := HookRequest{
		Event:     "pre_inference",
		RequestID: requestID,
		Route:     route,
	}

	// Try chat first.
	var chat api.ChatRequest
	if err := json.Unmarshal(body, &chat); err == nil && len(chat.Messages) > 0 {
		hp.Model = chat.Model
		hp.Messages = messagesToHook(chat.Messages)
		hp.Tools = toolsToHook(chat.Tools)
		hp.Options = chat.Options
		return hp, nil
	}

	// Then generate.
	var gen api.GenerateRequest
	if err := json.Unmarshal(body, &gen); err == nil && (gen.Prompt != "" || gen.System != "") {
		hp.Model = gen.Model
		if gen.System != "" {
			hp.Messages = append(hp.Messages, HookMessage{Role: "system", Content: gen.System})
		}
		if gen.Prompt != "" {
			hp.Messages = append(hp.Messages, HookMessage{Role: "user", Content: gen.Prompt})
		}
		hp.Options = gen.Options
		return hp, nil
	}

	return hp, errors.New("not a chat or generate request")
}

// applyPreModify replaces the messages in the body with those from the hook
// response. Preserves other fields on the request (model, options, tools,
// etc.) exactly. Works for both api.ChatRequest and api.GenerateRequest
// shapes.
func applyPreModify(body []byte, modified []HookMessage) ([]byte, error) {
	// Detect which request shape this is by probing for the "messages" key.
	// We decode to map[string]any to preserve all other fields as-is.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}
	if _, hasMessages := raw["messages"]; hasMessages {
		modRaw, err := json.Marshal(messagesFromHook(modified))
		if err != nil {
			return nil, err
		}
		raw["messages"] = modRaw
		return json.Marshal(raw)
	}
	// Generate request: pick the last message of role != system as prompt,
	// system message as system.
	var newSystem, newPrompt string
	for _, m := range modified {
		if strings.EqualFold(m.Role, "system") {
			newSystem = m.Content
		} else {
			newPrompt = m.Content
		}
	}
	if b, err := json.Marshal(newPrompt); err == nil {
		raw["prompt"] = b
	}
	if newSystem != "" {
		if b, err := json.Marshal(newSystem); err == nil {
			raw["system"] = b
		}
	}
	return json.Marshal(raw)
}

// Post-inference -------------------------------------------------------------

// PostInferenceResult is the outcome of a post-inference hook call, as seen
// by ChatHandler / GenerateHandler.
type PostInferenceResult struct {
	// Action is one of "allow" | "deny" | "ask" | "modify". When the hook
	// is not configured or the verdict is allow/modify, callers should emit
	// the response normally (using possibly-updated OutputText/ToolCalls).
	// "deny" and "ask" are terminal — callers must abort with the matching
	// status code (400 for deny, 403 for ask).
	Action     string
	Reason     string
	OutputText string
	ToolCalls  []HookToolCall
}

// HTTPStatus returns the status code to respond with when Action is a
// terminal verdict. Returns 0 for non-terminal actions (allow/modify).
func (r PostInferenceResult) HTTPStatus() int {
	switch r.Action {
	case "deny":
		return http.StatusBadRequest
	case "ask":
		return http.StatusForbidden
	default:
		return 0
	}
}

// Terminated reports whether the caller must abort the response rather than
// return the content.
func (r PostInferenceResult) Terminated() bool {
	return r.Action == "deny" || r.Action == "ask"
}

// callPostInference is an internal helper — most callers should use
// Server.PostInference(c, ...).
func (s *Server) callPostInference(c *gin.Context, route, model, outputText string, toolCalls []HookToolCall) PostInferenceResult {
	if s.inferenceHook == nil || s.inferenceHook.postURL == "" {
		return PostInferenceResult{Action: "allow", OutputText: outputText, ToolCalls: toolCalls}
	}

	reqID, _ := c.Get("inference_hook_request_id")
	reqIDStr, _ := reqID.(string)

	payload := HookRequest{
		Event:      "post_inference",
		RequestID:  reqIDStr,
		Route:      route,
		Model:      model,
		OutputText: outputText,
		ToolCalls:  toolCalls,
	}

	res, err := s.inferenceHook.call(c.Request.Context(), s.inferenceHook.postURL, "post_inference", payload)
	if err != nil {
		if s.inferenceHook.onError == "allow" {
			slog.Warn("inference hook: post call failed, fail-open", "route", route, "err", err)
			return PostInferenceResult{Action: "allow", OutputText: outputText, ToolCalls: toolCalls}
		}
		slog.Warn("inference hook: post call failed, fail-closed", "route", route, "err", err)
		return PostInferenceResult{Action: "deny", Reason: "post hook unavailable"}
	}

	switch res.Action {
	case "deny":
		return PostInferenceResult{Action: "deny", Reason: res.Reason}
	case "ask":
		return PostInferenceResult{Action: "ask", Reason: res.Reason}
	case "modify":
		if res.OutputText != "" {
			outputText = res.OutputText
		}
		if res.ToolCalls != nil {
			toolCalls = res.ToolCalls
		}
		return PostInferenceResult{Action: "modify", Reason: res.Reason, OutputText: outputText, ToolCalls: toolCalls}
	default:
		return PostInferenceResult{Action: "allow", OutputText: outputText, ToolCalls: toolCalls}
	}
}

// PostInference is the exported post-inference hook entry point used by
// ChatHandler/GenerateHandler. Returns an "allow" result when the hook is not
// configured, so handlers can call it unconditionally.
func (s *Server) PostInference(c *gin.Context, route, model, outputText string, toolCalls []HookToolCall) PostInferenceResult {
	return s.callPostInference(c, route, model, outputText, toolCalls)
}

// PostInferenceConfigured reports whether a post-inference URL has been
// configured. Handlers can skip assembling the hook payload when no one is
// listening.
func (s *Server) PostInferenceConfigured() bool {
	return s.inferenceHook != nil && s.inferenceHook.postURL != ""
}

// call is the shared HTTP round-trip helper for pre and post.
func (h *inferenceHook) call(ctx context.Context, url, event string, payload HookRequest) (HookResponse, error) {
	var out HookResponse
	buf, err := json.Marshal(payload)
	if err != nil {
		return out, err
	}

	reqCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return out, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Ollama-Hook-Event", event)
	req.Header.Set("X-Ollama-Request-Id", payload.RequestID)
	for k, vals := range h.headers {
		for _, v := range vals {
			req.Header.Add(k, v)
		}
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()

	// Read up to maxHookBody+1 bytes so we can distinguish "exactly at the
	// limit" from "oversized". Silently truncating to the limit and handing
	// a partial buffer to json.Unmarshal produces a confusing decode error
	// that hides the real cause.
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxHookBody+1))
	if err != nil {
		return out, fmt.Errorf("read hook body: %w", err)
	}
	if int64(len(respBody)) > maxHookBody {
		return out, fmt.Errorf("hook response body exceeds %d bytes", maxHookBody)
	}
	if resp.StatusCode >= 300 {
		return out, fmt.Errorf("hook http %d: %s", resp.StatusCode, truncateForError(string(respBody), 256))
	}
	if err := json.Unmarshal(respBody, &out); err != nil {
		return out, fmt.Errorf("hook body decode: %w", err)
	}
	return out, nil
}

// maxHookBody caps the size of a hook response body we're willing to buffer.
// Hooks that need to return larger payloads should use request rewriting or
// reduce verbosity.
const maxHookBody int64 = 4 << 20

// Conversions ---------------------------------------------------------------

func messagesToHook(in []api.Message) []HookMessage {
	out := make([]HookMessage, 0, len(in))
	for _, m := range in {
		hm := HookMessage{
			Role:       strings.ToLower(m.Role),
			Content:    m.Content,
			ToolCallID: m.ToolCallID,
			Name:       m.ToolName,
		}
		for _, tc := range m.ToolCalls {
			hm.ToolCalls = append(hm.ToolCalls, HookToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: HookToolCallFn{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments.String(),
				},
			})
		}
		out = append(out, hm)
	}
	return out
}

// toolCallsToHook converts api.ToolCall (used in assembled responses) to the
// wire-format HookToolCall. Arguments are serialized as a JSON string, which
// is the OpenAI tool-call convention and the shape hook servers expect.
func toolCallsToHook(in []api.ToolCall) []HookToolCall {
	if len(in) == 0 {
		return nil
	}
	out := make([]HookToolCall, 0, len(in))
	for _, tc := range in {
		out = append(out, HookToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: HookToolCallFn{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments.String(),
			},
		})
	}
	return out
}

// toolCallsFromHook reverses the conversion for post-inference modify. Parses
// the Arguments JSON string back into an ordered-map. On parse failure leaves
// the arguments empty rather than returning the attacker-controlled string —
// callers treat this as "hook returned unusable args, drop them".
func toolCallsFromHook(in []HookToolCall) []api.ToolCall {
	if len(in) == 0 {
		return nil
	}
	out := make([]api.ToolCall, 0, len(in))
	for _, tc := range in {
		args := api.NewToolCallFunctionArguments()
		if tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		out = append(out, api.ToolCall{
			ID: tc.ID,
			Function: api.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: args,
			},
		})
	}
	return out
}

// messagesFromHook reverses the conversion for modified messages. Loses any
// data that wasn't on the hook side (e.g., images) — the hook side operates
// on text only in v1.
func messagesFromHook(in []HookMessage) []api.Message {
	out := make([]api.Message, 0, len(in))
	for _, m := range in {
		out = append(out, api.Message{
			Role:       m.Role,
			Content:    m.Content,
			ToolName:   m.Name,
			ToolCallID: m.ToolCallID,
		})
	}
	return out
}

func toolsToHook(in api.Tools) []HookTool {
	if len(in) == 0 {
		return nil
	}
	out := make([]HookTool, 0, len(in))
	for _, t := range in {
		ht := HookTool{Type: t.Type}
		ht.Function.Name = t.Function.Name
		ht.Function.Description = t.Function.Description
		if t.Function.Parameters.Properties != nil {
			params := map[string]any{
				"type": t.Function.Parameters.Type,
			}
			// Best-effort serialization — we only need to hand something
			// off; we don't roundtrip it.
			if raw, err := json.Marshal(t.Function.Parameters); err == nil {
				var decoded map[string]any
				if json.Unmarshal(raw, &decoded) == nil {
					params = decoded
				}
			}
			ht.Function.Parameters = params
		}
		out = append(out, ht)
	}
	return out
}

// --- tiny helpers ---

// truncateForError keeps error messages a bounded length.
func truncateForError(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
