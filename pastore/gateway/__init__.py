"""The Ollama front door (R20 to R24, R32): attribution, per-host priority queues, and the streaming proxy.

Depends on: pastore.spec, pastore.policy and pastore.state (read-only); never imports dispatcher code paths that write.
"""
