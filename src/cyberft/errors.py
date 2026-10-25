"""Exception types. The library raises these; only ``cyberft.cli`` converts them to exit codes."""

from __future__ import annotations


class CyberftError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(CyberftError, ValueError):
    """The YAML config is invalid or contradicts a hard requirement."""


class NotBuiltError(CyberftError, RuntimeError):
    """A required artifact (processed data, adapter, llama.cpp) is missing."""


class DependencyError(CyberftError, RuntimeError):
    """A runtime dependency or hardware feature is unavailable."""



class NonFiniteLossError(CyberftError, RuntimeError):
    """Training or evaluation produced a NaN/inf loss."""
