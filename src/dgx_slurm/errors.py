"""Exception hierarchy for dgx_slurm.

Messages must never include passwords, .ovpn contents, certificate/key
material, the auth temp file, or full environment dumps.
"""

from __future__ import annotations


class DGXError(Exception):
    """Base class for all dgx_slurm errors."""


class ConfigurationError(DGXError):
    """Invalid or missing configuration was supplied."""


class VPNError(DGXError):
    """The OpenVPN connection could not be established or was lost."""


class SSHError(DGXError):
    """The SSH transport failed to connect or execute a command."""


class BundleError(DGXError):
    """The notebook bundle could not be validated or built."""


class SubmissionError(DGXError):
    """The job could not be submitted to the cluster."""


class SchedulerError(DGXError):
    """SLURM returned an unexpected or unparseable response."""


class NotebookExecutionError(DGXError):
    """The notebook failed during execution inside the container."""
