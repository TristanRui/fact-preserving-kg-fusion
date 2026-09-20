"""Role-aware residual evidence for hierarchical fusion."""

from .role_aware_sparse import (
    RoleAwareSparseIndex,
    RoleAwareSparseResidualConfig,
    fuse_role_aware_sparse_residual,
)

__all__ = [
    "RoleAwareSparseIndex",
    "RoleAwareSparseResidualConfig",
    "fuse_role_aware_sparse_residual",
]
