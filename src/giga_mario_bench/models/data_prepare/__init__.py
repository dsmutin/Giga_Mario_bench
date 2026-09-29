"""Raw benchmark rows to the integer sequences a model consumes.

Giga_Mario already owns ``parse_data`` for Caduceus and LegNet windows. New
models of that kind should be added there. The adapters in this package cover
the reverse-complement toy, which that parser does not emit.
"""

from giga_mario_bench.models.data_prepare.encode import prepare_records

__all__ = ["prepare_records"]
