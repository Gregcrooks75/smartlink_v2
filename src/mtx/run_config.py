"""Configuration for the active meter-to-transformer SmartLink pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Parameters used by the meter-to-transformer pipeline stages."""

    force_ami_refresh: bool = False
    force_transformer_refresh: bool = False
    force_itron_refresh: bool = False
    transformer_candidate_radius_ft: float = 550.0
    group_source: str = "itron"
    transformer_division_col: str = "DivName"
    overwrite_m2t_outputs: bool = False
    partitioned_divisions: tuple[str, ...] = ("Central", "Lancaster")
    ami_columns: tuple[str, ...] = (
        "Address", "ZipCode", "City", "Division", "LATITUDE",
        "LONGITUDE", "Circuit", "SuperiorFLOC", "ConnectionObject",
        "PremiseID",
    )
    similarity_columns: tuple[str, ...] = (
        "Address", "ConnectionObject", "PremiseID",
    )
    exact_match_columns: tuple[str, ...] = (
        "ZipCode", "City", "Circuit",
    )

    def __post_init__(self) -> None:
        if self.transformer_candidate_radius_ft <= 0:
            raise ValueError("transformer_candidate_radius_ft must be positive.")
        if self.group_source not in {"itron", "smartlink"}:
            raise ValueError("group_source must be 'itron' or 'smartlink'.")
        if not set(self.similarity_columns) <= set(self.ami_columns):
            raise ValueError("similarity_columns must be in ami_columns.")
        if not set(self.exact_match_columns) <= set(self.ami_columns):
            raise ValueError("exact_match_columns must be in ami_columns.")
