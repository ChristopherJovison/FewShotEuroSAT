from __future__ import annotations

DEFAULT_TEMPLATES = ["a Sentinel-2 satellite image of {}."]

ENSEMBLE_TEMPLATES = [
    "a Sentinel-2 satellite image of {}.",
    "a centered satellite image of {}.",
    "a remote sensing image of {}.",
    "an aerial image of {}.",
]


def build_prompts(
    class_names: list[str],
    ensemble: bool = True,
    templates: list[str] | None = None,
) -> list[list[str]]:
    if templates is None:
        templates = ENSEMBLE_TEMPLATES if ensemble else DEFAULT_TEMPLATES
    return [[template.format(name) for template in templates] for name in class_names]
