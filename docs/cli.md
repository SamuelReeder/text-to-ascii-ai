# Command-line reference

[Back to AsciiNet](../README.md)

## Options

| flag | default | |
|---|---|---|
| `-w/--width N` | terminal width (40–160) | characters per line (16–512; training used 16–128) |
| `--background dark\|light` | dark | your terminal/page background |
| `--invert auto\|yes\|no` | auto | auto: plain white backgrounds become empty space |
| `--focus auto\|select\|on\|off` | auto | fade background clutter around the subject |
| `--engine auto\|net\|pipeline` | auto | AsciiNet (yours in `checkpoints/`, else the released weights) or the pipeline |
| `--checkpoint FILE` | | another AsciiNet checkpoint (`.pt` or `.safetensors`) |
| `--fill match\|ramp` | match | pipeline: glyphs chosen by shape, or the classic density ramp |
| `--no-strokes` | | pipeline: don't draw contours with stroke characters |
| `--color` | | 24-bit ANSI color |
| `-o FILE` / `--png FILE` | | save plain text / a rendering |
| `-p TEXT` | | generate the picture from a prompt (`--candidates`, `--seed`, `--save-image`) |
| `--t2i sana-sprint\|sd-turbo` | sana-sprint | text → image model for prompts |
| `--fast` | | pipeline: skip the neural matte (no focus) |


Install the `pipeline` or `text` profile to use the teacher or prompt options.
