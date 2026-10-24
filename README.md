# ROE Map Parser

Extracts terrain and static collision geometry from the original Ring of Elysium game files for bot visibility and line-of-sight checks.

Requires Python 3.11+. Reading VFS assets requires Windows x64 and the game's original `VFS_x64.dll`.

## Install

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

## Export

```powershell
.venv\Scripts\python.exe -m roe_map_parser world --game-root "D:\Games\Ring of Elysium" --world europa --out output/europa
```

Replace `--game-root` with your game installation path. Use an empty output directory. Check `summary.json` and `unsupported.json` for export coverage.
