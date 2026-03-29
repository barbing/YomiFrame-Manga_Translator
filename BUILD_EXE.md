# Building YomiFrame EXE

## Prerequisites
1.  **Install PyInstaller**:
    ```powershell
    pip install pyinstaller
    ```
2.  **Ensure Dependencies are Installed**:
    ```powershell
    pip install -r requirements.txt
    ```

## Build Instructions
Run the following command in the terminal:

```powershell
pyinstaller manga_translator.spec
```

## Output
*   The built application will be in `dist/YomiFrame`.
*   Run `YomiFrame.exe` inside that folder.

## Models
The EXE does **not** bundle the large model files (to prevent the EXE from being 10GB+).
*   **Copy your `models/` folder** into the `dist/YomiFrame/` folder.
*   The first time you run it on a new machine, it may try to download huggingface models (MangaOCR/LaMa) if they are not cached.
    *   To make it truly offline, verify where `cache_dir` is or pre-download models into `~/.cache/huggingface` on the target machine.
