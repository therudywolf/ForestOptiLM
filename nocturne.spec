# -*- mode: python ; coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# PyInstaller spec for Nocturne Data Forge (ForestOptiLM) GUI.
# Build:  pyinstaller --noconfirm --clean nocturne.spec
# Output (one-dir, reliable):
#   Windows: dist/NocturneDataForge/NocturneDataForge.exe
#   Linux:   dist/NocturneDataForge/NocturneDataForge
#   macOS:   dist/NocturneDataForge/NocturneDataForge  +  dist/NocturneDataForge.app
#
# Cross-platform: the same spec builds on Windows, Linux and macOS (PyInstaller
# does NOT cross-compile — run it once per target OS, e.g. via CI matrix).
# One-dir is used intentionally: faiss/pandas/tcl-tk ship native libraries that
# are far more robust un-packed than inside a single self-extracting binary.
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas: list = []
binaries: list = []
hiddenimports: list = []

# Packages that carry data files / native libs / plugins PyInstaller can miss.
for pkg in (
    "customtkinter",   # theme assets
    "faiss",           # native libs
    "pdfminer",        # cmap data (pdfplumber backend)
    "pdfplumber",
    "fitz",            # PyMuPDF — native libs (PDF embedded-image extraction)
    "pymupdf",
    "odf",             # odfpy (ODT)
    "ebooklib",        # EPUB
    "striprtf",        # RTF
    "openpyxl",        # XLSX
    "bs4",             # HTML (веб-fetch/импорт)
    "certifi",         # CA-бандл для HTTPS (веб-стек — первый TLS-потребитель)
    "httpx",           # HTTP-клиент (LLM + веб-поиск/fetch/дипресёрч)
    "yaml",            # run profiles fallback parser
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# tiktoken registers encodings via the tiktoken_ext namespace + package metadata.
hiddenimports += collect_submodules("tiktoken_ext")
hiddenimports += ["tiktoken_ext.openai_public"]
hiddenimports += ["retrieval_enhance"]  # лениво импортится в notebook_chat (точный поиск)
hiddenimports += ["notebook_wiki"]      # лениво импортится в notebook_gui (компиляция вики)
hiddenimports += ["md3"]                # MD3-тема, импортится в gui.py/notebook_gui.py
hiddenimports += ["app_version"]        # версия для сайдбара (ленивый импорт в gui.py)
hiddenimports += ["md_render"]          # markdown→plain для чат-пузыря (импорт в notebook_gui)
# веб-стек (W1–W5): импортится в notebook_gui; перечисляем явно на всякий случай
hiddenimports += ["web_search", "web_fetch", "web_import", "deep_research"]
# faiss-cpu на PyPI НЕ содержит swigfaiss_avx2 в CI-колесе → faiss на старте
# логирует INFO «Could not load ... AVX2» и откатывается на базовый swigfaiss
# (retrieval работает, просто без AVX2). Бандлить нечего; шум в логе гасим в
# main.py (logging.getLogger("faiss") → WARNING). collect_all("faiss") выше уже
# кладёт то, что реально есть в колесе.
try:
    datas += copy_metadata("tiktoken")
except Exception:
    pass

# Bundle a prefetched tiktoken encoding so tokenization works fully offline.
_tk_cache = os.path.join(os.getcwd(), ".build", "tiktoken_cache")
if os.path.isdir(_tk_cache):
    datas += [(_tk_cache, "tiktoken_cache")]

# Run profiles (config/run_profiles.yaml) read relative to the package dir.
if os.path.isdir("config"):
    datas += [("config", "config")]

# App icon: bundle .ico/.png so the running app can set its window icon,
# and feed .ico/.icns to the EXE/BUNDLE for the file/dock icon.
for _ic in ("assets/icon.ico", "assets/icon.png"):
    if os.path.isfile(_ic):
        datas += [(_ic, "assets")]
_icon_ico = "assets/icon.ico" if os.path.isfile("assets/icon.ico") else None
_icon_icns = "assets/icon.icns" if os.path.isfile("assets/icon.icns") else None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "pytest", "_pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

# Windows .exe version resource (author/product metadata, shown in Properties).
_version_file = None
if sys.platform == "win32" and os.path.isfile(os.path.join("scripts", "version_info.txt")):
    _version_file = os.path.join("scripts", "version_info.txt")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NocturneDataForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # GUI app — no console window
    disable_windowed_traceback=False,
    icon=_icon_ico,           # Windows .exe icon (ignored on other OSes)
    version=_version_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NocturneDataForge",
)

# macOS: also wrap the one-dir build into a proper .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="NocturneDataForge.app",
        icon=_icon_icns,
        bundle_identifier="com.therudywolf.nocturnedataforge",
        info_plist={
            "CFBundleName": "Nocturne Data Forge",
            "CFBundleDisplayName": "Nocturne Data Forge",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Copyright (C) 2025 therudywolf — AGPL-3.0-or-later",
        },
    )
