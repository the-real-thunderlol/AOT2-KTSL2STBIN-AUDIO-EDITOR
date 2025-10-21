# AOT2 KTSL2STBIN AUDIO EDITOR BY THUNDERLOL - V1.0

**Complete toolkit for extracting and recompiling KVS/KTSL2STBIN audio files**

**Total package size:** ~845 MB (standalone, no installation required) (I am working on decreasing it)

---

## Quick Start

### 1. Extract Audio from Game

1. Run **KVS_Extractor.exe**
2. Select your game's `.ktsl2stbin` file
3. Choose output folder (for extracting ogg and ktsc files)
4. Wait for extraction to complete

**Output:**
- `ogg_files/` - Decrypted OGG audio files (edit these!)
- `kvs_files/` - Original KVS (sound) files

DO NOT TOUCH. THESE FILES HAVE ALL THE DATA NEEDED TO RECOMPILE
- `header.bin` - Game header
- `ogg_sizes.json` - Original sizes
- `padding_data.json` - Padding metadata
- `trailing_data.json` - Trailing metadata
- `loop_points.txt` - Loop positions

### 2. Replace Audio

1. Edit or replace files in ogg_files/
2. Keep same sample rate and channels as original
3. Use any audio editor (Audacity, etc.)

### 3. Recompile ktsl2stbin

1. Run **KVS_Compiler.exe**
2. Select `ogg_files/` folder (where all music files are)
3. Choose output location (FOR MODDED KTSL2STBIN)
4. Enter filename (`BGM_RECOMPILED.ktsl2stbin` is default name)
5. Wait for compilation.

### 4. Use Mod

1. Go To `C:\Program Files (x86)\Steam\steamapps\common\AoT2\FILE\SOUND\BGM`
2. Rename the original BGM.ktsl2bin file to "anything".
3. Copy and Paste the modded BGM_RECOMPILED.ktsl2stbin
4. RENAME BGM_RECOMPILED.ktsl2stbin to BGM.ktsl2stbin (takes over the original music pack)

---

### ADDITIONAL NOTES:
**For changing mp3 / any audio format to ogg, use this tool:**
https://www.freac.org/downloads-mainmenu-33/362-freac-117

**ID -> MUSIC NAME** (to check what you are editing) (WiP):
https://docs.google.com/spreadsheets/d/1FUFXEIM3AXOpg05Rq1yR0hf0UDvLssygsGvR8uv99vk/edit?usp=sharing

---

### WARNING:
DO NOT REPLACE SMALL .OGG FILES, THE WHOLE MUSIC PACK WILL BREAK (KNOWN: FILE 34 and 63)
- files that are about 1-2 seconds long
- you dont hear any sound when played
- only a few kilobytes (10KB-20KB)

**LINK TO MOD:** https://drive.google.com/drive/folders/1myaJEEMNdLp5tlpy9Ih-p14S7zKC67dl?usp=drive_link

**GITHUB:**https://github.com/the-real-thunderlol/AOT2-KTSL2STBIN-AUDIO-EDITOR
