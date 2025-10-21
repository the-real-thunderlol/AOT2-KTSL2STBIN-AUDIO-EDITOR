#!/usr/bin/env python
import os
import sys
import struct
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path


def decrypt_ogg_data(ogg_data):
    decrypted = bytearray(ogg_data)
    for i in range(min(256, len(decrypted))):
        decrypted[i] ^= i
    return bytes(decrypted)


def save_header(input_file, output_dir):
    with open(input_file, 'rb') as f:
        header = f.read(96)
    if not header.startswith(b'KTSR'):
        print("WARNING: File doesn't start with KTSR header!")
        print(f"         Found: {header[:4]}")
        response = messagebox.askyesno(
            "Unknown Header",
            "File doesn't have expected KTSR header.\n\nContinue anyway?"
        )
        if not response:
            return False
    header_path = os.path.join(output_dir, "_header.bin")
    with open(header_path, 'wb') as f:
        f.write(header)
    print(f"OK: Saved 96-byte KTSR header to _header.bin")
    return True


def extract_kvs_from_ktsc2bin(input_file, kvs_output_dir):
    print(f"\n[1/3] Extracting KVS files from {os.path.basename(input_file)}...")
    with open(input_file, 'rb') as f:
        byte_data = f.read()
    file_size = len(byte_data)
    kvs_positions = []
    start = 0
    while True:
        pos = byte_data.find(b'KOVS', start)
        if pos == -1:
            break
        kvs_positions.append(pos)
        start = pos + 1
    kns_positions = []
    start = 0
    while True:
        pos = byte_data.find(b'KTSS', start)
        if pos == -1:
            break
        kns_positions.append(pos)
        start = pos + 1
    if kvs_positions:
        positions = kvs_positions
        ext = '.kvs'
    elif kns_positions:
        positions = kns_positions
        ext = '.kns'
    else:
        print("ERROR: No audio files found in container!")
        return []
    print(f"Found {len(positions)} audio files")
    os.makedirs(kvs_output_dir, exist_ok=True)
    extracted_files = []
    for i, start_pos in enumerate(positions):
        if i < len(positions) - 1:
            end_pos = positions[i + 1]
        else:
            end_pos = file_size
        file_data = byte_data[start_pos:end_pos]
        filename = f"{i:03d}{ext}"
        output_path = os.path.join(kvs_output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(file_data)
        extracted_files.append(output_path)
        print(f"  [{i+1}/{len(positions)}] Extracted: {filename} ({len(file_data) / (1024*1024):.2f} MB)")
    print(f"Extraction complete! {len(positions)} files saved to: {kvs_output_dir}")
    return extracted_files


def extract_ogg_from_kvs(kvs_file, ogg_output_dir):
    with open(kvs_file, 'rb') as f:
        signature = f.read(4)
        if signature not in [b'KOVS', b'KTSS']:
            return False, None, None, None, None
        ogg_size = struct.unpack('<I', f.read(4))[0]
        loop_point = struct.unpack('<I', f.read(4))[0]
        padding = f.read(20)
        encrypted_ogg = f.read(ogg_size)
        trailing = f.read()
    decrypted_ogg = decrypt_ogg_data(encrypted_ogg)
    if not decrypted_ogg.startswith(b'OggS'):
        return False, None, None, None, None
    kvs_filename = Path(kvs_file).stem
    output_path = os.path.join(ogg_output_dir, f"{kvs_filename}.ogg")
    with open(output_path, 'wb') as f:
        f.write(decrypted_ogg)
    return True, output_path, loop_point, padding, trailing


def convert_all_kvs_to_ogg(kvs_dir, ogg_dir):
    print(f"\n[2/3] Decrypting and extracting OGG files from KVS...")
    kvs_files = sorted(Path(kvs_dir).glob('*.kvs'))
    kns_files = sorted(Path(kvs_dir).glob('*.kns'))
    all_files = kvs_files + kns_files
    if not all_files:
        print("ERROR: No .kvs or .kns files found!")
        return 0, {}, {}
    os.makedirs(ogg_dir, exist_ok=True)
    success_count = 0
    loop_points = {}
    ogg_sizes = {}
    padding_data = {}
    trailing_data = {}
    for i, kvs_file in enumerate(all_files, 1):
        print(f"  [{i}/{len(all_files)}] {kvs_file.name} -> ", end='', flush=True)
        success, output_path, loop_point, padding, trailing = extract_ogg_from_kvs(str(kvs_file), ogg_dir)
        if success:
            ogg_size = os.path.getsize(output_path)
            kvs_size = os.path.getsize(kvs_file)
            filename_stem = kvs_file.stem
            if loop_point > 0:
                loop_points[filename_stem] = loop_point
            ogg_sizes[filename_stem] = ogg_size
            padding_data[filename_stem] = padding.hex()
            if trailing:
                trailing_data[filename_stem] = trailing.hex()
            ogg_mb = ogg_size / (1024 * 1024)
            kvs_mb = kvs_size / (1024 * 1024)
            trail_info = f" [+{len(trailing)}B trailing]" if trailing else ""
            loop_info = f" [Loop: {loop_point}]" if loop_point > 0 else ""
            print(f"{filename_stem}.ogg ({ogg_mb:.2f} MB, was {kvs_mb:.2f} MB KVS){loop_info}{trail_info} OK")
            success_count += 1
        else:
            print("FAILED")
    print(f"\nConversion complete! {success_count}/{len(all_files)} files converted")
    if loop_points:
        loop_file = os.path.join(ogg_dir, "_loop_points.txt")
        with open(loop_file, 'w') as f:
            f.write("# Loop points for OGG files\n")
            for name, point in sorted(loop_points.items()):
                f.write(f"{name}.ogg: {point}\n")
        print(f"Loop points saved to: _loop_points.txt")
    sizes_file = os.path.join(ogg_dir, "_ogg_sizes.json")
    with open(sizes_file, 'w') as f:
        json.dump(ogg_sizes, f, indent=2, sort_keys=True)
    print(f"OGG sizes saved to: _ogg_sizes.json")
    padding_file = os.path.join(ogg_dir, "_padding_data.json")
    with open(padding_file, 'w') as f:
        json.dump(padding_data, f, indent=2, sort_keys=True)
    print(f"Padding data saved to: _padding_data.json")
    trailing_file = os.path.join(ogg_dir, "_trailing_data.json")
    with open(trailing_file, 'w') as f:
        json.dump(trailing_data, f, indent=2, sort_keys=True)
    print(f"Trailing data saved to: _trailing_data.json ({len(trailing_data)} files)")
    return success_count, loop_points, ogg_sizes


def main():
    root = tk.Tk()
    root.withdraw()
    print("=" * 70)
    print("AOT2 KTSL2STBIN Audio Extractor by Thunderlol")
    print("Extracts KTSL2STBIN -> KVS -> OGG")
    print("=" * 70)
    print("\n[Step 1] Select .ktsl2stbin file to extract...")
    input_file = filedialog.askopenfilename(
        title="Select .ktsl2stbin file",
        filetypes=[("KTSL2STBIN files", "*.ktsl2stbin"), ("All files", "*.*")]
    )
    if not input_file:
        print("No file selected. Exiting.")
        return
    print(f"Selected: {input_file}")
    print("\n[Step 2] Select output folder...")
    output_folder = filedialog.askdirectory(title="Select output folder")
    if not output_folder:
        print("No folder selected. Exiting.")
        return
    print(f"Output to: {output_folder}")
    kvs_dir = os.path.join(output_folder, "kvs_files")
    ogg_dir = os.path.join(output_folder, "ogg_files")
    if not save_header(input_file, output_folder):
        return
    extracted_files = extract_kvs_from_ktsc2bin(input_file, kvs_dir)
    if not extracted_files:
        messagebox.showerror("Error", "Extraction failed!")
        return
    success_count, loop_points, ogg_sizes = convert_all_kvs_to_ogg(kvs_dir, ogg_dir)
    if success_count > 0:
        loop_info = f"\nLoop points preserved: {len(loop_points)}" if loop_points else ""
        messagebox.showinfo(
            "Complete!",
            f"Extraction complete!\n\n"
            f"Output folder: {output_folder}\n"
            f"- _header.bin: Original KTSR header (96 bytes)\n"
            f"- kvs_files/: Original KVS files ({len(extracted_files)} files)\n"
            f"- ogg_files/: Decrypted OGG files ({success_count} files)\n"
            f"- _ogg_sizes.json: Original OGG sizes\n"
            f"- _padding_data.json: KVS padding metadata\n"
            f"- _trailing_data.json: KVS trailing metadata\n"
            f"{loop_info}\n\n"
            f"Ready for editing and recompilation!"
        )
    else:
        messagebox.showerror("Error", "OGG extraction failed!")


if __name__ == '__main__':
    main()
