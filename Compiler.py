#!/usr/bin/env python
import os
import sys
import json
import struct
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path


def get_bin_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'bin')


def check_ffmpeg():
    bin_path = get_bin_path()
    ffmpeg_exe = os.path.join(bin_path, 'ffmpeg.exe')
    if os.path.exists(ffmpeg_exe):
        try:
            subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True)
            return True
        except:
            pass
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_ffmpeg_path():
    bin_path = get_bin_path()
    ffmpeg_exe = os.path.join(bin_path, 'ffmpeg.exe')
    if os.path.exists(ffmpeg_exe):
        return ffmpeg_exe
    return 'ffmpeg'


def load_header(ogg_folder):
    header_path = os.path.join(ogg_folder, "..", "_header.bin")
    if not os.path.exists(header_path):
        header_path = os.path.join(ogg_folder, "_header.bin")
    if not os.path.exists(header_path):
        return None
    with open(header_path, 'rb') as f:
        header = f.read(96)
    if len(header) != 96:
        print(f"WARNING: Header is {len(header)} bytes, expected 96")
        return None
    return header


def load_ogg_sizes(ogg_folder):
    sizes_path = os.path.join(ogg_folder, "_ogg_sizes.json")
    if not os.path.exists(sizes_path):
        return None
    with open(sizes_path, 'r') as f:
        return json.load(f)


def load_loop_points(ogg_folder):
    loop_file = os.path.join(ogg_folder, "_loop_points.txt")
    loop_points = {}
    if not os.path.exists(loop_file):
        return loop_points
    try:
        with open(loop_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    filename, point = line.split(':', 1)
                    filename = filename.strip()
                    loop_points[filename] = int(point.strip())
    except Exception as e:
        print(f"Warning: Could not load loop points: {e}")
    return loop_points


def load_padding_data(ogg_folder):
    padding_file = os.path.join(ogg_folder, "_padding_data.json")
    if not os.path.exists(padding_file):
        return None
    try:
        with open(padding_file, 'r') as f:
            padding_data = json.load(f)
        return padding_data
    except Exception as e:
        print(f"Warning: Could not load padding data: {e}")
        return None


def load_trailing_data(ogg_folder):
    trailing_file = os.path.join(ogg_folder, "_trailing_data.json")
    if not os.path.exists(trailing_file):
        return None
    try:
        with open(trailing_file, 'r') as f:
            trailing_data = json.load(f)
        return trailing_data
    except Exception as e:
        print(f"Warning: Could not load trailing data: {e}")
        return None


def encrypt_ogg_data(ogg_data):
    encrypted = bytearray(ogg_data)
    for i in range(min(256, len(encrypted))):
        encrypted[i] ^= i
    return bytes(encrypted)


def create_kvs_from_ogg(ogg_file, loop_point=0, padding=None, trailing=None):
    with open(ogg_file, 'rb') as f:
        ogg_data = f.read()
    if not ogg_data.startswith(b'OggS'):
        raise ValueError(f"{os.path.basename(ogg_file)} is not a valid OGG file!")
    encrypted_ogg = encrypt_ogg_data(ogg_data)
    kvs_data = bytearray()
    kvs_data.extend(b'KOVS')
    kvs_data.extend(struct.pack('<I', len(encrypted_ogg)))
    kvs_data.extend(struct.pack('<I', loop_point))
    if padding is not None and len(padding) == 20:
        kvs_data.extend(padding)
    else:
        kvs_data.extend(b'\x00' * 20)
    kvs_data.extend(encrypted_ogg)
    if trailing is not None and len(trailing) > 0:
        kvs_data.extend(trailing)
    return bytes(kvs_data)


def compress_ogg(input_ogg, output_ogg, quality):
    try:
        ffmpeg = get_ffmpeg_path()
        subprocess.run([
            ffmpeg,
            '-i', input_ogg,
            '-c:a', 'libvorbis',
            '-q:a', str(quality),
            '-y',
            output_ogg
        ], capture_output=True, check=True, timeout=30)
        return True
    except:
        return False


def process_ogg_with_size_enforcement(ogg_file, target_ogg_size, loop_point=0, padding=None, trailing=None, temp_dir=None, has_ffmpeg=False):
    original_ogg = ogg_file
    current_ogg = ogg_file
    was_compressed = False
    compression_info = ""
    current_ogg_size = os.path.getsize(ogg_file)
    if current_ogg_size > target_ogg_size:
        if not has_ffmpeg:
            raise RuntimeError(
                f"File too large, ffmpeg needed!\n"
                f"File: {os.path.basename(ogg_file)}\n"
                f"Current: {current_ogg_size}, Target: {target_ogg_size}"
            )
        print(f"    Compressing ({current_ogg_size} -> {target_ogg_size})...")
        for quality in range(9, -1, -1):
            temp_ogg = os.path.join(temp_dir, f"temp_q{quality}.ogg")
            if compress_ogg(original_ogg, temp_ogg, quality):
                test_size = os.path.getsize(temp_ogg)
                if test_size <= target_ogg_size:
                    current_ogg = temp_ogg
                    current_ogg_size = test_size
                    was_compressed = True
                    reduction = (1 - current_ogg_size / os.path.getsize(original_ogg)) * 100
                    compression_info = f"Q{quality} (-{reduction:.1f}%)"
                    break
        if current_ogg_size > target_ogg_size:
            raise RuntimeError(
                f"Could not compress small enough!\n"
                f"File: {os.path.basename(ogg_file)}\n"
                f"Best: {current_ogg_size}, Target: {target_ogg_size}"
            )
    kvs_data = create_kvs_from_ogg(current_ogg, loop_point, padding, trailing)
    if current_ogg_size == target_ogg_size and not was_compressed:
        compression_info = "exact match"
    elif current_ogg_size < target_ogg_size:
        padding_needed = target_ogg_size - current_ogg_size
        if trailing is not None and len(trailing) > 0:
            kvs_data_without_trailing = kvs_data[:-len(trailing)]
            kvs_data_without_trailing += b'\x00' * padding_needed
            kvs_data = kvs_data_without_trailing + trailing
        else:
            kvs_data = kvs_data + (b'\x00' * padding_needed)
        if not was_compressed:
            compression_info = f"+{padding_needed}B padding"
    return kvs_data, was_compressed, compression_info


def convert_ogg_to_kvs_with_enforcement(ogg_dir, kvs_dir, ogg_sizes, loop_points, padding_data, trailing_data, has_ffmpeg):
    print(f"\n[1/2] Converting OGG files to KVS (with size enforcement)...")
    ogg_files = sorted(Path(ogg_dir).glob('*.ogg'))
    ogg_files = [f for f in ogg_files if not f.name.startswith('_')]
    if not ogg_files:
        print("ERROR: No .ogg files found!")
        return 0, [], {}
    os.makedirs(kvs_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="kvs_compile_")
    success_count = 0
    kvs_files = []
    stats = {'compressed': 0, 'padded': 0, 'exact': 0, 'errors': 0}
    for i, ogg_file in enumerate(ogg_files, 1):
        filename = ogg_file.stem
        print(f"  [{i}/{len(ogg_files)}] {ogg_file.name} -> ", end='', flush=True)
        try:
            loop_point = loop_points.get(ogg_file.name, 0)
            if ogg_sizes and filename in ogg_sizes:
                target_ogg_size = ogg_sizes[filename]
            else:
                target_ogg_size = os.path.getsize(ogg_file)
                print(f"NO SIZE DATA ", end='')
            file_padding = None
            if padding_data and filename in padding_data:
                file_padding = bytes.fromhex(padding_data[filename])
            file_trailing = None
            if trailing_data and filename in trailing_data:
                file_trailing = bytes.fromhex(trailing_data[filename])
            kvs_data, was_compressed, info = process_ogg_with_size_enforcement(
                str(ogg_file),
                target_ogg_size,
                loop_point,
                file_padding,
                file_trailing,
                temp_dir,
                has_ffmpeg
            )
            if was_compressed:
                stats['compressed'] += 1
            elif 'padding' in info:
                stats['padded'] += 1
            elif 'exact match' in info:
                stats['exact'] += 1
            output_filename = f"{filename}.kvs"
            output_path = os.path.join(kvs_dir, output_filename)
            with open(output_path, 'wb') as f:
                f.write(kvs_data)
            kvs_size = len(kvs_data) / (1024 * 1024)
            loop_info = f" [Loop: {loop_point}]" if loop_point > 0 else ""
            trail_info = f" [+{len(file_trailing)}B trail]" if file_trailing else ""
            size_info = f" [{info}]" if info else ""
            print(f"{output_filename} ({kvs_size:.2f} MB){size_info}{loop_info}{trail_info} OK")
            kvs_files.append(output_path)
            success_count += 1
        except Exception as e:
            print(f"FAILED ({e})")
            stats['errors'] += 1
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    print(f"\nConversion complete! {success_count}/{len(ogg_files)} files converted")
    print(f"  - Exact matches: {stats['exact']}")
    print(f"  - Compressed: {stats['compressed']}")
    print(f"  - Padded: {stats['padded']}")
    if stats['errors'] > 0:
        print(f"  - Errors: {stats['errors']}")
    return success_count, kvs_files, stats


def combine_kvs_to_ktsc2bin(kvs_dir, output_file, ktsr_header):
    print(f"\n[2/2] Combining KVS files into {os.path.basename(output_file)}...")
    kvs_files = sorted(Path(kvs_dir).glob('*.kvs'))
    if not kvs_files:
        print("ERROR: No .kvs files found!")
        return False
    with open(output_file, 'wb') as out:
        out.write(ktsr_header)
        for i, kvs_file in enumerate(kvs_files, 1):
            with open(kvs_file, 'rb') as f:
                kvs_data = f.read()
            out.write(kvs_data)
            print(f"  [{i}/{len(kvs_files)}] Added: {kvs_file.name} ({len(kvs_data) / (1024*1024):.2f} MB)")
    final_size = os.path.getsize(output_file)
    print(f"\nOutput file created: {output_file}")
    print(f"Final size: {final_size:,} bytes ({final_size / (1024*1024):.2f} MB)")
    return True


def main():
    root = tk.Tk()
    root.withdraw()
    print("=" * 70)
    print("AOT2 KTSL2STBIN Audio Compiler by Thunderlol")
    print("Compiles OGG -> KVS -> KTSL2STBIN")
    print("=" * 70)
    has_ffmpeg = check_ffmpeg()
    if has_ffmpeg:
        print("\nOK: ffmpeg detected (compression available)")
    else:
        print("\nWARNING: ffmpeg not found (compression disabled)")
        print("         Files larger than original will fail!")
    print("\n[Step 1] Select OGG files folder...")
    ogg_folder = filedialog.askdirectory(title="Select OGG files folder")
    if not ogg_folder:
        print("No folder selected. Exiting.")
        return
    print(f"Selected: {ogg_folder}")
    print("\n[Step 2] Loading metadata...")
    ktsr_header = load_header(ogg_folder)
    if ktsr_header is None:
        messagebox.showerror("Error", "Could not find _header.bin!\n\nMake sure you're selecting the ogg_files folder from extraction.")
        return
    print("OK: Loaded 96-byte KTSR header")
    ogg_sizes = load_ogg_sizes(ogg_folder)
    if ogg_sizes:
        print(f"OK: Loaded {len(ogg_sizes)} OGG size entries")
    else:
        print("WARNING: No _ogg_sizes.json found (size enforcement disabled)")
    loop_points = load_loop_points(ogg_folder)
    if loop_points:
        print(f"OK: Loaded {len(loop_points)} loop points")
    padding_data = load_padding_data(ogg_folder)
    if padding_data:
        print(f"OK: Loaded {len(padding_data)} padding data entries")
    else:
        print("WARNING: No _padding_data.json found (will use zero padding)")
    trailing_data = load_trailing_data(ogg_folder)
    if trailing_data:
        print(f"OK: Loaded {len(trailing_data)} trailing data entries")
    else:
        print("WARNING: No _trailing_data.json found (will omit trailing data)")
        print("         This will cause -4,736 byte size mismatch!")
    print("\n[Step 3] Select output location...")
    output_folder = filedialog.askdirectory(title="Select output folder")
    if not output_folder:
        print("No folder selected. Exiting.")
        return
    default_name = "BGM_RECOMPILED.ktsl2stbin"
    output_filename = simpledialog.askstring(
        "Output filename",
        "Enter output filename:",
        initialvalue=default_name
    )
    if not output_filename:
        print("No filename provided. Exiting.")
        return
    output_file = os.path.join(output_folder, output_filename)
    print(f"Output will be: {output_file}")
    kvs_temp_dir = os.path.join(output_folder, "kvs_files_compiled")
    success_count, kvs_files, stats = convert_ogg_to_kvs_with_enforcement(
        ogg_folder,
        kvs_temp_dir,
        ogg_sizes,
        loop_points,
        padding_data,
        trailing_data,
        has_ffmpeg
    )
    if success_count == 0:
        messagebox.showerror("Error", "Compilation failed!")
        return
    if not combine_kvs_to_ktsc2bin(kvs_temp_dir, output_file, ktsr_header):
        messagebox.showerror("Error", "Failed to create final file!")
        return
    final_size = os.path.getsize(output_file)
    messagebox.showinfo(
        "Complete!",
        f"Compilation complete!\n\n"
        f"Output file: {output_filename}\n"
        f"Location: {output_folder}\n"
        f"Size: {final_size:,} bytes ({final_size / (1024*1024):.2f} MB)\n"
        f"\nFiles processed: {success_count}\n"
        f"- Exact matches: {stats['exact']}\n"
        f"- Compressed: {stats['compressed']}\n"
        f"- Padded: {stats['padded']}\n"
        f"\nReady to use in game!"
    )


if __name__ == '__main__':
    main()
