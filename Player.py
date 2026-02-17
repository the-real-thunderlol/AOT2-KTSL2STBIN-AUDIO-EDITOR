#!/usr/bin/env python
"""
KVS Audio Player & Editor - Iteration 6

Features:
- Waveform zoom in/out
- Edit loop point
- Import mp4/ogg/wav, export to OGG
- Proper looping

Requirements: pip install sounddevice soundfile numpy
"""

import os
import csv
import json
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import sounddevice as sd
from PIL import Image, ImageDraw, ImageTk

CPU_COUNT = os.cpu_count() or 4


class AudioPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("AOT2 Audio Editor")
        self.root.geometry("1000x710")
        self.root.minsize(900, 600)

        self.ogg_files = []
        self.current_file = None
        self.audio_data = None
        self.audio_mono = None
        self.wave_drawing = False
        self.sample_rate = None
        self.duration = 0
        self.is_playing = False
        self.play_obj = None
        self.current_position = 0
        self.loop_points = {}
        self.loop_enabled = tk.BooleanVar(value=True)
        self.volume = 0.7
        self.stop_flag = False
        self.user_paused = False  # Track if user paused vs natural end
        self.playback_thread = None
        self.play_idx = 0

        self.zoom_level = 1.0
        self.view_start = 0.0
        self.edit_loop_sample = 0
        self.trim_start_sample = 0
        self.trim_end_sample = 0  # 0 means end of file
        self.stream = None

        self.prog_fill_id = None
        self.wave_pos_id = None

        # Store replaced audio in memory (file_stem -> (audio_data, sample_rate))
        self.replaced_audio = {}

        # Load song names from CSV
        self.song_names = {}
        self.skip_ids = set()
        self.load_song_names()

        self.build_ui()

    def load_song_names(self):
        """Load song names from CSV file"""
        csv_path = Path(__file__).parent / "data" / "sound_file_names.csv"
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        id_num = int(row['ID'])
                        name = row.get('Ingame Sound Name') or ''
                        name = name.strip()
                        if name == '/skip':
                            self.skip_ids.add(f"{id_num:03d}")
                        elif name:
                            self.song_names[id_num] = name
                    except (ValueError, KeyError, AttributeError):
                        pass

    def build_ui(self):
        # Prevent spacebar from activating focused buttons (space = play/pause only)
        self.root.bind_class("TButton", "<space>", lambda e: self.on_space(e))

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(top, text="Extract Music", command=self.extract_music).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(top, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Compile Folder", command=self.compile_folder).pack(side=tk.LEFT, padx=5)
        self.folder_label = ttk.Label(top, text="No folder loaded")
        self.folder_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text="About", command=self.show_about).pack(side=tk.RIGHT)

        middle = ttk.Frame(main)
        middle.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(middle, width=245)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        ttk.Label(left, text="Files:").pack(anchor=tk.W)
        scroll = ttk.Scrollbar(left)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(left, yscrollcommand=scroll.set)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<Double-Button-1>", self.on_select)
        self.listbox.bind("<Return>", self.on_select)
        self.listbox.bind("<space>", self.on_space)  # Prevent listbox capturing space
        scroll.config(command=self.listbox.yview)

        # Global keyboard shortcuts
        self.root.bind("<space>", self.on_space)
        self.root.bind("<Configure>", self.on_window_resize)

        right = ttk.Frame(middle)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        wave_frame = ttk.Frame(right)
        wave_frame.pack(fill=tk.X)

        ttk.Label(wave_frame, text="Waveform:").pack(side=tk.LEFT)
        ttk.Button(wave_frame, text="−", width=3, command=self.zoom_out).pack(side=tk.RIGHT, padx=2)
        ttk.Button(wave_frame, text="+", width=3, command=self.zoom_in).pack(side=tk.RIGHT, padx=2)
        ttk.Button(wave_frame, text="Fit", width=4, command=self.zoom_fit).pack(side=tk.RIGHT, padx=2)
        self.zoom_label = ttk.Label(wave_frame, text="100%")
        self.zoom_label.pack(side=tk.RIGHT, padx=5)

        self.wave_canvas = tk.Canvas(right, height=180, bg="#1a1a2e", highlightthickness=0)
        self.wave_canvas.pack(fill=tk.X, pady=5)
        self.wave_canvas.bind("<Button-1>", self.on_wave_click)
        self.wave_canvas.bind("<B1-Motion>", self.on_wave_drag)
        self.wave_canvas.bind("<Button-2>", self.on_middle_click)
        self.wave_canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.wave_canvas.bind("<Configure>", self.on_resize)
        self.wave_canvas.bind("<MouseWheel>", self.on_scroll)

        self.wave_scroll = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.on_wave_scroll)
        self.wave_scroll.pack(fill=tk.X)

        prog_frame = ttk.Frame(right)
        prog_frame.pack(fill=tk.X, pady=5)
        ttk.Label(prog_frame, text="Playback:").pack(side=tk.LEFT, padx=(0, 5))
        self.prog_canvas = tk.Canvas(prog_frame, height=30, bg="#16213e", highlightthickness=0)
        self.prog_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.prog_canvas.bind("<Button-1>", self.on_prog_click)
        self.prog_canvas.bind("<B1-Motion>", self.on_prog_drag)

        self.time_label = ttk.Label(right, text="0:00.000 / 0:00.000")
        self.time_label.pack(anchor=tk.W)

        ctrl = ttk.Frame(right)
        ctrl.pack(fill=tk.X, pady=10)

        # Player buttons: |<  <<  Play/Pause  >>  >|  Stop
        ttk.Button(ctrl, text="|◀", command=self.goto_start, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="◀◀", command=self.rewind_5s, width=4).pack(side=tk.LEFT, padx=2)
        self.play_btn = ttk.Button(ctrl, text="▶", command=self.toggle_play, width=6)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="▶▶", command=self.forward_5s, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="▶|", command=self.goto_end, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="⏹", command=self.stop_and_reset, width=4).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(ctrl, text="Loop", variable=self.loop_enabled).pack(side=tk.LEFT, padx=10)

        vol_frame = ttk.Frame(ctrl)
        vol_frame.pack(side=tk.RIGHT)
        ttk.Label(vol_frame, text="Vol:").pack(side=tk.LEFT)
        self.vol_slider = ttk.Scale(vol_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=80, command=self.set_vol)
        self.vol_slider.set(70)
        self.vol_slider.pack(side=tk.LEFT)

        # Replace Sound section
        replace_frame = ttk.LabelFrame(right, text="Replace Sound", padding=10)
        replace_frame.pack(fill=tk.X, pady=5)

        replace_row = ttk.Frame(replace_frame)
        replace_row.pack(fill=tk.X)

        ttk.Button(replace_row, text="Upload Sound", command=self.replace_sound).pack(side=tk.LEFT, padx=5)
        ttk.Button(replace_row, text="Revert", command=self.revert_sound).pack(side=tk.LEFT, padx=5)
        self.replace_label = ttk.Label(replace_row, text="")
        self.replace_label.pack(side=tk.LEFT, padx=10)

        # Edit Loop Point section
        edit_frame = ttk.LabelFrame(right, text="Edit Loop Point", padding=10)
        edit_frame.pack(fill=tk.X, pady=5)

        loop_row = ttk.Frame(edit_frame)
        loop_row.pack(fill=tk.X)

        ttk.Label(loop_row, text="Loop Time:", width=10).pack(side=tk.LEFT)
        self.loop_entry = ttk.Entry(loop_row, width=15)
        self.loop_entry.insert(0, "00:00:00.000")
        self.loop_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(loop_row, text="Update Time", command=self.apply_loop).pack(side=tk.LEFT, padx=5)

        # Orange button for loop
        style = ttk.Style()
        style.configure("Orange.TButton", background="#ffa500")
        loop_pos_btn = tk.Button(loop_row, text="Set to Playback Position", command=self.loop_from_pos,
                                  bg="#ffa500", activebackground="#ffb732")
        loop_pos_btn.pack(side=tk.LEFT, padx=5)

        # Trim Points section (for cutting audio when replacing)
        trim_frame = ttk.LabelFrame(right, text="Trim Points (for Replace)", padding=10)
        trim_frame.pack(fill=tk.X, pady=5)

        # Start point row
        start_row = ttk.Frame(trim_frame)
        start_row.pack(fill=tk.X, pady=2)

        ttk.Label(start_row, text="Start Time:", width=10).pack(side=tk.LEFT)
        self.trim_start_entry = ttk.Entry(start_row, width=15)
        self.trim_start_entry.insert(0, "00:00:00.000")
        self.trim_start_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(start_row, text="Update Time", command=self.apply_trim_start).pack(side=tk.LEFT, padx=5)

        # Green button for start
        start_pos_btn = tk.Button(start_row, text="Set to Playback Position", command=self.trim_start_from_pos,
                                   bg="#4ecca3", activebackground="#6fd9b5")
        start_pos_btn.pack(side=tk.LEFT, padx=5)

        # End point row
        end_row = ttk.Frame(trim_frame)
        end_row.pack(fill=tk.X, pady=2)

        ttk.Label(end_row, text="End Time:", width=10).pack(side=tk.LEFT)
        self.trim_end_entry = ttk.Entry(end_row, width=15)
        self.trim_end_entry.insert(0, "00:00:00.000")
        self.trim_end_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(end_row, text="Update Time", command=self.apply_trim_end).pack(side=tk.LEFT, padx=5)

        # Light red button for end
        end_pos_btn = tk.Button(end_row, text="Set to Playback Position", command=self.trim_end_from_pos,
                                 bg="#ff6b6b", activebackground="#ff8a8a")
        end_pos_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(end_row, text="Reset", command=self.reset_trim_points).pack(side=tk.LEFT, padx=10)

        # Apply Changes button (combines Replace OGG + Update Loop File + Trim)
        ttk.Button(right, text="Apply Changes", command=self.apply_changes).pack(fill=tk.X, pady=10)

        self.status = ttk.Label(main, text="Ready | Right-click waveform to set loop point")
        self.status.pack(anchor=tk.W)

    def load_folder(self):
        folder = filedialog.askdirectory(title="Select OGG folder")
        if not folder:
            return

        files = sorted(Path(folder).glob("*.ogg"))
        # Filter out files starting with _ and skip IDs 034, 063
        self.ogg_files = []
        for f in files:
            if f.name.startswith("_"):
                continue
            stem = f.stem
            if stem in self.skip_ids:
                continue
            self.ogg_files.append(f)

        self.current_folder = folder

        if not self.ogg_files:
            messagebox.showwarning("Warning", "No OGG files found")
            return

        # Create temp folder for waveform cache
        self.temp_folder = Path(folder) / "temp"
        self.temp_folder.mkdir(exist_ok=True)

        # Check which files need waveform cache generation
        missing_cache = []
        for f in self.ogg_files:
            cache_file = self.temp_folder / f"{f.stem}.npy"
            if not cache_file.exists():
                missing_cache.append(f)

        # Generate missing caches with progress
        if missing_cache:
            self.generate_waveform_caches(missing_cache)

        self.load_loops(folder)
        self.folder_label.config(text=folder)

        self.listbox.delete(0, tk.END)
        for f in self.ogg_files:
            display_name = self.get_display_name(f)
            self.listbox.insert(tk.END, display_name)

        self.status.config(text=f"Loaded {len(self.ogg_files)} files")

    def _process_single_waveform(self, f):
        """Worker function to process a single file's waveform (runs in thread)"""
        try:
            # Read audio and compute peaks
            data, sr = sf.read(str(f), dtype="float32")
            if len(data.shape) == 1:
                mono = data
            else:
                mono = np.mean(data, axis=1)

            # Calculate peaks for standard width (800 pixels)
            w = 800
            total_samples = len(mono)
            samples_per_pixel = max(1, total_samples // w)
            usable = (total_samples // samples_per_pixel) * samples_per_pixel

            if usable > 0:
                reshaped = np.abs(mono[:usable]).reshape(-1, samples_per_pixel)
                peaks = np.max(reshaped, axis=1).astype(np.float32)
            else:
                peaks = np.zeros(w, dtype=np.float32)

            # Save cache (peaks + duration + sample_rate)
            cache_file = self.temp_folder / f"{f.stem}.npy"
            cache_data = {
                'peaks': peaks,
                'duration': len(data) / sr,
                'sample_rate': sr
            }
            np.save(cache_file, cache_data, allow_pickle=True)
            return f.stem, True
        except Exception:
            return f.stem, False

    def generate_waveform_caches(self, files):
        """Generate waveform cache files with progress dialog using max threads"""
        total = len(files)
        if total == 0:
            return

        # Create progress window
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Generating Waveforms")
        progress_win.geometry("400x100")
        progress_win.transient(self.root)
        progress_win.grab_set()

        label = ttk.Label(progress_win, text="Preparing...")
        label.pack(pady=10)

        progress = ttk.Progressbar(progress_win, length=350, mode='determinate')
        progress.pack(pady=10)

        # Use maximum threads (CPU count)
        max_workers = os.cpu_count() or 4
        completed = [0]  # Use list for mutable counter in closure

        def update_progress():
            """Update progress bar from main thread"""
            if completed[0] < total:
                progress['value'] = (completed[0] / total) * 100
                label.config(text=f"Processing waveforms ({completed[0]}/{total}) - {max_workers} threads")
                progress_win.after(50, update_progress)

        def run_threaded():
            """Run all waveform processing in parallel"""
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_single_waveform, f): f for f in files}

                for future in as_completed(futures):
                    completed[0] += 1

            # Signal completion
            progress_win.after(0, finish_progress)

        def finish_progress():
            """Clean up progress window"""
            progress['value'] = 100
            progress_win.destroy()

        # Start progress updates
        update_progress()

        # Run processing in background thread
        thread = threading.Thread(target=run_threaded, daemon=True)
        thread.start()

        # Wait for completion while keeping UI responsive
        while thread.is_alive():
            progress_win.update()
            time.sleep(0.01)

    def get_display_name(self, file_path):
        """Get display name: 'XX - Song Name' or just filename"""
        stem = file_path.stem
        try:
            # Try to parse as number
            id_num = int(stem)
            if id_num in self.song_names:
                return f"{id_num:02d} - {self.song_names[id_num]}"
        except ValueError:
            pass
        return stem

    def import_file(self):
        path = filedialog.askopenfilename(
            title="Import Audio File",
            filetypes=[
                ("Audio files", "*.ogg *.wav *.flac *.mp3"),
                ("OGG", "*.ogg"),
                ("WAV", "*.wav"),
                ("FLAC", "*.flac"),
                ("All", "*.*")
            ]
        )
        if not path:
            return

        self.load_file(Path(path))

    def replace_sound(self):
        """Replace current sound with a new file (kept in memory until Save)"""
        if self.current_file is None:
            messagebox.showwarning("Warning", "No file selected")
            return

        path = filedialog.askopenfilename(
            title="Select Replacement Audio",
            filetypes=[
                ("Audio files", "*.ogg *.wav *.flac *.mp3"),
                ("OGG", "*.ogg"),
                ("WAV", "*.wav"),
                ("MP3", "*.mp3"),
                ("All", "*.*")
            ]
        )
        if not path:
            return

        try:
            # Load and process new audio data
            data, sr = sf.read(path, dtype="float32")

            # Convert to stereo
            if len(data.shape) == 1:
                data = np.column_stack([data, data])
            elif data.shape[1] > 2:
                data = data[:, :2]
            elif data.shape[1] == 1:
                data = np.column_stack([data[:, 0], data[:, 0]])

            # Apply trim points if set
            trim_start = self.trim_start_sample
            trim_end = self.trim_end_sample if self.trim_end_sample > 0 else len(data)

            # Clamp to valid range
            trim_start = max(0, min(trim_start, len(data) - 1))
            trim_end = max(trim_start + 1, min(trim_end, len(data)))

            if trim_start > 0 or trim_end < len(data):
                data = data[trim_start:trim_end]
                self.status.config(text=f"Trimmed: {trim_start} to {trim_end} samples")

            stem = self.current_file.stem

            # Convert to OGG using oggenc2 (via temp WAV)
            if hasattr(self, 'current_folder'):
                import subprocess
                import shutil
                pre_replace = self.temp_folder / "pre-replace" if self.temp_folder else Path(self.current_folder) / "temp" / "pre-replace"
                pre_replace.mkdir(exist_ok=True)

                temp_wav = pre_replace / f"_temp_{stem}.wav"
                pre_replace_ogg = pre_replace / f"{stem}.ogg"

                # Save temp WAV
                sf.write(str(temp_wav), data, sr, format="WAV", subtype="PCM_16")

                # Convert with oggenc2
                oggenc_path = Path(__file__).parent / "tool" / "oggenc2.exe"
                if not oggenc_path.exists():
                    temp_wav.unlink()
                    messagebox.showerror("Error", "oggenc2.exe not found in tool folder")
                    return

                result = subprocess.run([
                    str(oggenc_path),
                    "-q", "6",
                    "-o", str(pre_replace_ogg),
                    str(temp_wav)
                ], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)

                # Delete temp WAV immediately
                if temp_wav.exists():
                    temp_wav.unlink()

                if result.returncode != 0:
                    messagebox.showerror("Error", f"OGG conversion failed:\n{result.stderr}")
                    return

                # Backup original waveform cache
                if hasattr(self, 'temp_folder') and self.temp_folder:
                    if stem not in self.replaced_audio:  # First replacement
                        cache_file = self.temp_folder / f"{stem}.npy"
                        if cache_file.exists():
                            backup_cache = pre_replace / f"{stem}.npy"
                            if not backup_cache.exists():
                                shutil.copy2(cache_file, backup_cache)

            # Mark as replaced (for UI display)
            self.replaced_audio[stem] = True

            # Stop existing stream (sample rate may have changed)
            was_playing = self.is_playing
            self.stop_flag = True
            self.is_playing = False
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            self.play_btn.config(text="▶")

            # Set all internal state BEFORE any GUI updates
            self.audio_data = data
            self.audio_mono = np.mean(data, axis=1)
            self.sample_rate = sr
            self.duration = len(data) / sr
            self.current_position = 0
            self.zoom_level = 1.0
            self.view_start = 0.0
            self.play_idx = 0
            self.stop_flag = False

            # Reset trim points and loop point for the new uploaded audio
            self.trim_start_sample = 0
            self.trim_end_sample = 0
            self.edit_loop_sample = 0
            self.set_trim_start_entry(0)
            self.set_trim_end_entry(len(data))
            self.set_loop_entry(0)

            # Now update GUI (all data is ready)
            self.replace_label.config(text="[REPLACED]", foreground="red")
            self.status.config(text=f"Replaced with: {Path(path).name}")
            self.draw_wave()
            self.draw_prog(0)
            self.update_scroll()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    def revert_sound(self):
        """Revert to original sound file"""
        if self.current_file is None:
            return

        stem = self.current_file.stem
        if stem in self.replaced_audio:
            del self.replaced_audio[stem]

        # Restore original waveform cache from pre-replace backup
        if hasattr(self, 'current_folder') and hasattr(self, 'temp_folder') and self.temp_folder:
            import shutil
            pre_replace = self.temp_folder / "pre-replace"
            backup_cache = pre_replace / f"{stem}.npy"
            if backup_cache.exists():
                cache_file = self.temp_folder / f"{stem}.npy"
                shutil.copy2(backup_cache, cache_file)

        # Reload original file
        self.replace_label.config(text="", foreground="black")
        threading.Thread(target=self.load_file_threaded, args=(self.current_file,), daemon=True).start()
        self.status.config(text=f"Reverted to original: {self.current_file.name}")

    def load_loops(self, folder):
        self.loop_points = {}
        path = os.path.join(folder, "_loop_points.txt")
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    name, val = line.split(":", 1)
                    name = name.strip().replace(".ogg", "")
                    self.loop_points[name] = int(val.strip())

    def on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.ogg_files[idx]

        # Visual selection first
        self.listbox.see(idx)
        self.listbox.activate(idx)
        self.root.update_idletasks()

        # Stop audio instantly (close stream in background)
        self.stop_flag = True
        self.is_playing = False
        self.play_btn.config(text="▶")
        if hasattr(self, 'stream') and self.stream:
            threading.Thread(target=self._close_stream, daemon=True).start()

        self.current_file = path
        self.audio_data = None
        self.audio_mono = None
        self.wave_canvas.delete("all")
        self.draw_prog(0)
        self.status.config(text=f"Loading {path.name}...")
        self.root.update()

        # Prepare for progressive drawing
        self.wave_pixels = None
        self.wave_progress = None
        self.wave_drawing = False
        self.wave_w = 800
        self.wave_h = 180

        # Start update loop on main thread FIRST
        self.start_wave_update_loop()

        # Then load in background thread
        threading.Thread(target=self.load_file_threaded, args=(path,), daemon=True).start()

    def start_wave_update_loop(self):
        """Update loop running on main thread"""
        # Skip during resize
        if getattr(self, '_resizing', False):
            if self.wave_drawing or self.wave_pixels is None:
                self.root.after(16, self.start_wave_update_loop)
            return

        if self.wave_pixels is not None and self.wave_progress is not None:
            total_done = sum(self.wave_progress)
            img = Image.fromarray(self.wave_pixels.copy(), 'RGB')
            self.wave_photo = ImageTk.PhotoImage(img)
            self.wave_canvas.delete("all")
            self.wave_canvas.create_image(0, 0, anchor=tk.NW, image=self.wave_photo)
            self.status.config(text=f"Drawing waveform... {total_done}/{self.wave_w}")

        if self.wave_drawing or self.wave_pixels is None:
            self.root.after(16, self.start_wave_update_loop)  # 60fps

    def load_file_threaded(self, path):
        try:
            stem = path.stem
            use_cache = False
            cached_peaks = None

            # Check for waveform cache
            if hasattr(self, 'temp_folder') and self.temp_folder:
                cache_file = self.temp_folder / f"{stem}.npy"
                if cache_file.exists():
                    try:
                        cache_data = np.load(cache_file, allow_pickle=True).item()
                        cached_peaks = cache_data['peaks']
                        use_cache = True
                    except:
                        pass

            # Check if there's a replacement in memory
            is_replaced = stem in self.replaced_audio
            # Update label immediately with captured value
            self.root.after(0, lambda r=is_replaced: self.replace_label.config(
                text="[REPLACED]" if r else "",
                foreground="red" if r else "black"
            ))

            if is_replaced:
                data, sr = self.replaced_audio[stem]
                use_cache = False  # Don't use cache for replaced audio
            else:
                data, sr = sf.read(str(path), dtype="float32")

                # Convert to stereo
                if len(data.shape) == 1:
                    data = np.column_stack([data, data])
                elif data.shape[1] > 2:
                    data = data[:, :2]
                elif data.shape[1] == 1:
                    data = np.column_stack([data[:, 0], data[:, 0]])

            self.audio_data = data
            self.sample_rate = sr
            self.duration = len(data) / sr
            self.current_position = 0
            self.zoom_level = 1.0
            self.view_start = 0.0

            if path.stem in self.loop_points:
                self.edit_loop_sample = self.loop_points[path.stem]
            else:
                self.edit_loop_sample = 0

            # Schedule UI updates on main thread
            self.root.after(0, self.after_load, path)

            # Use cached waveform if available (instant), otherwise draw progressively
            if use_cache and cached_peaks is not None:
                self.audio_mono = np.mean(data, axis=1)
                self.root.after(0, lambda p=cached_peaks: self.draw_wave_from_cache(p))
            else:
                self.draw_wave_progressive(data)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

    def draw_wave_from_cache(self, peaks):
        """Draw waveform instantly from cached peaks"""
        w = self.wave_canvas.winfo_width()
        h = self.wave_canvas.winfo_height()
        if w < 10:
            w = 800
        if h < 10:
            h = 180

        self.wave_w = w
        self.wave_h = h
        mid = h // 2

        # Create pixel array
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        pixels[:, :] = [26, 26, 46]  # Background
        pixels[mid, :] = [51, 51, 51]  # Center line

        # Scale peaks to current width
        if len(peaks) != w:
            # Resample peaks to match current width
            indices = np.linspace(0, len(peaks) - 1, w).astype(int)
            peaks = peaks[indices]

        # Draw waveform from cached peaks
        for x, peak_val in enumerate(peaks):
            peak = int(peak_val * (mid - 5))
            if peak > 0:
                pixels[mid - peak:mid + peak + 1, x] = [78, 204, 163]

        # Draw markers if we have audio data
        if self.audio_data is not None and len(self.audio_data) > 0:
            total_samples = len(self.audio_data)

            # Draw trim start marker (green)
            if self.trim_start_sample > 0:
                tsx = int(self.trim_start_sample / total_samples * w)
                if 0 <= tsx < w:
                    pixels[:, tsx] = [0, 255, 0]  # Green

            # Draw trim end marker (orange)
            trim_end = self.trim_end_sample if self.trim_end_sample > 0 else total_samples
            if trim_end < total_samples:
                tex = int(trim_end / total_samples * w)
                if 0 <= tex < w:
                    pixels[:, tex] = [255, 165, 0]  # Orange

            # Draw loop marker (red)
            if self.edit_loop_sample > 0:
                lx = int(self.edit_loop_sample / total_samples * w)
                if 0 <= lx < w:
                    pixels[:, lx] = [255, 107, 107]  # Red

        img = Image.fromarray(pixels, 'RGB')
        draw = ImageDraw.Draw(img)

        # Add text labels for markers
        if self.audio_data is not None and len(self.audio_data) > 0:
            total_samples = len(self.audio_data)
            if self.trim_start_sample > 0:
                tsx = int(self.trim_start_sample / total_samples * w)
                if 0 <= tsx < w:
                    draw.text((tsx + 3, 2), "start", fill="#00ff00")
            trim_end = self.trim_end_sample if self.trim_end_sample > 0 else total_samples
            if trim_end < total_samples:
                tex = int(trim_end / total_samples * w)
                if 0 <= tex < w:
                    draw.text((tex + 3, 2), "end", fill="#ffa500")
            if self.edit_loop_sample > 0:
                lx = int(self.edit_loop_sample / total_samples * w)
                if 0 <= lx < w:
                    draw.text((lx + 3, 14), "L", fill="#ff6b6b")

        self.wave_photo = ImageTk.PhotoImage(img)
        self.wave_canvas.delete("all")
        self.wave_canvas.create_image(0, 0, anchor=tk.NW, image=self.wave_photo)

        # Create position line
        self.wave_pos_id = self.wave_canvas.create_line(0, 0, 0, h, fill="white", width=2)
        self.wave_start_sample = 0
        self.wave_end_sample = len(self.audio_data) if self.audio_data is not None else 0

        self.status.config(text=f"Loaded: {self.current_file.name} ({self.fmt_full(self.duration)}) | SR: {self.sample_rate}")

    def after_load(self, path):
        self.set_loop_entry(self.edit_loop_sample)
        # Reset trim points for new file
        self.trim_start_sample = 0
        self.trim_end_sample = 0
        self.set_trim_start_entry(0)
        if self.audio_data is not None:
            self.set_trim_end_entry(len(self.audio_data))
        self.draw_prog(0)
        self.update_scroll()

    def draw_wave_progressive(self, data):
        """Draw waveform progressively using all CPU cores with live updates"""
        w = self.wave_canvas.winfo_width()
        h = self.wave_canvas.winfo_height()
        if w < 10:
            w = 800
        if h < 10:
            h = 180

        self.wave_w = w
        self.wave_h = h
        mid = h // 2
        total_samples = len(data)

        # Calculate mono first
        mono = np.mean(data, axis=1)
        self.audio_mono = mono

        # Create shared pixel array - accessible by main thread
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        pixels[:, :] = [26, 26, 46]  # Background
        pixels[mid, :] = [51, 51, 51]  # Center line

        samples_per_pixel = max(1, total_samples // w)
        n_threads = CPU_COUNT
        chunk_width = max(1, w // n_threads)

        # Shared progress array
        progress = [0] * n_threads

        # Set shared refs for main thread update loop
        self.wave_pixels = pixels
        self.wave_progress = progress
        self.wave_drawing = True

        def process_chunk(thread_id):
            """Process a chunk of the waveform with live updates"""
            start_x = thread_id * chunk_width
            end_x = w if thread_id == n_threads - 1 else (thread_id + 1) * chunk_width
            local_count = 0

            for x in range(start_x, end_x):
                start_idx = x * samples_per_pixel
                end_idx = min((x + 1) * samples_per_pixel, total_samples)
                if start_idx >= total_samples:
                    break
                chunk = mono[start_idx:end_idx]
                if len(chunk) > 0:
                    peak = int(np.max(np.abs(chunk)) * (mid - 5))
                    if peak > 0:
                        pixels[mid - peak:mid + peak + 1, x] = [78, 204, 163]

                local_count += 1
                progress[thread_id] = local_count

            return thread_id

        # Run all chunks in parallel
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(process_chunk, i) for i in range(n_threads)]
            for f in futures:
                f.result()

        self.wave_drawing = False
        # Final update with loop marker
        self.root.after(50, self.finalize_wave)

    def update_wave_image(self, pixels, section, total):
        """Update waveform image on main thread"""
        img = Image.fromarray(pixels, 'RGB')
        self.wave_photo = ImageTk.PhotoImage(img)
        self.wave_canvas.delete("all")
        self.wave_canvas.create_image(0, 0, anchor=tk.NW, image=self.wave_photo)
        self.status.config(text=f"Drawing waveform... {section}/{total}")

    def finalize_wave(self):
        """Finalize waveform with markers"""
        if self.audio_mono is None:
            return
        self.draw_wave()
        self.status.config(text=f"Loaded: {self.current_file.name} ({self.fmt_full(self.duration)}) | SR: {self.sample_rate}")

    def draw_wave(self):
        self.wave_canvas.delete("all")
        if self.audio_data is None:
            return

        w = self.wave_canvas.winfo_width()
        h = self.wave_canvas.winfo_height()
        if w < 10:
            w = 800
        if h < 10:
            h = 180

        mid = h // 2
        mono = self.audio_mono  # Use cached mono
        total_samples = len(mono)

        visible_samples = int(total_samples / self.zoom_level)
        start_sample = int(self.view_start * total_samples)
        end_sample = min(start_sample + visible_samples, total_samples)

        if end_sample <= start_sample:
            return

        # Calculate peaks with vectorized numpy (much faster)
        view_data = mono[start_sample:end_sample]
        n_samples = len(view_data)

        # Create pixel array directly (faster than PIL draw)
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        pixels[:, :] = [26, 26, 46]  # Background #1a1a2e
        pixels[mid, :] = [51, 51, 51]  # Center line #333333

        # Full resolution (1 pixel per bar)
        samples_per_pixel = max(1, n_samples // w)
        usable = (n_samples // samples_per_pixel) * samples_per_pixel
        if usable > 0:
            reshaped = np.abs(view_data[:usable]).reshape(-1, samples_per_pixel)
            peaks = np.max(reshaped, axis=1)
            peaks = (peaks * (mid - 5)).astype(int)

            # Draw waveform at full resolution
            for x, peak in enumerate(peaks):
                if x < w and peak > 0:
                    pixels[mid - peak:mid + peak + 1, x] = [78, 204, 163]

        img = Image.fromarray(pixels, 'RGB')
        draw = ImageDraw.Draw(img)

        # Draw trim start marker (green)
        if self.trim_start_sample > 0:
            trim_start_ratio = (self.trim_start_sample - start_sample) / (end_sample - start_sample)
            if 0 <= trim_start_ratio <= 1:
                tsx = int(trim_start_ratio * w)
                draw.line([(tsx, 0), (tsx, h)], fill="#00ff00", width=2)
                draw.text((tsx + 3, 2), "start", fill="#00ff00")

        # Draw trim end marker (orange)
        trim_end = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)
        if trim_end < len(self.audio_data):
            trim_end_ratio = (trim_end - start_sample) / (end_sample - start_sample)
            if 0 <= trim_end_ratio <= 1:
                tex = int(trim_end_ratio * w)
                draw.line([(tex, 0), (tex, h)], fill="#ffa500", width=2)
                draw.text((tex + 3, 2), "end", fill="#ffa500")

        # Draw loop marker (red)
        if self.edit_loop_sample > 0:
            loop_ratio = (self.edit_loop_sample - start_sample) / (end_sample - start_sample)
            if 0 <= loop_ratio <= 1:
                lx = int(loop_ratio * w)
                draw.line([(lx, 0), (lx, h)], fill="#ff6b6b", width=2)
                draw.text((lx + 3, 14), "L", fill="#ff6b6b")

        # Convert to PhotoImage and display
        self.wave_photo = ImageTk.PhotoImage(img)
        self.wave_canvas.create_image(0, 0, anchor=tk.NW, image=self.wave_photo)

        # Create position line (will be moved during playback)
        pos_sample = int(self.current_position * self.sample_rate)
        pos_ratio = (pos_sample - start_sample) / (end_sample - start_sample)
        px = int(pos_ratio * w) if 0 <= pos_ratio <= 1 else 0
        self.wave_pos_id = self.wave_canvas.create_line(px, 0, px, h, fill="white", width=2)

        self.wave_w = w
        self.wave_h = h
        self.wave_start_sample = start_sample
        self.wave_end_sample = end_sample

    def draw_prog(self, ratio):
        self.prog_canvas.delete("all")
        w = self.prog_canvas.winfo_width()
        h = self.prog_canvas.winfo_height()
        if w < 10:
            w = 800
        if h < 10:
            h = 30

        px = int(ratio * w)
        self.prog_fill_id = self.prog_canvas.create_rectangle(0, 0, px, h, fill="#4ecca3", outline="")

        if self.audio_data is not None and len(self.audio_data) > 0:
            total = len(self.audio_data)

            # Draw trim start marker (green)
            if self.trim_start_sample > 0:
                tsx = int(self.trim_start_sample / total * w)
                self.prog_canvas.create_line(tsx, 0, tsx, h, fill="#00ff00", width=2)

            # Draw trim end marker (orange)
            trim_end = self.trim_end_sample if self.trim_end_sample > 0 else total
            if trim_end < total:
                tex = int(trim_end / total * w)
                self.prog_canvas.create_line(tex, 0, tex, h, fill="#ffa500", width=2)

            # Draw loop marker (red)
            if self.edit_loop_sample > 0:
                lx = int(self.edit_loop_sample / total * w)
                self.prog_canvas.create_line(lx, 0, lx, h, fill="#ff6b6b", width=2)

        self.time_label.config(text=f"{self.fmt_full(ratio * self.duration)} / {self.fmt_full(self.duration)}")

    def update_scroll(self):
        if self.zoom_level <= 1:
            self.wave_scroll.set(0, 1)
        else:
            visible = 1 / self.zoom_level
            self.wave_scroll.set(self.view_start, self.view_start + visible)

    def on_wave_scroll(self, *args):
        if args[0] == "moveto":
            self.view_start = max(0, min(float(args[1]), 1 - 1/self.zoom_level))
            self.draw_wave()
            self.update_scroll()

    def on_scroll(self, event):
        # Calculate cursor position as ratio of visible area
        if self.audio_data is not None:
            w = self.wave_canvas.winfo_width()
            cursor_ratio = event.x / w if w > 0 else 0.5
            # Convert to position in full audio
            visible = 1 / self.zoom_level
            cursor_pos = self.view_start + cursor_ratio * visible
        else:
            cursor_pos = 0.5

        if event.delta > 0:
            self.zoom_in(cursor_pos)
        else:
            self.zoom_out(cursor_pos)

    def zoom_in(self, center_pos=None):
        old_zoom = self.zoom_level
        self.zoom_level = min(self.zoom_level * 1.5, 50)
        if self.zoom_level == old_zoom:
            return  # Already at max, don't shift view
        self._center_on_position(center_pos)
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.draw_wave()
        self.update_scroll()

    def zoom_out(self, center_pos=None):
        old_zoom = self.zoom_level
        self.zoom_level = max(self.zoom_level / 1.5, 1)
        if self.zoom_level == old_zoom:
            return  # Already at min, don't shift view
        self._center_on_position(center_pos)
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        self.draw_wave()
        self.update_scroll()

    def _center_on_position(self, center_pos=None):
        # Center view on given position (or playback position if None)
        if center_pos is None:
            # Fall back to playback position for button clicks
            center_pos = self.current_position / self.duration if self.duration > 0 else 0

        visible = 1 / self.zoom_level
        self.view_start = max(0, min(center_pos - visible / 2, 1 - visible))

    def zoom_fit(self):
        self.zoom_level = 1.0
        self.view_start = 0.0
        self.zoom_label.config(text="100%")
        self.draw_wave()
        self.update_scroll()

    def on_window_resize(self, event):
        # Catch window maximize/restore from title bar
        if event.widget == self.root:
            self._resizing = True
            if hasattr(self, '_resize_clear_id'):
                self.root.after_cancel(self._resize_clear_id)
            self._resize_clear_id = self.root.after(300, self._clear_resizing)

    def on_resize(self, event):
        # Simple debounce - just cancel previous and schedule new
        self._resizing = True
        if hasattr(self, '_resize_after_id'):
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(50, self._do_resize)

    def _do_resize(self):
        if self.audio_data is not None:
            self.draw_wave()
            self.draw_prog(self.current_position / self.duration if self.duration else 0)
        # Keep _resizing True for a bit longer to prevent accidental clicks after resize
        if hasattr(self, '_resize_clear_id'):
            self.root.after_cancel(self._resize_clear_id)
        self._resize_clear_id = self.root.after(200, self._clear_resizing)

    def _clear_resizing(self):
        self._resizing = False

    def pos_from_wave_x(self, x):
        w = self.wave_canvas.winfo_width()
        total_samples = len(self.audio_data)
        visible_samples = int(total_samples / self.zoom_level)
        start_sample = int(self.view_start * total_samples)
        sample = start_sample + int((x / w) * visible_samples)
        return max(0, min(sample, total_samples - 1))

    def on_wave_click(self, event):
        if getattr(self, '_resizing', False):
            return
        if self.audio_data is None:
            return
        sample = self.pos_from_wave_x(event.x)
        self.current_position = sample / self.sample_rate
        ratio = self.current_position / self.duration
        self.update_position_only(ratio)

        # If playing, just update play index (no restart)
        if self.is_playing:
            self.play_idx = sample
        else:
            # Force canvas update when paused
            self.wave_canvas.update_idletasks()
            self.prog_canvas.update_idletasks()

    def on_wave_drag(self, event):
        self.on_wave_click(event)

    def on_middle_click(self, event):
        """Start middle-click panning"""
        self._pan_start_x = event.x
        self._pan_start_view = self.view_start

    def on_middle_drag(self, event):
        """Pan waveform with middle mouse drag"""
        if self.audio_data is None or self.zoom_level <= 1:
            return
        if not hasattr(self, '_pan_start_x'):
            return

        w = self.wave_canvas.winfo_width()
        dx = self._pan_start_x - event.x  # Inverted for natural panning
        visible = 1 / self.zoom_level
        shift = (dx / w) * visible

        new_start = self._pan_start_view + shift
        self.view_start = max(0, min(new_start, 1 - visible))
        self.draw_wave()
        self.update_scroll()

    def on_set_loop(self, event):
        if getattr(self, '_resizing', False):
            return
        if self.audio_data is None:
            return
        sample = self.pos_from_wave_x(event.x)
        # Validate: loop point cannot be before start point
        if sample < self.trim_start_sample:
            messagebox.showerror("Error", "Loop point cannot be before start point!")
            return
        # Validate: loop point cannot be after end point
        trim_end = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)
        if sample >= trim_end:
            messagebox.showerror("Error", "Loop point cannot be at or after end point!")
            return
        self.edit_loop_sample = sample
        self.set_loop_entry(sample)
        self.draw_wave()
        self.draw_prog(self.current_position / self.duration)
        t = sample / self.sample_rate
        self.status.config(text=f"Loop point set to {self.fmt_full(t)}")

    def on_prog_click(self, event):
        if getattr(self, '_resizing', False):
            return
        if self.audio_data is None:
            return
        ratio = max(0, min(1, event.x / self.prog_canvas.winfo_width()))
        self.current_position = ratio * self.duration
        self.update_position_only(ratio)

        # If playing, just update play index (no restart)
        if self.is_playing:
            self.play_idx = int(self.current_position * self.sample_rate)
        else:
            # Force canvas update when paused
            self.wave_canvas.update_idletasks()
            self.prog_canvas.update_idletasks()

    def on_prog_drag(self, event):
        self.on_prog_click(event)

    def apply_loop(self):
        if self.audio_data is None or self.sample_rate is None:
            return
        time_sec = self.parse_time(self.loop_entry.get())
        if time_sec is None:
            messagebox.showerror("Error", "Invalid time format. Use 00:00:00.000")
            return
        sample = int(time_sec * self.sample_rate)
        # Validate: loop point cannot be before start point
        if sample < self.trim_start_sample:
            messagebox.showerror("Error", "Loop point cannot be before start point!")
            return
        # Validate: loop point cannot be after end point
        trim_end = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)
        if sample >= trim_end:
            messagebox.showerror("Error", "Loop point cannot be at or after end point!")
            return
        self.edit_loop_sample = max(0, min(sample, len(self.audio_data) - 1))
        # Update entry with normalized format
        self.set_loop_entry(self.edit_loop_sample)
        self.draw_wave()
        self.draw_prog(self.current_position / self.duration)

    def loop_from_pos(self):
        if self.audio_data is None:
            return
        sample = int(self.current_position * self.sample_rate)
        # Validate: loop point cannot be before start point
        if sample < self.trim_start_sample:
            messagebox.showerror("Error", "Loop point cannot be before start point!")
            return
        # Validate: loop point cannot be after end point
        trim_end = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)
        if sample >= trim_end:
            messagebox.showerror("Error", "Loop point cannot be at or after end point!")
            return
        self.edit_loop_sample = sample
        self.set_loop_entry(sample)
        self.draw_wave()
        self.draw_prog(self.current_position / self.duration)

    def apply_trim_start(self):
        """Apply trim start time from entry"""
        if self.audio_data is None or self.sample_rate is None:
            return
        time_sec = self.parse_time(self.trim_start_entry.get())
        if time_sec is None:
            messagebox.showerror("Error", "Invalid time format. Use 00:00:00.000")
            return
        sample = int(time_sec * self.sample_rate)
        # Validate: start must be less than end (if end is set)
        end_sample = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)
        if sample >= end_sample:
            messagebox.showerror("Error", "Start time must be before end time!")
            return
        self.trim_start_sample = max(0, min(sample, len(self.audio_data) - 1))
        self.set_trim_start_entry(self.trim_start_sample)
        # Adjust loop point if it's now before start
        if self.edit_loop_sample < self.trim_start_sample:
            self.edit_loop_sample = self.trim_start_sample
            self.set_loop_entry(self.edit_loop_sample)
            self.status.config(text="Loop point adjusted to match start point")
        self.draw_wave()
        self.draw_prog(self.current_position / self.duration)

    def apply_trim_end(self):
        """Apply trim end time from entry"""
        if self.audio_data is None or self.sample_rate is None:
            return
        time_sec = self.parse_time(self.trim_end_entry.get())
        if time_sec is None:
            messagebox.showerror("Error", "Invalid time format. Use 00:00:00.000")
            return
        sample = int(time_sec * self.sample_rate)
        # Validate: end must be greater than start
        if sample <= self.trim_start_sample:
            messagebox.showerror("Error", "End time must be after start time!")
            return
        self.trim_end_sample = max(1, min(sample, len(self.audio_data)))
        self.set_trim_end_entry(self.trim_end_sample)
        # Adjust loop point if it's now at or after end
        if self.edit_loop_sample >= self.trim_end_sample:
            self.edit_loop_sample = self.trim_start_sample
            self.set_loop_entry(self.edit_loop_sample)
            self.status.config(text="Loop point adjusted to match start point")
        self.draw_wave()
        self.draw_prog(self.current_position / self.duration)

    def trim_start_from_pos(self):
        """Set trim start from current playback position"""
        if self.audio_data is None:
            return
        sample = int(self.current_position * self.sample_rate)
        end_sample = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)
        if sample >= end_sample:
            messagebox.showerror("Error", "Start time must be before end time!")
            return
        self.trim_start_sample = sample
        self.set_trim_start_entry(sample)
        # Adjust loop point if it's now before start
        if self.edit_loop_sample < self.trim_start_sample:
            self.edit_loop_sample = self.trim_start_sample
            self.set_loop_entry(self.edit_loop_sample)
            self.status.config(text="Loop point adjusted to match start point")
        self.draw_wave()
        self.draw_prog(self.current_position / self.duration)

    def trim_end_from_pos(self):
        """Set trim end from current playback position"""
        if self.audio_data is None:
            return
        sample = int(self.current_position * self.sample_rate)
        if sample <= self.trim_start_sample:
            messagebox.showerror("Error", "End time must be after start time!")
            return
        self.trim_end_sample = sample
        self.set_trim_end_entry(sample)
        # Adjust loop point if it's now at or after end
        if self.edit_loop_sample >= self.trim_end_sample:
            self.edit_loop_sample = self.trim_start_sample
            self.set_loop_entry(self.edit_loop_sample)
            self.status.config(text="Loop point adjusted to match start point")
        self.draw_wave()
        self.draw_prog(self.current_position / self.duration)

    def reset_trim_points(self):
        """Reset trim points to default (full audio)"""
        self.trim_start_sample = 0
        self.trim_end_sample = 0  # 0 means end of file
        self.set_trim_start_entry(0)
        if self.audio_data is not None:
            self.set_trim_end_entry(len(self.audio_data))
        else:
            self.trim_end_entry.delete(0, tk.END)
            self.trim_end_entry.insert(0, "00:00:00.000")
        self.draw_wave()
        if self.duration > 0:
            self.draw_prog(self.current_position / self.duration)

    def set_trim_start_entry(self, sample):
        """Set trim start entry with time format"""
        if self.sample_rate:
            t = sample / self.sample_rate
            self.trim_start_entry.delete(0, tk.END)
            self.trim_start_entry.insert(0, self.fmt_full(t))

    def set_trim_end_entry(self, sample):
        """Set trim end entry with time format"""
        if self.sample_rate:
            t = sample / self.sample_rate
            self.trim_end_entry.delete(0, tk.END)
            self.trim_end_entry.insert(0, self.fmt_full(t))

    def set_loop_entry(self, sample):
        """Set loop entry with time format"""
        if self.sample_rate:
            t = sample / self.sample_rate
            self.loop_entry.delete(0, tk.END)
            self.loop_entry.insert(0, self.fmt_full(t))

    def on_space(self, event):
        """Handle space key - toggle play/pause"""
        self.toggle_play()
        return "break"  # Prevent default listbox behavior

    def goto_start(self):
        """Go to trim start point"""
        if self.audio_data is None:
            return
        # Go to trim start (or 0 if not set)
        self.current_position = self.trim_start_sample / self.sample_rate
        if self.is_playing:
            self.play_idx = self.trim_start_sample
        self.update_position_only(self.current_position / self.duration)

    def goto_end(self):
        """Go to trim end point"""
        if self.audio_data is None:
            return
        # Go to just before trim end (or file end if not set)
        trim_end = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)
        end_pos = max(0, (trim_end / self.sample_rate) - 0.1)
        self.current_position = end_pos
        if self.is_playing:
            self.play_idx = int(self.current_position * self.sample_rate)
        self.update_position_only(self.current_position / self.duration)

    def rewind_5s(self):
        """Rewind 5 seconds"""
        if self.audio_data is None:
            return
        self.current_position = max(0, self.current_position - 5)
        if self.is_playing:
            self.play_idx = int(self.current_position * self.sample_rate)
        self.update_position_only(self.current_position / self.duration)

    def forward_5s(self):
        """Forward 5 seconds"""
        if self.audio_data is None:
            return
        self.current_position = min(self.duration, self.current_position + 5)
        if self.is_playing:
            self.play_idx = int(self.current_position * self.sample_rate)
        self.update_position_only(self.current_position / self.duration)

    def stop_and_reset(self):
        """Stop and go to beginning - instant"""
        self.stop_flag = True
        self.is_playing = False
        self.play_btn.config(text="▶")
        self.current_position = 0
        self.play_idx = 0
        self.update_position_only(0)
        # Close stream in background
        if hasattr(self, 'stream') and self.stream:
            threading.Thread(target=self._close_stream, daemon=True).start()

    def toggle_play(self):
        if self.audio_data is None:
            return
        if self.is_playing:
            self.pause()
        else:
            self.play(self.current_position)

    def pause(self):
        """Pause without closing stream - instant"""
        # Save current position FIRST before stopping
        if hasattr(self, 'play_idx') and self.sample_rate:
            saved_pos = self.play_idx / self.sample_rate
            self.current_position = saved_pos

        self.stop_flag = True
        self.user_paused = True  # Mark as user-initiated pause
        self.is_playing = False
        self.play_btn.config(text="▶")

        # Close stream in background to avoid delay
        if hasattr(self, 'stream') and self.stream:
            threading.Thread(target=self._close_stream, daemon=True).start()

    def _close_stream(self):
        """Close stream in background"""
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except:
            pass

    def play(self, start=0):
        self.is_playing = True
        self.play_btn.config(text="⏸")
        self.stop_flag = False
        self.user_paused = False

        start_idx = int(start * self.sample_rate)
        self.play_start_time = time.time()
        self.play_start_pos = start
        self.play_idx = start_idx

        def audio_callback(outdata, frames, time_info, status):
            if self.stop_flag:
                raise sd.CallbackStop()

            # Determine playback boundaries
            play_start = self.trim_start_sample
            play_end = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)

            # Determine loop point (must be >= play_start)
            loop_point = self.edit_loop_sample if self.edit_loop_sample >= play_start else play_start

            end_idx = self.play_idx + frames

            # Check if we've reached or passed the end point
            if self.play_idx >= play_end:
                if self.loop_enabled.get():
                    self.play_idx = loop_point
                    end_idx = self.play_idx + frames
                else:
                    outdata[:, :] = 0
                    raise sd.CallbackStop()

            # Calculate how much we can read before hitting end
            available = play_end - self.play_idx
            read_frames = min(frames, available)

            chunk = self.audio_data[self.play_idx:self.play_idx + read_frames].copy()

            if len(chunk) < frames:
                # Fill first part with remaining audio
                outdata[:len(chunk), :] = chunk * self.volume
                remaining = frames - len(chunk)

                if self.loop_enabled.get():
                    # Seamless loop: fill rest with audio from loop point
                    loop_chunk = self.audio_data[loop_point:loop_point + remaining].copy()
                    if len(loop_chunk) > 0:
                        outdata[len(chunk):len(chunk) + len(loop_chunk), :] = loop_chunk * self.volume
                    if len(loop_chunk) < remaining:
                        outdata[len(chunk) + len(loop_chunk):, :] = 0
                    self.play_idx = loop_point + len(loop_chunk)
                else:
                    outdata[len(chunk):, :] = 0
                    raise sd.CallbackStop()
            else:
                outdata[:, :] = chunk * self.volume
                self.play_idx = self.play_idx + read_frames

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=audio_callback,
            finished_callback=self.on_stream_end
        )
        self.stream.start()

        def update_ui():
            while not self.stop_flag and self.is_playing:
                # Skip scheduling during resize
                if not getattr(self, '_resizing', False):
                    pos = self.play_idx / self.sample_rate
                    self.current_position = pos
                    self.root.after(0, lambda p=pos: self.update_ui_pos(p))
                time.sleep(0.016)  # 60 fps

        self.playback_thread = threading.Thread(target=update_ui, daemon=True)
        self.playback_thread.start()

    def on_stream_end(self):
        self.root.after(0, self.on_end)

    def update_ui_pos(self, pos):
        if self.duration > 0:
            ratio = pos / self.duration
            self.update_position_only(ratio)

    def update_position_only(self, ratio):
        if self.audio_data is None:
            return
        w = self.prog_canvas.winfo_width()
        h = self.prog_canvas.winfo_height()
        if w < 10:
            return

        px = int(ratio * w)

        # Move progress fill
        if self.prog_fill_id:
            self.prog_canvas.coords(self.prog_fill_id, 0, 0, px, h)

        # Move waveform position line
        if hasattr(self, 'wave_start_sample') and self.wave_pos_id:
            total_samples = len(self.audio_data)
            pos_sample = int(ratio * total_samples)
            visible = self.wave_end_sample - self.wave_start_sample
            if visible > 0:
                wx = int((pos_sample - self.wave_start_sample) / visible * self.wave_w)
                self.wave_canvas.coords(self.wave_pos_id, wx, 0, wx, self.wave_h)

        # Update time label
        self.time_label.config(text=f"{self.fmt_full(ratio * self.duration)} / {self.fmt_full(self.duration)}")

    def on_end(self):
        # Never reset position here - user uses stop button for that
        self.is_playing = False
        self.play_btn.config(text="▶")

    def stop(self):
        self.stop_flag = True
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.play_btn.config(text="▶")

    def set_vol(self, val):
        self.volume = float(val) / 100

    def apply_changes(self):
        """Apply all changes: replacement OGG, loop points, and trim"""
        if self.current_file is None:
            messagebox.showwarning("Warning", "No file selected")
            return

        if not hasattr(self, 'current_folder') or not self.current_folder:
            messagebox.showwarning("Warning", "No folder loaded")
            return

        stem = self.current_file.stem
        path = str(self.current_file)
        changes_made = []

        # Stop stream before file operations
        self.stop_flag = True
        self.is_playing = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.play_btn.config(text="▶")

        import shutil
        import subprocess

        # Check if there's a replacement in pre-replace folder
        has_replacement = False
        if hasattr(self, 'temp_folder') and self.temp_folder:
            pre_replace = self.temp_folder / "pre-replace"
            pre_replace_ogg = pre_replace / f"{stem}.ogg"
            has_replacement = pre_replace_ogg.exists()

        # Determine if we need to trim the current file
        need_trim = (self.trim_start_sample > 0 or
                     (self.trim_end_sample > 0 and self.audio_data is not None and
                      self.trim_end_sample < len(self.audio_data)))

        try:
            # Step 1: Handle replacement first (if any)
            if has_replacement:
                # Move replacement from pre-replace to main folder
                shutil.move(str(pre_replace_ogg), path)
                changes_made.append("Replaced OGG")

                # Clear replacement marker
                if stem in self.replaced_audio:
                    del self.replaced_audio[stem]

                # Clean up pre-replace backup
                backup_cache = pre_replace / f"{stem}.npy"
                if backup_cache.exists():
                    backup_cache.unlink()
                if pre_replace.exists() and not any(pre_replace.iterdir()):
                    pre_replace.rmdir()

                self.replace_label.config(text="", foreground="black")

            # Step 2: Apply trim (works on replacement OR current file)
            if need_trim and self.audio_data is not None:
                trim_start = self.trim_start_sample
                trim_end = self.trim_end_sample if self.trim_end_sample > 0 else len(self.audio_data)

                trimmed_data = self.audio_data[trim_start:trim_end]

                # Save trimmed audio via temp WAV + oggenc2
                oggenc_path = Path(__file__).parent / "tool" / "oggenc2.exe"
                if not oggenc_path.exists():
                    messagebox.showerror("Error", "oggenc2.exe not found in tool folder")
                    return

                temp_wav = self.temp_folder / f"_temp_trim_{stem}.wav"
                sf.write(str(temp_wav), trimmed_data, self.sample_rate, format="WAV", subtype="PCM_16")

                result = subprocess.run([
                    str(oggenc_path), "-q", "6", "-o", path, str(temp_wav)
                ], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)

                temp_wav.unlink()

                if result.returncode != 0:
                    messagebox.showerror("Error", f"Trim failed:\n{result.stderr}")
                    return

                # Update internal state with trimmed data
                self.audio_data = trimmed_data
                self.audio_mono = np.mean(trimmed_data, axis=1)
                self.duration = len(trimmed_data) / self.sample_rate

                changes_made.append(f"Trimmed ({trim_start}-{trim_end})")

            # Step 2: Update loop points file
            loop_file_path = os.path.join(self.current_folder, "_loop_points.txt")
            name = self.current_file.stem

            # Adjust loop point relative to trim (if trimmed)
            if need_trim and self.edit_loop_sample >= self.trim_start_sample:
                # Make loop point relative to new trimmed start
                adjusted_loop = self.edit_loop_sample - self.trim_start_sample
                self.loop_points[name] = adjusted_loop
                self.edit_loop_sample = adjusted_loop
            else:
                self.loop_points[name] = self.edit_loop_sample

            # Write loop points file
            with open(loop_file_path, "w") as f:
                f.write("# Loop points for OGG files\n")
                for n, v in sorted(self.loop_points.items()):
                    f.write(f"{n}.ogg: {v}\n")
            changes_made.append("Updated loop points")

            # Step 3: Reset trim points after applying
            self.trim_start_sample = 0
            self.trim_end_sample = 0
            self.set_trim_start_entry(0)
            if self.audio_data is not None:
                self.set_trim_end_entry(len(self.audio_data))

            # Step 4: Update waveform cache
            if hasattr(self, 'temp_folder') and self.temp_folder:
                cache_file = self.temp_folder / f"{stem}.npy"
                self._save_waveform_cache(cache_file)

            # Step 5: Redraw
            self.set_loop_entry(self.edit_loop_sample)
            self.draw_wave()
            self.draw_prog(0)
            self.current_position = 0

            self.status.config(text=f"Applied: {', '.join(changes_made)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply changes: {e}")

    def _save_waveform_cache(self, cache_file):
        """Save waveform cache for current audio"""
        if self.audio_data is None:
            return
        try:
            mono = np.mean(self.audio_data, axis=1) if len(self.audio_data.shape) > 1 else self.audio_data
            w = 800
            total_samples = len(mono)
            samples_per_pixel = max(1, total_samples // w)
            usable = (total_samples // samples_per_pixel) * samples_per_pixel

            if usable > 0:
                reshaped = np.abs(mono[:usable]).reshape(-1, samples_per_pixel)
                peaks = np.max(reshaped, axis=1).astype(np.float32)
            else:
                peaks = np.zeros(w, dtype=np.float32)

            cache_data = {
                'peaks': peaks,
                'duration': self.duration,
                'sample_rate': self.sample_rate
            }
            np.save(cache_file, cache_data, allow_pickle=True)
        except Exception:
            pass  # Skip cache errors silently

    def save_loop_file(self):
        if not hasattr(self, "current_folder") or not self.current_file:
            messagebox.showwarning("Warning", "No folder loaded")
            return

        path = os.path.join(self.current_folder, "_loop_points.txt")
        name = self.current_file.stem

        self.loop_points[name] = self.edit_loop_sample

        try:
            with open(path, "w") as f:
                f.write("# Loop points for OGG files\n")
                for n, v in sorted(self.loop_points.items()):
                    f.write(f"{n}.ogg: {v}\n")
            self.status.config(text=f"Updated: {path}")

            self.listbox.delete(0, tk.END)
            for fl in self.ogg_files:
                display_name = self.get_display_name(fl)
                self.listbox.insert(tk.END, display_name)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def fmt(self, sec):
        return f"{int(sec // 60)}:{int(sec % 60):02d}"

    def fmt_full(self, sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def parse_time(self, time_str):
        """Parse 00:00:00.000 format to seconds"""
        try:
            parts = time_str.strip().split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            else:
                return float(time_str)
        except:
            return None

    def show_about(self):
        """Show about dialog with clickable links"""
        import webbrowser

        about_win = tk.Toplevel(self.root)
        about_win.title("About")
        about_win.geometry("350x220")
        about_win.resizable(False, False)
        about_win.transient(self.root)
        about_win.grab_set()

        # Center the window
        about_win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 175
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 110
        about_win.geometry(f"+{x}+{y}")

        frame = ttk.Frame(about_win, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="AOT2 Audio Editor", font=("", 12, "bold")).pack(pady=(0, 10))
        ttk.Label(frame, text="Made by thunderlol").pack()

        # Reddit link
        reddit_frame = ttk.Frame(frame)
        reddit_frame.pack(pady=5)
        ttk.Label(reddit_frame, text="Reddit: ").pack(side=tk.LEFT)
        reddit_link = ttk.Label(reddit_frame, text="u/_Thunderlol_", foreground="blue", cursor="hand2")
        reddit_link.pack(side=tk.LEFT)
        reddit_link.bind("<Button-1>", lambda e: webbrowser.open("https://reddit.com/u/_Thunderlol_"))

        # Discord link
        discord_frame = ttk.Frame(frame)
        discord_frame.pack(pady=5)
        ttk.Label(discord_frame, text="AOT2 Discord: ").pack(side=tk.LEFT)
        discord_link = ttk.Label(discord_frame, text="Join Server", foreground="blue", cursor="hand2")
        discord_link.pack(side=tk.LEFT)
        discord_link.bind("<Button-1>", lambda e: webbrowser.open("https://discord.com/invite/t8SwB7kkjy"))

        ttk.Button(frame, text="Close", command=about_win.destroy).pack(pady=(15, 0))

    def extract_music(self):
        """Extract music from ktsl2stbin file"""
        # Select input file
        input_file = filedialog.askopenfilename(
            title="Select .ktsl2stbin file",
            filetypes=[("KTSL2STBIN files", "*.ktsl2stbin"), ("All files", "*.*")]
        )
        if not input_file:
            return

        # Select output folder
        output_folder = filedialog.askdirectory(title="Select output folder")
        if not output_folder:
            return

        self.status.config(text="Extracting...")
        self.root.update()

        try:
            # Extract from ktsl2stbin
            kvs_dir = os.path.join(output_folder, "kvs_files")
            ogg_dir = os.path.join(output_folder, "ogg_files")

            # Save KTSR header
            with open(input_file, 'rb') as f:
                header = f.read(96)
            header_path = os.path.join(output_folder, "_header.bin")
            with open(header_path, 'wb') as f:
                f.write(header)

            # Extract KVS files
            with open(input_file, 'rb') as f:
                byte_data = f.read()

            # Find KOVS markers
            kvs_positions = []
            start = 0
            while True:
                pos = byte_data.find(b'KOVS', start)
                if pos == -1:
                    break
                kvs_positions.append(pos)
                start = pos + 1

            # Also check for KTSS markers
            ktss_positions = []
            start = 0
            while True:
                pos = byte_data.find(b'KTSS', start)
                if pos == -1:
                    break
                ktss_positions.append(pos)
                start = pos + 1

            # Determine format
            if kvs_positions:
                positions = kvs_positions
                ext = '.kvs'
                marker = b'KOVS'
            elif ktss_positions:
                positions = ktss_positions
                ext = '.kns'
                marker = b'KTSS'
            else:
                messagebox.showerror("Error", "No audio files found!")
                return

            os.makedirs(kvs_dir, exist_ok=True)
            os.makedirs(ogg_dir, exist_ok=True)

            extracted_files = []
            for i, start_pos in enumerate(positions):
                end_pos = positions[i + 1] if i < len(positions) - 1 else len(byte_data)
                file_data = byte_data[start_pos:end_pos]
                filename = f"{i:03d}{ext}"
                output_path = os.path.join(kvs_dir, filename)
                with open(output_path, 'wb') as f:
                    f.write(file_data)
                extracted_files.append(output_path)

            # Convert KVS to OGG
            loop_points = {}
            ogg_sizes = {}
            padding_data = {}
            trailing_data = {}

            for kvs_file in extracted_files:
                with open(kvs_file, 'rb') as f:
                    signature = f.read(4)
                    if signature not in [b'KOVS', b'KTSS']:
                        continue
                    ogg_size = struct.unpack('<I', f.read(4))[0]
                    loop_point = struct.unpack('<I', f.read(4))[0]
                    padding = f.read(20)
                    encrypted_ogg = f.read(ogg_size)
                    trailing = f.read()

                # Decrypt
                decrypted = bytearray(encrypted_ogg)
                for i in range(min(256, len(decrypted))):
                    decrypted[i] ^= i

                kvs_stem = Path(kvs_file).stem
                ogg_path = os.path.join(ogg_dir, f"{kvs_stem}.ogg")
                with open(ogg_path, 'wb') as f:
                    f.write(bytes(decrypted))

                if loop_point > 0:
                    loop_points[kvs_stem] = loop_point
                ogg_sizes[kvs_stem] = len(decrypted)
                padding_data[kvs_stem] = padding.hex()
                if trailing:
                    trailing_data[kvs_stem] = trailing.hex()

            # Save metadata
            if loop_points:
                with open(os.path.join(ogg_dir, "_loop_points.txt"), 'w') as f:
                    f.write("# Loop points for OGG files\n")
                    for name, point in sorted(loop_points.items()):
                        f.write(f"{name}.ogg: {point}\n")

            with open(os.path.join(ogg_dir, "_ogg_sizes.json"), 'w') as f:
                json.dump(ogg_sizes, f, indent=2, sort_keys=True)

            with open(os.path.join(ogg_dir, "_padding_data.json"), 'w') as f:
                json.dump(padding_data, f, indent=2, sort_keys=True)

            with open(os.path.join(ogg_dir, "_trailing_data.json"), 'w') as f:
                json.dump(trailing_data, f, indent=2, sort_keys=True)

            # Delete KVS files
            import shutil
            shutil.rmtree(kvs_dir)

            messagebox.showinfo("Extraction Complete",
                f"Extracted {len(extracted_files)} audio files to:\n{ogg_dir}\n\n"
                f"Format: {ext[1:].upper()}\n"
                f"Loop points: {len(loop_points)}\n"
                f"Ready for editing!")

            # Load the extracted folder
            self.load_folder_path(ogg_dir)

        except Exception as e:
            messagebox.showerror("Error", f"Extraction failed:\n{e}")

    def load_folder_path(self, folder):
        """Load a folder by path (used after extraction)"""
        files = sorted(Path(folder).glob("*.ogg"))
        self.ogg_files = []
        for f in files:
            if f.name.startswith("_"):
                continue
            stem = f.stem
            if stem in self.skip_ids:
                continue
            self.ogg_files.append(f)

        if not self.ogg_files:
            self.status.config(text="No valid OGG files found")
            return

        self.current_folder = folder
        self.temp_folder = Path(folder) / "temp"
        self.temp_folder.mkdir(exist_ok=True)

        missing_cache = []
        for f in self.ogg_files:
            cache_file = self.temp_folder / f"{f.stem}.npy"
            if not cache_file.exists():
                missing_cache.append(f)

        if missing_cache:
            self.generate_waveform_caches(missing_cache)

        self.load_loops(folder)
        self.folder_label.config(text=folder)

        self.listbox.delete(0, tk.END)
        for f in self.ogg_files:
            display_name = self.get_display_name(f)
            self.listbox.insert(tk.END, display_name)

        self.status.config(text=f"Loaded {len(self.ogg_files)} files")

    def compile_folder(self):
        """Compile current folder back to ktsl2stbin"""
        if not hasattr(self, 'current_folder') or not self.current_folder:
            messagebox.showwarning("Warning", "No folder loaded. Load a folder first.")
            return

        ogg_folder = self.current_folder
        parent_folder = Path(ogg_folder).parent

        # Check for header
        header_path = parent_folder / "_header.bin"
        if not header_path.exists():
            header_path = Path(ogg_folder) / "_header.bin"
        if not header_path.exists():
            messagebox.showerror("Error", "Could not find _header.bin!")
            return

        with open(header_path, 'rb') as f:
            ktsr_header = f.read(96)

        # Load metadata
        ogg_sizes = {}
        sizes_path = Path(ogg_folder) / "_ogg_sizes.json"
        if sizes_path.exists():
            with open(sizes_path, 'r') as f:
                ogg_sizes = json.load(f)

        loop_points = {}
        loop_file = Path(ogg_folder) / "_loop_points.txt"
        if loop_file.exists():
            with open(loop_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ':' in line:
                        filename, point = line.split(':', 1)
                        loop_points[filename.strip()] = int(point.strip())

        padding_data = {}
        padding_path = Path(ogg_folder) / "_padding_data.json"
        if padding_path.exists():
            with open(padding_path, 'r') as f:
                padding_data = json.load(f)

        trailing_data = {}
        trailing_path = Path(ogg_folder) / "_trailing_data.json"
        if trailing_path.exists():
            with open(trailing_path, 'r') as f:
                trailing_data = json.load(f)

        # Select output file
        output_file = filedialog.asksaveasfilename(
            title="Save compiled file",
            defaultextension=".ktsl2stbin",
            filetypes=[("KTSL2STBIN", "*.ktsl2stbin")],
            initialfile="BGM_RECOMPILED.ktsl2stbin"
        )
        if not output_file:
            return

        self.status.config(text="Compiling...")
        self.root.update()

        # Check for vorbis tools
        import subprocess
        import tempfile
        tool_dir = Path(__file__).parent / "tool"
        oggenc_path = tool_dir / "oggenc2.exe"  # Use oggenc2 for better quality
        oggdec_path = tool_dir / "oggdec.exe"
        has_vorbis = oggenc_path.exists() and oggdec_path.exists()

        try:
            ogg_files = sorted(Path(ogg_folder).glob('*.ogg'))
            ogg_files = [f for f in ogg_files if not f.name.startswith('_')]

            temp_dir = tempfile.mkdtemp(prefix="kvs_compile_")
            stats = {'exact': 0, 'compressed': 0, 'padded': 0}

            with open(output_file, 'wb') as out:
                out.write(ktsr_header)

                for idx, ogg_file in enumerate(ogg_files):
                    filename = ogg_file.stem
                    self.status.config(text=f"Compiling {idx+1}/{len(ogg_files)}: {filename}")
                    self.root.update()

                    with open(ogg_file, 'rb') as f:
                        ogg_data = f.read()

                    target_size = ogg_sizes.get(filename, len(ogg_data))
                    current_ogg = str(ogg_file)
                    was_compressed = False

                    # If file too large, compress it using smart quality calculation
                    if len(ogg_data) > target_size:
                        if not has_vorbis:
                            messagebox.showerror("Error",
                                f"File {filename}.ogg is too large!\n"
                                f"Current: {len(ogg_data)}, Target: {target_size}\n\n"
                                f"Need oggenc.exe + oggdec.exe for auto-compression.")
                            return

                        # Calculate compression ratio and estimate quality
                        # Target 99% of target_size to ensure we fit
                        target_99 = int(target_size * 0.99)
                        ratio = target_99 / len(ogg_data)

                        # Map ratio to quality (0-10 scale)
                        # ratio 0.1 -> quality 0, ratio 0.9 -> quality 8
                        # Lower ratio = more compression needed = lower quality
                        estimated_quality = max(0, min(9, int(ratio * 10) - 1))

                        temp_wav = os.path.join(temp_dir, f"temp_{filename}.wav")

                        # Decode to WAV once
                        subprocess.run([str(oggdec_path), '-o', temp_wav, current_ogg],
                            capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)

                        compressed = False
                        # Try estimated quality first, then go lower if needed (max 2-3 tries)
                        for quality in range(estimated_quality, -1, -1):
                            temp_ogg = os.path.join(temp_dir, f"temp_{filename}_q{quality}.ogg")

                            # Re-encode at calculated quality
                            subprocess.run([str(oggenc_path), '-q', str(quality), '-o', temp_ogg, temp_wav],
                                capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)

                            if os.path.exists(temp_ogg):
                                new_size = os.path.getsize(temp_ogg)
                                if new_size <= target_size:
                                    with open(temp_ogg, 'rb') as f:
                                        ogg_data = f.read()
                                    compressed = True
                                    was_compressed = True
                                    stats['compressed'] += 1
                                    # Clean up temp ogg
                                    os.remove(temp_ogg)
                                    break
                                # Clean up temp ogg if too large
                                os.remove(temp_ogg)

                        # Clean up temp wav
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)

                        if not compressed:
                            messagebox.showerror("Error",
                                f"Could not compress {filename}.ogg small enough!\n"
                                f"Current: {len(ogg_data)}, Target: {target_size}")
                            return

                    # Encrypt
                    encrypted = bytearray(ogg_data)
                    for i in range(min(256, len(encrypted))):
                        encrypted[i] ^= i

                    # Get metadata
                    loop_point = loop_points.get(ogg_file.name, 0)
                    file_padding = bytes.fromhex(padding_data.get(filename, '00' * 20))
                    file_trailing = bytes.fromhex(trailing_data.get(filename, '')) if filename in trailing_data else b''

                    # Build KVS
                    kvs_data = bytearray()
                    kvs_data.extend(b'KOVS')
                    kvs_data.extend(struct.pack('<I', len(encrypted)))
                    kvs_data.extend(struct.pack('<I', loop_point))
                    kvs_data.extend(file_padding if len(file_padding) == 20 else b'\x00' * 20)
                    kvs_data.extend(encrypted)

                    # Track stats and pad if needed
                    if len(ogg_data) < target_size:
                        kvs_data.extend(b'\x00' * (target_size - len(ogg_data)))
                        if not was_compressed:
                            stats['padded'] += 1
                    elif len(ogg_data) == target_size and not was_compressed:
                        stats['exact'] += 1

                    kvs_data.extend(file_trailing)
                    out.write(kvs_data)

            # Cleanup temp dir
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            final_size = os.path.getsize(output_file)
            messagebox.showinfo("Compilation Complete",
                f"Compiled {len(ogg_files)} files\n\n"
                f"Output: {output_file}\n"
                f"Size: {final_size:,} bytes ({final_size / (1024*1024):.2f} MB)\n\n"
                f"Stats:\n"
                f"- Exact matches: {stats['exact']}\n"
                f"- Compressed: {stats['compressed']}\n"
                f"- Padded: {stats['padded']}")
            self.status.config(text="Nothing to do")

        except Exception as e:
            messagebox.showerror("Error", f"Compilation failed:\n{e}")


def main():
    root = tk.Tk()
    app = AudioPlayer(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
