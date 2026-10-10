"""批量把 AlphaFold full_data JSON 中的 PAE 矩阵绘制为 600 DPI PNG。"""

from __future__ import annotations

import json
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "缺少依赖",
        "需要安装 numpy 和 matplotlib。\n\n请在命令行运行：\n"
        "py -m pip install numpy matplotlib\n\n"
        f"详细信息：{exc}",
    )
    raise SystemExit(1) from exc


DPI = 600


def chain_layout(chain_ids: list, matrix_size: int):
    """从连续的 token_chain_ids 得到链标签、中心位置和分界位置。"""
    if len(chain_ids) != matrix_size or not chain_ids:
        return [], [], []

    labels, centers, boundaries = [], [], []
    start = 0
    for index in range(1, len(chain_ids) + 1):
        if index == len(chain_ids) or chain_ids[index] != chain_ids[start]:
            labels.append(str(chain_ids[start]))
            centers.append((start + index - 1) / 2)
            if index < len(chain_ids):
                boundaries.append(index)
            start = index
    return labels, centers, boundaries


def extract_pae(data):
    """兼容 AF3 字典格式以及部分程序导出的列表包装格式。"""
    if isinstance(data, dict) and "pae" in data:
        return data["pae"], data.get("token_chain_ids", [])

    if isinstance(data, list) and data and isinstance(data[0], dict):
        record = data[0]
        for key in ("pae", "predicted_aligned_error"):
            if key in record:
                return record[key], record.get("token_chain_ids", [])

    if isinstance(data, dict) and "predicted_aligned_error" in data:
        return data["predicted_aligned_error"], data.get("token_chain_ids", [])

    raise ValueError("JSON 中未找到 'pae' 或 'predicted_aligned_error' 字段")


def make_plot(json_path: Path, output_dir: Path) -> Path:
    with json_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    raw_pae, chain_ids = extract_pae(data)
    pae = np.asarray(raw_pae, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE 应为方阵，当前形状为 {pae.shape}")
    if not np.isfinite(pae).all():
        raise ValueError("PAE 矩阵包含 NaN 或无穷值")

    labels, centers, boundaries = chain_layout(chain_ids, pae.shape[0])

    fig, ax = plt.subplots(figsize=(7, 6), dpi=DPI)
    image = ax.imshow(
        pae,
        cmap="Greens_r",
        vmin=0,
        vmax=30,
        origin="upper",
        interpolation="nearest",
    )

    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="black", linewidth=0.8)
        ax.axvline(boundary - 0.5, color="black", linewidth=0.8)

    if centers:
        ax.set_xticks(centers, labels)
        ax.set_yticks(centers, labels)
        ax.tick_params(axis="both", which="major", length=0)
    else:
        ax.set_xlabel("Scored residue / token")
        ax.set_ylabel("Aligned residue / token")

    if centers:
        ax.set_xlabel("Scored residue / token")
        ax.set_ylabel("Aligned residue / token")
    ax.set_title("Predicted Aligned Error (PAE)")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Expected position error (Å)")
    colorbar.set_ticks([0, 5, 10, 15, 20, 25, 30])

    # 按 JSON 主文件名命名，例如 sample.json -> sample.png。
    output_path = output_dir / f"{json_path.stem}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


class PAEApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PAE 批量绘图工具（600 DPI）")
        self.geometry("780x570")
        self.minsize(680, 480)
        self.files: list[Path] = []
        self.output_dir = tk.StringVar(value="")
        self.use_source_dir = tk.BooleanVar(value=True)
        self.events: queue.Queue = queue.Queue()
        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="PAE 批量绘图工具",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="选择一个或多个 AlphaFold JSON，批量输出同名的 600 DPI PNG。",
        ).pack(anchor="w", pady=(3, 12))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="添加 JSON…", command=self.add_files).pack(side="left")
        ttk.Button(buttons, text="移除选中", command=self.remove_selected).pack(
            side="left", padx=8
        )
        ttk.Button(buttons, text="清空", command=self.clear_files).pack(side="left")

        list_frame = ttk.Frame(outer)
        list_frame.pack(fill="both", expand=True, pady=10)
        self.file_list = tk.Listbox(list_frame, selectmode="extended")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scroll.set)
        self.file_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        output_box = ttk.LabelFrame(outer, text="输出位置", padding=10)
        output_box.pack(fill="x")
        ttk.Checkbutton(
            output_box,
            text="输出到每个 JSON 所在文件夹",
            variable=self.use_source_dir,
            command=self._toggle_output_controls,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        self.output_entry = ttk.Entry(output_box, textvariable=self.output_dir, state="disabled")
        self.output_entry.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.output_button = ttk.Button(
            output_box, text="选择文件夹…", command=self.choose_output, state="disabled"
        )
        self.output_button.grid(row=1, column=1, padx=(8, 0), pady=(8, 0))
        output_box.columnconfigure(0, weight=1)

        self.progress = ttk.Progressbar(outer, mode="determinate")
        self.progress.pack(fill="x", pady=(12, 5))
        self.status = ttk.Label(outer, text="尚未选择文件")
        self.status.pack(anchor="w")

        self.run_button = ttk.Button(outer, text="开始生成 PAE 图", command=self.start)
        self.run_button.pack(anchor="e", pady=(10, 0))

    def add_files(self):
        selected = filedialog.askopenfilenames(
            title="选择 AlphaFold JSON 文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        known = {str(path).lower() for path in self.files}
        for name in selected:
            path = Path(name)
            if str(path).lower() not in known:
                self.files.append(path)
                self.file_list.insert("end", str(path))
                known.add(str(path).lower())
        self.status.config(text=f"已选择 {len(self.files)} 个 JSON 文件")

    def remove_selected(self):
        for index in reversed(self.file_list.curselection()):
            self.file_list.delete(index)
            del self.files[index]
        self.status.config(text=f"已选择 {len(self.files)} 个 JSON 文件")

    def clear_files(self):
        self.files.clear()
        self.file_list.delete(0, "end")
        self.status.config(text="尚未选择文件")

    def _toggle_output_controls(self):
        state = "disabled" if self.use_source_dir.get() else "normal"
        self.output_entry.config(state=state)
        self.output_button.config(state=state)

    def choose_output(self):
        selected = filedialog.askdirectory(title="选择 PNG 输出文件夹")
        if selected:
            self.output_dir.set(selected)

    def start(self):
        if not self.files:
            messagebox.showwarning("未选择文件", "请先添加至少一个 JSON 文件。")
            return
        if not self.use_source_dir.get() and not self.output_dir.get().strip():
            messagebox.showwarning("未选择输出目录", "请选择 PNG 输出文件夹。")
            return

        self.run_button.config(state="disabled")
        self.progress.configure(maximum=len(self.files), value=0)
        self.status.config(text="正在生成……")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        successes, failures = [], []
        for number, json_path in enumerate(self.files, start=1):
            try:
                target_dir = (
                    json_path.parent
                    if self.use_source_dir.get()
                    else Path(self.output_dir.get().strip())
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                output = make_plot(json_path, target_dir)
                successes.append(output)
                detail = f"已完成：{json_path.name}"
            except Exception as exc:
                failures.append((json_path, str(exc)))
                detail = f"失败：{json_path.name}"
            self.events.put(("progress", number, detail))
        self.events.put(("done", successes, failures))

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, value, detail = event
                    self.progress.configure(value=value)
                    self.status.config(text=detail)
                elif event[0] == "done":
                    _, successes, failures = event
                    self.run_button.config(state="normal")
                    self._show_result(successes, failures)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _show_result(self, successes, failures):
        self.status.config(
            text=f"处理完成：成功 {len(successes)}，失败 {len(failures)}"
        )
        text = f"成功生成 {len(successes)} 张 PNG。"
        if failures:
            details = "\n".join(f"• {path.name}: {error}" for path, error in failures[:12])
            if len(failures) > 12:
                details += f"\n……另有 {len(failures) - 12} 个错误"
            messagebox.showwarning("处理完成（存在失败）", f"{text}\n\n{details}")
        else:
            messagebox.showinfo("处理完成", text)


if __name__ == "__main__":
    try:
        PAEApp().mainloop()
    except Exception:
        messagebox.showerror("程序错误", traceback.format_exc())

