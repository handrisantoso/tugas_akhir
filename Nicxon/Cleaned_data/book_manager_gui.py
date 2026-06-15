"""
book_manager_gui.py - GUI for managing book metadata:
cover images, TOC, description, and subjects.
"""
import json, os, re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(SCRIPT_DIR, "books_english_only.json")
COVER_DIR = os.path.join(SCRIPT_DIR, "cover_images")
os.makedirs(COVER_DIR, exist_ok=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.strip().replace(" ", "_")[:120]

def get_display_label(book: dict) -> str:
    title = book.get("title", "Untitled")
    authors = book.get("authors", [])
    author_str = authors[0].get("name", "") if authors else ""
    icons = []
    if book.get("has_cover_image"): icons.append("📷")
    if book.get("has_toc"): icons.append("📑")
    if book.get("has_desc"): icons.append("📝")
    label = title
    if author_str: label += f"  —  {author_str}"
    if icons: label += f"  [{' '.join(icons)}]"
    return label

class BookManagerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("📚 Book Manager")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#1e1e2e")
        self.books, self.filtered_indices = [], []
        self.selected_index = None
        self.cover_preview_image = None
        self._load_json()
        self._build_ui()
        self._populate_list()

    def _load_json(self):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            self.books = json.load(f)

    def _save_json(self):
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(self.books, f, indent=2, ensure_ascii=False)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        bg, surface, accent = "#1e1e2e", "#2a2a3d", "#7c3aed"
        text_color, muted = "#e2e2f0", "#8888aa"
        style.configure("TFrame", background=bg)
        style.configure("Surface.TFrame", background=surface)
        style.configure("TLabel", background=bg, foreground=text_color, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=bg, foreground="#fff", font=("Segoe UI", 14, "bold"))
        style.configure("Sub.TLabel", background=surface, foreground=muted, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=surface, foreground=text_color, font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background=accent, foreground="#fff", font=("Segoe UI", 10, "bold"), padding=(12,6))
        style.map("Accent.TButton", background=[("active", "#6d28d9")])
        style.configure("TButton", font=("Segoe UI", 10), padding=(10,5))

        # Left panel
        left = ttk.Frame(self.root)
        left.pack(side="left", fill="both", expand=False, padx=(12,6), pady=12)
        ttk.Label(left, text="📚 Book Library", style="Header.TLabel").pack(anchor="w", pady=(0,8))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._filter_list())
        ttk.Entry(left, textvariable=self.search_var, font=("Segoe UI", 10)).pack(fill="x", pady=(0,8))

        ff = ttk.Frame(left)
        ff.pack(fill="x", pady=(0,6))
        self.filter_var = tk.StringVar(value="all")
        for t, v in [("All","all"),("No Cover","no_cover"),("No TOC","no_toc"),("No Desc","no_desc"),("Has Cover","has_cover"),("Has TOC","has_toc"),("Has Desc","has_desc")]:
            ttk.Radiobutton(ff, text=t, variable=self.filter_var, value=v, command=self._filter_list).pack(side="left", padx=(0,4))

        lf = ttk.Frame(left, style="Surface.TFrame")
        lf.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(lf, bg="#2a2a3d", fg="#e2e2f0", selectbackground="#7c3aed",
            selectforeground="#fff", font=("Segoe UI", 10), borderwidth=0, highlightthickness=0, activestyle="none")
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.count_label = ttk.Label(left, text="", style="Sub.TLabel")
        self.count_label.pack(anchor="w", pady=(4,0))

        # Right panel with scrollable canvas
        right = ttk.Frame(self.root)
        right.pack(side="right", fill="both", expand=True, padx=(6,12), pady=12)
        ttk.Label(right, text="📖 Book Details", style="Header.TLabel").pack(anchor="w", pady=(0,10))

        canvas_frame = ttk.Frame(right, style="Surface.TFrame")
        canvas_frame.pack(fill="both", expand=True)
        self.detail_canvas = tk.Canvas(canvas_frame, bg=surface, highlightthickness=0)
        detail_sb = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.detail_canvas.yview)
        self.detail_canvas.configure(yscrollcommand=detail_sb.set)
        detail_sb.pack(side="right", fill="y")
        self.detail_canvas.pack(side="left", fill="both", expand=True)
        self.detail_inner = ttk.Frame(self.detail_canvas, style="Surface.TFrame", padding=16)
        self.detail_window = self.detail_canvas.create_window((0,0), window=self.detail_inner, anchor="nw")
        self.detail_inner.bind("<Configure>", lambda e: self.detail_canvas.configure(scrollregion=self.detail_canvas.bbox("all")))
        self.detail_canvas.bind("<Configure>", lambda e: self.detail_canvas.itemconfig(self.detail_window, width=e.width))

        df = self.detail_inner

        # Title & meta
        self.title_var = tk.StringVar(value="Select a book from the list")
        ttk.Label(df, textvariable=self.title_var, font=("Segoe UI",13,"bold"), background=surface, foreground="#fff", wraplength=600).pack(anchor="w", pady=(0,4))
        self.meta_var = tk.StringVar()
        ttk.Label(df, textvariable=self.meta_var, style="Sub.TLabel", wraplength=600).pack(anchor="w", pady=(0,12))

        # Status row
        sf = ttk.Frame(df, style="Surface.TFrame")
        sf.pack(fill="x", pady=(0,12))
        self.cover_status_var = tk.StringVar()
        self.toc_status_var = tk.StringVar()
        self.desc_status_var = tk.StringVar()
        ttk.Label(sf, textvariable=self.cover_status_var, style="Status.TLabel").pack(side="left", padx=(0,16))
        ttk.Label(sf, textvariable=self.toc_status_var, style="Status.TLabel").pack(side="left", padx=(0,16))
        ttk.Label(sf, textvariable=self.desc_status_var, style="Status.TLabel").pack(side="left")

        # Cover image section
        cs = ttk.LabelFrame(df, text="  Cover Image  ", padding=10)
        cs.pack(fill="x", pady=(0,10))
        self.cover_canvas = tk.Canvas(cs, width=160, height=220, bg="#1e1e2e", highlightthickness=1, highlightbackground="#444466")
        self.cover_canvas.pack(side="left", padx=(0,12))
        cb = ttk.Frame(cs)
        cb.pack(side="left", fill="y")
        self.upload_btn = ttk.Button(cb, text="📁 Upload Cover Image", style="Accent.TButton", command=self._upload_cover)
        self.upload_btn.pack(anchor="w", pady=(0,6))
        self.remove_cover_btn = ttk.Button(cb, text="🗑 Remove Cover", command=self._remove_cover)
        self.remove_cover_btn.pack(anchor="w")
        self.cover_path_var = tk.StringVar(value="No cover image")
        ttk.Label(cb, textvariable=self.cover_path_var, style="Sub.TLabel", wraplength=350).pack(anchor="w", pady=(8,0))

        # Description section
        ds = ttk.LabelFrame(df, text="  Description  ", padding=10)
        ds.pack(fill="x", pady=(0,10))
        self.desc_text = scrolledtext.ScrolledText(ds, height=4, font=("Consolas",10), bg="#1e1e2e", fg="#e2e2f0", insertbackground="#fff", borderwidth=0, wrap="word")
        self.desc_text.pack(fill="x", pady=(0,8))
        db = ttk.Frame(ds)
        db.pack(fill="x")
        self.save_desc_btn = ttk.Button(db, text="💾 Save Description", style="Accent.TButton", command=self._save_desc)
        self.save_desc_btn.pack(side="left", padx=(0,8))
        self.clear_desc_btn = ttk.Button(db, text="🗑 Clear Description", command=self._clear_desc)
        self.clear_desc_btn.pack(side="left")

        # Subjects section
        ss = ttk.LabelFrame(df, text="  Subjects  ", padding=10)
        ss.pack(fill="x", pady=(0,10))
        ttk.Label(ss, text="One subject per line", style="Sub.TLabel").pack(anchor="w", pady=(0,4))
        self.subjects_text = scrolledtext.ScrolledText(ss, height=4, font=("Consolas",10), bg="#1e1e2e", fg="#e2e2f0", insertbackground="#fff", borderwidth=0, wrap="word")
        self.subjects_text.pack(fill="x", pady=(0,8))
        sub_btns = ttk.Frame(ss)
        sub_btns.pack(fill="x")
        self.save_subj_btn = ttk.Button(sub_btns, text="💾 Save Subjects", style="Accent.TButton", command=self._save_subjects)
        self.save_subj_btn.pack(side="left", padx=(0,8))
        self.clear_subj_btn = ttk.Button(sub_btns, text="🗑 Clear Subjects", command=self._clear_subjects)
        self.clear_subj_btn.pack(side="left")

        # TOC section
        ts = ttk.LabelFrame(df, text="  Table of Contents  ", padding=10)
        ts.pack(fill="x", pady=(0,10))
        ttk.Label(ts, text="One chapter/section per line", style="Sub.TLabel").pack(anchor="w", pady=(0,4))
        self.toc_text = scrolledtext.ScrolledText(ts, height=6, font=("Consolas",10), bg="#1e1e2e", fg="#e2e2f0", insertbackground="#fff", borderwidth=0, wrap="word")
        self.toc_text.pack(fill="x", pady=(0,8))
        tb = ttk.Frame(ts)
        tb.pack(fill="x")
        self.save_toc_btn = ttk.Button(tb, text="💾 Save TOC", style="Accent.TButton", command=self._save_toc)
        self.save_toc_btn.pack(side="left", padx=(0,8))
        self.clear_toc_btn = ttk.Button(tb, text="🗑 Clear TOC", command=self._clear_toc)
        self.clear_toc_btn.pack(side="left")

        self._set_controls_state("disabled")

    # ── List ───────────────────────────────────────────────────────────
    def _populate_list(self): self._filter_list()

    def _filter_list(self):
        self.listbox.delete(0, "end")
        q = self.search_var.get().lower().strip()
        f = self.filter_var.get()
        self.filtered_indices = []
        for i, b in enumerate(self.books):
            if f == "no_cover" and b.get("has_cover_image"): continue
            if f == "no_toc" and b.get("has_toc"): continue
            if f == "no_desc" and b.get("has_desc"): continue
            if f == "has_cover" and not b.get("has_cover_image"): continue
            if f == "has_toc" and not b.get("has_toc"): continue
            if f == "has_desc" and not b.get("has_desc"): continue
            label = get_display_label(b)
            if q and q not in label.lower(): continue
            self.filtered_indices.append(i)
            self.listbox.insert("end", label)
        self.count_label.configure(text=f"Showing {len(self.filtered_indices)} of {len(self.books)} books")

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if not sel: return
        self.selected_index = self.filtered_indices[sel[0]]
        self._show_book_details(self.books[self.selected_index])
        self._set_controls_state("normal")

    # ── Details ────────────────────────────────────────────────────────
    def _show_book_details(self, book):
        title = book.get("title", "Untitled")
        sub = book.get("subtitle", "")
        self.title_var.set(f"{title} — {sub}" if sub else title)

        authors = ", ".join(a.get("name","") for a in book.get("authors",[]))
        pub = ", ".join(book.get("publishers",[]))
        date = book.get("publish_date","")
        pages = book.get("number_of_pages","")
        parts = []
        if authors: parts.append(f"Author: {authors}")
        if pub: parts.append(f"Publisher: {pub}")
        if date: parts.append(f"Published: {date}")
        if pages: parts.append(f"Pages: {pages}")
        self.meta_var.set("  |  ".join(parts) or "No metadata")

        hc, ht, hd = book.get("has_cover_image",False), book.get("has_toc",False), book.get("has_desc",False)
        self.cover_status_var.set(f"📷 Cover: {'✅' if hc else '❌'}")
        self.toc_status_var.set(f"📑 TOC: {'✅' if ht else '❌'}")
        self.desc_status_var.set(f"📝 Desc: {'✅' if hd else '❌'}")

        # Cover preview
        self.cover_canvas.delete("all")
        self.cover_preview_image = None
        cp = book.get("cover_image_path","")
        if cp and os.path.isfile(cp):
            try:
                img = Image.open(cp); img.thumbnail((160,220), Image.LANCZOS)
                self.cover_preview_image = ImageTk.PhotoImage(img)
                self.cover_canvas.create_image(80,110,image=self.cover_preview_image)
            except Exception:
                self.cover_canvas.create_text(80,110,text="Error",fill="#888")
            self.cover_path_var.set(cp)
        else:
            self.cover_canvas.create_text(80,110,text="No cover\nimage",fill="#666",font=("Segoe UI",10))
            self.cover_path_var.set("No cover image")

        # Description
        self.desc_text.delete("1.0","end")
        desc = book.get("description","")
        if isinstance(desc, dict): desc = desc.get("value","")
        if desc: self.desc_text.insert("1.0", desc)

        # Subjects
        self.subjects_text.delete("1.0","end")
        subjs = book.get("subjects",[])
        if subjs: self.subjects_text.insert("1.0", "\n".join(subjs))

        # TOC
        self.toc_text.delete("1.0","end")
        toc = book.get("table_of_contents")
        if toc:
            if isinstance(toc, list):
                lines = []
                for item in toc:
                    if isinstance(item, dict):
                        lines.append("  "*item.get("level",0) + item.get("title",""))
                    else: lines.append(str(item))
                self.toc_text.insert("1.0", "\n".join(lines))
            elif isinstance(toc, str):
                self.toc_text.insert("1.0", toc)

    def _set_controls_state(self, state):
        s = state if state == "normal" else "disabled"
        for w in [self.upload_btn, self.remove_cover_btn, self.save_toc_btn, self.clear_toc_btn,
                  self.save_desc_btn, self.clear_desc_btn, self.save_subj_btn, self.clear_subj_btn]:
            w.configure(state=s)
        for t in [self.toc_text, self.desc_text, self.subjects_text]:
            t.configure(state=s)

    def _refresh_current_list_item(self):
        sel = self.listbox.curselection()
        if not sel: return
        idx = sel[0]
        book = self.books[self.filtered_indices[idx]]
        self.listbox.delete(idx)
        self.listbox.insert(idx, get_display_label(book))
        self.listbox.selection_set(idx)

    # ── Cover ──────────────────────────────────────────────────────────
    def _upload_cover(self):
        if self.selected_index is None: return
        fp = filedialog.askopenfilename(title="Select Cover Image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.webp *.bmp *.gif"),("All","*.*")])
        if not fp: return
        book = self.books[self.selected_index]
        safe = sanitize_filename(book.get("title","untitled"))
        dest = os.path.join(COVER_DIR, f"{safe}.png")
        c = 1
        while os.path.exists(dest):
            dest = os.path.join(COVER_DIR, f"{safe}_{c}.png"); c += 1
        try:
            Image.open(fp).save(dest, "PNG")
            book["cover_image_path"] = dest
            book["has_cover_image"] = True
            self._save_json(); self._show_book_details(book); self._refresh_current_list_item()
            messagebox.showinfo("Success", f"Cover saved:\n{dest}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _remove_cover(self):
        if self.selected_index is None: return
        book = self.books[self.selected_index]
        if not book.get("cover_image_path"):
            messagebox.showinfo("Info","No cover to remove."); return
        if not messagebox.askyesno("Confirm","Remove cover image?"): return
        cp = book.get("cover_image_path","")
        if cp and os.path.isfile(cp):
            try: os.remove(cp)
            except OSError: pass
        book.pop("cover_image_path", None)
        book["has_cover_image"] = False
        self._save_json(); self._show_book_details(book); self._refresh_current_list_item()

    # ── Description ────────────────────────────────────────────────────
    def _save_desc(self):
        if self.selected_index is None: return
        book = self.books[self.selected_index]
        content = self.desc_text.get("1.0","end").strip()
        if not content:
            messagebox.showwarning("Warning","Description is empty. Use Clear to remove."); return
        book["description"] = content
        book["has_desc"] = True
        self._save_json(); self._show_book_details(book); self._refresh_current_list_item()
        messagebox.showinfo("Success","Description saved!")

    def _clear_desc(self):
        if self.selected_index is None: return
        book = self.books[self.selected_index]
        if not book.get("description"):
            messagebox.showinfo("Info","No description to clear."); return
        if not messagebox.askyesno("Confirm","Remove description?"): return
        book.pop("description", None)
        book["has_desc"] = False
        self._save_json(); self._show_book_details(book); self._refresh_current_list_item()

    # ── Subjects ───────────────────────────────────────────────────────
    def _save_subjects(self):
        if self.selected_index is None: return
        book = self.books[self.selected_index]
        raw = self.subjects_text.get("1.0","end").strip()
        subjects = [s.strip() for s in raw.split("\n") if s.strip()] if raw else []
        book["subjects"] = subjects
        self._save_json(); self._show_book_details(book); self._refresh_current_list_item()
        messagebox.showinfo("Success", f"Saved {len(subjects)} subject(s)!")

    def _clear_subjects(self):
        if self.selected_index is None: return
        book = self.books[self.selected_index]
        if not book.get("subjects"):
            messagebox.showinfo("Info","No subjects to clear."); return
        if not messagebox.askyesno("Confirm","Clear all subjects?"): return
        book["subjects"] = []
        self._save_json(); self._show_book_details(book); self._refresh_current_list_item()

    # ── TOC ────────────────────────────────────────────────────────────
    def _save_toc(self):
        if self.selected_index is None: return
        book = self.books[self.selected_index]
        content = self.toc_text.get("1.0","end").strip()
        if not content:
            messagebox.showwarning("Warning","TOC is empty. Use Clear to remove."); return
        book["table_of_contents"] = content
        book["has_toc"] = True
        self._save_json(); self._show_book_details(book); self._refresh_current_list_item()
        messagebox.showinfo("Success","Table of contents saved!")

    def _clear_toc(self):
        if self.selected_index is None: return
        book = self.books[self.selected_index]
        if not book.get("table_of_contents"):
            messagebox.showinfo("Info","No TOC to clear."); return
        if not messagebox.askyesno("Confirm","Remove table of contents?"): return
        book.pop("table_of_contents", None)
        book["has_toc"] = False
        self._save_json(); self._show_book_details(book); self._refresh_current_list_item()

if __name__ == "__main__":
    root = tk.Tk()
    BookManagerApp(root)
    root.mainloop()
