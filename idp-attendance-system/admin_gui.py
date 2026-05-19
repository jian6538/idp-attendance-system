"""
admin_gui.py
============
Tkinter admin dashboard — three tabs:

  Tab 1 · Timetable   — add / edit / delete course timeslots
  Tab 2 · Students    — manage enrolled students per course
  Tab 3 · Attendance  — browse logs, filter by date/course, export CSV

Run:
    python admin_gui.py
"""

from __future__ import annotations

import csv
import os
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk, filedialog
from typing import List, Optional

from schedule_manager import (
    DAY_MAP,
    CourseSlot,
    get_students_path,
    list_all_courses,
    load_enrolled_students,
    load_timetable,
    save_enrolled_students,
    save_timetable,
)

HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "attendance_logs")

DAYS_ORDERED = ["Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_time(s: str) -> bool:
    try:
        h, m = s.strip().split(":")
        return 0 <= int(h) <= 23 and 0 <= int(m) <= 59
    except Exception:
        return False


def _slot_to_row(s: CourseSlot):
    days_rev = {v: k.capitalize() for k, v in DAY_MAP.items()}
    return (
        s.course_code, s.course_name,
        days_rev[s.day],
        s.start_time.strftime("%H:%M"),
        s.end_time.strftime("%H:%M"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Timetable
# ──────────────────────────────────────────────────────────────────────────────

class TimetableTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build()
        self._refresh()

    def _build(self):
        # ── TreeView ──────────────────────────────────────────────────────
        cols = ("code", "name", "day", "start", "end")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=14)
        for col, hdr, w in zip(cols,
                               ("Course Code", "Course Name", "Day", "Start", "End"),
                               (100, 250, 100, 80, 80)):
            self.tree.heading(col, text=hdr)
            self.tree.column(col, width=w, anchor="center")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        vsb.grid(row=0, column=2, sticky="ns", pady=10)

        # ── Form ──────────────────────────────────────────────────────────
        form = ttk.LabelFrame(self, text="Add / Edit Slot")
        form.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        labels = ["Course Code", "Course Name", "Day", "Start (HH:MM)", "End (HH:MM)"]
        self._vars = [tk.StringVar() for _ in labels]
        for i, (lbl, var) in enumerate(zip(labels, self._vars)):
            ttk.Label(form, text=lbl).grid(row=0, column=i*2, padx=6, pady=6, sticky="e")
            if lbl == "Day":
                cb = ttk.Combobox(form, textvariable=var, values=DAYS_ORDERED,
                                  width=11, state="readonly")
                cb.grid(row=0, column=i*2+1, padx=4)
                cb.set("Monday")
            else:
                ttk.Entry(form, textvariable=var, width=18).grid(row=0, column=i*2+1, padx=4)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=6)
        ttk.Button(btn_frame, text="➕ Add",    command=self._add).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="✏ Load Selected", command=self._load_sel).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="💾 Save Edit",    command=self._save_edit).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="🗑 Delete",  command=self._delete).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="🔄 Refresh", command=self._refresh).pack(side="left", padx=4)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def _refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for s in load_timetable():
            self.tree.insert("", "end", values=_slot_to_row(s))

    def _form_values(self):
        return [v.get().strip() for v in self._vars]

    def _add(self):
        code, name, day, start, end = self._form_values()
        if not all([code, name, day, start, end]):
            messagebox.showwarning("Missing", "Please fill in all fields."); return
        if not _validate_time(start) or not _validate_time(end):
            messagebox.showwarning("Invalid time", "Use HH:MM format (e.g. 08:00)."); return
        slots = load_timetable()
        from datetime import time as dtime
        day_int = DAY_MAP[day.lower()]
        sh, sm = map(int, start.split(":")); eh, em = map(int, end.split(":"))
        slots.append(CourseSlot(code, name, day_int, dtime(sh,sm), dtime(eh,em)))
        save_timetable(slots)
        self._refresh()

    def _load_sel(self):
        sel = self.tree.selection()
        if not sel: return
        vals = self.tree.item(sel[0])["values"]
        for var, val in zip(self._vars, vals):
            var.set(val)

    def _save_edit(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a row to edit first."); return
        idx  = self.tree.index(sel[0])
        code, name, day, start, end = self._form_values()
        if not all([code, name, day, start, end]):
            messagebox.showwarning("Missing", "Fill all fields."); return
        if not _validate_time(start) or not _validate_time(end):
            messagebox.showwarning("Invalid", "Use HH:MM."); return
        from datetime import time as dtime
        day_int = DAY_MAP[day.lower()]
        sh, sm = map(int, start.split(":")); eh, em = map(int, end.split(":"))
        slots = load_timetable()
        slots[idx] = CourseSlot(code, name, day_int, dtime(sh,sm), dtime(eh,em))
        save_timetable(slots)
        self._refresh()

    def _delete(self):
        sel = self.tree.selection()
        if not sel: return
        if not messagebox.askyesno("Confirm", "Delete selected slot?"): return
        idxs = sorted([self.tree.index(s) for s in sel], reverse=True)
        slots = load_timetable()
        for i in idxs:
            slots.pop(i)
        save_timetable(slots)
        self._refresh()


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Students
# ──────────────────────────────────────────────────────────────────────────────

class StudentsTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build()
        self._load_courses()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Course:").pack(side="left")
        self._course_var = tk.StringVar()
        self._course_cb  = ttk.Combobox(top, textvariable=self._course_var,
                                        width=20, state="readonly")
        self._course_cb.pack(side="left", padx=6)
        self._course_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh())
        ttk.Button(top, text="🔄", command=self._load_courses, width=3).pack(side="left")

        # TreeView
        cols = ("name", "matrix")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=14)
        self.tree.heading("name",   text="Name");          self.tree.column("name",   width=220)
        self.tree.heading("matrix", text="Matrix Number"); self.tree.column("matrix", width=120, anchor="center")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10,0), pady=4)
        vsb.pack(side="left", fill="y", pady=4)

        # Form + buttons
        right = ttk.Frame(self)
        right.pack(side="left", fill="y", padx=14, pady=4)

        form = ttk.LabelFrame(right, text="Add Student")
        form.pack(fill="x", pady=6)
        ttk.Label(form, text="Name").grid(row=0, column=0, padx=6, pady=4, sticky="e")
        self._name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self._name_var, width=20).grid(row=0, column=1, padx=4)
        ttk.Label(form, text="Matrix No.").grid(row=1, column=0, padx=6, pady=4, sticky="e")
        self._mn_var = tk.StringVar()
        ttk.Entry(form, textvariable=self._mn_var, width=20).grid(row=1, column=1, padx=4)
        ttk.Button(form, text="➕ Add", command=self._add).grid(row=2, column=0, columnspan=2, pady=6)

        ttk.Button(right, text="🗑 Remove Selected", command=self._remove).pack(pady=4, fill="x")
        ttk.Button(right, text="📥 Import from CSV", command=self._import_csv).pack(pady=4, fill="x")

    def _load_courses(self):
        courses = list_all_courses()
        # Also add courses from timetable that don't have a student file yet
        for slot in load_timetable():
            if slot.course_code not in courses:
                courses.append(slot.course_code)
        courses = sorted(set(courses))
        self._course_cb["values"] = courses
        if courses and not self._course_var.get():
            self._course_var.set(courses[0])
            self._refresh()

    def _refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        code = self._course_var.get()
        if not code: return
        for s in load_enrolled_students(code):
            self.tree.insert("", "end", values=(s["name"], s["matrix_number"]))

    def _add(self):
        code = self._course_var.get()
        name = self._name_var.get().strip()
        mn   = self._mn_var.get().strip()
        if not code:
            messagebox.showwarning("No course", "Select a course first."); return
        if not name or not mn:
            messagebox.showwarning("Missing", "Enter both name and matrix number."); return
        students = load_enrolled_students(code)
        if any(s["matrix_number"].lower() == mn.lower() for s in students):
            messagebox.showinfo("Duplicate", f"{mn} is already in {code}."); return
        students.append({"name": name, "matrix_number": mn})
        save_enrolled_students(code, students)
        self._name_var.set(""); self._mn_var.set("")
        self._refresh()

    def _remove(self):
        sel = self.tree.selection()
        if not sel: return
        code     = self._course_var.get()
        students = load_enrolled_students(code)
        for item in sel:
            mn = self.tree.item(item)["values"][1]
            students = [s for s in students if s["matrix_number"] != str(mn)]
        save_enrolled_students(code, students)
        self._refresh()

    def _import_csv(self):
        code = self._course_var.get()
        if not code:
            messagebox.showwarning("No course", "Select a course first."); return
        path = filedialog.askopenfilename(
            title="Select student CSV",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All", "*.*")]
        )
        if not path: return
        existing = load_enrolled_students(code)
        existing_mns = {s["matrix_number"].lower() for s in existing}
        added = 0
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("name") or row.get("Name") or "").strip()
                    mn   = (row.get("matrix_number") or row.get("Matrix Number") or "").strip()
                    if name and mn and mn.lower() not in existing_mns:
                        existing.append({"name": name, "matrix_number": mn})
                        existing_mns.add(mn.lower())
                        added += 1
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        save_enrolled_students(code, existing)
        self._refresh()
        messagebox.showinfo("Imported", f"Added {added} student(s) to {code}.")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Attendance Log
# ──────────────────────────────────────────────────────────────────────────────

class AttendanceTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build()
        self._refresh()

    def _build(self):
        # Filter bar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=8)

        ttk.Label(bar, text="Date (YYYY-MM-DD):").pack(side="left")
        self._date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(bar, textvariable=self._date_var, width=14).pack(side="left", padx=4)

        ttk.Label(bar, text="Course:").pack(side="left", padx=(12,0))
        self._filter_course = tk.StringVar(value="All")
        self._course_cb = ttk.Combobox(bar, textvariable=self._filter_course,
                                       width=16, state="readonly")
        self._course_cb.pack(side="left", padx=4)

        ttk.Button(bar, text="🔍 Load",   command=self._refresh).pack(side="left", padx=6)
        ttk.Button(bar, text="📤 Export", command=self._export).pack(side="left", padx=4)

        # Stats label
        self._stats_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._stats_var, foreground="#444").pack(anchor="w", padx=12)

        # TreeView
        cols = ("name", "matrix", "course_code", "course_name", "date", "time", "status")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=18)
        headers  = ("Name", "Matrix No.", "Code", "Course Name", "Date", "Time", "Status")
        widths   = (160, 100, 80, 200, 100, 80, 80)
        for col, hdr, w in zip(cols, headers, widths):
            self.tree.heading(col, text=hdr, command=lambda c=col: self._sort(c))
            self.tree.column(col, width=w, anchor="center")

        # Colour tags
        self.tree.tag_configure("present", foreground="#1a7a1a")
        self.tree.tag_configure("late",    foreground="#b05000")

        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10,0))
        vsb.pack(side="left", fill="y")

        self._sort_col  = "time"
        self._sort_rev  = False

    def _load_courses_for_filter(self):
        courses = ["All"] + sorted(list_all_courses())
        self._course_cb["values"] = courses
        if self._filter_course.get() not in courses:
            self._filter_course.set("All")

    def _refresh(self):
        self._load_courses_for_filter()
        for row in self.tree.get_children():
            self.tree.delete(row)

        date_str = self._date_var.get().strip()
        log_path = os.path.join(LOG_DIR, f"attendance_{date_str}.csv")
        if not os.path.isfile(log_path):
            self._stats_var.set(f"No log found for {date_str}")
            return

        filter_course = self._filter_course.get()
        rows = []
        try:
            with open(log_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if filter_course != "All" and r.get("Course Code","").strip() != filter_course:
                        continue
                    rows.append(r)
        except OSError:
            return

        present_count = sum(1 for r in rows if r.get("Status","") == "Present")
        late_count    = sum(1 for r in rows if r.get("Status","") == "Late")
        self._stats_var.set(
            f"Total: {len(rows)}   ✓ Present: {present_count}   ⏱ Late: {late_count}"
        )

        for r in rows:
            status = r.get("Status", "")
            tag    = "present" if status == "Present" else ("late" if status == "Late" else "")
            self.tree.insert("", "end", tags=(tag,), values=(
                r.get("Name",""),
                r.get("Matrix Number",""),
                r.get("Course Code",""),
                r.get("Course Name",""),
                r.get("Date",""),
                r.get("Time",""),
                status,
            ))

    def _sort(self, col):
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        self._sort_rev = not self._sort_rev if col == self._sort_col else False
        self._sort_col = col
        items.sort(reverse=self._sort_rev)
        for i, (_, k) in enumerate(items):
            self.tree.move(k, "", i)

    def _export(self):
        date_str = self._date_var.get().strip()
        log_path = os.path.join(LOG_DIR, f"attendance_{date_str}.csv")
        if not os.path.isfile(log_path):
            messagebox.showinfo("Nothing to export", f"No log for {date_str}."); return
        dest = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"attendance_{date_str}_export.csv",
        )
        if not dest: return
        import shutil
        shutil.copy2(log_path, dest)
        messagebox.showinfo("Exported", f"Saved to:\n{dest}")


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

def run():
    root = tk.Tk()
    root.title("Attendance Admin — EE4001 IDP")
    root.geometry("900x600")
    root.minsize(800, 500)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TNotebook.Tab", padding=[14, 6], font=("Segoe UI", 10))
    style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    nb.add(TimetableTab(nb),  text="📅  Timetable")
    nb.add(StudentsTab(nb),   text="👥  Students")
    nb.add(AttendanceTab(nb), text="📋  Attendance Log")

    root.mainloop()


if __name__ == "__main__":
    run()
