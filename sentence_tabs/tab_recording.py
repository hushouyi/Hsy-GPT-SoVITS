"""
Recording tab page.

Displays 10 sentence rows with individual Gradio buttons for REC/PLAY/DEL.
All components are generated at build time for maximum Gradio compatibility.
"""

import os
import time
import gradio as gr
from functools import partial
from typing import List, Optional

from sentence_recorder.script_reader import ScriptReader, PageData
from sentence_recorder.mapping import MappingManager, MappingEntry
from sentence_recorder.recorder import RecordingManager
from sentence_recorder.project_manager import ProjectManager
from sentence_recorder.state import AppState


PAGE_SIZE = 10


def create_recording_tab():
    """Create the recording tab with per-row Gradio buttons."""
    # In-memory state
    _script_reader = ScriptReader()
    _mapping_mgr = MappingManager()
    _recorder = RecordingManager()

    # ==================== Helpers ====================

    def load_project(project_name: str):
        """Load project data into memory."""
        script_path = ProjectManager.get_script_path(project_name)
        mapping_path = ProjectManager.get_mapping_path(project_name)
        if os.path.exists(script_path):
            _script_reader.read(script_path)
        _mapping_mgr.load(mapping_path)

    def get_stats(page: int) -> tuple:
        """Return (progress_html, page_header, _, category_check, is_all_met)."""
        pg = _script_reader.get_page(page, PAGE_SIZE)
        if not pg:
            return "", "### No data", "", "", False
        mapping = _mapping_mgr.get_all()
        total = _script_reader.total_sentences

        # Overall progress
        recorded_total = sum(1 for e in mapping.values() if e.confirmed and e.wav_path)
        pct = int(recorded_total / max(total, 1) * 100)

        # Per-category
        cats = _script_reader.get_categories()
        cat_lines = []
        all_met = True
        for c in cats:
            cnt = sum(1 for idx in range(c.start_idx, c.end_idx + 1)
                      if mapping.get(idx) and mapping[idx].confirmed and mapping[idx].wav_path)
            met = cnt >= 5
            if not met:
                all_met = False
            cat_lines.append(f"{c.name[:6]} {cnt}/{c.total} {'✅' if met else '❌'}")

        prog_html = f"""<div style="padding:6px;background:#f5f5f5;border-radius:6px;border:1px solid #ddd;font-size:12px;">
<div><b>总体：</b> {recorded_total}/{total} ({pct}%)</div>
<div style="background:#e0e0e0;height:12px;border-radius:6px;margin:2px 0 6px 0;">
<div style="background:linear-gradient(90deg,#4CAF50,#81C784);width:{pct}%;height:12px;border-radius:6px;"></div></div>
<div style="display:flex;flex-wrap:wrap;gap:2px;">{"".join(f'<div style="flex:0 0 auto;min-width:80px;">{l}</div>' for l in cat_lines)}</div></div>"""

        header = f"### 第 {pg.page}/{pg.total_pages} 页 | {pg.category[:30] if pg.category else ''}"
        stats_text = f"**第 {pg.page} 页**  |  **共 {len(pg.sentences)} 句**"
        ck_text = "✅ 全部达标，可以训练！" if all_met else "❌ 有类别未达标（每类需≥5条）"
        return prog_html, header, stats_text, ck_text, all_met

    def get_sentence_texts(page: int) -> List[str]:
        """Return list of 10 sentence texts for the given page."""
        pg = _script_reader.get_page(page, PAGE_SIZE)
        if not pg:
            return [""] * 10
        texts = []
        for s in pg.sentences:
            txt = s.text
            if len(txt) > 55:
                txt = txt[:53] + "..."
            texts.append(txt)
        # Pad to 10
        while len(texts) < 10:
            texts.append("")
        return texts

    def get_sentence_indices(page: int) -> List[int]:
        """Return list of sentence indices for current page."""
        pg = _script_reader.get_page(page, PAGE_SIZE)
        if not pg:
            return [0] * 10
        indices = [s.idx for s in pg.sentences]
        while len(indices) < 10:
            indices.append(0)
        return indices

    def get_row_states(page: int) -> List[tuple]:
        """Return list of (dot, bg_color, rec_text, rec_disabled, play_disabled, del_disabled, variant) for 10 rows."""
        pg = _script_reader.get_page(page, PAGE_SIZE)
        if not pg:
            return [("", "", "", True, True, True, "secondary")] * 10
        mapping = _mapping_mgr.get_all()
        rows = []
        for s in pg.sentences:
            entry = mapping.get(s.idx)
            has = entry and entry.wav_path and os.path.exists(
                os.path.join(ProjectManager.get_project_dir(AppState.get_current_project()), entry.wav_path))
            conf = entry and entry.confirmed and has
            is_rec = _recorder.is_recording and _recorder.current_save_path and str(s.idx) in _recorder.current_save_path

            if is_rec:
                rows.append(("🔴", "#FFF0F0", "停止", True, True, True, "stop"))
            elif has and conf:
                rows.append(("🟢", "#F0FFF0", "录音", False, False, False, "secondary"))
            else:
                rows.append(("⚪", "", "录音", False, True, True, "secondary"))

        while len(rows) < 10:
            rows.append(("", "", "录音", True, True, True, "secondary"))
        return rows

    # ==================== Build UI ====================

    with gr.Column() as rec_tab:
        # ─── Project Bar ───
        with gr.Row():
            project_dropdown = gr.Dropdown(choices=[], label="项目", scale=3)
            btn_new_project = gr.Button("＋", size="sm", scale=0)
            btn_lock_project = gr.Button("🔓", size="sm", scale=0)
            btn_delete_project = gr.Button("✕", size="sm", scale=0, variant="stop")
            btn_refresh_projects = gr.Button("↻", size="sm", scale=0)

        gr.Markdown("---")

        # ─── Progress ───
        progress_html = gr.HTML(value="<div>加载中...</div>")
        page_header = gr.Markdown("### 加载中...")

        # Store current page number
        current_page = gr.State(value=1)

        # ─── 10 sentence rows with individual buttons ───
        row_cells = []
        btn_recs = []
        btn_plays = []
        btn_dels = []

        for i in range(10):
            with gr.Row() as row:
                cell = gr.HTML(value="")  # combined dot + label as HTML
                brec = gr.Button("录音", size="sm", scale=0, elem_id=f"rec_{i}")
                bplay = gr.Button("播放", size="sm", scale=0, elem_id=f"play_{i}")
                bdel = gr.Button("删除", size="sm", scale=0, elem_id=f"del_{i}")
            row_cells.append(cell)
            btn_recs.append(brec)
            btn_plays.append(bplay)
            btn_dels.append(bdel)

        gr.Markdown("---")
        page_display = gr.HTML(value="<b>第 1/16 页</b>")
        with gr.Row(elem_classes="pagination-wrap"):
            btn_prev = gr.Button("◀ 上一页", scale=0, size="sm")
            jump_input = gr.Number(value=1, label="", minimum=1, maximum=99, scale=0, elem_classes="page-jump")
            btn_jump = gr.Button("跳转", scale=0, size="sm")
            btn_next = gr.Button("下一页 ▶", scale=0, size="sm")
        category_check = gr.Markdown("")
        btn_done = gr.Button("❌ 完成录制（每类需≥5条）",
                            variant="primary", interactive=False, size="lg")

    # ==================== Project helpers ====================

    def get_project_ui_state() -> list:
        """Return [project_dropdown_update, lock_btn_update] for current state."""
        projects = ProjectManager.list_projects()
        current = AppState.get_current_project()
        locked = ProjectManager.is_locked(current)
        return [
            gr.update(choices=projects, value=current if current in projects else (projects[0] if projects else "")),
            gr.update(value="\U0001f512" if locked else "\U0001f513",
                      variant="stop" if locked else "secondary"),
        ]

    def switch_project(project_name: str) -> list:
        """Switch to a different project. Returns full UI update."""
        if not project_name or project_name == AppState.get_current_project():
            return get_project_ui_state() + [1] + refresh_page(1)
        if _recorder.is_recording:
            print("[WARN] Stop recording before switching project")
            return get_project_ui_state() + [1] + refresh_page(1)
        # Save current mapping
        _mapping_mgr.flush()
        # Load new project
        AppState.set_current_project(project_name)
        load_project(project_name)
        print(f"[OK] Switched to project: {project_name}")
        return get_project_ui_state() + [1] + refresh_page(1)

    # ==================== Page update function ====================

    def refresh_page(page: int) -> list:
        """Refresh all page UI elements. Returns values matching all_outputs. Callers must
        prepend get_project_ui_state() + [page] before these."""
        pg = _script_reader.get_page(page, PAGE_SIZE)
        if not pg:
            return [gr.update()] * (2 + 10*4 + 3)

        prog, hdr, _, ck, all_met = get_stats(page)
        states = get_row_states(page)
        texts = get_sentence_texts(page)
        indices = get_sentence_indices(page)
        total_p = pg.total_pages

        outputs = []
        # progress_html, page_header
        outputs.extend([prog, hdr])

        # 10 rows: combined_cell(dot+text), rec_btn, play_btn, del_btn
        for i in range(10):
            idx = indices[i]
            is_valid = idx > 0 and i < len(pg.sentences)
            if is_valid:
                dot_val = states[i][0]
                bg = states[i][1] if states[i][1] else "transparent"
                label_val = f"#{idx} {texts[i]}"
                rec_text = states[i][2]
                rec_dis = states[i][3]
                play_dis = states[i][4]
                del_dis = states[i][5]
                play_var = states[i][6] if len(states[i]) > 6 else "secondary"
            else:
                dot_val = ""
                bg = "transparent"
                label_val = ""
                rec_text = "录音"
                rec_dis = True
                play_dis = True
                del_dis = True
                play_var = "secondary"

            cell_html = f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="font-size:16px;">{dot_val}</span><span style="background:{bg};padding:1px 4px;border-radius:3px;font-size:13px;">{label_val}</span></span>'

            outputs.extend([
                cell_html,
                gr.update(value=rec_text, variant=play_var, interactive=not rec_dis),
                gr.update(interactive=not play_dis),
                gr.update(interactive=not del_dis),
            ])

        # Pagination + check + done button
        outputs.extend([
            f"<b>第 {page}/{total_p} 页</b>",
            ck,
            gr.update(
                value="✅ 完成录制 → 开始训练" if all_met
                      else "❌ 完成录制（每类需≥5条）",
                interactive=all_met,
                variant="primary" if all_met else "secondary"
            ),
        ])

        return outputs

    # ==================== Action functions ====================

    def on_rec_click(row_idx: int, page: int) -> list:
        """Handle record button click for row_idx on current page."""
        indices = get_sentence_indices(page)
        if row_idx >= len(indices):
            return get_project_ui_state() + [page] + refresh_page(page)
        idx = indices[row_idx]
        if idx <= 0:
            return get_project_ui_state() + [page] + refresh_page(page)

        if _recorder.is_recording:
            result = _recorder.stop_recording()
            if result and result["path"]:
                _mapping_mgr.update_field(idx, duration_sec=result["duration"],
                                          recorded_at=result.get("recorded_at", ""), confirmed=True)
                print(f"[OK] Recording stopped for #{idx}")
        else:
            if AppState.get_training():
                print("[WARN] Cannot record while training")
                return get_project_ui_state() + [page] + refresh_page(page)
            proj = AppState.get_current_project()
            rec_dir = ProjectManager.get_recorded_dir(proj)
            os.makedirs(rec_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            wav_name = f"p{page:03d}_i{idx:02d}_{ts}.wav"
            wav_path = os.path.join(rec_dir, wav_name)
            rel = f"recorded/{wav_name}"
            ok = _recorder.start_recording(wav_path)
            if ok:
                entries = _mapping_mgr.get_all()
                txt = entries[idx].text if idx in entries else ""
                _mapping_mgr.update(idx, MappingEntry(idx, txt, rel, True, 0, ts))
                print(f"[OK] Recording started for #{idx}")
        return get_project_ui_state() + [page] + refresh_page(page)

    def on_play_click(row_idx: int, page: int) -> list:
        """Handle play button click."""
        indices = get_sentence_indices(page)
        if row_idx >= len(indices):
            return get_project_ui_state() + [page] + refresh_page(page)
        idx = indices[row_idx]
        if idx <= 0:
            return get_project_ui_state() + [page] + refresh_page(page)
        entries = _mapping_mgr.get_all()
        entry = entries.get(idx)
        if entry and entry.wav_path:
            proj = AppState.get_current_project()
            full = os.path.join(ProjectManager.get_project_dir(proj), entry.wav_path)
            if os.path.exists(full):
                print(f"[OK] Would play: {full}")
            else:
                print(f"[WARN] File missing: {full}")
                _mapping_mgr.update_field(idx, wav_path="", confirmed=False)
        return get_project_ui_state() + [page] + refresh_page(page)

    def on_del_click(row_idx: int, page: int) -> list:
        """Handle delete button click."""
        indices = get_sentence_indices(page)
        if row_idx >= len(indices):
            return get_project_ui_state() + [page] + refresh_page(page)
        idx = indices[row_idx]
        if idx <= 0:
            return get_project_ui_state() + [page] + refresh_page(page)
        entries = _mapping_mgr.get_all()
        entry = entries.get(idx)
        if entry and entry.wav_path:
            proj = AppState.get_current_project()
            full = os.path.join(ProjectManager.get_project_dir(proj), entry.wav_path)
            if os.path.exists(full):
                os.remove(full)
            _mapping_mgr.update_field(idx, wav_path="", confirmed=False,
                                      duration_sec=0, recorded_at="0")
            print(f"[OK] Cleared #{idx}")
        _mapping_mgr.auto_flush()
        return get_project_ui_state() + [page] + refresh_page(page)

    def go_prev(page: int) -> list:
        p = max(1, page - 1)
        return get_project_ui_state() + [p] + refresh_page(p)

    def go_next(page: int) -> list:
        p = min(_script_reader.total_pages, page + 1)
        return get_project_ui_state() + [p] + refresh_page(p)

    def go_jump(target: int, page: int) -> list:
        p = max(1, min(_script_reader.total_pages, int(target)))
        return get_project_ui_state() + [p] + refresh_page(p)

    def on_new_project() -> list:
        """Create a new project with a dialog."""
        # For simplicity, just add numbered new project
        base = "new_project"
        name = base
        i = 1
        while os.path.exists(ProjectManager.get_project_dir(name)):
            name = f"{base}_{i}"
            i += 1
        ok = ProjectManager.create(name, script_source="default")
        if ok:
            _mapping_mgr.flush()
            AppState.set_current_project(name)
            load_project(name)
            print(f"[OK] Created project: {name}")
        return get_project_ui_state() + [1] + refresh_page(1)

    def on_delete_project() -> list:
        """Delete current project (if not locked)."""
        current = AppState.get_current_project()
        if current == "default":
            print("[WARN] Cannot delete default project")
            return get_project_ui_state() + [1] + refresh_page(1)
        if ProjectManager.is_locked(current):
            print(f"[WARN] Project {current} is locked")
            return get_project_ui_state() + [1] + refresh_page(1)
        _mapping_mgr.flush()
        ProjectManager.delete(current)
        # Switch to default
        AppState.set_current_project("default")
        load_project("default")
        print(f"[OK] Deleted project: {current}")
        return get_project_ui_state() + [1] + refresh_page(1)

    def on_toggle_lock() -> list:
        """Toggle lock on current project."""
        current = AppState.get_current_project()
        if ProjectManager.is_locked(current):
            ProjectManager.unlock(current)
            print(f"[OK] Unlocked project: {current}")
        else:
            ProjectManager.lock(current)
            print(f"[OK] Locked project: {current}")
        return get_project_ui_state() + [1] + refresh_page(1)

    def on_refresh_projects() -> list:
        """Refresh project list from disk."""
        return get_project_ui_state() + [1] + refresh_page(1)

    # ==================== Wire events ====================

    project_outs = [project_dropdown, btn_lock_project]
    partial_out = [current_page, progress_html, page_header]

    row_out = []
    for i in range(10):
        row_out.append(row_cells[i])
        row_out.append(btn_recs[i])
        row_out.append(btn_plays[i])
        row_out.append(btn_dels[i])

    all_outputs = project_outs + partial_out + row_out + [page_display, category_check, btn_done]

    # Project events
    project_dropdown.change(fn=switch_project, inputs=[project_dropdown], outputs=all_outputs)
    btn_new_project.click(fn=on_new_project, inputs=[], outputs=all_outputs)
    btn_delete_project.click(fn=on_delete_project, inputs=[], outputs=all_outputs)
    btn_lock_project.click(fn=on_toggle_lock, inputs=[], outputs=all_outputs)
    btn_refresh_projects.click(fn=on_refresh_projects, inputs=[], outputs=all_outputs)

    # Pagination
    btn_prev.click(fn=go_prev, inputs=[current_page], outputs=all_outputs)
    btn_next.click(fn=go_next, inputs=[current_page], outputs=all_outputs)
    btn_jump.click(fn=go_jump, inputs=[jump_input, current_page], outputs=all_outputs)

    # Per-row button clicks
    for i in range(10):
        btn_recs[i].click(fn=partial(on_rec_click, i), inputs=[current_page], outputs=all_outputs)
        btn_plays[i].click(fn=partial(on_play_click, i), inputs=[current_page], outputs=all_outputs)
        btn_dels[i].click(fn=partial(on_del_click, i), inputs=[current_page], outputs=all_outputs)

    # Return components
    return {
        "progress_html": progress_html,
        "page_header": page_header,
        "current_page": current_page,
        "project_dropdown": project_dropdown,
        "btn_lock_project": btn_lock_project,
        "btn_prev": btn_prev,
        "btn_next": btn_next,
        "jump_input": jump_input,
        "btn_jump": btn_jump,
        "category_check": category_check,
        "btn_done": btn_done,
        "load_project": load_project,
        "refresh_page": refresh_page,
        "script_reader": _script_reader,
        "mapping_mgr": _mapping_mgr,
        "all_outputs": all_outputs,
    }
