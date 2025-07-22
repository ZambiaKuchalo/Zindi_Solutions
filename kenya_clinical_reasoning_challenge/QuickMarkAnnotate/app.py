import streamlit as st
import sqlite3
import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

# Database Schema and Management
class DatabaseManager:
    def __init__(self, db_path: str = "quickmark_annotate.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Labels table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                color TEXT DEFAULT '#3498db',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                UNIQUE(project_id, name)
            )
        ''')
        
        # Dataset entries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_entries (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_text TEXT NOT NULL,
                target_text TEXT,
                original_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Annotations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL,
                field_type TEXT NOT NULL CHECK (field_type IN ('source', 'target')),
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                label_name TEXT NOT NULL,
                annotated_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entry_id) REFERENCES dataset_entries (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def create_project(self, name: str, description: str = "") -> str:
        """Create a new project and return its ID"""
        project_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO projects (id, name, description)
                VALUES (?, ?, ?)
            ''', (project_id, name, description))
            conn.commit()
            return project_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Project '{name}' already exists")
        finally:
            conn.close()
    
    def get_projects(self) -> List[Dict]:
        """Get all projects"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, created_at FROM projects ORDER BY created_at DESC')
        projects = [{'id': row[0], 'name': row[1], 'description': row[2], 'created_at': row[3]} 
                   for row in cursor.fetchall()]
        conn.close()
        return projects
    
    def add_label(self, project_id: str, name: str, color: str = '#3498db') -> str:
        """Add a label to a project"""
        label_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO labels (id, project_id, name, color)
                VALUES (?, ?, ?, ?)
            ''', (label_id, project_id, name, color))
            conn.commit()
            return label_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Label '{name}' already exists in this project")
        finally:
            conn.close()
    
    def get_labels(self, project_id: str) -> List[Dict]:
        """Get all labels for a project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, color FROM labels 
            WHERE project_id = ? ORDER BY name
        ''', (project_id,))
        labels = [{'id': row[0], 'name': row[1], 'color': row[2]} 
                 for row in cursor.fetchall()]
        conn.close()
        return labels
    
    def import_data(self, project_id: str, data: List[Dict]) -> int:
        """Import dataset entries into a project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Clear existing data for this project
        cursor.execute('DELETE FROM dataset_entries WHERE project_id = ?', (project_id,))
        
        # Insert new data
        count = 0
        for i, entry in enumerate(data):
            entry_id = str(uuid.uuid4())
            source_text = entry.get('source', '')
            target_text = entry.get('target', '')
            
            cursor.execute('''
                INSERT INTO dataset_entries (id, project_id, source_text, target_text, original_index)
                VALUES (?, ?, ?, ?, ?)
            ''', (entry_id, project_id, source_text, target_text, i))
            count += 1
        
        conn.commit()
        conn.close()
        return count
    
    def get_dataset_entries(self, project_id: str, offset: int = 0, limit: int = 10) -> Tuple[List[Dict], int]:
        """Get dataset entries with pagination"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM dataset_entries WHERE project_id = ?', (project_id,))
        total_count = cursor.fetchone()[0]
        
        # Get entries with pagination
        cursor.execute('''
            SELECT id, source_text, target_text, original_index
            FROM dataset_entries 
            WHERE project_id = ?
            ORDER BY original_index
            LIMIT ? OFFSET ?
        ''', (project_id, limit, offset))
        
        entries = []
        for row in cursor.fetchall():
            entries.append({
                'id': row[0],
                'source_text': row[1],
                'target_text': row[2],
                'original_index': row[3]
            })
        
        conn.close()
        return entries, total_count
    
    def save_annotation(self, entry_id: str, field_type: str, start_pos: int, 
                       end_pos: int, label_name: str, annotated_text: str):
        """Save an annotation"""
        annotation_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO annotations 
            (id, entry_id, field_type, start_pos, end_pos, label_name, annotated_text, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (annotation_id, entry_id, field_type, start_pos, end_pos, 
              label_name, annotated_text, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def update_entry_text(self, entry_id: str, source_text: str = None, target_text: str = None):
        """Update the text content of an entry"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if source_text is not None and target_text is not None:
            cursor.execute('''
                UPDATE dataset_entries 
                SET source_text = ?, target_text = ?
                WHERE id = ?
            ''', (source_text, target_text, entry_id))
        elif source_text is not None:
            cursor.execute('''
                UPDATE dataset_entries 
                SET source_text = ?
                WHERE id = ?
            ''', (source_text, entry_id))
        elif target_text is not None:
            cursor.execute('''
                UPDATE dataset_entries 
                SET target_text = ?
                WHERE id = ?
            ''', (target_text, entry_id))
        
        conn.commit()
        conn.close()
    
    def clear_annotations_for_entry(self, entry_id: str, field_type: str = None):
        """Clear annotations for an entry (optionally for specific field type)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if field_type:
            cursor.execute('DELETE FROM annotations WHERE entry_id = ? AND field_type = ?', 
                          (entry_id, field_type))
        else:
            cursor.execute('DELETE FROM annotations WHERE entry_id = ?', (entry_id,))
        
        conn.commit()
        conn.close()
    
    def get_annotations(self, entry_id: str) -> List[Dict]:
        """Get all annotations for an entry"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, field_type, start_pos, end_pos, label_name, annotated_text
            FROM annotations 
            WHERE entry_id = ?
            ORDER BY start_pos
        ''', (entry_id,))
        
        annotations = []
        for row in cursor.fetchall():
            annotations.append({
                'id': row[0],
                'field_type': row[1],
                'start_pos': row[2],
                'end_pos': row[3],
                'label_name': row[4],
                'annotated_text': row[5]
            })
        
        conn.close()
        return annotations
    
    def delete_annotation(self, annotation_id: str):
        """Delete an annotation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM annotations WHERE id = ?', (annotation_id,))
        conn.commit()
        conn.close()
    
    def export_project_data(self, project_id: str) -> List[Dict]:
        """Export all project data including annotations"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT de.id, de.source_text, de.target_text, de.original_index
            FROM dataset_entries de
            WHERE de.project_id = ?
            ORDER BY de.original_index
        ''', (project_id,))
        
        entries = []
        for row in cursor.fetchall():
            entry = {
                'original_index': row[3],
                'source': row[1],
                'target': row[2],
                'annotations': []
            }
            
            # Get annotations for this entry
            cursor.execute('''
                SELECT field_type, start_pos, end_pos, label_name, annotated_text
                FROM annotations
                WHERE entry_id = ?
                ORDER BY field_type, start_pos
            ''', (row[0],))
            
            for ann_row in cursor.fetchall():
                entry['annotations'].append({
                    'field_type': ann_row[0],
                    'start_pos': ann_row[1],
                    'end_pos': ann_row[2],
                    'label_name': ann_row[3],
                    'annotated_text': ann_row[4]
                })
            
            entries.append(entry)
        
        conn.close()
        return entries
    
    def add_dataset_entry(self, project_id: str, source_text: str, target_text: str = "") -> str:
        """Insert a single new (source, target) entry at the end of the dataset."""
        entry_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()

        # compute new original_index = max+1
        cursor.execute(
            'SELECT COALESCE(MAX(original_index), -1) FROM dataset_entries WHERE project_id = ?',
            (project_id,)
        )
        max_idx = cursor.fetchone()[0]
        new_idx = max_idx + 1

        cursor.execute('''
            INSERT INTO dataset_entries
              (id, project_id, source_text, target_text, original_index)
            VALUES (?, ?, ?, ?, ?)
        ''', (entry_id, project_id, source_text, target_text, new_idx))

        conn.commit()
        conn.close()
        return entry_id

    def search_entries(self, project_id: str, query: str) -> List[Dict]:
        """Return all entries whose source or target contains the query substring."""
        conn = self.get_connection()
        cursor = conn.cursor()
        like = f"%{query}%"
        cursor.execute('''
            SELECT id, source_text, target_text, original_index
            FROM dataset_entries
            WHERE project_id = ?
              AND (LOWER(source_text) LIKE ? OR LOWER(target_text) LIKE ?)
            ORDER BY original_index
        ''', (project_id, like, like))
    
        results = [{
            'id': row[0],
            'source_text': row[1],
            'target_text': row[2],
            'original_index': row[3]
        } for row in cursor.fetchall()]
    
        conn.close()
        return results

# Text Processing and Annotation Utils
class AnnotationProcessor:
    @staticmethod
    def apply_annotations_to_text(text: str, annotations: List[Dict]) -> str:
        """Apply annotations to text, wrapping spans with special tokens"""
        if not annotations:
            return text
        
        # Sort annotations by start position (descending) to apply from end to start
        sorted_annotations = sorted(annotations, key=lambda x: x['start_pos'], reverse=True)
        
        result = text
        for ann in sorted_annotations:
            start, end = ann['start_pos'], ann['end_pos']
            label = ann['label_name']
            
            if 0 <= start < end <= len(text):
                span_text = result[start:end]
                wrapped_span = f"<{label}>{span_text}</{label}>"
                result = result[:start] + wrapped_span + result[end:]
        
        return result
    
    @staticmethod
    def render_text_with_highlights(text: str, annotations: List[Dict], labels: Dict[str, str]) -> str:
        """Render text with HTML highlights for display"""
        if not annotations:
            return text
        
        # Sort annotations by start position (descending)
        sorted_annotations = sorted(annotations, key=lambda x: x['start_pos'], reverse=True)
        
        result = text
        for ann in sorted_annotations:
            start, end = ann['start_pos'], ann['end_pos']
            label = ann['label_name']
            color = labels.get(label, '#3498db')
            
            if 0 <= start < end <= len(result):
                span_text = result[start:end]
                highlighted_span = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px;" title="{label}">{span_text}</span>'
                result = result[:start] + highlighted_span + result[end:]
        
        return result
    
    @staticmethod
    def find_sentence_boundaries(text: str) -> List[int]:
        """Find sentence start positions for sectioner placement"""
        import re
        
        # Pattern to match sentence boundaries (basic implementation)
        sentence_pattern = r'(?<=[.!?])\s+'
        boundaries = [0]  # Start of text
        
        for match in re.finditer(sentence_pattern, text):
            boundaries.append(match.end())
        
        return boundaries
    
    @staticmethod
    def find_paragraph_boundaries(text: str) -> List[int]:
        """Find paragraph start positions for sectioner placement"""
        import re
        
        boundaries = [0]  # Start of text
        
        # Find paragraph breaks (double newlines or single newlines followed by capital letters)
        paragraph_pattern = r'\n\s*(?=[A-Z])|(?:\n\s*\n)'
        
        for match in re.finditer(paragraph_pattern, text):
            boundaries.append(match.end())
        
        return sorted(list(set(boundaries)))  # Remove duplicates and sort
    
    @staticmethod
    def insert_sectioner(text: str, position: int, sectioner: str) -> str:
        """Insert sectioner at specified position"""
        if not sectioner.startswith('[') or not sectioner.endswith(']'):
            sectioner = f"[{sectioner}]"
        
        # Add space after sectioner if the next character isn't whitespace
        if position < len(text) and not text[position].isspace():
            sectioner += " "
        
        return text[:position] + sectioner + text[position:]

# Streamlit App
def initialize_session_state():
    """Initialize session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = None
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    if 'entries_per_page' not in st.session_state:
        st.session_state.entries_per_page = 10
    
    if 'selected_text' not in st.session_state:
        st.session_state.selected_text = None
    
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = {}
    
    if 'sectioner_mode' not in st.session_state:
        st.session_state.sectioner_mode = {}

def project_management_section():
    """Project creation and selection section"""
    st.header("📁 Project Management")
    
    # Project selection
    projects = st.session_state.db_manager.get_projects()
    
    if projects:
        project_options = {p['name']: p['id'] for p in projects}
        selected_project_name = st.selectbox(
            "Select Project",
            options=list(project_options.keys()),
            index=0 if st.session_state.current_project_id is None else 
                  list(project_options.values()).index(st.session_state.current_project_id) 
                  if st.session_state.current_project_id in project_options.values() else 0
        )
        st.session_state.current_project_id = project_options[selected_project_name]
    else:
        st.info("No projects found. Create a new project to get started.")
        st.session_state.current_project_id = None
    
    # Create new project
    with st.expander("➕ Create New Project"):
        with st.form("create_project_form"):
            project_name = st.text_input("Project Name", help="Enter a unique name for your project")
            project_description = st.text_area("Description (Optional)", help="Brief description of the project")
            
            if st.form_submit_button("Create Project"):
                if project_name.strip():
                    try:
                        project_id = st.session_state.db_manager.create_project(
                            project_name.strip(), project_description.strip()
                        )
                        st.success(f"Project '{project_name}' created successfully!")
                        st.session_state.current_project_id = project_id
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.error("Project name cannot be empty")

def label_management_section():
    """Label creation and management section"""
    if not st.session_state.current_project_id:
        return
    
    st.header("🏷️ Label Management")
    
    # Display existing labels
    labels = st.session_state.db_manager.get_labels(st.session_state.current_project_id)
    
    if labels:
        st.subheader("Current Labels")
        cols = st.columns(min(len(labels), 4))
        for i, label in enumerate(labels):
            with cols[i % 4]:
                st.markdown(f'<span style="background-color: {label["color"]}; color: white; padding: 4px 8px; border-radius: 4px; margin: 2px;">{label["name"]}</span>', unsafe_allow_html=True)
    
    # Add new label
    with st.expander("➕ Add New Label"):
        with st.form("add_label_form"):
            col1, col2 = st.columns([3, 1])
            with col1:
                label_name = st.text_input("Label Name", help="e.g., summary, important, claim")
            with col2:
                label_color = st.color_picker("Color", value="#3498db")
            
            if st.form_submit_button("Add Label"):
                if label_name.strip():
                    try:
                        st.session_state.db_manager.add_label(
                            st.session_state.current_project_id,
                            label_name.strip(),
                            label_color
                        )
                        st.success(f"Label '{label_name}' added successfully!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.error("Label name cannot be empty")

def data_import_export_section():
    """Data import and export section"""
    if not st.session_state.current_project_id:
        return
    
    st.header("📥📤 Data Import/Export")
    
    tab1, tab2 = st.tabs(["Import Data", "Export Data"])
    
    with tab1:
        st.subheader("Import Dataset")
        
        upload_method = st.radio("Upload Method", ["File Upload", "Text Input"])
        
        if upload_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'json'],
                help="Upload a CSV or JSON file with your dataset"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        data = df.to_dict('records')
                    elif uploaded_file.name.endswith('.json'):
                        data = json.load(uploaded_file)
                        if not isinstance(data, list):
                            data = [data]
                    
                    st.write(f"Preview ({len(data)} entries):")
                    st.dataframe(pd.DataFrame(data).head())
                    
                    if st.button("Import Data"):
                        count = st.session_state.db_manager.import_data(
                            st.session_state.current_project_id, data
                        )
                        st.success(f"Successfully imported {count} entries!")
                        st.session_state.current_page = 0
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        else:  # Text Input
            st.write("Enter data in JSON format:")
            sample_data = [
                {"source": "Original text to be summarized", "target": "Summary of the text"},
                {"source": "Another piece of text", "target": "Its summary"}
            ]
            text_input = st.text_area(
                "JSON Data",
                value=json.dumps(sample_data, indent=2),
                height=200,
                help="Enter a JSON array of objects with 'source' and optionally 'target' fields"
            )
            
            if st.button("Import from Text"):
                try:
                    data = json.loads(text_input)
                    if not isinstance(data, list):
                        data = [data]
                    
                    count = st.session_state.db_manager.import_data(
                        st.session_state.current_project_id, data
                    )
                    st.success(f"Successfully imported {count} entries!")
                    st.session_state.current_page = 0
                    st.rerun()
                    
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                except Exception as e:
                    st.error(f"Error importing data: {str(e)}")
                    
    with tab2:
        st.subheader("Export Annotated Data")
    
        if st.button("Export as JSON"):
            data = st.session_state.db_manager.export_project_data(st.session_state.current_project_id)
    
            if data:
                json_str = json.dumps(data, indent=2)
                st.download_button(
                    label="📥 Download Annotated JSON",
                    data=json_str,
                    file_name=f"annotated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                st.success(f"{len(data)} entries exported successfully!")
            else:
                st.info("No data available to export.")

                    
                    
def data_editing_section():
    """Data editing interface (with Add & Search)."""
    if not st.session_state.current_project_id:
        st.info("Please select or create a project to edit data.")
        return

    project_id = st.session_state.current_project_id

    # — Add new entry —
    with st.expander("➕ Add New Entry"):
        new_src = st.text_area("New Source Text", height=150)
        new_tgt = st.text_area("New Target Text", height=150)
        if st.button("Add Entry"):
            st.session_state.db_manager.add_dataset_entry(
                project_id, new_src.strip(), new_tgt.strip()
            )
            st.success("Entry added successfully!")
            st.session_state.current_page = 0
            st.rerun()

    # — Search —
    search_q = st.text_input("🔎 Search entries", help="Filter by source or target text")
    if search_q:
        entries = st.session_state.db_manager.search_entries(project_id, search_q.strip())
        total_count = len(entries)
    else:
        entries, total_count = st.session_state.db_manager.get_dataset_entries(
            project_id,
            offset=st.session_state.current_page * st.session_state.entries_per_page,
            limit=st.session_state.entries_per_page
        )

    st.header("📝 Edit Data")
    entries, total_count = st.session_state.db_manager.get_dataset_entries(
        st.session_state.current_project_id,
        offset=st.session_state.current_page * st.session_state.entries_per_page,
        limit=st.session_state.entries_per_page
    )
    if not entries:
        st.info("No data imported yet. Please import some data first.")
        return

    # Pagination (unchanged) …
    total_pages = (total_count - 1) // st.session_state.entries_per_page + 1
    col_prev, col_info, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("◀ Previous Page", disabled=st.session_state.current_page == 0):
            st.session_state.current_page -= 1; st.rerun()
    with col_info:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages} ({total_count} entries)")
    with col_next:
        if st.button("Next Page ▶", disabled=st.session_state.current_page >= total_pages-1):
            st.session_state.current_page += 1; st.rerun()

    st.divider()

    for i, entry in enumerate(entries):
        entry_idx = st.session_state.current_page * st.session_state.entries_per_page + i
        eid = entry["id"]

        with st.container():
            st.subheader(f"Entry {entry_idx+1}")
            is_edit = st.session_state.edit_mode.get(eid, False)
            is_section = st.session_state.sectioner_mode.get(eid, False)

            # --- mode toggles ---
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                if st.button("✏️ Edit Text" if not is_edit else "👁️ View Mode", key=f"toggle_edit_{eid}"):
                    st.session_state.edit_mode[eid] = not is_edit
                    st.session_state.sectioner_mode[eid] = False
                    st.rerun()
            with c2:
                if st.button("🏷️ Add Sectioners" if not is_section else "👁️ View Mode", key=f"toggle_section_{eid}"):
                    st.session_state.sectioner_mode[eid] = not is_section
                    st.session_state.edit_mode[eid] = False
                    st.rerun()

            # --- EDIT MODE ---
            if is_edit:
                src_col, tgt_col = st.columns(2)
                with src_col:
                    st.write("**Source Text**")
                    new_src = st.text_area(
                        "Edit source:", value=entry["source_text"],
                        height=200, key=f"edit_src_{eid}"
                    )
                with tgt_col:
                    st.write("**Target Text**")
                    new_tgt = st.text_area(
                        "Edit target:", value=entry.get("target_text",""),
                        height=200, key=f"edit_tgt_{eid}"
                    )

                save_col, cancel_col = st.columns([1,1])
                with save_col:
                    if st.button("💾 Save Changes", key=f"save_all_{eid}"):
                        st.session_state.db_manager.update_entry_text(
                            eid,
                            source_text=new_src,
                            target_text=new_tgt
                        )
                        # clear any stale annotations on both sides
                        st.session_state.db_manager.clear_annotations_for_entry(eid, "source")
                        st.session_state.db_manager.clear_annotations_for_entry(eid, "target")
                        st.session_state.edit_mode[eid] = False
                        st.success("Both source and target updated! Annotations cleared.")
                        st.rerun()
                with cancel_col:
                    if st.button("❌ Cancel Editing", key=f"cancel_edit_{eid}"):
                        st.session_state.edit_mode[eid] = False
                        st.rerun()

            # --- SECTIONER MODE (unchanged) ---
            elif is_section:
                # … your existing sectioner code for source and target …
                pass

            # --- VIEW MODE (unchanged) ---
            else:
                v1, v2 = st.columns(2)
                with v1:
                    st.write("**Source:**")
                    st.text_area("", entry["source_text"], height=200, disabled=True, key=f"view_src_{eid}")
                with v2:
                    if entry.get("target_text"):
                        st.write("**Target:**")
                        st.text_area("", entry["target_text"], height=200, disabled=True, key=f"view_tgt_{eid}")
                    else:
                        st.info("No target text for this entry.")

            st.divider()

# def data_editing_section():
#     """Data editing interface"""
#     if not st.session_state.current_project_id:
#         st.info("Please select or create a project to edit data.")
#         return
    
#     st.header("📝 Edit Data")
    
#     # Get entries with pagination
#     entries, total_count = st.session_state.db_manager.get_dataset_entries(
#         st.session_state.current_project_id,
#         offset=st.session_state.current_page * st.session_state.entries_per_page,
#         limit=st.session_state.entries_per_page
#     )
    
#     if not entries:
#         st.info("No data imported yet. Please import some data first.")
#         return
    
#     # Pagination controls
#     total_pages = (total_count - 1) // st.session_state.entries_per_page + 1
    
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col1:
#         if st.button("◀ Previous Page", disabled=st.session_state.current_page == 0):
#             st.session_state.current_page -= 1
#             st.rerun()
    
#     with col2:
#         st.write(f"Page {st.session_state.current_page + 1} of {total_pages} ({total_count} total entries)")
    
#     with col3:
#         if st.button("Next Page ▶", disabled=st.session_state.current_page >= total_pages - 1):
#             st.session_state.current_page += 1
#             st.rerun()
    
#     st.divider()
    
#     # Edit interface for each entry
#     for i, entry in enumerate(entries):
#         entry_index = st.session_state.current_page * st.session_state.entries_per_page + i
#         entry_id = entry['id']
        
#         with st.container():
#             st.subheader(f"Entry {entry_index + 1}")
            
#             # Check if this entry is in edit mode
#             is_editing = st.session_state.edit_mode.get(entry_id, False)
#             is_sectioning = st.session_state.sectioner_mode.get(entry_id, False)
            
#             # Mode toggle buttons
#             col_edit, col_section, col_save, col_cancel = st.columns([1, 1, 1, 1])
            
#             with col_edit:
#                 if st.button("✏️ Edit Text" if not is_editing else "👁️ View Mode", 
#                            key=f"toggle_edit_{entry_id}"):
#                     st.session_state.edit_mode[entry_id] = not is_editing
#                     st.session_state.sectioner_mode[entry_id] = False  # Exit sectioner mode
#                     st.rerun()
            
#             with col_section:
#                 if st.button("🏷️ Add Sectioners" if not is_sectioning else "👁️ View Mode", 
#                            key=f"toggle_section_{entry_id}"):
#                     st.session_state.sectioner_mode[entry_id] = not is_sectioning
#                     st.session_state.edit_mode[entry_id] = False  # Exit edit mode
#                     st.rerun()
            
#             if is_editing or is_sectioning:
#                 with col_save:
#                     save_clicked = st.button("💾 Save Changes", key=f"save_{entry_id}")
                
#                 with col_cancel:
#                     if st.button("❌ Cancel", key=f"cancel_{entry_id}"):
#                         st.session_state.edit_mode[entry_id] = False
#                         st.session_state.sectioner_mode[entry_id] = False
#                         st.rerun()
            
#             # Create columns for source and target
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.write("**Source Text:**")
                
#                 if is_editing:
#                     # Editable text area
#                     new_source = st.text_area(
#                         "Edit source text:",
#                         value=entry['source_text'],
#                         height=200,
#                         key=f"edit_source_{entry_id}"
#                     )
                    
#                     if save_clicked:
#                         st.session_state.db_manager.update_entry_text(entry_id, source_text=new_source)
#                         # Clear annotations that might be invalid after text change
#                         st.session_state.db_manager.clear_annotations_for_entry(entry_id, 'source')
#                         st.session_state.edit_mode[entry_id] = False
#                         st.success("Source text updated! Note: Existing annotations were cleared.")
#                         st.rerun()
                
#                 elif is_sectioning:
#                     # Sectioner interface
#                     st.write("**Add Section Markers:**")
                    
#                     # Show current text
#                     st.text_area(
#                         "Current text:",
#                         value=entry['source_text'],
#                         height=150,
#                         disabled=True,
#                         key=f"view_source_{entry_id}"
#                     )
                    
#                     # Sectioner controls
#                     sectioner_name = st.text_input(
#                         "Section name (e.g., HISTORY, BACKGROUND, SUMMARY):",
#                         key=f"sectioner_name_source_{entry_id}"
#                     )
                    
#                     placement_type = st.radio(
#                         "Placement:",
#                         ["At sentence start", "At paragraph start", "At specific position"],
#                         key=f"placement_source_{entry_id}"
#                     )
                    
#                     if placement_type == "At sentence start":
#                         boundaries = AnnotationProcessor.find_sentence_boundaries(entry['source_text'])
#                         position_options = {}
#                         for pos in boundaries:
#                             preview = entry['source_text'][pos:pos+50] + "..." if len(entry['source_text']) > pos+50 else entry['source_text'][pos:]
#                             position_options[f"Position {pos}: {preview}"] = pos
                        
#                         if position_options:
#                             selected_pos_label = st.selectbox(
#                                 "Choose sentence:",
#                                 list(position_options.keys()),
#                                 key=f"sentence_pos_source_{entry_id}"
#                             )
#                             selected_pos = position_options[selected_pos_label]
#                         else:
#                             selected_pos = 0
                    
#                     elif placement_type == "At paragraph start":
#                         boundaries = AnnotationProcessor.find_paragraph_boundaries(entry['source_text'])
#                         position_options = {}
#                         for pos in boundaries:
#                             preview = entry['source_text'][pos:pos+50] + "..." if len(entry['source_text']) > pos+50 else entry['source_text'][pos:]
#                             position_options[f"Position {pos}: {preview}"] = pos
                        
#                         if position_options:
#                             selected_pos_label = st.selectbox(
#                                 "Choose paragraph:",
#                                 list(position_options.keys()),
#                                 key=f"para_pos_source_{entry_id}"
#                             )
#                             selected_pos = position_options[selected_pos_label]
#                         else:
#                             selected_pos = 0
                    
#                     else:  # Specific position
#                         selected_pos = st.number_input(
#                             "Character position:",
#                             min_value=0,
#                             max_value=len(entry['source_text']),
#                             value=0,
#                             key=f"custom_pos_source_{entry_id}"
#                         )
                    
#                     if sectioner_name and st.button("Add Sectioner to Source", key=f"add_sectioner_source_{entry_id}"):
#                         new_text = AnnotationProcessor.insert_sectioner(
#                             entry['source_text'], selected_pos, sectioner_name
#                         )
#                         st.session_state.db_manager.update_entry_text(entry_id, source_text=new_text)
#                         # Clear annotations that might be invalid after text change
#                         st.session_state.db_manager.clear_annotations_for_entry(entry_id, 'source')
#                         st.success(f"Sectioner [{sectioner_name}] added to source! Existing annotations were cleared.")
#                         st.rerun()
                
#                 else:
#                     # View mode
#                     st.text_area(
#                         "Source:",
#                         value=entry['source_text'],
#                         height=200,
#                         disabled=True,
#                         key=f"view_only_source_{entry_id}"
#                     )
            
#             with col2:
#                 if entry['target_text']:
#                     st.write("**Target Text:**")
                    
#                     if is_editing:
#                         # Editable text area
#                         new_target = st.text_area(
#                             "Edit target text:",
#                             value=entry['target_text'],
#                             height=200,
#                             key=f"edit_target_{entry_id}"
#                         )
                        
#                         if save_clicked:
#                             st.session_state.db_manager.update_entry_text(entry_id, target_text=new_target)
#                             # Clear annotations that might be invalid after text change
#                             st.session_state.db_manager.clear_annotations_for_entry(entry_id, 'target')
#                             st.session_state.edit_mode[entry_id] = False
#                             st.success("Target text updated! Note: Existing annotations were cleared.")
#                             st.rerun()
                    
#                     elif is_sectioning:
#                         # Sectioner interface
#                         st.write("**Add Section Markers:**")
                        
#                         # Show current text
#                         st.text_area(
#                             "Current text:",
#                             value=entry['target_text'],
#                             height=150,
#                             disabled=True,
#                             key=f"view_target_{entry_id}"
#                         )
                        
#                         # Sectioner controls
#                         sectioner_name = st.text_input(
#                             "Section name (e.g., HISTORY, BACKGROUND, SUMMARY):",
#                             key=f"sectioner_name_target_{entry_id}"
#                         )
                        
#                         placement_type = st.radio(
#                             "Placement:",
#                             ["At sentence start", "At paragraph start", "At specific position"],
#                             key=f"placement_target_{entry_id}"
#                         )
                        
#                         if placement_type == "At sentence start":
#                             boundaries = AnnotationProcessor.find_sentence_boundaries(entry['target_text'])
#                             position_options = {}
#                             for pos in boundaries:
#                                 preview = entry['target_text'][pos:pos+50] + "..." if len(entry['target_text']) > pos+50 else entry['target_text'][pos:]
#                                 position_options[f"Position {pos}: {preview}"] = pos
                            
#                             if position_options:
#                                 selected_pos_label = st.selectbox(
#                                     "Choose sentence:",
#                                     list(position_options.keys()),
#                                     key=f"sentence_pos_target_{entry_id}"
#                                 )
#                                 selected_pos = position_options[selected_pos_label]
#                             else:
#                                 selected_pos = 0
                        
#                         elif placement_type == "At paragraph start":
#                             boundaries = AnnotationProcessor.find_paragraph_boundaries(entry['target_text'])
#                             position_options = {}
#                             for pos in boundaries:
#                                 preview = entry['target_text'][pos:pos+50] + "..." if len(entry['target_text']) > pos+50 else entry['target_text'][pos:]
#                                 position_options[f"Position {pos}: {preview}"] = pos
                            
#                             if position_options:
#                                 selected_pos_label = st.selectbox(
#                                     "Choose paragraph:",
#                                     list(position_options.keys()),
#                                     key=f"para_pos_target_{entry_id}"
#                                 )
#                                 selected_pos = position_options[selected_pos_label]
#                             else:
#                                 selected_pos = 0
                        
#                         else:  # Specific position
#                             selected_pos = st.number_input(
#                                 "Character position:",
#                                 min_value=0,
#                                 max_value=len(entry['target_text']),
#                                 value=0,
#                                 key=f"custom_pos_target_{entry_id}"
#                             )
                        
#                         if sectioner_name and st.button("Add Sectioner to Target", key=f"add_sectioner_target_{entry_id}"):
#                             new_text = AnnotationProcessor.insert_sectioner(
#                                 entry['target_text'], selected_pos, sectioner_name
#                             )
#                             st.session_state.db_manager.update_entry_text(entry_id, target_text=new_text)
#                             # Clear annotations that might be invalid after text change
#                             st.session_state.db_manager.clear_annotations_for_entry(entry_id, 'target')
#                             st.success(f"Sectioner [{sectioner_name}] added to target! Existing annotations were cleared.")
#                             st.rerun()
                    
#                     else:
#                         # View mode
#                         st.text_area(
#                             "Target:",
#                             value=entry['target_text'],
#                             height=200,
#                             disabled=True,
#                             key=f"view_only_target_{entry_id}"
#                         )
#                 else:
#                     st.info("No target text available for this entry")
            
#             st.divider()
    
#     with tab2:
#         st.subheader("Export Annotated Data")
        
#         if st.button("Export as JSON"):
#             data = st.session_state.db_manager.export_project_data(st.session_state.current_project_id)
            
#             if data:
#                 json_str = json.dumps(data, indent=2)
#                 st.download_button(
#                     label="Download JSON",
#                     data=json_str,
#                     file_name=f"annotated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                     mime="application/json"
#                 )
#                 st.success(f"Ready to download {len(data)} annotated entries!")
#             else:
#                 st.info("No data to export")

def annotation_interface():
    """Main annotation interface"""
    if not st.session_state.current_project_id:
        st.info("Please select or create a project to start annotating.")
        return
    
    st.header("✏️ Annotation Interface")
    
    # Get entries with pagination
    entries, total_count = st.session_state.db_manager.get_dataset_entries(
        st.session_state.current_project_id,
        offset=st.session_state.current_page * st.session_state.entries_per_page,
        limit=st.session_state.entries_per_page
    )
    
    if not entries:
        st.info("No data imported yet. Please import some data first.")
        return
    
    # Pagination controls
    total_pages = (total_count - 1) // st.session_state.entries_per_page + 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Previous", disabled=st.session_state.current_page == 0):
            st.session_state.current_page -= 1
            st.rerun()
    
    with col2:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages} ({total_count} total entries)")
    
    with col3:
        if st.button("Next ▶", disabled=st.session_state.current_page >= total_pages - 1):
            st.session_state.current_page += 1
            st.rerun()
    
    # Get labels for current project
    labels = st.session_state.db_manager.get_labels(st.session_state.current_project_id)
    label_colors = {label['name']: label['color'] for label in labels}
    
    if not labels:
        st.warning("No labels defined for this project. Please add some labels first.")
        return
    
    # Annotation interface for each entry
    for i, entry in enumerate(entries):
        entry_index = st.session_state.current_page * st.session_state.entries_per_page + i
        
        with st.container():
            st.subheader(f"Entry {entry_index + 1}")
            
            # Get existing annotations
            annotations = st.session_state.db_manager.get_annotations(entry['id'])
            source_annotations = [ann for ann in annotations if ann['field_type'] == 'source']
            target_annotations = [ann for ann in annotations if ann['field_type'] == 'target']
            
            # Create two columns for source and target
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Source Text:**")
                
                # Text selection interface for source
                source_key = f"source_text_{entry['id']}"
                
                # Display annotated text
                if source_annotations:
                    annotated_html = AnnotationProcessor.render_text_with_highlights(
                        entry['source_text'], source_annotations, label_colors
                    )
                    st.markdown(annotated_html, unsafe_allow_html=True)
                else:
                    st.write(entry['source_text'])
                
                # Text selection for annotation
                st.write("**Select text to annotate:**")
                selected_source = st.text_input(
                    "Selected text (source)",
                    key=f"selected_source_{entry['id']}",
                    help="Copy and paste text from above to annotate"
                )
                
                if selected_source:
                    # Find the position of selected text
                    start_pos = entry['source_text'].find(selected_source)
                    if start_pos != -1:
                        end_pos = start_pos + len(selected_source)
                        
                        # Label selection
                        selected_label = st.selectbox(
                            "Choose label",
                            options=[label['name'] for label in labels],
                            key=f"label_source_{entry['id']}"
                        )
                        
                        if st.button(f"Annotate Source", key=f"annotate_source_{entry['id']}"):
                            st.session_state.db_manager.save_annotation(
                                entry['id'], 'source', start_pos, end_pos,
                                selected_label, selected_source
                            )
                            st.success("Annotation saved!")
                            st.rerun()
                    else:
                        st.warning("Selected text not found in source. Please copy exact text.")
                
                # Display existing annotations
                if source_annotations:
                    st.write("**Existing annotations:**")
                    for ann in source_annotations:
                        col_ann, col_del = st.columns([4, 1])
                        with col_ann:
                            st.text(f"'{ann['annotated_text'][:50]}...' → {ann['label_name']}")
                        with col_del:
                            if st.button("🗑️", key=f"del_source_{ann['id']}"):
                                st.session_state.db_manager.delete_annotation(ann['id'])
                                st.rerun()
            
            with col2:
                if entry['target_text']:
                    st.write("**Target Text:**")
                    
                    # Display annotated text
                    if target_annotations:
                        annotated_html = AnnotationProcessor.render_text_with_highlights(
                            entry['target_text'], target_annotations, label_colors
                        )
                        st.markdown(annotated_html, unsafe_allow_html=True)
                    else:
                        st.write(entry['target_text'])
                    
                    # Text selection for annotation
                    st.write("**Select text to annotate:**")
                    selected_target = st.text_input(
                        "Selected text (target)",
                        key=f"selected_target_{entry['id']}",
                        help="Copy and paste text from above to annotate"
                    )
                    
                    if selected_target:
                        # Find the position of selected text
                        start_pos = entry['target_text'].find(selected_target)
                        if start_pos != -1:
                            end_pos = start_pos + len(selected_target)
                            
                            # Label selection
                            selected_label = st.selectbox(
                                "Choose label",
                                options=[label['name'] for label in labels],
                                key=f"label_target_{entry['id']}"
                            )
                            
                            if st.button(f"Annotate Target", key=f"annotate_target_{entry['id']}"):
                                st.session_state.db_manager.save_annotation(
                                    entry['id'], 'target', start_pos, end_pos,
                                    selected_label, selected_target
                                )
                                st.success("Annotation saved!")
                                st.rerun()
                        else:
                            st.warning("Selected text not found in target. Please copy exact text.")
                    
                    # Display existing annotations
                    if target_annotations:
                        st.write("**Existing annotations:**")
                        for ann in target_annotations:
                            col_ann, col_del = st.columns([4, 1])
                            with col_ann:
                                st.text(f"'{ann['annotated_text'][:50]}...' → {ann['label_name']}")
                            with col_del:
                                if st.button("🗑️", key=f"del_target_{ann['id']}"):
                                    st.session_state.db_manager.delete_annotation(ann['id'])
                                    st.rerun()
                else:
                    st.info("No target text available for this entry")
            
            st.divider()

def main():
    """Main application function"""
    st.set_page_config(
        page_title="QuickMark Annotate",
        page_icon="✏️",
        layout="wide"
    )
    
    st.title("✏️ QuickMark Annotate")
    st.markdown("*Lightweight Summarization Dataset Annotation Tool*")
    
    # Initialize session state
    initialize_session_state()
    
    # Main interface sections
    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Go to:",
            ["Projects", "Labels", "Import/Export", "Edit Data", "Annotate"]
        )
    
    if section == "Projects":
        project_management_section()
    elif section == "Labels":
        label_management_section()
    elif section == "Import/Export":
        data_import_export_section()
    elif section == "Edit Data":
        data_editing_section()
    elif section == "Annotate":
        annotation_interface()

if __name__ == "__main__":
    main()