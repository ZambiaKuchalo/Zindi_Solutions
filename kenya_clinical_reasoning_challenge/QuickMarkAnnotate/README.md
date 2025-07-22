# QuickMark Annotate 📝

**Developer**: Daniel Kasonde

QuickMark Annotate is a lightweight, user-friendly tool built with **Streamlit** for annotating summarization datasets. It offers an end-to-end interface for **project management**, **label creation**, **data import/export**, **data editing**, and **interactive annotation** — all with a focus on simplicity and flexibility.

---

## 🚀 Features

- **📁 Project Management**: Create and manage multiple annotation projects.
- **🏷️ Label Management**: Add custom labels with color coding.
- **📥 Data Import**: Upload datasets via CSV, JSON, or raw text input.
- **📤 Data Export**: Export annotated datasets in JSON format.
- **📝 Edit Data**: Add, edit, and search text entries.
- **✏️ Annotation Interface**: Intuitive text annotation with label assignment.
- **📌 Section Insertion**: Add custom section markers at sentence or paragraph levels.
- **🔎 Search Entries**: Quickly filter entries by content.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** for the frontend interface
- **SQLite** for local data storage
- **Pandas & JSON** for data manipulation
- **uuid, datetime** for unique identification and timestamps

---

## 🖥️ How to Run

### 1. Install Requirements

```bash
pip install streamlit pandas
```

SQLite is built-in with Python.

### 2. Run the App

```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## 📂 Folder Structure

```
.
├── app.py               # Main Streamlit application
└── quickmark_annotate.db (auto-created) # SQLite database for local storage
```

---

## 📌 How to Use

1. **Create a Project**: Go to the "Projects" tab to set up a new annotation project.
2. **Add Labels**: Use "Labels" to define custom annotation tags.
3. **Import Data**: Upload datasets through "Import/Export".
4. **Edit & Annotate**: Edit your entries and annotate them using the respective tabs.
5. **Export Results**: Download your work in structured JSON.

---

## 💡 Example Data Format

```json
[
  {
    "source": "This is the full text.",
    "target": "This is the summary."
  }
]
```

---

## 🎁 Highlights

- Fully offline & local — no external server dependencies.
- Designed for simplicity — no setup complexity.
- Flexible enough to support both source and target annotations.

---

## 📜 License

This project is open-source. Feel free to modify and use it under your terms.

---

## 🙌 Acknowledgements

Developed by **Daniel Kasonde** — for efficient text annotation and summarization dataset management.