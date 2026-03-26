## **DEDUPE ML**

This repository contains a full-stack application developed to detect and manage duplicate data across multiple file formats. The project demonstrates how intelligent similarity techniques can be applied to solve real-world data redundancy and storage optimization problems.

---

## **PROJECT OVERVIEW**

DeDupe ML is an advanced file management system that identifies both exact and near-duplicate content across text, PDFs, audio, and images. Unlike traditional methods, it uses similarity-based techniques such as Locality Sensitive Hashing (LSH) to perform efficient and scalable comparisons.

The system is built with a Flask backend and provides an interactive dashboard for users to upload, analyze, and manage files.

---

## **KEY FEATURES**

* Multimodal support for Text, PDF, Audio, and Image file formats
* Efficient similarity detection using MinHash and Locality Sensitive Hashing (LSH)
* Cross-format matching to identify duplicate content across different file types
* Automated storage optimization by removing redundant data
* Interactive dashboard to track uploads and duplicate history
* Secure authentication using GitHub OAuth

---

## **TECHNICAL ARCHITECTURE**

| Component         | Description                      |
| ----------------- | -------------------------------- |
| Backend           | Flask (Python)                   |
| Similarity Engine | MinHash, LSH, Jaccard Similarity |
| File Processing   | PyMuPDF, python-docx, Librosa    |
| Database          | SQLAlchemy                       |
| Security          | GitHub OAuth, Session Management |

---

## **FILES**

| File/Folder  | Description                         |
| ------------ | ----------------------------------- |
| `app.py`     | Main Flask application              |
| `utils/`     | Core logic for similarity detection |
| `models.py`  | Database models                     |
| `templates/` | Frontend UI (HTML/Jinja)            |
| `static/`    | CSS and JavaScript files            |

---

## **HOW TO USE**

1. Clone or download this repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Configure GitHub OAuth credentials
4. Run the application:

   ```bash
   python app.py
   ```
5. Upload files through the dashboard to test duplicate detection

---

## **ABOUT ME**

I am a final-year Computer Science (Artificial Intelligence) student with a strong interest in building intelligent systems and scalable applications. This project highlights my ability to design backend systems, implement efficient algorithms, and solve real-world data challenges.

I welcome feedback and collaboration opportunities.
