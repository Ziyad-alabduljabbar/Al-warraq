# 📚 Al-Warraq | الوراق
### Academic Book Recommender System for University Students

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-al--warraq.onrender.com-blue)](https://al-warraq.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-Al--warraq-black?logo=github)](https://github.com/Ziyad-alabduljabbar/Al-warraq)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-black?logo=flask)](https://flask.palletsprojects.com)

---

## 🔍 About

**Al-Warraq** is a web-based academic book recommender system designed specifically for university students. It helps students discover relevant academic resources efficiently using **Content-Based Filtering** powered by **TF-IDF** and **Cosine Similarity**.

The system also features an **AI-powered chatbot librarian** that helps users understand book content, and personalized recommendations based on user interests and favorites.

---

## ✨ Features

- 🔎 **Smart Search** — TF-IDF + Cosine Similarity recommendation engine
- 🧠 **NLP Spell Correction** — SymSpell-powered query error correction with academic vocabulary
- 🎯 **Personalized Recommendations** — Based on user interests and favorite books
- 🤖 **AI Chatbot Librarian** — Powered by Groq (primary) and SambaNova (fallback) APIs
- ⭐ **Favorites Management** — Save and manage favorite books
- 👤 **User Authentication** — Secure registration, login, and password management
- 🛡️ **Rate Limiting** — Brute-force protection with Flask-Limiter

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML Engine | TF-IDF, Cosine Similarity, scikit-learn |
| NLP | SymSpell, Query Expansion |
| AI Chatbot | Groq API (llama-3.3-70b), SambaNova API |
| Database | SQLite, Flask-SQLAlchemy |
| Frontend | HTML, CSS, JavaScript |


---

## 📊 Performance

| Metric | Result |
|--------|--------|
| Favorites Hit Rate @12 | 58.3% (+26.8pp vs Standard) |
| Interests Hit Rate @12 | 52.8% (+21.4pp vs Standard) |
| Avg Query Latency | < 35ms |
| Concurrent Users (15) | 3.77× degradation — PASS |
| Chatbot Pass Rate | 5/5 (100%) |

---

## 🚀 Live Demo

👉 **[https://al-warraq.onrender.com](https://al-warraq.onrender.com)**

> Note: The app may take 30-60 seconds to wake up on first visit (free tier).

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Ziyad-alabduljabbar/Al-warraq.git
cd Al-warraq

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "SECRET_KEY=your_secret_key" > .env
echo "GROQ_API_KEY=your_groq_key" >> .env
echo "SAMBANOVA_API_KEY=your_sambanova_key" >> .env

# Run the app
python app.py
```

---

## 👥 Team

| Name |
|------|
| Abdulmohsen Aljomah |
| Abdulaziz Alhoshany |
| Ziyad Othman Al-Abduljabbar |
| Faisal Alahheadeb |
| Abdulrahman Alamer |
| Mohammed Ahmad Alghanam |

**Supervisor:** Dr. Ahmed Shahin  
**Institution:** Majmaah University — College of Computer and Information Sciences  
**Year:** 2026
