from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import re  # Regex
import numpy as np
from functools import lru_cache
import time
import json
import threading
import pkg_resources
from symspellpy import SymSpell, Verbosity


load_dotenv()


# ⚙️ App setup

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///alwarraq.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://", 
    default_limits=["200 per day"] 
)


# 📚 Query Expansion
ABBREVIATIONS_MAP = {
    'ai': 'artificial intelligence', 'sql': 'structured query language',
    'ml': 'machine learning', 'nlp': 'natural language processing',
    'html': 'hypertext markup language', 'css': 'cascading style sheets',
    'api': 'application programming interface', 'cpu': 'central processing unit',
    'os': 'operating system', 'iot': 'internet of things',
    'ui': 'user interface', 'ux': 'user experience',
    'db': 'database', 'knn': 'k-nearest neighbors',
    'svm': 'support vector machine', 'ann': 'artificial neural network',
    'cnn': 'convolutional neural network', 'rnn': 'recurrent neural network',
    'ide': 'integrated development environment', 'http': 'hypertext transfer protocol',
    'tcp': 'transmission control protocol', 'ip': 'internet protocol',
    'aws': 'amazon web services', 'saas': 'software as a service',
    'paas': 'platform as a service', 'iaas': 'infrastructure as a service',
    'ds': 'data science', 'cv': 'computer vision',
    'gan': 'generative adversarial network', 'bert': 'bidirectional encoder representations from transformers',
    'gpt': 'generative pre-trained transformer', 'llm': 'large language model'
}



# 1.SymSpell 
print("⏳ Loading SymSpell engine... (occurs only once)")
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Academic dictionary of terminology
CUSTOM_VOCAB = [
    'frontend', 'backend', 'linux', 'queries', 'neural', 'cloud', 'kubernetes', 'cybersecurity',
    'database', 'algorithms', 'blockchain', 'infrastructure', 'network', 'architecture', 
    'cryptography', 'recommendation', 'operating', 'javascript', 'microservices', 'iot',
    'discrete', 'linear', 'algebra', 'matrices', 'probability', 'statistics', 'numerical',
    'calculus', 'quantum', 'mechanics', 'electrical', 'nanotechnology', 'mechanical', 'civil',
    'neuroscience', 'anatomy', 'psychiatry', 'clinical', 'nutrition', 'entrepreneurship', 
    'strategy', 'financial', 'accounting', 'agile', 'cognitive', 'anthropology', 'cultural'
]

custom_words = list(ABBREVIATIONS_MAP.keys()) + CUSTOM_VOCAB
for word in custom_words:
    sym_spell.create_dictionary_entry(word, 999999999)

# Stubborn words (rapid intervention)
HARDCODED_TYPOS = {
    'dat': 'data', 'scince': 'science', 'linar': 'linear', 'discrt': 'discrete',
    'proect': 'project', 'quries': 'queries', 'acounting': 'accounting',
    'algbra': 'algebra', 'mechnical': 'mechanical', 'computr': 'computer',
    'fronend': 'frontend', 'kuberntes': 'kubernetes'
}
print("✅ SymSpell Engine is ready")

@lru_cache(maxsize=2048)
def correct_word(word: str) -> str:
    # 1. Rapid intervention of stubborn words
    if word in HARDCODED_TYPOS:
        return HARDCODED_TYPOS[word]
        
    # 2.Protect shortcuts from modification
    if word in ABBREVIATIONS_MAP:
        return word
        
    # 3. Correction via SymSpell
    suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
        
    return word 

def preprocess_user_query(query):
    if not query:
        return ""

    raw_words = str(query).lower().split()
    corrected_words = [correct_word(w) for w in raw_words]

    expanded_query = []
    for word in corrected_words:
        expanded_query.append(word)
        if word in ABBREVIATIONS_MAP:
            expanded_query.append(ABBREVIATIONS_MAP[word])

    return " ".join(expanded_query)

# 🗄️ DATABASE MODELS
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False) 
    password = db.Column(db.String(100), nullable=False)
    
    # (Relationships)
    interests = db.relationship('Interest', backref='user', lazy=True, cascade="all, delete-orphan")
    favorites = db.relationship('Favorite', backref='user', lazy=True, cascade="all, delete-orphan")


class Interest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    interest_name = db.Column(db.String(100), nullable=False)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    book_title = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 🧠 TFIDFRecommender 
class TFIDFRecommender:
    def __init__(self, models_dir='models'):
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_path, models_dir)

        self.models_dir = models_dir
        self.vectorizer = None
        self.matrix = None
        self.data = None
        self.load_artifacts()

    def load_artifacts(self):
        print("⏳ Recommender: Loading search artifacts...")
        try:
            self.vectorizer = joblib.load(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
            self.matrix = joblib.load(os.path.join(self.models_dir, 'tfidf_matrix.pkl'))
            self.data = joblib.load(os.path.join(self.models_dir, 'books_processed.pkl'))
            if 'category' in self.data.columns:
                self.data['category'] = self.data['category'].fillna('General')

            # 🌟 Fair Tiered Quality Boost 
            desc_lengths = self.data['description'].astype(str).apply(len)
            
            # Tiers
            conditions = [
                (desc_lengths < 50),
                (desc_lengths >= 50) & (desc_lengths <= 200),
                (desc_lengths > 200)
            ]
            
            
            choices = [1.0, 1.05, 1.10]
            
            # apply equation
            self.quality_boost = np.select(conditions, choices, default=1.0)
        except Exception as e:
            print(f"❌ Recommender Error: {e}")

    

    def get_recommendations(self, query, interests=None, top_k=12, boost_factor=0.35):
        if not self.vectorizer or not query:
            return []
        # 1. Basic retrieval
        words = str(query).split()
        if len(words) <= 10:
            search_text = preprocess_user_query(str(query))
        else:
            search_text = str(query).lower()
            expanded = []
            for word in search_text.split():
                expanded.append(word)
                if word in ABBREVIATIONS_MAP:
                    expanded.append(ABBREVIATIONS_MAP[word])
            search_text = " ".join(expanded)
        query_vec = self.vectorizer.transform([search_text])
        final_sim = cosine_similarity(query_vec, self.matrix).flatten()
        # 2.  OPTIMIZED personalization layer
        if interests:
            user_interests_set = set()
            

            if isinstance(interests, str):
                raw_items = [i.strip() for i in interests.split(',')] if ',' in interests else [interests]
            elif hasattr(interests, 'all'):
                raw_items = list(interests.all())
            elif hasattr(interests, '__iter__') and not isinstance(interests, dict):
                raw_items = list(interests)
            else:
                raw_items = [interests]
            

            for item in raw_items:
                if hasattr(item, 'interest_name'):
                    val = str(getattr(item, 'interest_name', '')).lower().strip()
                elif isinstance(item, (tuple, list)) and len(item) > 0:
                    val = str(item[0]).lower().strip()
                else:
                    val = str(item).lower().strip()
                val = val.replace('_', ' ').replace('interest name', '').replace('object', '').strip()
                if val and val not in ('general', 'none', 'nan', ''):
                    for part in val.split(','):
                        clean = part.strip()
                        if clean:
                            user_interests_set.add(clean)
            if user_interests_set:
                max_query_score = np.max(final_sim) if np.max(final_sim) > 0 else 0.5
                dynamic_boost = boost_factor
                #  Vectorized boost calculation 
                categories_lower = self.data['category'].fillna('').str.lower()
                category_boost = np.where(
                    categories_lower.isin(user_interests_set),
                    dynamic_boost * 0.3,
                    0.0
                )
                # Tags boost (vectorized)
                tags_col = self.data.get("tags", self.data.get("category", pd.Series([""] * len(self.data))))
                tags_lower = tags_col.fillna('').astype(str).str.lower()
                # Only once regex pattern for interests Convert the

                interest_pattern = '|'.join(re.escape(i) for i in user_interests_set)
                tag_boost = np.where(
                    tags_lower.str.contains(interest_pattern, case=False, na=False, regex=True),
                    dynamic_boost * 0.5,
                    0.0
                )
                interest_boost = category_boost + tag_boost
                final_sim = final_sim + interest_boost
        # 3. Quality layer (already vectorized - no change needed)
        if 'description' in self.data.columns:
            desc_lengths = self.data['description'].fillna('').str.len()
            quality_boost = np.where(desc_lengths > 200, 1.10,
                            np.where(desc_lengths >= 50, 1.05, 1.0))
            final_sim = final_sim + (quality_boost - 1) * 0.1
        # 4. Normalization
        if np.max(final_sim) > 0:
            final_sim = final_sim / (np.max(final_sim) + 1e-9)
        # 5. Final ranking
        related_indices = final_sim.argsort()[:-top_k-1:-1]
        results = []
        for i in related_indices:
            score = final_sim[i]
            if score > 0:
                book = self.data.iloc[i]
                results.append(self.format_book(book, score))
        return results[:top_k]

    def get_by_category(self, categories_str, limit=12):
        if not categories_str or 'category' not in self.data.columns:
            return self.get_random_books(limit)
        cats = [c.strip().lower() for c in categories_str.split(',')]
        mask = self.data['category'].str.lower().apply(lambda x: any(c in str(x).lower() for c in cats))
        filtered = self.data[mask]
        if filtered.empty: return self.get_random_books(limit)
        return [self.format_book(row) for _, row in filtered.sample(min(len(filtered), limit)).iterrows()]

    def get_book_details(self, title):
        book = self.data[self.data['title'] == title]
        if book.empty: return None
        return self.format_book(book.iloc[0])

    def get_random_books(self, n=12):
        return [self.format_book(row) for _, row in self.data.sample(n).iterrows()]

    def format_book(self, book_row, score=0):
        raw_tags = str(book_row.get('tags',  book_row.get('category', 'General')))
        clean_tags = raw_tags.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
        return {
            'title': book_row.get('title', 'Unknown'),
            'author': book_row.get('authors', 'Unknown'),
            'category': book_row.get('category', 'General'),
            'tags': clean_tags, 
            'description': str(book_row.get('description', ''))[:300] + "...",
            'full_desc': str(book_row.get('description', '')), 
            'score': round(float(score), 2)
        }


# 🤖 CLASS 2: ChatBot 
class ChatBot:
    def __init__(self):
        self.groq_client = None
        self.sambanova_client = None
        self.last_request_time = None
        self.setup_clients()
    def _wait_if_needed(self):
        
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < 2: 
                time.sleep(2 - elapsed)
        self.last_request_time = datetime.now()
    def setup_clients(self):
        print("⏳ ChatBot: Initializing Multi-Cloud System (Groq + SambaNova)...")
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            try:
                self.groq_client = Groq(api_key=groq_key)
                print("✅ ChatBot: Groq API Connected! (Primary)")
            except Exception as e:
                print(f"⚠️ Groq Init Error: {e}")
        else:
            print("⚠️ GROQ_API_KEY not found in .env")

        sambanova_key = os.environ.get("SAMBANOVA_API_KEY")
        if sambanova_key:
            try:
                self.sambanova_client = OpenAI(
                    base_url="https://api.sambanova.ai/v1",
                    api_key=sambanova_key
                )
                print("✅ ChatBot: SambaNova API Connected! (Fallback)")
            except Exception as e:
                print(f"⚠️ SambaNova Init Error: {e}")
        else:
            print("⚠️ SAMBANOVA_API_KEY not found in .env")

    def generate_reply(self, user_msg, context_data, chat_history=None):
        
        title = context_data.get('title', 'Unknown')
        author = context_data.get('author', 'Unknown')
        category = context_data.get('category', 'General')
        tags = context_data.get('tags', 'None')
        desc = str(context_data.get('full_desc', '')).strip()
        similar_books = context_data.get('similar_books', [])

        if similar_books:
            similar_books_text = "\n".join([f"- {book}" for book in similar_books])
        else:
            similar_books_text = "No similar books available at the moment."     

        
        system_prompt = f"""You are "Al-Warraq AI Librarian", an expert and supremely CONFIDENT academic assistant. 
        
        Current Book Details:
        - Title: "{title}"
        - Author: {author}
        - Category: {category}
        - Description: {desc}
        - Similar Books: {similar_books_text}

        CRITICAL RULES (FOLLOW STRICTLY):
        1. NO REPETITIVE OPENINGS: You MUST NOT use repetitive or robotic introductions like "على الرغم من أن الوصف المحفوظ..." or "According to the description". Start answering the user's question DIRECTLY and naturally.
        
        2. ADAPTIVE MULTILINGUAL: You MUST mirror the user's language. If the user asks in English, reply entirely in English. If the user asks in Arabic, reply entirely in high-level academic Arabic. NEVER mix languages unprofessionally.
        
        3. ABSOLUTE CONFIDENCE: Base your answers on the 'Current Book Details'. If asked about specific chapters not in the description, use your pre-trained memory to provide highly probable academic contents for THIS specific book confidently. NEVER use phrases like "I cannot confirm", "guess", "لا أستطيع التأكيد", "بدون وصف أكثر".
        
        4. STRICT RECOMMENDATIONS: For book recommendations, ONLY list books from the 'Similar Books' list. NEVER invent outside books.
        
        5. FORMATTING & READABILITY (CRITICAL): You MUST use Markdown formatting to make your answer highly readable. Use **bold text** for book titles, core concepts, and emphasis. Always use bullet points for any lists. Keep paragraphs short and visually appealing.
        """

        messages_format = [{"role": "system", "content": system_prompt}]
        
        #  chat history
        if chat_history:
            messages_format.extend(chat_history)
            
        messages_format.append({"role": "user", "content": user_msg})

        # Groq
        if self.groq_client:
            try:
                self._wait_if_needed()
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages_format,
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=300,
                    timeout=10
                )
                return chat_completion.choices[0].message.content

            except Exception as e:
                error_msg = str(e).lower()

                # ✅ التعرف على Rate Limit Error
                if "rate" in error_msg or "429" in error_msg or "quota" in error_msg:
                    print("⏳ Groq Rate Limit reached. Waiting 3s then trying SambaNova...")
                    time.sleep(3)
                else:
                    print(f"☁️ Groq Error: {e}. Switching to SambaNova...")

        # SambaNova Fallback
        if self.sambanova_client:
            try:
                self._wait_if_needed()
                print("🔄 Using SambaNova API as Fallback...")
                chat_completion = self.sambanova_client.chat.completions.create(
                    messages=messages_format,
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=300,
                    timeout=10
                )
                return chat_completion.choices[0].message.content

            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "429" in error_msg:
                    return "⏸️ Both AI services are busy right now. Please try again in 1 minute."
                else:
                    print(f"❌ SambaNova Error: {e}")
                    return "Sorry, both AI services are temporarily unavailable."

        return "Sorry, the AI librarian is currently unavailable. Please check API keys."
    

# 🎮 CLASS 3: Flask Server
recommender = TFIDFRecommender()
bot = ChatBot()

@app.route('/', methods=['GET', 'POST'])
def home():
    books = []
    message = ""
    search_query = ""
    selected_mode = "tfidf" 

    # 
    user_interests_list = []
    user_interests_str = ""  # 
    if current_user.is_authenticated and current_user.interests:
        user_interests_list = [i.interest_name for i in current_user.interests]
        user_interests_str = ",".join(user_interests_list)  

    if request.method == 'POST':
        search_query = request.form.get('query', '')
        selected_mode = request.form.get('mode', 'tfidf')
        
        if search_query:
            if current_user.is_authenticated:
                if selected_mode == 'tfidf_interests':
                    if user_interests_list: 
                        books = recommender.get_recommendations(search_query, interests=user_interests_list, boost_factor=0.4)
                    else: 
                     message = "You don't have any interests set up yet. Using standard search."
                     books = recommender.get_recommendations(search_query)
                
                elif selected_mode == 'tfidf_favorites':
                    favs = Favorite.query.filter_by(user_id=current_user.id).all()
                    if favs:
                        fav_titles = [f.book_title for f in favs[:5]]
                        fav_books_batch = recommender.data[recommender.data['title'].isin(fav_titles)]

                        if not fav_books_batch.empty:
                            enriched_parts = [search_query]
                            enriched_parts.extend(fav_books_batch['title'].tolist())
                            enriched_parts.extend(fav_books_batch['description'].fillna('').str[:200].tolist())
                            enriched_query = " ".join(str(p) for p in enriched_parts if p)

                            fav_signals = []
                            valid_cats = fav_books_batch['category'].dropna()
                            valid_cats = valid_cats[valid_cats != 'General']
                            fav_signals.extend(valid_cats.tolist())

                            if 'tags' in fav_books_batch.columns:
                                all_tags = fav_books_batch['tags'].fillna('').astype(str)
                                for tag_str in all_tags:
                                    clean_tags = tag_str.replace("{","").replace("}","").replace("'","").replace('"','')
                                    for tag in clean_tags.split(','):
                                        tag = tag.strip()
                                        if tag and tag.lower() not in ('general', 'none', 'nan', ''):
                                            fav_signals.append(tag)

                            unique_signals = list(set(fav_signals)) if fav_signals else None
                            books = recommender.get_recommendations(enriched_query, interests=unique_signals)
                            if not books:
                                books = recommender.get_recommendations(search_query)
                        else:
                            books = recommender.get_recommendations(search_query)
                    else:
                        message = "You don't have any favorites yet. Using standard search."
                        books = recommender.get_recommendations(search_query)
                else:
                    books = recommender.get_recommendations(search_query)
            else:
                books = recommender.get_recommendations(search_query)
        else:
            message = "Please enter a search term."
            
    else:  
        if current_user.is_authenticated:
            # 1. Attempting to make a recommendation based on preferences
            favs = Favorite.query.filter_by(user_id=current_user.id).all()
            if favs:
                fav_query = " ".join([f.book_title for f in favs])
                books = recommender.get_recommendations(fav_query, top_k=12)
                message = "Recommended for you based on your Favorites"
            
            elif user_interests_str:
                books = recommender.get_by_category(user_interests_str, limit=12)
                message = "Top picks based on your interests"
            
            # If the user is new
            else:
                books = recommender.get_random_books(12)
                message = "Welcome! Explore our collection"
        else:
            # For the guest user
            books = recommender.get_random_books(12)
            message = "Discover our popular books"

    return render_template('home.html', 
                           recommendations=books, 
                           search_query=search_query, 
                           selected_mode=selected_mode, 
                           message=message)


@app.route('/book/<path:title>')
def book_details(title):
    import urllib.parse
    decoded_title = urllib.parse.unquote(title)
    book = recommender.get_book_details(decoded_title)
    if not book: return "Book Not Found", 404
    
    # Check the status of favorites
    book_is_favorite = False
    if current_user.is_authenticated:
        book_is_favorite = any(fav.book_title == book['title'] for fav in current_user.favorites)
    
    raw_similar = recommender.get_recommendations(book['description'], top_k=7)
    similar_books = [b for b in raw_similar if b['title'] != book['title']][:6]
    
    return render_template('book_details.html', 
                           book=book, 
                           similar_books=similar_books, 
                           book_is_favorite=book_is_favorite)

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute", methods=["POST"])
def login():
    if request.method == 'POST':
        email = request.form.get('email') 
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first() 
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST']) 
def register():
    if request.method == 'POST':
        email = request.form.get('email') 
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # 🛡️ Regex
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_regex, email):
            flash('Invalid email format! Please enter a valid email.', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first(): 
            flash('Email already exists', 'error')
            return redirect(url_for('register'))
            
        new_user = User(
            email=email, 
            password=generate_password_hash(password, method='scrypt')
        )
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('interests'))
        
    return render_template('register.html')

@app.route('/favorites')
@login_required
def favorites():
    favs = Favorite.query.filter_by(user_id=current_user.id).all()
    books = []
    for f in favs:
        details = recommender.get_book_details(f.book_title)
        if details: books.append(details)
    return render_template('favorites.html', favorite_books=books)

@app.route('/interests', methods=['GET', 'POST'])
@login_required
def interests():
    categories = sorted(recommender.data['category'].dropna().unique().astype(str))[:30]
    
    if request.method == 'POST':
        selected = request.form.getlist('interests')
        # Use set() to automatically delete any duplicate values
        unique_selected = list(set(selected)) 
        
        Interest.query.filter_by(user_id=current_user.id).delete()
        for item in unique_selected:
            new_interest = Interest(interest_name=item, user_id=current_user.id)
            db.session.add(new_interest)
            
        db.session.commit()
        flash('Preferences saved successfully!', 'success')
        return redirect(url_for('home'))
        
    current_interests = [i.interest_name for i in current_user.interests] if current_user.interests else []
    
    return render_template('interests.html', 
                           available_interests=categories, 
                           saved_interests=current_interests)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/api/search', methods=['POST'])
def search_api():
    data = request.json
    query = data.get('query', '')
    use_interests = data.get('use_interests', False)
    
    user_interests = None
    if current_user.is_authenticated and use_interests and current_user.interests:
        
        user_interests = [i.interest_name for i in current_user.interests]

    results = recommender.get_recommendations(query, interests=user_interests)
    return jsonify(results)

@app.route('/api/chat', methods=['POST'])
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    user_msg = data.get('message', '').strip() 
    context_data = data.get('context', {}) 
    
    #  Protection 1: Block empty messages
    if not user_msg:
        return jsonify({'response': 'Sorry, I cannot respond to an empty message.'})
        
    if 'chat_history' not in session:
        session['chat_history'] = []
        
    #  Protection 2: Handle unexpected network failures
    try:
        response = bot.generate_reply(
            user_msg=user_msg, 
            context_data=context_data, 
            chat_history=session['chat_history']
        )
    except Exception as e:
        print(f"❌ Route Error: {e}")
        response = "Sorry, there is pressure on the Warraq library at the moment. Please try again later."
    
    
    session['chat_history'].append({"role": "user", "content": user_msg})
    session['chat_history'].append({"role": "assistant", "content": response})
    
    if len(session['chat_history']) > 6:
        session['chat_history'] = session['chat_history'][-6:]
        
    session.modified = True

    return jsonify({'response': response})

@app.route('/add_favorite/<path:title>')
@login_required
def add_favorite(title):
    import urllib.parse
    decoded_title = urllib.parse.unquote(title)
    existing_fav = Favorite.query.filter_by(user_id=current_user.id, book_title=decoded_title).first()
    
    if not existing_fav:
        new_fav = Favorite(user_id=current_user.id, book_title=decoded_title)
        db.session.add(new_fav)
        db.session.commit()
        flash('Book added to favorites successfully!', 'success')
    else:
        flash('This book is already in your favorites.', 'info')
        
    return redirect(url_for('book_details', title=title))

@app.route('/remove_favorite/<path:title>')
@login_required
def remove_favorite(title):
    import urllib.parse
    decoded_title = urllib.parse.unquote(title)
    
    fav_book = Favorite.query.filter_by(user_id=current_user.id, book_title=decoded_title).first()
    
    if fav_book:
        db.session.delete(fav_book)
        db.session.commit()
        flash('The book has been deleted from favorites!', 'success')
    else:
        flash('The book is not in your favorites.', 'error')
        
    return redirect(request.referrer or url_for('favorites'))

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    # 1. Verify the current password
    if not check_password_hash(current_user.password, current_password):
        flash('The current password is incorrect', 'error')
        return redirect(url_for('profile'))

    # 2.Verify that the new password matches
    if new_password != confirm_password:
        flash('The new password does not match', 'error')
        return redirect(url_for('profile'))

    # 3.update password
    current_user.password = generate_password_hash(new_password, method='scrypt')
    db.session.commit()
    
    flash('Password changed successfully!', 'success')
    return redirect(url_for('profile'))

@app.errorhandler(429)
def ratelimit_handler(e):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "error": "Too many attempts! Please wait a minute before trying again."
        }), 429
    else:
        flash('Too many attempts! For security reasons, please wait a minute.', 'error')
        return redirect(request.url)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run( debug=True, port=5000)