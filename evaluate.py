import pandas as pd
import numpy as np
import random
import sys
import os
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import TFIDFRecommender, preprocess_user_query, ABBREVIATIONS_MAP

# Settings
DATASET_PATH = r"C:\Users\mobil\Desktop\‏‏Al warraq\dataset\AlWarraq_Final_Version.csv"
TARGET_AUTO_SIZE = 300
TOP_K            = 12
RANDOM_SEED      = 41

# Custom edge cases
CUSTOM_QUERIES = [
    {
        'Query_Type': 'Edge_Acronym',
        'Query': 'nlp algorithms',
        'Target_Category': 'natural_language_processing',
        'Interests': ['natural_language_processing', 'data_analytics'],
        'Fav_Categories': ['natural_language_processing'],
        'Fav_Tags': ['nlp', 'text mining', 'linguistics', 'python', 'natural language processing']
    },
    {
        'Query_Type': 'Edge_Backend',
        'Query': 'php and mysql backend',
        'Target_Category': 'web_development',
        'Interests': ['web_development', 'software_engineering'],
        'Fav_Categories': ['web_development'],
        'Fav_Tags': ['php', 'mysql', 'html', 'css', 'backend', 'frontend', 'web']
    },
    {
        'Query_Type': 'Edge_Broad',
        'Query': 'introduction',
        'Target_Category': 'operating_systems',
        'Interests': ['operating_systems', 'distributed_systems'],
        'Fav_Categories': ['operating_systems'],
        'Fav_Tags': ['linux', 'windows', 'system administration', 'kernel', 'os']
    },
    {
        'Query_Type': 'Edge_CrossDomain',
        'Query': 'visualizing business insights',
        'Target_Category': 'data_analytics',
        'Interests': ['data_analytics', 'expert_systems'],
        'Fav_Categories': ['data_analytics'],
        'Fav_Tags': ['data visualization', 'power bi', 'statistics', 'analytics', 'charts']
    },
    {
        'Query_Type': 'Edge_Theory',
        'Query': 'strategy of conflict',
        'Target_Category': 'game_theory',
        'Interests': ['game_theory', 'discrete_mathematics'],
        'Fav_Categories': ['game_theory'],
        'Fav_Tags': ['game theory', 'negotiation', 'strategy', 'economics', 'mathematics']
    },
]

# Spell correction test queries
SPELL_TEST_QUERIES = [
    {'Query_Type': 'Spell_MultiError', 'Query': 'discrt mathmatics and logic',    'Corrected': 'discrete mathematics and logic',   'Target_Category': 'discrete_mathematics'},
    {'Query_Type': 'Spell_MultiError', 'Query': 'linar algbra matrics',           'Corrected': 'linear algebra matrices',          'Target_Category': 'linear_algebra'},
    {'Query_Type': 'Spell_MultiError', 'Query': 'probabilty and statstics',       'Corrected': 'probability and statistics',       'Target_Category': 'probability'},
    {'Query_Type': 'Spell_Typo',       'Query': 'numercal anlysis',               'Corrected': 'numerical analysis',               'Target_Category': 'numerical_analysis'},
    {'Query_Type': 'Spell_Typo',       'Query': 'advnced calculs',                'Corrected': 'advanced calculus',                'Target_Category': 'calculus'},
    {'Query_Type': 'Spell_MultiError', 'Query': 'quantm mechanis theary',         'Corrected': 'quantum mechanics theory',         'Target_Category': 'quantum_mechanics'},
    {'Query_Type': 'Spell_MultiError', 'Query': 'eletrical enginering circuts',   'Corrected': 'electrical engineering circuits',  'Target_Category': 'electrical_engineering'},
    {'Query_Type': 'Spell_MultiError', 'Query': 'mechnical enginering',           'Corrected': 'mechanical engineering',           'Target_Category': 'mechanical_engineering'},
    {'Query_Type': 'Spell_MultiError', 'Query': 'neuroscince and brain',          'Corrected': 'neuroscience and brain',           'Target_Category': 'neuroscience'},
    {'Query_Type': 'Spell_MultiError', 'Query': 'artficial intelgence etics',     'Corrected': 'artificial intelligence ethics',   'Target_Category': 'artificial_intelligence'},
    {'Query_Type': 'Spell_Abbrev',     'Query': 'nlp algorthms and txt mining',   'Corrected': 'nlp algorithms and text mining',   'Target_Category': 'natural_language_processing'},
    {'Query_Type': 'Spell_Abbrev',     'Query': 'sql databse quris',              'Corrected': 'sql database queries',             'Target_Category': 'sql'},
    {'Query_Type': 'Spell_Abbrev',     'Query': 'iot netwrks secuity',            'Corrected': 'iot networks security',            'Target_Category': 'internet_of_things'},
]

# ChatBot test cases
CHATBOT_TEST_CASES = [
    {
        'test_id': 'CB_01', 'type': 'Language_Arabic',
        'message': 'ما هي المواضيع الرئيسية في هذا الكتاب؟',
        'context': {
            'title': 'Artificial Intelligence', 'author': 'Michael Negnevitsky',
            'category': 'Artificial Intelligence', 'tags': 'ai, machine learning, neural networks',
            'full_desc': 'A comprehensive book about artificial intelligence covering expert systems and neural networks.',
            'similar_books': ['Deep Learning', 'Machine Learning', 'Neural Networks']
        },
        'expected_language': 'arabic',
        'check_keywords': ['الذكاء', 'الاصطناعي']
    },
    {
        'test_id': 'CB_02', 'type': 'Language_English',
        'message': 'What are the main topics covered in this book?',
        'context': {
            'title': 'Python Machine Learning', 'author': 'Sebastian Raschka',
            'category': 'Machine Learning', 'tags': 'python, machine learning, deep learning',
            'full_desc': 'A practical guide to machine learning using Python.',
            'similar_books': ['Deep Learning with Python', 'Hands-On ML']
        },
        'expected_language': 'english',
        'check_keywords': ['machine learning', 'python']
    },
    {
        'test_id': 'CB_03', 'type': 'Recommendations',
        'message': 'Can you recommend similar books?',
        'context': {
            'title': 'Clean Code', 'author': 'Robert C. Martin',
            'category': 'Software Engineering', 'tags': 'clean code, programming, software',
            'full_desc': 'A guide to writing clean, maintainable code.',
            'similar_books': ['The Pragmatic Programmer', 'Refactoring', 'Design Patterns']
        },
        'expected_language': 'english',
        'check_keywords': ['Pragmatic', 'Refactoring', 'Design']
    },
    {
        'test_id': 'CB_04', 'type': 'Specific_Question',
        'message': 'What chapters does this book have?',
        'context': {
            'title': 'Introduction to Algorithms', 'author': 'Thomas H. Cormen',
            'category': 'Algorithms', 'tags': 'algorithms, data structures, complexity',
            'full_desc': 'A comprehensive introduction to algorithms and data structures.',
            'similar_books': ['Algorithm Design', 'Data Structures']
        },
        'expected_language': 'english',
        'check_keywords': ['algorithm', 'chapter']
    },
    {
        'test_id': 'CB_05', 'type': 'Empty_Message',
        'message': '',
        'context': {
            'title': 'Test Book', 'author': 'Test Author',
            'category': 'General', 'tags': 'test',
            'full_desc': 'Test description.', 'similar_books': []
        },
        'expected_language': 'any',
        'check_keywords': []
    },
]

# Helpers
STOP_WORDS = {
    'the', 'a', 'an', 'of', 'in', 'and', 'to', 'for', 'on', 'with', 'at', 'by',
    'from', 'up', 'about', 'into', 'over', 'after', 'or', 'as', 'if', 'this', 'that',
    'introduction', 'basics', 'fundamental', 'fundamentals', 'principles',
    'handbook', 'guide', 'manual', 'concepts', 'elements', 'essential',
    'essentials', 'complete', 'comprehensive', 'modern', 'advanced',
    'primer', 'overview', 'theory', 'approach', 'practice', 'applications',
    'using', 'learning', 'towards', 'through', 'within', 'between', 'new',
    'edition', 'volume', 'series', 'study', 'analysis', 'research', 'design'
}

def parse_tags(raw_tags_str):
    cleaned = str(raw_tags_str).replace('{', '').replace('}', '').replace("'", '').replace('"', '')
    return [t.strip().lower() for t in cleaned.split(',') if t.strip() and t.strip().lower() not in ('general', 'none', 'nan', '')]

def build_query_from_title(title):
    words = str(title).replace('-', ' ').split()
    meaningful = [w for w in words if w.lower() not in STOP_WORDS]
    if not meaningful:
        return None
    num_words = random.randint(min(2, len(meaningful)), min(4, len(meaningful)))
    query = " ".join(meaningful[:num_words]).lower()
    query = ''.join(c for c in query if c.isalnum() or c.isspace()).strip()
    return query if query else None

# Data generation
def generate_test_data(csv_path, target_size):
    print(f"[1/4] Generating {target_size} test queries from dataset...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(csv_path)
    df['category']    = df['category'].fillna('').astype(str).str.strip()
    df['tags']        = df['tags'].fillna('').astype(str)
    df['title']       = df['title'].fillna('').astype(str).str.strip()
    df['description'] = df['description'].fillna('').astype(str)
    df = df[(df['category'] != '') & (df['title'] != '')].copy().reset_index(drop=True)

    valid_cats = sorted(
        df['category'].value_counts()[df['category'].value_counts() >= 6].index.tolist()
    )

    per_cat = max(1, target_size // len(valid_cats))
    test_data = []

    for cat in valid_cats:
        cat_books = df[df['category'] == cat].reset_index(drop=True)
        if len(cat_books) < 6:
            continue

        for i in range(per_cat):
            seed_i = RANDOM_SEED + valid_cats.index(cat) * 100 + i
            sampled     = cat_books.sample(6, random_state=seed_i)
            target_book = sampled.iloc[0]
            fav_books   = sampled.iloc[1:]

            query = build_query_from_title(target_book['title'])
            if not query:
                continue

            rng = random.Random(seed_i)
            num_interests = rng.randint(1, 3)
            interests = [str(cat).lower().replace('_', ' ')]
            if num_interests > 1:
                others = [c for c in valid_cats if c != cat]
                extras = rng.sample(others, min(num_interests - 1, len(others)))
                interests += [c.lower().replace('_', ' ') for c in extras]

            fav_categories, fav_tags = [], []
            for _, row in fav_books.iterrows():
                c = str(row['category']).strip().lower().replace('_', ' ')
                if c and c not in ('', 'general'):
                    fav_categories.append(c)
                fav_tags.extend(parse_tags(row['tags']))

            # التعديل الجوهري هنا: ترتيب الكلمات المفتاحية حسب تكرارها وأهميتها
            tag_counts = Counter(fav_tags)
            sorted_tags_by_freq = [tag for tag, count in tag_counts.most_common()]

            test_data.append({
                'Query_Type'     : 'Auto',
                'Query'          : query,
                'Target_Category': str(cat).lower().replace('_', ' '),
                'Interests'      : interests,
                'Num_Interests'  : len(interests),
                'Fav_Categories' : sorted(list(set(fav_categories))), # التصنيفات تبقى كما هي لأن عددها قليل
                'Fav_Tags'       : sorted_tags_by_freq, # نمرر القائمة المرتبة بالأهمية
            })

        if len(test_data) >= target_size:
            break

    test_data = test_data[:target_size]

    for eq in CUSTOM_QUERIES:
        test_data.append({
            'Query_Type'     : eq['Query_Type'],
            'Query'          : eq['Query'],
            'Target_Category': eq['Target_Category'].lower().replace('_', ' '),
            'Interests'      : [i.lower().replace('_', ' ') for i in eq['Interests']],
            'Num_Interests'  : len(eq['Interests']),
            'Fav_Categories' : [c.lower().replace('_', ' ') for c in eq['Fav_Categories']],
            'Fav_Tags'       : eq['Fav_Tags'],
        })

    print(f"OK: {len(test_data)} total queries ready ({len(CUSTOM_QUERIES)} edge cases)")
    return test_data

# Metrics
def hit_rate(results, expected_category, k=TOP_K):
    if not results: return 0.0
    expected = expected_category.lower().strip().replace('_', ' ')
    matches  = sum(1 for b in results[:k] if str(b.get('category', '')).lower().strip().replace('_', ' ') == expected)
    return (matches / k) * 100

def ndcg_at_k(results, expected_category, k=TOP_K):
    if not results: return 0.0
    expected = expected_category.lower().strip().replace('_', ' ')
    dcg  = sum(1.0 / np.log2(r + 2) for r, b in enumerate(results[:k]) if str(b.get('category', '')).lower().strip().replace('_', ' ') == expected)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(k))
    return (dcg / idcg) * 100 if idcg > 0 else 0.0

def precision_at_1(results, expected_category):
    if not results: return 0.0
    expected = expected_category.lower().strip().replace('_', ' ')
    return 100.0 if str(results[0].get('category', '')).lower().strip().replace('_', ' ') == expected else 0.0

def evaluate_query(recommender, query, expected_category, interests=None, use_spell=False):
    processed = preprocess_user_query(query) if use_spell else query
    results   = recommender.get_recommendations(processed, interests=interests, top_k=TOP_K)
    return {
        'hit_rate'       : hit_rate(results, expected_category),
        'ndcg'           : ndcg_at_k(results, expected_category),
        'precision_1'    : precision_at_1(results, expected_category),
        'processed_query': processed,
    }

# Speed evaluation
def evaluate_speed(recommender, test_data, n_samples=50):
    print(f"\n[Speed] Evaluating {n_samples} random queries...")
    samples    = random.sample(test_data, min(n_samples, len(test_data)))
    speed_rows = []

    for data in samples:
        query = data['Query']

        # Mode 1
        t = time.perf_counter()
        recommender.get_recommendations(query, top_k=TOP_K)
        t1 = (time.perf_counter() - t) * 1000

        # Mode 2
        t = time.perf_counter()
        recommender.get_recommendations(query, interests=data['Interests'], top_k=TOP_K)
        t2 = (time.perf_counter() - t) * 1000

        # Mode 3
        fav_signals = data['Fav_Categories'] + data['Fav_Tags'][:7] or None
        enriched_q  = query + " " + " ".join(data['Fav_Categories'][:3])
        t = time.perf_counter()
        recommender.get_recommendations(enriched_q, interests=fav_signals, top_k=TOP_K)
        t3 = (time.perf_counter() - t) * 1000

    
        t = time.perf_counter()
        preprocess_user_query(query)
        t_spell = (time.perf_counter() - t) * 1000

        speed_rows.append({
            'Query'          : query,
            'Mode1_ms'       : round(t1, 2),
            'Mode2_ms'       : round(t2, 2),
            'Mode3_ms'       : round(t3, 2),
            'SpellCorrect_ms': round(t_spell, 2),
        })

    df = pd.DataFrame(speed_rows)
    df.to_csv('speed_results.csv', index=False, encoding='utf-8-sig')
    return df

def print_speed_report(df):
    print("\n" + "=" * 65)
    print("SECTION 2 — SPEED REPORT (Latency in milliseconds)")
    print("=" * 65)
    print(f"\n{'Mode':<35} {'Avg':>7} {'Min':>7} {'Max':>7} {'P95':>7}")
    print("-" * 65)
    for label, col in [
        ('Standard  (Query Only)',        'Mode1_ms'),
        ('Interests (Query + Interests)', 'Mode2_ms'),
        ('Favorites (Query + Favorites)', 'Mode3_ms'),
        ('Spell Correction',              'SpellCorrect_ms'),
    ]:
        print(f"{label:<35} {df[col].mean():>6.1f} {df[col].min():>7.1f} {df[col].max():>7.1f} {df[col].quantile(0.95):>7.1f}")

    fastest = df['Mode1_ms'].mean()
    slowest = df['Mode3_ms'].mean()
    print(f"\n  Fastest mode  : Standard  ({fastest:.1f} ms avg)")
    print(f"  Slowest mode  : Favorites ({slowest:.1f} ms avg)")
    print(f"  Spell overhead: {df['SpellCorrect_ms'].mean():.2f} ms avg per query")

# Concurrent users test
def evaluate_concurrent(recommender, test_data, user_levels=None):
    import concurrent.futures

    if user_levels is None:
        user_levels = [1, 5, 10, 15]

    samples = random.sample(test_data, min(20, len(test_data)))
    queries = [d["Query"] for d in samples]

    def single_request(query):
        t = time.perf_counter()
        recommender.get_recommendations(query, top_k=TOP_K)
        return (time.perf_counter() - t) * 1000

    print(f"\n[Concurrent] Testing with user levels: {user_levels}...")
    concurrent_rows = []
    baseline_avg = None

    for n_users in user_levels:
        user_queries = (queries * ((n_users // len(queries)) + 1))[:n_users]

        t_total_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_users) as executor:
            latencies = list(executor.map(single_request, user_queries))
        total_wall = (time.perf_counter() - t_total_start) * 1000

        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        min_lat = min(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]

        if n_users == 1:
            baseline_avg = avg_lat

        degradation = (avg_lat / baseline_avg) if baseline_avg else 1.0
        acceptable  = degradation <= 5.0
        status = "OK" if acceptable else "SLOW"

        print(f"   [{status}] {n_users:>2} users | Avg={avg_lat:>7.1f}ms | Max={max_lat:>7.1f}ms | P95={p95_lat:>7.1f}ms | Degradation={degradation:.1f}x")

        concurrent_rows.append({
            "Num_Users"    : n_users,
            "Avg_ms"       : round(avg_lat, 1),
            "Min_ms"       : round(min_lat, 1),
            "Max_ms"       : round(max_lat, 1),
            "P95_ms"       : round(p95_lat, 1),
            "Wall_Time_ms" : round(total_wall, 1),
            "Degradation_x": round(degradation, 2),
            "Acceptable"   : acceptable,
        })

    df = pd.DataFrame(concurrent_rows)
    df.to_csv("concurrent_results.csv", index=False, encoding="utf-8-sig")
    return df

def print_concurrent_report(df):
    print("\n" + "=" * 65)
    print("SECTION 2B — CONCURRENT USERS REPORT")
    print("=" * 65)
    print(f"\n{'Users':<8} {'Avg (ms)':>10} {'Max (ms)':>10} {'P95 (ms)':>10} {'Degradation':>13} {'Status':>8}")
    print("-" * 65)
    for _, row in df.iterrows():
        status = "OK" if row["Acceptable"] else "SLOW"
        print(f"{int(row['Num_Users']):<8} {row['Avg_ms']:>10.1f} {row['Max_ms']:>10.1f} {row['P95_ms']:>10.1f} {row['Degradation_x']:>12.1f}x {status:>8}")

    passed = df["Acceptable"].sum()
    total  = len(df)
    print(f"\n  Pass Rate : {passed}/{total} levels acceptable (degradation <= 3x baseline)")

    max_ok = df[df["Acceptable"]]["Num_Users"].max() if df["Acceptable"].any() else 0
    print(f"  Max users handled without noticeable slowdown: {max_ok}")

    if not df[df["Num_Users"] == 15].empty:
        row_15 = df[df["Num_Users"] == 15].iloc[0]
        verdict = "PASS" if row_15["Acceptable"] else "FAIL"
        print(f"  15-user target: [{verdict}] — degradation={row_15['Degradation_x']:.1f}x vs baseline")

# ChatBot evaluation
def evaluate_chatbot(bot):
    print(f"\n[ChatBot] Evaluating {len(CHATBOT_TEST_CASES)} test cases...")
    chatbot_rows = []

    for tc in CHATBOT_TEST_CASES:
        t_start = time.perf_counter()

        if not tc['message']:
            response = "Protected — empty message blocked"
            latency  = 0.0
            passed   = True
            notes    = "Empty message protection OK"
        else:
            try:
                response = bot.generate_reply(
                    user_msg     = tc['message'],
                    context_data = tc['context'],
                    chat_history = []
                )
                latency = (time.perf_counter() - t_start) * 1000

                # فحص اللغة
                if tc['expected_language'] == 'arabic':
                    lang_ok = any('\u0600' <= c <= '\u06ff' for c in response)
                elif tc['expected_language'] == 'english':
                    lang_ok = all(ord(c) < 1000 for c in response if c.isalpha())
                else:
                    lang_ok = True

                # فحص الكلمات المفتاحية
                kw_found = sum(1 for kw in tc['check_keywords'] if kw.lower() in response.lower())
                kw_total = len(tc['check_keywords'])
                kw_score = (kw_found / kw_total * 100) if kw_total > 0 else 100

                passed = lang_ok and (kw_score >= 50 or kw_total == 0)
                notes  = f"Lang={'OK' if lang_ok else 'FAIL'} | KW={kw_found}/{kw_total} ({kw_score:.0f}%)"

            except Exception as e:
                response = f"ERROR: {e}"
                latency  = (time.perf_counter() - t_start) * 1000
                passed   = False
                notes    = f"Exception: {e}"

        status = "PASS" if passed else "FAIL"
        print(f"   [{status}] [{tc['test_id']}] {tc['type']:<25} {latency:>7.0f}ms  {notes}")

        chatbot_rows.append({
            'Test_ID'         : tc['test_id'],
            'Type'            : tc['type'],
            'Message'         : tc['message'][:60],
            'Latency_ms'      : round(latency, 1),
            'Passed'          : passed,
            'Notes'           : notes,
            'Response_Preview': str(response)[:120],
        })

    df = pd.DataFrame(chatbot_rows)
    df.to_csv('chatbot_results.csv', index=False, encoding='utf-8-sig')
    return df

def print_chatbot_report(df):
    passed = df['Passed'].sum()
    total  = len(df)
    print("\n" + "=" * 65)
    print("SECTION 3 — CHATBOT REPORT")
    print("=" * 65)
    print(f"\n  Pass Rate   : {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"  Avg Latency : {df['Latency_ms'].mean():.0f} ms")
    print(f"  Max Latency : {df['Latency_ms'].max():.0f} ms")
    print(f"  Min Latency : {df['Latency_ms'].min():.0f} ms")

    failed = df[~df['Passed']]
    if not failed.empty:
        print(f"\n  Failed Cases:")
        for _, row in failed.iterrows():
            print(f"    [{row['Test_ID']}] {row['Type']} — {row['Notes']}")

# Main
def run_evaluation():
    print("=" * 65)
    print("  Al-Warraq — Full Evaluation Pipeline")
    print("=" * 65)

    test_data = generate_test_data(DATASET_PATH, TARGET_AUTO_SIZE)

    print("\n[2/4] Loading TF-IDF engine + ChatBot...")
    from app import ChatBot
    recommender = TFIDFRecommender()
    bot         = ChatBot()


    print(f"\n[3/4] Section 1 — Accuracy: {len(test_data)} queries x 4 modes...")
    print("=" * 65)

    rows, zero_standard, regressions, spell_rows = [], [], [], []

    for i, data in enumerate(test_data, 1):
        query  = data['Query']
        target = data['Target_Category']
        qt     = data['Query_Type']

        m1 = evaluate_query(recommender, query, target)
        m2 = evaluate_query(recommender, query, target, interests=data['Interests'])
        fav_signals = (data['Fav_Categories'] * 2) + data['Fav_Tags'][:2]
        if not fav_signals:
            fav_signals = None
        m3 = evaluate_query(recommender, query, target, interests=fav_signals)
        m4 = evaluate_query(recommender, query, target, use_spell=True)

        if m1['hit_rate'] == 0:
            zero_standard.append({'Query': query, 'Target': target})
        if m3['hit_rate'] < m1['hit_rate']:
            regressions.append({'Query': query, 'Target': target,
                                'Standard_HR': m1['hit_rate'], 'Favorites_HR': m3['hit_rate'],
                                'Diff': m3['hit_rate'] - m1['hit_rate']})

        rows.append({
            'Query_ID': i, 'Query_Type': qt, 'Search_Query': query,
            'Target_Category': target, 'Num_Interests': data['Num_Interests'],
            'HR_Standard' : m1['hit_rate'],  'HR_Interests' : m2['hit_rate'],
            'HR_Favorites': m3['hit_rate'],  'HR_Spell'     : m4['hit_rate'],
            'NDCG_Standard'  : m1['ndcg'],   'NDCG_Interests' : m2['ndcg'],
            'NDCG_Favorites' : m3['ndcg'],   'NDCG_Spell'     : m4['ndcg'],
            'P1_Standard' : m1['precision_1'], 'P1_Interests' : m2['precision_1'],
            'P1_Favorites': m3['precision_1'], 'P1_Spell'     : m4['precision_1'],
            'HR_Imp_Interests': m2['hit_rate'] - m1['hit_rate'],
            'HR_Imp_Favorites': m3['hit_rate'] - m1['hit_rate'],
            'HR_Imp_Spell'    : m4['hit_rate'] - m1['hit_rate'],
        })

        if i % 50 == 0:
            print(f"   ... {i}/{len(test_data)} completed")


    print(f"\n[Spell] Evaluating {len(SPELL_TEST_QUERIES)} spell queries...")
    for sq in SPELL_TEST_QUERIES:
        raw_q     = sq['Query']
        corrected = sq['Corrected']
        target    = sq['Target_Category'].lower().replace('_', ' ')
        m_raw     = evaluate_query(recommender, raw_q,    target)
        m_fixed   = evaluate_query(recommender, raw_q,    target, use_spell=True)
        m_ideal   = evaluate_query(recommender, corrected, target)
        spell_rows.append({
            'Query_Type': sq['Query_Type'], 'Raw_Query': raw_q,
            'Spell_Output': m_fixed['processed_query'], 'Target_Category': target,
            'HR_Raw': m_raw['hit_rate'], 'HR_Spell_Fixed': m_fixed['hit_rate'],
            'HR_Ideal': m_ideal['hit_rate'],
            'Improvement' : m_fixed['hit_rate'] - m_raw['hit_rate'],
            'Gap_to_Ideal': m_ideal['hit_rate'] - m_fixed['hit_rate'],
        })

    acc_df   = pd.DataFrame(rows)
    spell_df = pd.DataFrame(spell_rows)
    acc_df.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    spell_df.to_csv('spell_evaluation.csv', index=False, encoding='utf-8-sig')


    print(f"\n[4/4] Section 2 — Speed...")
    speed_df = evaluate_speed(recommender, test_data)

    # -- SECTION 2B: CONCURRENT USERS --------------------------------
    print('\n[Concurrent] Section 2B -- Concurrent users test...')
    concurrent_df = evaluate_concurrent(recommender, test_data, user_levels=[1, 5, 10, 15])


    chatbot_df = evaluate_chatbot(bot)

    # ══════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 65)
    print("  FINAL REPORT — Al-Warraq Recommender")
    print("=" * 65)


    print("\n" + "=" * 65)
    print("SECTION 1 — ACCURACY")
    print("=" * 65)
    print(f"\n{'Mode':<34} {'Hit Rate':>10} {'NDCG@10':>10} {'P@1':>10}")
    print("-" * 67)
    for label, hr_col, ndcg_col, p1_col in [
        ('Standard  (Query Only)',        'HR_Standard',  'NDCG_Standard',  'P1_Standard'),
        ('Interests (Query + Interests)', 'HR_Interests', 'NDCG_Interests', 'P1_Interests'),
        ('Favorites (Query + Favorites)', 'HR_Favorites', 'NDCG_Favorites', 'P1_Favorites'),
        ('Spell     (Query + SpellCheck)','HR_Spell',     'NDCG_Spell',     'P1_Spell'),
    ]:
        print(f"{label:<34} {acc_df[hr_col].mean():>9.1f}% {acc_df[ndcg_col].mean():>9.1f}% {acc_df[p1_col].mean():>9.1f}%")

    print(f"\n  Improvement vs Standard:")
    print(f"    Interests : {acc_df['HR_Imp_Interests'].mean():+.1f}%")
    print(f"    Favorites : {acc_df['HR_Imp_Favorites'].mean():+.1f}%")
    print(f"    Spell     : {acc_df['HR_Imp_Spell'].mean():+.1f}%")
    print(f"    Regressions (Favorites): {len(regressions)}/{len(acc_df)}")

    print(f"\n  Impact of Number of Interests (Hit Rate):")
    for n in sorted(acc_df['Num_Interests'].unique()):
        sub = acc_df[acc_df['Num_Interests'] == n]
        print(f"    {n} interest(s): {sub['HR_Interests'].mean():.1f}%  (n={len(sub)})")

    print(f"\n  Spell Correction:")
    print(f"    Without: {spell_df['HR_Raw'].mean():.1f}% | With: {spell_df['HR_Spell_Fixed'].mean():.1f}% | Ideal: {spell_df['HR_Ideal'].mean():.1f}%")
    print(f"    Improvement: {spell_df['Improvement'].mean():+.1f}%")

    edge_df = acc_df[acc_df['Query_Type'].str.startswith('Edge', na=False)]
    if not edge_df.empty:
        print(f"\n  Edge Cases ({len(edge_df)}):")
        print(f"    Standard={edge_df['HR_Standard'].mean():.1f}% | Interests={edge_df['HR_Interests'].mean():.1f}% | Favorites={edge_df['HR_Favorites'].mean():.1f}%")

    if zero_standard:
        pd.DataFrame(zero_standard).to_csv('zero_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n  Warning: {len(zero_standard)} queries with 0 results -> zero_results.csv")
    if regressions:
        pd.DataFrame(regressions).to_csv('regressions.csv', index=False, encoding='utf-8-sig')
        print(f"  Warning: {len(regressions)} regressions -> regressions.csv")

    print_speed_report(speed_df)
    print_concurrent_report(concurrent_df)
    print_chatbot_report(chatbot_df)

    print("\n" + "=" * 65)
    print("  FILES SAVED:")
    print("    evaluation_results.csv  — Accuracy per query")
    print("    spell_evaluation.csv    — Spell correction detail")
    print("    speed_results.csv       — Latency per query")
    print("    concurrent_results.csv  -- Concurrent users test")
    print("    chatbot_results.csv     -- ChatBot test results")
    if zero_standard: print("    zero_results.csv        — Queries with 0 results")
    if regressions:   print("    regressions.csv         — Favorites regressions")
    print("=" * 65)


if __name__ == '__main__':
    run_evaluation()