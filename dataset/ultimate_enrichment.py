import pandas as pd
import requests
import time
from tqdm import tqdm
import os


file_name = 'AlWarraq_NLP_Ready.csv'

print("⏳ Loading dataset...")
df = pd.read_csv(file_name)


def get_verified_google_description(title, author):
    try:
        query = f"intitle:{title}+inauthor:{author}"
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&langRestrict=en"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and len(data['items']) > 0:
                desc = data['items'][0].get('volumeInfo', {}).get('description', '')
                if desc and len(desc) > 150:
                    return desc.replace('<br>', '\n').replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
    except:
        pass
    return None


bad_data_condition = (
    df['description'].astype(str).str.contains('A book about', case=False, na=False) | 
    (df['description'].astype(str).str.len() < 150)
)

books_to_update = df[bad_data_condition]
print(f"🎯 Found {len(books_to_update)} books remaining to update.")

print("🚀 Starting/Resuming Process...")
updated_count = 0

try:
    for index, row in tqdm(books_to_update.iterrows(), total=len(books_to_update), desc="Updating"):
        title = str(row['title'])
        author = str(row['author'])
        
        verified_desc = get_verified_google_description(title, author)
        
        if verified_desc:
            df.at[index, 'description'] = verified_desc
            updated_count += 1
        
        # حفظ فوري كل 10 كتب عشان لو قفلت السكربت فجأة ما يضيع شيء
        if updated_count % 10 == 0:
            df.to_csv(file_name, index=False)
            
        time.sleep(1) # احترام قوانين جوجل

except KeyboardInterrupt:
    print("\n🛑 Stopped by user. Saving progress...")

# حفظ نهائي عند الإغلاق
df.to_csv(file_name, index=False)
print(f"✅ Progress saved! Updated {updated_count} books in this session.")