import pandas as pd
import numpy as np
import re
import warnings

def add_feature_engineering(df):
    df = df.copy()  # 📋 Work on a copy to avoid modifying the original DataFrame

    # 🎓 Extract seniority level based on keywords in job title
    def extract_seniority(title):
        title = title.lower()  # Make case-insensitive
        if re.search(r'(intern|trainee|junior)', title):
            return 'junior'
        elif re.search(r'(senior|sr|lead)', title):
            return 'senior'
        elif re.search(r'(manager|director|head|chief)', title):
            return 'management'
        else:
            return 'mid'  # Default if no clear match found

    df['seniority_level'] = df['job_title'].apply(extract_seniority)

    # 🕒 Bin years of experience into custom ranges
    def bin_experience(x):
        if x < 2:
            return '0–2'
        elif x < 5:
            return '2–5'
        elif x < 10:
            return '5–10'
        else:
            return '10+'

    df['experience_bin'] = df['years_experience'].apply(bin_experience)

    # 🏠 Create a binary flag for full remote jobs (remote_ratio = 100%)
    df['remote_flag'] = df['remote_ratio'].apply(lambda x: 1 if x == 100 else 0)

    # 🌍 Map company location (country code) to a continent
    # Extend this dictionary as needed for other country codes
    continent_map = {
        'US': 'North America', 'CA': 'North America',
        'IN': 'Asia', 'CN': 'Asia', 'JP': 'Asia',
        'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'IT': 'Europe',
        'AU': 'Oceania', 'NZ': 'Oceania',
        'BR': 'South America', 'AR': 'South America',
        'ZA': 'Africa', 'NG': 'Africa'
    }
    df['continent'] = df['company_location'].map(continent_map).fillna('Other')

    # 💱 Classify currency into strength categories (manual list)
    strong_currencies = ['USD', 'EUR', 'GBP', 'CHF']  # Strong/Stable currencies
    weak_currencies = ['INR', 'BRL', 'IDR', 'ZAR']     # Weak/Volatile currencies

    def currency_strength(curr):
        if curr in strong_currencies:
            return 'strong'
        elif curr in weak_currencies:
            return 'weak'
        else:
            return 'mid'  # Default/mid-tier currency

    df['currency_strength'] = df['currency'].apply(currency_strength)

    return df  # ✅ Return the transformed DataFrame
