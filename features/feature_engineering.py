import pandas as pd
import numpy as np
import re
import warnings

def add_feature_engineering(df):
    df = df.copy()
    configure_yaml_dumper()
    # 🎓 Seniority extraction from job title
    def extract_seniority(title):
        title = title.lower()
        if re.search(r'(intern|trainee|junior)', title):
            return 'junior'
        elif re.search(r'(senior|sr|lead)', title):
            return 'senior'
        elif re.search(r'(manager|director|head|chief)', title):
            return 'management'
        else:
            return 'mid'

    df['seniority_level'] = df['job_title'].apply(extract_seniority)

    # 🕒 Experience binning
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

    # 🏠 Remote flag from remote_ratio
    df['remote_flag'] = df['remote_ratio'].apply(lambda x: 1 if x == 100 else 0)

    # 🌍 Continent mapping from company location (you can expand this as needed)
    continent_map = {
        'US': 'North America', 'CA': 'North America',
        'IN': 'Asia', 'CN': 'Asia', 'JP': 'Asia',
        'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'IT': 'Europe',
        'AU': 'Oceania', 'NZ': 'Oceania',
        'BR': 'South America', 'AR': 'South America',
        'ZA': 'Africa', 'NG': 'Africa'
    }
    df['continent'] = df['company_location'].map(continent_map).fillna('Other')

    # 💱 Currency strength category (manual — you may update rates)
    strong_currencies = ['USD', 'EUR', 'GBP', 'CHF']
    weak_currencies = ['INR', 'BRL', 'IDR', 'ZAR']

    def currency_strength(curr):
        if curr in strong_currencies:
            return 'strong'
        elif curr in weak_currencies:
            return 'weak'
        else:
            return 'mid'

    df['currency_strength'] = df['currency'].apply(currency_strength)

    return df
