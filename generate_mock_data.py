"""
Mock Data Generator for Power BI Analytics Report
Generates 4 separate CSV files: marketing, sessions, transactions, products
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import argparse
import os


# Configuration
CHANNEL_GROUPINGS = {
    'Organic Search': {
        'sources': ['google', 'bing', 'yahoo', 'duckduckgo'],
        'mediums': ['organic', 'search'],
        'has_ads': False
    },
    'Direct': {
        'sources': ['(direct)', '(none)'],
        'mediums': ['(none)', '(not set)'],
        'has_ads': False
    },
    'Referral': {
        'sources': ['facebook.com', 'twitter.com', 'linkedin.com', 'reddit.com', 'medium.com'],
        'mediums': ['referral'],
        'has_ads': False
    },
    'Paid Search': {
        'sources': ['google', 'bing', 'yahoo'],
        'mediums': ['cpc', 'ppc'],
        'has_ads': True
    },
    'Social': {
        'sources': ['facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com', 'tiktok.com'],
        'mediums': ['social', 'social-media'],
        'has_ads': False
    },
    'Display': {
        'sources': ['google', 'doubleclick', 'adform', 'criteo'],
        'mediums': ['display', 'banner', 'cpm'],
        'has_ads': True
    },
    'Affiliates': {
        'sources': ['affiliate1.com', 'affiliate2.com', 'partner.com'],
        'mediums': ['affiliate', 'affiliate-program'],
        'has_ads': True  # Has clicks and cost, but no impressions
    },
    '(Other)': {
        'sources': ['email', 'newsletter', 'app'],
        'mediums': ['email', 'notification'],
        'has_ads': False
    }
}

COUNTRIES = ['US', 'UK', 'CA', 'DE', 'FR', 'AU', 'NL', 'SE', 'NO', 'DK', 'IT', 'ES', 'PL', 'BR', 'MX']
PLATFORMS = ['Desktop', 'Mobile', 'Tablet']

PRODUCT_CATEGORIES = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Health & Beauty']
PRODUCTS = {
    'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Smartwatch', 'Headphones', 'Camera'],
    'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Dress', 'Hat'],
    'Home & Garden': ['Furniture', 'Kitchen Tools', 'Garden Supplies', 'Lighting', 'Decorative Items'],
    'Books': ['Fiction', 'Non-Fiction', 'Technical', 'Biography', 'Children Books'],
    'Sports': ['Running Shoes', 'Yoga Mat', 'Dumbbells', 'Bicycle', 'Tennis Racket'],
    'Health & Beauty': ['Skincare', 'Makeup', 'Vitamins', 'Haircare', 'Fragrance']
}

# Price ranges by category (min, max)
PRICE_RANGES = {
    'Electronics': (50, 2000),
    'Clothing': (15, 300),
    'Home & Garden': (20, 800),
    'Books': (5, 50),
    'Sports': (25, 600),
    'Health & Beauty': (10, 150)
}

# Product IDs mapping for consistency
PRODUCT_IDS = {}
product_id_counter = 1


def get_product_id(category: str, product_name: str) -> str:
    """Get or create product_id for a product."""
    global product_id_counter
    key = f"{category}|{product_name}"
    if key not in PRODUCT_IDS:
        PRODUCT_IDS[key] = f"prod_{product_id_counter:05d}"
        product_id_counter += 1
    return PRODUCT_IDS[key]


def generate_channel_from_grouping(channel: str) -> Tuple[str, str]:
    """Generate random source and medium for a given channel grouping."""
    config = CHANNEL_GROUPINGS[channel]
    source = random.choice(config['sources'])
    medium = random.choice(config['mediums'])
    return source, medium


def generate_ad_metrics(channel: str, has_ads: bool) -> Tuple[int, int, float]:
    """Generate impressions, clicks and cost for paid channels."""
    if not has_ads:
        return 0, 0, 0.0

    # Special handling for Affiliates - no impressions
    if channel == 'Affiliates':
        min_clicks, max_clicks = (500, 25000)
        clicks = random.randint(min_clicks, max_clicks)
        # CPC for affiliates typically $0.3-$1.5
        cpc = random.uniform(0.3, 1.5)
        cost = clicks * cpc
        return 0, clicks, round(cost, 2)

    # Generate realistic ad metrics for other paid channels
    base_impressions_range = {
        'Paid Search': (50000, 500000),
        'Display': (10000, 200000)
    }

    if channel in base_impressions_range:
        min_imp, max_imp = base_impressions_range[channel]
        impressions = random.randint(min_imp, max_imp)

        # CTR typically 1-3% for paid search, 0.5-2% for display
        ctr = random.uniform(0.01, 0.03) if channel == 'Paid Search' else random.uniform(0.005, 0.02)
        clicks = int(impressions * ctr)

        # CPC typically $0.5-$5 for paid search, $0.1-$2 for display
        cpc = random.uniform(0.5, 5.0) if channel == 'Paid Search' else random.uniform(0.1, 2.0)
        cost = clicks * cpc

        return impressions, clicks, round(cost, 2)

    return 0, 0, 0.0


def generate_user_pool(num_users: int = 50000) -> List[Dict]:
    """Generate pool of users with their typical attributes."""
    users = []
    for i in range(num_users):
        # Assign user to a channel with some distribution
        channel_weights = {
            'Organic Search': 0.35,
            'Direct': 0.25,
            'Referral': 0.15,
            'Paid Search': 0.10,
            'Social': 0.08,
            'Display': 0.04,
            'Affiliates': 0.02,
            '(Other)': 0.01
        }
        channel = np.random.choice(list(channel_weights.keys()), p=list(channel_weights.values()))

        source, medium = generate_channel_from_grouping(channel)
        country = random.choice(COUNTRIES)
        platform = np.random.choice(PLATFORMS, p=[0.5, 0.4, 0.1])

        users.append({
            'user_id': f'user_{i:06d}',
            'preferred_channel': channel,
            'source': source,
            'medium': medium,
            'country': country,
            'platform': platform
        })

    return users


def generate_marketing_data(
    start_date: str,
    months: int = 6
) -> pd.DataFrame:
    """Generate aggregated marketing data by day, source_medium, channel, country, platform."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = start + timedelta(days=months * 30)

    marketing_records = []
    current_date = start

    while current_date < end:
        date_str = current_date.strftime('%Y-%m-%d')

        # Generate marketing data for each channel/country/platform combination
        for channel in CHANNEL_GROUPINGS.keys():
            source, medium = generate_channel_from_grouping(channel)
            source_medium = f"{source}/{medium}"
            has_ads = CHANNEL_GROUPINGS[channel]['has_ads']

            # Generate impressions, clicks and cost only for paid channels
            if has_ads:
                # Add some variation day by day
                base_impressions, base_clicks, base_cost = generate_ad_metrics(channel, has_ads)
                daily_variation = random.uniform(0.8, 1.2)  # ±20% variation
                impressions = int(base_impressions * daily_variation)
                clicks = int(base_clicks * daily_variation)
                cost = round(base_cost * daily_variation, 2)

                # Generate data for each country/platform combination for paid channels
                for country in COUNTRIES[:5]:  # Limit countries for paid channels
                    for platform in PLATFORMS:
                        country_factor = random.uniform(0.3, 1.5)
                        platform_factor = random.uniform(0.5, 1.5)

                        marketing_records.append({
                            'date': date_str,
                            'source_medium': source_medium,
                            'channel_grouping': channel,
                            'country': country,
                            'platform': platform,
                            'impressions': int(impressions * country_factor * platform_factor / 15) if impressions > 0 else 0,
                            'clicks': int(clicks * country_factor * platform_factor / 15),
                            'cost': round(cost * country_factor * platform_factor / 15, 2)
                        })
            else:
                # For non-paid channels, impressions, clicks and cost are 0
                for country in COUNTRIES:
                    for platform in PLATFORMS:
                        marketing_records.append({
                            'date': date_str,
                            'source_medium': source_medium,
                            'channel_grouping': channel,
                            'country': country,
                            'platform': platform,
                            'impressions': 0,
                            'clicks': 0,
                            'cost': 0.0
                        })

        current_date += timedelta(days=1)

    return pd.DataFrame(marketing_records)


def generate_sessions_data(
    start_date: str,
    months: int = 6,
    users: List[Dict] = None
) -> pd.DataFrame:
    """Generate web sessions data."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = start + timedelta(days=months * 30)

    sessions = []
    session_counter = 0
    current_date = start

    while current_date < end:
        date_str = current_date.strftime('%Y-%m-%d')

        # Generate sessions per day (vary by weekday/weekend)
        base_sessions = 15000 if current_date.weekday() < 5 else 8000
        daily_sessions = int(np.random.normal(base_sessions, base_sessions * 0.2))
        daily_sessions = max(1000, daily_sessions)

        # Sample users for this day
        daily_users = random.sample(users, min(daily_sessions, len(users)))

        for user in daily_users:
            # Sometimes users come from different sources
            if random.random() < 0.15:  # 15% chance to use different source
                channel = np.random.choice(list(CHANNEL_GROUPINGS.keys()))
                source, medium = generate_channel_from_grouping(channel)
                country = random.choice(COUNTRIES)
                platform = np.random.choice(PLATFORMS, p=[0.5, 0.4, 0.1])
            else:
                channel = user['preferred_channel']
                source = user['source']
                medium = user['medium']
                country = user['country']
                platform = user['platform']

            source_medium = f"{source}/{medium}"

            sessions.append({
                'date': date_str,
                'user_id': user['user_id'],
                'session_id': f'session_{session_counter:08d}',
                'source_medium': source_medium,
                'channel_grouping': channel,
                'country': country,
                'platform': platform
            })

            session_counter += 1

        current_date += timedelta(days=1)

    return pd.DataFrame(sessions)


def generate_transactions_data(
    sessions_df: pd.DataFrame,
    conversion_rate: float = 0.03
) -> pd.DataFrame:
    """Generate transactions data linked to sessions."""
    transactions = []
    transaction_counter = 0

    # Sample sessions that convert to transactions
    num_transactions = int(len(sessions_df) * conversion_rate)
    converting_sessions = sessions_df.sample(n=min(num_transactions, len(sessions_df)), random_state=42)

    for _, session in converting_sessions.iterrows():
        # Transaction happens same day or within 7 days of session
        session_date = datetime.strptime(session['date'], '%Y-%m-%d')
        days_delay = random.randint(0, 7) if random.random() < 0.7 else 0  # 70% same day
        transaction_date = session_date + timedelta(days=days_delay)

        transaction_id = f'trans_{transaction_counter:08d}'

        transactions.append({
            'date': transaction_date.strftime('%Y-%m-%d'),
            'user_id': session['user_id'],
            'session_id': session['session_id'],
            'transaction_id': transaction_id,
            'source_medium': session['source_medium'],
            'channel_grouping': session['channel_grouping'],
            'country': session['country'],
            'platform': session['platform']
        })

        transaction_counter += 1

    return pd.DataFrame(transactions)


def generate_products_data(
    transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate products data for each transaction."""
    products = []

    for _, transaction in transactions_df.iterrows():
        # Determine number of items in transaction (1-4 items typical)
        num_items = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])

        for item_num in range(num_items):
            category = random.choice(PRODUCT_CATEGORIES)
            product_name = random.choice(PRODUCTS[category])
            product_id = get_product_id(category, product_name)
            min_price, max_price = PRICE_RANGES[category]
            product_cost = round(random.uniform(min_price, max_price), 2)

            products.append({
                'transaction_id': transaction['transaction_id'],
                'product_id': product_id,
                'product_name': product_name,
                'product_category': category,
                'product_cost': product_cost
            })

    return pd.DataFrame(products)


def main():
    parser = argparse.ArgumentParser(description='Generate mock web analytics data')
    parser.add_argument('--start-date', type=str, default='2025-01-01',
                       help='Start date in YYYY-MM-DD format (default: 2025-01-01)')
    parser.add_argument('--months', type=int, default=6,
                       help='Number of months to generate (default: 6)')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for CSV files (default: ./data)')
    parser.add_argument('--users', type=int, default=50000,
                       help='Number of unique users in pool (default: 50000)')

    args = parser.parse_args()

    print(f"Generating mock data starting from {args.start_date} for {args.months} months...")
    print(f"Creating user pool of {args.users} users...")

    # Generate user pool
    users = generate_user_pool(args.users)

    # Generate marketing data (aggregated by day)
    print("Generating marketing data...")
    marketing_df = generate_marketing_data(args.start_date, args.months)
    print(f"Generated {len(marketing_df):,} marketing records")

    # Generate sessions
    print("Generating web sessions...")
    sessions_df = generate_sessions_data(args.start_date, args.months, users)
    print(f"Generated {len(sessions_df):,} sessions")

    # Generate transactions linked to sessions
    print("Generating transactions...")
    transactions_df = generate_transactions_data(sessions_df, conversion_rate=0.03)
    print(f"Generated {len(transactions_df):,} transactions")

    # Generate products for transactions
    print("Generating products...")
    products_df = generate_products_data(transactions_df)
    print(f"Generated {len(products_df):,} product records")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to CSV
    marketing_file = os.path.join(args.output_dir, 'marketing.csv')
    sessions_file = os.path.join(args.output_dir, 'sessions.csv')
    transactions_file = os.path.join(args.output_dir, 'transactions.csv')
    products_file = os.path.join(args.output_dir, 'products.csv')

    marketing_df.to_csv(marketing_file, index=False)
    sessions_df.to_csv(sessions_file, index=False)
    transactions_df.to_csv(transactions_file, index=False)
    products_df.to_csv(products_file, index=False)

    print(f"\nData saved to:")
    print(f"  - {marketing_file}")
    print(f"  - {sessions_file}")
    print(f"  - {transactions_file}")
    print(f"  - {products_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nMarketing - Total Impressions: {marketing_df['impressions'].sum():,}")
    print(f"Marketing - Total Clicks: {marketing_df['clicks'].sum():,}")
    print(f"Marketing - Total Cost: ${marketing_df['cost'].sum():,.2f}")

    print(f"\nSessions by Channel Grouping:")
    print(sessions_df['channel_grouping'].value_counts())

    print(f"\nTotal Transactions: {len(transactions_df):,}")
    print(f"Total Revenue: ${products_df['product_cost'].sum():,.2f}")
    print(f"Total Ad Costs: ${marketing_df['cost'].sum():,.2f}")
    print(f"Net Profit: ${products_df['product_cost'].sum() - marketing_df['cost'].sum():,.2f}")

    print(f"\nProducts by Category:")
    print(products_df['product_category'].value_counts())

    print("\n=== Sample Data ===")
    print("\nFirst 5 marketing records:")
    print(marketing_df.head())
    print("\nFirst 5 sessions:")
    print(sessions_df.head())
    print("\nFirst 5 transactions:")
    print(transactions_df.head())
    print("\nFirst 5 products:")
    print(products_df.head())


if __name__ == '__main__':
    main()