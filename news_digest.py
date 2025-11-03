# news_digest.py
import os
import json
import time
import ujson
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
import feedparser
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Table, MetaData
from sqlalchemy.exc import OperationalError
import dateparser
from dateutil import tz
import openai
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv()

# CONFIG from env
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
RSS_FEEDS = os.getenv("RSS_FEEDS", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
TO_EMAIL = os.getenv("TO_EMAIL")
MAX_STORIES = int(os.getenv("MAX_STORIES", "15"))
TIMEZONE = os.getenv("TIMEZONE", "UTC")

# DB cache (SQLite)
DB_FILE = os.getenv("CACHE_DB", "seen_articles.db")
engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)
meta = MetaData()

articles_table = Table(
    'seen_articles', meta,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('article_id', String, unique=True, nullable=False),
    Column('seen_at', DateTime, nullable=False)
)

meta.create_all(engine)

# Setup OpenAI
openai.api_key = OPENAI_API_KEY

# Jinja env
env = Environment(loader=FileSystemLoader('.'), autoescape=True)
template = env.get_template('email_template.html')

def add_seen(article_id):
    ins = articles_table.insert().values(article_id=article_id, seen_at=datetime.utcnow())
    conn = engine.connect()
    try:
        conn.execute(ins)
    except Exception:
        pass
    conn.close()

def is_seen(article_id):
    sel = articles_table.select().where(articles_table.c.article_id == article_id)
    conn = engine.connect()
    res = conn.execute(sel).fetchone()
    conn.close()
    return res is not None

def fetch_newsapi(q='(business OR technology) AND (startup OR ai OR product OR market)', page_size=50):
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    articles = []
    if data.get('status') == 'ok':
        for a in data.get('articles', []):
            art_id = a.get('url')
            published = a.get('publishedAt')
            articles.append({
                "id": art_id,
                "title": a.get('title'),
                "url": a.get('url'),
                "source": a.get('source', {}).get('name'),
                "published": published,
                "raw": a
            })
    return articles

def fetch_rss_feeds():
    feeds = [f.strip() for f in RSS_FEEDS.split(',') if f.strip()]
    results = []
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed)
            for e in parsed.entries:
                art_id = e.get('link') or e.get('id') or (e.get('title') + e.get('published', ''))
                pub = e.get('published') or e.get('updated') or ''
                results.append({
                    "id": art_id,
                    "title": e.get('title'),
                    "url": e.get('link'),
                    "source": parsed.feed.get('title') or feed,
                    "published": pub,
                    "raw": e
                })
        except Exception:
            continue
    return results

def normalize_date(d):
    if not d:
        return None
    dt = dateparser.parse(d)
    if not dt:
        return None
    # convert to timezone name
    return dt.astimezone(tz.gettz(TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")

def dedupe_and_filter(articles):
    unique = {}
    for a in articles:
        key = a.get('id') or a.get('url') or a.get('title')
        if not key:
            continue
        if is_seen(key):
            continue
        if key in unique:
            continue
        unique[key] = a
    return list(unique.values())

# LLM summarization prompt
SUMMARIZE_PROMPT = """
You are a concise news summarizer. Given an article title and url (and optionally a short snippet),
produce:
- a 1-2 sentence factual summary (no opinion)
- topic tags (choose from: AI, Markets, Startups, Product, Regulation, M&A, Hiring, Research, Other)
- a short sentiment label (Positive, Neutral, Negative)
- a relevance score 0-1 (as a float)
Return JSON: {{ "summary": "...", "tags": ["AI"], "sentiment": "Neutral", "score": 0.83 }}
"""

def call_llm_for_article(article):
    # Build context (title + url + excerpt)
    text = f"Title: {article.get('title')}\nURL: {article.get('url')}\n\nIf you can fetch the article, summarize; if not, summarize from the title and short snippet.\n\n"
    try:
        content = f"{text}"
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"You are a precise news summarizer."},
                      {"role":"user","content": SUMMARIZE_PROMPT + "\n\n" + content}],
            max_tokens=400,
            temperature=0.0
        )
        out = resp['choices'][0]['message']['content'].strip()
        # Expect JSON in output — try to parse; otherwise fallback to simple text parse
        try:
            parsed = json.loads(out)
            return parsed
        except Exception:
            # fallback: create a simple summary
            return {
                "summary": out[:400],
                "tags": ["Other"],
                "sentiment": "Neutral",
                "score": 0.5
            }
    except Exception as e:
        # Fail gracefully
        return {
            "summary": article.get('title') or "",
            "tags": ["Other"],
            "sentiment": "Neutral",
            "score": 0.5
        }

def group_by_topic(items):
    grouped = {}
    for it in items:
        tags = it.get('tags') or ["Other"]
        primary = tags[0] if isinstance(tags, list) and tags else "Other"
        if primary not in grouped:
            grouped[primary] = []
        grouped[primary].append(it)
    return grouped

def send_email_via_sendgrid(subject, html_content):
    if not SENDGRID_API_KEY or not FROM_EMAIL or not TO_EMAIL:
        raise RuntimeError("SendGrid keys or emails not configured.")
    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=subject,
        html_content=html_content
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print("Email sent, status:", response.status_code)
    except Exception as e:
        print("SendGrid send error:", e)
        raise

def build_and_send(stories):
    generated_at = datetime.now(tz.gettz(TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")
    grouped = group_by_topic(stories)
    html = template.render(stories=grouped, generated_at=generated_at, total_stories=len(stories), timezone=TIMEZONE)
    subj = f"Business & Tech Digest — {generated_at} — {len(stories)} stories"
    send_email_via_sendgrid(subj, html)

def main():
    # Fetch
    newsapi_articles = fetch_newsapi()
    rss_articles = fetch_rss_feeds()
    all_articles = newsapi_articles + rss_articles

    all_articles = sorted(all_articles, key=lambda x: x.get('published') or "", reverse=True)

    # dedupe & filter by seen
    candidates = dedupe_and_filter(all_articles)

    # Keep only top MAX_STORIES
    candidates = candidates[:MAX_STORIES]

    processed = []
    for a in candidates:
        info = {
            "title": a.get('title'),
            "url": a.get('url'),
            "source": a.get('source'),
            "published": normalize_date(a.get('published')),
        }
        # Call LLM
        llm = call_llm_for_article(a)
        info['summary'] = llm.get('summary') if isinstance(llm, dict) else str(llm)[:400]
        info['tags'] = llm.get('tags') if isinstance(llm, dict) else ["Other"]
        info['sentiment'] = llm.get('sentiment') if isinstance(llm, dict) else "Neutral"
        info['score'] = llm.get('score') if isinstance(llm, dict) else 0.5

        processed.append(info)
        add_seen(a.get('id'))

    if processed:
        build_and_send(processed)
    else:
        print("No new stories to send at this time.")

if __name__ == "__main__":
    main()