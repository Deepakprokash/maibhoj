import re
import time
import argparse
import os
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def get_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[{}"\[\]]', '', text)
    return text.strip()

def get_article_urls(driver, base_url, max_articles=10):
    """Get actual article URLs from the website"""
    article_urls = []
    domain = urlparse(base_url).netloc
    
    print(f"Getting articles from: {base_url}")
    driver.get(base_url)
    time.sleep(8)
    soup = BeautifulSoup(driver.page_source, "lxml")
    
    # Find all links on the page - use the same logic as find_articles.py
    all_links = soup.find_all("a", href=True)
    print(f"Found {len(all_links)} total links on page")
    
    for link in all_links:
        href = link.get('href', '')
        text = link.get_text().strip()
        
        # Use the exact same criteria that worked in find_articles.py
        if ('khabarbhojpuri.com' in href and 
            len(text) > 15 and
            not any(skip in href.lower() for skip in ['about', 'contact', 'category', 'tag', 'author', 'facebook', 'twitter', 'instagram'])):
            
            if href not in article_urls:
                article_urls.append(href)
                print(f"Found article {len(article_urls)}: {href}")
                print(f"  Title: {text[:80]}")
                
                if len(article_urls) >= max_articles:
                    break
    
    print(f"Total article URLs found: {len(article_urls)}")
    return article_urls

def extract_article_content(driver, url):
    """Extract article content ensuring URL, headline, and content match"""
    try:
        print(f"Processing: {url}")
        driver.get(url)
        time.sleep(8)
        
        soup = BeautifulSoup(driver.page_source, "lxml")
        
        # Extract headline
        headline = None
        title_selectors = ["h1", "h1.entry-title", "h1.post-title", ".post-title h1"]
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                headline = title_elem.get_text().strip()
                break
        
        if not headline:
            # Fallback to page title
            title_tag = soup.find('title')
            if title_tag:
                headline = title_tag.get_text().strip().split(' - ')[0]
        
        # Extract article content - target article body specifically
        article_content = None
        
        # Try to find article content container
        article_containers = [
            soup.find('div', class_=re.compile(r'entry-content|post-content|article-content|content-area')),
            soup.find('article'),
            soup.find('div', class_='post')
        ]
        
        for container in article_containers:
            if container:
                # Get all paragraphs from the container
                paragraphs = container.find_all('p')
                
                # Filter and collect paragraph text
                valid_paragraphs = []
                for p in paragraphs:
                    text = p.get_text().strip()
                    
                    # Skip if too short
                    if len(text) < 50:
                        continue
                    
                    # Skip author bio patterns
                    if any(skip in text for skip in ['सिविल इंजीनियर', 'ब्लॉगर', 'कमेंटेटर', 'anuragranjan', '@gmail.com']):
                        continue
                    
                    # Skip "Read also" and related articles sections
                    if any(skip in text.lower() for skip in ['read also', 'ई भी पढ़ीं', 'इहो पढ़ीं', 'related', 'संबंधित', 'अउरी पढ़ीं']):
                        break  # Stop processing when we hit related articles
                    
                    # Skip JSON and metadata
                    if any(skip in text.lower() for skip in ['@context', '@type', 'schema.org', 'json', '{', '}', '"@']):
                        continue
                    
                    # Check for Devanagari content
                    devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
                    if devanagari_count > 30:
                        valid_paragraphs.append(text)
                
                # Combine paragraphs
                if valid_paragraphs:
                    article_content = ' '.join(valid_paragraphs)
                    break
        
        # Fallback: if no content found, try finding largest text block (excluding author bio)
        if not article_content or len(article_content) < 200:
            text_elements = soup.find_all(string=True)
            best_content = ""
            best_score = 0
            
            for text_elem in text_elements:
                text_content = text_elem.strip()
                if len(text_content) > 200:
                    # Skip author bio
                    if any(skip in text_content for skip in ['सिविल इंजीनियर', 'ब्लॉगर', 'कमेंटेटर', 'anuragranjan', '@gmail.com']):
                        continue
                    
                    # Skip "Read also" and related articles sections
                    if any(skip in text_content.lower() for skip in ['read also', 'ई भी पढ़ीं', 'इहो पढ़ीं', 'related', 'संबंधित', 'अउरी पढ़ीं']):
                        continue
                    
                    devanagari_count = sum(1 for c in text_content if '\u0900' <= c <= '\u097F')
                    
                    # Skip JSON and metadata
                    if any(skip in text_content.lower() for skip in ['@context', '@type', 'schema.org', 'json', '{', '}', '"@']):
                        continue
                    
                    # Score by Devanagari content and length
                    score = devanagari_count * len(text_content)
                    
                    if score > best_score and devanagari_count > 50:
                        best_score = score
                        best_content = text_content
            
            if best_content and len(best_content) > len(article_content or ""):
                article_content = best_content
        
        # Clean the content
        if article_content:
            # Remove unwanted patterns
            article_content = re.sub(r'खबर भोजपुरी\s*-?\s*', '', article_content)
            article_content = re.sub(r'Khabar Bhojpuri\s*-?\s*', '', article_content)
            
            # Remove "Read also" sections and everything after
            read_also_patterns = [
                r'ई भी पढ़ीं.*',
                r'इहो पढ़ीं.*',
                r'अउरी पढ़ीं.*',
                r'Read also.*',
                r'संबंधित.*'
            ]
            for pattern in read_also_patterns:
                article_content = re.sub(pattern, '', article_content, flags=re.DOTALL | re.IGNORECASE)
            
            article_content = clean_text(article_content)
        
        if headline and article_content and len(article_content) > 200:
            print(f"✓ Extracted: {len(article_content)} chars")
            return {
                "headline": headline,
                "content": article_content
            }
        else:
            print(f"✗ Failed: headline={bool(headline)}, content_len={len(article_content) if article_content else 0}")
            return None
            
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--demo", type=int, default=10)
    args = parser.parse_args()

    print(f"Scraping {args.demo} articles from {args.site}")
    
    driver = get_driver()
    try:
        # Get article URLs
        article_urls = get_article_urls(driver, args.site, args.demo)
        
        if not article_urls:
            print("No article URLs found!")
            return
        
        # Extract content from each article
        data = []
        for i, url in enumerate(article_urls[:args.demo], 1):
            result = extract_article_content(driver, url)
            if result:
                data.append({
                    "website": urlparse(args.site).netloc,
                    "article_url": url,
                    "language": "bhojpuri",
                    "headline": result["headline"],
                    "article_text": result["content"]
                })
                print(f"✓ Successfully extracted article {i}")
            else:
                print(f"✗ Failed to extract article {i}")
        
        # Save to CSV
        if not os.path.exists("output"):
            os.makedirs("output")
        
        if data:
            df = pd.DataFrame(data)
            output_file = f"output/bhojpuri_articles_{len(data)}_final.csv"
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\nCompleted! Saved {len(data)} articles to {output_file}")
        else:
            print("No articles were successfully extracted!")
            
    finally:
        driver.quit()

if __name__ == "__main__":
    main()