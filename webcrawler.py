import os
import asyncio
import aiohttp
import aiofiles
import re
import json
import html2text
import urllib.parse
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from io import BytesIO
import base64
import hashlib
import time
from typing import List, Dict, Set, Tuple, Optional, Any
import nest_asyncio

# Apply nest_asyncio to avoid asyncio loop issues when using with Streamlit
nest_asyncio.apply()

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 20, max_depth: int = 3, 
                 download_images: bool = True, image_folder: str = "./crawled_images"):
        """
        Initialize the web crawler.
        
        Args:
            base_url: The starting URL to crawl
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth of crawling from the base URL
            download_images: Whether to download images
            image_folder: Folder to save downloaded images
        """
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.download_images = download_images
        self.image_folder = image_folder
        self.visited_urls: Set[str] = set()
        self.pages_data: List[Dict] = []
        self.images_data: List[Dict] = []
        
        # Create image folder if it doesn't exist
        if self.download_images and not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        
        # Initialize HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_tables = False
        
        # Initialize Selenium options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
    
    async def fetch_url(self, url: str, session: aiohttp.ClientSession) -> Tuple[str, str]:
        """Fetch a URL and return the HTML content."""
        try:
            async with session.get(url, timeout=15) as response:
                return url, await response.text()
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return url, ""
    
    def extract_links(self, url: str, html_content: str) -> List[str]:
        """Extract links from HTML content."""
        base_url = url
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            
            # Only include links from the same domain
            if urlparse(full_url).netloc == self.base_domain:
                links.append(full_url)
        
        return links
    
    def extract_text_and_images(self, url: str, html_content: str) -> Tuple[str, List[Dict]]:
        """Extract text and image URLs from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No Title"
        
        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and "content" in meta_tag.attrs:
            meta_desc = meta_tag["content"]
        
        # Extract header content
        headers_content = []
        for header in soup.find_all(['h1', 'h2', 'h3']):
            if header.text.strip():
                headers_content.append(header.text.strip())
        
        # Extract all text content and clean it
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Convert HTML to markdown-style text to preserve some structure
        markdown_text = self.html_converter.handle(html_content)
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            alt = img.get('alt', '')
            
            # Convert relative URLs to absolute
            img_url = urljoin(url, src)
            
            # Only process images from the same domain or with complete URLs
            if self.base_domain in urlparse(img_url).netloc or src.startswith(('http://', 'https://')):
                img_data = {
                    'url': img_url,
                    'alt_text': alt,
                    'page_url': url,
                    'filename': None
                }
                images.append(img_data)
        
        # Combine all text elements
        full_text = f"Title: {title}\n\nURL: {url}\n\nDescription: {meta_desc}\n\n"
        if headers_content:
            full_text += "Headers:\n" + "\n".join(headers_content) + "\n\n"
        full_text += "Content:\n" + markdown_text
        
        return full_text, images
    
    async def download_image(self, image_data: Dict, session: aiohttp.ClientSession) -> Dict:
        """Download an image and save it to disk."""
        if not self.download_images:
            return image_data
        
        try:
            img_url = image_data['url']
            
            # Generate a filename based on URL hash
            url_hash = hashlib.md5(img_url.encode()).hexdigest()
            extension = os.path.splitext(urlparse(img_url).path)[1]
            if not extension or len(extension) > 5:  # Handle cases with no extension or invalid ones
                extension = '.jpg'  # Default to jpg
            
            filename = f"{url_hash}{extension}"
            filepath = os.path.join(self.image_folder, filename)
            
            # Download and save the image
            async with session.get(img_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Verify the content is a valid image
                    try:
                        Image.open(BytesIO(content))
                        
                        # Save the image
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(content)
                        
                        # Update image data with local filename
                        image_data['filename'] = filename
                        image_data['local_path'] = filepath
                    except Exception as e:
                        print(f"Invalid image format for {img_url}: {str(e)}")
                
            return image_data
        except Exception as e:
            print(f"Error downloading image {image_data['url']}: {str(e)}")
            return image_data
    
    async def crawl_with_selenium(self, url: str) -> Dict:
        """
        Crawl a JavaScript-heavy page using Selenium.
        Used as a fallback when regular crawling doesn't yield good results.
        """
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.chrome_options)
            
            # Load the page with Selenium
            driver.get(url)
            time.sleep(3)  # Wait for JS to execute
            
            # Get the rendered HTML
            html_content = driver.page_source
            
            # Process the content
            text, images = self.extract_text_and_images(url, html_content)
            
            # Create page data
            page_data = {
                'url': url,
                'text': text,
                'images': images
            }
            
            driver.quit()
            return page_data
            
        except Exception as e:
            print(f"Selenium crawling error for {url}: {str(e)}")
            return {'url': url, 'text': '', 'images': []}
    
    async def crawl(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Crawl the website starting from the base URL.
        Returns a tuple of (pages_data, images_data).
        """
        to_visit = [(self.base_url, 0)]  # (URL, depth)
        visited_count = 0
        
        async with aiohttp.ClientSession() as session:
            while to_visit and visited_count < self.max_pages:
                # Get next URL to visit
                current_url, depth = to_visit.pop(0)
                
                # Skip if already visited or too deep
                if current_url in self.visited_urls or depth > self.max_depth:
                    continue
                
                # Mark as visited
                self.visited_urls.add(current_url)
                visited_count += 1
                
                print(f"Crawling ({visited_count}/{self.max_pages}): {current_url}")
                
                # Fetch the URL
                url, html_content = await self.fetch_url(current_url, session)
                
                if not html_content:
                    # Try with Selenium as fallback
                    page_data = await self.crawl_with_selenium(current_url)
                    if page_data['text']:
                        self.pages_data.append(page_data)
                        # Process images if any
                        if page_data['images']:
                            image_download_tasks = []
                            for img_data in page_data['images']:
                                task = self.download_image(img_data, session)
                                image_download_tasks.append(task)
                            
                            downloaded_images = await asyncio.gather(*image_download_tasks)
                            self.images_data.extend([img for img in downloaded_images if img.get('filename')])
                    continue
                
                # Extract text and images
                text, images = self.extract_text_and_images(url, html_content)
                
                # Create page data
                page_data = {
                    'url': url,
                    'text': text,
                    'images': images
                }
                self.pages_data.append(page_data)
                
                # Download images
                if self.download_images and images:
                    image_download_tasks = []
                    for img_data in images:
                        task = self.download_image(img_data, session)
                        image_download_tasks.append(task)
                    
                    downloaded_images = await asyncio.gather(*image_download_tasks)
                    self.images_data.extend([img for img in downloaded_images if img.get('filename')])
                
                # Extract links if not at max depth
                if depth < self.max_depth:
                    links = self.extract_links(url, html_content)
                    
                    # Add new links to visit
                    for link in links:
                        if link not in self.visited_urls:
                            to_visit.append((link, depth + 1))
        
        return self.pages_data, self.images_data
    
    def save_results(self, output_folder: str = "./crawled_data") -> Tuple[str, str]:
        """Save crawled data to JSON files."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        pages_file = os.path.join(output_folder, "pages.json")
        images_file = os.path.join(output_folder, "images.json")
        
        with open(pages_file, 'w', encoding='utf-8') as f:
            json.dump(self.pages_data, f, ensure_ascii=False, indent=2)
        
        with open(images_file, 'w', encoding='utf-8') as f:
            json.dump(self.images_data, f, ensure_ascii=False, indent=2)
        
        return pages_file, images_file

# Example usage
async def main():
    crawler = WebCrawler("https://example.com", max_pages=5)
    pages, images = await crawler.crawl()
    print(f"Crawled {len(pages)} pages and found {len(images)} images")
    pages_file, images_file = crawler.save_results()
    print(f"Results saved to {pages_file} and {images_file}")

if __name__ == "__main__":
    asyncio.run(main())
