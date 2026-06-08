"""Parses legal documents into structured, clean text."""
from typing import List
from bs4 import BeautifulSoup
import re

class DocumentParser:
    """Parses legal documents into structured, clean text."""

    def __init__(self):
        self.html_parser = "html.parser"

    def parse_html(self, html_content: str) -> str:
        """Extract clean text from HTML content."""
        soup = BeautifulSoup(html_content, self.html_parser)
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text

    def parse_robots_txt(self, robots_content: str) -> dict:
        """Parse robots.txt into structured data."""
        rules = {"allow": [], "disallow": [], "sitemaps": [], "crawl-delay": []}
        for line in robots_content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lower_line = line.lower()
            if lower_line.startswith('user-agent:'):
                pass
            elif lower_line.startswith('allow:'):
                rules["allow"].append(line.split(':', 1)[1].strip())
            elif lower_line.startswith('disallow:'):
                rules["disallow"].append(line.split(':', 1)[1].strip())
            elif lower_line.startswith('sitemap:'):
                rules["sitemaps"].append(line.split(':', 1)[1].strip())
            elif lower_line.startswith('crawl-delay:'):
                try:
                    rules["crawl-delay"].append(float(line.split(':', 1)[1].strip()))
                except ValueError:
                    pass
        return rules

    def extract_key_sections(self, text: str) -> List[dict]:
        """Extract key sections from legal documents for better chunking."""
        sections = []
        lines = text.split('\n')
        current_section = {"title": "", "content": []}
        for line in lines:
            if self._is_section_header(line):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"title": line.strip(), "content": []}
            else:
                current_section["content"].append(line)
        if current_section["content"]:
            sections.append(current_section)
        return sections

    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is a section header."""
        line = line.strip()
        return (
            line.isupper() or
            bool(re.match(r'^(Section|Article|Clause)\s+\d+', line, re.I)) or
            bool(re.match(r'^\d+\.\s+[A-Z]', line))
