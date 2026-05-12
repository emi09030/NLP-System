import re
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


INPUT_TXT = "fit_hcmute_thong_bao.txt"
OUTPUT_TXT = "fit_hcmute_crawled_all.txt"
OUTPUT_JSON = "fit_hcmute_crawled_all.json"
BASE_URL = "https://fit.hcmute.edu.vn"


session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
})


def clean_line(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def clean_text(text):
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def read_urls_from_txt(file_path):
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    urls = re.findall(r"https?://[^\s\"'<>]+", text)

    result = []

    for url in urls:
        url = url.strip().rstrip(".,;)")
        url = url.replace("/Default.aspx?ArticleId=", "/?ArticleId=")

        if "fit.hcmute.edu.vn" in url:
            result.append(url)

    return list(dict.fromkeys(result))


def get_soup(url):
    response = session.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return BeautifulSoup(response.text, "lxml")


def remove_noise(soup):
    for tag in soup(["script", "style", "noscript", "iframe", "input", "button", "select", "textarea"]):
        tag.decompose()

    for tag in soup.select(".RadEditor, .reContentArea, .footer, .menu, .navbar, .breadcrumb"):
        tag.decompose()

    return soup


def normalize_spaces(text):
    text = text.replace("\xa0", " ")
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def block_to_text(block):
    for br in block.find_all("br"):
        br.replace_with("\n")

    for tag in block.find_all(["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = block.get_text("\n")
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    return normalize_spaces("\n".join(lines))


def find_main_content(soup):
    selectors = [
        "div.articleContent",
        ".articleContent",
        "div.ct_tin_display .articleContent",
        "div.ct_tin_display",
        ".ct_tin_display",
        ".article-content",
        ".news-content",
        ".detail-content",
        ".post-content",
        ".content",
        ".main-content",
        "#content"
    ]

    for selector in selectors:
        block = soup.select_one(selector)

        if block:
            text = block_to_text(block)

            if len(text) >= 30:
                return block

    candidates = []

    for div in soup.find_all("div"):
        text = block_to_text(div)

        if len(text) >= 100:
            score = len(text)

            if "Tác giả" in text:
                score += 1000

            if "Các tin khác" in text:
                score -= 500

            if "Góp ý" in text:
                score -= 500

            candidates.append((score, div))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return soup.body if soup.body else soup


def extract_title(soup, content_block):
    article_title = soup.select_one(".articleTitle")

    if article_title:
        text = clean_line(article_title.get_text(" "))

        if len(text) >= 5:
            return text

    for selector in ["h1", "h2", "h3", ".title", ".news-title", ".post-title", ".article-title"]:
        tag = soup.select_one(selector)

        if tag:
            text = clean_line(tag.get_text(" "))

            if len(text) >= 5:
                return text

    for tag in content_block.find_all(["strong", "b", "span", "p"]):
        text = clean_line(tag.get_text(" "))

        if len(text) >= 10:
            return text

    if soup.title:
        return clean_line(soup.title.get_text(" "))

    return ""


def extract_date(text):
    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        r"\bNgày\s+\d{1,2}/\d{1,2}/\d{4}\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return match.group(0)

    return ""


def extract_links(block):
    links = []

    for a in block.find_all("a", href=True):
        text = clean_line(a.get_text(" "))
        href = urljoin(BASE_URL, a["href"])

        if href:
            links.append({
                "text": text,
                "url": href
            })

    return links


def filter_content(text):
    stop_keywords = [
        "Các tin khác",
        "Tin khác",
        "Bài viết liên quan",
        "Góp ý",
        "Họ và tên",
        "Email:",
        "Nội dung góp ý",
        "Mã xác nhận",
        "Việc làm - Doanh nghiệp",
        "Liên kết website",
        "Copyright",
        "Địa chỉ:",
        "Điện thoại:",
        "E-mail:",
        "Truy cập tháng",
        "Tổng truy cập",
        "RadEditor"
    ]

    remove_lines = [
        "Tác giả :",
        "Tác giả:",
        "Ngày đăng :",
        "Ngày đăng:"
    ]

    result = []

    for line in text.splitlines():
        line = clean_line(line)

        if not line:
            continue

        if any(keyword.lower() in line.lower() for keyword in stop_keywords):
            break

        if line in remove_lines:
            continue

        result.append(line)

    return "\n".join(result).strip()


def crawl_article(url):
    soup = get_soup(url)
    soup = remove_noise(soup)

    content_block = find_main_content(soup)
    title = extract_title(soup, content_block)
    raw_content = block_to_text(content_block)
    content = filter_content(raw_content)
    date = extract_date(title + "\n" + content)
    links = extract_links(content_block)

    return {
        "url": url,
        "title": title,
        "date": date,
        "content": content,
        "links": links
    }


def save_txt(articles):
    data = []

    for index, article in enumerate(articles, start=1):
        data.append("=" * 120)
        data.append(f"BÀI {index}")
        data.append(f"TIÊU ĐỀ: {article['title']}")
        data.append(f"NGÀY: {article['date']}")
        data.append(f"URL: {article['url']}")
        data.append("-" * 120)
        data.append(article["content"])

        if article["links"]:
            data.append("")
            data.append("LINK TRONG BÀI:")

            for link in article["links"]:
                if link["text"]:
                    data.append(f"- {link['text']}: {link['url']}")
                else:
                    data.append(f"- {link['url']}")

        data.append("")

    Path(OUTPUT_TXT).write_text("\n".join(data), encoding="utf-8")


def save_json(articles):
    Path(OUTPUT_JSON).write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main():
    urls = read_urls_from_txt(INPUT_TXT)

    print(f"Tìm thấy {len(urls)} URL trong file {INPUT_TXT}")

    articles = []

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Đang cào: {url}")

        try:
            article = crawl_article(url)

            if article["content"]:
                articles.append(article)
                print(f"Đã lấy {len(article['content'])} ký tự.")
            else:
                print("Không lấy được nội dung.")

            time.sleep(0.7)

        except Exception as e:
            print(f"Lỗi: {url} - {e}")

    save_txt(articles)
    save_json(articles)

    print(f"Đã lưu {len(articles)} bài vào:")
    print(OUTPUT_TXT)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()